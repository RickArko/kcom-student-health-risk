"""Model utilities for Student Health Risk Prediction.

Includes a LightGBM single-model baseline plus a stacking ensemble of
HistGradientBoosting / CatBoost / XGBoost / LightGBM (inspired by
https://www.kaggle.com/code/kospintr/health-stacked-hgbc-catb-xgb-lgbm-baseline).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

N_CLASSES = 3

TARGET_LABELS = ["fit", "at-risk", "unhealthy"]

LGBM_DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

MODEL_REGISTRY: dict[str, type] = {
    "lgbm": lgb.LGBMClassifier,
    "histgbm": HistGradientBoostingClassifier,
}

SEED_PARAM_MAP: dict[str, str] = {
    "lgbm": "random_state",
    "xgb": "random_state",
    "catboost": "random_seed",
    "histgbm": "random_state",
}

try:
    from xgboost import XGBClassifier

    MODEL_REGISTRY["xgb"] = XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[misc, assignment]

try:
    from catboost import CatBoostClassifier

    MODEL_REGISTRY["catboost"] = CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[misc, assignment]


def load_model(model_path: str | Path) -> lgb.LGBMClassifier:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_model(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", path)


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
    params = {**LGBM_DEFAULT_PARAMS, **kwargs}
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame | None = None,
    n_folds: int = 5,
    random_state: int = 42,
    params: dict | None = None,
) -> dict:
    if params is None:
        params = LGBM_DEFAULT_PARAMS
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_proba = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    oof_labels = np.zeros(len(X), dtype=np.int32)
    test_proba = (
        np.zeros((len(X_test), N_CLASSES), dtype=np.float32) if X_test is not None else None
    )
    scores = []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
        )

        val_proba = model.predict_proba(X_val)
        oof_proba[val_idx] = val_proba
        oof_labels[val_idx] = val_proba.argmax(axis=1)

        if test_proba is not None:
            test_proba += model.predict_proba(X_test) / n_folds

        fold_ba = balanced_accuracy_score(y_val, oof_labels[val_idx])
        scores.append(fold_ba)
        logger.info("Fold %d/%d — Balanced Acc: %.4f", fold + 1, n_folds, fold_ba)

    oof_ba = balanced_accuracy_score(y, oof_labels)
    logger.info(
        "OOF Balanced Acc: %.4f (mean %.4f ± %.4f)",
        oof_ba,
        np.mean(scores),
        np.std(scores),
    )
    return {
        "oof_proba": oof_proba,
        "oof_labels": oof_labels,
        "test_proba": test_proba,
        "scores": scores,
        "oof_ba": oof_ba,
    }


# ── Stacking meta-models ──────────────────────────────────────────────────────


class SimpleAverageMeta(BaseEstimator, ClassifierMixin):
    """Average base-model probabilities and take argmax (no learning)."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_models_ = X.shape[1] // len(self.classes_)
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        reshaped = X.reshape(X.shape[0], self.n_models_, n_classes)
        return reshaped.mean(axis=1)


class ModelWeightMeta(BaseEstimator, ClassifierMixin):
    """Search simplex weights over base models to maximise balanced accuracy.

    Matches the WEIGHTING path in the kospintr stacked baseline: try uniform,
    single-model, and random Dirichlet weights; keep the best OOF blend.
    """

    def __init__(self, n_trials: int = 1000, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_models = X.shape[1] // n_classes
        reshaped = X.reshape(X.shape[0], n_models, n_classes)

        candidates = [np.ones(n_models) / n_models]
        for i in range(n_models):
            w = np.zeros(n_models)
            w[i] = 1.0
            candidates.append(w)

        rng = np.random.default_rng(self.random_state)
        candidates.extend(rng.dirichlet(np.ones(n_models), size=self.n_trials))

        best_score = -1.0
        best_weights = candidates[0]
        for weights in candidates:
            blend = np.tensordot(weights, reshaped, axes=(0, 1))
            score = balanced_accuracy_score(y, blend.argmax(axis=1))
            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        self.weights_ = best_weights
        self.n_models_ = n_models
        self.best_score_ = best_score
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        reshaped = X.reshape(X.shape[0], self.n_models_, n_classes)
        return np.tensordot(self.weights_, reshaped, axes=(0, 1))


class WeightedAverageMeta(BaseEstimator, ClassifierMixin):
    """Per-model per-class weights via L-BFGS-B (non-negative)."""

    def fit(self, X, y):
        from scipy.optimize import minimize

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_models = X.shape[1] // n_classes
        reshaped = X.reshape(X.shape[0], n_models, n_classes)

        def neg_balanced_acc(weights):
            w = weights.reshape(n_models, n_classes)
            scores = (reshaped * w[np.newaxis, :, :]).sum(axis=1)
            return -balanced_accuracy_score(y, scores.argmax(axis=1))

        x0 = np.ones(n_models * n_classes)
        bounds = [(0, None)] * (n_models * n_classes)
        result = minimize(neg_balanced_acc, x0, method="L-BFGS-B", bounds=bounds)
        self.weights_ = result.x.reshape(n_models, n_classes)
        self.n_models_ = n_models
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        reshaped = X.reshape(X.shape[0], self.n_models_, n_classes)
        scores = (reshaped * self.weights_[np.newaxis, :, :]).sum(axis=1)
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return scores / row_sums


class StackingEnsemble:
    """Stacking ensemble with stratified k-fold CV and OOF meta-features.

    Trains base models via CV, stacks OOF probabilities, then fits a meta-model
    (default: ModelWeightMeta for BA-optimised blending).
    """

    def __init__(
        self,
        base_models: list[tuple[str, object]],
        meta_model: object | None = None,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or ModelWeightMeta()
        self.fold_models_: list[dict[str, object]] = []
        self.meta_model_: object | None = None
        self.label_encoder_: LabelEncoder | None = None
        self.n_classes_: int | None = None
        self.valid_scores_: list[float] = []
        self.per_model_oof_scores_: dict[str, float] = {}
        self.overall_oof_score_: float | None = None
        self.oof_meta_: np.ndarray | None = None
        self.test_meta_: np.ndarray | None = None
        self.y_enc_: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        cv: object,
        X_test: pd.DataFrame | None = None,
        model_fit_kwargs: dict[str, dict] | None = None,
    ) -> StackingEnsemble:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.label_encoder_ = le
        self.n_classes_ = len(le.classes_)
        self.y_enc_ = y_enc

        n_train = len(X)
        has_test = X_test is not None
        n_test = len(X_test) if has_test else 0

        oof_probas: dict[str, np.ndarray] = {
            name: np.zeros((n_train, self.n_classes_), dtype=np.float32)
            for name, _ in self.base_models
        }
        if has_test:
            test_probas: dict[str, np.ndarray] = {
                name: np.zeros((n_test, self.n_classes_), dtype=np.float32)
                for name, _ in self.base_models
            }

        n_splits = cv.get_n_splits()
        fold_iter = list(enumerate(cv.split(X, y_enc)))
        for fold, (train_idx, val_idx) in fold_iter:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

            fold_models: dict[str, object] = {}
            for name, model in self.base_models:
                m = clone(model)
                kwargs = (model_fit_kwargs or {}).get(name, {})
                m.fit(X_tr, y_tr, **kwargs)
                fold_models[name] = m
                oof_probas[name][val_idx] = m.predict_proba(X_val)
                if has_test:
                    test_probas[name] += m.predict_proba(X_test) / n_splits

            self.fold_models_.append(fold_models)

            fold_blend = sum(oof_probas[n][val_idx] for n, _ in self.base_models)
            fold_preds = fold_blend.argmax(axis=1)
            score = balanced_accuracy_score(y_val, fold_preds)
            self.valid_scores_.append(score)
            logger.info(
                "Fold %d/%d — blend BA: %.4f",
                fold + 1,
                n_splits,
                score,
            )

        for name, _ in self.base_models:
            preds = oof_probas[name].argmax(axis=1)
            self.per_model_oof_scores_[name] = balanced_accuracy_score(y_enc, preds)
            logger.info("  %-12s OOF BA: %.4f", name, self.per_model_oof_scores_[name])

        oof_meta = np.hstack([oof_probas[n] for n, _ in self.base_models])
        self.oof_meta_ = oof_meta
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(oof_meta, y_enc)

        oof_final = self.meta_model_.predict(oof_meta)
        self.overall_oof_score_ = balanced_accuracy_score(y_enc, oof_final)
        logger.info(
            "Meta OOF BA: %.4f (fold mean %.4f ± %.4f)",
            self.overall_oof_score_,
            float(np.mean(self.valid_scores_)),
            float(np.std(self.valid_scores_)),
        )

        self.test_meta_ = (
            np.hstack([test_probas[n] for n, _ in self.base_models]) if has_test else None
        )
        return self

    def predict_proba(self, X: pd.DataFrame | None = None) -> np.ndarray:
        """Predict probabilities. If X is None, use cached test_meta_ from fit."""
        if X is None:
            if self.test_meta_ is None:
                raise ValueError("No cached test predictions; pass X or fit with X_test.")
            return self.meta_model_.predict_proba(self.test_meta_)

        test_meta = self._build_meta(X)
        return self.meta_model_.predict_proba(test_meta)

    def predict(self, X: pd.DataFrame | None = None) -> np.ndarray:
        """Predict string labels. If X is None, use cached test_meta_ from fit."""
        if X is None:
            if self.test_meta_ is None:
                raise ValueError("No cached test predictions; pass X or fit with X_test.")
            preds_enc = self.meta_model_.predict(self.test_meta_)
        else:
            test_meta = self._build_meta(X)
            preds_enc = self.meta_model_.predict(test_meta)
        return self.label_encoder_.inverse_transform(preds_enc)

    def _build_meta(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        n_folds = len(self.fold_models_)
        test_probas: dict[str, np.ndarray] = {
            name: np.zeros((n, self.n_classes_), dtype=np.float32) for name, _ in self.base_models
        }
        for fold_models in self.fold_models_:
            for name, _ in self.base_models:
                test_probas[name] += fold_models[name].predict_proba(X) / n_folds
        return np.hstack([test_probas[n] for n, _ in self.base_models])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> StackingEnsemble:
        import joblib

        return joblib.load(path)


def default_stack_base_models(
    random_state: int = 42,
    *,
    n_estimators: int = 400,
) -> list[tuple[str, object]]:
    """Build the HGBC / CatBoost / XGB / LGBM base-model set."""
    if XGBClassifier is None or CatBoostClassifier is None:
        raise ImportError(
            "xgboost and catboost are required for the stack. "
            "Install with: uv sync && uv add xgboost catboost"
        )

    return [
        (
            "hgbc",
            HistGradientBoostingClassifier(
                max_iter=n_estimators,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.0,
                early_stopping=True,
                n_iter_no_change=30,
                scoring="balanced_accuracy",
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        (
            "catboost",
            CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=9,
                auto_class_weights="Balanced",
                loss_function="MultiClass",
                eval_metric="TotalF1",
                random_seed=random_state,
                verbose=0,
                allow_writing_files=False,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=N_CLASSES,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
        (
            "lgbm",
            lgb.LGBMClassifier(
                objective="multiclass",
                num_class=N_CLASSES,
                metric="multi_logloss",
                n_estimators=n_estimators,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]


def build_meta_model(meta_type: str = "model_weight", **kwargs) -> object:
    """Construct a meta-learner by name."""
    if meta_type in {"model_weight", "weighting"}:
        return ModelWeightMeta(
            n_trials=kwargs.get("n_trials", 1000),
            random_state=kwargs.get("random_state", 42),
        )
    if meta_type in {"simple_average", "average"}:
        return SimpleAverageMeta()
    if meta_type in {"weighted_average", "per_class_weight"}:
        return WeightedAverageMeta()
    if meta_type in {"logistic", "logistic_regression"}:
        return LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=kwargs.get("random_state", 42),
        )
    if meta_type in {"hgbc", "histgbm", "stacking"}:
        return HistGradientBoostingClassifier(
            max_iter=kwargs.get("max_iter", 500),
            learning_rate=kwargs.get("learning_rate", 0.05),
            early_stopping=True,
            n_iter_no_change=40,
            scoring="balanced_accuracy",
            class_weight="balanced",
            random_state=kwargs.get("random_state", 42),
        )
    raise ValueError(f"Unknown meta_type: {meta_type!r}")
