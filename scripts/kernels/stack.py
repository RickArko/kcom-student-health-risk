"""Kaggle Kernel: Stacked HGBC / CatBoost / XGB / LightGBM.

Paste the contents of this file into a Kaggle Notebook cell and run.
Or run locally: uv run python scripts/kernels/stack.py

Inspired by:
  https://www.kaggle.com/code/kospintr/health-stacked-hgbc-catb-xgb-lgbm-baseline

Strategy:
- Median / mode impute → label-encode categoricals → interaction features
- Stratified 5-fold CV on HGBC + CatBoost + XGBoost + LightGBM
- Blend OOF probabilities with BA-optimised simplex weights (WEIGHTING)
- Submit argmax of the blended test probabilities

Evaluation metric: Balanced Accuracy (mean recall per class).
"""

# === SETUP ===
from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Auto-detect Kaggle data path
KAGGLE_DATA = None

try:
    import kagglehub

    _p = kagglehub.competition_download("playground-series-s6e7")
    KAGGLE_DATA = Path(_p)
except Exception:
    pass

if KAGGLE_DATA is None:
    for p in [
        "/kaggle/input/playground-series-s6e7",
        "/kaggle/input/competitions/playground-series-s6e7",
        "data/raw",
        "data",
    ]:
        dp = Path(p)
        if dp.exists() and (dp / "train.csv").exists():
            KAGGLE_DATA = dp
            break

if KAGGLE_DATA is None:
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        for subdir in sorted(input_dir.iterdir()):
            if subdir.is_dir() and (subdir / "train.csv").exists():
                KAGGLE_DATA = subdir
                break

if KAGGLE_DATA is None:
    KAGGLE_DATA = Path("data/raw")

os.system(
    "pip install -q scikit-learn>=1.5 lightgbm>=4.3 xgboost>=2.0 catboost>=1.2 "
    "scipy>=1.13 joblib>=1.3"
)

# === VENDORED CODE ===

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("student_health_stack")

NUM_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
CAT_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
TARGET = "health_condition"
N_CLASSES = 3
RANDOM_STATE = 42
N_FOLDS = 5
N_ESTIMATORS = 400
META_N_TRIALS = 1000


class ModelWeightMeta(BaseEstimator, ClassifierMixin):
    """Search simplex weights over base models to maximise balanced accuracy."""

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


def preprocess(df, *, is_train=True, num_medians=None, cat_modes=None):
    df = df.copy()
    if is_train:
        num_medians, cat_modes = {}, {}
        for col in NUM_COLS:
            if col in df.columns:
                med = df[col].median()
                num_medians[col] = med
                df[col] = df[col].fillna(med)
        for col in CAT_COLS:
            if col in df.columns:
                mode_val = df[col].mode(dropna=True)
                mode = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
                cat_modes[col] = mode
                df[col] = df[col].fillna(mode)
    else:
        for col in NUM_COLS:
            if col in df.columns and num_medians is not None and col in num_medians:
                df[col] = df[col].fillna(num_medians[col])
        for col in CAT_COLS:
            if col in df.columns and cat_modes is not None and col in cat_modes:
                df[col] = df[col].fillna(cat_modes[col])
    return df, num_medians, cat_modes


def encode_categoricals(df, *, fit=True, encoders=None):
    df = df.copy()
    if fit:
        encoders = {}
        for col in CAT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col in CAT_COLS:
            if col in df.columns and encoders is not None and col in encoders:
                le = encoders[col]
                unseen = ~df[col].astype(str).isin(le.classes_)
                df.loc[unseen, col] = le.classes_[0]
                df[col] = le.transform(df[col].astype(str))
    return df, encoders


def add_features(df):
    df = df.copy()
    if "bmi" in df.columns and "exercise_duration" in df.columns:
        df["bmi_exercise_interaction"] = df["bmi"] * df["exercise_duration"]
    if "step_count" in df.columns and "calorie_expenditure" in df.columns:
        df["efficiency_ratio"] = df["calorie_expenditure"] / (df["step_count"] + 1)
    if "heart_rate" in df.columns and "bmi" in df.columns:
        df["heart_bmi_ratio"] = df["heart_rate"] / (df["bmi"] + 1e-8)
    return df


def make_base_models():
    return [
        (
            "hgbc",
            HistGradientBoostingClassifier(
                max_iter=N_ESTIMATORS,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                early_stopping=True,
                n_iter_no_change=30,
                scoring="balanced_accuracy",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "catboost",
            CatBoostClassifier(
                iterations=N_ESTIMATORS,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=9,
                auto_class_weights="Balanced",
                loss_function="MultiClass",
                random_seed=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=N_ESTIMATORS,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=N_CLASSES,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
        (
            "lgbm",
            lgb.LGBMClassifier(
                objective="multiclass",
                num_class=N_CLASSES,
                n_estimators=N_ESTIMATORS,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]


def main() -> None:
    logger.info("=" * 60)
    logger.info("Student Health Risk — Stacked HGBC/CatB/XGB/LGBM")
    logger.info("=" * 60)

    # ── 1. Load ──
    logger.info("[1/5] Loading data ...")
    t0 = time.time()
    train = pd.read_csv(Path(KAGGLE_DATA) / "train.csv")
    test = pd.read_csv(Path(KAGGLE_DATA) / "test.csv")
    test_ids = test["id"].copy()
    logger.info("  Train: %s  Test: %s  (%.1fs)", train.shape, test.shape, time.time() - t0)

    # ── 2. Preprocess ──
    logger.info("[2/5] Preprocessing ...")
    t0 = time.time()
    train, num_medians, cat_modes = preprocess(train, is_train=True)
    test, _, _ = preprocess(test, is_train=False, num_medians=num_medians, cat_modes=cat_modes)
    train, encoders = encode_categoricals(train, fit=True)
    test, _ = encode_categoricals(test, fit=False, encoders=encoders)

    le_target = LabelEncoder()
    y = le_target.fit_transform(train[TARGET])

    train = add_features(train)
    test = add_features(test)

    feature_cols = [c for c in train.columns if c not in ("id", TARGET)]
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    for c in X.columns:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test = X_test[X.columns]
    for col in X.select_dtypes("float64").columns:
        X[col] = X[col].astype("float32")
    for col in X_test.select_dtypes("float64").columns:
        X_test[col] = X_test[col].astype("float32")
    logger.info("  Features: %d  (%.1fs)", len(feature_cols), time.time() - t0)

    # ── 3. CV stack ──
    logger.info("[3/5] Training stack (%d-fold) ...", N_FOLDS)
    t0 = time.time()
    base_models = make_base_models()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_probas = {name: np.zeros((len(X), N_CLASSES), dtype=np.float32) for name, _ in base_models}
    test_probas = {
        name: np.zeros((len(X_test), N_CLASSES), dtype=np.float32) for name, _ in base_models
    }
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        for name, model in base_models:
            m = clone(model)
            m.fit(X_tr, y_tr)
            oof_probas[name][val_idx] = m.predict_proba(X_val)
            test_probas[name] += m.predict_proba(X_test) / N_FOLDS

        blend = sum(oof_probas[n][val_idx] for n, _ in base_models)
        fold_ba = balanced_accuracy_score(y_val, blend.argmax(axis=1))
        fold_scores.append(fold_ba)
        logger.info("  Fold %d/%d  blend BA: %.4f", fold + 1, N_FOLDS, fold_ba)

    for name, _ in base_models:
        ba = balanced_accuracy_score(y, oof_probas[name].argmax(axis=1))
        logger.info("  %-10s OOF BA: %.4f", name, ba)

    # ── 4. Meta blend ──
    logger.info("[4/5] BA-weight blending ...")
    oof_meta = np.hstack([oof_probas[n] for n, _ in base_models])
    test_meta = np.hstack([test_probas[n] for n, _ in base_models])
    meta = ModelWeightMeta(n_trials=META_N_TRIALS, random_state=RANDOM_STATE)
    meta.fit(oof_meta, y)
    oof_ba = balanced_accuracy_score(y, meta.predict(oof_meta))
    logger.info("  Weights: %s", dict(zip([n for n, _ in base_models], meta.weights_)))
    logger.info(
        "  Meta OOF BA: %.4f (fold mean %.4f ± %.4f)",
        oof_ba,
        float(np.mean(fold_scores)),
        float(np.std(fold_scores)),
    )
    logger.info("  Time: %.1fs", time.time() - t0)

    # ── 5. Submission ──
    logger.info("[5/5] Generating submission ...")
    test_labels = meta.predict(test_meta)
    pred_labels = le_target.inverse_transform(test_labels)

    # Kaggle writes to cwd; local repo uses data/submissions/
    if Path("/kaggle").exists():
        out_path = Path("submission.csv")
    else:
        out_dir = Path("data/submissions")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "submission.csv"

    submission = pd.DataFrame({"id": test_ids, "health_condition": pred_labels})
    submission.to_csv(out_path, index=False)
    logger.info("  Submission saved: %s (%d rows)", out_path, len(submission))
    dist = submission["health_condition"].value_counts().to_dict()
    logger.info("  Predicted distribution: %s", dist)
    logger.info("Done ✓  OOF Balanced Acc: %.4f", oof_ba)


if __name__ == "__main__":
    main()
