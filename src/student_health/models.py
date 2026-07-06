"""Model utilities for Student Health Risk Prediction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

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
