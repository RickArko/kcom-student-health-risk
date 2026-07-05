"""Model utilities for Student Health Risk Prediction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd

logger = logging.getLogger(__name__)

N_CLASSES = 3


def load_model(model_path: str | Path) -> lgb.LGBMClassifier:
    """Load a trained LightGBM model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_model(model, path: str | Path):
    """Save a trained model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", path)


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
    """Train a LightGBM multi-class model."""
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        verbose=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model
