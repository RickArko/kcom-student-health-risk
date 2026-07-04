"""Model utilities for Student Health Risk Prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_model(model_path: str | Path) -> lgb.LGBMClassifier:
    """Load a trained LightGBM model."""
    return lgb.LGBMClassifier(model_path=model_path)


def save_model(model, path: str | Path):
    """Save a trained model to disk."""
    model.save_model(path)
    logger.info("Saved model to %s", path)


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5, **kwargs):
    """Train a LightGBM model."""
    model = lgb.LGBMClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model
