from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONFIG = {
    "target_encoding": "binary",
    "numeric_scale": "minmax",
    "categorical_handling": "one-hot",
    "split_strategy": "temporal",
    "cv_folds": 5,
    "model_type": "lightgbm",
    "submission_format": "probability",
}


def save_config():
    config_path = Path("experiments/latest_config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    logger.info("Saved config to %s", config_path)


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def evaluate_predictions(y_true, y_pred, metric: str = "auc"):
    if metric == "auc":
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(y_true, y_pred)
    elif metric == "f1":
        from sklearn.metrics import f1_score

        y_pred_class = (y_pred > 0.5).astype(int)
        return f1_score(y_true, y_pred_class)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def log_metrics(metrics: dict):
    metrics_path = Path("experiments/latest_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Logged metrics to %s", metrics_path)
