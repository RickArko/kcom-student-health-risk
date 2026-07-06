from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

CONFIG = {
    "target_encoding": "multiclass",
    "n_classes": 3,
    "model_type": "lightgbm",
    "submission_format": "class_label",
    "metric": "balanced_accuracy",
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


def evaluate_predictions(y_true, y_pred, metric: str = "balanced_accuracy"):
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "f1":
        return f1_score(y_true, y_pred, average="macro")
    elif metric == "auc":
        if y_pred.ndim == 1:
            y_pred = np.eye(CONFIG["n_classes"])[y_pred]
        return roc_auc_score(y_true, y_pred, multi_class="ovr")
    else:
        raise ValueError(f"Unknown metric: {metric}")


def log_metrics(metrics: dict):
    metrics_path = Path("experiments/latest_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Logged metrics to %s", metrics_path)
