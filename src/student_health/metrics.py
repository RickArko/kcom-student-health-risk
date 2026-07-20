"""Balanced-accuracy helpers and BA-aware decision thresholds."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def ba_from_proba(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Balanced accuracy using argmax over class probabilities."""
    return float(balanced_accuracy_score(y_true, proba.argmax(axis=1)))


def apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Predict class by scaled probabilities: argmax(p / t).

    ``thresholds`` length equals n_classes. Values near 1.0 recover argmax;
    lowering a class threshold boosts that class's recall (useful for BA).
    """
    t = np.asarray(thresholds, dtype=np.float64)
    t = np.clip(t, 1e-6, None)
    return (proba / t).argmax(axis=1)


def tune_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    n_rounds: int = 3,
) -> tuple[np.ndarray, float]:
    """Coordinate-ascent search over per-class thresholds to maximize BA.

    Returns ``(thresholds, best_ba)``. Starts from all-ones (argmax) and
    never returns a worse score than argmax BA.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    n_classes = proba.shape[1]
    if grid is None:
        grid = np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2], dtype=np.float64)

    thresholds = np.ones(n_classes, dtype=np.float64)
    best_ba = float(balanced_accuracy_score(y_true, apply_thresholds(proba, thresholds)))

    for _ in range(n_rounds):
        improved = False
        for c in range(n_classes):
            local_best_t = thresholds[c]
            local_best_ba = best_ba
            for t in grid:
                trial = thresholds.copy()
                trial[c] = float(t)
                ba = float(balanced_accuracy_score(y_true, apply_thresholds(proba, trial)))
                if ba > local_best_ba + 1e-12:
                    local_best_ba = ba
                    local_best_t = float(t)
            if local_best_t != thresholds[c]:
                thresholds[c] = local_best_t
                best_ba = local_best_ba
                improved = True
        if not improved:
            break

    return thresholds, best_ba


def encode_labels(y: np.ndarray | list, mapping: dict[str, int]) -> np.ndarray:
    """Map string labels to integers with a fixed class order."""
    if hasattr(y, "map"):
        encoded = y.map(mapping)
        if encoded.isna().any():
            bad = sorted(set(y[encoded.isna()].astype(str)))
            raise ValueError(f"Unknown labels: {bad}")
        return encoded.to_numpy(dtype=np.int32)
    arr = np.asarray(y)
    if arr.dtype.kind in {"i", "u"}:
        return arr.astype(np.int32)
    return np.array([mapping[str(v)] for v in arr], dtype=np.int32)


def decode_labels(y_enc: np.ndarray, labels: list[str]) -> np.ndarray:
    """Map integer class indices back to string labels."""
    return np.asarray([labels[int(i)] for i in y_enc], dtype=object)
