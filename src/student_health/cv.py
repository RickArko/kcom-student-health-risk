"""Fixed stratified CV helpers and OOF artifact I/O."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


def make_cv(
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> StratifiedKFold:
    """Return the competition's canonical stratified K-fold splitter."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def fold_indices(
    y: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize (train_idx, val_idx) pairs for reuse across models."""
    cv = make_cv(n_splits=n_splits, random_state=random_state)
    X_dummy = np.zeros(len(y))
    return list(cv.split(X_dummy, y))


def save_fold_indices(folds: list[tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """Persist fold indices as a JSON list of {train, val} arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"train": tr.tolist(), "val": va.tolist()} for tr, va in folds]
    path.write_text(json.dumps(payload))
    logger.info("Saved %d fold index pairs → %s", len(folds), path)


def load_fold_indices(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load fold indices written by :func:`save_fold_indices`."""
    payload = json.loads(Path(path).read_text())
    return [
        (np.asarray(item["train"], dtype=np.int64), np.asarray(item["val"], dtype=np.int64))
        for item in payload
    ]


def save_oof_artifacts(
    out_dir: Path,
    *,
    oof_proba: np.ndarray,
    test_proba: np.ndarray | None,
    metrics: dict,
    name: str = "model",
) -> Path:
    """Save OOF/test probability arrays and metrics under ``out_dir/name/``."""
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof_proba.npy", oof_proba.astype(np.float32))
    if test_proba is not None:
        np.save(out_dir / "test_proba.npy", test_proba.astype(np.float32))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved OOF artifacts → %s (BA=%.4f)", out_dir, metrics.get("oof_ba", float("nan")))
    return out_dir


def load_oof_artifacts(path: Path) -> dict:
    """Load OOF/test probabilities and metrics from an artifact directory."""
    path = Path(path)
    result: dict = {
        "oof_proba": np.load(path / "oof_proba.npy"),
        "metrics": json.loads((path / "metrics.json").read_text()),
    }
    test_path = path / "test_proba.npy"
    if test_path.exists():
        result["test_proba"] = np.load(test_path)
    return result
