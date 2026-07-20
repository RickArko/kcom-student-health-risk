#!/usr/bin/env python3
"""TabPFN stratified OOF verification baseline (comparable to the GBDT ensemble).

Runs CV on a stratified subsample (TabPFN sweet spot) so CPU/GPU runs finish
in reasonable time. Requires TABPFN_TOKEN (https://ux.priorlabs.ai).

Usage:
    uv run python scripts/verify_tabpfn.py
    uv run python scripts/verify_tabpfn.py --sample-size 8000 --n-folds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from student_health.cv import fold_indices
from student_health.features import CAT_COLS, N_CLASSES, NUM_COLS, TARGET_COL, TARGET_MAPPING
from student_health.metrics import encode_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("verify_tabpfn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TabPFN OOF verification baseline")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/tabpfn"))
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument(
        "--sample-size",
        type=int,
        default=8000,
        help="Stratified subsample of train used for the entire OOF CV",
    )
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def _ensure_token() -> None:
    load_dotenv()
    token = os.environ.get("TABPFN_TOKEN") or os.environ.get("KAGGLE_SECRET_TABPFN_TOKEN")
    if not token:
        raise SystemExit(
            "TABPFN_TOKEN not set. Accept the license at https://ux.priorlabs.ai "
            "and export TABPFN_TOKEN=<api-key>."
        )
    os.environ["TABPFN_TOKEN"] = token


def _prepare_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for c in NUM_COLS:
        out[c] = out[c].astype("float32")
    for c in CAT_COLS:
        out[c] = out[c].astype(str)
    return out


def main() -> None:
    args = parse_args()
    _ensure_token()
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    from tabpfn import TabPFNClassifier

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    feature_cols = NUM_COLS + CAT_COLS
    y_full = encode_labels(train[TARGET_COL], TARGET_MAPPING)

    # CV entirely inside a stratified subsample (keeps predict sets small)
    sample_n = min(args.sample_size, len(train))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_n, random_state=args.random_state)
    idx, _ = next(sss.split(train, y_full))
    sample = train.iloc[idx].reset_index(drop=True)
    y = encode_labels(sample[TARGET_COL], TARGET_MAPPING)
    logger.info("TabPFN OOF on stratified subsample n=%d", len(sample))

    folds = fold_indices(y, n_splits=args.n_folds, random_state=args.random_state)
    oof_proba = np.zeros((len(sample), N_CLASSES), dtype=np.float32)
    scores: list[float] = []
    cat_indices = list(range(len(NUM_COLS), len(feature_cols)))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        logger.info("Fold %d/%d — TabPFN fit ...", fold + 1, args.n_folds)
        t0 = time.time()
        X_fit = _prepare_frame(sample.iloc[trn_idx], feature_cols)
        y_fit = y[trn_idx]
        X_val = _prepare_frame(sample.iloc[val_idx], feature_cols)

        model = TabPFNClassifier(
            categorical_features_indices=cat_indices,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            show_progress_bar=False,
            ignore_pretraining_limits=True,
        )
        model.fit(X_fit, y_fit)
        val_arr = model.predict_proba(X_val).astype(np.float32)
        oof_proba[val_idx] = val_arr

        fold_ba = float(balanced_accuracy_score(y[val_idx], val_arr.argmax(axis=1)))
        scores.append(fold_ba)
        logger.info(
            "  Fold %d BA: %.4f  (fit_n=%d, val_n=%d, %.1fs)",
            fold + 1,
            fold_ba,
            len(trn_idx),
            len(val_idx),
            time.time() - t0,
        )

    oof_ba = float(balanced_accuracy_score(y, oof_proba.argmax(axis=1)))
    metrics = {
        "oof_ba": oof_ba,
        "fold_scores": scores,
        "sample_size": sample_n,
        "n_folds": args.n_folds,
        "n_estimators": args.n_estimators,
        "note": "OOF computed on stratified subsample (not full train)",
    }
    np.save(args.output_dir / "oof_proba.npy", oof_proba)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("TabPFN OOF BA: %.4f (mean %.4f ± %.4f)", oof_ba, np.mean(scores), np.std(scores))
    logger.info("Artifacts → %s", args.output_dir)


if __name__ == "__main__":
    main()
