#!/usr/bin/env python3
"""Run AutoGluon Tabular as an automated verification baseline (same folds).

Requires optional deps:
    uv sync --extra verify

Usage:
    uv run python scripts/verify_autogluon.py
    uv run python scripts/verify_autogluon.py --time-limit 600
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from student_health.cv import fold_indices, save_oof_artifacts
from student_health.features import N_CLASSES, TARGET_COL, TARGET_LABELS, TARGET_MAPPING
from student_health.metrics import encode_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("verify_autogluon")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoGluon OOF verification baseline")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/autogluon"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--time-limit", type=int, default=600, help="Seconds per fold")
    p.add_argument("--presets", type=str, default="medium_quality")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Optional stratified subsample size (0 = full train)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise SystemExit(
            "autogluon.tabular is not installed. Run: uv sync --extra verify\n"
            f"Original error: {exc}"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading data from %s", args.data_dir)
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")

    feature_cols = [c for c in train.columns if c not in ("id", TARGET_COL)]
    y = encode_labels(train[TARGET_COL], TARGET_MAPPING)

    if args.sample and args.sample < len(train):
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=args.sample, random_state=args.random_state
        )
        idx, _ = next(sss.split(train, y))
        train = train.iloc[idx].reset_index(drop=True)
        y = encode_labels(train[TARGET_COL], TARGET_MAPPING)
        logger.info("Subsampled to %d rows", len(train))

    folds = fold_indices(y, n_splits=args.n_folds, random_state=args.random_state)
    oof_proba = np.zeros((len(train), N_CLASSES), dtype=np.float32)
    test_proba = np.zeros((len(test), N_CLASSES), dtype=np.float32)
    scores: list[float] = []

    label_order = TARGET_LABELS  # fit, at-risk, unhealthy

    for fold, (trn_idx, val_idx) in enumerate(folds):
        logger.info(
            "Fold %d/%d — fitting AutoGluon (%ss) ...",
            fold + 1,
            args.n_folds,
            args.time_limit,
        )
        t0 = time.time()
        fold_dir = args.output_dir / f"fold_{fold}"
        if fold_dir.exists():
            import shutil

            shutil.rmtree(fold_dir)

        train_fold = train.iloc[trn_idx][feature_cols + [TARGET_COL]].copy()
        val_fold = train.iloc[val_idx][feature_cols].copy()

        predictor = TabularPredictor(
            label=TARGET_COL,
            problem_type="multiclass",
            eval_metric="balanced_accuracy",
            path=str(fold_dir),
            verbosity=1,
        )
        predictor.fit(
            train_fold,
            time_limit=args.time_limit,
            presets=args.presets,
        )

        val_pred = predictor.predict_proba(val_fold)
        # Align columns to TARGET_LABELS order
        val_arr = np.column_stack([val_pred[c].to_numpy() for c in label_order]).astype(np.float32)
        oof_proba[val_idx] = val_arr

        te_pred = predictor.predict_proba(test[feature_cols])
        te_arr = np.column_stack([te_pred[c].to_numpy() for c in label_order]).astype(np.float32)
        test_proba += te_arr / args.n_folds

        fold_ba = float(balanced_accuracy_score(y[val_idx], val_arr.argmax(axis=1)))
        scores.append(fold_ba)
        logger.info("  Fold %d BA: %.4f  (%.1fs)", fold + 1, fold_ba, time.time() - t0)

    oof_ba = float(balanced_accuracy_score(y, oof_proba.argmax(axis=1)))
    metrics = {
        "oof_ba": oof_ba,
        "fold_scores": scores,
        "presets": args.presets,
        "time_limit": args.time_limit,
        "n_folds": args.n_folds,
        "n_rows": len(train),
    }
    save_oof_artifacts(
        args.output_dir.parent if args.output_dir.name == "autogluon" else args.output_dir,
        oof_proba=oof_proba,
        test_proba=test_proba,
        metrics=metrics,
        name="autogluon" if args.output_dir.name == "autogluon" else args.output_dir.name,
    )
    # Also write canonical metrics path expected by compare_baselines
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(args.output_dir / "oof_proba.npy", oof_proba)
    np.save(args.output_dir / "test_proba.npy", test_proba)

    logger.info(
        "AutoGluon OOF BA: %.4f (mean %.4f ± %.4f)",
        oof_ba,
        np.mean(scores),
        np.std(scores),
    )
    logger.info("Artifacts → %s", args.output_dir)


if __name__ == "__main__":
    main()
