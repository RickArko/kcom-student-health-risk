#!/usr/bin/env python3
"""Student Health Risk Prediction - End-to-End Training Pipeline

Usage:
    uv run python scripts/train.py [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from student_health.data import load_data, preprocess, get_feature_cols
from student_health.features import build_features, get_X_y
from student_health.models import train_lightgbm, save_model
from student_health.tracking import log_metrics, evaluate_predictions, CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Student Health Risk Prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to directory with train.csv",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="health_risk_model",
        help="Name for saved model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/latest_run"),
        help="Directory to save outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Student Health Risk Prediction - Training Pipeline")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/6] Loading training data...")
    t0 = time.time()
    df_train, _ = load_data(args.data_dir)
    print(f"  Loaded {len(df_train):,} rows × {df_train.shape[1]} columns  ({time.time() - t0:.1f}s)")

    # ── 2. Preprocess ──────────────────────────────────────────────
    print("\n[2/6] Preprocessing...")
    t0 = time.time()
    df_train = preprocess(df_train)
    feature_cols = get_feature_cols(df_train)
    df_train_feat = build_features(df_train, train=True)
    print(f"  After feature engineering: {len(get_feature_cols(df_train_feat)):,} columns  ({time.time() - t0:.1f}s)")

    # ── 3. Train / val split ──────────────────────────────────────
    print(f"\n[3/6] Splitting data (val_frac={args.val_frac})...")
    # Simple temporal split
    split_idx = int(len(df_train_feat) * (1 - args.val_frac))
    df_train_split, df_val = df_train_feat.iloc[:split_idx], df_train_feat.iloc[split_idx:]
    X_train, y_train = get_X_y(df_train_split)
    X_val, y_val = get_X_y(df_val)
    print(f"  Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    # ── 4. Train model ────────────────────────────────────────────
    print(f"\n[4/6] Training LightGBM model ({args.folds}-fold CV)...")
    t0 = time.time()
    model = train_lightgbm(X_train, y_train, n_splits=args.folds)
    print(f"  Model trained in {time.time() - t0:.1f}s")

    # ── 5. Evaluate ───────────────────────────────────────────────
    print("\n[5/6] Evaluating on validation set...")
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = evaluate_predictions(y_val, val_pred, "auc")
    val_f1 = evaluate_predictions(y_val, val_pred, "f1")
    print(f"  AUC: {val_auc:.4f} | F1: {val_f1:.4f}")

    metrics = {
        "auc": val_auc,
        "f1": val_f1,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_train.shape[1],
    }

    # ── 6. Save model & metrics ────────────────────────────────────
    print("\n[6/6] Saving model and metrics...")
    model_path = output_dir / f"{args.model_name}.pkl"
    save_model(model, model_path)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    log_metrics({
        "auc": val_auc,
        "f1": val_f1,
        "model_path": str(model_path),
    })

    print(f"\nDone ✓")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
