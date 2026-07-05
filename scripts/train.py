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

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from student_health.data import load_train
from student_health.features import build_features, get_X_y
from student_health.models import save_model, train_lightgbm
from student_health.tracking import CONFIG, evaluate_predictions, log_metrics

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
        "--model-name",
        type=str,
        default="health_risk_model",
        help="Name for saved model",
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
    print("\n[1/5] Loading training data...")
    t0 = time.time()
    df_train = load_train(args.data_dir)
    print(
        f"  Loaded {len(df_train):,} rows × {df_train.shape[1]} columns  ({time.time() - t0:.1f}s)"
    )

    # ── 2. Feature engineering ─────────────────────────────────────
    print("\n[2/5] Feature engineering...")
    t0 = time.time()
    df_train_feat = build_features(df_train, train=True)
    print(
        f"  After feature engineering: {df_train_feat.shape[1]} columns  ({time.time() - t0:.1f}s)"
    )

    # ── 3. Train / val split ──────────────────────────────────────
    print(f"\n[3/5] Splitting data (val_frac={args.val_frac})...")
    split_idx = int(len(df_train_feat) * (1 - args.val_frac))
    df_train_split, df_val = df_train_feat.iloc[:split_idx], df_train_feat.iloc[split_idx:]
    X_train, y_train = get_X_y(df_train_split)
    X_val, y_val = get_X_y(df_val)
    print(f"  Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    # ── 4. Train model ────────────────────────────────────────────
    print("\n[4/5] Training LightGBM model...")
    t0 = time.time()
    model = train_lightgbm(X_train, y_train)
    print(f"  Model trained in {time.time() - t0:.1f}s")

    # ── 5. Evaluate ───────────────────────────────────────────────
    print("\n[5/5] Evaluating on validation set...")
    val_pred_classes = model.predict(X_val)
    val_pred_proba = model.predict_proba(X_val)
    val_acc = evaluate_predictions(y_val, val_pred_classes, "accuracy")
    val_f1 = evaluate_predictions(y_val, val_pred_classes, "f1")
    val_auc = evaluate_predictions(y_val, val_pred_proba, "auc")
    print(f"  Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

    metrics = {
        "accuracy": val_acc,
        "f1": val_f1,
        "auc": val_auc,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_train.shape[1],
    }

    # ── 6. Save model & metrics ────────────────────────────────────
    print("\nSaving model and metrics...")
    model_path = output_dir / f"{args.model_name}.pkl"
    save_model(model, model_path)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    log_metrics(
        {
            "accuracy": val_acc,
            "f1": val_f1,
            "auc": val_auc,
            "model_path": str(model_path),
        }
    )

    print("\nDone ✓")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
