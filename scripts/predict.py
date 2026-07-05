#!/usr/bin/env python3
"""Student Health Risk Prediction - Prediction Pipeline

Usage:
    uv run python scripts/predict.py --model model.pkl \\
        --data data/raw/test.csv --output submission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from student_health.features import TARGET_MAPPING, build_features
from student_health.models import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions for Student Health Risk Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.pkl)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/test.csv"),
        help="Path to test data CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/submission.csv"),
        help="Path to output submission CSV",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Path to run directory containing model.pkl (if available)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Student Health Risk Prediction - Prediction Pipeline")
    print("=" * 70)

    # ── 1. Load model ──────────────────────────────────────────────
    print(f"\nLoading model from: {args.model}")
    if not args.model.exists():
        if args.run_dir and (args.run_dir / "model.pkl").exists():
            model_path = args.run_dir / "model.pkl"
            logger.info("Using model from run dir: %s", model_path)
        else:
            raise FileNotFoundError(f"Model not found at {args.model}")
    else:
        model_path = args.model

    model = load_model(model_path)

    # ── 2. Load and preprocess test data ─────────────────────────
    print(f"\nLoading test data from: {args.data}")
    df = pd.read_csv(args.data)

    print(f"Processing {len(df):,} test samples...")
    df_feat = build_features(df, train=False)
    print(f"  After feature engineering: {df_feat.shape[1]} columns")

    # ── 3. Generate predictions ───────────────────────────────────
    print("\nGenerating predictions...")
    pred_labels = model.predict(df_feat.drop(columns=["id"]))
    rev_map = {v: k for k, v in TARGET_MAPPING.items()}
    predictions = [rev_map[p] for p in pred_labels]

    # ── 4. Create submission ──────────────────────────────────────
    print("\nCreating submission file...")
    submission = pd.DataFrame(
        {
            "id": df["id"],
            "health_condition": predictions,
        }
    )

    submission.to_csv(args.output, index=False)
    print(f"Submission saved to: {args.output}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
