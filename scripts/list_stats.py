#!/usr/bin/env python3
"""Student Health Risk Prediction - Statistics and Listing Script

Usage:
    uv run python scripts/list_stats.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="List dataset statistics and pipeline info")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stats.json"),
        help="Path to output statistics JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = {}

    print("=" * 70)
    print("Student Health Risk Prediction - Statistics")
    print("=" * 70)

    # ── 1. Check data files ───────────────────────────────────────
    print("\n[1/5] Checking data files...")
    train_file = args.data_dir / "train.csv"
    test_file = args.data_dir / "test.csv"

    if train_file.exists():
        df_train = pd.read_csv(train_file)
        stats["train"] = {
            "rows": len(df_train),
            "columns": df_train.shape[1],
            "target_sum": int(df_train["health_risk"].sum()),
            "target_mean": float(df_train["health_risk"].mean()),
            "target_std": float(df_train["health_risk"].std()),
        }
        print(f"  ✓ train.csv: {len(df_train):,} rows, {df_train.shape[1]} columns")

    if test_file.exists():
        df_test = pd.read_csv(test_file)
        stats["test"] = {
            "rows": len(df_test),
            "columns": df_test.shape[1],
        }
        print(f"  ✓ test.csv: {len(df_test):,} rows, {df_test.shape[1]} columns")

    # ── 2. Check experiments ─────────────────────────────────────
    print("\n[2/5] Checking experiments...")
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        exp_dirs = list(experiments_dir.glob("run_*"))
        stats["experiments"] = {
            "count": len(exp_dirs),
            "dirs": [d.name for d in exp_dirs],
        }
        print(f"  ✓ Found {len(exp_dirs)} experiment runs")

    # ── 3. Check models ───────────────────────────────────────────
    print("\n[3/5] Checking models...")
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        stats["models"] = {
            "count": len(model_files),
            "files": [f.name for f in model_files],
        }
        print(f"  ✓ Found {len(model_files)} model files")

    # ── 4. Check submissions ─────────────────────────────────────
    print("\n[4/5] Checking submissions...")
    submissions_dir = Path("submissions")
    if submissions_dir.exists():
        submission_files = list(submissions_dir.glob("*.csv"))
        stats["submissions"] = {
            "count": len(submission_files),
            "files": [f.name for f in submission_files],
        }
        print(f"  ✓ Found {len(submission_files)} submission files")

    # ── 5. Save stats ──────────────────────────────────────────────
    print("\n[5/5] Saving statistics...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Statistics saved to: {args.output}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
