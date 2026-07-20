#!/usr/bin/env python3
"""Train the HGBC/CatB/XGB/LGBM stack and write a submission.

Usage:
    uv run python scripts/train_stack.py
    uv run python scripts/train_stack.py --n-estimators 100 --n-folds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from student_health.data import load_data
from student_health.features import TARGET_COL, HealthPreprocessor
from student_health.models import StackingEnsemble, build_meta_model, default_stack_base_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("train_stack")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train stacked HGBC/CatB/XGB/LGBM")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/stack"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--meta", type=str, default="model_weight")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = Path("data/submissions")
    sub_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", args.data_dir)
    t0 = time.time()
    train, test = load_data(args.data_dir)
    test_ids = test["id"].copy()

    prep = HealthPreprocessor()
    X, feature_cols = prep.get_feature_matrix(train, fit=True)
    X_test, _ = prep.get_feature_matrix(test, fit=False)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)
    y = train[TARGET_COL]
    logger.info("Features: %d  (%.1fs)", len(feature_cols), time.time() - t0)

    base_models = default_stack_base_models(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    meta = build_meta_model(args.meta, random_state=args.random_state)
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)

    logger.info("Fitting stack (%d models × %d folds) ...", len(base_models), args.n_folds)
    t0 = time.time()
    ensemble = StackingEnsemble(base_models=base_models, meta_model=meta)
    ensemble.fit(X, y, cv=cv, X_test=X_test)
    logger.info("Train wall time: %.1fs", time.time() - t0)

    pred_labels = ensemble.predict()
    submission = pd.DataFrame({"id": test_ids, "health_condition": pred_labels})
    sub_path = sub_dir / "submission_stack.csv"
    submission.to_csv(sub_path, index=False)

    model_path = args.output_dir / "ensemble.joblib"
    ensemble.save(model_path)

    metrics = {
        "oof_ba": ensemble.overall_oof_score_,
        "fold_scores": ensemble.valid_scores_,
        "per_model_oof": ensemble.per_model_oof_scores_,
        "n_features": len(feature_cols),
        "n_estimators": args.n_estimators,
        "meta": args.meta,
    }
    if hasattr(ensemble.meta_model_, "weights_"):
        metrics["blend_weights"] = {
            n: float(w) for (n, _), w in zip(base_models, ensemble.meta_model_.weights_)
        }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("OOF BA: %.4f", ensemble.overall_oof_score_)
    logger.info("Submission: %s", sub_path)
    logger.info("Model: %s", model_path)
    logger.info("Metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
