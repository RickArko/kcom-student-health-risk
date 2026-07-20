#!/usr/bin/env python3
"""Train LGBM + XGB + CatBoost OOF ensemble with BA blend + thresholds.

Usage:
    uv run python scripts/train_ensemble.py
    uv run python scripts/train_ensemble.py --config config/ensemble.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from student_health.cv import fold_indices, save_fold_indices, save_oof_artifacts
from student_health.ensemble import fit_blend, predict_with_blend
from student_health.features import (
    TARGET_COL,
    TARGET_LABELS,
    TARGET_MAPPING,
    HealthPreprocessor,
)
from student_health.metrics import decode_labels, encode_labels
from student_health.models import train_cv_cat, train_cv_lgbm, train_cv_xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("train_ensemble")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BA-verified GBDT ensemble")
    p.add_argument("--config", type=Path, default=Path("config/ensemble.yaml"))
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=None, help="Override all model estimators")
    p.add_argument("--skip-xgb", action="store_true")
    p.add_argument("--skip-cat", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        logger.warning("Config %s missing — using defaults", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_dir = Path(args.data_dir or cfg.get("paths", {}).get("data", "data/raw"))
    exp_default = "experiments/ensemble"
    out_dir = Path(args.output_dir or cfg.get("paths", {}).get("experiments", exp_default))
    sub_dir = Path(cfg.get("paths", {}).get("submissions", "data/submissions"))
    cv_cfg = cfg.get("cv", {})
    n_folds = args.n_folds or int(cv_cfg.get("n_splits", 5))
    random_state = int(cv_cfg.get("random_state", 42))
    blend_cfg = cfg.get("blend", {})
    model_cfg = cfg.get("models", {})

    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Student Health Risk — GBDT Ensemble (BA-verified)")
    logger.info("=" * 60)

    logger.info("[1/5] Loading data from %s", data_dir)
    t0 = time.time()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    test_ids = test["id"].copy()
    logger.info("  Train %s  Test %s  (%.1fs)", train.shape, test.shape, time.time() - t0)

    logger.info("[2/5] Preprocessing (missing indicators + impute + encode)")
    t0 = time.time()
    miss = bool(cfg.get("preprocessing", {}).get("missing_indicators", True))
    prep = HealthPreprocessor(missing_indicators=miss)
    X, feature_cols = prep.get_feature_matrix(train, fit=True)
    X_test, _ = prep.get_feature_matrix(test, fit=False)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)
    y = encode_labels(train[TARGET_COL], TARGET_MAPPING)
    logger.info("  Features: %d  (%.1fs)", len(feature_cols), time.time() - t0)

    folds = fold_indices(y, n_splits=n_folds, random_state=random_state)
    save_fold_indices(folds, out_dir / "folds.json")

    lgbm_params = dict(model_cfg.get("lgbm", {}))
    xgb_params = dict(model_cfg.get("xgb", {}))
    cat_params = dict(model_cfg.get("catboost", {}))
    if args.n_estimators is not None:
        lgbm_params["n_estimators"] = args.n_estimators
        xgb_params["n_estimators"] = args.n_estimators
        cat_params["iterations"] = args.n_estimators

    logger.info("[3/5] Training base models (%d-fold CV)", n_folds)
    results: dict[str, dict] = {}

    t0 = time.time()
    results["lgbm"] = train_cv_lgbm(
        X, y, X_test, folds=folds, params=lgbm_params or None, random_state=random_state
    )
    save_oof_artifacts(
        out_dir,
        oof_proba=results["lgbm"]["oof_proba"],
        test_proba=results["lgbm"]["test_proba"],
        metrics={
            "oof_ba": results["lgbm"]["oof_ba"],
            "fold_scores": results["lgbm"]["scores"],
        },
        name="lgbm",
    )
    logger.info("  LGBM wall: %.1fs", time.time() - t0)

    if not args.skip_xgb:
        t0 = time.time()
        results["xgb"] = train_cv_xgb(
            X, y, X_test, folds=folds, params=xgb_params or None, random_state=random_state
        )
        save_oof_artifacts(
            out_dir,
            oof_proba=results["xgb"]["oof_proba"],
            test_proba=results["xgb"]["test_proba"],
            metrics={
                "oof_ba": results["xgb"]["oof_ba"],
                "fold_scores": results["xgb"]["scores"],
            },
            name="xgb",
        )
        logger.info("  XGB wall: %.1fs", time.time() - t0)

    if not args.skip_cat:
        t0 = time.time()
        results["catboost"] = train_cv_cat(
            X, y, X_test, folds=folds, params=cat_params or None, random_state=random_state
        )
        save_oof_artifacts(
            out_dir,
            oof_proba=results["catboost"]["oof_proba"],
            test_proba=results["catboost"]["test_proba"],
            metrics={
                "oof_ba": results["catboost"]["oof_ba"],
                "fold_scores": results["catboost"]["scores"],
            },
            name="catboost",
        )
        logger.info("  CatBoost wall: %.1fs", time.time() - t0)

    logger.info("[4/5] Hill-climb blend + BA thresholds")
    oof_probas = {k: v["oof_proba"] for k, v in results.items()}
    test_probas = {k: v["test_proba"] for k, v in results.items()}
    blend = fit_blend(
        y,
        oof_probas,
        test_probas,
        n_trials=int(blend_cfg.get("n_trials", 2000)),
        random_state=int(blend_cfg.get("random_state", random_state)),
        tune_thresh=bool(blend_cfg.get("tune_thresholds", True)),
    )

    save_oof_artifacts(
        out_dir,
        oof_proba=blend.oof_proba,
        test_proba=blend.test_proba,
        metrics={
            "oof_ba": blend.oof_ba_tuned,
            "oof_ba_argmax": blend.oof_ba_argmax,
            "weights": {n: float(w) for n, w in zip(blend.model_names, blend.weights)},
            "thresholds": blend.thresholds.tolist(),
            "accepted": blend.accepted,
            "single_best": blend.single_best_name,
            "single_best_ba": blend.single_best_ba,
            "per_model": {k: v["oof_ba"] for k, v in results.items()},
        },
        name="blend",
    )

    logger.info("[5/5] Writing submission")
    pred_enc = predict_with_blend(blend)
    pred_labels = decode_labels(pred_enc, TARGET_LABELS)
    submission = pd.DataFrame({"id": test_ids, "health_condition": pred_labels})
    sub_path = sub_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)

    summary = {
        "oof_ba": blend.oof_ba_tuned,
        "oof_ba_argmax": blend.oof_ba_argmax,
        "per_model": {k: v["oof_ba"] for k, v in results.items()},
        "weights": {n: float(w) for n, w in zip(blend.model_names, blend.weights)},
        "thresholds": blend.thresholds.tolist(),
        "n_features": len(feature_cols),
        "n_folds": n_folds,
        "submission": str(sub_path),
        "pred_dist": submission["health_condition"].value_counts().to_dict(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("OOF BA (tuned): %.4f", blend.oof_ba_tuned)
    logger.info("Submission: %s", sub_path)
    logger.info("Metrics: %s", out_dir / "metrics.json")
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
