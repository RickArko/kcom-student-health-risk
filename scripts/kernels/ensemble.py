"""Kaggle Kernel: LGBM + XGB + CatBoost OOF ensemble (BA blend + thresholds).

Paste into a Kaggle Notebook cell, or run locally:
    uv run python scripts/kernels/ensemble.py

Locally this delegates to scripts/train_ensemble.py.
On Kaggle it installs deps and runs the package pipeline against competition data.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("student_health_ensemble")


def _detect_data_dir() -> Path:
    try:
        import kagglehub

        p = Path(kagglehub.competition_download("playground-series-s6e7"))
        if (p / "train.csv").exists():
            return p
    except Exception:
        pass

    for candidate in [
        Path("/kaggle/input/playground-series-s6e7"),
        Path("/kaggle/input/competitions/playground-series-s6e7"),
        Path("data/raw"),
        Path("data"),
    ]:
        if (candidate / "train.csv").exists():
            return candidate

    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        for subdir in sorted(input_dir.iterdir()):
            if subdir.is_dir() and (subdir / "train.csv").exists():
                return subdir
    return Path("data/raw")


def main() -> None:
    on_kaggle = Path("/kaggle").exists()
    if on_kaggle:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "scikit-learn>=1.5",
                "lightgbm>=4.3",
                "xgboost>=2.0",
                "catboost>=1.2",
                "pyyaml>=6.0",
                "scipy>=1.13",
                "pandas>=2.2",
                "numpy>=1.26",
            ]
        )

    # Prefer invoking the repo training script when present
    repo_script = Path(__file__).resolve().parents[1] / "train_ensemble.py"
    data_dir = _detect_data_dir()
    logger.info("Data dir: %s", data_dir)

    if repo_script.exists():
        cmd = [
            sys.executable,
            str(repo_script),
            "--data-dir",
            str(data_dir),
            "--config",
            str(Path(__file__).resolve().parents[2] / "config" / "ensemble.yaml"),
        ]
        logger.info("Running: %s", " ".join(cmd))
        env = os.environ.copy()
        src = Path(__file__).resolve().parents[2] / "src"
        if src.exists():
            env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}"
        raise SystemExit(subprocess.call(cmd, env=env))

    # Fallback: import package API directly (editable install / Kaggle with src uploaded)
    from student_health.cv import fold_indices, save_oof_artifacts
    from student_health.ensemble import fit_blend, predict_with_blend
    from student_health.features import (
        TARGET_COL,
        TARGET_LABELS,
        TARGET_MAPPING,
        HealthPreprocessor,
    )
    from student_health.metrics import decode_labels, encode_labels
    from student_health.models import train_cv_cat, train_cv_lgbm, train_cv_xgb
    import pandas as pd

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    prep = HealthPreprocessor(missing_indicators=True)
    X, cols = prep.get_feature_matrix(train, fit=True)
    X_test, _ = prep.get_feature_matrix(test, fit=False)
    X_test = X_test.reindex(columns=cols, fill_value=0)
    y = encode_labels(train[TARGET_COL], TARGET_MAPPING)
    folds = fold_indices(y, n_splits=5, random_state=42)

    results = {
        "lgbm": train_cv_lgbm(X, y, X_test, folds=folds),
        "xgb": train_cv_xgb(X, y, X_test, folds=folds),
        "catboost": train_cv_cat(X, y, X_test, folds=folds),
    }
    blend = fit_blend(
        y,
        {k: v["oof_proba"] for k, v in results.items()},
        {k: v["test_proba"] for k, v in results.items()},
    )
    out_dir = Path("experiments/ensemble")
    save_oof_artifacts(
        out_dir,
        oof_proba=blend.oof_proba,
        test_proba=blend.test_proba,
        metrics={"oof_ba": blend.oof_ba_tuned},
        name="blend",
    )
    pred = decode_labels(predict_with_blend(blend), TARGET_LABELS)
    sub_dir = Path("data/submissions")
    sub_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "health_condition": pred}).to_csv(
        sub_dir / "submission.csv", index=False
    )
    logger.info("OOF BA: %.4f", blend.oof_ba_tuned)


if __name__ == "__main__":
    main()
