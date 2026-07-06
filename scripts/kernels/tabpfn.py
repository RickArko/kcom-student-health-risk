"""Kaggle Kernel: TabPFN v3 for Student Health Risk Prediction.

Paste into a Kaggle Notebook cell and run (GPU recommended).
TabPFN is a prior-data fitted transformer — no tuning needed.

Strategy:
- TabPFN handles missing values natively
- Pass categoricals as strings (TabPFN auto-encodes)
- Stratified subsample to 15K rows (TabPFN sweet spot)
- Single fast run (skip CV, rely on Kaggle LB for eval)
"""

# === SETUP ===
from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

KAGGLE_DATA = None

try:
    import kagglehub

    _p = kagglehub.competition_download("playground-series-s6e7")
    KAGGLE_DATA = Path(_p)
except Exception:
    pass

if KAGGLE_DATA is None:
    for p in [
        "/kaggle/input/playground-series-s6e7",
        "/kaggle/input/competitions/playground-series-s6e7",
        "data",
    ]:
        dp = Path(p)
        if dp.exists() and (dp / "train.csv").exists():
            KAGGLE_DATA = dp
            break

if KAGGLE_DATA is None:
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        for subdir in sorted(input_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / "train.csv").exists():
                KAGGLE_DATA = subdir
                break

if KAGGLE_DATA is None:
    KAGGLE_DATA = Path("data")

os.system("pip install -q tabpfn>=8.0 scikit-learn>=1.5")

# === VENDORED CODE ===

from sklearn.model_selection import StratifiedShuffleSplit
from tabpfn import TabPFNClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("student_health_tabpfn")

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
CAT_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
TARGET = "health_condition"
TARGET_MAP = {"fit": 0, "at-risk": 1, "unhealthy": 2}
RANDOM_STATE = 42


def main() -> None:
    logger.info("=" * 60)
    logger.info("Student Health Risk — TabPFN v3")
    logger.info("=" * 60)

    # ── 1. Load ──
    logger.info("[1/4] Loading data ...")
    data_path = Path(KAGGLE_DATA)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    test_ids = test["id"].copy()
    logger.info("  Train: %s  Test: %s", train.shape, test.shape)

    # ── 2. Prepare ──
    logger.info("[2/4] Stratified subsample + prepare ...")
    SAMPLE_SIZE = 15000
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    idx, _ = next(sss.split(train, train[TARGET]))
    sample = train.iloc[idx].reset_index(drop=True)
    logger.info(
        "  Sampled %d rows (class dist: %s)", SAMPLE_SIZE, sample[TARGET].value_counts().to_dict()
    )

    feature_cols = NUM_COLS + CAT_COLS
    X_tr = sample[feature_cols].copy()
    y_tr = sample[TARGET].map(TARGET_MAP).values

    X_te = test[feature_cols].copy()

    # TabPFN handles NaNs natively — no imputation
    for c in NUM_COLS:
        X_tr[c] = X_tr[c].astype("float32")
        X_te[c] = X_te[c].astype("float32")
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype(str)
        X_te[c] = X_te[c].astype(str)

    cat_indices = list(range(len(NUM_COLS), len(feature_cols)))

    # ── 3. Fit TabPFN ──
    logger.info("[3/4] Fitting TabPFN (n_estimators=8, %d samples) ...", len(X_tr))
    t0 = time.time()
    model = TabPFNClassifier(
        categorical_features_indices=cat_indices,
        n_estimators=8,
        random_state=RANDOM_STATE,
        show_progress_bar=False,
    )
    model.fit(X_tr, y_tr)
    logger.info("  Fit complete (%.1fs)", time.time() - t0)

    # ── 4. Predict + Submit ──
    logger.info("[4/4] Predicting ...")
    t0 = time.time()
    preds = model.predict_proba(X_te)
    test_labels = preds.argmax(axis=1)
    rev_map = {v: k for k, v in TARGET_MAP.items()}
    pred_labels = [rev_map[i] for i in test_labels]
    logger.info("  Predict complete (%.1fs)", time.time() - t0)

    out_dir = Path("data/submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "health_condition": pred_labels})
    submission.to_csv(out_dir / "submission.csv", index=False)
    logger.info("  Saved: data/submissions/submission.csv")
    logger.info("  Distribution: %s", submission["health_condition"].value_counts().to_dict())
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
