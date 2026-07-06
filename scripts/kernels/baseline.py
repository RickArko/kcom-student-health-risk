"""Kaggle Kernel for Student Health Risk Prediction.

Paste the contents of this file into a Kaggle Notebook cell and run.
Or upload as a script and run with: python kaggle_kernel.py

Evaluation metric: Balanced Accuracy (mean of recall per class).
"""

# === SETUP ===
from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Auto-detect Kaggle data path
KAGGLE_DATA = None

# 0. Use kagglehub to attach the competition (works in Kaggle Notebooks)
try:
    import kagglehub

    _p = kagglehub.competition_download("playground-series-s6e7")
    KAGGLE_DATA = Path(_p)
except Exception:
    pass

# 1. Check explicit candidates
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

# 2. Brute-force scan /kaggle/input/
if KAGGLE_DATA is None:
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        for subdir in sorted(input_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / "train.csv").exists():
                KAGGLE_DATA = subdir
                break

# 3. Final fallback
if KAGGLE_DATA is None:
    KAGGLE_DATA = Path("data")

os.system("pip install -q scikit-learn>=1.5 lightgbm>=4.3 scipy>=1.13 joblib>=1.3")

# === VENDORED CODE ===

import joblib
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("student_health")

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
TARGET_LABELS = ["fit", "at-risk", "unhealthy"]
N_CLASSES = 3
RANDOM_STATE = 42


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSV files."""
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    logger.info("Loaded train %s, test %s", train.shape, test.shape)
    return train, test


def preprocess(
    df: pd.DataFrame,
    *,
    is_train: bool = True,
    num_medians: dict | None = None,
    cat_modes: dict | None = None,
) -> pd.DataFrame:
    """Handle missing values. Returns processed frame."""
    df = df.copy()

    if is_train:
        num_medians = {}
        cat_modes = {}
        for col in NUM_COLS:
            if col in df.columns:
                med = df[col].median()
                num_medians[col] = med
                df[col] = df[col].fillna(med)
        for col in CAT_COLS:
            if col in df.columns:
                mode_val = df[col].mode(dropna=True)
                mode = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
                cat_modes[col] = mode
                df[col] = df[col].fillna(mode)
    else:
        for col in NUM_COLS:
            if col in df.columns and num_medians is not None and col in num_medians:
                df[col] = df[col].fillna(num_medians[col])
        for col in CAT_COLS:
            if col in df.columns and cat_modes is not None and col in cat_modes:
                df[col] = df[col].fillna(cat_modes[col])

    return df, num_medians, cat_modes


def encode_categoricals(
    df: pd.DataFrame, *, fit: bool = True, encoders: dict | None = None
) -> pd.DataFrame:
    """Label-encode categorical features."""
    df = df.copy()
    if fit:
        encoders = {}
        for col in CAT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col in CAT_COLS:
            if col in df.columns and encoders is not None and col in encoders:
                le = encoders[col]
                unseen = ~df[col].astype(str).isin(le.classes_)
                df.loc[unseen, col] = le.classes_[0]
                df[col] = le.transform(df[col].astype(str))
    return df, encoders


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple interaction features."""
    df = df.copy()
    if "bmi" in df.columns and "exercise_duration" in df.columns:
        df["bmi_exercise_interaction"] = df["bmi"] * df["exercise_duration"]
    if "step_count" in df.columns and "calorie_expenditure" in df.columns:
        df["efficiency_ratio"] = df["calorie_expenditure"] / (df["step_count"] + 1)
    if "heart_rate" in df.columns and "bmi" in df.columns:
        df["heart_bmi_ratio"] = df["heart_rate"] / (df["bmi"] + 1e-8)
    return df


def main() -> None:
    cfg = {
        "n_folds": 5,
        "lgbm_params": {
            "objective": "multiclass",
            "num_class": N_CLASSES,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        },
    }

    logger.info("=" * 60)
    logger.info("Student Health Risk Prediction — Baseline Kernel")
    logger.info("=" * 60)
    logger.info("Evaluation: Balanced Accuracy")

    # ── 1. Load data ──
    logger.info("[1/5] Loading data ...")
    t0 = time.time()
    train, test = load_data(KAGGLE_DATA)
    test_ids = test["id"].copy()
    logger.info("  Train: %s  Test: %s  (%.1fs)", train.shape, test.shape, time.time() - t0)

    # ── 2. Preprocess ──
    logger.info("[2/5] Preprocessing ...")
    t0 = time.time()
    train, num_medians, cat_modes = preprocess(train, is_train=True)
    test, _, _ = preprocess(test, is_train=False, num_medians=num_medians, cat_modes=cat_modes)

    # Encode categoricals
    train, encoders = encode_categoricals(train, fit=True)
    test, _ = encode_categoricals(test, fit=False, encoders=encoders)

    # Target encoding
    le_target = LabelEncoder()
    y = le_target.fit_transform(train[TARGET])

    # Feature engineering
    train = add_features(train)
    test = add_features(test)
    logger.info("  Features: %d  (%.1fs)", train.shape[1] - 2, time.time() - t0)

    # ── 3. Prepare feature matrix ──
    feature_cols = [c for c in train.columns if c not in ("id", TARGET)]
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # Align test columns
    for c in X.columns:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test = X_test[X.columns]

    # Downcast for memory
    for col in X.select_dtypes("float64").columns:
        X[col] = X[col].astype("float32")
    for col in X_test.select_dtypes("float64").columns:
        X_test[col] = X_test[col].astype("float32")

    # ── 4. Cross-validation training ──
    cv = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=RANDOM_STATE)
    logger.info("[3/5] Training with %d-fold CV ...", cfg["n_folds"])
    t0 = time.time()

    oof_preds = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    test_preds = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)
    oof_labels = np.zeros(len(X), dtype=np.int32)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**cfg["lgbm_params"])
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.log_evaluation(0)],
        )

        val_prob = model.predict_proba(X_val)
        oof_preds[val_idx] = val_prob
        oof_labels[val_idx] = val_prob.argmax(axis=1)
        test_preds += model.predict_proba(X_test) / cfg["n_folds"]

        fold_score = balanced_accuracy_score(y_val, oof_labels[val_idx])
        scores.append(fold_score)
        logger.info("  Fold %d/%d  Balanced Acc: %.4f", fold + 1, cfg["n_folds"], fold_score)

    oof_score = balanced_accuracy_score(y, oof_labels)
    logger.info(
        "  OOF Balanced Acc: %.4f (mean %.4f ± %.4f)", oof_score, np.mean(scores), np.std(scores)
    )
    logger.info("  Time: %.1fs", time.time() - t0)

    # ── 5. Generate submission ──
    logger.info("[4/5] Generating submission ...")
    test_labels = test_preds.argmax(axis=1)
    pred_labels = le_target.inverse_transform(test_labels)

    submission = pd.DataFrame({"id": test_ids, "health_condition": pred_labels})
    submission.to_csv("submission.csv", index=False)
    logger.info("  Submission saved: submission.csv (%d rows)", len(submission))
    pred_dist = submission["health_condition"].value_counts().to_dict()
    logger.info("  Predicted distribution: %s", pred_dist)

    # ── 6. Save model (optional) ──
    logger.info("[5/5] Saving model ...")
    model.fit(X, y)
    joblib.dump(
        {
            "model": model,
            "num_medians": num_medians,
            "cat_modes": cat_modes,
            "encoders": encoders,
            "le_target": le_target,
            "feature_cols": feature_cols,
        },
        "model.joblib",
    )
    logger.info("  Model saved: model.joblib")

    logger.info("Done ✓  OOF Balanced Acc: %.4f", oof_score)


if __name__ == "__main__":
    import lightgbm as lgb

    main()
