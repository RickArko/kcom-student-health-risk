"""Feature engineering for Student Health Risk Prediction."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

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
TARGET_COL = "health_condition"

TARGET_MAPPING = {"fit": 0, "at-risk": 1, "unhealthy": 2}


def build_features(df: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """Build features from raw data."""
    df = df.copy()

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("missing")
            df[col] = df[col].astype("category")

    if "bmi" in df.columns and "exercise_duration" in df.columns:
        df["bmi_exercise_interaction"] = df["bmi"] * df["exercise_duration"]

    if "step_count" in df.columns and "calorie_expenditure" in df.columns:
        df["efficiency_ratio"] = df["calorie_expenditure"] / (df["step_count"] + 1)

    logger.info(
        "Built features: %d columns, %d numeric, %d categorical",
        df.shape[1],
        len([c for c in NUM_COLS if c in df.columns]),
        len([c for c in CAT_COLS if c in df.columns]),
    )
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from dataframe."""
    X = df.drop(columns=["id", TARGET_COL], errors="ignore")
    y = df[TARGET_COL].map(TARGET_MAPPING)
    return X, y
