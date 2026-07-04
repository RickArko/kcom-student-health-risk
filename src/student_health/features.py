"""Feature engineering for Student Health Risk Prediction."""

from __future__ import annotations

import logging
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """Build features from raw data."""
    df = df.copy()

    # Engineered features
    df["age_group"] = pd.cut(df["age"], bins=[0, 10, 18, 25, 40, 100], labels=["child", "teen", "young", "adult", "senior"])
    df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 50], labels=["underweight", "normal", "overweight", "obese"])

    # Interaction features
    df["age_bmi_interaction"] = df["age"] * df["bmi"]
    df["stress_activity_interaction"] = df["stress_level"] * df["physical_activity_level"]

    # Categorical encoding
    df["age_group_cat"] = df["age_group"].astype("category")
    df["bmi_category_cat"] = df["bmi_category"].astype("category")

    # Derived scores
    df["health_score"] = (
        df["sleep_hours"] / 10
        + df["physical_activity_level"] / 10
        - df["stress_level"] / 10
        + df["diet_quality_score"] / 10
    )

    df = df.drop(columns=["age_group", "bmi_category"])

    feature_cols = [c for c in df.columns if not c.startswith("id") and c != "target"]

    if train:
        logger.info("Built features from %d columns to %d columns", len(feature_cols), len(feature_cols) - 1)

    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from dataframe."""
    X = df.drop(columns=["id", "target"])
    y = df["target"]
    return X, y
