"""Feature engineering for Student Health Risk Prediction."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple physics / lifestyle interaction features."""
    df = df.copy()
    if "bmi" in df.columns and "exercise_duration" in df.columns:
        df["bmi_exercise_interaction"] = df["bmi"] * df["exercise_duration"]
    if "step_count" in df.columns and "calorie_expenditure" in df.columns:
        df["efficiency_ratio"] = df["calorie_expenditure"] / (df["step_count"] + 1)
    if "heart_rate" in df.columns and "bmi" in df.columns:
        df["heart_bmi_ratio"] = df["heart_rate"] / (df["bmi"] + 1e-8)
    return df


class HealthPreprocessor:
    """Leakage-safe median/mode impute → label-encode → interactions."""

    def __init__(self):
        self.num_medians_: dict[str, float] = {}
        self.cat_modes_: dict[str, str] = {}
        self.encoders_: dict[str, LabelEncoder] = {}
        self.feature_cols_: list[str] = []

    def fit(self, df: pd.DataFrame) -> HealthPreprocessor:
        for col in NUM_COLS:
            if col in df.columns:
                self.num_medians_[col] = float(df[col].median())
        for col in CAT_COLS:
            if col in df.columns:
                mode_val = df[col].mode(dropna=True)
                self.cat_modes_[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "missing"
        # Fit encoders on imputed categoricals
        tmp = self._impute(df)
        for col in CAT_COLS:
            if col in tmp.columns:
                le = LabelEncoder()
                le.fit(tmp[col].astype(str))
                self.encoders_[col] = le
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._impute(df)
        for col, le in self.encoders_.items():
            if col not in out.columns:
                continue
            values = out[col].astype(str)
            unseen = ~values.isin(le.classes_)
            if unseen.any():
                values = values.copy()
                values.loc[unseen] = le.classes_[0]
            out[col] = le.transform(values)
        out = add_interactions(out)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_feature_matrix(
        self, df: pd.DataFrame, *, fit: bool = False
    ) -> tuple[pd.DataFrame, list[str]]:
        processed = self.fit_transform(df) if fit else self.transform(df)
        feature_cols = [c for c in processed.columns if c not in ("id", TARGET_COL)]
        self.feature_cols_ = feature_cols
        X = processed[feature_cols].copy()
        for col in X.select_dtypes("float64").columns:
            X[col] = X[col].astype("float32")
        return X, feature_cols

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, med in self.num_medians_.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)
        for col, mode in self.cat_modes_.items():
            if col in out.columns:
                out[col] = out[col].fillna(mode)
        return out


def build_features(df: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """Build features from raw data (legacy helper used by scripts/train.py)."""
    del train  # medians are fit on the passed frame
    df = df.copy()

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("missing")
            df[col] = df[col].astype("category")

    df = add_interactions(df)

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
