"""Test data module for Student Health Risk Prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_train(data_dir: Path | None = None) -> pd.DataFrame:
    """Load training data from CSV file."""
    data_dir = data_dir or Path("data/raw")
    return pd.read_csv(data_dir / "train.csv")


def split_train_val(df: pd.DataFrame, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets based on temporal split."""
    split_idx = int(len(df) * (1 - val_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def fill_missing(df: pd.DataFrame, strategy: str = "forward") -> pd.DataFrame:
    """Fill missing values in numeric columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["int64", "float64"]:
                if strategy == "forward":
                    df[col].fillna(method="ffill", inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
    return df


def winsorize(
    df: pd.DataFrame, feat_cols: list[str], lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    """Winsorize numeric feature columns."""
    df = df.copy()
    for col in feat_cols:
        if df[col].dtype in ["int64", "float64"]:
            lower_val = df[col].quantile(lower)
            upper_val = df[col].quantile(upper)
            df[col] = df[col].clip(lower=lower_val, upper=upper_val)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns from dataframe."""
    return [c for c in df.columns if c not in ["id", "health_risk", "target"]]
