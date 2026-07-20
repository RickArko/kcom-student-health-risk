"""Preprocessing facade — re-exports the leakage-safe HealthPreprocessor."""

from __future__ import annotations

from student_health.features import (  # noqa: F401
    CAT_COLS,
    MISS_INDICATOR_COLS,
    N_CLASSES,
    NUM_COLS,
    TARGET_COL,
    TARGET_LABELS,
    TARGET_MAPPING,
    HealthPreprocessor,
    add_interactions,
    add_missing_indicators,
    build_features,
    get_X_y,
)
