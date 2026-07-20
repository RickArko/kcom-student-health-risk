"""Tests for BA thresholds, blending, class order, and missing indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from student_health.ensemble import blend_probas, fit_blend, hillclimb_weights
from student_health.features import (
    TARGET_LABELS,
    TARGET_MAPPING,
    HealthPreprocessor,
    add_missing_indicators,
)
from student_health.metrics import (
    apply_thresholds,
    ba_from_proba,
    decode_labels,
    encode_labels,
    tune_thresholds,
)


class TestClassOrder:
    def test_mapping_matches_label_list(self):
        assert [TARGET_MAPPING[c] for c in TARGET_LABELS] == [0, 1, 2]
        assert TARGET_MAPPING["fit"] == 0
        assert TARGET_MAPPING["at-risk"] == 1
        assert TARGET_MAPPING["unhealthy"] == 2

    def test_encode_decode_roundtrip(self):
        y = pd.Series(["unhealthy", "fit", "at-risk", "fit"])
        enc = encode_labels(y, TARGET_MAPPING)
        assert enc.tolist() == [2, 0, 1, 0]
        back = decode_labels(enc, TARGET_LABELS)
        assert back.tolist() == y.tolist()


class TestThresholds:
    def test_tune_thresholds_beats_or_matches_argmax(self):
        rng = np.random.default_rng(0)
        n, n_classes = 500, 3
        # Imbalanced labels
        y = rng.choice([0, 1, 2], size=n, p=[0.1, 0.8, 0.1])
        # Noisy probs biased toward majority
        proba = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float64)
        proba[:, 1] += 0.3
        proba = proba / proba.sum(axis=1, keepdims=True)
        # Inject signal for minorities
        for i, yi in enumerate(y):
            if yi != 1:
                proba[i, yi] += 0.4
        proba = proba / proba.sum(axis=1, keepdims=True)

        argmax_ba = ba_from_proba(y, proba)
        thresholds, tuned_ba = tune_thresholds(y, proba)
        assert tuned_ba >= argmax_ba - 1e-12
        preds = apply_thresholds(proba, thresholds)
        assert preds.shape == (n,)


class TestBlend:
    def test_weights_sum_to_one_and_beat_weak_models(self):
        rng = np.random.default_rng(1)
        n, n_classes = 400, 3
        y = rng.integers(0, n_classes, size=n)
        strong = np.eye(n_classes)[y] * 0.85 + 0.05
        weak = rng.random((n, n_classes))
        weak = weak / weak.sum(axis=1, keepdims=True)
        weights, ba = hillclimb_weights(y, [strong, weak, weak], n_trials=300, random_state=0)
        assert abs(weights.sum() - 1.0) < 1e-9
        assert weights[0] >= weights[1] - 1e-9
        assert ba >= ba_from_proba(y, weak) - 1e-12

    def test_fit_blend_rejects_worse_blend(self):
        rng = np.random.default_rng(2)
        n, n_classes = 300, 3
        y = rng.integers(0, n_classes, size=n)
        strong = np.eye(n_classes)[y] * 0.9 + 0.033
        # Correlated near-copy — blend should still accept or fall back cleanly
        result = fit_blend(
            y,
            {"strong": strong, "copy": strong.copy()},
            {"strong": strong, "copy": strong.copy()},
            n_trials=50,
            tune_thresh=True,
        )
        assert abs(result.weights.sum() - 1.0) < 1e-9
        assert result.oof_ba_tuned >= result.single_best_ba - 1e-12
        blended = blend_probas([strong, strong], result.weights)
        assert blended.shape == strong.shape


class TestMissingIndicators:
    def test_stress_missing_flag_present(self):
        n = 40
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "sleep_duration": np.linspace(5, 9, n),
                "heart_rate": np.linspace(60, 90, n),
                "bmi": np.linspace(20, 30, n),
                "calorie_expenditure": np.linspace(1500, 2500, n),
                "step_count": np.linspace(4000, 12000, n),
                "exercise_duration": np.linspace(10, 60, n),
                "water_intake": np.linspace(1, 3, n),
                "diet_type": ["veg"] * n,
                "stress_level": ["low"] * n,
                "sleep_quality": ["good"] * n,
                "physical_activity_level": ["moderate"] * n,
                "smoking_alcohol": ["none"] * n,
                "gender": ["female"] * n,
                "health_condition": ["at-risk"] * n,
            }
        )
        df.loc[:5, "stress_level"] = np.nan
        flagged = add_missing_indicators(df)
        assert flagged["stress_level_missing"].sum() == 6

        prep = HealthPreprocessor(missing_indicators=True)
        X, cols = prep.get_feature_matrix(df, fit=True)
        assert "stress_level_missing" in cols
        assert not X.isna().any().any()
