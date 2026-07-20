"""Unit tests for stacking ensemble and preprocessor (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from student_health.features import CAT_COLS, NUM_COLS, TARGET_COL, HealthPreprocessor
from student_health.models import (
    ModelWeightMeta,
    SimpleAverageMeta,
    StackingEnsemble,
    build_meta_model,
    default_stack_base_models,
)


def _synthetic(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(
        {
            "id": np.arange(n),
            "sleep_duration": rng.normal(7, 1.5, n),
            "heart_rate": rng.normal(70, 10, n),
            "bmi": rng.normal(24, 4, n),
            "calorie_expenditure": rng.normal(2000, 400, n),
            "step_count": rng.normal(8000, 2000, n),
            "exercise_duration": rng.normal(40, 15, n),
            "water_intake": rng.normal(2.0, 0.5, n),
            "diet_type": rng.choice(["veg", "non-veg", "vegan"], n),
            "stress_level": rng.choice(["low", "medium", "high"], n),
            "sleep_quality": rng.choice(["poor", "average", "good"], n),
            "physical_activity_level": rng.choice(["low", "moderate", "high"], n),
            "smoking_alcohol": rng.choice(["none", "occasional", "regular"], n),
            "gender": rng.choice(["male", "female"], n),
        }
    )
    # Inject missingness
    for col in NUM_COLS[:2]:
        train.loc[rng.choice(n, size=n // 10, replace=False), col] = np.nan
    for col in CAT_COLS[:2]:
        train.loc[rng.choice(n, size=n // 10, replace=False), col] = np.nan

    # Label roughly from BMI + stress for signal
    score = train["bmi"].fillna(24) + (train["stress_level"] == "high").astype(float) * 5
    labels = np.where(score < 22, "fit", np.where(score > 28, "unhealthy", "at-risk"))
    train[TARGET_COL] = labels

    test = train.drop(columns=[TARGET_COL]).copy()
    test["id"] = np.arange(n, 2 * n)
    return train, test


class TestHealthPreprocessor:
    def test_fit_transform_shapes(self):
        train, test = _synthetic()
        prep = HealthPreprocessor()
        X, cols = prep.get_feature_matrix(train, fit=True)
        X_test, _ = prep.get_feature_matrix(test, fit=False)
        X_test = X_test.reindex(columns=cols, fill_value=0)
        assert X.shape[0] == len(train)
        assert X_test.shape[1] == X.shape[1]
        assert TARGET_COL not in X.columns
        assert "bmi_exercise_interaction" in cols

    def test_no_nan_after_transform(self):
        train, _ = _synthetic()
        prep = HealthPreprocessor()
        X, _ = prep.get_feature_matrix(train, fit=True)
        assert not X.isna().any().any()


class TestMetaModels:
    def test_model_weight_meta_improves_or_matches_uniform(self):
        rng = np.random.default_rng(0)
        n, n_classes = 300, 3
        y = rng.integers(0, n_classes, size=n)
        # One strong model (near-perfect), two weak
        strong = np.eye(n_classes)[y] * 0.9 + 0.05
        weak = rng.random((n, n_classes))
        weak = weak / weak.sum(axis=1, keepdims=True)
        X = np.hstack([strong, weak, weak])
        meta = ModelWeightMeta(n_trials=200, random_state=0)
        meta.fit(X, y)
        assert meta.weights_[0] >= meta.weights_[1] - 1e-9
        assert meta.best_score_ >= 0.5

    def test_simple_average_meta(self):
        # Two models × three classes; y must include all classes so n_classes_ is correct
        X = np.array(
            [
                [0.8, 0.1, 0.1, 0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1, 0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8, 0.1, 0.2, 0.7],
            ],
            dtype=float,
        )
        y = np.array([0, 1, 2])
        meta = SimpleAverageMeta().fit(X, y)
        proba = meta.predict_proba(X)
        assert proba.shape == (3, 3)
        np.testing.assert_allclose(proba[0], [0.75, 0.15, 0.1])


class TestStackingEnsemble:
    def test_fit_predict_synthetic(self):
        train, test = _synthetic(n=300)
        prep = HealthPreprocessor()
        X, cols = prep.get_feature_matrix(train, fit=True)
        X_test, _ = prep.get_feature_matrix(test, fit=False)
        X_test = X_test.reindex(columns=cols, fill_value=0)
        y = train[TARGET_COL]

        base_models = [
            (
                "hgbc",
                HistGradientBoostingClassifier(
                    max_iter=40,
                    learning_rate=0.1,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=40,
                    learning_rate=0.1,
                    class_weight="balanced",
                    verbose=-1,
                    random_state=0,
                ),
            ),
        ]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        ens = StackingEnsemble(base_models, meta_model=ModelWeightMeta(n_trials=50))
        ens.fit(X, y, cv=cv, X_test=X_test)

        assert ens.overall_oof_score_ is not None
        assert 0.3 <= ens.overall_oof_score_ <= 1.0
        assert len(ens.valid_scores_) == 3
        assert set(ens.per_model_oof_scores_) == {"hgbc", "lgbm"}

        preds = ens.predict()
        assert len(preds) == len(test)
        assert set(preds).issubset({"fit", "at-risk", "unhealthy"})

        proba = ens.predict_proba()
        assert proba.shape == (len(test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_default_stack_builds(self):
        models = default_stack_base_models(n_estimators=10)
        assert [n for n, _ in models] == ["hgbc", "catboost", "xgb", "lgbm"]

    def test_build_meta_model_names(self):
        assert isinstance(build_meta_model("model_weight"), ModelWeightMeta)
        assert isinstance(build_meta_model("simple_average"), SimpleAverageMeta)
