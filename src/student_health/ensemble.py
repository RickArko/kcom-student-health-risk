"""OOF probability blending with hill-climbing weights and BA thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from student_health.metrics import apply_thresholds, ba_from_proba, tune_thresholds

logger = logging.getLogger(__name__)


@dataclass
class BlendResult:
    weights: np.ndarray
    oof_proba: np.ndarray
    test_proba: np.ndarray | None
    oof_ba_argmax: float
    thresholds: np.ndarray
    oof_ba_tuned: float
    model_names: list[str]
    single_best_ba: float
    single_best_name: str
    accepted: bool


def _normalize(weights: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def blend_probas(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted average of probability arrays (same shape)."""
    w = _normalize(weights)
    out = np.zeros_like(probas[0], dtype=np.float64)
    for wi, p in zip(w, probas):
        out += wi * p
    return out.astype(np.float32)


def hillclimb_weights(
    y_true: np.ndarray,
    oof_probas: list[np.ndarray],
    *,
    n_trials: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Search simplex weights maximizing OOF balanced accuracy (argmax).

    Candidates: uniform, one-hot (single models), and Dirichlet samples.
    """
    n_models = len(oof_probas)
    stacked = np.stack(oof_probas, axis=0)  # (M, N, C)

    candidates = [np.ones(n_models) / n_models]
    for i in range(n_models):
        w = np.zeros(n_models)
        w[i] = 1.0
        candidates.append(w)

    rng = np.random.default_rng(random_state)
    candidates.extend(rng.dirichlet(np.ones(n_models), size=n_trials))

    best_w = candidates[0]
    best_ba = -1.0
    for w in candidates:
        blend = np.tensordot(_normalize(w), stacked, axes=(0, 0))
        ba = float(balanced_accuracy_score(y_true, blend.argmax(axis=1)))
        if ba > best_ba:
            best_ba = ba
            best_w = _normalize(w)

    return best_w.astype(np.float64), best_ba


def fit_blend(
    y_true: np.ndarray,
    oof_probas: dict[str, np.ndarray],
    test_probas: dict[str, np.ndarray] | None = None,
    *,
    n_trials: int = 2000,
    random_state: int = 42,
    tune_thresh: bool = True,
) -> BlendResult:
    """Hill-climb blend + optional BA threshold tuning.

    Rejects the blend (falls back to best single model) if it does not beat
    the best single-model OOF BA.
    """
    names = list(oof_probas.keys())
    oof_list = [oof_probas[n] for n in names]

    single_scores = {n: ba_from_proba(y_true, oof_probas[n]) for n in names}
    best_name = max(single_scores, key=single_scores.get)
    best_single = single_scores[best_name]
    logger.info("Single-model OOF BA: %s", {k: f"{v:.4f}" for k, v in single_scores.items()})

    weights, blend_ba = hillclimb_weights(
        y_true, oof_list, n_trials=n_trials, random_state=random_state
    )
    oof_blend = blend_probas(oof_list, weights)
    logger.info(
        "Hill-climb blend OOF BA: %.4f  weights=%s",
        blend_ba,
        {n: float(w) for n, w in zip(names, weights)},
    )

    accepted = blend_ba >= best_single - 1e-12
    if not accepted:
        logger.warning(
            "Blend (%.4f) did not beat best single %s (%.4f) — falling back",
            blend_ba,
            best_name,
            best_single,
        )
        weights = np.array([1.0 if n == best_name else 0.0 for n in names], dtype=np.float64)
        oof_blend = oof_probas[best_name].copy()
        blend_ba = best_single

    thresholds = np.ones(oof_blend.shape[1], dtype=np.float64)
    tuned_ba = blend_ba
    if tune_thresh:
        thresholds, tuned_ba = tune_thresholds(y_true, oof_blend)
        logger.info(
            "Threshold-tuned OOF BA: %.4f  thresholds=%s",
            tuned_ba,
            thresholds.tolist(),
        )

    test_blend = None
    if test_probas is not None:
        test_list = [test_probas[n] for n in names]
        test_blend = blend_probas(test_list, weights)

    return BlendResult(
        weights=weights,
        oof_proba=oof_blend,
        test_proba=test_blend,
        oof_ba_argmax=blend_ba,
        thresholds=thresholds,
        oof_ba_tuned=tuned_ba,
        model_names=names,
        single_best_ba=best_single,
        single_best_name=best_name,
        accepted=accepted,
    )


def predict_with_blend(result: BlendResult, test_proba: np.ndarray | None = None) -> np.ndarray:
    """Apply frozen thresholds to blended test probabilities."""
    proba = test_proba if test_proba is not None else result.test_proba
    if proba is None:
        raise ValueError("No test probabilities available")
    return apply_thresholds(proba, result.thresholds)
