"""Conformal prediction sets for hub assignment.

Provides empirical coverage guarantees (not mathematical — LOFO models != deployment model
violates exchangeability). Conservative bias is expected and documented.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from tract.config import PHASE1C_CONFORMAL_ALPHA

logger = logging.getLogger(__name__)


def compute_conformal_quantile(
    probs: NDArray[np.floating],
    valid_hub_indices: list[list[int]],
    alpha: float = PHASE1C_CONFORMAL_ALPHA,
) -> float:
    """Compute conformal quantile from multi-label nonconformity scores.

    Nonconformity score: 1 - sum(P(hub) for hub in valid_hubs)
    Quantile: ceil((n+1)*(1-alpha))/n percentile of scores.
    """
    n = len(probs)
    scores = np.empty(n)
    for i, valid_indices in enumerate(valid_hub_indices):
        p_valid = sum(probs[i, j] for j in valid_indices)
        scores[i] = 1.0 - p_valid

    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    quantile = float(np.quantile(scores, q_level))

    logger.info(
        "Conformal quantile: %.4f (alpha=%.2f, n=%d, q_level=%.4f)",
        quantile, alpha, n, q_level,
    )
    return quantile


def build_prediction_sets(
    probs: NDArray[np.floating],
    hub_ids: list[str],
    quantile: float,
) -> list[set[str]]:
    """Build prediction sets: {hub : P(hub) >= 1-quantile}.

    Args:
        probs: (n, n_hubs) calibrated probabilities.
        hub_ids: Ordered hub IDs matching probs columns.
        quantile: Conformal quantile from compute_conformal_quantile.

    Returns:
        List of sets, one per item.
    """
    threshold = 1.0 - quantile
    result: list[set[str]] = []
    for i in range(len(probs)):
        pset = {hub_ids[j] for j in range(len(hub_ids)) if probs[i, j] >= threshold}
        result.append(pset)
    return result


def compute_conformal_coverage(
    prediction_sets: list[set[str]],
    valid_hub_sets: list[frozenset[str]],
) -> float:
    """Empirical coverage: fraction where ANY valid hub is in prediction set."""
    n = len(prediction_sets)
    if n == 0:
        return 0.0
    covered = sum(
        1 for pset, vset in zip(prediction_sets, valid_hub_sets)
        if pset & vset
    )
    return covered / n
