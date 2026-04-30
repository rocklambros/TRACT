"""Calibration diagnostic metrics: ECE, KS-test, coverage."""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from tract.config import PHASE1C_ECE_BOOTSTRAP_N, PHASE1C_ECE_N_BINS

logger = logging.getLogger(__name__)


def expected_calibration_error(
    confidences: NDArray[np.floating],
    accuracies: NDArray[np.floating],
    n_bins: int = PHASE1C_ECE_N_BINS,
) -> float:
    """Equal-width binned Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (n_bin / n) * abs(avg_acc - avg_conf)

    return float(ece)


def bootstrap_ece(
    confidences: NDArray[np.floating],
    accuracies: NDArray[np.floating],
    n_bins: int = PHASE1C_ECE_N_BINS,
    n_bootstrap: int = PHASE1C_ECE_BOOTSTRAP_N,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for ECE."""
    rng = np.random.default_rng(seed)
    n = len(confidences)
    ece_samples = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ece_samples[b] = expected_calibration_error(
            confidences[idx], accuracies[idx], n_bins=n_bins
        )

    point_ece = expected_calibration_error(confidences, accuracies, n_bins=n_bins)
    return {
        "ece": point_ece,
        "ci_low": float(np.percentile(ece_samples, 2.5)),
        "ci_high": float(np.percentile(ece_samples, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def ks_test_similarity_distributions(
    traditional_max_sims: NDArray[np.floating],
    ai_max_sims: NDArray[np.floating],
) -> dict:
    """Two-sample KS test between traditional and AI max-cosine similarity distributions."""
    stat, p_value = ks_2samp(traditional_max_sims, ai_max_sims)
    result = {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "n_traditional": len(traditional_max_sims),
        "n_ai": len(ai_max_sims),
    }

    if p_value < 0.01:
        logger.warning(
            "Domain mismatch: KS p=%.4f (stat=%.3f) between traditional (n=%d) "
            "and AI (n=%d) similarity distributions",
            p_value, stat, len(traditional_max_sims), len(ai_max_sims),
        )
    return result


def full_recall_coverage(
    prediction_sets: list[set[str]],
    valid_hub_sets: list[frozenset[str]],
) -> float:
    """Fraction of multi-label items where ALL valid hubs are in the prediction set."""
    if len(prediction_sets) != len(valid_hub_sets):
        raise ValueError("Length mismatch between prediction_sets and valid_hub_sets")
    n = len(prediction_sets)
    if n == 0:
        return 0.0
    covered = sum(
        1 for pset, vset in zip(prediction_sets, valid_hub_sets)
        if vset.issubset(pset)
    )
    return covered / n
