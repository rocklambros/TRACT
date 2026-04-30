"""Out-of-distribution detection for hub assignment.

OOD items (non-security content) get flagged and routed to
hub proposal pipeline (Phase 1D) instead of receiving an assignment.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from tract.config import PHASE1C_OOD_PERCENTILE, PHASE1C_OOD_SEPARATION_GATE

logger = logging.getLogger(__name__)


def compute_ood_threshold(
    max_sims: NDArray[np.floating],
    percentile: int = PHASE1C_OOD_PERCENTILE,
) -> float:
    """Compute OOD threshold as the p-th percentile of in-distribution max cosine similarities."""
    threshold = float(np.percentile(max_sims, percentile))
    logger.info(
        "OOD threshold: %.4f (%dth percentile of %d in-distribution items)",
        threshold, percentile, len(max_sims),
    )
    return threshold


def validate_ood_threshold(
    ood_max_sims: NDArray[np.floating],
    threshold: float,
    gate: float = PHASE1C_OOD_SEPARATION_GATE,
) -> dict:
    """Validate OOD threshold against synthetic non-security texts.

    Returns dict with separation_rate, n_below, n_total, gate_passed.
    """
    n_below = int((ood_max_sims < threshold).sum())
    n_total = len(ood_max_sims)
    separation_rate = n_below / n_total if n_total > 0 else 0.0
    passed = separation_rate >= gate

    if not passed:
        logger.warning(
            "OOD gate FAILED: %.1f%% separation (need ≥%.0f%%), threshold=%.4f",
            separation_rate * 100, gate * 100, threshold,
        )
    else:
        logger.info(
            "OOD gate passed: %.1f%% separation, threshold=%.4f",
            separation_rate * 100, threshold,
        )

    return {
        "separation_rate": separation_rate,
        "n_below": n_below,
        "n_total": n_total,
        "threshold": threshold,
        "gate_passed": passed,
    }


def flag_ood_items(
    max_sims: NDArray[np.floating],
    threshold: float,
) -> list[bool]:
    """Flag items as OOD if their max cosine similarity is below threshold."""
    return [bool(sim < threshold) for sim in max_sims]
