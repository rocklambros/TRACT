"""Canary item management for active learning quality measurement.

Two canary types:
1. Pre-labeled AI controls (from unmapped pool, expert labels BEFORE seeing predictions)
2. Traditional holdout canaries (from 20-item partition, ground truth from OpenCRE)
"""
from __future__ import annotations

import logging

import numpy as np

from tract.config import PHASE1C_N_AI_CANARIES

logger = logging.getLogger(__name__)


def select_ai_canaries(
    unmapped_controls: list[dict],
    n: int = PHASE1C_N_AI_CANARIES,
    seed: int = 42,
) -> list[dict]:
    """Select n controls from the unmapped pool for canary pre-labeling.

    Returns the selected controls (expert must label these before AL begins).
    """
    if len(unmapped_controls) < n:
        raise ValueError(
            f"Not enough unmapped controls for canaries: need {n}, have {len(unmapped_controls)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(unmapped_controls), size=n, replace=False)
    selected = [unmapped_controls[i] for i in sorted(indices)]

    logger.info("Selected %d AI canary controls for pre-labeling", len(selected))
    return selected


def evaluate_canary_accuracy(
    canary_labels: dict[str, frozenset[str]],
    review_items: list[dict],
) -> float:
    """Evaluate expert accuracy on canary items.

    Compares expert's accepted/corrected hub against the pre-labeled ground truth.
    Only counts items the expert actually reviewed (not skipped).
    """
    canary_ids = set(canary_labels.keys())
    correct = 0
    total = 0

    for item in review_items:
        cid = item["control_id"]
        if cid not in canary_ids:
            continue

        review = item.get("review")
        if review is None or review["status"] == "rejected":
            continue

        total += 1
        if review["status"] == "corrected":
            assigned_hub = review.get("corrected_hub_id", "")
        else:
            preds = item.get("predictions", [])
            assigned_hub = preds[0]["hub_id"] if preds else ""

        if assigned_hub in canary_labels[cid]:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("Canary accuracy: %.1f%% (%d/%d)", accuracy * 100, correct, total)
    return accuracy
