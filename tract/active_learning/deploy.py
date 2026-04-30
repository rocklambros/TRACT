"""Deployment model training, loading, and inference utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tract.config import (
    PHASE1C_HOLDOUT_CALIBRATION,
    PHASE1C_HOLDOUT_CANARY,
)
from tract.training.data_quality import AI_FRAMEWORK_NAMES, TieredLink

logger = logging.getLogger(__name__)


def select_holdout(
    tiered_links: list[TieredLink],
    n_cal: int = PHASE1C_HOLDOUT_CALIBRATION,
    n_canary: int = PHASE1C_HOLDOUT_CANARY,
    seed: int = 42,
) -> tuple[list[TieredLink], list[TieredLink], list[TieredLink]]:
    """Select holdout links for calibration and canaries.

    Only traditional (non-AI) links are eligible for holdout.

    Returns:
        (calibration_links, canary_links, remaining_links)
    """
    traditional = [l for l in tiered_links if l.link.get("standard_name", "") not in AI_FRAMEWORK_NAMES]
    ai_links = [l for l in tiered_links if l.link.get("standard_name", "") in AI_FRAMEWORK_NAMES]

    n_total = n_cal + n_canary
    if len(traditional) < n_total:
        raise ValueError(
            f"Not enough traditional links for holdout: need {n_total}, have {len(traditional)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(traditional))

    holdout_indices = set(indices[:n_total].tolist())
    cal_indices = indices[:n_cal]
    canary_indices = indices[n_cal:n_total]

    calibration = [traditional[i] for i in cal_indices]
    canaries = [traditional[i] for i in canary_indices]
    remaining_trad = [traditional[i] for i in range(len(traditional)) if i not in holdout_indices]

    remaining = remaining_trad + ai_links

    fw_counts: dict[str, int] = {}
    for link in calibration:
        fw = link.link.get("standard_name", "unknown")
        fw_counts[fw] = fw_counts.get(fw, 0) + 1
    max_fw_pct = max(fw_counts.values()) / n_cal if fw_counts else 0
    if max_fw_pct > 0.5:
        logger.warning(
            "Holdout dominated by single framework: %s (%.0f%%)",
            max(fw_counts, key=fw_counts.get), max_fw_pct * 100,
        )

    logger.info(
        "Holdout: %d calibration + %d canary from %d traditional links, %d remaining",
        len(calibration), len(canaries), len(traditional), len(remaining),
    )
    return calibration, canaries, remaining


def holdout_to_eval(link: TieredLink) -> dict:
    """Convert a TieredLink to an eval-compatible record.

    Returns dict with keys: control_text, framework, valid_hub_ids.
    """
    control_text = link.link.get("section_name") or link.link.get("section_id", "")
    section_id = link.link.get("section_id", "")
    if section_id and section_id not in control_text:
        control_text = f"{section_id}: {control_text}"

    return {
        "control_text": control_text,
        "framework": link.link.get("standard_name", ""),
        "valid_hub_ids": frozenset({link.link["cre_id"]}),
    }
