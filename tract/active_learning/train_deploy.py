"""Deployment model training wrapper for Phase 1C."""
from __future__ import annotations

import logging

from tract.training.data_quality import TieredLink

logger = logging.getLogger(__name__)


def prepare_deployment_training_data(
    all_links: list[TieredLink],
    holdout_links: list[TieredLink],
) -> list[TieredLink]:
    """Remove holdout links from training data.

    Uses (section_name, cre_id) as the identity key for matching.
    """
    holdout_keys = {
        (l.link.get("section_name", ""), l.link.get("cre_id", ""))
        for l in holdout_links
    }

    remaining = [
        l for l in all_links
        if (l.link.get("section_name", ""), l.link.get("cre_id", "")) not in holdout_keys
    ]

    logger.info(
        "Deployment training data: %d links (%d removed as holdout)",
        len(remaining), len(all_links) - len(remaining),
    )
    return remaining
