"""Classify CRE hubs as AI-only, traditional-only, naturally bridged, or unlinked."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tract.config import BRIDGE_AI_FRAMEWORK_IDS
from tract.io import load_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubClassification:
    ai_only: list[str]
    trad_only: list[str]
    naturally_bridged: list[str]
    unlinked: list[str]


def classify_hubs(
    hub_links_path: Path | str,
    all_hub_ids: list[str],
) -> HubClassification:
    """Classify hubs by which framework types link to them.

    Args:
        hub_links_path: Path to hub_links_by_framework.json
            (dict keyed by framework_id, values are lists of link dicts with 'cre_id')
        all_hub_ids: All hub IDs with embeddings (from deployment_artifacts.npz)

    Returns:
        HubClassification with sorted lists of hub IDs per category.
    """
    hub_links: dict[str, list[dict]] = load_json(hub_links_path)

    ai_hubs: set[str] = set()
    trad_hubs: set[str] = set()

    for framework_id, links in hub_links.items():
        for link in links:
            cre_id = link["cre_id"]
            if framework_id in BRIDGE_AI_FRAMEWORK_IDS:
                ai_hubs.add(cre_id)
            else:
                trad_hubs.add(cre_id)

    all_hub_set = set(all_hub_ids)
    ai_hubs &= all_hub_set
    trad_hubs &= all_hub_set
    linked_hubs = ai_hubs | trad_hubs

    result = HubClassification(
        ai_only=sorted(ai_hubs - trad_hubs),
        trad_only=sorted(trad_hubs - ai_hubs),
        naturally_bridged=sorted(ai_hubs & trad_hubs),
        unlinked=sorted(all_hub_set - linked_hubs),
    )

    logger.info(
        "Hub classification: %d AI-only, %d trad-only, %d bridged, %d unlinked",
        len(result.ai_only), len(result.trad_only),
        len(result.naturally_bridged), len(result.unlinked),
    )
    return result
