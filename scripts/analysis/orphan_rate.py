"""Count AI hubs with no traditional supervision. Phase 2C's Gate 1 statistic.

An AI hub is an ORPHAN under the strict all-AI firewall when no traditional
framework links to it. Hold every AI framework out -- which is what the strict
firewall does -- and nothing remains that positions that hub for the model. The
78 AI hubs are today orphaned 78 of 78, which is the same fact as "the AI and
traditional hub regions are disjoint" and is the reason Phase 2C exists.

Gate 1 is a reduction in this count. It is free to compute, needs no model, and
answers a question about the corpus rather than about a run.

TWO AI-FRAMEWORK DEFINITIONS EXIST. `AI_FRAMEWORK_NAMES` in scripts/phase0
is the five-framework LOFO eval roster. `BRIDGE_AI_FRAMEWORK_IDS` is the
eight-framework AI region, adding ENISA, ETSI and BIML. The region is the right
one here, and the difference is not cosmetic:

    5-framework roster : 73 AI hubs, 57 of them apparently bridged
    8-framework region : 78 AI hubs, 0 bridged

All 57 of those apparent bridges are supplied by ENISA (51), ETSI (28) or BIML
(11). No traditional framework links to any AI hub. The narrow definition is
what manufactures the appearance of existing bridges, so this module keys on
framework ids and takes the region.

Read-only. Loads no model and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from tract.config import BRIDGE_AI_FRAMEWORK_IDS, TRAINING_DIR

if TYPE_CHECKING:
    from tract.bridge.links import BridgeLink

logger = logging.getLogger(__name__)

CURATED_BY_FRAMEWORK_PATH: Final[Path] = (
    TRAINING_DIR / "hub_links_by_framework_curated.json"
)

# One (framework_id, cre_id) edge.
LinkPair = tuple[str, str]


def load_framework_hub_links(
    path: Path = CURATED_BY_FRAMEWORK_PATH,
) -> list[LinkPair]:
    """Every (framework_id, hub_id) edge in the curated link set.

    The by-framework file rather than hub_links_curated.jsonl: both hold the
    same 4,405 links, but this one is keyed by framework_id, which is the key
    BRIDGE_AI_FRAMEWORK_IDS is expressed in. Going through standard_name would
    need a third copy of the AI-framework list to map across.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; cannot count orphans.")
    payload: dict[str, list[dict[str, str]]] = json.loads(
        path.read_text(encoding="utf-8")
    )
    return [
        (framework_id, link["cre_id"])
        for framework_id, links in payload.items()
        for link in links
    ]


def bridge_link_pairs(bridge: Iterable[BridgeLink]) -> list[LinkPair]:
    """Bridge links as (framework_id, hub_id) edges, for the same count."""
    return [(b.framework_id, b.cre_id) for b in bridge]


def strict_firewall_orphans(
    links: Iterable[LinkPair],
    ai_framework_ids: Iterable[str] = BRIDGE_AI_FRAMEWORK_IDS,
) -> tuple[int, int]:
    """(orphaned, total) distinct AI gold hubs.

    An AI hub is any hub some AI framework links to. It is orphaned when no
    non-AI framework links to it. Counted over HUBS, not links, so mapping many
    controls onto one hub rescues it exactly once -- otherwise a productive
    annotator could clear Gate 1 without widening coverage at all.

    Raises on input it cannot measure. An empty set would return (0, 0), which
    reads as every hub rescued.
    """
    ai_ids = frozenset(ai_framework_ids)
    frameworks_by_hub: dict[str, set[str]] = defaultdict(set)
    for framework_id, hub_id in links:
        frameworks_by_hub[hub_id].add(framework_id)

    if not frameworks_by_hub:
        raise ValueError(
            "Refusing to count orphans over no links: the result would be "
            "(0, 0), which reads as a perfect score."
        )

    ai_hubs = {
        hub for hub, frameworks in frameworks_by_hub.items() if frameworks & ai_ids
    }
    if not ai_hubs:
        raise ValueError(
            "No AI-framework link is present, so there are no AI hubs to be "
            f"orphaned. Expected some of {sorted(ai_ids)}."
        )

    orphaned = {hub for hub in ai_hubs if not (frameworks_by_hub[hub] - ai_ids)}
    return len(orphaned), len(ai_hubs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--links",
        type=Path,
        default=CURATED_BY_FRAMEWORK_PATH,
        help="Curated links grouped by framework id.",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=None,
        help=(
            "Tier-2 bridge link JSONL to include. NOTE: this counts EVERY "
            "link, including those below the Gate 1 confidence floor, because "
            "this module is the raw graph arithmetic. It is not the gate. Use "
            "scripts/analysis/gate1_report.py for a Gate 1 verdict."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pairs = load_framework_hub_links(args.links)
    if args.bridge is not None:
        from tract.bridge.links import load_bridge_links

        bridge = load_bridge_links(args.bridge)
        pairs = pairs + bridge_link_pairs(bridge)
        logger.info("Included %d bridge links from %s", len(bridge), args.bridge)
        logger.warning(
            "This is the raw orphan rate over ALL links, with no confidence "
            "floor applied. It is NOT the Gate 1 verdict: Gate 1 also binds "
            "Q1-Q4, and a corpus of low-confidence links moves this number "
            "while counting for nothing. Run "
            "`python -m scripts.analysis.gate1_report %s` instead.",
            args.bridge,
        )

    orphaned, total = strict_firewall_orphans(pairs)
    logger.info(
        "AI hubs with no traditional supervision: %d of %d (%.1f%%)",
        orphaned, total, 100 * orphaned / total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
