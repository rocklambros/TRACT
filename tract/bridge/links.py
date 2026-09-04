"""Tier-2 bridge links: human-curated traditional-control -> AI-hub mappings.

Phase 2C exists because the AI and traditional hub regions are disjoint. The
curated link set puts 78 hubs in the AI region and 380 in the traditional one,
and the intersection is exactly 0 -- so a model trained on it never sees a
traditional control positioned against an AI hub, and the PRD's bridging
capability has no supervision behind it.

These links are the supervision. They are Tier 2 under CAMPAIGN3.md Section 2:
independently human-authored, with no model output shown to the annotator. That
is a weaker claim than Tier 1 (OpenCRE-curated independently of TRACT) and a far
stronger one than Tier 3 (produced by, or ratified in the presence of, a model).

They are kept in their own file, and design decision D3 justifies that on the
grounds that the tier boundary is then a file boundary. `bridge_training_records`
is what makes that true past the boundary: it stamps every record with
`link_type` so `assign_quality_tier` can return T2 rather than falling through
to T1. Without the stamp a bridge link is a traditional framework with no
link_type, which is precisely the shape of an ordinary Tier-1 gold link.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from tract.config import BRIDGE_LINK_TYPE

if TYPE_CHECKING:
    from scripts.phase0.common import HubStandardLink

# The only tier a file of these may declare. A record claiming Tier 1 is
# claiming OpenCRE curated it, and a record claiming Tier 3 does not belong in
# a training corpus at all; both are rejected rather than coerced.
REQUIRED_TIER: Final[int] = 2

__all__ = [
    "BRIDGE_LINK_TYPE",
    "REQUIRED_TIER",
    "BridgeLink",
    "bridge_training_records",
    "load_bridge_links",
    "merge_for_training",
]


@dataclass(frozen=True)
class BridgeLink:
    """One annotator's mapping of a traditional control onto an AI hub."""

    framework_id: str
    standard_name: str
    section_id: str
    section_name: str
    cre_id: str
    tier: int
    annotator_id: str
    created_at: str
    confidence: int
    rationale: str


def load_bridge_links(path: Path) -> list[BridgeLink]:
    """Read and validate a JSONL file of bridge links.

    Raises rather than returning a partial list. An annotation round that
    silently dropped malformed records would report a smaller corpus as though
    it were the whole one, and the count is what Gate 1 is measured on.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Returning an empty list here would train "
            "with no bridge supervision while reporting success."
        )

    expected = {f.name for f in fields(BridgeLink)}
    out: list[BridgeLink] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {lineno}: not valid JSON") from exc

            present = set(record)
            missing = expected - present
            if missing:
                raise ValueError(
                    f"{path} line {lineno}: missing required field(s) "
                    f"{sorted(missing)}"
                )
            unknown = present - expected
            if unknown:
                # Not ignored. A misspelled field name silently discards the
                # annotator data it was carrying.
                raise ValueError(
                    f"{path} line {lineno}: unknown field(s) {sorted(unknown)}"
                )
            if record["tier"] != REQUIRED_TIER:
                raise ValueError(
                    f"{path} line {lineno}: tier is {record['tier']!r}, but "
                    f"this file may only hold tier {REQUIRED_TIER} links"
                )
            out.append(BridgeLink(**record))
    return out


def bridge_training_records(bridge: list[BridgeLink]) -> list[dict[str, str]]:
    """Bridge links as training-pipeline link dicts, carrying their tier.

    This is the form `filter_training_links` and `assign_quality_tier` consume.
    The `link_type` stamp is the whole point: without it a bridge link is a
    traditional standard_name with no link_type, `assign_quality_tier` falls
    through both branches, and it is tiered T1 -- which asserts OpenCRE curated
    it independently of TRACT.
    """
    return [
        {
            "cre_id": b.cre_id,
            "cre_name": "",
            "standard_name": b.standard_name,
            "section_id": b.section_id,
            "section_name": b.section_name,
            "link_type": BRIDGE_LINK_TYPE,
        }
        for b in bridge
    ]


def merge_for_training(
    curated: list[HubStandardLink], bridge: list[BridgeLink]
) -> list[HubStandardLink]:
    """Append bridge links to a curated set. TRAINING ONLY.

    Never pass the result to `build_evaluation_corpus` for a scored run. The
    147-item evaluation corpus is byte-identical either way -- bridge links name
    traditional sections, which the AI-framework filter excludes -- and
    `tests/test_bridge_links.py` asserts that on the real corpus rather than
    trusting the reasoning. The prohibition stands anyway: the property holds
    because of what today's bridge links happen to contain, and a future round
    that bridged an AI framework would break it silently.

    `HubStandardLink` has no tier field, so this conversion DROPS provenance.
    That is acceptable only on the corpus path, where nothing tiers anything.
    For the training pipeline use `bridge_training_records`, which keeps it.
    """
    from scripts.phase0.common import HubStandardLink

    return list(curated) + [
        HubStandardLink(
            cre_id=b.cre_id,
            cre_name="",
            standard_name=b.standard_name,
            section_id=b.section_id,
            section_name=b.section_name,
        )
        for b in bridge
    ]
