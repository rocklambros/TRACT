"""Training data quality pipeline.

Filters curated hub links by quality, assigns tier metadata,
and computes data hash chain for provenance tracking.

Quality tiers:
  T1     — Human LinkedTo with descriptive text (traditional frameworks)
  T1-AI  — Human-curated AI framework links
  T3     — AutomaticallyLinkedTo with descriptive text
  DROPPED — Bare-ID, short text, or from dropped frameworks
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tract.config import (
    PHASE1B_DROPPED_FRAMEWORKS,
    PHASE1B_MIN_SECTION_TEXT_LENGTH,
    TRAINING_DIR,
)

logger = logging.getLogger(__name__)

AI_FRAMEWORK_NAMES: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS",
    "NIST AI 100-2",
    "OWASP AI Exchange",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
})

CURATED_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"
TRAINING_OUTPUT_PATH: Final[Path] = TRAINING_DIR / "hub_links_training.jsonl"


class QualityTier(enum.Enum):
    T1 = "T1"
    T1_AI = "T1-AI"
    T3 = "T3"
    DROPPED = "DROPPED"


@dataclass(frozen=True)
class TieredLink:
    link: dict[str, str]
    tier: QualityTier


def _has_descriptive_text(link: dict[str, str]) -> bool:
    """Check if section_name has enough descriptive content."""
    section = link.get("section_name", "")
    return len(section) >= PHASE1B_MIN_SECTION_TEXT_LENGTH


def assign_quality_tier(link: dict[str, str]) -> QualityTier:
    """Assign a quality tier to a single hub link."""
    framework_id = link.get("framework_id", "")
    standard_name = link.get("standard_name", "")
    link_type = link.get("link_type", "")

    if framework_id in PHASE1B_DROPPED_FRAMEWORKS:
        return QualityTier.DROPPED

    if not _has_descriptive_text(link):
        return QualityTier.DROPPED

    if standard_name in AI_FRAMEWORK_NAMES:
        return QualityTier.T1_AI

    if link_type == "AutomaticallyLinkedTo":
        return QualityTier.T3

    return QualityTier.T1


def filter_training_links(links: list[dict[str, str]]) -> list[TieredLink]:
    """Filter links by quality and assign tier metadata.

    Returns non-DROPPED links with their tier assignment.
    """
    result: list[TieredLink] = []
    tier_counts: dict[QualityTier, int] = {t: 0 for t in QualityTier}

    for link in links:
        tier = assign_quality_tier(link)
        tier_counts[tier] += 1
        if tier != QualityTier.DROPPED:
            result.append(TieredLink(link=link, tier=tier))

    for tier, count in tier_counts.items():
        logger.info("Quality tier %s: %d links", tier.value, count)

    return result


def compute_data_hash(data: list[dict]) -> str:
    """Compute deterministic SHA-256 hash of structured data."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_and_filter_curated_links(
    path: Path | None = None,
) -> tuple[list[TieredLink], str]:
    """Load curated links, filter by quality, return with data hash.

    Returns:
        Tuple of (filtered links with tiers, SHA-256 hash of raw data).
    """
    p = path or CURATED_PATH
    raw_links: list[dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_links.append(json.loads(line))

    raw_hash = compute_data_hash(raw_links)
    logger.info("Loaded %d curated links (hash=%s)", len(raw_links), raw_hash[:16])

    filtered = filter_training_links(raw_links)
    logger.info(
        "After quality filter: %d usable links (dropped %d)",
        len(filtered),
        len(raw_links) - len(filtered),
    )

    return filtered, raw_hash


def save_training_links(
    links: list[TieredLink],
    raw_hash: str,
    path: Path | None = None,
) -> str:
    """Save filtered training links to JSONL with tier metadata.

    Returns SHA-256 hash of the output data.
    """
    p = path or TRAINING_OUTPUT_PATH
    output_records: list[dict] = []
    for tiered in links:
        record = dict(tiered.link)
        record["quality_tier"] = tiered.tier.value
        output_records.append(record)

    output_hash = compute_data_hash(output_records)

    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info(
        "Saved %d training links to %s (hash=%s, raw_hash=%s)",
        len(output_records),
        p.name,
        output_hash[:16],
        raw_hash[:16],
    )
    return output_hash
