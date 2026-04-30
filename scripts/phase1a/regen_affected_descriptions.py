"""Selective hub description regeneration for audit-affected hubs.

Phase 0R audit corrected 56 AI-framework links. Hub descriptions generated
before the audit used uncurated links as prompt context. This script:

1. Identifies hubs where >50% of linked-section prompt context came from
   AI framework links AND those links changed in the audit
2. Regenerates descriptions for those hubs using curated links
3. Preserves expert review status for unaffected hubs
4. Marks regenerated hubs as review_status="pending"

Usage:
    python -m scripts.phase1a.regen_affected_descriptions
    python -m scripts.phase1a.regen_affected_descriptions --dry-run
    python -m scripts.phase1a.regen_affected_descriptions --threshold 0.3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from tract.config import (
    PHASE1A_DESCRIPTION_MAX_CONCURRENT,
    PHASE1A_DESCRIPTION_MAX_TOKENS,
    PHASE1A_DESCRIPTION_MODEL,
    PHASE1A_DESCRIPTION_TEMPERATURE,
    PHASE1A_DESCRIPTION_TIMEOUT_S,
    PROCESSED_DIR,
    TRAINING_DIR,
)
from tract.descriptions import (
    DESCRIPTION_SYSTEM_PROMPT,
    HubDescription,
    HubDescriptionSet,
    build_description_prompt,
)
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.sanitize import sanitize_text

try:
    from anthropic.types import TextBlock
except ImportError:
    TextBlock = None  # type: ignore[assignment,misc]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AI_FRAMEWORKS: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS",
    "NIST AI 100-2",
    "OWASP AI Exchange",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
})

UNCURATED_PATH: Final[Path] = TRAINING_DIR / "hub_links.jsonl"
CURATED_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"
REVIEWED_PATH: Final[Path] = PROCESSED_DIR / "hub_descriptions_reviewed.json"
OUTPUT_PATH: Final[Path] = PROCESSED_DIR / "hub_descriptions_reviewed.json"
BACKUP_PATH: Final[Path] = PROCESSED_DIR / "hub_descriptions_reviewed.pre_regen.json"


def _load_links(path: Path) -> list[dict]:
    links = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            links.append(json.loads(line))
    return links


def _hub_section_map(links: list[dict]) -> dict[str, set[tuple[str, str]]]:
    """Build hub_id -> set of (standard_name, section_name) for AI frameworks."""
    result: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for link in links:
        if link["standard_name"] in AI_FRAMEWORKS:
            result[link["cre_id"]].add(
                (link["standard_name"], link.get("section_name", ""))
            )
    return dict(result)


def _hub_all_section_names(links: list[dict]) -> dict[str, set[str]]:
    """Build hub_id -> set of all section names (all frameworks)."""
    result: dict[str, set[str]] = defaultdict(set)
    for link in links:
        section = link.get("section_name") or link.get("section_id", "")
        if section:
            result[link["cre_id"]].add(section)
    return dict(result)


def identify_affected_hubs(threshold: float = 0.5) -> list[str]:
    """Identify hubs needing description regeneration.

    A hub is affected if:
    1. AI framework links make up > threshold of its total linked sections
    2. Those AI links changed between uncurated and curated versions
    """
    uncurated = _load_links(UNCURATED_PATH)
    curated = _load_links(CURATED_PATH)

    uncurated_ai = _hub_section_map(uncurated)
    curated_ai = _hub_section_map(curated)
    all_sections = _hub_all_section_names(uncurated)

    affected: list[str] = []

    all_hub_ids = set(uncurated_ai.keys()) | set(curated_ai.keys())
    for hub_id in sorted(all_hub_ids):
        total_sections = len(all_sections.get(hub_id, set()))
        ai_sections = len(uncurated_ai.get(hub_id, set()))

        if total_sections == 0:
            continue

        ai_fraction = ai_sections / total_sections
        if ai_fraction <= threshold:
            continue

        old = uncurated_ai.get(hub_id, set())
        new = curated_ai.get(hub_id, set())
        if old != new:
            affected.append(hub_id)

    return affected


def _build_curated_hub_sections() -> dict[str, list[str]]:
    """Build hub_id -> sorted list of section names from CURATED links."""
    curated = _load_links(CURATED_PATH)
    hub_sections: dict[str, list[str]] = defaultdict(list)
    for link in curated:
        section = link.get("section_name") or link.get("section_id", "")
        if section:
            hub_sections[link["cre_id"]].append(section)
    for hub_id in hub_sections:
        hub_sections[hub_id] = sorted(set(hub_sections[hub_id]))
    return dict(hub_sections)


def _get_api_key() -> str:
    import os
    import subprocess

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    result = subprocess.run(
        ["pass", "anthropic/api-key"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    api_key = result.stdout.strip()
    if not api_key:
        raise RuntimeError("pass returned empty API key")
    return api_key


async def regenerate_descriptions(
    hub_ids: list[str],
    hierarchy: CREHierarchy,
    hub_sections: dict[str, list[str]],
) -> dict[str, HubDescription]:
    """Regenerate descriptions for specified hubs using curated link context."""
    import anthropic

    api_key = _get_api_key()
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        max_retries=3,
        timeout=PHASE1A_DESCRIPTION_TIMEOUT_S + 30,
    )
    semaphore = asyncio.Semaphore(PHASE1A_DESCRIPTION_MAX_CONCURRENT)
    results: dict[str, HubDescription] = {}

    async def generate_one(hub_id: str) -> tuple[str, HubDescription | Exception]:
        node = hierarchy.hubs[hub_id]
        siblings = hierarchy.get_siblings(hub_id)
        sibling_names = [s.name for s in siblings]
        linked_sections = hub_sections.get(hub_id, [])

        prompt = build_description_prompt(
            hub_name=node.name,
            hierarchy_path=node.hierarchy_path,
            sibling_names=sibling_names,
            linked_section_names=linked_sections,
        )

        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=PHASE1A_DESCRIPTION_MODEL,
                        max_tokens=PHASE1A_DESCRIPTION_MAX_TOKENS,
                        temperature=PHASE1A_DESCRIPTION_TEMPERATURE,
                        system=DESCRIPTION_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=PHASE1A_DESCRIPTION_TIMEOUT_S,
                )
                content_block = response.content[0]
                if TextBlock is not None and not isinstance(content_block, TextBlock):
                    raise TypeError(f"Expected TextBlock, got {type(content_block).__name__}")
                raw_text = content_block.text.strip()
                clean_text = sanitize_text(raw_text)

                desc = HubDescription(
                    hub_id=hub_id,
                    hub_name=node.name,
                    hierarchy_path=node.hierarchy_path,
                    description=clean_text,
                    model=PHASE1A_DESCRIPTION_MODEL,
                    temperature=PHASE1A_DESCRIPTION_TEMPERATURE,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    review_status="pending",
                    reviewed_description=None,
                    reviewer_notes=None,
                )
                return hub_id, desc
            except Exception as exc:
                logger.error("Failed to generate for %s (%s): %s", hub_id, node.name, exc)
                return hub_id, exc

    try:
        tasks = [generate_one(hid) for hid in hub_ids]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        failures: list[str] = []
        for i, item in enumerate(raw_results):
            if isinstance(item, BaseException):
                failures.append(hub_ids[i])
                logger.error("Unhandled exception for %s: %s", hub_ids[i], item)
                continue
            hub_id, result = item
            if isinstance(result, Exception):
                failures.append(hub_id)
                continue
            results[hub_id] = result

        if failures:
            logger.error("Failed hubs (%d): %s", len(failures), failures)
            if len(failures) == len(hub_ids):
                raise RuntimeError("All regeneration attempts failed")
    finally:
        await client.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate descriptions for audit-affected hubs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Identify affected hubs without regenerating",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="AI-link fraction threshold (default: 0.5 = hubs where >50%% of context was AI)",
    )
    args = parser.parse_args()

    affected = identify_affected_hubs(threshold=args.threshold)
    logger.info("Identified %d affected hubs (threshold=%.0f%%)", len(affected), args.threshold * 100)

    if not affected:
        logger.info("No hubs affected — nothing to regenerate.")
        return

    hierarchy_data = load_json(PROCESSED_DIR / "cre_hierarchy.json")
    hierarchy = CREHierarchy.model_validate(hierarchy_data)

    reviewed_data = load_json(REVIEWED_PATH)
    desc_set = HubDescriptionSet.model_validate(reviewed_data)

    for hub_id in affected:
        old_desc = desc_set.descriptions.get(hub_id)
        old_status = old_desc.review_status if old_desc else "missing"
        old_sections_preview = ""
        if old_desc:
            old_sections_preview = old_desc.description[:80] + "..."
        logger.info(
            "  %s (%s) — was: %s — %s",
            hub_id,
            hierarchy.hubs[hub_id].name,
            old_status,
            old_sections_preview,
        )

    if args.dry_run:
        logger.info("Dry run — no changes made. %d hubs would be regenerated.", len(affected))
        return

    logger.info("Backing up reviewed descriptions to %s", BACKUP_PATH)
    atomic_write_json(reviewed_data, BACKUP_PATH)

    curated_sections = _build_curated_hub_sections()

    logger.info("Regenerating %d descriptions using curated links...", len(affected))
    start = time.time()
    new_descs = asyncio.run(regenerate_descriptions(affected, hierarchy, curated_sections))
    elapsed = time.time() - start
    logger.info("Regenerated %d/%d descriptions in %.1fs", len(new_descs), len(affected), elapsed)

    preserved = 0
    replaced = 0
    for hub_id in affected:
        if hub_id in new_descs:
            old = desc_set.descriptions.get(hub_id)
            old_text = old.description if old else "(none)"
            new_text = new_descs[hub_id].description
            if old_text != new_text:
                logger.info(
                    "  CHANGED %s: '%s' → '%s'",
                    hub_id, old_text[:60], new_text[:60],
                )
            else:
                logger.info("  UNCHANGED %s (same text despite different context)", hub_id)
            desc_set.descriptions[hub_id] = new_descs[hub_id]
            replaced += 1
        else:
            logger.warning("  KEPT %s (regeneration failed, preserving old)", hub_id)
            preserved += 1

    desc_set.total_pending_review = sum(
        1 for d in desc_set.descriptions.values() if d.review_status == "pending"
    )
    desc_set.generation_timestamp = datetime.now(timezone.utc).isoformat()

    atomic_write_json(desc_set.model_dump(), OUTPUT_PATH)
    logger.info(
        "Saved updated descriptions: %d replaced (now pending review), "
        "%d preserved, %d total pending review",
        replaced, preserved, desc_set.total_pending_review,
    )


if __name__ == "__main__":
    main()
