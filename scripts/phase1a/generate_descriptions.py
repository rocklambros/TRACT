"""Generate hub descriptions for all 400 leaf hubs via Opus.

Usage:
    python -m scripts.phase1a.generate_descriptions
    python -m scripts.phase1a.generate_descriptions --limit 10  # generate first 10 only
    python -m scripts.phase1a.generate_descriptions --dry-run   # show what would be generated
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

from tract.config import (
    PHASE1A_DESCRIPTION_MAX_CONCURRENT,
    PHASE1A_DESCRIPTION_MAX_TOKENS,
    PHASE1A_DESCRIPTION_MODEL,
    PHASE1A_DESCRIPTION_SAVE_INTERVAL,
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
except ImportError:  # pragma: no cover
    TextBlock = None  # type: ignore[assignment,misc]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PARTIAL_PATH = PROCESSED_DIR / "hub_descriptions_partial.json"
OUTPUT_PATH = PROCESSED_DIR / "hub_descriptions.json"


def _get_api_key() -> str:
    """Retrieve Anthropic API key from env or pass manager."""
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


def _load_hub_links() -> dict[str, list[str]]:
    """Load hub links and return hub_id -> list of section names."""
    links_path = TRAINING_DIR / "hub_links.jsonl"
    if not links_path.exists():
        raise FileNotFoundError(
            f"Hub links not found at {links_path}. "
            "Run parsers/extract_hub_links.py first."
        )

    import json
    hub_sections: dict[str, list[str]] = defaultdict(list)
    with open(links_path, encoding="utf-8") as f:
        for line in f:
            link = json.loads(line)
            section = link.get("section_name") or link.get("section_id", "")
            if section:
                hub_sections[link["cre_id"]].append(section)

    for hub_id in hub_sections:
        hub_sections[hub_id] = sorted(set(hub_sections[hub_id]))

    return dict(hub_sections)


def _load_existing_descriptions() -> dict[str, HubDescription]:
    """Load any previously generated descriptions for resume support."""
    if PARTIAL_PATH.exists():
        logger.info("Found partial file, loading for resume: %s", PARTIAL_PATH)
        data = load_json(PARTIAL_PATH)
        desc_set = HubDescriptionSet.model_validate(data)
        valid = {
            hub_id: desc
            for hub_id, desc in desc_set.descriptions.items()
            if desc.description
        }
        logger.info("Resuming with %d existing descriptions", len(valid))
        return valid
    return {}


async def _generate_all(
    hierarchy: CREHierarchy,
    hub_sections: dict[str, list[str]],
    existing: dict[str, HubDescription],
    limit: int | None,
) -> dict[str, HubDescription]:
    """Generate descriptions for all leaf hubs not already in existing."""
    import anthropic

    api_key = _get_api_key()
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        max_retries=3,
        timeout=PHASE1A_DESCRIPTION_TIMEOUT_S + 30,
    )
    semaphore = asyncio.Semaphore(PHASE1A_DESCRIPTION_MAX_CONCURRENT)
    descriptions = dict(existing)

    hub_ids_to_generate = [
        hid for hid in hierarchy.label_space
        if hid not in existing
    ]
    if limit is not None:
        hub_ids_to_generate = hub_ids_to_generate[:limit]

    logger.info(
        "Generating descriptions: %d to generate, %d already done, %d total leaf hubs",
        len(hub_ids_to_generate), len(existing), len(hierarchy.label_space),
    )

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
                    raise TypeError(
                        f"Expected TextBlock, got {type(content_block).__name__}"
                    )
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
        generated_count = 0
        tasks = [generate_one(hid) for hid in hub_ids_to_generate]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        failures: list[str] = []
        for hub_id, result in results:
            if isinstance(result, Exception):
                failures.append(hub_id)
                continue
            descriptions[hub_id] = result
            generated_count += 1
            if generated_count % PHASE1A_DESCRIPTION_SAVE_INTERVAL == 0:
                _save_partial(descriptions, hierarchy)
                logger.info("Intermediate save at %d descriptions", generated_count)

        if failures:
            fail_rate = len(failures) / len(hub_ids_to_generate)
            logger.error(
                "%d/%d generations failed (%.1f%%): %s",
                len(failures), len(hub_ids_to_generate),
                fail_rate * 100, failures[:10],
            )
            if fail_rate > 0.10:
                raise RuntimeError(
                    f"Failure rate {fail_rate:.1%} exceeds 10% threshold"
                )
    finally:
        await client.close()

    return descriptions


def _save_partial(
    descriptions: dict[str, HubDescription],
    hierarchy: CREHierarchy,
) -> None:
    """Save intermediate results."""
    desc_set = HubDescriptionSet(
        descriptions=descriptions,
        generation_model=PHASE1A_DESCRIPTION_MODEL,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        data_hash=hierarchy.data_hash,
        total_generated=len(descriptions),
        total_pending_review=sum(
            1 for d in descriptions.values() if d.review_status == "pending"
        ),
    )
    atomic_write_json(desc_set.model_dump(), PARTIAL_PATH)


def _save_final(
    descriptions: dict[str, HubDescription],
    hierarchy: CREHierarchy,
) -> None:
    """Save final results and validate cross-references."""
    leaf_ids = set(hierarchy.label_space)
    desc_ids = set(descriptions.keys())
    missing = leaf_ids - desc_ids
    extra = desc_ids - leaf_ids

    if missing:
        logger.warning("Missing descriptions for %d leaf hubs: %s", len(missing), sorted(missing)[:10])
    if extra:
        raise ValueError(f"Descriptions exist for non-leaf hubs: {sorted(extra)[:10]}")

    desc_set = HubDescriptionSet(
        descriptions=descriptions,
        generation_model=PHASE1A_DESCRIPTION_MODEL,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        data_hash=hierarchy.data_hash,
        total_generated=len(descriptions),
        total_pending_review=sum(
            1 for d in descriptions.values() if d.review_status == "pending"
        ),
    )
    atomic_write_json(desc_set.model_dump(), OUTPUT_PATH)
    logger.info("Saved %d descriptions to %s", len(descriptions), OUTPUT_PATH)

    # Clean up partial file
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        logger.info("Removed partial file %s", PARTIAL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hub descriptions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of hubs to generate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without calling API")
    args = parser.parse_args()

    hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
    if not hierarchy_path.exists():
        raise FileNotFoundError(
            f"Hierarchy not found at {hierarchy_path}. "
            "Run scripts/phase1a/build_hierarchy.py first."
        )

    hierarchy = CREHierarchy.load(hierarchy_path)
    hub_sections = _load_hub_links()
    existing = _load_existing_descriptions()

    to_generate = [hid for hid in hierarchy.label_space if hid not in existing]
    if args.limit:
        to_generate = to_generate[:args.limit]

    if args.dry_run:
        logger.info("DRY RUN: would generate %d descriptions", len(to_generate))
        for hid in to_generate[:5]:
            node = hierarchy.hubs[hid]
            logger.info("  %s: %s (%s)", hid, node.name, node.hierarchy_path)
        if len(to_generate) > 5:
            logger.info("  ... and %d more", len(to_generate) - 5)
        return

    t0 = time.monotonic()
    descriptions = asyncio.run(
        _generate_all(hierarchy, hub_sections, existing, args.limit)
    )
    elapsed = time.monotonic() - t0

    _save_final(descriptions, hierarchy)
    logger.info("Done in %.1fs (%.1fs per hub)", elapsed, elapsed / max(len(to_generate), 1))


if __name__ == "__main__":
    main()
