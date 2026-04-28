"""Extract traditional framework controls from OpenCRE link metadata.

Extracts the 19 OpenCRE frameworks that lack primary-source parsers
and writes per-framework JSON + unified all_controls.json.

Usage: python -m scripts.phase1a.extract_traditional_frameworks
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tract.config import (
    AI_PARSER_FRAMEWORK_IDS,
    COUNT_TOLERANCE,
    OPENCRE_EXTRACT_FRAMEWORK_IDS,
    OPENCRE_FRAMEWORK_ID_MAP,
    PHASE1A_FRAMEWORK_SLUG_RE,
    PROCESSED_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    RAW_OPENCRE_DIR,
)
from tract.io import atomic_write_json, load_json
from tract.sanitize import sanitize_text
from tract.schema import Control, FrameworkOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(PHASE1A_FRAMEWORK_SLUG_RE)
_BARE_ID_FRAMEWORKS = {"CAPEC", "CWE"}


def validate_framework_slug(slug: str) -> None:
    """Validate a framework_id slug. Raises ValueError if invalid."""
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"Invalid framework slug: {slug!r} "
            f"(must match {PHASE1A_FRAMEWORK_SLUG_RE})"
        )


def slugify(name: str) -> str:
    """Generate a URL-safe slug from a section name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80]
    if not slug:
        raise ValueError(f"Slugified name is empty. Original: {name!r}")
    return slug


def _normalize_link_type(raw: str) -> str | None:
    """Normalize link type string. Returns None if not a training link."""
    lower = raw.replace(" ", "").lower()
    if lower == "linkedto":
        return "LinkedTo"
    if lower == "automaticallylinkedto":
        return "AutomaticallyLinkedTo"
    return None


def extract_framework_controls(
    cres: list[dict[str, Any]],
    framework_names: set[str],
    framework_id: str,
) -> list[Control]:
    """Extract unique controls for a framework from OpenCRE CRE records."""
    seen: set[tuple[str, str, str]] = set()
    controls: list[Control] = []
    bare_id_count = 0

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        for link in cre.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") != "Standard":
                continue

            standard_name = doc.get("name", "")
            if standard_name not in framework_names:
                continue

            raw_ltype = link.get("ltype", link.get("type", ""))
            link_type = _normalize_link_type(raw_ltype)
            if link_type is None:
                continue

            section_id = str(doc.get("sectionID", "")).strip()
            section_name = str(doc.get("section", "")).strip()

            dedup_key = (framework_id, section_id, section_name)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Build control_id
            if section_id:
                control_id_suffix = section_id
            else:
                control_id_suffix = slugify(section_name)

            control_id = f"{framework_id}:{control_id_suffix}"

            # Build title
            if section_name:
                title = section_name
            elif section_id and standard_name in _BARE_ID_FRAMEWORKS:
                title = f"{standard_name}-{section_id}"
                bare_id_count += 1
            else:
                title = section_id or "(unknown)"

            try:
                description = sanitize_text(title)
            except ValueError:
                description = title if title else "(no description)"

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=description,
                full_text=None,
                hierarchy_level=None,
                parent_id=None,
                parent_name=None,
                metadata={
                    "opencre_standard_name": standard_name,
                    "opencre_section_id": section_id,
                    "link_type": link_type,
                },
            ))

    if bare_id_count > 0:
        logger.info(
            "%s: %d bare-ID controls (no section name)",
            framework_id, bare_id_count,
        )

    controls.sort(key=lambda c: c.control_id)
    return controls


def _count_links_per_framework(
    cres: list[dict[str, Any]],
) -> dict[str, int]:
    """Count raw training links per framework_id."""
    counts: dict[str, int] = defaultdict(int)
    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        for link in cre.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") != "Standard":
                continue
            raw_ltype = link.get("ltype", link.get("type", ""))
            if _normalize_link_type(raw_ltype) is None:
                continue
            standard_name = doc.get("name", "")
            fw_id = OPENCRE_FRAMEWORK_ID_MAP.get(standard_name)
            if fw_id:
                counts[fw_id] += 1
    return dict(counts)


def _build_id_to_names_map() -> dict[str, set[str]]:
    """Build map: framework_id -> set of OpenCRE standard names."""
    id_to_names: dict[str, set[str]] = defaultdict(set)
    for name, fwid in OPENCRE_FRAMEWORK_ID_MAP.items():
        id_to_names[fwid].add(name)
    return dict(id_to_names)


def main() -> None:
    opencre_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    if not opencre_path.exists():
        raise FileNotFoundError(f"OpenCRE data not found at {opencre_path}")

    data = load_json(opencre_path)
    cres = data["cres"]
    fetch_timestamp = data.get("fetch_timestamp", "unknown")

    # Parse date from fetch_timestamp for version string
    try:
        fetch_date = datetime.fromisoformat(fetch_timestamp).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        fetch_date = "unknown"

    link_counts = _count_links_per_framework(cres)
    id_to_names = _build_id_to_names_map()

    PROCESSED_FRAMEWORKS_DIR.mkdir(parents=True, exist_ok=True)

    extracted_frameworks: list[FrameworkOutput] = []

    for framework_id in sorted(OPENCRE_EXTRACT_FRAMEWORK_IDS):
        validate_framework_slug(framework_id)

        names = id_to_names.get(framework_id)
        if not names:
            logger.warning("No OpenCRE names mapped to framework_id=%s, skipping", framework_id)
            continue

        controls = extract_framework_controls(cres, names, framework_id)
        if not controls:
            logger.warning("No controls extracted for %s, skipping", framework_id)
            continue

        # Pick the shortest name as framework_name
        framework_name = min(names, key=len)

        fw_output = FrameworkOutput(
            framework_id=framework_id,
            framework_name=framework_name,
            version=f"opencre-{fetch_date}",
            source_url="https://opencre.org",
            fetched_date=fetch_timestamp,
            mapping_unit_level="section",
            controls=controls,
        )

        output_path = PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
        atomic_write_json(fw_output.model_dump(), output_path)

        raw_links = link_counts.get(framework_id, 0)
        deviation = abs(len(controls) - raw_links) / max(raw_links, 1)
        if deviation > COUNT_TOLERANCE:
            logger.warning(
                "%s: %d controls vs %d raw links (%.1f%% deviation)",
                framework_id, len(controls), raw_links, deviation * 100,
            )

        logger.info(
            "Extracted %s: %d controls (%d raw links)",
            framework_id, len(controls), raw_links,
        )
        extracted_frameworks.append(fw_output)

    # Build all_controls.json: AI parsers take precedence
    all_frameworks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # First: load existing AI parser outputs
    for fw_path in sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json")):
        fw_id = fw_path.stem
        if fw_id in AI_PARSER_FRAMEWORK_IDS:
            fw_data = load_json(fw_path)
            all_frameworks.append(fw_data)
            seen_ids.add(fw_id)

    # Then: add extracted traditional frameworks
    for fw in extracted_frameworks:
        if fw.framework_id not in seen_ids:
            all_frameworks.append(fw.model_dump())
            seen_ids.add(fw.framework_id)

    all_frameworks.sort(key=lambda f: f["framework_id"])
    total_controls = sum(len(f["controls"]) for f in all_frameworks)

    all_controls = {
        "framework_count": len(all_frameworks),
        "total_controls": total_controls,
        "generated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "frameworks": all_frameworks,
    }
    atomic_write_json(all_controls, PROCESSED_DIR / "all_controls.json")

    logger.info(
        "all_controls.json: %d frameworks, %d total controls",
        len(all_frameworks), total_controls,
    )


if __name__ == "__main__":
    main()
