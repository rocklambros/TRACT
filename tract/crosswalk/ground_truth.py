"""Ground truth import — multi-strategy section ID resolver and assignment creation."""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ResolverResult:
    resolved: dict[str, str]           # gt_key → control_id
    unresolved: list[dict]             # list of unresolvable GT links
    strategy_counts: dict[str, int]    # strategy_name → count


def build_control_lookups(
    conn: sqlite3.Connection,
    framework_id: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build three lookup dictionaries for a framework's controls.

    Returns (section_id_map, title_map, normalized_map) where each maps
    the lookup key → control_id.
    """
    rows = conn.execute(
        "SELECT id, section_id, title FROM controls WHERE framework_id = ?",
        (framework_id,),
    ).fetchall()

    section_id_map: dict[str, str] = {}
    title_map: dict[str, str] = {}
    normalized_map: dict[str, str] = {}

    for row in rows:
        ctrl_id = row["id"]
        sid = row["section_id"]
        title = row["title"] or ""

        section_id_map[sid] = ctrl_id
        if title:
            title_map[title] = ctrl_id
            title_map[title.lower()] = ctrl_id
        normalized_key = re.sub(r"[^a-z0-9]", "", sid.lower())
        if normalized_key:
            normalized_map[normalized_key] = ctrl_id

    return section_id_map, title_map, normalized_map


def resolve_section_id(
    gt_section_id: str,
    framework_id: str,
    section_id_map: dict[str, str],
    title_map: dict[str, str],
    normalized_map: dict[str, str],
) -> tuple[str | None, str]:
    """Try 5 strategies to resolve a GT section_id to a control_id.

    Returns (control_id, strategy_name) or (None, "unresolved").
    """
    # Strategy 1: Direct match
    if gt_section_id in section_id_map:
        return section_id_map[gt_section_id], "direct"

    # Strategy 2: Prefixed match
    prefixed = f"{framework_id}:{gt_section_id}"
    if prefixed in section_id_map:
        return section_id_map[prefixed], "prefixed"

    # Strategy 3: Title exact match
    if gt_section_id in title_map:
        return title_map[gt_section_id], "title_exact"

    # Strategy 4: Title case-insensitive
    if gt_section_id.lower() in title_map:
        return title_map[gt_section_id.lower()], "title_case_insensitive"

    # Strategy 5: Normalized
    gt_normalized = re.sub(r"[^a-z0-9]", "", gt_section_id.lower())
    if gt_normalized and gt_normalized in normalized_map:
        return normalized_map[gt_normalized], "normalized"

    return None, "unresolved"


def resolve_framework_links(
    conn: sqlite3.Connection,
    framework_id: str,
    links: list[dict],
) -> ResolverResult:
    """Resolve all GT links for a single framework.

    Returns ResolverResult with resolved mappings, unresolved links, and strategy counts.
    """
    section_id_map, title_map, normalized_map = build_control_lookups(conn, framework_id)

    resolved: dict[str, str] = {}
    unresolved: list[dict] = []
    strategy_counts: dict[str, int] = {}

    for link in links:
        gt_sid = link["section_id"]
        gt_key = f"{framework_id}:{gt_sid}"

        control_id, strategy = resolve_section_id(
            gt_sid, framework_id, section_id_map, title_map, normalized_map,
        )

        if control_id is not None:
            resolved[gt_key] = control_id
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        else:
            unresolved.append(link)
            logger.warning(
                "Unresolvable GT link: framework=%s section_id=%s",
                framework_id, gt_sid,
            )

    return ResolverResult(
        resolved=resolved,
        unresolved=unresolved,
        strategy_counts=strategy_counts,
    )
