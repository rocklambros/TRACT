"""OpenCRE export filter pipeline (spec §5).

Filters applied in SQL:
1. Ground truth exclusion (provenance != 'ground_truth_T1-AI')
2. NULL confidence exclusion
3. OOD exclusion (is_ood != 1)
4. Only accepted review_status
Per-framework confidence floor applied in Python.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict, cast

from tract.config import PHASE5_GROUND_TRUTH_PROVENANCE
from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


class ExportableAssignment(TypedDict):
    """One assignment row that passed every export filter.

    Mirrors the SELECT column list in query_exportable_assignments. Declared
    so downstream consumers (the export manifest, coverage gaps) index it
    against real key names instead of dict[str, object], which erased the
    value types and made every read unindexable.
    """

    control_id: str
    hub_id: str
    hub_name: str
    confidence: float
    is_ood: int
    provenance: str
    framework_id: str
    section_id: str
    title: str
    description: str


def query_exportable_assignments(
    db_path: Path,
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    framework_filter: str | None = None,
) -> list[ExportableAssignment]:
    """Query assignments passing all export filters.

    Returns list of dicts with keys: control_id, hub_id, hub_name,
    confidence, framework_id, section_id, title, description.
    Sorted by (hub_id, framework_id, section_id).
    """
    conn = get_connection(db_path)
    try:
        query = (
            "SELECT a.control_id, a.hub_id, h.name AS hub_name, "
            "a.confidence, a.is_ood, a.provenance, "
            "c.framework_id, c.section_id, c.title, c.description "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN hubs h ON a.hub_id = h.id "
            "WHERE a.review_status = 'accepted' "
            "AND a.provenance != ? "
            "AND a.confidence IS NOT NULL "
            "AND a.is_ood != 1 "
        )
        params: list[str] = [PHASE5_GROUND_TRUTH_PROVENANCE]

        if framework_filter:
            query += "AND c.framework_id = ? "
            params.append(framework_filter)

        query += "ORDER BY a.hub_id, c.framework_id, c.section_id"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    results: list[ExportableAssignment] = []
    for row in rows:
        fw_id = row["framework_id"]
        floor = confidence_overrides.get(fw_id, confidence_floor)
        if row["confidence"] < floor:
            logger.debug(
                "Excluded %s: confidence %.3f < floor %.3f (framework=%s)",
                row["control_id"], row["confidence"], floor, fw_id,
            )
            continue
        results.append(cast(ExportableAssignment, dict(row)))

    logger.info(
        "Export filter: %d assignments passed (%d excluded by confidence floor)",
        len(results), len(rows) - len(results),
    )
    return results


def compute_filter_stats(
    db_path: Path,
    exported_rows: list[ExportableAssignment],
    confidence_floor: float,
    confidence_overrides: dict[str, float],
) -> dict[str, dict[str, int]]:
    """Compute per-framework filter statistics for the export manifest."""
    conn = get_connection(db_path)
    try:
        all_rows = conn.execute(
            "SELECT a.control_id, a.hub_id, a.confidence, a.is_ood, "
            "a.provenance, a.review_status, c.framework_id "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id"
        ).fetchall()
    finally:
        conn.close()

    exported_keys = {(r["control_id"], r["hub_id"]) for r in exported_rows}

    stats: dict[str, dict[str, int]] = {}
    for row in all_rows:
        fw_id = row["framework_id"]
        if fw_id not in stats:
            stats[fw_id] = {
                "exported": 0,
                "excluded_ground_truth": 0,
                "excluded_confidence": 0,
                "excluded_ood": 0,
                "excluded_null_confidence": 0,
                "excluded_not_accepted": 0,
            }
        s = stats[fw_id]
        key = (row["control_id"], row["hub_id"])

        if row["provenance"] == PHASE5_GROUND_TRUTH_PROVENANCE:
            s["excluded_ground_truth"] += 1
        elif row["review_status"] != "accepted":
            s["excluded_not_accepted"] += 1
        elif row["confidence"] is None:
            s["excluded_null_confidence"] += 1
        elif row["is_ood"] == 1:
            s["excluded_ood"] += 1
        elif key in exported_keys:
            s["exported"] += 1
        else:
            floor = confidence_overrides.get(fw_id, confidence_floor)
            if row["confidence"] < floor:
                s["excluded_confidence"] += 1

    return stats
