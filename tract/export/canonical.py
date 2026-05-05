"""Canonical export: snapshot builder, differ, and serializer (spec §§2-8).

Produces per-framework JSON snapshots and changesets for OpenCRE's
incremental import RFC. The export_history table tracks prior exports
for changeset generation.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tract.crosswalk.schema import get_connection
from tract.export.canonical_schema import (
    CanonicalControl,
    Changeset,
    ChangesetEntry,
    ChangesetSummary,
    CREMapping,
    FilterPolicy,
    ImpactAnalysis,
    StandardSnapshot,
    compute_content_hash,
)

logger = logging.getLogger(__name__)

EXPORT_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS export_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    framework_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    export_date TEXT NOT NULL DEFAULT (datetime('now')),
    snapshot_json TEXT NOT NULL,
    filter_policy_json TEXT NOT NULL,
    assignment_count INTEGER NOT NULL,
    control_count INTEGER NOT NULL,
    tract_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_export_history_fw
    ON export_history(framework_id, export_date);
"""


def ensure_export_history_table(db_path: Path) -> None:
    """Create the export_history table if it does not exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(EXPORT_HISTORY_DDL)
        conn.commit()
    finally:
        conn.close()


def _query_canonical_assignments(
    db_path: Path,
    framework_id: str,
    confidence_floor: float,
    confidence_overrides: dict[str, float],
) -> list[dict]:
    """Query assignments passing all export filters, returning fields needed for canonical export."""
    from tract.config import PHASE5_GROUND_TRUTH_PROVENANCE

    conn = get_connection(db_path)
    try:
        floor = confidence_overrides.get(framework_id, confidence_floor)
        rows = conn.execute(
            "SELECT a.control_id, a.hub_id, h.name AS hub_name, "
            "a.confidence, a.provenance, "
            "c.framework_id, c.section_id, c.title, c.description "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN hubs h ON a.hub_id = h.id "
            "WHERE a.review_status = 'accepted' "
            "AND a.provenance != ? "
            "AND a.confidence IS NOT NULL "
            "AND a.is_ood != 1 "
            "AND c.framework_id = ? "
            "AND a.confidence >= ? "
            "ORDER BY a.hub_id, c.section_id",
            [PHASE5_GROUND_TRUTH_PROVENANCE, framework_id, floor],
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def build_snapshot(
    db_path: Path,
    framework_id: str,
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    model_adapter_hash: str,
    tract_version: str,
    hyperlink_fn: Callable[[str, str], str],
    framework_name: str | None = None,
) -> StandardSnapshot:
    """Build a StandardSnapshot from live DB data."""
    if framework_name is None:
        from tract.export.opencre_names import get_opencre_name
        framework_name = get_opencre_name(framework_id)

    rows = _query_canonical_assignments(
        db_path, framework_id, confidence_floor, confidence_overrides,
    )

    seen_controls: dict[str, CanonicalControl] = {}
    control_mappings: defaultdict[str, list[dict]] = defaultdict(list)

    for row in rows:
        cid = row["control_id"]
        if cid not in seen_controls:
            seen_controls[cid] = CanonicalControl(
                control_id=cid,
                framework_id=row["framework_id"],
                section_id=row["section_id"],
                title=row["title"],
                description=row["description"],
                hyperlink=hyperlink_fn(row["framework_id"], row["section_id"]),
            )
        control_mappings[cid].append(row)

    controls = sorted(seen_controls.values(), key=lambda c: c.control_id)

    mappings: list[CREMapping] = []
    for cid in sorted(control_mappings.keys()):
        ranked = sorted(control_mappings[cid], key=lambda r: -r["confidence"])
        for rank_idx, row in enumerate(ranked, start=1):
            mappings.append(CREMapping(
                control_id=row["control_id"],
                hub_id=row["hub_id"],
                hub_name=row["hub_name"],
                confidence=row["confidence"],
                rank=rank_idx,
                provenance=row["provenance"],
                model_version=model_adapter_hash,
            ))

    filter_policy = FilterPolicy(
        confidence_floor=confidence_floor,
        confidence_override=confidence_overrides.get(framework_id),
    )

    snapshot = StandardSnapshot(
        framework_id=framework_id,
        framework_name=framework_name,
        export_date=datetime.now(timezone.utc).isoformat(),
        content_hash="placeholder",
        tract_version=tract_version,
        model_adapter_hash=model_adapter_hash,
        filter_policy=filter_policy,
        controls=controls,
        mappings=mappings,
    )
    snapshot.content_hash = compute_content_hash(snapshot)
    return snapshot
