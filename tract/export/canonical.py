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


# ── Control diff helpers ────────────────────────────────────────────────

_CONTROL_MUTABLE_FIELDS = ("title", "description", "hyperlink")
_MAPPING_MUTABLE_FIELDS = ("confidence", "rank", "provenance", "model_version")


def _diff_controls(
    prior_controls: dict[str, CanonicalControl],
    current_controls: dict[str, CanonicalControl],
) -> list[ChangesetEntry]:
    ops: list[ChangesetEntry] = []
    all_keys = sorted(set(prior_controls) | set(current_controls))
    for key in all_keys:
        old = prior_controls.get(key)
        new = current_controls.get(key)
        if old is None and new is not None:
            ops.append(ChangesetEntry(operation="ADD_CONTROL", entity=new))
        elif old is not None and new is None:
            ops.append(ChangesetEntry(operation="DELETE_CONTROL", key=key))
        elif old is not None and new is not None:
            if any(getattr(old, f) != getattr(new, f) for f in _CONTROL_MUTABLE_FIELDS):
                ops.append(ChangesetEntry(
                    operation="UPDATE_CONTROL", entity=new, before=old,
                ))
    return ops


def _diff_mappings(
    prior_mappings: dict[tuple[str, str], CREMapping],
    current_mappings: dict[tuple[str, str], CREMapping],
) -> list[ChangesetEntry]:
    ops: list[ChangesetEntry] = []
    all_keys = sorted(set(prior_mappings) | set(current_mappings))
    for key in all_keys:
        old = prior_mappings.get(key)
        new = current_mappings.get(key)
        if old is None and new is not None:
            ops.append(ChangesetEntry(operation="ADD_MAPPING", entity=new))
        elif old is not None and new is None:
            ops.append(ChangesetEntry(
                operation="DELETE_MAPPING", key=f"{key[0]}|{key[1]}",
            ))
        elif old is not None and new is not None:
            if any(getattr(old, f) != getattr(new, f) for f in _MAPPING_MUTABLE_FIELDS):
                ops.append(ChangesetEntry(
                    operation="UPDATE_MAPPING", entity=new, before=old,
                ))
    return ops


def _compute_impact(
    operations: list[ChangesetEntry],
    framework_id: str,
    db_path: Path | None = None,
) -> ImpactAnalysis:
    affected_hubs: set[str] = set()
    for op in operations:
        if op.entity and isinstance(op.entity, CREMapping):
            affected_hubs.add(op.entity.hub_id)
        if op.before and isinstance(op.before, CREMapping):
            affected_hubs.add(op.before.hub_id)
        if op.key and "|" in op.key:
            affected_hubs.add(op.key.split("|")[1])

    co_mapped = 0
    affected_frameworks: list[str] = [framework_id]
    if db_path is not None and affected_hubs:
        conn = get_connection(db_path)
        try:
            placeholders = ",".join("?" for _ in affected_hubs)
            rows = conn.execute(
                f"SELECT DISTINCT c.framework_id FROM assignments a "
                f"JOIN controls c ON a.control_id = c.id "
                f"WHERE a.hub_id IN ({placeholders}) "
                f"AND c.framework_id != ?",
                [*affected_hubs, framework_id],
            ).fetchall()
            other_fws = [r["framework_id"] for r in rows]
            affected_frameworks.extend(sorted(other_fws))
            co_mapped_rows = conn.execute(
                f"SELECT COUNT(DISTINCT a.control_id) FROM assignments a "
                f"JOIN controls c ON a.control_id = c.id "
                f"WHERE a.hub_id IN ({placeholders}) "
                f"AND c.framework_id != ?",
                [*affected_hubs, framework_id],
            ).fetchone()
            co_mapped = co_mapped_rows[0] if co_mapped_rows else 0
        finally:
            conn.close()

    n_ops = len(operations)
    has_delete_mapping = any(op.operation == "DELETE_MAPPING" for op in operations)
    has_delete_control = any(op.operation == "DELETE_CONTROL" for op in operations)

    if n_ops > 50 or has_delete_control:
        scope = "major"
    elif n_ops >= 10 or has_delete_mapping:
        scope = "moderate"
    else:
        scope = "minor"

    return ImpactAnalysis(
        affected_hubs=sorted(affected_hubs),
        affected_frameworks=affected_frameworks,
        co_mapped_changes=co_mapped,
        scope=scope,
    )


def diff_snapshots(
    prior: StandardSnapshot | None,
    current: StandardSnapshot,
    db_path: Path | None = None,
) -> Changeset:
    """Compute changeset between two snapshots (spec §3)."""
    if prior is None:
        ops: list[ChangesetEntry] = []
        for ctrl in current.controls:
            ops.append(ChangesetEntry(operation="ADD_CONTROL", entity=ctrl))
        for mapping in current.mappings:
            ops.append(ChangesetEntry(operation="ADD_MAPPING", entity=mapping))
    else:
        prior_controls = {c.control_id: c for c in prior.controls}
        current_controls = {c.control_id: c for c in current.controls}
        prior_mappings = {(m.control_id, m.hub_id): m for m in prior.mappings}
        current_mappings = {(m.control_id, m.hub_id): m for m in current.mappings}

        ops = _diff_controls(prior_controls, current_controls)
        ops.extend(_diff_mappings(prior_mappings, current_mappings))

    summary = ChangesetSummary(
        controls_added=sum(1 for o in ops if o.operation == "ADD_CONTROL"),
        controls_updated=sum(1 for o in ops if o.operation == "UPDATE_CONTROL"),
        controls_deleted=sum(1 for o in ops if o.operation == "DELETE_CONTROL"),
        mappings_added=sum(1 for o in ops if o.operation == "ADD_MAPPING"),
        mappings_updated=sum(1 for o in ops if o.operation == "UPDATE_MAPPING"),
        mappings_deleted=sum(1 for o in ops if o.operation == "DELETE_MAPPING"),
    )

    impact = _compute_impact(ops, current.framework_id, db_path)

    return Changeset(
        framework_id=current.framework_id,
        from_version=prior.content_hash if prior else None,
        to_version=current.content_hash,
        export_date=current.export_date,
        operations=ops,
        summary=summary,
        impact=impact,
    )
