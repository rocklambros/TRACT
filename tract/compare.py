"""Cross-framework comparison via shared CRE hubs.

Factored from T5's _export_cross_framework_matrix() into a reusable
library function. Queries crosswalk.db live — no cached data.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tract.crosswalk.schema import get_connection
from tract.hierarchy import CREHierarchy

logger = logging.getLogger(__name__)


@dataclass
class Equivalence:
    hub_id: str
    hub_name: str
    controls: list[dict]  # [{control_id, framework_id, title}]
    frameworks: list[str]


@dataclass
class RelatedPair:
    hub_a: str
    hub_b: str
    parent_hub: str
    controls_a: list[dict]
    controls_b: list[dict]


@dataclass
class CrossFrameworkResult:
    equivalences: list[Equivalence]
    related: list[RelatedPair]
    gap_controls: dict[str, list[str]]
    framework_pair_overlap: dict[tuple[str, str], int]
    total_shared_hubs: int


def cross_framework_matrix(
    db_path: Path,
    framework_ids: list[str],
    hierarchy: CREHierarchy,
) -> CrossFrameworkResult:
    """Live query: relationship matrix between 2+ frameworks.

    Equivalent: controls assigned to the same hub.
    Related: controls whose hubs share a parent (via hierarchy.get_parent()).
    Gap: controls with no cross-framework match.
    """
    fw_set = set(framework_ids)

    conn = get_connection(db_path)
    try:
        placeholders = ",".join("?" * len(framework_ids))
        rows = conn.execute(
            f"SELECT a.control_id, a.hub_id, c.framework_id, c.title, "
            f"h.name AS hub_name "
            f"FROM assignments a "
            f"JOIN controls c ON a.control_id = c.id "
            f"JOIN hubs h ON a.hub_id = h.id "
            f"WHERE c.framework_id IN ({placeholders}) "
            f"AND a.review_status IN ('accepted', 'ground_truth') "
            f"ORDER BY a.hub_id",
            framework_ids,
        ).fetchall()
    finally:
        conn.close()

    hub_to_controls: dict[str, list[dict]] = defaultdict(list)
    control_hubs: dict[str, str] = {}
    all_control_ids_by_fw: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        hub_to_controls[row["hub_id"]].append({
            "control_id": row["control_id"],
            "framework_id": row["framework_id"],
            "title": row["title"],
        })
        control_hubs[row["control_id"]] = row["hub_id"]
        all_control_ids_by_fw[row["framework_id"]].add(row["control_id"])

    equivalences: list[Equivalence] = []
    matched_controls: set[str] = set()
    for hub_id, controls in hub_to_controls.items():
        fws_in_hub = {c["framework_id"] for c in controls} & fw_set
        if len(fws_in_hub) >= 2:
            hub_name = controls[0].get("title", hub_id)
            for row in rows:
                if row["hub_id"] == hub_id:
                    hub_name = row["hub_name"]
                    break
            equivalences.append(Equivalence(
                hub_id=hub_id,
                hub_name=hub_name,
                controls=controls,
                frameworks=sorted(fws_in_hub),
            ))
            matched_controls.update(c["control_id"] for c in controls)

    equivalences.sort(key=lambda e: (-len(e.frameworks), e.hub_id))

    related: list[RelatedPair] = []
    hub_ids_list = list(hub_to_controls.keys())
    for i, hub_a in enumerate(hub_ids_list):
        parent_a = hierarchy.get_parent(hub_a) if hub_a in hierarchy.hubs else None
        if parent_a is None:
            continue
        for hub_b in hub_ids_list[i + 1:]:
            parent_b = hierarchy.get_parent(hub_b) if hub_b in hierarchy.hubs else None
            if parent_b is None:
                continue
            if parent_a.hub_id == parent_b.hub_id:
                fws_a = {c["framework_id"] for c in hub_to_controls[hub_a]} & fw_set
                fws_b = {c["framework_id"] for c in hub_to_controls[hub_b]} & fw_set
                if fws_a != fws_b or len(fws_a) >= 2:
                    related.append(RelatedPair(
                        hub_a=hub_a,
                        hub_b=hub_b,
                        parent_hub=parent_a.hub_id,
                        controls_a=hub_to_controls[hub_a],
                        controls_b=hub_to_controls[hub_b],
                    ))
                    matched_controls.update(
                        c["control_id"] for c in hub_to_controls[hub_a]
                    )
                    matched_controls.update(
                        c["control_id"] for c in hub_to_controls[hub_b]
                    )

    gap_controls: dict[str, list[str]] = {}
    for fw_id in framework_ids:
        unmatched = sorted(all_control_ids_by_fw.get(fw_id, set()) - matched_controls)
        if unmatched:
            gap_controls[fw_id] = unmatched

    fw_pair_overlap: dict[tuple[str, str], int] = defaultdict(int)
    for eq in equivalences:
        for i, fw_a in enumerate(eq.frameworks):
            for fw_b in eq.frameworks[i + 1:]:
                pair = (min(fw_a, fw_b), max(fw_a, fw_b))
                fw_pair_overlap[pair] += 1

    return CrossFrameworkResult(
        equivalences=equivalences,
        related=related,
        gap_controls=gap_controls,
        framework_pair_overlap=dict(fw_pair_overlap),
        total_shared_hubs=len(equivalences),
    )
