"""Transform hierarchy/framework/link data into crosswalk DB record format."""
from __future__ import annotations

import logging

from tract.hierarchy import CREHierarchy
from tract.training.data_quality import TieredLink

logger = logging.getLogger(__name__)


def build_hub_records(hierarchy: CREHierarchy) -> list[dict]:
    """Convert CREHierarchy hubs to insert_hubs() format.

    Topologically sorted: parents before children (required by FK constraint).
    """
    from collections import deque

    all_ids = set(hierarchy.hubs.keys())
    children_map: dict[str | None, list[str]] = {}
    for hub_id, hub in hierarchy.hubs.items():
        parent = hub.parent_id if hub.parent_id in all_ids else None
        children_map.setdefault(parent, []).append(hub_id)

    ordered: list[str] = []
    queue: deque[str | None] = deque([None])
    while queue:
        parent = queue.popleft()
        for child_id in sorted(children_map.get(parent, [])):
            ordered.append(child_id)
            queue.append(child_id)

    records = []
    for hub_id in ordered:
        hub = hierarchy.hubs[hub_id]
        records.append({
            "id": hub.hub_id,
            "name": hub.name,
            "path": hub.hierarchy_path,
            "parent_id": hub.parent_id if hub.parent_id in all_ids else None,
        })
    return records


def build_framework_records(frameworks_data: list[dict]) -> list[dict]:
    """Convert parsed framework JSON metadata to insert_frameworks() format."""
    records = []
    for fw in frameworks_data:
        records.append({
            "id": fw["framework_id"],
            "name": fw.get("framework_name", fw["framework_id"]),
            "version": fw.get("version"),
            "fetch_date": fw.get("fetched_date"),
            "control_count": len(fw.get("controls", [])),
        })
    return records


def build_control_records(frameworks_data: list[dict]) -> list[dict]:
    """Convert parsed framework controls to insert_controls() format.

    Each control gets a composite ID: framework_id:control_id.
    Deduplicates by composite ID (keeps first occurrence).
    """
    seen: set[str] = set()
    records = []
    for fw in frameworks_data:
        fid = fw["framework_id"]
        for ctrl in fw.get("controls", []):
            cid = ctrl.get("control_id", "")
            composite = f"{fid}:{cid}"
            if composite in seen:
                continue
            seen.add(composite)
            records.append({
                "id": composite,
                "framework_id": fid,
                "section_id": cid,
                "title": ctrl.get("title", ""),
                "description": ctrl.get("description", ""),
                "full_text": ctrl.get("full_text", ""),
            })
    return records


def build_training_assignments(
    tiered_links: list[TieredLink],
    control_id_map: dict[tuple[str, str], str],
) -> list[dict]:
    """Convert TieredLinks to insert_assignments() format.

    Args:
        tiered_links: Filtered training links.
        control_id_map: Maps (standard_name, section_id) -> composite control_id in DB.
    """
    records = []
    skipped = 0
    for link in tiered_links:
        fw = link.link.get("standard_name", "")
        sid = link.link.get("section_id", "")
        key = (fw, sid)
        control_id = control_id_map.get(key)
        if control_id is None:
            skipped += 1
            continue

        records.append({
            "control_id": control_id,
            "hub_id": link.link["cre_id"],
            "confidence": None,
            "in_conformal_set": None,
            "is_ood": 0,
            "provenance": f"training_{link.tier.value}",
            "source_link_id": link.link.get("link_id"),
            "model_version": None,
            "review_status": "ground_truth",
        })

    if skipped:
        logger.warning("Skipped %d links with no matching control in DB", skipped)
    logger.info("Built %d training assignments from %d links", len(records), len(tiered_links))
    return records
