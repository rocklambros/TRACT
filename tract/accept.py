"""Accept reviewed ingest predictions into crosswalk DB.

Reads a _review.json produced by `tract ingest`, processes human review
decisions, and commits framework + controls + assignments to crosswalk.db.
"""
from __future__ import annotations

import logging
from pathlib import Path

from tract.crosswalk.schema import get_connection
from tract.crosswalk.store import insert_assignments, insert_controls, insert_frameworks

logger = logging.getLogger(__name__)


def accept_review(
    db_path: Path,
    review_data: dict,
    force: bool = False,
) -> dict:
    """Process reviewed ingest file and commit to crosswalk DB.

    Args:
        db_path: Path to crosswalk.db.
        review_data: Parsed review JSON from `tract ingest`.
        force: If True, delete existing framework data before inserting.

    Returns:
        Summary dict with counts of inserted/skipped items.

    Raises:
        ValueError: If framework already exists (without --force), or if
            a corrected control is missing corrected_hub_id.
    """
    fw_id = review_data["framework_id"]
    fw_name = review_data["framework_name"]
    model_version = review_data.get("model_version", "unknown")

    conn = get_connection(db_path)
    try:
        existing = conn.execute(
            "SELECT id FROM frameworks WHERE id = ?", (fw_id,)
        ).fetchone()
    finally:
        conn.close()

    if existing and not force:
        raise ValueError(
            f"Framework '{fw_id}' already exists in crosswalk DB. "
            "Use --force to replace."
        )

    if existing and force:
        _delete_framework_data(db_path, fw_id)
        logger.info("Deleted existing framework data for '%s'", fw_id)

    controls_to_insert: list[dict] = []
    assignments_to_insert: list[dict] = []
    accepted_count = 0
    rejected_count = 0
    corrected_count = 0
    pending_count = 0
    provenance = f"ingest_{fw_id}"

    for ctrl in review_data["controls"]:
        control_db_id = f"{fw_id}:{ctrl['control_id']}"

        controls_to_insert.append({
            "id": control_db_id,
            "framework_id": fw_id,
            "section_id": ctrl["control_id"],
            "title": ctrl.get("title", ""),
            "description": ctrl.get("description", ""),
            "full_text": ctrl.get("full_text"),
        })

        review = ctrl.get("review", {})
        status = review.get("status", "pending") if review else "pending"

        if status == "pending":
            pending_count += 1
            logger.warning("Control %s still pending review — skipping assignment", ctrl["control_id"])
            continue

        if status == "rejected":
            rejected_count += 1
            continue

        if status == "corrected":
            corrected_hub_id = review.get("corrected_hub_id")
            if not corrected_hub_id:
                raise ValueError(
                    f"Control {ctrl['control_id']} marked as corrected but "
                    "missing corrected_hub_id in review"
                )
            hub_id = corrected_hub_id
            confidence = None
            in_conformal_set = None
            corrected_count += 1
        elif status == "accepted":
            preds = ctrl.get("predictions", [])
            if not preds:
                logger.warning("Control %s accepted but has no predictions — skipping", ctrl["control_id"])
                continue
            top = preds[0]
            hub_id = top["hub_id"]
            confidence = top.get("calibrated_confidence", top.get("confidence"))
            in_conformal_set = 1 if top.get("in_conformal_set", False) else 0
            accepted_count += 1
        else:
            logger.warning("Unknown review status '%s' for control %s — skipping", status, ctrl["control_id"])
            continue

        assignments_to_insert.append({
            "control_id": control_db_id,
            "hub_id": hub_id,
            "confidence": confidence,
            "in_conformal_set": in_conformal_set,
            "is_ood": 1 if ctrl.get("is_ood", False) else 0,
            "provenance": provenance,
            "source_link_id": None,
            "model_version": model_version,
            "review_status": status,
        })

    insert_frameworks(db_path, [{
        "id": fw_id,
        "name": fw_name,
        "version": review_data.get("version", ""),
        "fetch_date": review_data.get("fetched_date", ""),
        "control_count": len(controls_to_insert),
    }])

    if controls_to_insert:
        insert_controls(db_path, controls_to_insert)

    if assignments_to_insert:
        insert_assignments(db_path, assignments_to_insert)

    summary = {
        "framework_id": fw_id,
        "framework_inserted": True,
        "controls_inserted": len(controls_to_insert),
        "assignments_created": len(assignments_to_insert),
        "accepted": accepted_count,
        "corrected": corrected_count,
        "rejected": rejected_count,
        "pending": pending_count,
    }

    logger.info(
        "Accepted framework '%s': %d controls, %d assignments "
        "(%d accepted, %d corrected, %d rejected, %d pending)",
        fw_id, len(controls_to_insert), len(assignments_to_insert),
        accepted_count, corrected_count, rejected_count, pending_count,
    )

    return summary


def _delete_framework_data(db_path: Path, framework_id: str) -> None:
    """Delete a framework and all its controls and assignments."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "DELETE FROM assignments WHERE control_id IN "
                "(SELECT id FROM controls WHERE framework_id = ?)",
                (framework_id,),
            )
            conn.execute(
                "DELETE FROM controls WHERE framework_id = ?",
                (framework_id,),
            )
            conn.execute(
                "DELETE FROM frameworks WHERE id = ?",
                (framework_id,),
            )
    finally:
        conn.close()
