"""TRACT review import — apply expert review decisions to crosswalk.db."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tract.crosswalk.ground_truth import _backup_database
from tract.crosswalk.schema import get_connection, migrate_schema
from tract.review.validate import validate_review_json

logger = logging.getLogger(__name__)


def apply_review_decisions(
    db_path: Path,
    review_path: Path,
    reviewer: str,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Apply expert review decisions to crosswalk.db.

    Updates assignments in a single transaction. Returns summary dict.
    Skips calibration items (id < 0) and pending items.

    Args:
        db_path: Path to crosswalk.db.
        review_path: Path to reviewed predictions JSON.
        reviewer: Name of the reviewer (stored in assignments.reviewer).
        dry_run: If True, roll back instead of committing.

    Returns:
        Summary dict with counts: accepted, rejected, reassigned,
        skipped_pending, skipped_calibration, total.

    Raises:
        ValueError: If validation fails (errors in the review JSON).
    """
    migrate_schema(db_path)

    result = validate_review_json(review_path, db_path)
    if not result.valid:
        raise ValueError(
            f"Review JSON validation failed with {len(result.errors)} error(s): "
            + "; ".join(result.errors)
        )

    if not dry_run:
        _backup_database(db_path)

    data = json.loads(review_path.read_text(encoding="utf-8"))
    predictions = data["predictions"]

    conn = get_connection(db_path)
    accepted = 0
    rejected = 0
    reassigned = 0
    skipped_pending = 0
    skipped_calibration = 0

    try:
        for pred in predictions:
            pred_id: int = pred["id"]
            status: str = pred["status"]

            if pred_id < 0:
                skipped_calibration += 1
                continue

            if status == "pending":
                skipped_pending += 1
                continue

            existing = conn.execute(
                "SELECT reviewer, hub_id FROM assignments WHERE id = ?",
                (pred_id,),
            ).fetchone()

            if existing is not None and existing["reviewer"] is not None:
                if existing["reviewer"] != reviewer:
                    logger.warning(
                        "Assignment %d already reviewed by '%s', overriding with '%s'",
                        pred_id, existing["reviewer"], reviewer,
                    )

            notes: str = pred.get("reviewer_notes") or ""

            if status == "accepted":
                conn.execute(
                    "UPDATE assignments "
                    "SET review_status = 'accepted', "
                    "    reviewer = ?, "
                    "    review_date = datetime('now'), "
                    "    reviewer_notes = ? "
                    "WHERE id = ?",
                    (reviewer, notes or None, pred_id),
                )
                accepted += 1

            elif status == "reassigned":
                new_hub_id: str = pred["reviewer_hub_id"]
                old_hub_id: str = pred.get("assigned_hub_id", "")
                reassign_note = f"[Reassigned from hub {old_hub_id}]"
                if notes:
                    reassign_note = f"{reassign_note} {notes}"

                conn.execute(
                    "UPDATE assignments "
                    "SET original_hub_id = hub_id, "
                    "    hub_id = ?, "
                    "    confidence = NULL, "
                    "    review_status = 'accepted', "
                    "    reviewer = ?, "
                    "    review_date = datetime('now'), "
                    "    reviewer_notes = ? "
                    "WHERE id = ?",
                    (new_hub_id, reviewer, reassign_note, pred_id),
                )
                reassigned += 1

            elif status == "rejected":
                conn.execute(
                    "UPDATE assignments "
                    "SET review_status = 'rejected', "
                    "    reviewer = ?, "
                    "    review_date = datetime('now'), "
                    "    reviewer_notes = ? "
                    "WHERE id = ?",
                    (reviewer, notes or None, pred_id),
                )
                rejected += 1

        if dry_run:
            conn.rollback()
            logger.info("Dry run — rolled back %d updates", accepted + rejected + reassigned)
        else:
            conn.commit()
            logger.info(
                "Applied review decisions: accepted=%d, rejected=%d, reassigned=%d",
                accepted, rejected, reassigned,
            )

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "accepted": accepted,
        "rejected": rejected,
        "reassigned": reassigned,
        "skipped_pending": skipped_pending,
        "skipped_calibration": skipped_calibration,
        "total": accepted + rejected + reassigned,
    }
