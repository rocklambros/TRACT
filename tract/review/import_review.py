"""TRACT review import — apply expert review decisions to crosswalk.db."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Final

from tract.crosswalk.ground_truth import _backup_database
from tract.crosswalk.schema import get_connection, migrate_schema
from tract.review.validate import validate_review_json

logger = logging.getLogger(__name__)

# The provenance values generate_review_export is willing to put in front of a
# reviewer (export.py:65). The import used to trust the `id` in the returned
# document and nothing else: validate.py checks that the id EXISTS, not that it
# belongs to the scope that was exported, so a returned file naming any positive
# id rewrote that row -- including the ground_truth_T1-AI rows that are the
# calibration control over the reviewer submitting the file. Scoping the UPDATE
# itself is what makes the export's narrowing mean something on the way back in.
REVIEWABLE_PROVENANCE: Final[tuple[str, ...]] = (
    "active_learning_round_2", "model_prediction",
)
_PROVENANCE_SQL: Final[str] = (
    " AND provenance IN (" + ", ".join("?" * len(REVIEWABLE_PROVENANCE)) + ")"
)


def _require_in_scope(cursor: sqlite3.Cursor, pred_id: int, action: str) -> None:
    """Refuse a decision that matched no in-scope row, rather than counting it.

    rowcount is 0 when the id exists but its provenance is outside
    REVIEWABLE_PROVENANCE -- a returned document naming a ground-truth or
    calibration row. Silently skipping it would leave the caller's accepted /
    reassigned / rejected tally reporting work that never happened, which is the
    same shape of failure as the decision landing: the operator believes the
    store now says something it does not.
    """
    if cursor.rowcount == 0:
        raise ValueError(
            f"Review decision '{action}' names assignment id {pred_id}, which is "
            f"not an in-scope reviewable row. Only "
            f"{', '.join(REVIEWABLE_PROVENANCE)} rows are exported for review, "
            f"so this id was either never sent out or points at ground-truth or "
            f"calibration data. Refusing the whole import: a review file that "
            f"names rows it was not given is not one to apply selectively."
        )


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
                cur = conn.execute(
                    "UPDATE assignments "
                    "SET review_status = 'accepted', "
                    "    reviewer = ?, "
                    "    review_date = datetime('now'), "
                    "    reviewer_notes = ? "
                    "WHERE id = ?" + _PROVENANCE_SQL,
                    (reviewer, notes or None, pred_id, *REVIEWABLE_PROVENANCE),
                )
                _require_in_scope(cur, pred_id, "accept")
                accepted += 1

            elif status == "reassigned":
                new_hub_id: str = pred["reviewer_hub_id"]
                old_hub_id: str = pred.get("assigned_hub_id", "")
                reassign_note = f"[Reassigned from hub {old_hub_id}]"
                if notes:
                    reassign_note = f"{reassign_note} {notes}"

                cur = conn.execute(
                    "UPDATE assignments "
                    "SET original_hub_id = hub_id, "
                    "    hub_id = ?, "
                    "    confidence = NULL, "
                    "    review_status = 'accepted', "
                    "    reviewer = ?, "
                    "    review_date = datetime('now'), "
                    "    reviewer_notes = ? "
                    "WHERE id = ?" + _PROVENANCE_SQL,
                    (new_hub_id, reviewer, reassign_note, pred_id,
                     *REVIEWABLE_PROVENANCE),
                )
                _require_in_scope(cur, pred_id, "reassign")
                reassigned += 1

            elif status == "rejected":
                cur = conn.execute(
                    "UPDATE assignments "
                    "SET review_status = 'rejected', "
                    "    reviewer = ?, "
                    "    review_date = datetime('now'), "
                    "    reviewer_notes = ? "
                    "WHERE id = ?" + _PROVENANCE_SQL,
                    (reviewer, notes or None, pred_id, *REVIEWABLE_PROVENANCE),
                )
                _require_in_scope(cur, pred_id, "reject")
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
