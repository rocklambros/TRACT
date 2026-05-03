"""TRACT review validation — pre-import checks for reviewed prediction JSON."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)

_VALID_STATUSES = frozenset({"pending", "accepted", "reassigned", "rejected"})


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_review_json(
    review_path: Path,
    db_path: Path,
) -> ValidationResult:
    """Validate a reviewed predictions JSON file.

    Checks:
    - JSON parse succeeds (report line context on failure)
    - Top-level structure: metadata + predictions array
    - Every non-pending status is valid (accepted/reassigned/rejected)
    - Every reviewer_hub_id (on reassigned) is a valid hub ID in DB
    - Every id matches an existing assignment (skip calibration items with id < 0)
    - Warns (not fails) if pending items remain
    """
    errors: list[str] = []
    warnings: list[str] = []

    raw = review_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(
            f"JSON parse error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        )
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if not isinstance(data, dict):
        errors.append("Top-level value must be a JSON object")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if "metadata" not in data:
        errors.append("Missing required top-level key: 'metadata'")
    if "predictions" not in data:
        errors.append("Missing required top-level key: 'predictions'")
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    predictions = data["predictions"]
    if not isinstance(predictions, list):
        errors.append("'predictions' must be a JSON array")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    conn = get_connection(db_path)
    try:
        hub_ids: set[str] = {
            row[0] for row in conn.execute("SELECT id FROM hubs").fetchall()
        }
        assignment_ids: set[int] = {
            row[0]
            for row in conn.execute("SELECT id FROM assignments").fetchall()
        }
    finally:
        conn.close()

    pending_count = 0

    for idx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            errors.append(f"predictions[{idx}]: expected object, got {type(pred).__name__}")
            continue

        if "id" not in pred:
            errors.append(f"predictions[{idx}]: missing required field 'id'")
        else:
            pred_id = pred["id"]
            if isinstance(pred_id, int) and pred_id >= 0 and pred_id not in assignment_ids:
                errors.append(
                    f"predictions[{idx}]: assignment id={pred_id} not found in database"
                )

        if "status" not in pred:
            errors.append(f"predictions[{idx}]: missing required field 'status'")
            continue

        status = pred["status"]
        if status not in _VALID_STATUSES:
            errors.append(
                f"predictions[{idx}]: invalid status '{status}' "
                f"(expected one of: {', '.join(sorted(_VALID_STATUSES))})"
            )
            continue

        if status == "pending":
            pending_count += 1

        if status == "reassigned":
            reviewer_hub = pred.get("reviewer_hub_id")
            if not reviewer_hub:
                errors.append(
                    f"predictions[{idx}]: status is 'reassigned' but "
                    f"'reviewer_hub_id' is missing or empty"
                )
            elif reviewer_hub not in hub_ids:
                errors.append(
                    f"predictions[{idx}]: reviewer_hub_id '{reviewer_hub}' "
                    f"not found in hubs table"
                )

    if pending_count > 0:
        warnings.append(f"{pending_count} prediction(s) still pending review")

    result = ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )

    if result.valid:
        logger.info("Validation passed (%d warnings)", len(warnings))
    else:
        logger.warning("Validation failed with %d error(s)", len(errors))

    return result
