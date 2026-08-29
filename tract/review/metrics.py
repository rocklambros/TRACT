"""TRACT review metrics — coverage, rates, calibration quality, confidence analysis."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Final

from tract.review.types import (
    AcceptanceRate,
    CalibrationDisagreement,
    CalibrationQuality,
    ConfidenceAnalysis,
    CoverageMetrics,
    FrameworkRates,
    OverallRates,
    ReviewExportDocument,
    ReviewItem,
    ReviewMetrics,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# The provenance the export draws calibration items from. Items in a review
# file carrying it are calibration; the main review query excludes this
# provenance, so nothing else in the file can have it.
CALIBRATION_PROVENANCE: Final[str] = "ground_truth_T1-AI"


def _calibration_ids_from_db(db_path: Path) -> frozenset[int]:
    """Assignment ids the export would have drawn calibration items from.

    A superset of what any one export selected -- selection samples this pool --
    but sound as a membership test for ids that are actually in a review file,
    because the reviewable query excludes this provenance entirely.

    Returns an empty set when the database has no such rows, which is the
    honest answer: an export built against that database had no calibration
    items either, and `_compute_calibration_quality` already reports a null
    quality score for an empty set.
    """
    from tract.crosswalk.schema import get_connection  # noqa: PLC0415

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id FROM assignments WHERE provenance = ?",
            (CALIBRATION_PROVENANCE,),
        ).fetchall()
    finally:
        conn.close()
    return frozenset(int(r["id"]) for r in rows)


def compute_review_metrics(
    db_path: Path,
    review_data: ReviewExportDocument,
    output_path: Path,
) -> ReviewMetrics:
    """Compute review metrics and save to JSON.

    Includes: coverage, overall rates, per-framework breakdown,
    reviewer quality (calibration items), confidence analysis.

    Args:
        db_path: Path to crosswalk.db (reserved for future use).
        review_data: Full review JSON dict with "metadata" and "predictions".
        output_path: Path to write review_metrics.json.

    Returns:
        The full metrics dict.
    """
    predictions = review_data.get("predictions", [])

    # Calibration membership comes from the database, not from a marker in the
    # reviewed file. It used to be `id < 0`, which is exactly the tell that let
    # the reviewer these items measure pick them out of their own worklist
    # (F19). The reviewer-facing export now carries real assignment ids and a
    # uniform provenance, so the only surviving discriminator is the store --
    # which the reviewer never sees and which cannot drift from it.
    calibration_ids = _calibration_ids_from_db(db_path)
    real = [p for p in predictions if p.get("id") not in calibration_ids]
    calibration = [p for p in predictions if p.get("id") in calibration_ids]

    coverage = _compute_coverage(real)
    overall = _compute_overall_rates(real)
    per_framework = _compute_per_framework(real)
    reviewer_quality = _compute_calibration_quality(calibration)
    confidence = _compute_confidence_analysis(real)

    import_round = 1
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            import_round = existing.get("import_round", 0) + 1
        except (json.JSONDecodeError, OSError):
            pass

    metrics: ReviewMetrics = {
        "import_round": import_round,
        "coverage": coverage,
        "overall": overall,
        "per_framework": per_framework,
        "reviewer_quality": reviewer_quality,
        "confidence_analysis": confidence,
    }

    _write_metrics(output_path, metrics)
    logger.info(
        "Review metrics written: round=%d, reviewed=%d/%d (%.1f%%)",
        import_round,
        coverage["reviewed"],
        coverage["total_predictions"],
        coverage["completion_pct"],
    )
    return metrics


def _compute_coverage(real: list[ReviewItem]) -> CoverageMetrics:
    total = len(real)
    reviewed = sum(1 for p in real if p.get("status") != "pending")
    pending = total - reviewed
    completion_pct = (reviewed / total * 100) if total > 0 else 0.0
    return {
        "total_predictions": total,
        "reviewed": reviewed,
        "pending": pending,
        "completion_pct": round(completion_pct, 1),
    }


def _compute_overall_rates(real: list[ReviewItem]) -> OverallRates:
    reviewed = [p for p in real if p.get("status") != "pending"]
    n = len(reviewed)
    accepted = sum(1 for p in reviewed if p.get("status") == "accepted")
    rejected = sum(1 for p in reviewed if p.get("status") == "rejected")
    reassigned = sum(1 for p in reviewed if p.get("status") == "reassigned")
    return {
        "accepted": accepted,
        "accepted_rate": round(accepted / n * 100, 1) if n > 0 else 0.0,
        "rejected": rejected,
        "rejected_rate": round(rejected / n * 100, 1) if n > 0 else 0.0,
        "reassigned": reassigned,
        "reassigned_rate": round(reassigned / n * 100, 1) if n > 0 else 0.0,
    }


def _compute_per_framework(real: list[ReviewItem]) -> dict[str, FrameworkRates]:
    frameworks: dict[str, FrameworkRates] = {}
    for p in real:
        if p.get("status") == "pending":
            continue
        fw_id = p.get("framework_id", "unknown")
        if fw_id not in frameworks:
            frameworks[fw_id] = {
                "framework_name": p.get("framework_name", fw_id),
                "accepted": 0,
                "rejected": 0,
                "reassigned": 0,
            }
        # Explicit branches: a TypedDict cannot be indexed by a str variable.
        status = p.get("status")
        if status == "accepted":
            frameworks[fw_id]["accepted"] += 1
        elif status == "rejected":
            frameworks[fw_id]["rejected"] += 1
        elif status == "reassigned":
            frameworks[fw_id]["reassigned"] += 1
    return frameworks


def _compute_calibration_quality(calibration: list[ReviewItem]) -> CalibrationQuality:
    reviewed = [c for c in calibration if c.get("status") != "pending"]
    if not reviewed:
        return {
            "total_calibration": len(calibration),
            "reviewed": 0,
            "agreed": 0,
            "quality_score": None,
            "disagreements": [],
        }

    agreed = 0
    disagreements: list[CalibrationDisagreement] = []
    for c in reviewed:
        # `reviewed` already excludes status == "pending", but .get() still
        # types as str | None and CalibrationDisagreement declares status as
        # str. Defaulting names the impossible case rather than asserting it
        # away: an item with no status at all is a malformed record, and
        # recording it as such is more useful in a disagreement report than
        # crashing the metrics run that would have surfaced it.
        status = c.get("status") or "missing"
        if status == "accepted":
            agreed += 1
        else:
            disagreements.append({
                "id": c.get("id"),
                "assigned_hub_id": c.get("assigned_hub_id"),
                "status": status,
                "reviewer_hub_id": c.get("reviewer_hub_id"),
            })

    quality_score = agreed / len(reviewed) if reviewed else None
    return {
        "total_calibration": len(calibration),
        "reviewed": len(reviewed),
        "agreed": agreed,
        "quality_score": round(quality_score, 3) if quality_score is not None else None,
        "disagreements": disagreements,
    }


def _compute_confidence_analysis(real: list[ReviewItem]) -> ConfidenceAnalysis:
    reviewed = [p for p in real if p.get("status") != "pending"]
    if not reviewed:
        return {
            "high_confidence": {"total": 0, "accepted": 0, "acceptance_rate": 0.0},
            "low_confidence": {"total": 0, "accepted": 0, "acceptance_rate": 0.0},
            "ood_items": {"total": 0, "accepted": 0, "acceptance_rate": 0.0},
        }

    threshold = _DEFAULT_CONFIDENCE_THRESHOLD

    high = [p for p in reviewed if (p.get("confidence") or 0) > threshold]
    low = [p for p in reviewed if (p.get("confidence") or 0) <= threshold]
    ood = [p for p in reviewed if p.get("is_ood")]

    def _rate(items: list[ReviewItem]) -> AcceptanceRate:
        total = len(items)
        accepted = sum(1 for i in items if i.get("status") == "accepted")
        return {
            "total": total,
            "accepted": accepted,
            "acceptance_rate": round(accepted / total * 100, 1) if total > 0 else 0.0,
        }

    return {
        "high_confidence": _rate(high),
        "low_confidence": _rate(low),
        "ood_items": _rate(ood),
    }


def _write_metrics(output_path: Path, data: ReviewMetrics) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".review_metrics.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
