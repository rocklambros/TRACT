"""TRACT review export — re-run inference and build reviewer-ready JSON.

Queries in-scope assignments (active_learning_round_2 + model_prediction,
unreviewed, not GT-confirmed), re-runs inference from scratch to get fresh
confidence + raw_similarity values, and writes a structured JSON file for
human review.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from tract.config import (
    PHASE3_GT_PROVENANCE,
    PHASE3_MODEL_PROVENANCE,
    PHASE3_TEXT_QUALITY_HIGH_THRESHOLD,
    PHASE3_TEXT_QUALITY_LOW_THRESHOLD,
)
from tract.crosswalk.schema import get_connection
from tract.io import load_json

logger = logging.getLogger(__name__)

# SQL to fetch assignments that need review:
#   - provenance is active_learning_round_2 or model_prediction
#   - not yet reviewed (reviewer IS NULL)
#   - not already confirmed by a GT assignment for the same (control, hub) pair
_REVIEW_QUERY = """
SELECT
    a.id,
    a.control_id,
    a.hub_id,
    a.provenance,
    a.is_ood,
    c.title,
    c.description,
    c.full_text,
    c.framework_id,
    c.section_id,
    f.name AS framework_name
FROM assignments a
JOIN controls c ON a.control_id = c.id
JOIN frameworks f ON c.framework_id = f.id
WHERE a.provenance IN ('active_learning_round_2', 'model_prediction')
  AND a.reviewer IS NULL
  AND NOT EXISTS (
    SELECT 1 FROM assignments a2
    WHERE a2.control_id = a.control_id
      AND a2.hub_id = a.hub_id
      AND a2.provenance = 'opencre_ground_truth'
  )
ORDER BY f.name, a.id
"""


def _compute_text_quality(text_length: int) -> str:
    """Classify combined inference text length into quality tier.

    Args:
        text_length: Character count of the combined control text.

    Returns:
        "high" if >= PHASE3_TEXT_QUALITY_HIGH_THRESHOLD,
        "medium" if >= PHASE3_TEXT_QUALITY_LOW_THRESHOLD,
        "low" otherwise.
    """
    if text_length >= PHASE3_TEXT_QUALITY_HIGH_THRESHOLD:
        return "high"
    if text_length >= PHASE3_TEXT_QUALITY_LOW_THRESHOLD:
        return "medium"
    return "low"


def _compute_review_priority(
    confidence: float,
    is_ood: bool,
    text_quality: str,
    global_threshold: float,
) -> str:
    """Classify assignment into review priority tier.

    Priority rules (applied in order):
        critical — confidence at or below threshold AND text_quality is "low"
        careful  — confidence at or below threshold OR is_ood
        routine  — confidence above threshold AND not OOD

    Args:
        confidence: Calibrated confidence from fresh inference.
        is_ood: Whether the assignment is out-of-distribution.
        text_quality: One of "high", "medium", "low".
        global_threshold: Decision threshold from calibration.json.

    Returns:
        One of "critical", "careful", "routine".
    """
    below_threshold = confidence <= global_threshold
    if below_threshold and text_quality == "low":
        return "critical"
    if below_threshold or is_ood:
        return "careful"
    return "routine"


def generate_review_export(
    db_path: Path,
    model_dir: Path,
    output_dir: Path,
    calibration_path: Path,
) -> dict:
    """Build reviewer-ready JSON from in-scope assignments.

    Queries unreviewed, non-GT-confirmed assignments, re-runs inference to
    obtain fresh confidence + raw_similarity values, and writes a structured
    JSON file to output_dir/review_export.json.

    Args:
        db_path: Path to crosswalk.db.
        model_dir: Directory containing the deployment model and artifacts.
        output_dir: Directory to write review_export.json into.
        calibration_path: Path to calibration.json with global_threshold.

    Returns:
        The metadata dict (same object written into the JSON file).

    Raises:
        FileNotFoundError: If db_path, model_dir, or calibration_path do
            not exist.
        ValueError: If calibration.json is missing the global_threshold key.
        KeyError: If calibration.json is malformed.
    """
    # Lazy import — avoids loading heavyweight model at module import time.
    from tract.inference import TRACTPredictor

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    calibration = load_json(calibration_path)
    if "global_threshold" not in calibration:
        raise ValueError(
            f"calibration.json is missing required key 'global_threshold': {calibration_path}"
        )
    global_threshold: float = float(calibration["global_threshold"])

    logger.info("Loading TRACTPredictor from %s", model_dir)
    predictor = TRACTPredictor(model_dir)
    model_version: str = predictor._artifacts.model_adapter_hash[:12]

    # ── Query in-scope assignments ─────────────────────────────────────────
    conn = get_connection(db_path)
    try:
        rows = conn.execute(_REVIEW_QUERY).fetchall()
        logger.info("Found %d in-scope assignments for review export", len(rows))

        # Fetch hub metadata (name, path) for all referenced hub IDs.
        all_hub_ids: set[str] = {row["hub_id"] for row in rows}
        hub_rows = conn.execute(
            "SELECT id, name, path FROM hubs WHERE id IN ({})".format(
                ",".join("?" * len(all_hub_ids))
            ),
            list(all_hub_ids),
        ).fetchall() if all_hub_ids else []
        hub_meta: dict[str, dict[str, str]] = {
            r["id"]: {"name": r["name"], "path": r["path"] or ""}
            for r in hub_rows
        }
    finally:
        conn.close()

    if not rows:
        logger.warning("No assignments found for review export.")
        metadata: dict = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
            "total_predictions": 0,
            "calibration_items": 0,
            "framework_breakdown": {},
            "priority_breakdown": {"critical": 0, "careful": 0, "routine": 0},
        }
        _write_export(output_dir, {"metadata": metadata, "predictions": []})
        return metadata

    # ── Re-run inference on all controls ─────────────────────────────────
    texts: list[str] = []
    for row in rows:
        combined = " ".join(
            part for part in [row["title"], row["description"], row["full_text"]]
            if part
        )
        texts.append(combined)

    logger.info("Running batch inference on %d texts (top_k=3)", len(texts))
    batch_predictions = predictor.predict_batch(texts, top_k=3)

    # ── Build prediction records ──────────────────────────────────────────
    predictions: list[dict] = []
    framework_breakdown: dict[str, int] = {}
    priority_breakdown: dict[str, int] = {"critical": 0, "careful": 0, "routine": 0}

    for row, preds in zip(rows, batch_predictions):
        assigned_hub_id: str = row["hub_id"]
        text: str = texts[len(predictions)]
        text_quality = _compute_text_quality(len(text))

        # Find the prediction matching the assigned hub.
        assigned_pred = None
        for p in preds:
            if p.hub_id == assigned_hub_id:
                assigned_pred = p
                break

        if assigned_pred is None:
            # Assigned hub not in top-3; use the first prediction's OOD/conformal
            # flags but record confidence as 0.0 and similarity as 0.0 to signal
            # the hub ranked below our top-3 window.
            logger.debug(
                "Assigned hub %s not in top-3 for assignment %d; confidence=0.0",
                assigned_hub_id, row["id"],
            )
            first_pred = preds[0] if preds else None
            is_ood_val: bool = bool(first_pred.is_ood) if first_pred else True
            in_conformal_set_val: bool = False
            confidence_val: float = 0.0
            raw_similarity_val: float = 0.0
        else:
            is_ood_val = bool(assigned_pred.is_ood)
            in_conformal_set_val = bool(assigned_pred.in_conformal_set)
            confidence_val = float(assigned_pred.calibrated_confidence)
            raw_similarity_val = float(assigned_pred.raw_similarity)

        review_priority = _compute_review_priority(
            confidence_val, is_ood_val, text_quality, global_threshold,
        )

        # Collect alternative hubs: top-3 excluding the assigned hub.
        alternative_hubs: list[dict] = []
        for p in preds:
            if p.hub_id != assigned_hub_id:
                alternative_hubs.append({
                    "hub_id": p.hub_id,
                    "hub_name": p.hub_name,
                    "confidence": float(p.calibrated_confidence),
                })
            if len(alternative_hubs) >= 2:
                break

        hub_info = hub_meta.get(assigned_hub_id, {"name": assigned_hub_id, "path": ""})
        framework_id: str = row["framework_id"]

        predictions.append({
            "id": row["id"],
            "control_id": row["control_id"],
            "framework_id": framework_id,
            "framework_name": row["framework_name"],
            "section_id": row["section_id"],
            "control_title": row["title"] or "",
            "control_text": text,
            "assigned_hub_id": assigned_hub_id,
            "assigned_hub_name": hub_info["name"],
            "assigned_hub_path": hub_info["path"],
            "confidence": confidence_val,
            "raw_similarity": raw_similarity_val,
            "is_ood": is_ood_val,
            "in_conformal_set": in_conformal_set_val,
            "text_quality": text_quality,
            "review_priority": review_priority,
            "provenance": row["provenance"],
            "alternative_hubs": alternative_hubs,
            "decision": None,
            "reviewer_hub_id": None,
            "reviewer_notes": None,
            "status": "pending",
        })

        framework_breakdown[framework_id] = framework_breakdown.get(framework_id, 0) + 1
        priority_breakdown[review_priority] = priority_breakdown.get(review_priority, 0) + 1

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "total_predictions": len(predictions),
        "calibration_items": 0,
        "framework_breakdown": framework_breakdown,
        "priority_breakdown": priority_breakdown,
    }

    export_doc: dict = {"metadata": metadata, "predictions": predictions}
    _write_export(output_dir, export_doc)

    logger.info(
        "Review export written: %d predictions (%s)",
        len(predictions),
        ", ".join(f"{k}={v}" for k, v in priority_breakdown.items()),
    )
    return metadata


def _write_export(output_dir: Path, data: dict) -> Path:
    """Atomically write export_doc to output_dir/review_export.json.

    Uses write-to-temp-then-rename to prevent partial writes on crash.

    Args:
        output_dir: Target directory (created if absent).
        data: JSON-serializable export document.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "review_export.json"

    fd, tmp_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=".review_export.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.debug("Atomically wrote review export to %s", target)
    return target
