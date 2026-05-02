"""Bridge candidate review validation and hierarchy commit."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from tract.config import HIERARCHY_BRIDGE_VERSION
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json

logger = logging.getLogger(__name__)

VALID_STATUSES = {"accepted", "rejected"}


def validate_candidates(
    candidates_data: dict,
    hierarchy_data: dict,
) -> list[str]:
    """Validate reviewed candidates. Returns list of error messages (empty = valid).

    Checks:
    - All candidates have status 'accepted' or 'rejected' (no 'pending')
    - All hub IDs exist in the hierarchy
    - Required keys present in each candidate
    """
    errors: list[str] = []
    hub_ids = set(hierarchy_data.get("hubs", {}).keys())
    required_keys = {"ai_hub_id", "trad_hub_id", "status"}

    for i, candidate in enumerate(candidates_data.get("candidates", [])):
        missing = required_keys - set(candidate.keys())
        if missing:
            errors.append(f"Candidate {i}: missing keys {missing}")
            continue

        status = candidate["status"]
        if status not in VALID_STATUSES:
            errors.append(
                f"Candidate {i} ({candidate['ai_hub_id']} -> {candidate['trad_hub_id']}): "
                f"invalid status '{status}', must be one of {VALID_STATUSES}"
            )

        for key in ("ai_hub_id", "trad_hub_id"):
            if candidate[key] not in hub_ids:
                errors.append(
                    f"Candidate {i}: {key} '{candidate[key]}' not in hierarchy"
                )

    return errors


def commit_bridges(
    candidates_data: dict,
    hierarchy_path: Path,
    report_path: Path,
) -> dict:
    """Commit accepted bridges to hierarchy and write bridge_report.json.

    Args:
        candidates_data: Reviewed bridge_candidates.json content.
        hierarchy_path: Path to cre_hierarchy.json (modified in place).
        report_path: Path to write bridge_report.json.

    Returns:
        The bridge report dict.
    """
    hier_data = load_json(hierarchy_path)

    errors = validate_candidates(candidates_data, hier_data)
    if errors:
        raise ValueError(
            f"Candidate validation failed with {len(errors)} errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    accepted = [
        c for c in candidates_data["candidates"]
        if c["status"] == "accepted"
    ]
    rejected = [
        c for c in candidates_data["candidates"]
        if c["status"] == "rejected"
    ]

    if accepted:
        for bridge in accepted:
            ai_id = bridge["ai_hub_id"]
            trad_id = bridge["trad_hub_id"]

            ai_related = hier_data["hubs"][ai_id].get("related_hub_ids", [])
            if trad_id not in ai_related:
                ai_related.append(trad_id)
            hier_data["hubs"][ai_id]["related_hub_ids"] = sorted(ai_related)

            trad_related = hier_data["hubs"][trad_id].get("related_hub_ids", [])
            if ai_id not in trad_related:
                trad_related.append(ai_id)
            hier_data["hubs"][trad_id]["related_hub_ids"] = sorted(trad_related)

        hier_data["version"] = HIERARCHY_BRIDGE_VERSION

        updated = CREHierarchy.model_validate(hier_data)
        updated.validate_integrity()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "method": candidates_data.get("method", "top_k_per_ai_hub"),
            "top_k": candidates_data.get("top_k", 3),
        },
        "counts": {
            "total": len(candidates_data.get("candidates", [])),
            "accepted": len(accepted),
            "rejected": len(rejected),
        },
        "candidates": [
            {
                "ai_hub_id": c["ai_hub_id"],
                "trad_hub_id": c["trad_hub_id"],
                "cosine_similarity": c["cosine_similarity"],
                "status": c["status"],
            }
            for c in candidates_data["candidates"]
        ],
        "similarity_stats": candidates_data.get("similarity_stats", {}),
    }

    atomic_write_json(report, report_path)
    logger.info("Wrote bridge report to %s", report_path)

    if accepted:
        updated.save(hierarchy_path)
        logger.info("Updated hierarchy with %d bridges", len(accepted))

    return report
