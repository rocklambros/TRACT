"""Coverage gaps report for OpenCRE export.

For each exported framework, identifies controls that were NOT included
in the export and why. Written alongside CSVs as coverage_gaps.json.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from tract.config import PHASE5_GROUND_TRUTH_PROVENANCE
from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)

# Rows come back from sqlite3 via dict(row), so the value type is genuinely
# heterogeneous (TEXT, REAL, INTEGER, NULL) and Any is the honest annotation
# rather than a placeholder.
AssignmentRow = dict[str, Any]
ControlRow = dict[str, Any]


class MissingControl(TypedDict):
    """One control that did not make it into the export, and why."""

    section_id: str
    title: str
    reason: str
    detail: str


class FrameworkCoverage(TypedDict):
    """Per-framework coverage block of coverage_gaps.json."""

    total_controls: int
    exported_controls: int
    coverage_pct: float
    missing_controls: list[MissingControl]


def query_coverage_gaps(
    db_path: Path,
    exported_keys: set[tuple[str, str]],
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    framework_ids: list[str],
) -> dict[str, FrameworkCoverage]:
    """Build per-framework coverage gaps report.

    Returns a dict keyed by framework_id, each containing:
      - total_controls: int
      - exported_controls: int
      - missing_controls: list of {section_id, title, reason, detail}
    """
    conn = get_connection(db_path)
    try:
        controls = conn.execute(
            "SELECT c.id, c.framework_id, c.section_id, c.title "
            "FROM controls c "
            "WHERE c.framework_id IN ({}) "
            "ORDER BY c.framework_id, c.section_id".format(
                ",".join("?" for _ in framework_ids)
            ),
            framework_ids,
        ).fetchall()

        assignments = conn.execute(
            "SELECT a.control_id, a.hub_id, a.confidence, a.is_ood, "
            "a.provenance, a.review_status "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "WHERE c.framework_id IN ({})".format(
                ",".join("?" for _ in framework_ids)
            ),
            framework_ids,
        ).fetchall()
    finally:
        conn.close()

    assignments_by_control: dict[str, list[AssignmentRow]] = {}
    for a in assignments:
        assignments_by_control.setdefault(a["control_id"], []).append(dict(a))

    controls_by_fw: dict[str, list[ControlRow]] = {}
    for c in controls:
        controls_by_fw.setdefault(c["framework_id"], []).append(dict(c))

    exported_control_ids = {cid for cid, _ in exported_keys}

    report: dict[str, FrameworkCoverage] = {}
    for fw_id in framework_ids:
        fw_controls = controls_by_fw.get(fw_id, [])
        floor = confidence_overrides.get(fw_id, confidence_floor)
        missing: list[MissingControl] = []

        for ctrl in fw_controls:
            if ctrl["id"] in exported_control_ids:
                continue

            ctrl_assignments = assignments_by_control.get(ctrl["id"], [])
            reason, detail = _classify_gap(ctrl_assignments, floor)
            missing.append({
                "section_id": ctrl["section_id"],
                "title": ctrl["title"],
                "reason": reason,
                "detail": detail,
            })

        exported_count = len(fw_controls) - len(missing)
        report[fw_id] = {
            "total_controls": len(fw_controls),
            "exported_controls": exported_count,
            "coverage_pct": round(exported_count / len(fw_controls) * 100, 1) if fw_controls else 0.0,
            "missing_controls": missing,
        }

    return report


def _classify_gap(
    assignments: list[AssignmentRow],
    confidence_floor: float,
) -> tuple[str, str]:
    """Determine why a control was excluded from export.

    Returns (reason_code, human_readable_detail).
    """
    if not assignments:
        return "no_assignment", "No hub assignment exists for this control"

    dominated_by: dict[str, list[AssignmentRow]] = {
        "not_accepted": [],
        "ground_truth": [],
        "null_confidence": [],
        "out_of_distribution": [],
        "below_confidence_floor": [],
    }

    for a in assignments:
        if a["provenance"] == PHASE5_GROUND_TRUTH_PROVENANCE:
            dominated_by["ground_truth"].append(a)
        elif a["review_status"] != "accepted":
            dominated_by["not_accepted"].append(a)
        elif a["confidence"] is None:
            dominated_by["null_confidence"].append(a)
        elif a["is_ood"] == 1:
            dominated_by["out_of_distribution"].append(a)
        elif a["confidence"] < confidence_floor:
            dominated_by["below_confidence_floor"].append(a)

    if dominated_by["below_confidence_floor"]:
        best = max(dominated_by["below_confidence_floor"], key=lambda a: a["confidence"])
        return (
            "below_confidence_floor",
            f"Best confidence {best['confidence']:.3f} < floor {confidence_floor:.2f}",
        )
    if dominated_by["not_accepted"]:
        statuses = {a["review_status"] for a in dominated_by["not_accepted"]}
        return "not_accepted", f"Review status: {', '.join(sorted(statuses))}"
    if dominated_by["null_confidence"]:
        return "null_confidence", "Assignment exists but confidence score is NULL"
    if dominated_by["out_of_distribution"]:
        return "out_of_distribution", "Flagged as out-of-distribution"
    if dominated_by["ground_truth"]:
        return "ground_truth", "Reserved as ground truth training data"

    return "unknown", "Excluded for unknown reason"
