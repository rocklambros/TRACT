"""TRACT dataset bundler — assemble staging directory with all dataset files."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from tract.config import PHASE3_PROVENANCE_PRIORITY
from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)

_CROSSWALK_QUERY = """
SELECT
    a.control_id,
    a.hub_id,
    a.confidence,
    a.provenance,
    a.source_link_id,
    a.review_status,
    a.reviewer_notes,
    a.original_hub_id,
    c.title AS control_title,
    c.section_id,
    c.framework_id,
    f.name AS framework_name,
    h.name AS hub_name,
    h.path AS hub_path
FROM assignments a
JOIN controls c ON a.control_id = c.id
JOIN frameworks f ON c.framework_id = f.id
JOIN hubs h ON a.hub_id = h.id
ORDER BY a.control_id, a.hub_id,
         CASE a.provenance
           WHEN 'opencre_ground_truth' THEN 1
           WHEN 'ground_truth_T1-AI' THEN 2
           WHEN 'active_learning_round_2' THEN 3
           WHEN 'model_prediction' THEN 4
           ELSE 5
         END
"""

_GT_PROVENANCES = frozenset({"opencre_ground_truth", "ground_truth_T1-AI"})

_LICENSE_TEXT = """\
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

You are free to:
  Share — copy and redistribute the material in any medium or format
  Adapt — remix, transform, and build upon the material for any purpose

Under the following terms:
  Attribution — You must give appropriate credit, provide a link to the
  license, and indicate if changes were made.
  ShareAlike — If you remix, transform, or build upon the material, you
  must distribute your contributions under the same license.

Full text: https://creativecommons.org/licenses/by-sa/4.0/legalcode
"""


def _derive_assignment_type(row: dict) -> str:
    """Derive assignment_type from provenance, review_status, and original_hub_id."""
    provenance = row["provenance"]
    review_status = row["review_status"]
    original_hub_id = row["original_hub_id"]
    source_link_id = row["source_link_id"]

    if provenance == "opencre_ground_truth":
        if source_link_id == "LinkedTo":
            return "ground_truth_linked"
        return "ground_truth_auto"

    if review_status == "rejected":
        return "model_rejected"

    if review_status == "accepted" and original_hub_id is not None:
        return "model_reassigned"

    if review_status == "accepted":
        return "model_accepted"

    return "model_accepted"


def _build_crosswalk_jsonl(db_path: Path, output_path: Path) -> int:
    """Query all assignments, dedup by (control_id, hub_id), write JSONL.

    Rows are priority-ordered by provenance via SQL CASE. Dedup keeps the
    first row per (control_id, hub_id) group.

    Returns row count after dedup.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(_CROSSWALK_QUERY).fetchall()
    finally:
        conn.close()

    seen: set[tuple[str, str]] = set()
    count = 0

    fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".crosswalk.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                key = (row["control_id"], row["hub_id"])
                if key in seen:
                    continue
                seen.add(key)

                record = {
                    "control_id": row["control_id"],
                    "framework_id": row["framework_id"],
                    "framework_name": row["framework_name"],
                    "section_id": row["section_id"],
                    "control_title": row["control_title"] or "",
                    "hub_id": row["hub_id"],
                    "hub_name": row["hub_name"],
                    "hub_path": row["hub_path"] or "",
                    "assignment_type": _derive_assignment_type(dict(row)),
                    "confidence": row["confidence"],
                    "provenance": row["provenance"],
                    "review_status": row["review_status"],
                    "reviewer_notes": row["reviewer_notes"],
                }
                fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
                fh.write("\n")
                count += 1

        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Wrote %d deduplicated rows to %s", count, output_path)
    return count


def _build_framework_metadata(db_path: Path, output_path: Path) -> list[dict]:
    """Generate per-framework statistics.

    Returns list of framework metadata dicts, also written to output_path.
    """
    conn = get_connection(db_path)
    try:
        frameworks = conn.execute(
            "SELECT id, name, control_count FROM frameworks ORDER BY id"
        ).fetchall()

        metadata: list[dict] = []
        for fw in frameworks:
            fw_id = fw["id"]

            assigned_controls = conn.execute(
                "SELECT COUNT(DISTINCT control_id) FROM assignments "
                "WHERE control_id IN (SELECT id FROM controls WHERE framework_id = ?)",
                (fw_id,),
            ).fetchone()[0]

            assignment_count = conn.execute(
                "SELECT COUNT(*) FROM assignments "
                "WHERE control_id IN (SELECT id FROM controls WHERE framework_id = ?)",
                (fw_id,),
            ).fetchone()[0]

            provenances = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT provenance FROM assignments "
                    "WHERE control_id IN (SELECT id FROM controls WHERE framework_id = ?)",
                    (fw_id,),
                ).fetchall()
            }

            has_gt = bool(provenances & _GT_PROVENANCES)
            has_model = bool(provenances - _GT_PROVENANCES)

            if has_gt and has_model:
                coverage_type = "mixed"
            elif has_gt:
                coverage_type = "ground_truth"
            else:
                coverage_type = "model_prediction"

            metadata.append({
                "framework_id": fw_id,
                "framework_name": fw["name"],
                "total_controls": fw["control_count"] or 0,
                "assigned_controls": assigned_controls,
                "assignment_count": assignment_count,
                "coverage_type": coverage_type,
            })
    finally:
        conn.close()

    _atomic_json_write(output_path, metadata)
    logger.info("Wrote framework metadata for %d frameworks", len(metadata))
    return metadata


def _build_zenodo_metadata(output_path: Path) -> None:
    """Generate zenodo_metadata.json for manual Zenodo upload."""
    zenodo = {
        "title": "TRACT Crosswalk Dataset v1.0",
        "description": (
            "Human-reviewed security framework crosswalk mapping "
            "31 frameworks to 522 CRE hubs"
        ),
        "creators": [{"name": "Lambros, Rock", "orcid": ""}],
        "license": "CC-BY-SA-4.0",
        "keywords": [
            "security",
            "crosswalk",
            "CRE",
            "AI-security",
            "framework-mapping",
        ],
        "version": "1.0",
        "resource_type": "dataset",
    }
    _atomic_json_write(output_path, zenodo)
    logger.info("Wrote Zenodo metadata to %s", output_path)


def _atomic_json_write(output_path: Path, data: object) -> None:
    """Atomically write JSON data to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
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


def bundle_dataset(
    db_path: Path,
    staging_dir: Path,
    hierarchy_path: Path,
    hub_descriptions_path: Path,
    bridge_report_path: Path,
    review_metrics_path: Path,
) -> dict:
    """Assemble staging directory with all dataset files.

    Creates: crosswalk_v1.0.jsonl, framework_metadata.json,
    cre_hierarchy_v1.1.json, hub_descriptions_v1.0.json,
    bridge_report.json, review_metrics.json, LICENSE, zenodo_metadata.json

    Returns bundle stats dict with total_rows, frameworks count, and file list.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    total_rows = _build_crosswalk_jsonl(
        db_path, staging_dir / "crosswalk_v1.0.jsonl",
    )

    fw_metadata = _build_framework_metadata(
        db_path, staging_dir / "framework_metadata.json",
    )

    _build_zenodo_metadata(staging_dir / "zenodo_metadata.json")

    shutil.copy2(hierarchy_path, staging_dir / "cre_hierarchy_v1.1.json")
    shutil.copy2(hub_descriptions_path, staging_dir / "hub_descriptions_v1.0.json")
    shutil.copy2(bridge_report_path, staging_dir / "bridge_report.json")
    shutil.copy2(review_metrics_path, staging_dir / "review_metrics.json")

    license_path = staging_dir / "LICENSE"
    license_path.write_text(_LICENSE_TEXT, encoding="utf-8")

    files = sorted(p.name for p in staging_dir.iterdir() if p.is_file())

    logger.info(
        "Dataset bundle complete: %d rows, %d frameworks, %d files",
        total_rows, len(fw_metadata), len(files),
    )

    return {
        "total_rows": total_rows,
        "frameworks": len(fw_metadata),
        "files": files,
    }
