"""End-to-end integration test: GT import → review export → review import → dataset bundle."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tract.crosswalk.ground_truth import import_ground_truth
from tract.crosswalk.schema import create_database, get_connection
from tract.dataset.bundle import bundle_dataset
from tract.review.export import generate_review_export
from tract.review.import_review import apply_review_decisions


def _make_prediction(
    hub_id: str, hub_name: str, confidence: float, *, is_ood: bool = False,
) -> MagicMock:
    p = MagicMock()
    p.hub_id = hub_id
    p.hub_name = hub_name
    p.calibrated_confidence = confidence
    p.raw_similarity = confidence * 0.8
    p.is_ood = is_ood
    p.in_conformal_set = confidence > 0.5
    return p


def _setup_db(db_path: Path) -> None:
    """Populate a test DB with frameworks, controls, hubs, and GT-T1 assignments."""
    create_database(db_path)
    conn = get_connection(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")

    conn.execute(
        "INSERT INTO frameworks (id, name, version) VALUES (?, ?, ?)",
        ("fw_alpha", "Alpha Framework", "1.0"),
    )
    conn.execute(
        "INSERT INTO frameworks (id, name, version) VALUES (?, ?, ?)",
        ("fw_beta", "Beta Framework", "2.0"),
    )

    for i in range(1, 6):
        conn.execute(
            "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
            (f"hub-{i}", f"Hub {i}", f"/root/hub-{i}"),
        )

    conn.execute(
        "INSERT INTO controls (id, framework_id, section_id, title, description) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_alpha:A.1", "fw_alpha", "A.1", "Alpha Control One",
         "Description for alpha control one " * 20),
    )
    conn.execute(
        "INSERT INTO controls (id, framework_id, section_id, title, description) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_alpha:A.2", "fw_alpha", "A.2", "Alpha Control Two",
         "Description for alpha control two " * 20),
    )
    conn.execute(
        "INSERT INTO controls (id, framework_id, section_id, title, description) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_alpha:A.3", "fw_alpha", "A.3", "Alpha Control Three",
         "Description for alpha control three " * 20),
    )
    conn.execute(
        "INSERT INTO controls (id, framework_id, section_id, title, description) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_beta:B.1", "fw_beta", "B.1", "Beta Control One",
         "Description for beta control one " * 20),
    )
    conn.execute(
        "INSERT INTO controls (id, framework_id, section_id, title, description) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_beta:B.2", "fw_beta", "B.2", "Beta Control Two",
         "Description for beta control two " * 20),
    )

    # Pre-existing AL assignments (from Phase 1).
    # A.2 → hub-2 (AL), A.3 → hub-3 (AL),
    # B.1 → hub-1 (AL), B.2 → hub-4 (AL)
    for ctrl, hub in [
        ("fw_alpha:A.2", "hub-2"),
        ("fw_alpha:A.3", "hub-3"),
        ("fw_beta:B.1", "hub-1"),
        ("fw_beta:B.2", "hub-4"),
    ]:
        conn.execute(
            "INSERT INTO assignments (control_id, hub_id, confidence, provenance, review_status) "
            "VALUES (?, ?, ?, ?, ?)",
            (ctrl, hub, 0.75, "active_learning_round_2", "pending"),
        )

    # ground_truth_T1-AI assignments (calibration source).
    # A.2→hub-2 overlaps with AL above (tests GT-confirmed exclusion in export).
    conn.execute(
        "INSERT INTO assignments (control_id, hub_id, provenance, review_status, source_link_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_alpha:A.2", "hub-2", "ground_truth_T1-AI", "ground_truth", "LinkedTo"),
    )
    conn.execute(
        "INSERT INTO assignments (control_id, hub_id, provenance, review_status, source_link_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_alpha:A.3", "hub-5", "ground_truth_T1-AI", "ground_truth", "LinkedTo"),
    )

    conn.commit()
    conn.close()


@patch("tract.inference.TRACTPredictor")
def test_full_pipeline(mock_predictor_cls: MagicMock, tmp_path: Path) -> None:
    """GT import → review export → review import → bundle."""
    db_path = tmp_path / "integration.db"
    _setup_db(db_path)

    # ── Step 1: Import ground truth ──────────────────────────────────
    hub_links = {
        "fw_alpha": [
            {"section_id": "A.1", "cre_id": "hub-1", "link_type": "LinkedTo"},
            {"section_id": "A.2", "cre_id": "hub-2", "link_type": "AutomaticallyLinkedTo"},
            {"section_id": "A.3", "cre_id": "hub-3", "link_type": "LinkedTo"},
        ],
    }
    hub_links_path = tmp_path / "hub_links.json"
    hub_links_path.write_text(json.dumps(hub_links), encoding="utf-8")

    gt_result = import_ground_truth(db_path, hub_links_path, dry_run=False)

    # A.1→hub-1: no pre-existing assignment → imported (1)
    # A.2→hub-2: exists as AL → skipped duplicate
    # A.3→hub-3: exists as AL → skipped duplicate
    assert gt_result["imported"] == 1
    assert gt_result["skipped_duplicate"] == 2

    conn = get_connection(db_path)
    gt_rows = conn.execute(
        "SELECT * FROM assignments WHERE provenance = 'opencre_ground_truth'"
    ).fetchall()
    assert len(gt_rows) == 1
    assert gt_rows[0]["control_id"] == "fw_alpha:A.1"

    # Add an AL assignment for A.1→hub-1 AFTER GT import to test exclusion.
    # This simulates an AL prediction that overlaps with GT — should be excluded
    # from review export by the NOT EXISTS clause.
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        "INSERT INTO assignments (control_id, hub_id, confidence, provenance, review_status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("fw_alpha:A.1", "hub-1", 0.80, "active_learning_round_2", "pending"),
    )
    conn.commit()
    conn.close()

    # ── Step 2: Review export (mocked inference) ─────────────────────
    mock_pred = MagicMock()
    mock_pred._artifacts = MagicMock()
    mock_pred._artifacts.model_adapter_hash = "abc123def456"

    def _predict_batch(texts: list[str], top_k: int = 3) -> list[list[MagicMock]]:
        return [
            [
                _make_prediction("hub-1", "Hub 1", 0.9),
                _make_prediction("hub-2", "Hub 2", 0.7),
                _make_prediction("hub-3", "Hub 3", 0.5),
            ]
            for _ in texts
        ]

    mock_pred.predict_batch.side_effect = _predict_batch
    mock_predictor_cls.return_value = mock_pred

    cal_path = tmp_path / "calibration.json"
    cal_path.write_text('{"global_threshold": 0.5}', encoding="utf-8")
    output_dir = tmp_path / "review_output"

    metadata = generate_review_export(db_path, tmp_path / "model", output_dir, cal_path)

    export_path = output_dir / "review_export.json"
    assert export_path.exists()
    review_data = json.loads(export_path.read_text(encoding="utf-8"))

    # The review query excludes GT-confirmed overlaps.
    # A.1→hub-1 has opencre_ground_truth → AL(A.1→hub-1) EXCLUDED by NOT EXISTS
    # A.2→hub-2 has GT-T1 but NOT opencre_ground_truth → INCLUDED
    # A.3→hub-3 no opencre_ground_truth → INCLUDED
    # B.1→hub-1 no opencre_ground_truth → INCLUDED
    # B.2→hub-4 no opencre_ground_truth → INCLUDED
    real_preds = [p for p in review_data["predictions"] if p["id"] >= 0]
    calibration_preds = [p for p in review_data["predictions"] if p["id"] < 0]

    assert len(real_preds) == 4
    real_ctrl_ids = {p["control_id"] for p in real_preds}
    # A.1→hub-1 excluded, but A.2, A.3, B.1, B.2 included
    assert "fw_alpha:A.2" in real_ctrl_ids
    assert "fw_beta:B.1" in real_ctrl_ids

    # Calibration items come from ground_truth_T1-AI (2 items, but we have < 20
    # so we get whatever is available).
    assert all(p["id"] < 0 for p in calibration_preds)
    assert all(p["status"] == "pending" for p in calibration_preds)

    # ── Step 3: Simulate expert review ───────────────────────────────
    pred_ids = sorted(p["id"] for p in real_preds)
    assert len(pred_ids) == 4

    for pred in review_data["predictions"]:
        if pred["id"] < 0:
            pred["status"] = "accepted"
        elif pred["id"] == pred_ids[0]:
            pred["status"] = "accepted"
            pred["reviewer_notes"] = "Looks correct"
        elif pred["id"] == pred_ids[1]:
            pred["status"] = "accepted"
        elif pred["id"] == pred_ids[2]:
            pred["status"] = "reassigned"
            pred["reviewer_hub_id"] = "hub-5"
            pred["reviewer_notes"] = "Better fit"
        elif pred["id"] == pred_ids[3]:
            pred["status"] = "rejected"
            pred["reviewer_notes"] = "No appropriate hub"

    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(
        json.dumps(review_data, sort_keys=True), encoding="utf-8",
    )

    # ── Step 4: Import review decisions ──────────────────────────────
    import_summary = apply_review_decisions(db_path, reviewed_path, "expert_1")

    assert import_summary["accepted"] == 2
    assert import_summary["reassigned"] == 1
    assert import_summary["rejected"] == 1
    assert import_summary["skipped_calibration"] == len(calibration_preds)

    # Verify reassigned assignment has original_hub_id
    conn = get_connection(db_path)
    reassigned_row = conn.execute(
        "SELECT * FROM assignments WHERE id = ?", (pred_ids[2],)
    ).fetchone()
    assert reassigned_row["hub_id"] == "hub-5"
    assert reassigned_row["original_hub_id"] is not None
    assert reassigned_row["confidence"] is None
    assert reassigned_row["reviewer"] == "expert_1"
    assert reassigned_row["review_status"] == "accepted"

    # Verify rejected assignment
    rejected_row = conn.execute(
        "SELECT * FROM assignments WHERE id = ?", (pred_ids[3],)
    ).fetchone()
    assert rejected_row["review_status"] == "rejected"
    assert rejected_row["reviewer"] == "expert_1"
    conn.close()

    # ── Step 5: Bundle dataset ───────────────────────────────────────
    # Create dummy supporting files
    for fname in ["hierarchy.json", "hub_descriptions.json",
                  "bridge_report.json", "review_metrics.json"]:
        (tmp_path / fname).write_text("{}", encoding="utf-8")

    staging_dir = tmp_path / "staging"
    stats = bundle_dataset(
        db_path,
        staging_dir,
        tmp_path / "hierarchy.json",
        tmp_path / "hub_descriptions.json",
        tmp_path / "bridge_report.json",
        tmp_path / "review_metrics.json",
    )

    # JSONL should exist and be non-empty
    jsonl_path = staging_dir / "crosswalk_v1.0.jsonl"
    assert jsonl_path.exists()
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert stats["total_rows"] == len(rows)
    assert stats["total_rows"] > 0

    # Check dedup: each (control_id, hub_id) appears at most once
    pairs = [(r["control_id"], r["hub_id"]) for r in rows]
    assert len(pairs) == len(set(pairs)), "JSONL has duplicate (control_id, hub_id)"

    # Verify assignment_type values are present and valid
    valid_types = {
        "ground_truth_linked", "ground_truth_auto", "model_accepted",
        "model_reassigned", "model_rejected",
    }
    for row in rows:
        assert "assignment_type" in row
        assert row["assignment_type"] in valid_types

    # The GT-linked A.3 should be ground_truth_linked
    a3_rows = [r for r in rows if r["control_id"] == "fw_alpha:A.3"]
    assert len(a3_rows) >= 1
    a3_gt = [r for r in a3_rows if r["provenance"] == "opencre_ground_truth"]
    if a3_gt:
        assert a3_gt[0]["assignment_type"] == "ground_truth_linked"

    # Reassigned assignment should be model_reassigned
    reassigned_rows = [r for r in rows if r["assignment_type"] == "model_reassigned"]
    assert len(reassigned_rows) >= 1

    # Verify framework_metadata and LICENSE exist
    assert (staging_dir / "framework_metadata.json").exists()
    assert (staging_dir / "LICENSE").exists()
    assert (staging_dir / "zenodo_metadata.json").exists()
