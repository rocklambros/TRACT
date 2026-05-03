"""Tests for review import — apply_review_decisions (UPDATE-in-place)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tract.crosswalk.schema import create_database, get_connection
from tract.review.import_review import apply_review_decisions


@pytest.fixture()
def review_import_db(tmp_path: Path) -> Path:
    """Create a test database with frameworks, controls, hubs, and assignments."""
    db_path = tmp_path / "review_import.db"
    create_database(db_path)

    conn = get_connection(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")

    conn.execute(
        "INSERT INTO frameworks (id, name) VALUES (?, ?)",
        ("fw_alpha", "Framework Alpha"),
    )

    for i in range(1, 5):
        conn.execute(
            "INSERT INTO controls (id, framework_id, section_id, title) VALUES (?, ?, ?, ?)",
            (f"ctrl-{i}", "fw_alpha", f"A.{i}", f"Control {i}"),
        )

    for i in range(1, 5):
        conn.execute(
            "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
            (f"hub-{i}", f"Hub {i}", f"/root/hub-{i}"),
        )

    for i in range(1, 5):
        conn.execute(
            "INSERT INTO assignments (control_id, hub_id, confidence, provenance, review_status) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"ctrl-{i}", f"hub-{i}", 0.85 + i * 0.01, "active_learning_round_2", "pending"),
        )

    conn.commit()
    conn.close()
    return db_path


def _write_review_json(path: Path, predictions: list[dict]) -> Path:
    """Write a review JSON file with the given predictions."""
    review_path = path / "review_predictions.json"
    data = {
        "metadata": {"total_predictions": len(predictions)},
        "predictions": predictions,
    }
    review_path.write_text(json.dumps(data), encoding="utf-8")
    return review_path


def _make_prediction(
    pred_id: int,
    status: str = "pending",
    *,
    assigned_hub_id: str = "hub-1",
    reviewer_hub_id: str | None = None,
    reviewer_notes: str | None = None,
) -> dict:
    """Build a minimal prediction dict for testing."""
    return {
        "id": pred_id,
        "control_id": f"ctrl-{pred_id}",
        "assigned_hub_id": assigned_hub_id,
        "assigned_hub_name": f"Hub {pred_id}",
        "confidence": 0.85,
        "status": status,
        "reviewer_hub_id": reviewer_hub_id,
        "reviewer_notes": reviewer_notes,
    }


class TestAccepted:
    def test_accepted_updates_review_fields(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        predictions = [_make_prediction(1, "accepted")]
        review_path = _write_review_json(tmp_path, predictions)

        result = apply_review_decisions(review_import_db, review_path, "expert_1")

        assert result["accepted"] == 1
        assert result["total"] == 1

        conn = get_connection(review_import_db)
        try:
            row = conn.execute(
                "SELECT review_status, reviewer, review_date FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["review_status"] == "accepted"
            assert row["reviewer"] == "expert_1"
            assert row["review_date"] is not None
        finally:
            conn.close()


class TestReassigned:
    def test_reassigned_updates_hub_and_tracks_original(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        predictions = [
            _make_prediction(
                1, "reassigned",
                assigned_hub_id="hub-1",
                reviewer_hub_id="hub-3",
                reviewer_notes="Better fit",
            ),
        ]
        review_path = _write_review_json(tmp_path, predictions)

        result = apply_review_decisions(review_import_db, review_path, "expert_1")

        assert result["reassigned"] == 1

        conn = get_connection(review_import_db)
        try:
            row = conn.execute(
                "SELECT hub_id, original_hub_id, confidence, review_status, "
                "reviewer, reviewer_notes FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["hub_id"] == "hub-3"
            assert row["original_hub_id"] == "hub-1"
            assert row["confidence"] is None
            assert row["review_status"] == "accepted"
            assert row["reviewer"] == "expert_1"
            assert "[Reassigned from hub hub-1]" in row["reviewer_notes"]
            assert "Better fit" in row["reviewer_notes"]
        finally:
            conn.close()


class TestRejected:
    def test_rejected_updates_status_and_notes(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        predictions = [
            _make_prediction(1, "rejected", reviewer_notes="No hub fits this control"),
        ]
        review_path = _write_review_json(tmp_path, predictions)

        result = apply_review_decisions(review_import_db, review_path, "expert_1")

        assert result["rejected"] == 1

        conn = get_connection(review_import_db)
        try:
            row = conn.execute(
                "SELECT review_status, reviewer, reviewer_notes FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["review_status"] == "rejected"
            assert row["reviewer"] == "expert_1"
            assert row["reviewer_notes"] == "No hub fits this control"
        finally:
            conn.close()


class TestPendingSkipped:
    def test_pending_items_not_updated(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        predictions = [
            _make_prediction(1, "pending"),
            _make_prediction(2, "accepted"),
        ]
        review_path = _write_review_json(tmp_path, predictions)

        result = apply_review_decisions(review_import_db, review_path, "expert_1")

        assert result["skipped_pending"] == 1
        assert result["accepted"] == 1

        conn = get_connection(review_import_db)
        try:
            row = conn.execute(
                "SELECT review_status, reviewer FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["review_status"] == "pending"
            assert row["reviewer"] is None
        finally:
            conn.close()


class TestCalibrationSkipped:
    def test_negative_id_predictions_skipped(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        predictions = [
            _make_prediction(-1, "accepted"),
            _make_prediction(-5, "rejected"),
            _make_prediction(1, "accepted"),
        ]
        review_path = _write_review_json(tmp_path, predictions)

        result = apply_review_decisions(review_import_db, review_path, "expert_1")

        assert result["skipped_calibration"] == 2
        assert result["accepted"] == 1
        assert result["total"] == 1


class TestIdempotent:
    def test_import_twice_same_result(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        predictions = [
            _make_prediction(1, "accepted"),
            _make_prediction(2, "rejected", reviewer_notes="Doesn't fit"),
        ]
        review_path = _write_review_json(tmp_path, predictions)

        first = apply_review_decisions(review_import_db, review_path, "expert_1")
        second = apply_review_decisions(review_import_db, review_path, "expert_1")

        assert first["accepted"] == second["accepted"]
        assert first["rejected"] == second["rejected"]

        conn = get_connection(review_import_db)
        try:
            row = conn.execute(
                "SELECT review_status, reviewer FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["review_status"] == "accepted"
            assert row["reviewer"] == "expert_1"
        finally:
            conn.close()


class TestAtomicRollback:
    def test_error_during_processing_rolls_back(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        from tract.review.validate import ValidationResult

        # Write a sabotaged file: first prediction is valid, second is
        # reassigned but missing reviewer_hub_id key → KeyError during processing.
        sabotaged_data = {
            "metadata": {"total_predictions": 2},
            "predictions": [
                _make_prediction(1, "accepted"),
                {"id": 2, "status": "reassigned"},
            ],
        }
        review_path = tmp_path / "review_predictions.json"
        review_path.write_text(json.dumps(sabotaged_data), encoding="utf-8")

        # Bypass validation so the sabotaged file is processed.
        with patch("tract.review.import_review.validate_review_json") as mock_val:
            mock_val.return_value = ValidationResult(valid=True)
            with pytest.raises(KeyError):
                apply_review_decisions(review_import_db, review_path, "expert_1")

        conn_after = get_connection(review_import_db)
        try:
            row = conn_after.execute(
                "SELECT review_status, reviewer FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["review_status"] == "pending"
            assert row["reviewer"] is None
        finally:
            conn_after.close()


class TestReviewerOverride:
    def test_different_reviewer_logs_warning_and_overrides(
        self, review_import_db: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        predictions = [_make_prediction(1, "accepted")]
        review_path = _write_review_json(tmp_path, predictions)

        apply_review_decisions(review_import_db, review_path, "expert_1")

        with caplog.at_level("WARNING"):
            apply_review_decisions(review_import_db, review_path, "expert_2")

        assert any("overriding" in r.message.lower() for r in caplog.records)

        conn = get_connection(review_import_db)
        try:
            row = conn.execute(
                "SELECT reviewer FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["reviewer"] == "expert_2"
        finally:
            conn.close()


class TestMigrateSchema:
    def test_columns_exist_after_import(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        predictions = [_make_prediction(1, "accepted")]
        review_path = _write_review_json(tmp_path, predictions)

        apply_review_decisions(review_import_db, review_path, "expert_1")

        conn = sqlite3.connect(str(review_import_db))
        cols = [row[1] for row in conn.execute("PRAGMA table_info(assignments)").fetchall()]
        conn.close()

        assert "reviewer_notes" in cols
        assert "original_hub_id" in cols


class TestValidationBlocksImport:
    def test_invalid_review_json_raises_valueerror(
        self, review_import_db: Path, tmp_path: Path,
    ) -> None:
        review_path = tmp_path / "review_predictions.json"
        data = {
            "metadata": {"total_predictions": 1},
            "predictions": [{"id": 1, "status": "approved"}],
        }
        review_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="validation failed"):
            apply_review_decisions(review_import_db, review_path, "expert_1")

        conn = get_connection(review_import_db)
        try:
            row = conn.execute(
                "SELECT review_status, reviewer FROM assignments WHERE id = 1",
            ).fetchone()
            assert row["review_status"] == "pending"
            assert row["reviewer"] is None
        finally:
            conn.close()
