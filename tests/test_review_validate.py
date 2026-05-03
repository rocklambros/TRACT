"""Tests for review JSON validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.crosswalk.schema import create_database, get_connection
from tract.review.validate import ValidationResult, validate_review_json


@pytest.fixture()
def validation_db(tmp_path: Path) -> Path:
    """Create a DB with frameworks, controls, hubs, and assignments for validation tests."""
    db_path = tmp_path / "validate.db"
    create_database(db_path)
    conn = get_connection(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        "INSERT INTO frameworks (id, name) VALUES (?, ?)",
        ("fw_a", "Framework A"),
    )
    conn.execute(
        "INSERT INTO controls (id, framework_id, section_id, title) VALUES (?, ?, ?, ?)",
        ("fw_a:ctrl-1", "fw_a", "ctrl-1", "Control One"),
    )
    conn.execute(
        "INSERT INTO controls (id, framework_id, section_id, title) VALUES (?, ?, ?, ?)",
        ("fw_a:ctrl-2", "fw_a", "ctrl-2", "Control Two"),
    )
    conn.execute(
        "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
        ("hub-1", "Hub One", "/Security/Hub One"),
    )
    conn.execute(
        "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
        ("hub-2", "Hub Two", "/Security/Hub Two"),
    )
    conn.execute(
        "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
        ("hub-3", "Hub Three", "/Security/Hub Three"),
    )
    conn.execute(
        "INSERT INTO assignments (control_id, hub_id, confidence, provenance) "
        "VALUES (?, ?, ?, ?)",
        ("fw_a:ctrl-1", "hub-1", 0.85, "active_learning_round_2"),
    )
    conn.execute(
        "INSERT INTO assignments (control_id, hub_id, confidence, provenance) "
        "VALUES (?, ?, ?, ?)",
        ("fw_a:ctrl-2", "hub-2", 0.72, "active_learning_round_2"),
    )
    conn.commit()
    conn.close()
    return db_path


def _write_review_json(path: Path, data: object) -> Path:
    """Write a review JSON file and return its path."""
    review_path = path / "review.json"
    review_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return review_path


def _write_review_raw(path: Path, text: str) -> Path:
    """Write raw text as a review JSON file (for syntax error tests)."""
    review_path = path / "review.json"
    review_path.write_text(text, encoding="utf-8")
    return review_path


def _minimal_prediction(
    pred_id: int = 1,
    status: str = "accepted",
    **overrides: object,
) -> dict:
    """Build a minimal valid prediction dict."""
    pred: dict = {"id": pred_id, "status": status}
    pred.update(overrides)
    return pred


class TestValidFilePassesValidation:
    def test_all_statuses_valid(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {"total_predictions": 3},
            "predictions": [
                _minimal_prediction(1, "accepted"),
                _minimal_prediction(2, "rejected", reviewer_notes="No fit"),
                _minimal_prediction(-1, "accepted"),
            ],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is True
        assert result.errors == []

    def test_reassigned_with_valid_hub(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {},
            "predictions": [
                _minimal_prediction(1, "reassigned", reviewer_hub_id="hub-3"),
            ],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is True
        assert result.errors == []


class TestJsonSyntaxError:
    def test_reports_line_and_column(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        broken = '{\n  "metadata": {},\n  "predictions": [bad]\n}'
        result = validate_review_json(
            _write_review_raw(tmp_path, broken), validation_db,
        )
        assert result.valid is False
        assert len(result.errors) == 1
        assert "line" in result.errors[0].lower()


class TestInvalidStatus:
    def test_catches_unknown_status(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {},
            "predictions": [_minimal_prediction(1, "approved")],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is False
        assert any("invalid status" in e and "'approved'" in e for e in result.errors)


class TestInvalidReviewerHubId:
    def test_catches_nonexistent_hub(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {},
            "predictions": [
                _minimal_prediction(1, "reassigned", reviewer_hub_id="nonexistent"),
            ],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is False
        assert any("nonexistent" in e and "not found" in e for e in result.errors)

    def test_catches_missing_hub_on_reassigned(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {},
            "predictions": [_minimal_prediction(1, "reassigned")],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is False
        assert any("missing or empty" in e for e in result.errors)


class TestNonExistentAssignmentId:
    def test_catches_unknown_assignment(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {},
            "predictions": [_minimal_prediction(99999, "accepted")],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is False
        assert any("99999" in e and "not found" in e for e in result.errors)


class TestCalibrationItemsSkipped:
    def test_negative_id_not_validated_against_db(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {},
            "predictions": [
                _minimal_prediction(-5, "accepted"),
                _minimal_prediction(-1, "rejected"),
            ],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is True
        assert result.errors == []


class TestPendingItemsProduceWarning:
    def test_pending_is_warning_not_error(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {
            "metadata": {},
            "predictions": [
                _minimal_prediction(1, "pending"),
                _minimal_prediction(2, "pending"),
            ],
        }
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is True
        assert result.errors == []
        assert len(result.warnings) == 1
        assert "2 prediction(s) still pending" in result.warnings[0]


class TestMissingTopLevelKeys:
    def test_missing_predictions_key(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {"metadata": {}}
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is False
        assert any("'predictions'" in e for e in result.errors)

    def test_missing_metadata_key(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {"predictions": [_minimal_prediction(1, "accepted")]}
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is False
        assert any("'metadata'" in e for e in result.errors)

    def test_predictions_not_a_list(
        self, tmp_path: Path, validation_db: Path,
    ) -> None:
        data = {"metadata": {}, "predictions": "not a list"}
        result = validate_review_json(
            _write_review_json(tmp_path, data), validation_db,
        )
        assert result.valid is False
        assert any("must be a JSON array" in e for e in result.errors)
