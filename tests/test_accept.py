"""Tests for tract accept command — committing reviewed predictions to crosswalk DB."""
from __future__ import annotations

import json

import pytest

from tract.accept import accept_review
from tract.crosswalk.schema import create_database, get_connection
from tract.crosswalk.store import insert_hubs


def _make_review_data(controls: list[dict] | None = None) -> dict:
    """Build a minimal review file structure."""
    if controls is None:
        controls = [
            {
                "control_id": "TC-01",
                "title": "Access Control",
                "description": "Enforce access control for AI models",
                "full_text": None,
                "is_ood": False,
                "predictions": [
                    {"hub_id": "h1", "hub_name": "Hub 1", "calibrated_confidence": 0.85,
                     "raw_similarity": 0.42, "in_conformal_set": True, "is_ood": False},
                    {"hub_id": "h2", "hub_name": "Hub 2", "calibrated_confidence": 0.60,
                     "raw_similarity": 0.35, "in_conformal_set": True, "is_ood": False},
                ],
                "duplicates": [],
                "similar": [],
                "review": {"status": "accepted"},
            },
        ]
    return {
        "framework_id": "test_fw",
        "framework_name": "Test Framework",
        "version": "1.0",
        "fetched_date": "2026-05-01",
        "source_url": "https://example.com",
        "generated_at": "2026-05-01T00:00:00+00:00",
        "model_version": "deployment_round2",
        "context": "ingestion",
        "summary": {"total_controls": len(controls)},
        "controls": controls,
    }


@pytest.fixture
def accept_db(tmp_path):
    db_path = tmp_path / "accept_test.db"
    create_database(db_path)
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
        {"id": "h3", "name": "Hub 3", "path": "R > H3", "parent_id": None},
    ])
    return db_path


class TestAcceptReview:
    def test_accepted_control_inserts_framework_and_assignment(self, accept_db) -> None:
        review = _make_review_data()
        result = accept_review(accept_db, review)

        assert result["framework_inserted"] is True
        assert result["controls_inserted"] == 1
        assert result["assignments_created"] == 1
        assert result["rejected"] == 0
        assert result["pending"] == 0

        conn = get_connection(accept_db)
        try:
            fw = conn.execute("SELECT * FROM frameworks WHERE id = 'test_fw'").fetchone()
            assert fw is not None
            assert fw["name"] == "Test Framework"
            assert fw["control_count"] == 1

            ctrl = conn.execute("SELECT * FROM controls WHERE id = 'test_fw:TC-01'").fetchone()
            assert ctrl is not None
            assert ctrl["section_id"] == "TC-01"
            assert ctrl["title"] == "Access Control"
            assert ctrl["description"] == "Enforce access control for AI models"

            assignment = conn.execute(
                "SELECT * FROM assignments WHERE control_id = 'test_fw:TC-01'"
            ).fetchone()
            assert assignment is not None
            assert assignment["hub_id"] == "h1"
            assert assignment["confidence"] == 0.85
            assert assignment["review_status"] == "accepted"
            assert assignment["provenance"] == "ingest_test_fw"
        finally:
            conn.close()

    def test_rejected_control_no_assignment(self, accept_db) -> None:
        controls = [
            {
                "control_id": "TC-01",
                "title": "Rejected Control",
                "description": "Will be rejected",
                "full_text": None,
                "is_ood": False,
                "predictions": [
                    {"hub_id": "h1", "hub_name": "Hub 1", "calibrated_confidence": 0.50,
                     "raw_similarity": 0.30, "in_conformal_set": True, "is_ood": False},
                ],
                "duplicates": [],
                "similar": [],
                "review": {"status": "rejected"},
            },
        ]
        result = accept_review(accept_db, _make_review_data(controls))

        assert result["assignments_created"] == 0
        assert result["rejected"] == 1
        assert result["controls_inserted"] == 1

        conn = get_connection(accept_db)
        try:
            ctrl = conn.execute("SELECT * FROM controls WHERE id = 'test_fw:TC-01'").fetchone()
            assert ctrl is not None

            assignment = conn.execute(
                "SELECT * FROM assignments WHERE control_id = 'test_fw:TC-01'"
            ).fetchone()
            assert assignment is None
        finally:
            conn.close()

    def test_corrected_control_uses_corrected_hub(self, accept_db) -> None:
        controls = [
            {
                "control_id": "TC-01",
                "title": "Corrected Control",
                "description": "Expert corrected the hub",
                "full_text": None,
                "is_ood": False,
                "predictions": [
                    {"hub_id": "h1", "hub_name": "Hub 1", "calibrated_confidence": 0.60,
                     "raw_similarity": 0.35, "in_conformal_set": True, "is_ood": False},
                ],
                "duplicates": [],
                "similar": [],
                "review": {"status": "corrected", "corrected_hub_id": "h3"},
            },
        ]
        result = accept_review(accept_db, _make_review_data(controls))

        assert result["assignments_created"] == 1
        assert result["corrected"] == 1

        conn = get_connection(accept_db)
        try:
            assignment = conn.execute(
                "SELECT * FROM assignments WHERE control_id = 'test_fw:TC-01'"
            ).fetchone()
            assert assignment["hub_id"] == "h3"
            assert assignment["review_status"] == "corrected"
        finally:
            conn.close()

    def test_pending_control_skipped_with_warning(self, accept_db) -> None:
        controls = [
            {
                "control_id": "TC-01",
                "title": "Pending Control",
                "description": "Not yet reviewed",
                "full_text": None,
                "is_ood": False,
                "predictions": [
                    {"hub_id": "h1", "hub_name": "Hub 1", "calibrated_confidence": 0.80,
                     "raw_similarity": 0.40, "in_conformal_set": True, "is_ood": False},
                ],
                "duplicates": [],
                "similar": [],
                "review": {"status": "pending"},
            },
        ]
        result = accept_review(accept_db, _make_review_data(controls))

        assert result["pending"] == 1
        assert result["assignments_created"] == 0
        assert result["controls_inserted"] == 1

    def test_mixed_statuses(self, accept_db) -> None:
        controls = [
            {
                "control_id": "TC-01", "title": "Accepted", "description": "D1",
                "full_text": None, "is_ood": False,
                "predictions": [{"hub_id": "h1", "hub_name": "H1", "calibrated_confidence": 0.90,
                                 "raw_similarity": 0.45, "in_conformal_set": True, "is_ood": False}],
                "duplicates": [], "similar": [],
                "review": {"status": "accepted"},
            },
            {
                "control_id": "TC-02", "title": "Rejected", "description": "D2",
                "full_text": None, "is_ood": False,
                "predictions": [{"hub_id": "h2", "hub_name": "H2", "calibrated_confidence": 0.50,
                                 "raw_similarity": 0.30, "in_conformal_set": True, "is_ood": False}],
                "duplicates": [], "similar": [],
                "review": {"status": "rejected"},
            },
            {
                "control_id": "TC-03", "title": "Corrected", "description": "D3",
                "full_text": None, "is_ood": True,
                "predictions": [{"hub_id": "h1", "hub_name": "H1", "calibrated_confidence": 0.40,
                                 "raw_similarity": 0.25, "in_conformal_set": False, "is_ood": True}],
                "duplicates": [], "similar": [],
                "review": {"status": "corrected", "corrected_hub_id": "h3"},
            },
            {
                "control_id": "TC-04", "title": "Pending", "description": "D4",
                "full_text": None, "is_ood": False,
                "predictions": [{"hub_id": "h2", "hub_name": "H2", "calibrated_confidence": 0.70,
                                 "raw_similarity": 0.38, "in_conformal_set": True, "is_ood": False}],
                "duplicates": [], "similar": [],
                "review": {"status": "pending"},
            },
        ]
        result = accept_review(accept_db, _make_review_data(controls))

        assert result["controls_inserted"] == 4
        assert result["assignments_created"] == 2
        assert result["rejected"] == 1
        assert result["corrected"] == 1
        assert result["pending"] == 1

    def test_force_replaces_existing_framework(self, accept_db) -> None:
        review = _make_review_data()
        accept_review(accept_db, review)

        updated_controls = [
            {
                "control_id": "TC-01", "title": "Updated Control", "description": "New desc",
                "full_text": "Extended text", "is_ood": False,
                "predictions": [{"hub_id": "h2", "hub_name": "H2", "calibrated_confidence": 0.95,
                                 "raw_similarity": 0.48, "in_conformal_set": True, "is_ood": False}],
                "duplicates": [], "similar": [],
                "review": {"status": "accepted"},
            },
        ]
        result = accept_review(accept_db, _make_review_data(updated_controls), force=True)

        assert result["framework_inserted"] is True

        conn = get_connection(accept_db)
        try:
            ctrl = conn.execute("SELECT * FROM controls WHERE id = 'test_fw:TC-01'").fetchone()
            assert ctrl["title"] == "Updated Control"
            assert ctrl["description"] == "New desc"

            assignment = conn.execute(
                "SELECT * FROM assignments WHERE control_id = 'test_fw:TC-01'"
            ).fetchone()
            assert assignment["hub_id"] == "h2"
        finally:
            conn.close()

    def test_duplicate_framework_without_force_raises(self, accept_db) -> None:
        review = _make_review_data()
        accept_review(accept_db, review)

        with pytest.raises(ValueError, match="already exists"):
            accept_review(accept_db, review, force=False)

    def test_ood_flag_preserved(self, accept_db) -> None:
        controls = [
            {
                "control_id": "TC-01", "title": "OOD Control", "description": "Out of dist",
                "full_text": None, "is_ood": True,
                "predictions": [{"hub_id": "h1", "hub_name": "H1", "calibrated_confidence": 0.70,
                                 "raw_similarity": 0.38, "in_conformal_set": True, "is_ood": True}],
                "duplicates": [], "similar": [],
                "review": {"status": "accepted"},
            },
        ]
        accept_review(accept_db, _make_review_data(controls))

        conn = get_connection(accept_db)
        try:
            assignment = conn.execute(
                "SELECT * FROM assignments WHERE control_id = 'test_fw:TC-01'"
            ).fetchone()
            assert assignment["is_ood"] == 1
        finally:
            conn.close()

    def test_conformal_set_preserved(self, accept_db) -> None:
        controls = [
            {
                "control_id": "TC-01", "title": "Test", "description": "Conformal check",
                "full_text": None, "is_ood": False,
                "predictions": [{"hub_id": "h1", "hub_name": "H1", "calibrated_confidence": 0.80,
                                 "raw_similarity": 0.40, "in_conformal_set": False, "is_ood": False}],
                "duplicates": [], "similar": [],
                "review": {"status": "accepted"},
            },
        ]
        accept_review(accept_db, _make_review_data(controls))

        conn = get_connection(accept_db)
        try:
            assignment = conn.execute(
                "SELECT * FROM assignments WHERE control_id = 'test_fw:TC-01'"
            ).fetchone()
            assert assignment["in_conformal_set"] == 0
        finally:
            conn.close()

    def test_corrected_without_hub_id_raises(self, accept_db) -> None:
        controls = [
            {
                "control_id": "TC-01", "title": "Bad Correction", "description": "Missing hub",
                "full_text": None, "is_ood": False,
                "predictions": [{"hub_id": "h1", "hub_name": "H1", "calibrated_confidence": 0.80,
                                 "raw_similarity": 0.40, "in_conformal_set": True, "is_ood": False}],
                "duplicates": [], "similar": [],
                "review": {"status": "corrected"},
            },
        ]
        with pytest.raises(ValueError, match="corrected_hub_id"):
            accept_review(accept_db, _make_review_data(controls))
