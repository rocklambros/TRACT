"""Tests for reviewer guide and hub reference generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.crosswalk.schema import create_database, get_connection
from tract.review.guide import generate_hub_reference, generate_reviewer_guide


@pytest.fixture()
def guide_db(tmp_path: Path) -> Path:
    """Create a test DB with hub hierarchy for guide/reference tests."""
    db_path = tmp_path / "guide.db"
    create_database(db_path)
    conn = get_connection(db_path)
    try:
        conn.executemany(
            "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
            [
                ("root-1", "Security", "Security", None),
                ("child-1", "Authentication", "Security/Authentication", "root-1"),
                ("child-2", "Cryptography", "Security/Cryptography", "root-1"),
                ("leaf-1", "MFA", "Security/Authentication/MFA", "child-1"),
                ("leaf-2", "Key Management", "Security/Cryptography/Key Management", "child-2"),
                ("root-2", "Operations", "Operations", None),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture()
def sample_metadata() -> dict:
    return {
        "generated_at": "2026-05-03T12:00:00+00:00",
        "model_version": "abc123def456",
        "total_predictions": 150,
        "calibration_items": 20,
        "framework_breakdown": {"nist_csf": 50, "iso_27001": 100},
        "priority_breakdown": {"critical": 10, "careful": 40, "routine": 100},
    }


class TestGenerateReviewerGuide:
    def test_guide_file_created(
        self, tmp_path: Path, sample_metadata: dict,
    ) -> None:
        result = generate_reviewer_guide(tmp_path / "output", sample_metadata)
        assert result.exists()
        assert result.name == "reviewer_guide.md"

    def test_guide_contains_key_sections(
        self, tmp_path: Path, sample_metadata: dict,
    ) -> None:
        result = generate_reviewer_guide(tmp_path / "output", sample_metadata)
        content = result.read_text(encoding="utf-8")
        assert "## Role" in content
        assert "## Step-by-Step Process" in content
        assert "## Decision Criteria" in content
        assert "## Common Pitfalls" in content
        assert "## Editor Requirement" in content
        assert "## Saving Progress" in content
        assert "## Time Estimate" in content

    def test_guide_contains_role_persona(
        self, tmp_path: Path, sample_metadata: dict,
    ) -> None:
        result = generate_reviewer_guide(tmp_path / "output", sample_metadata)
        content = result.read_text(encoding="utf-8")
        assert "cybersecurity domain expert" in content
        assert "peer-reviewed research dataset" in content

    def test_guide_uses_metadata_total(
        self, tmp_path: Path, sample_metadata: dict,
    ) -> None:
        result = generate_reviewer_guide(tmp_path / "output", sample_metadata)
        content = result.read_text(encoding="utf-8")
        assert "150 predictions" in content

    def test_guide_uses_priority_breakdown(
        self, tmp_path: Path, sample_metadata: dict,
    ) -> None:
        result = generate_reviewer_guide(tmp_path / "output", sample_metadata)
        content = result.read_text(encoding="utf-8")
        assert "100 routine" in content
        assert "40 careful" in content
        assert "10 critical" in content

    def test_guide_contains_all_nine_steps(
        self, tmp_path: Path, sample_metadata: dict,
    ) -> None:
        result = generate_reviewer_guide(tmp_path / "output", sample_metadata)
        content = result.read_text(encoding="utf-8")
        assert "review_predictions.json" in content
        assert "control_text" in content
        assert '"status": "accepted"' in content
        assert '"status": "reassigned"' in content
        assert '"status": "rejected"' in content

    def test_guide_contains_all_five_pitfalls(
        self, tmp_path: Path, sample_metadata: dict,
    ) -> None:
        result = generate_reviewer_guide(tmp_path / "output", sample_metadata)
        content = result.read_text(encoding="utf-8")
        assert "MITRE ATLAS" in content
        assert "alternative_hubs" in content
        assert "is_ood" in content
        assert "NIST AI RMF" in content
        assert "text_quality" in content


class TestGenerateHubReference:
    def test_hub_reference_file_created(
        self, tmp_path: Path, guide_db: Path,
    ) -> None:
        result = generate_hub_reference(guide_db, tmp_path / "output")
        assert result.exists()
        assert result.name == "hub_reference.json"

    def test_hub_reference_contains_all_hubs(
        self, tmp_path: Path, guide_db: Path,
    ) -> None:
        result = generate_hub_reference(guide_db, tmp_path / "output")
        data = json.loads(result.read_text(encoding="utf-8"))
        assert len(data) == 6

    def test_hub_reference_sorted_by_path(
        self, tmp_path: Path, guide_db: Path,
    ) -> None:
        result = generate_hub_reference(guide_db, tmp_path / "output")
        data = json.loads(result.read_text(encoding="utf-8"))
        paths = [h["path"] for h in data]
        assert paths == sorted(paths)

    def test_hub_reference_has_required_fields(
        self, tmp_path: Path, guide_db: Path,
    ) -> None:
        result = generate_hub_reference(guide_db, tmp_path / "output")
        data = json.loads(result.read_text(encoding="utf-8"))
        required = {"hub_id", "name", "path", "parent_id", "is_leaf"}
        for hub in data:
            assert set(hub.keys()) == required

    def test_is_leaf_correctly_computed(
        self, tmp_path: Path, guide_db: Path,
    ) -> None:
        result = generate_hub_reference(guide_db, tmp_path / "output")
        data = json.loads(result.read_text(encoding="utf-8"))
        by_id = {h["hub_id"]: h for h in data}

        assert by_id["root-1"]["is_leaf"] is False
        assert by_id["child-1"]["is_leaf"] is False
        assert by_id["child-2"]["is_leaf"] is False
        assert by_id["leaf-1"]["is_leaf"] is True
        assert by_id["leaf-2"]["is_leaf"] is True
        assert by_id["root-2"]["is_leaf"] is True

    def test_parent_id_preserved(
        self, tmp_path: Path, guide_db: Path,
    ) -> None:
        result = generate_hub_reference(guide_db, tmp_path / "output")
        data = json.loads(result.read_text(encoding="utf-8"))
        by_id = {h["hub_id"]: h for h in data}

        assert by_id["root-1"]["parent_id"] is None
        assert by_id["child-1"]["parent_id"] == "root-1"
        assert by_id["leaf-1"]["parent_id"] == "child-1"
