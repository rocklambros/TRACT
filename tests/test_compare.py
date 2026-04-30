"""Tests for tract.compare module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    """Create a crosswalk DB with two frameworks sharing a hub."""
    from tract.crosswalk.schema import create_database
    from tract.crosswalk.store import (
        insert_assignments,
        insert_controls,
        insert_frameworks,
        insert_hubs,
    )

    db_path = tmp_path / "test.db"
    create_database(db_path)

    insert_hubs(db_path, [
        {"id": "hub-1", "name": "Access Control", "path": "Root > Access Control", "parent_id": None},
        {"id": "hub-2", "name": "Input Validation", "path": "Root > Input Validation", "parent_id": None},
        {"id": "hub-3", "name": "Encryption", "path": "Root > Encryption", "parent_id": None},
    ])
    insert_frameworks(db_path, [
        {"id": "fw_a", "name": "Framework A", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2},
        {"id": "fw_b", "name": "Framework B", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2},
    ])
    insert_controls(db_path, [
        {"id": "fw_a::c1", "framework_id": "fw_a", "section_id": "c1", "title": "AC Policy", "description": "Access control", "full_text": None},
        {"id": "fw_a::c2", "framework_id": "fw_a", "section_id": "c2", "title": "Input Check", "description": "Validate input", "full_text": None},
        {"id": "fw_b::c1", "framework_id": "fw_b", "section_id": "c1", "title": "Auth", "description": "Authentication", "full_text": None},
        {"id": "fw_b::c2", "framework_id": "fw_b", "section_id": "c2", "title": "Encrypt", "description": "Encryption", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw_a::c1", "hub_id": "hub-1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0, "provenance": "ground_truth", "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw_b::c1", "hub_id": "hub-1", "confidence": 0.85, "in_conformal_set": 1, "is_ood": 0, "provenance": "ground_truth", "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw_a::c2", "hub_id": "hub-2", "confidence": 0.7, "in_conformal_set": 1, "is_ood": 0, "provenance": "model", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw_b::c2", "hub_id": "hub-3", "confidence": 0.8, "in_conformal_set": 1, "is_ood": 0, "provenance": "model", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
    ])
    return db_path


@pytest.fixture
def mock_hierarchy() -> object:
    """Minimal hierarchy with get_parent support."""
    from unittest.mock import MagicMock

    hierarchy = MagicMock()

    hub_nodes = {
        "hub-1": MagicMock(hub_id="hub-1", name="Access Control", parent_id=None, hierarchy_path="Root > Access Control"),
        "hub-2": MagicMock(hub_id="hub-2", name="Input Validation", parent_id=None, hierarchy_path="Root > Input Validation"),
        "hub-3": MagicMock(hub_id="hub-3", name="Encryption", parent_id=None, hierarchy_path="Root > Encryption"),
    }
    hierarchy.hubs = hub_nodes
    hierarchy.get_parent.return_value = None
    return hierarchy


class TestCrossFrameworkMatrix:
    def test_finds_equivalences(self, populated_db: Path, mock_hierarchy: object) -> None:
        from tract.compare import cross_framework_matrix

        result = cross_framework_matrix(populated_db, ["fw_a", "fw_b"], mock_hierarchy)
        assert len(result.equivalences) == 1
        assert result.equivalences[0].hub_id == "hub-1"
        assert result.total_shared_hubs == 1

    def test_gap_controls(self, populated_db: Path, mock_hierarchy: object) -> None:
        from tract.compare import cross_framework_matrix

        result = cross_framework_matrix(populated_db, ["fw_a", "fw_b"], mock_hierarchy)
        assert "fw_a" in result.gap_controls or "fw_b" in result.gap_controls

    def test_framework_pair_overlap(self, populated_db: Path, mock_hierarchy: object) -> None:
        from tract.compare import cross_framework_matrix

        result = cross_framework_matrix(populated_db, ["fw_a", "fw_b"], mock_hierarchy)
        assert ("fw_a", "fw_b") in result.framework_pair_overlap or ("fw_b", "fw_a") in result.framework_pair_overlap
