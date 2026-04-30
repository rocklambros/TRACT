"""Tests for crosswalk store CRUD operations."""
from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def db_path(tmp_path):
    from tract.crosswalk.schema import create_database
    path = tmp_path / "test.db"
    create_database(path)
    return path


class TestInsertHub:
    def test_insert_and_retrieve(self, db_path) -> None:
        from tract.crosswalk.store import insert_hubs, get_hub

        hubs = [{"id": "236-712", "name": "Access Control", "path": "Root > Access", "parent_id": None}]
        insert_hubs(db_path, hubs)

        hub = get_hub(db_path, "236-712")
        assert hub is not None
        assert hub["name"] == "Access Control"
        assert hub["path"] == "Root > Access"


class TestInsertFramework:
    def test_insert_and_count(self, db_path) -> None:
        from tract.crosswalk.store import insert_frameworks, count_frameworks

        fws = [{"id": "owasp_ai_exchange", "name": "OWASP AI Exchange", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 54}]
        insert_frameworks(db_path, fws)

        assert count_frameworks(db_path) == 1


class TestInsertControl:
    def test_insert_with_framework_fk(self, db_path) -> None:
        from tract.crosswalk.store import insert_frameworks, insert_controls, get_controls_by_framework

        insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
        insert_controls(db_path, [{"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "Control 1", "description": "Desc", "full_text": None}])

        controls = get_controls_by_framework(db_path, "fw1")
        assert len(controls) == 1
        assert controls[0]["title"] == "Control 1"


class TestInsertAssignment:
    def test_insert_and_query(self, db_path) -> None:
        from tract.crosswalk.store import (
            insert_assignments,
            insert_controls,
            insert_frameworks,
            insert_hubs,
            get_assignments_by_control,
        )

        insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
        insert_hubs(db_path, [{"id": "hub1", "name": "Hub 1", "path": "R > H1", "parent_id": None}])
        insert_controls(db_path, [{"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": None, "full_text": None}])

        assignments = [{
            "control_id": "fw1:c1",
            "hub_id": "hub1",
            "confidence": 0.85,
            "in_conformal_set": 1,
            "is_ood": 0,
            "provenance": "active_learning_round_1",
            "source_link_id": None,
            "model_version": "abc123",
            "review_status": "pending",
        }]
        insert_assignments(db_path, assignments)

        result = get_assignments_by_control(db_path, "fw1:c1")
        assert len(result) == 1
        assert result[0]["confidence"] == 0.85
        assert result[0]["provenance"] == "active_learning_round_1"


class TestUpdateReviewStatus:
    def test_update_status(self, db_path) -> None:
        from tract.crosswalk.store import (
            insert_assignments,
            insert_controls,
            insert_frameworks,
            insert_hubs,
            update_review_status,
            get_assignments_by_control,
        )

        insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
        insert_hubs(db_path, [{"id": "hub1", "name": "Hub 1", "path": "R", "parent_id": None}])
        insert_controls(db_path, [{"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": None, "full_text": None}])
        insert_assignments(db_path, [{"control_id": "fw1:c1", "hub_id": "hub1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0, "provenance": "al_r1", "source_link_id": None, "model_version": "v1", "review_status": "pending"}])

        assignments = get_assignments_by_control(db_path, "fw1:c1")
        update_review_status(db_path, assignments[0]["id"], "accepted", reviewer="expert1")

        updated = get_assignments_by_control(db_path, "fw1:c1")
        assert updated[0]["review_status"] == "accepted"
        assert updated[0]["reviewer"] == "expert1"


class TestTransactionAtomicity:
    def test_failed_insert_rolls_back(self, db_path) -> None:
        from tract.crosswalk.store import insert_hubs, count_hubs

        insert_hubs(db_path, [{"id": "h1", "name": "Hub 1", "path": "R", "parent_id": None}])

        with pytest.raises(sqlite3.IntegrityError):
            insert_hubs(db_path, [
                {"id": "h2", "name": "Hub 2", "path": "R", "parent_id": None},
                {"id": "h1", "name": "Duplicate", "path": "R", "parent_id": None},
            ])

        assert count_hubs(db_path) == 1
