"""Tests for pre-export CRE ID staleness check."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import insert_hubs


@pytest.fixture
def staleness_db(tmp_path):
    db_path = tmp_path / "staleness.db"
    create_database(db_path)
    insert_hubs(db_path, [
        {"id": "111-222", "name": "Hub A", "path": "R > A", "parent_id": None},
        {"id": "333-444", "name": "Hub B", "path": "R > B", "parent_id": None},
        {"id": "555-666", "name": "Hub C", "path": "R > C", "parent_id": None},
    ])
    return db_path


class TestStalenessCheck:
    def test_all_match_returns_pass(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness
        upstream_ids = {"111-222", "333-444", "555-666"}
        with patch("tract.export.staleness._fetch_upstream_cre_ids", return_value=upstream_ids):
            result = check_staleness(staleness_db)
        assert result["status"] == "pass"
        assert result["stale_ids"] == []

    def test_tract_has_extra_ids_returns_warn(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness
        upstream_ids = {"111-222", "333-444"}
        with patch("tract.export.staleness._fetch_upstream_cre_ids", return_value=upstream_ids):
            result = check_staleness(staleness_db)
        assert result["status"] == "warn"
        assert "555-666" in result["stale_ids"]

    def test_upstream_has_extra_ids_returns_pass(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness
        upstream_ids = {"111-222", "333-444", "555-666", "999-999"}
        with patch("tract.export.staleness._fetch_upstream_cre_ids", return_value=upstream_ids):
            result = check_staleness(staleness_db)
        assert result["status"] == "pass"
        assert result["upstream_only"] == ["999-999"]

    def test_network_error_returns_error(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness
        with patch("tract.export.staleness._fetch_upstream_cre_ids", side_effect=ConnectionError("timeout")):
            result = check_staleness(staleness_db)
        assert result["status"] == "error"
        assert "timeout" in result["message"]

    def test_get_tract_hub_ids(self, staleness_db) -> None:
        from tract.export.staleness import _get_tract_hub_ids
        ids = _get_tract_hub_ids(staleness_db)
        assert ids == {"111-222", "333-444", "555-666"}
