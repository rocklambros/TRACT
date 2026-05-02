"""Tests for tract.bridge.review — candidate validation and bridge commit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_hierarchy_data(hub_ids: list[str]) -> dict:
    """Build a minimal hierarchy dict for testing."""
    hubs = {}
    for hid in hub_ids:
        hubs[hid] = {
            "hub_id": hid, "name": f"Hub {hid}", "parent_id": None,
            "children_ids": [], "depth": 0, "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}", "is_leaf": True,
            "sibling_hub_ids": [], "related_hub_ids": [],
        }
    return {
        "hubs": hubs, "roots": sorted(hub_ids), "label_space": sorted(hub_ids),
        "fetch_timestamp": "2026-01-01T00:00:00", "data_hash": "test",
        "version": "1.1",
    }


def _make_candidates_data(candidates: list[dict]) -> dict:
    return {
        "generated_at": "2026-05-02T00:00:00",
        "method": "top_k_per_ai_hub",
        "top_k": 3,
        "similarity_stats": {"matrix_shape": [2, 3], "mean": 0.5},
        "candidates": candidates,
        "negative_controls": [],
        "unclassified_leaf_hubs": [],
    }


class TestValidateCandidates:

    def test_rejects_pending_status(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "pending", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert any("pending" in e.lower() for e in errors)

    def test_rejects_unknown_status(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "maybe", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert any("maybe" in e for e in errors)

    def test_rejects_nonexistent_hub_id(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "GHOST",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "accepted", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert any("GHOST" in e for e in errors)

    def test_accepts_all_reviewed(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "accepted", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert errors == []

    def test_accepts_rejected(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "rejected", "reviewer_notes": "weak"},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert errors == []


class TestCommitBridges:

    def test_accepted_creates_bidirectional_links(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1", "TRAD-2"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.7, "rank_for_ai_hub": 1,
             "description": "bridge", "status": "accepted", "reviewer_notes": ""},
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-2",
             "cosine_similarity": 0.3, "rank_for_ai_hub": 2,
             "description": "weak", "status": "rejected", "reviewer_notes": "too weak"},
        ])
        report = commit_bridges(candidates_data, hier_path, tmp_path / "report.json")

        updated = json.loads(hier_path.read_text())
        assert "TRAD-1" in updated["hubs"]["AI-1"]["related_hub_ids"]
        assert "AI-1" in updated["hubs"]["TRAD-1"]["related_hub_ids"]
        assert "TRAD-2" not in updated["hubs"]["AI-1"]["related_hub_ids"]

    def test_zero_accepted_is_valid(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.3, "rank_for_ai_hub": 1,
             "description": "weak", "status": "rejected", "reviewer_notes": "no bridge"},
        ])
        report = commit_bridges(candidates_data, hier_path, tmp_path / "report.json")
        assert report["counts"]["accepted"] == 0

    def test_report_json_written(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))
        report_path = tmp_path / "bridge_report.json"

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.7, "rank_for_ai_hub": 1,
             "description": "yes", "status": "accepted", "reviewer_notes": ""},
        ])
        commit_bridges(candidates_data, hier_path, report_path)
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["counts"]["accepted"] == 1
        assert report["counts"]["rejected"] == 0

    def test_hierarchy_validates_after_commit(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        from tract.hierarchy import CREHierarchy
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.7, "rank_for_ai_hub": 1,
             "description": "yes", "status": "accepted", "reviewer_notes": ""},
        ])
        commit_bridges(candidates_data, hier_path, tmp_path / "report.json")
        hier = CREHierarchy.load(hier_path)
        assert "TRAD-1" in hier.hubs["AI-1"].related_hub_ids
