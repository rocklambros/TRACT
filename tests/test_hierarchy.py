"""Tests for tract.hierarchy — CRE hierarchy tree model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase1a_mini_cres.json"


@pytest.fixture
def mini_cres_data() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def hierarchy(mini_cres_data: dict):
    from tract.hierarchy import CREHierarchy
    return CREHierarchy.from_opencre(
        cres=mini_cres_data["cres"],
        fetch_timestamp=mini_cres_data["fetch_timestamp"],
        data_hash="abc123",
    )


class TestCREHierarchyConstruction:

    def test_hub_count(self, hierarchy) -> None:
        assert len(hierarchy.hubs) == 10

    def test_roots_sorted(self, hierarchy) -> None:
        assert hierarchy.roots == sorted(hierarchy.roots)
        assert set(hierarchy.roots) == {"ORPHAN-D", "ROOT-A", "ROOT-B", "ROOT-C"}

    def test_label_space_contains_leaves_only(self, hierarchy) -> None:
        for hub_id in hierarchy.label_space:
            assert hierarchy.hubs[hub_id].is_leaf

    def test_label_space_sorted(self, hierarchy) -> None:
        assert hierarchy.label_space == sorted(hierarchy.label_space)

    def test_label_space_count(self, hierarchy) -> None:
        # LEAF-A1a, LEAF-A1b, PAR-A2, LEAF-B1, LEAF-B2, ROOT-C, ORPHAN-D = 7 leaves
        assert len(hierarchy.label_space) == 7

    def test_depth_root(self, hierarchy) -> None:
        assert hierarchy.hubs["ROOT-A"].depth == 0

    def test_depth_parent(self, hierarchy) -> None:
        assert hierarchy.hubs["PAR-A1"].depth == 1

    def test_depth_leaf(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].depth == 2

    def test_parent_id(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].parent_id == "PAR-A1"
        assert hierarchy.hubs["PAR-A1"].parent_id == "ROOT-A"
        assert hierarchy.hubs["ROOT-A"].parent_id is None

    def test_children_ids(self, hierarchy) -> None:
        assert set(hierarchy.hubs["PAR-A1"].children_ids) == {"LEAF-A1a", "LEAF-A1b"}
        assert hierarchy.hubs["LEAF-A1a"].children_ids == []

    def test_hierarchy_path(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].hierarchy_path == "Root A > Parent A1 > Leaf A1a"
        assert hierarchy.hubs["ROOT-A"].hierarchy_path == "Root A"

    def test_branch_root_id(self, hierarchy) -> None:
        assert hierarchy.hubs["LEAF-A1a"].branch_root_id == "ROOT-A"
        assert hierarchy.hubs["ROOT-A"].branch_root_id == "ROOT-A"

    def test_orphan_is_leaf_and_root(self, hierarchy) -> None:
        orphan = hierarchy.hubs["ORPHAN-D"]
        assert orphan.is_leaf
        assert orphan.parent_id is None
        assert orphan.children_ids == []
        assert orphan.depth == 0
        assert orphan.branch_root_id == "ORPHAN-D"
        assert "ORPHAN-D" in hierarchy.label_space

    def test_orphan_root_is_leaf(self, hierarchy) -> None:
        rootc = hierarchy.hubs["ROOT-C"]
        assert rootc.is_leaf
        assert rootc.parent_id is None
        assert "ROOT-C" in hierarchy.label_space

    def test_sibling_hub_ids(self, hierarchy) -> None:
        node = hierarchy.hubs["LEAF-A1a"]
        assert "LEAF-A1b" in node.sibling_hub_ids
        assert "LEAF-A1a" not in node.sibling_hub_ids

    def test_version_and_metadata(self, hierarchy) -> None:
        assert hierarchy.version == "1.0"
        assert hierarchy.data_hash == "abc123"
        assert hierarchy.fetch_timestamp == "2026-04-28T12:00:00Z"
