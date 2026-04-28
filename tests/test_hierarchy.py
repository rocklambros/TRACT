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


class TestCREHierarchyValidation:

    def test_validate_passes_on_good_data(self, hierarchy) -> None:
        hierarchy.validate_integrity()

    def test_detects_dangling_parent(self, mini_cres_data: dict) -> None:
        from tract.hierarchy import CREHierarchy
        cres = mini_cres_data["cres"]
        # Inject a hub with a bogus parent
        cres.append({
            "doctype": "CRE", "id": "BAD-HUB", "name": "Bad Hub",
            "tags": [], "links": [],
        })
        h = CREHierarchy.from_opencre(cres, "2026-01-01T00:00:00Z", "test")
        # Now create a corrupted version
        bad_node = h.hubs["BAD-HUB"].model_copy(update={"parent_id": "NONEXISTENT"})
        bad_hubs = dict(h.hubs)
        bad_hubs["BAD-HUB"] = bad_node
        bad_h = h.model_copy(update={"hubs": bad_hubs})
        with pytest.raises(ValueError, match="dangling parent_id"):
            bad_h.validate_integrity()

    def test_detects_unsorted_label_space(self, hierarchy) -> None:
        reversed_ls = list(reversed(hierarchy.label_space))
        bad_h = hierarchy.model_copy(update={"label_space": reversed_ls})
        with pytest.raises(ValueError, match="not sorted"):
            bad_h.validate_integrity()


class TestCREHierarchyQueries:

    def test_leaf_hub_ids_returns_label_space(self, hierarchy) -> None:
        assert hierarchy.leaf_hub_ids() == list(hierarchy.label_space)

    def test_get_parent(self, hierarchy) -> None:
        parent = hierarchy.get_parent("LEAF-A1a")
        assert parent is not None
        assert parent.hub_id == "PAR-A1"

    def test_get_parent_of_root(self, hierarchy) -> None:
        assert hierarchy.get_parent("ROOT-A") is None

    def test_get_children(self, hierarchy) -> None:
        children = hierarchy.get_children("PAR-A1")
        child_ids = {c.hub_id for c in children}
        assert child_ids == {"LEAF-A1a", "LEAF-A1b"}

    def test_get_children_of_leaf(self, hierarchy) -> None:
        assert hierarchy.get_children("LEAF-A1a") == []

    def test_get_siblings(self, hierarchy) -> None:
        siblings = hierarchy.get_siblings("LEAF-A1a")
        sib_ids = {s.hub_id for s in siblings}
        assert sib_ids == {"LEAF-A1b"}

    def test_get_branch_hub_ids(self, hierarchy) -> None:
        branch = hierarchy.get_branch_hub_ids("ROOT-A")
        assert "ROOT-A" in branch
        assert "LEAF-A1a" in branch
        assert "LEAF-B1" not in branch

    def test_get_hierarchy_path(self, hierarchy) -> None:
        assert hierarchy.get_hierarchy_path("LEAF-A1a") == "Root A > Parent A1 > Leaf A1a"

    def test_hub_by_name(self, hierarchy) -> None:
        node = hierarchy.hub_by_name("leaf a1a")
        assert node is not None
        assert node.hub_id == "LEAF-A1a"

    def test_hub_by_name_not_found(self, hierarchy) -> None:
        assert hierarchy.hub_by_name("nonexistent") is None
