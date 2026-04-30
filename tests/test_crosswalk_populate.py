"""Tests for crosswalk DB population helpers."""
from __future__ import annotations

from tract.hierarchy import CREHierarchy, HubNode
from tract.training.data_quality import TieredLink, QualityTier


def _make_hierarchy(hubs: dict[str, HubNode]) -> CREHierarchy:
    """Build a minimal CREHierarchy for testing."""
    return CREHierarchy(
        hubs=hubs,
        roots=["root1"],
        label_space=list(hubs.keys()),
        fetch_timestamp="2026-04-30T00:00:00Z",
        data_hash="testhash",
    )


class TestBuildHubRecords:
    def test_converts_hierarchy_to_hub_dicts(self) -> None:
        from tract.crosswalk.populate import build_hub_records

        hierarchy = _make_hierarchy({
            "h1": HubNode(hub_id="h1", name="Hub One", parent_id="root1", depth=1, branch_root_id="root1", hierarchy_path="/Root/Hub One", is_leaf=True),
            "h2": HubNode(hub_id="h2", name="Hub Two", parent_id="root1", depth=1, branch_root_id="root1", hierarchy_path="/Root/Hub Two", is_leaf=True),
        })
        records = build_hub_records(hierarchy)
        assert len(records) == 2
        assert records[0]["id"] == "h1"
        assert records[0]["name"] == "Hub One"
        assert records[0]["path"] == "/Root/Hub One"
        assert records[0]["parent_id"] == "root1"


class TestBuildFrameworkRecords:
    def test_converts_framework_metadata(self) -> None:
        from tract.crosswalk.populate import build_framework_records

        fw_data = [
            {
                "framework_id": "csa_aicm",
                "framework_name": "CSA AI Controls Matrix",
                "version": "1.0",
                "fetched_date": "2026-04-28",
                "controls": [{"control_id": "A01"}, {"control_id": "A02"}],
            }
        ]
        records = build_framework_records(fw_data)
        assert len(records) == 1
        assert records[0]["id"] == "csa_aicm"
        assert records[0]["control_count"] == 2


class TestBuildControlRecords:
    def test_converts_controls_with_framework_prefix(self) -> None:
        from tract.crosswalk.populate import build_control_records

        fw_data = [
            {
                "framework_id": "csa_aicm",
                "framework_name": "CSA AI Controls Matrix",
                "controls": [
                    {"control_id": "A01", "title": "Audit", "description": "Audit desc", "full_text": "Full text"},
                ],
            }
        ]
        records = build_control_records(fw_data)
        assert len(records) == 1
        assert records[0]["id"] == "csa_aicm:A01"
        assert records[0]["framework_id"] == "csa_aicm"
        assert records[0]["section_id"] == "A01"


class TestBuildTrainingAssignments:
    def test_converts_tiered_links(self) -> None:
        from tract.crosswalk.populate import build_training_assignments

        links = [
            TieredLink(
                link={"cre_id": "h1", "standard_name": "ASVS", "section_name": "s1", "section_id": "4.1.1", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            ),
        ]
        control_id_map = {("ASVS", "4.1.1"): "asvs:4.1.1"}
        records = build_training_assignments(links, control_id_map)
        assert len(records) == 1
        assert records[0]["control_id"] == "asvs:4.1.1"
        assert records[0]["hub_id"] == "h1"
        assert records[0]["provenance"] == "training_T1"
        assert records[0]["review_status"] == "ground_truth"

    def test_skips_unmapped_links(self) -> None:
        from tract.crosswalk.populate import build_training_assignments

        links = [
            TieredLink(
                link={"cre_id": "h1", "standard_name": "Unknown", "section_name": "s1", "section_id": "x", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            ),
        ]
        records = build_training_assignments(links, {})
        assert len(records) == 0
