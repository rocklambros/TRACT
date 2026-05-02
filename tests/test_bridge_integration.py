"""End-to-end bridge analysis integration test with synthetic data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_LINKS = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"
ALL_HUB_IDS = [
    "AI-1", "AI-2", "AI-3", "BOTH-1",
    "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5",
    "UNLINKED-1",
]


@pytest.fixture
def bridge_workspace(tmp_path):
    """Set up a complete bridge analysis workspace."""
    n = len(ALL_HUB_IDS)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, 1024)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    artifacts_path = tmp_path / "artifacts.npz"
    np.savez(
        str(artifacts_path),
        hub_embeddings=emb,
        control_embeddings=np.zeros((5, 1024), dtype=np.float32),
        hub_ids=np.array(ALL_HUB_IDS),
        control_ids=np.array([f"c-{i}" for i in range(5)]),
    )

    hubs = {}
    for hid in ALL_HUB_IDS:
        hubs[hid] = {
            "hub_id": hid, "name": f"Hub {hid}", "parent_id": None,
            "children_ids": [], "depth": 0, "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}", "is_leaf": True,
            "sibling_hub_ids": [], "related_hub_ids": [],
        }
    hier_data = {
        "hubs": hubs, "roots": sorted(ALL_HUB_IDS),
        "label_space": sorted(ALL_HUB_IDS),
        "fetch_timestamp": "2026-01-01T00:00:00", "data_hash": "test",
        "version": "1.1",
    }
    hier_path = tmp_path / "cre_hierarchy.json"
    hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

    return {
        "artifacts_path": artifacts_path,
        "hierarchy_path": hier_path,
        "tmp_path": tmp_path,
    }


class TestBridgeEndToEnd:

    def test_full_pipeline(self, bridge_workspace) -> None:
        from tract.bridge import run_bridge_analysis
        from tract.bridge.review import commit_bridges
        from tract.io import load_json

        ws = bridge_workspace
        output_dir = ws["tmp_path"] / "output"

        candidates_path = run_bridge_analysis(
            artifacts_path=ws["artifacts_path"],
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=ws["hierarchy_path"],
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )

        data = load_json(candidates_path)
        assert len(data["candidates"]) == 6  # 3 AI × top-2
        assert len(data["negative_controls"]) == 3  # 1 per AI hub

        data["candidates"][0]["status"] = "accepted"
        for c in data["candidates"][1:]:
            c["status"] = "rejected"

        candidates_path.write_text(json.dumps(data, sort_keys=True, indent=2))

        report_path = output_dir / "bridge_report.json"
        report = commit_bridges(data, ws["hierarchy_path"], report_path)

        assert report["counts"]["accepted"] == 1
        assert report["counts"]["rejected"] == 5
        assert report_path.exists()

        from tract.hierarchy import CREHierarchy
        hier = CREHierarchy.load(ws["hierarchy_path"])
        accepted = data["candidates"][0]
        ai_id = accepted["ai_hub_id"]
        trad_id = accepted["trad_hub_id"]
        assert trad_id in hier.hubs[ai_id].related_hub_ids
        assert ai_id in hier.hubs[trad_id].related_hub_ids
