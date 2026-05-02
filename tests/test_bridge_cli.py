"""Tests for bridge CLI command and orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _make_mini_artifacts(tmp_path: Path, hub_ids: list[str]) -> Path:
    """Create a minimal deployment_artifacts.npz."""
    n = len(hub_ids)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, 1024)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    path = tmp_path / "deployment_artifacts.npz"
    np.savez(
        str(path),
        hub_embeddings=emb,
        control_embeddings=np.zeros((10, 1024), dtype=np.float32),
        hub_ids=np.array(hub_ids),
        control_ids=np.array([f"ctrl-{i}" for i in range(10)]),
    )
    return path


def _make_mini_hierarchy(tmp_path: Path, hub_ids: list[str]) -> Path:
    """Create a minimal cre_hierarchy.json."""
    hubs = {}
    for hid in hub_ids:
        hubs[hid] = {
            "hub_id": hid, "name": f"Hub {hid}", "parent_id": None,
            "children_ids": [], "depth": 0, "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}", "is_leaf": True,
            "sibling_hub_ids": [], "related_hub_ids": [],
        }
    data = {
        "hubs": hubs, "roots": sorted(hub_ids), "label_space": sorted(hub_ids),
        "fetch_timestamp": "2026-01-01T00:00:00", "data_hash": "test",
        "version": "1.1",
    }
    path = tmp_path / "cre_hierarchy.json"
    path.write_text(json.dumps(data, sort_keys=True, indent=2))
    return path


FIXTURE_LINKS = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"
ALL_HUB_IDS = ["AI-1", "AI-2", "AI-3", "BOTH-1", "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5", "UNLINKED-1"]


class TestRunBridgeAnalysis:

    def test_generates_candidates_file(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        candidates_path = output_dir / "bridge_candidates.json"
        assert candidates_path.exists()
        data = json.loads(candidates_path.read_text())
        assert len(data["candidates"]) == 6  # 3 AI-only × top-2

    def test_candidates_all_pending(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        data = json.loads((output_dir / "bridge_candidates.json").read_text())
        for c in data["candidates"]:
            assert c["status"] == "pending"

    def test_includes_negative_controls(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        data = json.loads((output_dir / "bridge_candidates.json").read_text())
        assert len(data["negative_controls"]) == 3  # 1 per AI-only hub

    def test_includes_similarity_stats(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        data = json.loads((output_dir / "bridge_candidates.json").read_text())
        assert "matrix_shape" in data["similarity_stats"]


class TestBridgeCLIParsing:

    def test_bridge_subcommand_exists(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["bridge", "--skip-descriptions"])
        assert args.command == "bridge"
        assert args.skip_descriptions is True

    def test_bridge_top_k_default(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["bridge"])
        assert args.top_k == 3

    def test_bridge_commit_mode(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["bridge", "--commit", "--candidates", "cands.json"])
        assert args.commit is True
        assert args.candidates == "cands.json"
