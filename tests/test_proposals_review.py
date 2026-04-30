"""Tests for proposal writing and review session."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tract.proposals.cluster import Cluster
from tract.proposals.guardrails import GuardrailResult


@pytest.fixture
def passing_result() -> GuardrailResult:
    rng = np.random.default_rng(42)
    centroid = rng.standard_normal(1024).astype(np.float32)
    centroid = centroid / np.linalg.norm(centroid)
    cluster = Cluster(
        cluster_id=0,
        control_ids=["fw_a::c1", "fw_a::c2", "fw_b::c3"],
        centroid=centroid,
        nearest_hub_id="hub-1",
        nearest_hub_similarity=0.4,
        member_frameworks={"fw_a", "fw_b"},
    )
    return GuardrailResult(
        cluster=cluster, passed=True, rejection_reasons=[],
        parent_hub_id="hub-1", parent_similarity=0.4,
        uncertain_placement=False, evidence_score=6.0,
    )


class TestWriteProposalRound:
    def test_creates_round_directory(self, tmp_path: Path, passing_result: GuardrailResult) -> None:
        from tract.proposals.review import write_proposal_round

        output_dir = tmp_path / "hub_proposals"
        path = write_proposal_round(
            results=[passing_result],
            names={0: "AI Model Governance"},
            output_dir=output_dir,
            round_num=1,
        )
        assert path.exists()
        assert (path / "proposals.json").exists()

    def test_proposal_json_schema(self, tmp_path: Path, passing_result: GuardrailResult) -> None:
        from tract.proposals.review import write_proposal_round

        output_dir = tmp_path / "hub_proposals"
        path = write_proposal_round(
            results=[passing_result],
            names={0: "AI Model Governance"},
            output_dir=output_dir,
            round_num=1,
        )

        data = json.loads((path / "proposals.json").read_text())
        assert "round" in data
        assert "proposals" in data
        assert data["round"] == 1
        assert len(data["proposals"]) == 1

        proposal = data["proposals"][0]
        assert "cluster_id" in proposal
        assert "proposed_name" in proposal
        assert "control_ids" in proposal
        assert "parent_hub_id" in proposal
        assert "evidence_score" in proposal

    def test_only_passing_written(self, tmp_path: Path, passing_result: GuardrailResult) -> None:
        from tract.proposals.review import write_proposal_round

        failing = GuardrailResult(
            cluster=passing_result.cluster, passed=False,
            rejection_reasons=["test"], parent_hub_id=None,
            parent_similarity=0.0, uncertain_placement=False, evidence_score=0.0,
        )

        output_dir = tmp_path / "hub_proposals"
        path = write_proposal_round(
            results=[passing_result, failing],
            names={0: "Test"},
            output_dir=output_dir,
            round_num=1,
        )

        data = json.loads((path / "proposals.json").read_text())
        assert len(data["proposals"]) == 1
