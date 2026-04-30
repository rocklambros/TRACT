"""Tests for LLM-based hub naming."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tract.proposals.cluster import Cluster
from tract.proposals.guardrails import GuardrailResult


class TestGenerateHubNames:
    def test_returns_names_for_passing_clusters(self) -> None:
        from tract.proposals.naming import generate_hub_names

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
        result = GuardrailResult(
            cluster=cluster, passed=True, rejection_reasons=[],
            parent_hub_id="hub-1", parent_similarity=0.4,
            uncertain_placement=False, evidence_score=6.0,
        )

        mock_hierarchy = MagicMock()
        mock_hierarchy.hubs = {"hub-1": MagicMock(name="Access Control", hierarchy_path="Root > Access Control")}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="AI Model Governance")]
        mock_client.messages.create.return_value = mock_response

        with patch("tract.proposals.naming._get_anthropic_client", return_value=mock_client):
            names = generate_hub_names([result], mock_hierarchy, {})

        assert 0 in names
        assert isinstance(names[0], str)
        assert len(names[0]) > 0

    def test_placeholder_names_without_llm(self) -> None:
        from tract.proposals.naming import generate_placeholder_names

        rng = np.random.default_rng(42)
        centroid = rng.standard_normal(1024).astype(np.float32)
        centroid = centroid / np.linalg.norm(centroid)

        cluster = Cluster(
            cluster_id=0,
            control_ids=["fw_a::c1"],
            centroid=centroid,
            nearest_hub_id="hub-1",
            nearest_hub_similarity=0.4,
            member_frameworks={"fw_a"},
        )
        result = GuardrailResult(
            cluster=cluster, passed=True, rejection_reasons=[],
            parent_hub_id="hub-1", parent_similarity=0.4,
            uncertain_placement=False, evidence_score=2.0,
        )

        hub_names = {"hub-1": "Access Control"}
        names = generate_placeholder_names([result], hub_names)
        assert 0 in names
        assert "Access Control" in names[0]
