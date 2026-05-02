"""Tests for tract.bridge.describe — LLM bridge description generation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"


@pytest.fixture
def hub_links():
    from tract.io import load_json
    return load_json(FIXTURE_PATH)


@pytest.fixture
def mini_hierarchy():
    from tract.hierarchy import HubNode, CREHierarchy
    hubs = {}
    for hid, name in [
        ("AI-1", "AI Hub 1"), ("AI-2", "AI Hub 2"), ("AI-3", "AI Hub 3"),
        ("TRAD-1", "Trad Hub 1"), ("TRAD-2", "Trad Hub 2"),
        ("BOTH-1", "Both Hub 1"),
    ]:
        hubs[hid] = HubNode(
            hub_id=hid, name=name, depth=0, branch_root_id=hid,
            hierarchy_path=name, is_leaf=True,
        )
    return CREHierarchy(
        hubs=hubs, roots=sorted(hubs), label_space=sorted(hubs),
        fetch_timestamp="2026-01-01T00:00:00", data_hash="test",
    )


class TestGenerateBridgeDescriptions:

    def test_adds_description_field(self, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Both hubs address access control concerns.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        with patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client):
            candidates = [
                {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
                 "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
                 "cosine_similarity": 0.5, "rank_for_ai_hub": 1},
            ]
            generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
            assert candidates[0]["description"] != ""

    def test_sanitizes_description(self, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="  Has  <b>tags</b>  and   spaces  ")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        with patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client):
            candidates = [
                {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
                 "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
                 "cosine_similarity": 0.5, "rank_for_ai_hub": 1},
            ]
            generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
            assert "<b>" not in candidates[0]["description"]
            assert "  " not in candidates[0]["description"]

    def test_api_failure_sets_empty(self, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        with patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client):
            candidates = [
                {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
                 "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
                 "cosine_similarity": 0.5, "rank_for_ai_hub": 1},
            ]
            generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
            assert candidates[0]["description"] == ""

    def test_skips_existing_descriptions(self, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_client = MagicMock()

        with patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client):
            candidates = [
                {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
                 "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
                 "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
                 "description": "Already has one"},
            ]
            generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
            assert candidates[0]["description"] == "Already has one"
            mock_client.messages.create.assert_not_called()


class TestGenerateNegativeDescriptions:

    def test_returns_one_per_ai_hub(self, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_negative_descriptions
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="These hubs are unrelated.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp

        with patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client):
            negatives_input = [
                {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-2", "cosine_similarity": 0.01, "is_negative": True},
            ]
            result = generate_negative_descriptions(negatives_input, mini_hierarchy, hub_links)
            assert len(result) == 1
            assert result[0]["is_negative"] is True
            assert result[0]["description"] != ""


class TestCountControlsForHub:

    def test_counts_across_frameworks(self, hub_links) -> None:
        from tract.bridge.describe import count_controls_for_hub
        count = count_controls_for_hub("TRAD-1", hub_links)
        assert count == 2  # asvs + cwe

    def test_zero_for_unknown_hub(self, hub_links) -> None:
        from tract.bridge.describe import count_controls_for_hub
        assert count_controls_for_hub("NONEXISTENT", hub_links) == 0
