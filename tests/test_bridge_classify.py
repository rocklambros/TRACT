"""Tests for tract.bridge.classify — hub classification by framework type."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"

ALL_HUB_IDS = [
    "AI-1", "AI-2", "AI-3",
    "BOTH-1",
    "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5",
    "UNLINKED-1", "UNLINKED-2",
]


@pytest.fixture
def classification():
    from tract.bridge.classify import classify_hubs
    return classify_hubs(FIXTURE_PATH, ALL_HUB_IDS)


class TestClassifyHubs:

    def test_ai_only_count(self, classification) -> None:
        assert len(classification.ai_only) == 3

    def test_ai_only_ids(self, classification) -> None:
        assert set(classification.ai_only) == {"AI-1", "AI-2", "AI-3"}

    def test_trad_only_count(self, classification) -> None:
        assert len(classification.trad_only) == 5

    def test_trad_only_ids(self, classification) -> None:
        assert set(classification.trad_only) == {"TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5"}

    def test_naturally_bridged(self, classification) -> None:
        assert classification.naturally_bridged == ["BOTH-1"]

    def test_unlinked(self, classification) -> None:
        assert set(classification.unlinked) == {"UNLINKED-1", "UNLINKED-2"}

    def test_all_lists_sorted(self, classification) -> None:
        assert classification.ai_only == sorted(classification.ai_only)
        assert classification.trad_only == sorted(classification.trad_only)
        assert classification.naturally_bridged == sorted(classification.naturally_bridged)
        assert classification.unlinked == sorted(classification.unlinked)

    def test_no_overlap(self, classification) -> None:
        sets = [
            set(classification.ai_only),
            set(classification.trad_only),
            set(classification.naturally_bridged),
            set(classification.unlinked),
        ]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert sets[i].isdisjoint(sets[j])

    def test_all_hubs_accounted_for(self, classification) -> None:
        total = (
            len(classification.ai_only)
            + len(classification.trad_only)
            + len(classification.naturally_bridged)
            + len(classification.unlinked)
        )
        assert total == len(ALL_HUB_IDS)

    def test_hub_not_in_links(self) -> None:
        """Hub with no links at all classified as unlinked."""
        from tract.bridge.classify import classify_hubs
        result = classify_hubs(FIXTURE_PATH, ["TOTALLY-NEW"])
        assert result.unlinked == ["TOTALLY-NEW"]
        assert result.ai_only == []
        assert result.trad_only == []
