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


AI_SECURITY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "bridge_ai_security_hub_links.json"
)

AI_SECURITY_HUB_IDS = [
    "AI-1", "ENISA-ONLY-1", "BIML-ONLY-1", "ETSI-ONLY-1", "TRAD-1", "BOTH-1",
]


class TestAiSecurityFrameworksAreNotTraditional:
    """ENISA, BIML and ETSI are AI-security frameworks, not traditional ones.

    They were absent from BRIDGE_AI_FRAMEWORK_IDS, so `classify_hubs` counted
    them on the traditional side. That is what produced the published model
    card's claim of 60 "naturally bridged" hubs with the worked example
    "Data poisoning (linked by both ATLAS and CWE)" -- measured against the
    curated links, MITRE ATLAS hubs and CWE hubs intersect in ZERO hubs, and
    the traditional side of every one of those bridges came from ENISA, ETSI
    or BIML. Under the eight-framework definition the count is 0, which is
    what PRD.md:58 has said all along.
    """

    @pytest.fixture
    def classification(self):
        from tract.bridge.classify import classify_hubs
        return classify_hubs(AI_SECURITY_FIXTURE, AI_SECURITY_HUB_IDS)

    def test_a_hub_linked_only_by_ai_frameworks_is_not_a_bridge(
        self, classification,
    ) -> None:
        # AI-1 is linked by MITRE ATLAS and ENISA. Both are AI-security, so
        # this is not a bridge to traditional security.
        assert "AI-1" not in classification.naturally_bridged
        assert "AI-1" in classification.ai_only

    def test_enisa_only_hub_is_ai_only(self, classification) -> None:
        assert "ENISA-ONLY-1" in classification.ai_only

    def test_biml_only_hub_is_ai_only(self, classification) -> None:
        assert "BIML-ONLY-1" in classification.ai_only

    def test_etsi_only_hub_is_ai_only(self, classification) -> None:
        assert "ETSI-ONLY-1" in classification.ai_only

    def test_a_genuine_ai_to_traditional_bridge_is_still_found(
        self, classification,
    ) -> None:
        # BOTH-1 is MITRE ATLAS + ASVS. That is a real bridge and must survive.
        assert classification.naturally_bridged == ["BOTH-1"]

    def test_a_traditional_only_hub_is_unaffected(self, classification) -> None:
        assert classification.trad_only == ["TRAD-1"]
