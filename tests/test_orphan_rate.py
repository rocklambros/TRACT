"""Gate 1's free check: how many AI hubs have no traditional supervision.

Phase 2C exists because the AI and traditional hub regions are disjoint. An AI
hub with no link from any traditional framework is an ORPHAN under the strict
all-AI firewall: hold every AI framework out, and nothing remains that teaches
the model where that hub sits. Gate 1 is a reduction in this count.

TWO AI-FRAMEWORK DEFINITIONS EXIST AND ONLY ONE IS RIGHT HERE.
`scripts.phase0.common.AI_FRAMEWORK_NAMES` is the five-framework LOFO EVAL
ROSTER -- the frameworks that get held out and scored. `BRIDGE_AI_FRAMEWORK_IDS`
is the eight-framework AI REGION, which additionally contains ENISA, ETSI and
BIML. Region is the right one for an orphan count, and mixing them up is not
a rounding error:

    5-framework roster: 73 AI hubs, 57 of them apparently "bridged"
    8-framework region: 78 AI hubs, 0 bridged

Every one of those 57 apparent bridges comes from ENISA (51), ETSI (28) or BIML
(11) -- all AI frameworks. **No traditional framework links to any AI hub.** So
the 78-of-78 figure is not an artifact of the wider definition; the narrower one
is the artifact.

Keyed on framework_id rather than standard_name so the eight-id constant applies
directly and no third copy of the AI-framework list gets invented.
"""

from __future__ import annotations

import pytest

from scripts.analysis.orphan_rate import (
    bridge_link_pairs,
    load_framework_hub_links,
    strict_firewall_orphans,
)


@pytest.fixture(scope="module")
def pairs() -> list[tuple[str, str]]:
    return load_framework_hub_links()


class TestTheMeasuredBaseline:
    def test_reproduces_78_of_78(self, pairs: list[tuple[str, str]]) -> None:
        """docs/campaign3-audit-mechanism.md section 6g, and the whole premise."""
        assert strict_firewall_orphans(pairs) == (78, 78)

    def test_the_regions_are_disjoint(self, pairs: list[tuple[str, str]]) -> None:
        """Restating the baseline as the property it actually asserts.

        orphaned == total is exactly "no AI hub has a traditional link". If
        these ever diverge, Phase 2C's premise has changed and the gate needs
        rebaselining before it can be run.
        """
        orphaned, total = strict_firewall_orphans(pairs)
        assert orphaned == total

    def test_the_narrow_roster_definition_would_give_a_different_answer(
        self, pairs: list[tuple[str, str]]
    ) -> None:
        """Pin the trap, so a future edit to the id set is a deliberate act."""
        five = frozenset({
            "mitre_atlas", "nist_ai_100_2", "owasp_ai_exchange",
            "owasp_llm_top10", "owasp_ml_top10",
        })
        assert strict_firewall_orphans(pairs, ai_framework_ids=five) == (16, 73)


class TestABridgeRescuesItsHub:
    def test_one_traditional_link_reduces_the_orphan_count_by_one(
        self, pairs: list[tuple[str, str]]
    ) -> None:
        from tract.bridge.links import BridgeLink

        base, total = strict_firewall_orphans(pairs)
        target = sorted(
            {hub for fid, hub in pairs if fid in _ai_ids()}
        )[0]
        bridge = [
            BridgeLink(
                framework_id="nist_800_53",
                standard_name="NIST 800-53 v5",
                section_id="AC-3",
                section_name="Access Enforcement",
                cre_id=target,
                tier=2,
                annotator_id="a1",
                created_at="2026-09-01T00:00:00Z",
                confidence=3,
                rationale="test",
            )
        ]
        after, after_total = strict_firewall_orphans(
            list(pairs) + bridge_link_pairs(bridge)
        )
        assert after == base - 1
        assert after_total == total, "the denominator must not move"

    def test_a_second_link_to_the_same_hub_rescues_nothing_further(
        self, pairs: list[tuple[str, str]]
    ) -> None:
        """The count is over hubs, not links. Two links, one hub, one rescue.

        Without this a productive annotator could clear the gate by mapping
        many controls onto the same handful of hubs.
        """
        from tract.bridge.links import BridgeLink

        target = sorted({hub for fid, hub in pairs if fid in _ai_ids()})[0]

        def link(section: str) -> BridgeLink:
            return BridgeLink(
                framework_id="nist_800_53",
                standard_name="NIST 800-53 v5",
                section_id=section,
                section_name=section,
                cre_id=target,
                tier=2,
                annotator_id="a1",
                created_at="2026-09-01T00:00:00Z",
                confidence=3,
                rationale="test",
            )

        base, _ = strict_firewall_orphans(pairs)
        one = strict_firewall_orphans(list(pairs) + bridge_link_pairs([link("AC-3")]))
        two = strict_firewall_orphans(
            list(pairs) + bridge_link_pairs([link("AC-3"), link("AC-4")])
        )
        assert one[0] == base - 1
        assert two[0] == base - 1

    def test_an_ai_framework_bridge_rescues_nothing(
        self, pairs: list[tuple[str, str]]
    ) -> None:
        """Only traditional supervision de-orphans. An AI link is what it had."""
        from tract.bridge.links import BridgeLink

        target = sorted({hub for fid, hub in pairs if fid in _ai_ids()})[0]
        base, _ = strict_firewall_orphans(pairs)
        after, _ = strict_firewall_orphans(
            list(pairs)
            + bridge_link_pairs([
                BridgeLink(
                    framework_id="enisa",
                    standard_name="ENISA",
                    section_id="X",
                    section_name="X",
                    cre_id=target,
                    tier=2,
                    annotator_id="a1",
                    created_at="2026-09-01T00:00:00Z",
                    confidence=3,
                    rationale="test",
                )
            ])
        )
        assert after == base


class TestItRefusesInputItCannotMeasure:
    def test_an_empty_link_set_raises(self) -> None:
        """(0, 0) would read as a perfect score."""
        with pytest.raises(ValueError, match="no links"):
            strict_firewall_orphans([])

    def test_a_link_set_with_no_ai_framework_raises(self) -> None:
        with pytest.raises(ValueError, match="[Nn]o AI-framework"):
            strict_firewall_orphans([("nist_800_53", "111-111")])


def _ai_ids() -> frozenset[str]:
    from tract.config import BRIDGE_AI_FRAMEWORK_IDS

    return frozenset(BRIDGE_AI_FRAMEWORK_IDS)
