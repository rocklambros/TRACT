"""The placebo must differ from the treatment in judgement and nothing else.

Under the strict all-AI firewall the bridge-free comparator has zero training
positives for all 62 scored hubs, so `A1 - A0` asks "does any supervision beat
none" rather than "do these human-curated links help". A0R is what makes the
question the second one.

Every property held constant here is load-bearing. If the placebo's hub
distribution drifts, the exposure partition drifts with it, and `A1 - A0R` then
confounds the annotators' judgement with coverage.
"""

from __future__ import annotations

from pathlib import Path

import collections
import pytest

from scripts.build_placebo_corpus import build_placebo, candidate_controls
from tract.bridge.links import BridgeLink, load_bridge_links

REAL = Path("data/training/hub_links_bridge.r2.jsonl")


def _link(section: str, hub: str, *, confidence: int = 3) -> BridgeLink:
    return BridgeLink(
        framework_id="nist_800_53", standard_name="NIST 800-53 v5",
        section_id=section, section_name=f"name {section}", cre_id=hub,
        tier=2, annotator_id="vol-01", created_at="2026-09-07",
        confidence=confidence, rationale="because",
    )


CANDIDATES = {f"AC-{i}": f"control {i}" for i in range(60)}


class TestWhatIsHeldConstant:
    def test_same_number_of_edges(self) -> None:
        real = [_link("AC-1", "010-108"), _link("AC-2", "010-109")]
        placebo = build_placebo(real, CANDIDATES, seed=1)
        assert len(placebo) == len(real)

    def test_identical_hub_distribution(self) -> None:
        """The property the exposure partition depends on."""
        real = [
            _link("AC-1", "010-108"), _link("AC-2", "010-108"),
            _link("AC-3", "010-109"),
        ]
        placebo = build_placebo(real, CANDIDATES, seed=1)
        assert collections.Counter(b.cre_id for b in placebo) == (
            collections.Counter(b.cre_id for b in real)
        )

    def test_same_source_framework(self) -> None:
        real = [_link("AC-1", "010-108")]
        placebo = build_placebo(real, CANDIDATES, seed=1)
        assert placebo[0].standard_name == "NIST 800-53 v5"
        assert placebo[0].framework_id == "nist_800_53"

    def test_tier_is_preserved_so_it_trains_the_same_way(self) -> None:
        """A different tier would change assign_quality_tier's branch."""
        placebo = build_placebo([_link("AC-1", "010-108")], CANDIDATES, seed=1)
        assert placebo[0].tier == 2


class TestWhatIsDeliberatelyDifferent:
    def test_no_real_pair_is_reproduced(self) -> None:
        """A placebo overlapping the real corpus is a weaker comparator.

        By exactly the overlap, and silently: the arm would just under-perform
        its own label with nothing to show why.
        """
        real = [_link(f"AC-{i}", "010-108") for i in range(5)]
        placebo = build_placebo(real, CANDIDATES, seed=1)
        assert not (
            {(b.section_id, b.cre_id) for b in real}
            & {(b.section_id, b.cre_id) for b in placebo}
        )

    def test_it_is_labelled_as_not_a_human_judgement(self) -> None:
        """So a placebo record can never be mistaken for Tier-2 annotator work."""
        placebo = build_placebo([_link("AC-1", "010-108")], CANDIDATES, seed=1)
        assert placebo[0].annotator_id == "placebo"
        assert "Not a human judgement" in placebo[0].rationale


class TestDeterminism:
    def test_same_seed_gives_the_same_corpus(self) -> None:
        real = [_link(f"AC-{i}", "010-108") for i in range(5)]
        assert [b.section_id for b in build_placebo(real, CANDIDATES, seed=7)] == [
            b.section_id for b in build_placebo(real, CANDIDATES, seed=7)
        ]

    def test_different_seeds_give_different_corpora(self) -> None:
        real = [_link(f"AC-{i}", "010-108") for i in range(5)]
        assert [b.section_id for b in build_placebo(real, CANDIDATES, seed=7)] != [
            b.section_id for b in build_placebo(real, CANDIDATES, seed=8)
        ]

    def test_input_order_does_not_change_the_output(self) -> None:
        """The draw is over sorted inputs, not over whatever order arrived."""
        real = [_link(f"AC-{i}", "010-108") for i in range(5)]
        assert sorted(
            (b.section_id, b.cre_id)
            for b in build_placebo(real, CANDIDATES, seed=7)
        ) == sorted(
            (b.section_id, b.cre_id)
            for b in build_placebo(list(reversed(real)), CANDIDATES, seed=7)
        )


class TestItRefusesRatherThanDegrading:
    def test_an_empty_real_corpus_raises(self) -> None:
        with pytest.raises(ValueError, match="nothing to match"):
            build_placebo([], CANDIDATES, seed=1)

    def test_too_few_candidates_raises(self) -> None:
        """Rather than reusing controls and shrinking the effective corpus."""
        real = [_link(f"AC-{i}", "010-108") for i in range(5)]
        with pytest.raises(ValueError, match="candidate"):
            build_placebo(real, {"AC-0": "one", "AC-1": "two"}, seed=1)


class TestAgainstTheRealCorpus:
    def test_the_built_placebo_matches_the_real_one(self) -> None:
        if not REAL.is_file():
            pytest.skip(f"{REAL} is gitignored and absent on this machine")
        real = load_bridge_links(REAL)
        placebo = build_placebo(real, candidate_controls())
        assert len(placebo) == len(real) == 124
        assert collections.Counter(b.cre_id for b in placebo) == (
            collections.Counter(b.cre_id for b in real)
        )
        assert not (
            {(b.section_id, b.cre_id) for b in real}
            & {(b.section_id, b.cre_id) for b in placebo}
        )
