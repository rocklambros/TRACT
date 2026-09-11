"""Holding out eight frameworks at once, which Gate 2 requires and nothing did.

`run_single_fold` took `held_out_framework: str`, singular, and the exclusion in
`build_training_pairs` was `standard_name == excluded_framework`. Gate 2 scores
ENISA, BIML and ETSI under a firewall that must hold out the whole AI region --
all eight frameworks -- because the AI and traditional hub regions are disjoint
and leaving any AI framework in training supplies the very positives the bridge
corpus is bought to provide.

The dangerous shape is not the missing feature. It is that
`--split validation --framework ENISA` runs TODAY, passes every guard, holds out
ENISA alone, leaves the other seven AI frameworks in training -- 52 of 56 scored
hubs still supervised -- and writes a fold record indistinguishable from a
firewalled one. Nothing in the record names the firewall.

And a set passed into the old `==` comparison would have been permanently False:
nothing excluded, everything trained, hit@1 around 0.9, no guard objecting. So
the exclusion is set membership now, which is correct for a name or a set, and
there is a converse assertion that the exclusion actually removed what it said.
"""

from __future__ import annotations

import pytest

from tract.training.data import build_training_pairs, excluded_framework_set
from tract.training.data_quality import QualityTier, TieredLink
from tract.training.firewall import assert_exclusion_fired


def _Link(standard: str, section: str, hub: str) -> TieredLink:
    """A real TieredLink, not a stand-in.

    An earlier draft of this file used a stub with a `quality_tier` attribute,
    which `build_training_pairs` does not read -- it reads `tier.value`. The
    stub would have passed a test of my own invention against a shape the
    production code never sees.
    """
    return TieredLink(
        link={
            "standard_name": standard,
            "section_id": section,
            "section_name": f"title of {section}",
            "cre_id": hub,
        },
        tier=QualityTier.T1,
    )


def _links() -> list[TieredLink]:
    return [
        _Link("ENISA", "E-1", "010-108"),
        _Link("BIML", "B-1", "010-109"),
        _Link("ETSI", "T-1", "010-110"),
        _Link("MITRE ATLAS", "A-1", "010-111"),
        _Link("NIST 800-53 v5", "AC-1", "020-200"),
        _Link("CAPEC", "C-1", "020-201"),
    ]


HUBS = {f"010-10{i}": f"path | hub {i}" for i in range(8, 10)} | {
    "010-110": "path | hub ten",
    "010-111": "path | hub eleven",
    "020-200": "path | trad one",
    "020-201": "path | trad two",
}


class TestTheNormaliser:
    def test_none_excludes_nothing(self) -> None:
        assert excluded_framework_set(None) == frozenset()

    def test_a_bare_name_still_works(self) -> None:
        """Every existing caller passes a string and must keep working."""
        assert excluded_framework_set("ENISA") == frozenset({"ENISA"})

    def test_a_set_is_carried_through(self) -> None:
        assert excluded_framework_set({"ENISA", "BIML"}) == frozenset(
            {"ENISA", "BIML"}
        )

    def test_an_empty_string_excludes_nothing(self) -> None:
        """Not a framework named "". The old `if excluded_framework and` guard
        treated empty as absent and that behaviour is preserved."""
        assert excluded_framework_set("") == frozenset()


class TestTheExclusionActuallyExcludes:
    def test_a_single_name_excludes_one_framework(self) -> None:
        pairs = build_training_pairs(
            _links(), HUBS, excluded_framework="ENISA"  # type: ignore[arg-type]
        )
        frameworks = {p.framework for p in pairs}
        assert "ENISA" not in frameworks
        assert "BIML" in frameworks

    def test_a_set_excludes_every_named_framework(self) -> None:
        """The case that was silently a no-op before: `== {...}` is never True."""
        pairs = build_training_pairs(
            _links(),
            HUBS,
            excluded_framework=frozenset({"ENISA", "BIML", "ETSI"}),  # type: ignore[arg-type]
        )
        frameworks = {p.framework for p in pairs}
        assert frameworks.isdisjoint({"ENISA", "BIML", "ETSI"})
        assert "NIST 800-53 v5" in frameworks

    def test_the_whole_ai_region_can_be_held_out(self) -> None:
        from tract.config import PHASE2C_GATE2_HELD_OUT

        pairs = build_training_pairs(
            _links(), HUBS,
            excluded_framework=PHASE2C_GATE2_HELD_OUT,  # type: ignore[arg-type]
        )
        frameworks = {p.framework for p in pairs}
        assert frameworks.isdisjoint(PHASE2C_GATE2_HELD_OUT)
        assert frameworks, "the traditional links must survive"


class TestTheConverseAssertion:
    """Nothing asserted that the exclusion removed what it claimed to.

    `build_training_pairs` logged a count and moved on, so a mis-plumbed
    exclusion set is a silent hit@1 of 0.9 rather than a failure.
    """

    def test_it_passes_when_the_exclusion_fired(self) -> None:
        links = _links()
        pairs = build_training_pairs(
            links, HUBS, excluded_framework="ENISA"  # type: ignore[arg-type]
        )
        assert_exclusion_fired(pairs, links, frozenset({"ENISA"}))

    def test_it_raises_when_the_name_matches_nothing(self) -> None:
        """The silent failure: a typo, or a roster that drifted from the corpus.

        "OWASP LLM Top 10" against a corpus that says "OWASP Top10 for LLM"
        removes zero links and trains on everything.
        """
        links = _links()
        pairs = build_training_pairs(
            links, HUBS,
            excluded_framework=frozenset({"OWASP LLM Top 10"}),  # type: ignore[arg-type]
        )
        with pytest.raises(AssertionError, match="matched no links"):
            assert_exclusion_fired(pairs, links, frozenset({"OWASP LLM Top 10"}))

    def test_it_raises_when_an_excluded_framework_survived(self) -> None:
        """Direct check: no surviving pair may name an excluded framework."""
        links = _links()
        unfiltered = build_training_pairs(links, HUBS, excluded_framework=None)
        with pytest.raises(AssertionError, match="still carry excluded"):
            assert_exclusion_fired(unfiltered, links, frozenset({"ENISA"}))

    def test_an_empty_exclusion_is_a_no_op(self) -> None:
        """A fold that holds out nothing is not a firewall failure."""
        links = _links()
        pairs = build_training_pairs(links, HUBS, excluded_framework=None)
        assert_exclusion_fired(pairs, links, frozenset())

    def test_the_whole_ai_region_fires(self) -> None:
        from tract.config import PHASE2C_GATE2_HELD_OUT

        links = _links()
        pairs = build_training_pairs(
            links, HUBS,
            excluded_framework=PHASE2C_GATE2_HELD_OUT,  # type: ignore[arg-type]
        )
        assert_exclusion_fired(pairs, links, PHASE2C_GATE2_HELD_OUT)


class TestHubTextsAcceptASet:
    def test_build_all_hub_texts_takes_a_set(self) -> None:
        from tract.hierarchy import CREHierarchy
        from tract.config import PROCESSED_DIR
        from tract.io import load_json

        hierarchy = CREHierarchy.model_validate(
            load_json(PROCESSED_DIR / "cre_hierarchy.json")
        )
        from tract.training.firewall import build_all_hub_texts

        texts = build_all_hub_texts(
            hierarchy, excluded_framework=frozenset({"ENISA", "BIML"})
        )
        assert len(texts) == len(hierarchy.hubs)
