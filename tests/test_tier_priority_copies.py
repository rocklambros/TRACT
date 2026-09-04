"""Tier priority: one definition, and every tier ranked.

There used to be two tables. `tract/training/data.py` held one and
`tract/ceiling_study.py` held a copy, duplicated on purpose and with the reason
documented: importing the first pulls in torch, sentence-transformers and
datasets, and the ceiling study has to run without a GPU or those packages.

The duplication then did exactly what duplication does. Neither copy learned
about T2 when the tier was added, and the lookup is `TIER_PRIORITY.get(tier, 99)`
-- so a missing tier does not raise, it takes the worst rank and loses every
deduplication contest. A human-authored Tier-2 bridge link would have been
discarded in favour of an automatically-linked one, silently, inverting the
ordering the table exists to express.

The first fix here was a test binding the two copies. That was the wrong fix
twice over: it left the duplication in place, and it had to import
`tract.training.data` to read one of them, which made the test module import
torch -- absent from CI's requirements. Both new test modules did that, both
raised at collection, and **pytest aborts the whole run on a collection error**,
so CI executed 15 tests while reading as an ordinary red. That is premortem
finding A2 recurring, in the session that documented A2.

So the definition moved to `tract/config.py`, which imports nothing heavy.
`tract.training.data` and `tract.ceiling_study` both import it, there is nothing
left to drift, and this module reads it without torch.
"""

from __future__ import annotations

from tract.config import TIER_PRIORITY
from tract.training.data_quality import QualityTier


class TestThereIsOnlyOneTable:
    def test_the_ceiling_study_alias_is_the_same_object(self) -> None:
        """Not "equal to" -- the same object, so drift is impossible.

        `tract.ceiling_study` is torch-free, so this import is safe here.
        """
        from tract.ceiling_study import _TIER_PRIORITY

        assert _TIER_PRIORITY is TIER_PRIORITY

    def test_no_module_redefines_it(self) -> None:
        """A future copy-paste reintroduces the bug this file is named after."""
        from pathlib import Path

        from tract.config import PROJECT_ROOT

        definers: list[str] = []
        for path in sorted((PROJECT_ROOT / "tract").rglob("*.py")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if rel == "tract/config.py":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(("TIER_PRIORITY", "_TIER_PRIORITY")) and (
                    "{" in stripped
                ):
                    definers.append(f"{rel}: {stripped}")
        assert not definers, (
            "TIER_PRIORITY is defined outside tract/config.py again: "
            f"{definers}. Import it instead; the duplicate drifted last time "
            "and cost a silent tier inversion."
        )


class TestEveryTierIsRanked:
    def test_no_non_dropped_tier_is_missing(self) -> None:
        """The load-bearing assertion.

        A missing tier does not raise -- `.get(tier, 99)` gives it the worst
        rank. Adding a QualityTier member without ranking it must fail here
        rather than quietly reorder deduplication.
        """
        ranked = set(TIER_PRIORITY)
        expected = {t.value for t in QualityTier} - {QualityTier.DROPPED.value}
        assert expected <= ranked, (
            f"Tiers with no priority entry: {sorted(expected - ranked)}. "
            "They will be ranked 99 and lose every dedup contest, including "
            "against automatically-linked records."
        )

    def test_dropped_is_not_ranked(self) -> None:
        """A dropped link never reaches deduplication; ranking it would mislead."""
        assert QualityTier.DROPPED.value not in TIER_PRIORITY

    def test_ranks_are_distinct(self) -> None:
        """Two tiers sharing a rank makes their contest order arbitrary."""
        assert len(set(TIER_PRIORITY.values())) == len(TIER_PRIORITY)

    def test_nothing_is_ranked_that_is_not_a_tier(self) -> None:
        stray = set(TIER_PRIORITY) - {t.value for t in QualityTier}
        assert not stray, f"Ranked names that are not QualityTier values: {stray}"


class TestTheOrderingIsTheDocumentedOne:
    def test_human_links_outrank_automatic_ones(self) -> None:
        assert TIER_PRIORITY["T1"] < TIER_PRIORITY["T3"]
        assert TIER_PRIORITY["T1-AI"] < TIER_PRIORITY["T3"]

    def test_bridge_links_outrank_automatic_ones(self) -> None:
        """T2 is human-authored. Losing to an automatic link is the defect."""
        assert TIER_PRIORITY["T2"] < TIER_PRIORITY["T3"]

    def test_opencre_curated_links_outrank_bridge_links(self) -> None:
        """T1 asserts independent OpenCRE curation; T2 asserts one annotator."""
        assert TIER_PRIORITY["T1"] < TIER_PRIORITY["T2"]
        assert TIER_PRIORITY["T1-AI"] < TIER_PRIORITY["T2"]
