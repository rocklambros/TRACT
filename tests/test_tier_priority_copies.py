"""Two tier-priority tables, duplicated on purpose, and neither knew about T2.

`tract/training/data.py:TIER_PRIORITY` and
`tract/ceiling_study.py:_TIER_PRIORITY` hold the same mapping. The duplication
is deliberate and documented: `tract.training.data` pulls in torch,
sentence-transformers and datasets, and the ceiling study has to run without a
GPU or those packages.

Deliberate duplication still needs a test binding the copies, and it needs one
more than accidental duplication does, because the reason for it discourages
the import that would keep them in step.

WHY IT MATTERS HERE. The lookup is `TIER_PRIORITY.get(tier, 99)`. A tier absent
from the table does not raise -- it is assigned 99 and therefore loses every
deduplication contest. When Phase 2C bridge links arrive as T2, that silently
discards a human-authored link in favour of an automatic (T3) one, which
inverts the ordering the table exists to express.

So these tests assert three things: the copies agree, every non-dropped tier
has an entry, and the ordering is the one the comment claims.
"""

from __future__ import annotations

from tract.ceiling_study import _TIER_PRIORITY
from tract.training.data import TIER_PRIORITY
from tract.training.data_quality import QualityTier


class TestTheCopiesAgree:
    def test_both_tables_are_identical(self) -> None:
        assert TIER_PRIORITY == _TIER_PRIORITY, (
            "The two tier-priority tables have diverged. They are duplicated "
            "because tract.training.data imports torch; that is a reason to "
            "test them together, not a reason to let them drift."
        )


class TestEveryTierIsRanked:
    def test_no_non_dropped_tier_is_missing(self) -> None:
        """The load-bearing assertion.

        A missing tier does not raise -- `.get(tier, 99)` gives it the worst
        rank. Adding a QualityTier member without updating both tables must
        fail here rather than quietly reorder the deduplication.
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


class TestTheOrderingIsTheDocumentedOne:
    def test_human_links_outrank_automatic_ones(self) -> None:
        """The property the table's own comment states."""
        assert TIER_PRIORITY[QualityTier.T1.value] < TIER_PRIORITY[
            QualityTier.T3.value
        ]
        assert TIER_PRIORITY[QualityTier.T1_AI.value] < TIER_PRIORITY[
            QualityTier.T3.value
        ]

    def test_bridge_links_outrank_automatic_ones(self) -> None:
        """T2 is human-authored. Losing to an automatic link is the defect."""
        assert TIER_PRIORITY[QualityTier.T2.value] < TIER_PRIORITY[
            QualityTier.T3.value
        ]

    def test_opencre_curated_links_outrank_bridge_links(self) -> None:
        """T1 asserts independent OpenCRE curation; T2 asserts one annotator."""
        assert TIER_PRIORITY[QualityTier.T1.value] < TIER_PRIORITY[
            QualityTier.T2.value
        ]
        assert TIER_PRIORITY[QualityTier.T1_AI.value] < TIER_PRIORITY[
            QualityTier.T2.value
        ]
