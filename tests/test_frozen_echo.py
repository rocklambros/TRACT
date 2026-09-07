"""The echo partition must be a property of the item, not of the run.

`results/phase1b/CAMPAIGN3.md` makes the non-echo stratum a binding side
condition on the gate. Campaign 2's partition was computed per arm from the
TRUNCATED prose anchor, so it moved with `max_seq_length`: 38 echo items at the
2,150-char budget, 41 at 4,300, 44 at 8,601. Under the truncated split the
non-echo stratum is 98 items; untruncated it is 93, and the items that move
carry trained hit == zero-shot hit — so re-partitioning alone shifts the
side-condition metric with no change in model behaviour.

These tests pin the property that makes the condition bindable: the partition
does not move when the budget does.
"""
from __future__ import annotations

import json

import pytest

from tract.config import PROCESSED_DIR, max_anchor_chars
from tract.training.echo import UNTRUNCATED, frozen_echo_indices, is_echo


@pytest.fixture(scope="module")
def pieces():  # type: ignore[no-untyped-def]
    from scripts.phase0.common import (
        AI_FRAMEWORK_NAMES,
        build_evaluation_corpus,
        load_curated_links,
    )
    from tract.framework_identity import filter_set
    from tract.hierarchy import CREHierarchy
    from tract.text_selection import ProseIndex

    path = PROCESSED_DIR / "cre_hierarchy.json"
    if not path.is_file():
        pytest.skip(f"{path} absent")
    hierarchy = CREHierarchy.model_validate(
        json.loads(path.read_text(encoding="utf-8")),
    )
    corpus = build_evaluation_corpus(
        load_curated_links(), AI_FRAMEWORK_NAMES, {},
    )
    stopwords = filter_set(use_stopwords=True, use_framework_identity=False)
    return corpus, hierarchy, ProseIndex.load(), stopwords


class TestIsEcho:
    def test_requires_every_content_word(self) -> None:
        assert is_echo("we perform anomalous ai input handling here",
                       "Anomalous AI input handling")
        assert not is_echo("we handle input", "Anomalous AI input handling")

    def test_function_words_do_not_carry_the_match(self) -> None:
        """"of the" in a hub name must not be what makes an anchor echo."""
        assert is_echo("logging", "Logging of the")

    def test_a_hub_with_no_content_words_is_never_echo(self) -> None:
        """Otherwise the empty subset makes every item echo vacuously."""
        assert not is_echo("anything at all", "the of and")


class TestThePartitionDoesNotMoveWithTheBudget:
    """The property the binding side condition depends on."""

    def test_identical_at_every_anchor_budget(self, pieces) -> None:  # type: ignore[no-untyped-def]
        corpus, hierarchy, prose, stopwords = pieces
        reference = frozen_echo_indices(corpus, hierarchy, prose, stopwords)
        assert reference, "no echo items found; fixture is not exercising this"
        for seq_len in (512, 1024, 2048, 4096):
            again = frozen_echo_indices(corpus, hierarchy, prose, stopwords)
            assert again == reference, (
                f"partition changed at max_seq_length={seq_len} "
                f"(budget {max_anchor_chars(seq_len)} chars)"
            )

    def test_it_differs_from_the_truncated_partition(self, pieces) -> None:  # type: ignore[no-untyped-def]
        """The bug is real, not hypothetical: truncation hides echo items.

        If these ever coincide, either the corpus stopped truncating or the
        frozen computation silently started using a budget.
        """
        from tract.text_selection import apply_prose_to_corpus

        corpus, hierarchy, prose, stopwords = pieces
        names = {h: n.name for h, n in hierarchy.hubs.items()}

        def echo_at(budget: int) -> set[tuple[str, str]]:
            items = apply_prose_to_corpus(
                corpus, prose, stopwords, max_chars=budget,
            )
            return {
                (i.framework_name, i.section_id) for i in items
                if is_echo(i.control_text, names.get(i.ground_truth_hub_id, ""))
            }

        truncated = echo_at(max_anchor_chars(512))
        untruncated = echo_at(UNTRUNCATED)
        assert untruncated > truncated, (
            "restoring the truncated tail revealed no new echo items; the "
            "moving-ruler defect this module exists for is not reproducible "
            "and the module's rationale needs rewriting"
        )

    def test_frozen_is_a_superset_of_every_budget_partition(
        self, pieces,  # type: ignore[no-untyped-def]
    ) -> None:
        """Frozen must be the most generous definition, so non-echo is conservative.

        A claim made on the non-echo stratum is a claim about items that are not
        echo under ANY text the project could show the model.
        """
        from tract.text_selection import apply_prose_to_corpus

        corpus, hierarchy, prose, stopwords = pieces
        names = {h: n.name for h, n in hierarchy.hubs.items()}
        frozen = frozen_echo_indices(corpus, hierarchy, prose, stopwords)

        for seq_len in (512, 1024, 2048):
            items = apply_prose_to_corpus(
                corpus, prose, stopwords, max_chars=max_anchor_chars(seq_len),
            )
            at_budget = {
                idx for idx, i in enumerate(items)
                if is_echo(i.control_text, names.get(i.ground_truth_hub_id, ""))
            }
            assert at_budget <= frozen, (
                f"at max_seq_length={seq_len}, {len(at_budget - frozen)} items "
                "are echo but are not in the frozen partition, so the non-echo "
                "stratum would contain a lexical shortcut"
            )
