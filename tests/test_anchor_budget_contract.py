"""One budget must govern training anchors, eval anchors, and the truncation count.

Three defects made "raise the context budget" a change that would not have done
what it says, and would have corrupted the rebaseline rather than failing loudly.

1. `build_training_pairs` declared `max_chars` and never forwarded it to
   `select_control_text`. Every caller computed
   `max_anchor_chars(config.max_seq_length)` and passed it in good faith; every
   training anchor was cut at the module-level MAX_ANCHOR_CHARS regardless.
   Campaign 2 escaped it only because it ran at 512 tokens, where the two
   coincide. Raising the budget would have lengthened EVAL anchors and left
   TRAINING anchors at 2,150 -- train/eval skew introduced by the flag meant to
   remove one, and the delta would have measured the text change.

2. `run_experiment` built the eval corpus with no `max_chars` at all while
   `run_fold.py` passed one, so the two entrypoints produced different anchors
   for any configuration that was not 512 tokens.

3. The fold record's truncation count was re-derived as
   `len(anchor) >= MAX_ANCHOR_CHARS`. `prepare_anchor` rstrips AFTER cutting, so
   a truncated anchor can be shorter than the budget it was cut to; the
   heuristic reported 39 truncated eval anchors across Campaign 2's test round
   where the real figure is 55.

None of these tests loads a model.
"""
from __future__ import annotations

import json
from typing import Final

import pytest

from tract.config import MAX_ANCHOR_CHARS, PROCESSED_DIR, max_anchor_chars

# `tract.training.data` imports torch at module scope, and CI's light `test`
# job does not install it -- that is what the separate `training-stack` job is
# for. Same convention as tests/test_branch_balancing.py.
pytest.importorskip("torch")

# Well under any real budget, so a forwarded value is unmistakable and a
# dropped one is too.
TINY_BUDGET: Final[int] = 400


@pytest.fixture(scope="module")
def corpus_inputs():  # type: ignore[no-untyped-def]
    """Hierarchy, links and prose index — the real ones, not a fixture."""
    from scripts.phase0.common import load_curated_links
    from tract.hierarchy import CREHierarchy
    from tract.text_selection import ProseIndex

    path = PROCESSED_DIR / "cre_hierarchy.json"
    if not path.is_file():
        pytest.skip(f"{path} absent")
    hierarchy = CREHierarchy.model_validate(
        json.loads(path.read_text(encoding="utf-8")),
    )
    return hierarchy, load_curated_links(), ProseIndex.load()


class TestTrainingAnchorsHonourTheBudget:
    """The bug that would have silently broken a long-context rebaseline."""

    def test_max_chars_reaches_the_training_anchors(self, corpus_inputs) -> None:  # type: ignore[no-untyped-def]
        """No training anchor may exceed the budget the caller asked for.

        Before the fix, `max_chars=50` produced 4,337 anchors ALL longer than
        50, the longest exactly MAX_ANCHOR_CHARS.
        """
        from tract.training.data import build_training_pairs
        from tract.training.firewall import build_all_hub_texts
        from tract.training.orchestrate import load_and_filter_curated_links

        hierarchy, _links, prose = corpus_inputs
        tiered, _ = load_and_filter_curated_links()
        pairs = build_training_pairs(
            tiered, build_all_hub_texts(hierarchy),
            prose_index=prose, max_chars=TINY_BUDGET,
        )
        assert pairs, "no training pairs built"
        longest = max(len(p.control_text) for p in pairs)
        over = sum(1 for p in pairs if len(p.control_text) > TINY_BUDGET)
        assert over == 0, (
            f"{over} of {len(pairs)} training anchors exceed the requested "
            f"max_chars={TINY_BUDGET} (longest {longest}). The parameter is "
            "not reaching select_control_text, so training anchors are pinned "
            "at MAX_ANCHOR_CHARS whatever the config asks for."
        )

    def test_a_larger_budget_actually_produces_longer_anchors(
        self, corpus_inputs,  # type: ignore[no-untyped-def]
    ) -> None:
        """The positive control: the parameter must MOVE the anchors, not just bound them.

        Bounding alone would pass if the function ignored max_chars and every
        anchor happened to be short.
        """
        from tract.training.data import build_training_pairs
        from tract.training.firewall import build_all_hub_texts
        from tract.training.orchestrate import load_and_filter_curated_links

        hierarchy, _links, prose = corpus_inputs
        tiered, _ = load_and_filter_curated_links()
        hub_texts = build_all_hub_texts(hierarchy)
        small = build_training_pairs(
            tiered, hub_texts, prose_index=prose, max_chars=MAX_ANCHOR_CHARS,
        )
        large = build_training_pairs(
            tiered, hub_texts, prose_index=prose,
            max_chars=max_anchor_chars(2048),
        )
        small_total = sum(len(p.control_text) for p in small)
        large_total = sum(len(p.control_text) for p in large)
        assert large_total > small_total, (
            "doubling the token budget did not lengthen any training anchor; "
            f"{small_total} chars at {MAX_ANCHOR_CHARS} vs {large_total} at "
            f"{max_anchor_chars(2048)}"
        )


class TestEvalAnchorsMatchAcrossEntrypoints:
    """run_experiment and run_fold must build the same anchors."""

    def test_both_paths_pass_the_config_derived_budget(self) -> None:
        """Asserted on the source, because running both paths needs a GPU.

        run_experiment omitted max_chars entirely, so it silently used the
        module constant while run_fold.py used the config. A structural check
        is the cheap way to keep them in step.
        """
        import inspect

        from tract.training import orchestrate

        source = inspect.getsource(orchestrate.run_experiment)
        assert "max_chars=max_anchor_chars(config.max_seq_length)" in source, (
            "run_experiment builds the eval corpus without the config-derived "
            "anchor budget, so it disagrees with scripts/phase1b/run_fold.py "
            "for any max_seq_length that is not 512"
        )


class TestTruncationIsCountedNotGuessed:
    """The fold record must report what was cut, not what looks cut."""

    def test_rstrip_makes_the_length_heuristic_undercount(self) -> None:
        """The mechanism, demonstrated on one string.

        This is why the count has to come from SelectionStats: a truncated
        anchor can be shorter than its own budget, so no length comparison can
        recover the flag.
        """
        from tract.text_selection import prepare_anchor

        budget = 40
        text = "word " * 40  # trailing space lands exactly on the cut
        prepared, truncated = prepare_anchor(text, budget)
        assert truncated, "this input must truncate for the test to mean anything"
        assert len(prepared) < budget, (
            "expected rstrip to pull the result under the budget; without that "
            "the heuristic would happen to be right and this test is not "
            "exercising the defect"
        )
        assert not (len(prepared) >= budget), (
            "the old heuristic `len(anchor) >= budget` reports NOT TRUNCATED "
            "for an anchor that was truncated"
        )

    def test_run_single_fold_prefers_supplied_stats(self) -> None:
        """Structural: the fold must read corpus_selection when it is given.

        Running a real fold needs a GPU, so this asserts the wiring. The
        companion assertion -- that both entrypoints supply it -- is below.
        """
        import inspect

        from tract.training import orchestrate

        source = inspect.getsource(orchestrate.run_single_fold)
        assert "corpus_selection" in source
        assert "truncated_by_framework" in source, (
            "run_single_fold no longer reads the authoritative per-framework "
            "truncation counts from the supplied SelectionStats"
        )

    def test_both_entrypoints_supply_the_stats(self) -> None:
        import inspect
        from pathlib import Path

        from tract.training import orchestrate

        experiment = inspect.getsource(orchestrate.run_experiment)
        assert "corpus_selection=corpus_selection" in experiment, (
            "run_experiment does not hand its SelectionStats to run_single_fold"
        )
        run_fold = (
            Path(__file__).resolve().parent.parent
            / "scripts" / "phase1b" / "run_fold.py"
        ).read_text(encoding="utf-8")
        assert "corpus_selection=selection_stats" in run_fold, (
            "scripts/phase1b/run_fold.py does not hand its SelectionStats to "
            "run_single_fold, so pod folds fall back to the undercount"
        )


class TestFrameworkNameAliasing:
    """The lookup crosses a naming boundary and must not silently return zero."""

    def test_every_fold_name_resolves_to_its_stats_key(self) -> None:
        """SelectionStats is keyed by CANONICAL name; folds use the roster name.

        These differ for at least one framework: the roster says
        "OWASP Top10 for LLM" and the stats key is
        "OWASP Top 10 for LLM Applications 2025". A missing
        canonical_framework() call would make `.get(name, 0)` return 0 and the
        fold would report NO truncation for a fold that is 100% truncated --
        failing silently in the direction that looks like success.
        """
        from scripts.phase0.common import (
            AI_FRAMEWORK_NAMES,
            build_evaluation_corpus,
            load_curated_links,
        )
        from tract.framework_identity import filter_set
        from tract.text_selection import (
            ProseIndex,
            SelectionStats,
            apply_prose_to_corpus,
            canonical_framework,
        )

        stats = SelectionStats()
        corpus = build_evaluation_corpus(
            load_curated_links(), AI_FRAMEWORK_NAMES, {},
        )
        apply_prose_to_corpus(
            corpus, ProseIndex.load(),
            filter_set(use_stopwords=True, use_framework_identity=False),
            stats=stats, max_chars=MAX_ANCHOR_CHARS,
        )
        assert stats.n_truncated > 0, "fixture produced no truncation to attribute"

        via_lookup = sum(
            stats.truncated_by_framework.get(canonical_framework(fw), 0)
            for fw in AI_FRAMEWORK_NAMES
        )
        assert via_lookup == stats.n_truncated, (
            f"per-fold lookups recover {via_lookup} of {stats.n_truncated} "
            "truncated anchors. A fold name is not resolving to its stats key, "
            "so that fold will report zero truncation."
        )
