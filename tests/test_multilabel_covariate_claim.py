"""Premortem finding B3's delta split does not reproduce on any committed run.

B3 (round 1, High/Likely) reads:

    rotating the roster is confounded with a 4x multi-label shift. Multi-label
    share: incumbents 8.8%, candidates 36.5% (ENISA alone 51.5%). Single-label
    delta +0.1165 vs multi-label -0.1429; difference P(<=0)=0.044 -- better
    established than the H3 split the whole document rests on.

It produced remediation item 10, "pre-register multi-label density as a
covariate, or the predicted delta drop is uninterpretable".

THE COMPOSITION HALF IS RIGHT. Multi-label share among incumbents is 13 of 147
= 8.8%, exactly as stated.

THE DELTA HALF IS NOT. Measured over every committed run that carries
predictions:

    run                            single      multi      P(<=0)
    c2r_TEST_A3_prose_sw_qwen06b   +0.1493   +0.0000       0.073
    c3_TEST_..._seq1024            +0.1567   +0.0000       0.167
    lofo_prose                     +0.0672   +0.2308       0.955
    lofo_prose_desconly            +0.0746   +0.2308       0.937
    lofo_prose_stopwords           +0.0522   +0.3077       0.987
    lofo_title_only                +0.1119   +0.3077       0.957

The multi-label delta is never negative. On four of the six runs it is larger
than the single-label delta -- the opposite of the claimed direction -- and no
run reaches the claimed P(<=0)=0.044. The -0.1429 figure is 1/7, which no
n=13 stratum can produce.

WHAT SURVIVES. The confound is real in principle and does not depend on sign:
if the roster moves multi-label density from 8.8% to 36.5%, and multi-label
items behave differently in EITHER direction, the pooled delta moves for
compositional reasons. So multi-label density is worth disclosing and
stratifying on. What is not supported is the directional prediction that
rotating the roster will depress the delta, or the claim that this split is
better established than any other.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tract.config import PHASE1B_RESULTS_DIR

CLAIMED_MULTI_DELTA = -0.1429
CLAIMED_P = 0.044


def _split(run_dir):  # type: ignore[no-untyped-def]
    from scripts.analysis.audit_mechanism_probe import build_rows, contrast, score

    logging.disable(logging.INFO)
    try:
        rows = build_rows(run_dir)
    finally:
        logging.disable(logging.NOTSET)
    single = [r for r in rows if r["n_valid_hubs"] == 1]
    multi = [r for r in rows if r["n_valid_hubs"] > 1]
    rng = np.random.default_rng(42)
    return (
        score(single, rng),
        score(multi, rng),
        contrast(single, multi, "single", "multi", rng),
    )


def _runs() -> list:  # type: ignore[no-untyped-def]
    """Runs that can answer the question: all five AI folds, with predictions.

    Globbing `fold_*/predictions.json` is not enough. `build_rows` resolves the
    five AI folds through an explicit map and raises on a partial run rather
    than silently reporting a delta over whichever folds are on disk -- so a
    directory holding two of them is not a smaller sample, it is not a sample.
    """
    from scripts.analysis.audit_mechanism_probe import FOLD_DIRS

    from tract.config import FOLD_RESULT_FILENAME

    if not PHASE1B_RESULTS_DIR.is_dir():
        return []
    return [
        d
        for d in sorted(PHASE1B_RESULTS_DIR.iterdir())
        if d.is_dir()
        and all(
            (d / fold / name).is_file()
            for fold in FOLD_DIRS.values()
            for name in ("predictions.json", FOLD_RESULT_FILENAME)
        )
    ]


class TestTheCompositionHalfHolds:
    def test_incumbent_multilabel_share_is_the_stated_8_8_percent(self) -> None:
        from scripts.phase0.common import (
            AI_FRAMEWORK_NAMES,
            build_evaluation_corpus,
            load_curated_links,
        )

        logging.disable(logging.INFO)
        try:
            corpus = build_evaluation_corpus(
                load_curated_links(), AI_FRAMEWORK_NAMES, {}
            )
        finally:
            logging.disable(logging.NOTSET)
        multi = sum(1 for i in corpus if len(i.valid_hub_ids) > 1)
        assert (multi, len(corpus)) == (13, 147)
        assert round(100 * multi / len(corpus), 1) == 8.8


class TestTheDeltaHalfDoesNotReproduce:
    @pytest.mark.parametrize("run_dir", _runs(), ids=lambda d: d.name)
    def test_multilabel_delta_is_never_negative(self, run_dir) -> None:  # type: ignore[no-untyped-def]
        """The claim's sign, checked on every run that can answer it."""
        _, multi, _ = _split(run_dir)
        assert multi["delta_mean"] >= 0.0, (
            f"{run_dir.name}: multi-label delta {multi['delta_mean']:+.4f}. If "
            "a run now reproduces B3's negative split, re-open remediation "
            "item 10 rather than deleting this test."
        )

    @pytest.mark.parametrize("run_dir", _runs(), ids=lambda d: d.name)
    def test_no_run_reaches_the_claimed_significance(self, run_dir) -> None:  # type: ignore[no-untyped-def]
        _, _, c = _split(run_dir)
        assert c["p_difference_le_zero"] > CLAIMED_P, (
            f"{run_dir.name}: P(<=0)={c['p_difference_le_zero']:.3f}, at or "
            f"below the claimed {CLAIMED_P}. B3's establishment claim would "
            "then hold on this run and the correction needs revisiting."
        )

    def test_at_least_one_run_was_actually_checked(self) -> None:
        """A parametrize over an empty list is a silent pass."""
        assert _runs(), (
            "No run directory carries fold_*/predictions.json, so both "
            "parametrized tests above collapsed to nothing."
        )


class TestTheSurvivingConcernIsStatedWithoutDirection:
    def test_multilabel_items_do_differ_from_single_label_items(self) -> None:
        """What justifies stratifying: they behave differently, sign aside.

        Measured on the campaign's own test run. The magnitude of the gap is
        what makes composition worth disclosing; its sign is not stable across
        runs and must not be predicted.
        """
        target = PHASE1B_RESULTS_DIR / "c3_TEST_A3_prose_sw_qwen06b_seq1024"
        if not target.is_dir():
            pytest.skip(f"{target} absent")
        single, multi, _ = _split(target)
        assert abs(single["delta_mean"] - multi["delta_mean"]) > 0.10
