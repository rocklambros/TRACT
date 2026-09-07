"""A stratum's interval must be a property of that stratum, not of call order.

`gate_rule_candidates` had this defect, it was measured (+0.4595 printed where
the 500k reference is +0.4324), and it was fixed there with content-hashed
per-stratum seeding. `audit_mechanism_probe` -- which produced the published
audit-stratified figures in docs/campaign3-audit-mechanism.md -- kept the shape
the fix was written for: `main()` builds ONE generator and threads it through
every `score()` and `contrast()` call in sequence.

So each stratum's draws depend on how many draws every prior stratum consumed.
Reordering two log lines moves a published interval. These tests pin the
property that makes that impossible.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.audit_mechanism_probe import (
    N_RESAMPLES,
    ProbeRow,
    contrast,
    delta_distribution,
    score,
)


def _rows(n: int, fold: str, hit_pattern: tuple[int, ...]) -> list[ProbeRow]:
    """Deterministic rows; content differs per fold so the hash differs."""
    return [
        ProbeRow(
            framework="F",
            fold_dir=fold,
            section=f"{fold}-{i}",
            trained_hit1=hit_pattern[i % len(hit_pattern)],
            zero_shot_hit1=(i % 2),
            audit_touched=False,
            gold_degree_max=3,
            gold_degree_primary=3,
            n_valid_hubs=2,
            verdict=None,
            degree_change=None,
        )
        for i in range(n)
    ]


STRATUM_A = _rows(40, "fold_a", (1, 1, 0, 1))
STRATUM_B = _rows(35, "fold_b", (0, 1, 0, 0))


class TestScoreIsOrderIndependent:
    """Assert on the whole resample distribution, not two percentiles of it.

    A first version of these three compared `ci_low`/`ci_high` and PASSED
    against the unfixed code. On a binary fixture the 2.5th percentile is
    discrete: it lands on the same value whichever stream produced it, so the
    comparison cannot discriminate a shared generator from a per-stratum one.
    Comparing the full array is what makes these tests capable of failing.
    """

    def test_same_stratum_draws_identically_regardless_of_what_ran_before(
        self,
    ) -> None:
        """Draw A first; then draw A again after B has consumed from the stream."""
        _, first = delta_distribution(STRATUM_A, np.random.default_rng(42))

        rng = np.random.default_rng(42)
        delta_distribution(STRATUM_B, rng)
        _, after_b = delta_distribution(STRATUM_A, rng)

        np.testing.assert_array_equal(after_b, first)

    def test_a_stratum_is_not_perturbed_by_an_unrelated_upstream_draw(
        self,
    ) -> None:
        """An unrelated draw off the parent must not move a published interval."""
        _, clean = delta_distribution(STRATUM_A, np.random.default_rng(42))

        polluted_rng = np.random.default_rng(42)
        polluted_rng.integers(0, 100, 7919)
        _, polluted = delta_distribution(STRATUM_A, polluted_rng)

        np.testing.assert_array_equal(polluted, clean)

    def test_different_strata_still_get_different_streams(self) -> None:
        """Content-derived seeding must not collapse to one shared stream.

        If every stratum drew the same numbers, the contrast between two
        strata would lose its independent-resampling meaning.
        """
        _, a = delta_distribution(STRATUM_A, np.random.default_rng(42))
        _, b = delta_distribution(STRATUM_B, np.random.default_rng(42))
        assert not np.array_equal(a, b)

    def test_score_carries_the_property_through_to_its_percentiles(self) -> None:
        """The exported entry point inherits it -- what callers actually use."""
        first = score(STRATUM_A, np.random.default_rng(42))
        rng = np.random.default_rng(42)
        score(STRATUM_B, rng)
        after_b = score(STRATUM_A, rng)

        assert after_b["ci_low"] == pytest.approx(first["ci_low"])
        assert after_b["ci_high"] == pytest.approx(first["ci_high"])


class TestContrastIsOrderIndependent:
    def test_contrast_is_stable_across_repeated_independent_calls(self) -> None:
        first = contrast(STRATUM_A, STRATUM_B, "a", "b", np.random.default_rng(42))

        rng = np.random.default_rng(42)
        contrast(STRATUM_B, STRATUM_A, "b", "a", rng)
        again = contrast(STRATUM_A, STRATUM_B, "a", "b", rng)

        assert again["ci_low"] == pytest.approx(first["ci_low"])
        assert again["ci_high"] == pytest.approx(first["ci_high"])
        assert again["p_difference_le_zero"] == pytest.approx(
            first["p_difference_le_zero"]
        )

    def test_swapping_the_arguments_negates_the_difference(self) -> None:
        """a-b and b-a must be exact mirrors, which needs per-stratum streams.

        With one threaded generator the two calls draw different numbers for
        the same stratum, so the mirror only held approximately -- and how
        closely depended on the resample count.
        """
        ab = contrast(STRATUM_A, STRATUM_B, "a", "b", np.random.default_rng(42))
        ba = contrast(STRATUM_B, STRATUM_A, "b", "a", np.random.default_rng(42))

        assert ba["difference"] == pytest.approx(-ab["difference"])
        assert ba["ci_low"] == pytest.approx(-ab["ci_high"])
        assert ba["ci_high"] == pytest.approx(-ab["ci_low"])


class TestResampleCountMatchesTheSiblingModule:
    def test_resamples_are_not_left_at_the_count_that_produced_the_artifact(
        self,
    ) -> None:
        """10,000 is where the sibling module's artifact was measured.

        At 10k, 11 of 12 seeds reproduced the reference contrast; at 100k,
        12 of 12 did. The probe publishes intervals into a results document,
        so it takes the count that was shown to be stable.
        """
        assert N_RESAMPLES >= 100_000
