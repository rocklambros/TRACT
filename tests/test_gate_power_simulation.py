"""Tests for the gate power simulation.

The simulation's credibility rests on two things a reader cannot check by
eye: that the simulated study really has the mean delta it was asked for, and
that the gate rule is calibrated -- it must fire at ~alpha when the true effect
sits exactly on the threshold. Both are tested here. Everything runs on
synthetic draws, so no corpus or result directory is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.gate_power_simulation import (
    DISCORDANT_RATE,
    GATE_ALPHA,
    GATE_THRESHOLD,
    cluster_bootstrap_pass,
    power_at,
    simulate_study,
)


class TestSimulateStudy:
    def test_returns_k_frameworks_of_n_items(self) -> None:
        folds, _ = simulate_study([40] * 7, 0.15, 0.05, DISCORDANT_RATE,
                               np.random.default_rng(0))
        assert len(folds) == 7
        assert all(len(f) == 40 for f in folds)

    def test_items_are_mcnemar_cells(self) -> None:
        folds, _ = simulate_study([200] * 3, 0.15, 0.05, DISCORDANT_RATE,
                               np.random.default_rng(0))
        values = set(np.concatenate(folds).tolist())
        assert values <= {-1.0, 0.0, 1.0}

    def test_grand_mean_recovers_mu(self) -> None:
        # Many frameworks, many items: the pooled mean must land on mu.
        folds, _ = simulate_study([400] * 200, 0.15, 0.05, DISCORDANT_RATE,
                               np.random.default_rng(1))
        assert np.concatenate(folds).mean() == pytest.approx(0.15, abs=0.01)

    def test_discordant_rate_is_preserved(self) -> None:
        folds, _ = simulate_study([400] * 100, 0.15, 0.02, DISCORDANT_RATE,
                               np.random.default_rng(2))
        flat = np.concatenate(folds)
        assert float((flat != 0.0).mean()) == pytest.approx(
            DISCORDANT_RATE, abs=0.02)

    def test_tau_zero_makes_frameworks_identical_in_expectation(self) -> None:
        folds, _ = simulate_study([500] * 60, 0.15, 0.0, DISCORDANT_RATE,
                               np.random.default_rng(3))
        spread = float(np.std([f.mean() for f in folds]))
        # With tau=0 the only spread is within-framework sampling noise,
        # ~sqrt(0.29/500) ~ 0.024. Anything near tau=0.10 would fail this.
        assert spread < 0.05

    def test_larger_tau_produces_more_spread_between_frameworks(self) -> None:
        rng = np.random.default_rng(4)
        low, _ = simulate_study([500] * 60, 0.15, 0.02, DISCORDANT_RATE, rng)
        high, _ = simulate_study([500] * 60, 0.15, 0.20, DISCORDANT_RATE, rng)
        assert np.std([f.mean() for f in high]) > \
            np.std([f.mean() for f in low])

    def test_is_deterministic_for_a_seed(self) -> None:
        a = simulate_study([30] * 5, 0.15, 0.08, DISCORDANT_RATE,
                           np.random.default_rng(9))
        b = simulate_study([30] * 5, 0.15, 0.08, DISCORDANT_RATE,
                           np.random.default_rng(9))
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)


class TestClusterBootstrapPass:
    def test_an_overwhelming_effect_passes(self) -> None:
        folds = [np.ones(50) for _ in range(9)]
        assert cluster_bootstrap_pass(folds, 400, np.random.default_rng(0))

    def test_a_zero_effect_fails(self) -> None:
        folds = [np.zeros(50) for _ in range(9)]
        assert not cluster_bootstrap_pass(folds, 400, np.random.default_rng(0))

    def test_an_effect_exactly_at_the_threshold_fails(self) -> None:
        # Every fold sits exactly on 0.10, so P(delta <= 0.10) is ~1.
        folds = [np.array([1.0] * 5 + [0.0] * 45) for _ in range(9)]
        assert not cluster_bootstrap_pass(folds, 400, np.random.default_rng(0))

    def test_between_framework_spread_can_sink_a_high_mean(self) -> None:
        # Same grand mean either way, but the spread-out version must be
        # harder to pass -- that is the entire point of clustering.
        tight = [np.array([1.0] * 20 + [0.0] * 30) for _ in range(9)]
        spread = [np.ones(50) if i < 4 else np.zeros(50) for i in range(9)]
        rng = np.random.default_rng(0)
        # Compare the pass PROBABILITY, not a 5-trial count. The count
        # version was satisfied by 5 >= 5 and passed with cluster resampling
        # deleted outright.
        from scripts.analysis.gate_power_simulation import _pass_probability
        tight_p = _pass_probability(tight, 2000, rng)
        spread_p = _pass_probability(spread, 2000, rng)
        assert spread_p > tight_p, (
            "a spread-out set of frameworks must be HARDER to pass; "
            f"tight P(delta<=gate)={tight_p:.3f} spread={spread_p:.3f}")


class TestCalibration:
    def test_power_at_the_threshold_is_near_alpha(self) -> None:
        """The headline validity check: a true effect exactly at the gate
        threshold must pass at roughly the nominal rate, not more."""
        power, _ = power_at(
            [68] * 9, GATE_THRESHOLD, 0.05,
            n_studies=200, n_bootstrap=400, rng=np.random.default_rng(11),
        )
        assert power < 3 * GATE_ALPHA, (
            f"gate fired at {power:.1%} when the true effect sits exactly on "
            f"the {GATE_THRESHOLD} threshold; the rule is anti-conservative"
        )

    def test_power_rises_with_the_true_effect(self) -> None:
        rng = np.random.default_rng(12)
        low, _ = power_at([68] * 9, 0.15, 0.05, 150, 400, rng)
        high, _ = power_at([68] * 9, 0.25, 0.05, 150, 400, rng)
        assert high > low

    def test_power_falls_as_between_framework_spread_grows(self) -> None:
        rng = np.random.default_rng(13)
        tight, _ = power_at([68] * 9, 0.20, 0.02, 150, 400, rng)
        loose, _ = power_at([68] * 9, 0.20, 0.20, 150, 400, rng)
        assert tight > loose


class TestMeasuresTheGatesOwnEstimand:
    """The simulation used equal n per fold, which makes it a MACRO average.

    The gate reports the item-weighted MICRO average. On the real primary the
    two differ by 0.1701 -- 1.7x the gate threshold: micro +0.1000 (the
    published figure), macro +0.2701. A power surface computed on the macro
    statistic sizes a design for a number nobody reports.
    """

    def test_pooled_delta_is_item_weighted(self) -> None:
        from scripts.analysis.gate_power_simulation import pooled_delta
        # 90 items at 0, 10 items at 1. micro = 0.10; macro would be 0.50.
        folds = [np.zeros(90), np.ones(10)]
        assert pooled_delta(folds) == pytest.approx(0.10)

    def test_unequal_fold_sizes_are_honoured(self) -> None:
        from scripts.analysis.gate_power_simulation import simulate_study
        folds, _ = simulate_study([63, 30, 11, 4, 2], 0.15, 0.05,
                               DISCORDANT_RATE, np.random.default_rng(0))
        assert [len(f) for f in folds] == [63, 30, 11, 4, 2]

    def test_a_big_fold_dominates_the_pooled_estimate(self) -> None:
        from scripts.analysis.gate_power_simulation import pooled_delta
        # The real design is 63/30/11/4/2: OWASP AI Exchange alone is 57% of it.
        big, small = np.zeros(63), np.ones(2)
        assert pooled_delta([big, small]) < 0.05


class TestTauGridCoversTheCorpusEstimate:
    """The two clamp tests that lived here are superseded.

    They asserted against `realised_parameters`, which measured realised fold
    means and was wrong in both directions -- see
    TestDiagnosticsMeasureTheDrawNotTheOutcome, which replaces them with
    assertions against the drawn deltas, where the clamp actually binds.
    """

    def test_the_grid_covers_the_corpuss_own_tau_estimate(self) -> None:
        from scripts.analysis.gate_power_simulation import TAU_GRID
        # docs/campaign3-audit-mechanism.md 6e: tau = 0.3702 over all five
        # folds. A grid stopping at 0.20 never simulates the branch that says
        # the instrument should be replaced rather than re-powered.
        assert max(TAU_GRID) >= 0.37


class TestClusterResamplingActuallyHappens:
    """Deleting framework resampling left the previous suite entirely green."""

    def test_removing_framework_resampling_changes_the_result(self) -> None:
        from scripts.analysis.gate_power_simulation import _pass_probability
        # Every fold is internally CONSTANT, so resampling items within a fold
        # cannot move its mean at all -- the item bootstrap has exactly zero
        # variance here. All the uncertainty is between frameworks: one fold at
        # 1.0 and eight at 0.0 give a pooled 40/360 = 0.111, just above the
        # gate, and omitting the one fold that matters drops it to 0.0.
        #
        # This is the fixture the previous version needed and did not have.
        # With mixed folds, item resampling supplies enough spread on its own to
        # mask whether the cluster layer is running at all.
        folds = [np.ones(40)] + [np.zeros(40) for _ in range(8)]
        rng = np.random.default_rng(0)
        with_clusters = _pass_probability(folds, 2000, rng,
                                          resample_frameworks=True)
        without = _pass_probability(folds, 2000, rng,
                                    resample_frameworks=False)
        assert abs(with_clusters - without) > 0.10, (
            "cluster resampling made no difference -- the estimator is an "
            "item bootstrap wearing the name of a cluster bootstrap"
        )


class TestDiagnosticsMeasureTheDrawNotTheOutcome:
    """`realised_parameters` measured fold MEANS, which is not what it claimed.

    `clamped_fraction` counted realised fold means above the discordant rate.
    The clamp binds on the DRAWN Normal(mu, tau) delta, before any items exist.
    At the observed fold sizes -- two folds of n=2 and n=4 -- a fold mean clears
    0.30 by sampling noise alone, so at mu=0.10, tau=0.00 it reported 0.148
    where the true clamp probability is exactly 0.000.

    `tau` had the mirror defect: the SD of observed fold means is
    sqrt(tau^2 + within-fold noise), so it read ~0.20 when the true tau was 0.

    Both were written into results/analysis/power_surface.json and cited in
    docs/campaign3-audit-mechanism.md as "the clamp's effect reported". A reader
    would conclude the tau=0 row already delivered tau ~ 0.20 and that the
    design tolerates a tau the surface says it does not -- the round-1 pattern
    of a diagnostic that flatters the design, inside the fix for it.
    """

    def test_no_clamping_is_reported_when_the_clamp_cannot_bind(self) -> None:
        from scripts.analysis.gate_power_simulation import (DISCORDANT_RATE,
                                                            OBSERVED_FOLD_SIZES,
                                                            simulate_study)
        # mu well inside the discordant rate, tau = 0: the drawn delta is
        # always 0.10, so the clamp can never bind.
        _, diag = simulate_study(OBSERVED_FOLD_SIZES, 0.10, 0.0,
                                 DISCORDANT_RATE, np.random.default_rng(0))
        assert diag["clamped_fraction"] == 0.0

    def test_clamping_is_reported_when_it_does_bind(self) -> None:
        from scripts.analysis.gate_power_simulation import (DISCORDANT_RATE,
                                                            OBSERVED_FOLD_SIZES,
                                                            simulate_study)
        # mu above the discordant rate with tau = 0: it always binds.
        _, diag = simulate_study(OBSERVED_FOLD_SIZES, 0.50, 0.0,
                                 DISCORDANT_RATE, np.random.default_rng(0))
        assert diag["clamped_fraction"] == 1.0

    def test_drawn_tau_is_the_generating_tau_not_the_observed_spread(
        self,
    ) -> None:
        from scripts.analysis.gate_power_simulation import (DISCORDANT_RATE,
                                                            OBSERVED_FOLD_SIZES,
                                                            simulate_study)
        _, diag = simulate_study(OBSERVED_FOLD_SIZES, 0.10, 0.0,
                                 DISCORDANT_RATE, np.random.default_rng(0))
        # True tau is 0. The observed SD of fold means at these sizes is ~0.20.
        assert diag["drawn_tau"] < 0.05

    def test_fold_mean_spread_is_reported_separately_and_is_larger(
        self,
    ) -> None:
        from scripts.analysis.gate_power_simulation import (DISCORDANT_RATE,
                                                            OBSERVED_FOLD_SIZES,
                                                            simulate_study)
        # Reported, but never as "tau": a reader needs to know the observed
        # spread is dominated by within-fold noise at these sizes.
        _, diag = simulate_study(OBSERVED_FOLD_SIZES, 0.10, 0.0,
                                 DISCORDANT_RATE, np.random.default_rng(0))
        assert diag["fold_mean_spread"] > diag["drawn_tau"]
