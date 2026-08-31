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
        folds = simulate_study(7, 40, 0.15, 0.05, DISCORDANT_RATE,
                               np.random.default_rng(0))
        assert len(folds) == 7
        assert all(len(f) == 40 for f in folds)

    def test_items_are_mcnemar_cells(self) -> None:
        folds = simulate_study(3, 200, 0.15, 0.05, DISCORDANT_RATE,
                               np.random.default_rng(0))
        values = set(np.concatenate(folds).tolist())
        assert values <= {-1.0, 0.0, 1.0}

    def test_grand_mean_recovers_mu(self) -> None:
        # Many frameworks, many items: the pooled mean must land on mu.
        folds = simulate_study(200, 400, 0.15, 0.05, DISCORDANT_RATE,
                               np.random.default_rng(1))
        assert np.concatenate(folds).mean() == pytest.approx(0.15, abs=0.01)

    def test_discordant_rate_is_preserved(self) -> None:
        folds = simulate_study(100, 400, 0.15, 0.02, DISCORDANT_RATE,
                               np.random.default_rng(2))
        flat = np.concatenate(folds)
        assert float((flat != 0.0).mean()) == pytest.approx(
            DISCORDANT_RATE, abs=0.02)

    def test_tau_zero_makes_frameworks_identical_in_expectation(self) -> None:
        folds = simulate_study(60, 500, 0.15, 0.0, DISCORDANT_RATE,
                               np.random.default_rng(3))
        spread = float(np.std([f.mean() for f in folds]))
        # With tau=0 the only spread is within-framework sampling noise,
        # ~sqrt(0.29/500) ~ 0.024. Anything near tau=0.10 would fail this.
        assert spread < 0.05

    def test_larger_tau_produces_more_spread_between_frameworks(self) -> None:
        rng = np.random.default_rng(4)
        low = simulate_study(60, 500, 0.15, 0.02, DISCORDANT_RATE, rng)
        high = simulate_study(60, 500, 0.15, 0.20, DISCORDANT_RATE, rng)
        assert np.std([f.mean() for f in high]) > \
            np.std([f.mean() for f in low])

    def test_is_deterministic_for_a_seed(self) -> None:
        a = simulate_study(5, 30, 0.15, 0.08, DISCORDANT_RATE,
                           np.random.default_rng(9))
        b = simulate_study(5, 30, 0.15, 0.08, DISCORDANT_RATE,
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
        tight_pass = sum(
            cluster_bootstrap_pass(tight, 300, rng) for _ in range(5))
        spread_pass = sum(
            cluster_bootstrap_pass(spread, 300, rng) for _ in range(5))
        assert tight_pass >= spread_pass


class TestCalibration:
    def test_power_at_the_threshold_is_near_alpha(self) -> None:
        """The headline validity check: a true effect exactly at the gate
        threshold must pass at roughly the nominal rate, not more."""
        power = power_at(
            k=9, n_per=68, mu=GATE_THRESHOLD, tau=0.05,
            n_studies=200, n_bootstrap=400, rng=np.random.default_rng(11),
        )
        assert power < 3 * GATE_ALPHA, (
            f"gate fired at {power:.1%} when the true effect sits exactly on "
            f"the {GATE_THRESHOLD} threshold; the rule is anti-conservative"
        )

    def test_power_rises_with_the_true_effect(self) -> None:
        rng = np.random.default_rng(12)
        low = power_at(9, 68, 0.15, 0.05, 150, 400, rng)
        high = power_at(9, 68, 0.25, 0.05, 150, 400, rng)
        assert high > low

    def test_power_falls_as_between_framework_spread_grows(self) -> None:
        rng = np.random.default_rng(13)
        tight = power_at(9, 68, 0.20, 0.02, 150, 400, rng)
        loose = power_at(9, 68, 0.20, 0.20, 150, 400, rng)
        assert tight > loose
