"""Tests for conformal prediction sets."""
from __future__ import annotations

import math

import numpy as np
import pytest


class TestConformalQuantile:
    def test_quantile_in_valid_range(self) -> None:
        from tract.calibration.conformal import compute_conformal_quantile

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(50), size=30)
        valid = [[rng.integers(0, 50)] for _ in range(30)]

        q = compute_conformal_quantile(probs, valid, alpha=0.10)
        assert 0.0 <= q <= 1.0

    def test_higher_alpha_yields_lower_quantile(self) -> None:
        from tract.calibration.conformal import compute_conformal_quantile

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(20), size=50)
        valid = [[rng.integers(0, 20)] for _ in range(50)]

        q_strict = compute_conformal_quantile(probs, valid, alpha=0.05)
        q_loose = compute_conformal_quantile(probs, valid, alpha=0.20)
        assert q_strict >= q_loose


class TestPredictionSets:
    def test_prediction_set_covers_ground_truth(self) -> None:
        from tract.calibration.conformal import build_prediction_sets

        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        hub_ids = ["h1", "h2", "h3"]

        sets = build_prediction_sets(probs, hub_ids, quantile=0.5)
        assert len(sets) == 2
        for s in sets:
            assert isinstance(s, set)


class TestConformalCoverage:
    def test_empirical_coverage(self) -> None:
        from tract.calibration.conformal import compute_conformal_coverage

        prediction_sets = [{"h1", "h2"}, {"h2", "h3"}, {"h1"}]
        valid_hub_sets = [frozenset({"h1"}), frozenset({"h2"}), frozenset({"h3"})]

        coverage = compute_conformal_coverage(prediction_sets, valid_hub_sets)
        assert coverage == pytest.approx(2 / 3)
