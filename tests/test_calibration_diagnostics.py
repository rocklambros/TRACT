"""Tests for calibration diagnostic metrics."""
from __future__ import annotations

import numpy as np
import pytest


class TestECE:
    def test_perfect_calibration_ece_near_zero(self) -> None:
        from tract.calibration.diagnostics import expected_calibration_error

        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        accuracies = np.array([0, 0, 1, 1, 1] * 20)
        ece = expected_calibration_error(confidences, accuracies, n_bins=5)
        assert ece < 0.30

    def test_ece_in_valid_range(self) -> None:
        from tract.calibration.diagnostics import expected_calibration_error

        rng = np.random.default_rng(42)
        confidences = rng.uniform(0, 1, size=100)
        accuracies = rng.integers(0, 2, size=100).astype(float)
        ece = expected_calibration_error(confidences, accuracies, n_bins=5)
        assert 0.0 <= ece <= 1.0


class TestBootstrapECE:
    def test_returns_ci(self) -> None:
        from tract.calibration.diagnostics import bootstrap_ece

        rng = np.random.default_rng(42)
        confidences = rng.uniform(0, 1, size=50)
        accuracies = rng.integers(0, 2, size=50).astype(float)

        result = bootstrap_ece(confidences, accuracies, n_bins=5, n_bootstrap=100, seed=42)
        assert "ece" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert result["ci_low"] <= result["ece"] <= result["ci_high"]


class TestKSTest:
    def test_identical_distributions_high_p(self) -> None:
        from tract.calibration.diagnostics import ks_test_similarity_distributions

        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, size=100)
        b = rng.uniform(0, 1, size=100)

        result = ks_test_similarity_distributions(a, b)
        assert result["p_value"] > 0.01

    def test_different_distributions_low_p(self) -> None:
        from tract.calibration.diagnostics import ks_test_similarity_distributions

        a = np.ones(100) * 0.9
        b = np.ones(100) * 0.1

        result = ks_test_similarity_distributions(a, b)
        assert result["p_value"] < 0.01


class TestFullRecallCoverage:
    def test_all_hubs_covered(self) -> None:
        from tract.calibration.diagnostics import full_recall_coverage

        prediction_sets = [{"h1", "h2"}, {"h3"}]
        valid_hub_sets = [frozenset({"h1", "h2"}), frozenset({"h3"})]
        assert full_recall_coverage(prediction_sets, valid_hub_sets) == 1.0

    def test_partial_coverage(self) -> None:
        from tract.calibration.diagnostics import full_recall_coverage

        prediction_sets = [{"h1"}, {"h3"}]
        valid_hub_sets = [frozenset({"h1", "h2"}), frozenset({"h3"})]
        assert full_recall_coverage(prediction_sets, valid_hub_sets) == 0.5
