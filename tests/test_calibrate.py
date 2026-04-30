"""Tests for temperature scaling calibration."""
from __future__ import annotations

import numpy as np
import pytest


class TestTemperatureScaling:
    """Test optimal temperature finding."""

    def test_finds_reasonable_temperature(self) -> None:
        from tract.training.calibrate import find_optimal_temperature
        rng = np.random.default_rng(42)
        n_samples, n_hubs = 20, 50
        similarities = rng.uniform(-0.5, 0.5, size=(n_samples, n_hubs))
        ground_truth_indices = rng.integers(0, n_hubs, size=n_samples)
        for i in range(n_samples):
            similarities[i, ground_truth_indices[i]] += 0.5

        temp = find_optimal_temperature(similarities, ground_truth_indices)
        assert 0.01 <= temp <= 5.0

    def test_calibrated_probabilities_sum_to_one(self) -> None:
        from tract.training.calibrate import calibrate_similarities
        sims = np.array([[0.3, 0.7, 0.1], [0.9, 0.2, 0.4]])
        probs = calibrate_similarities(sims, temperature=1.0)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_temperature_affects_sharpness(self) -> None:
        from tract.training.calibrate import calibrate_similarities
        sims = np.array([[0.1, 0.5, 0.2]])
        probs_sharp = calibrate_similarities(sims, temperature=0.1)
        probs_smooth = calibrate_similarities(sims, temperature=5.0)
        assert probs_sharp[0, 1] > probs_smooth[0, 1]


class TestGlobalThreshold:
    """Test global threshold at max-F1."""

    def test_threshold_in_valid_range(self) -> None:
        from tract.training.calibrate import find_global_threshold
        rng = np.random.default_rng(42)
        n_samples, n_hubs = 30, 10
        similarities = rng.uniform(0, 1, size=(n_samples, n_hubs))
        ground_truth_indices = rng.integers(0, n_hubs, size=n_samples)
        for i in range(n_samples):
            similarities[i, ground_truth_indices[i]] += 0.3

        threshold = find_global_threshold(similarities, ground_truth_indices, temperature=1.0)
        assert 0.0 < threshold < 1.0
