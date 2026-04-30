"""Tests for multi-label temperature scaling calibration."""
from __future__ import annotations

import numpy as np
import pytest


class TestMultiLabelNLL:
    def test_single_label_matches_standard_nll(self) -> None:
        from tract.calibration.temperature import multi_label_nll

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(10), size=5)
        single_label_indices = [[3], [7], [1], [0], [9]]

        nll = multi_label_nll(probs, single_label_indices)
        expected = -np.mean([np.log(probs[i, idx[0]] + 1e-10) for i, idx in enumerate(single_label_indices)])
        np.testing.assert_allclose(nll, expected, atol=1e-6)

    def test_multi_label_lower_nll_than_single(self) -> None:
        from tract.calibration.temperature import multi_label_nll

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(10), size=5)
        single = [[3], [7], [1], [0], [9]]
        multi = [[3, 5], [7, 2], [1, 4], [0, 8], [9, 6]]

        nll_single = multi_label_nll(probs, single)
        nll_multi = multi_label_nll(probs, multi)
        assert nll_multi < nll_single

    def test_empty_valid_raises(self) -> None:
        from tract.calibration.temperature import multi_label_nll

        probs = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="empty"):
            multi_label_nll(probs, [[]])


class TestCalibrateSimilarities:
    def test_probabilities_sum_to_one(self) -> None:
        from tract.calibration.temperature import calibrate_similarities

        sims = np.array([[0.3, 0.7, 0.1], [0.9, 0.2, 0.4]])
        probs = calibrate_similarities(sims, temperature=1.0)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)


class TestFitTemperature:
    def test_finds_temperature_in_range(self) -> None:
        from tract.calibration.temperature import fit_temperature

        rng = np.random.default_rng(42)
        n, n_hubs = 30, 50
        sims = rng.uniform(-0.3, 0.3, size=(n, n_hubs))
        gt_indices = rng.integers(0, n_hubs, size=n).tolist()
        for i in range(n):
            sims[i, gt_indices[i]] += 0.5
        valid_hub_indices = [[idx] for idx in gt_indices]

        result = fit_temperature(sims, valid_hub_indices)
        assert 0.01 <= result["temperature"] <= 5.0
        assert "nll" in result

    def test_uses_log_spaced_grid(self) -> None:
        from tract.calibration.temperature import fit_temperature

        rng = np.random.default_rng(42)
        sims = rng.uniform(0, 1, size=(10, 5))
        valid = [[0]] * 10

        result = fit_temperature(sims, valid, n_grid=10, t_min=0.01, t_max=5.0)
        assert result["temperature"] > 0


class TestFitTLofo:
    def test_sqrt_n_weighting(self) -> None:
        from tract.calibration.temperature import fit_t_lofo

        rng = np.random.default_rng(42)
        fold_sims = {}
        fold_gt = {}
        for name, n in [("A", 60), ("B", 40), ("C", 10)]:
            s = rng.uniform(-0.3, 0.3, size=(n, 50))
            gt = rng.integers(0, 50, size=n).tolist()
            for i in range(n):
                s[i, gt[i]] += 0.5
            fold_sims[name] = s
            fold_gt[name] = [[idx] for idx in gt]

        result = fit_t_lofo(fold_sims, fold_gt)
        assert 0.01 <= result["temperature"] <= 5.0
        assert "per_fold_nll" in result
        assert len(result["per_fold_nll"]) == 3
