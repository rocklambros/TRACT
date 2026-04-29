"""Tests for LOFO evaluation harness."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class FakeEvalItem:
    control_text: str
    ground_truth_hub_id: str
    valid_hub_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.valid_hub_ids is None:
            self.valid_hub_ids = frozenset({self.ground_truth_hub_id})


class TestRankHubsBySimilarity:
    """Test cosine similarity ranking."""

    def test_returns_ranked_predictions(self) -> None:
        from tract.training.evaluate import rank_hubs_by_similarity

        query_emb = np.array([0.5, 0.5, 0.0])
        hub_embs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.4, 0.0],
        ])
        hub_ids = ["h1", "h2", "h3"]
        ranked = rank_hubs_by_similarity(query_emb, hub_embs, hub_ids)
        assert ranked[0] == "h3"
        assert len(ranked) == 3

    def test_exact_match_first(self) -> None:
        from tract.training.evaluate import rank_hubs_by_similarity

        query = np.array([1.0, 0.0])
        hubs = np.array([[1.0, 0.0], [0.0, 1.0]])
        ranked = rank_hubs_by_similarity(query, hubs, ["a", "b"])
        assert ranked[0] == "a"

    def test_rejects_2d_query(self) -> None:
        from tract.training.evaluate import rank_hubs_by_similarity

        with pytest.raises(ValueError, match="1-D"):
            rank_hubs_by_similarity(
                np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), ["a"]
            )

    def test_rejects_mismatched_hub_count(self) -> None:
        from tract.training.evaluate import rank_hubs_by_similarity

        with pytest.raises(ValueError, match="hub_ids length"):
            rank_hubs_by_similarity(
                np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), ["a", "b"]
            )


class TestFoldStratifiedBootstrap:
    """Test fold-stratified micro-average bootstrap CIs."""

    def test_preserves_fold_sizes(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        fold_hit1s = [
            np.array([1.0, 0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 1.0]),
        ]
        result = fold_stratified_bootstrap_ci(fold_hit1s, n_resamples=1000, seed=42)
        assert "mean" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert "n_total" in result
        assert result["n_total"] == 8
        assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    def test_perfect_scores_give_tight_ci(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        fold_hit1s = [np.ones(10), np.ones(5)]
        result = fold_stratified_bootstrap_ci(fold_hit1s, n_resamples=1000, seed=42)
        assert result["mean"] == 1.0
        assert result["ci_low"] == 1.0

    def test_zero_scores_give_tight_ci(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        fold_hit1s = [np.zeros(10), np.zeros(5)]
        result = fold_stratified_bootstrap_ci(fold_hit1s, n_resamples=1000, seed=42)
        assert result["mean"] == 0.0
        assert result["ci_high"] == 0.0

    def test_deterministic_with_same_seed(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        folds = [np.array([1.0, 0.0, 1.0, 0.0, 1.0])]
        r1 = fold_stratified_bootstrap_ci(folds, n_resamples=500, seed=99)
        r2 = fold_stratified_bootstrap_ci(folds, n_resamples=500, seed=99)
        assert r1["ci_low"] == r2["ci_low"]
        assert r1["ci_high"] == r2["ci_high"]

    def test_different_seeds_differ(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        rng = np.random.default_rng(42)
        folds = [rng.uniform(0, 1, size=50)]
        r1 = fold_stratified_bootstrap_ci(folds, n_resamples=500, seed=1)
        r2 = fold_stratified_bootstrap_ci(folds, n_resamples=500, seed=2)
        assert r1["ci_low"] != r2["ci_low"] or r1["ci_high"] != r2["ci_high"]

    def test_rejects_empty_fold_list(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        with pytest.raises(ValueError, match="non-empty"):
            fold_stratified_bootstrap_ci([], n_resamples=100, seed=42)

    def test_rejects_empty_fold(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        with pytest.raises(ValueError, match="empty"):
            fold_stratified_bootstrap_ci(
                [np.array([1.0]), np.array([])], n_resamples=100, seed=42
            )

    def test_realistic_fold_sizes(self) -> None:
        """Test with LOFO-realistic fold sizes: 65/45/64/13/10 = 197."""
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        rng = np.random.default_rng(42)
        folds = [
            rng.binomial(1, 0.5, size=65).astype(float),
            rng.binomial(1, 0.4, size=45).astype(float),
            rng.binomial(1, 0.6, size=64).astype(float),
            rng.binomial(1, 0.35, size=13).astype(float),
            rng.binomial(1, 0.55, size=10).astype(float),
        ]
        result = fold_stratified_bootstrap_ci(folds, n_resamples=10_000, seed=42)
        assert result["n_total"] == 197
        assert 0.0 < result["ci_low"] < result["ci_high"] < 1.0

    def test_vectorized_speed(self) -> None:
        """10K resamples on 197 items should complete in under 1 second."""
        from tract.training.evaluate import fold_stratified_bootstrap_ci

        rng = np.random.default_rng(42)
        folds = [
            rng.binomial(1, 0.5, size=65).astype(float),
            rng.binomial(1, 0.4, size=45).astype(float),
            rng.binomial(1, 0.6, size=64).astype(float),
            rng.binomial(1, 0.35, size=13).astype(float),
            rng.binomial(1, 0.55, size=10).astype(float),
        ]
        start = time.perf_counter()
        fold_stratified_bootstrap_ci(folds, n_resamples=10_000, seed=42)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Bootstrap took {elapsed:.2f}s — should be < 1s"


class TestPairedBootstrapDelta:
    """Test paired bootstrap for method comparison."""

    def test_positive_delta_detected(self) -> None:
        from tract.training.evaluate import paired_bootstrap_delta

        fold_a = [np.array([0.0, 0.0, 0.0, 0.0, 0.0])]
        fold_b = [np.array([1.0, 1.0, 1.0, 1.0, 1.0])]
        result = paired_bootstrap_delta(fold_a, fold_b, n_resamples=1000, seed=42)
        assert result["delta_mean"] > 0
        assert result["ci_low"] > 0

    def test_no_difference_includes_zero(self) -> None:
        from tract.training.evaluate import paired_bootstrap_delta

        rng = np.random.default_rng(99)
        data = rng.binomial(1, 0.5, size=50).astype(float)
        fold_a = [data]
        fold_b = [data]
        result = paired_bootstrap_delta(fold_a, fold_b, n_resamples=1000, seed=42)
        assert result["delta_mean"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0

    def test_p_value_significant_for_clear_difference(self) -> None:
        from tract.training.evaluate import paired_bootstrap_delta

        fold_a = [np.zeros(50)]
        fold_b = [np.ones(50)]
        result = paired_bootstrap_delta(fold_a, fold_b, n_resamples=1000, seed=42)
        assert result["p_value"] == 0.0

    def test_rejects_fold_count_mismatch(self) -> None:
        from tract.training.evaluate import paired_bootstrap_delta

        with pytest.raises(ValueError, match="Fold count"):
            paired_bootstrap_delta(
                [np.array([1.0])], [np.array([1.0]), np.array([0.0])],
                n_resamples=100, seed=42,
            )

    def test_rejects_fold_size_mismatch(self) -> None:
        from tract.training.evaluate import paired_bootstrap_delta

        with pytest.raises(ValueError, match="size mismatch"):
            paired_bootstrap_delta(
                [np.array([1.0, 0.0])], [np.array([1.0])],
                n_resamples=100, seed=42,
            )

    def test_multi_fold_realistic(self) -> None:
        """Paired delta with LOFO-realistic fold structure."""
        from tract.training.evaluate import paired_bootstrap_delta

        rng = np.random.default_rng(42)
        sizes = [65, 45, 64, 13, 10]
        fold_a = [rng.binomial(1, 0.4, size=s).astype(float) for s in sizes]
        fold_b = [rng.binomial(1, 0.6, size=s).astype(float) for s in sizes]
        result = paired_bootstrap_delta(fold_a, fold_b, n_resamples=5000, seed=42)
        assert result["delta_mean"] > 0
        assert "p_value" in result

    def test_vectorized_speed(self) -> None:
        """10K paired resamples should complete in under 1 second."""
        from tract.training.evaluate import paired_bootstrap_delta

        rng = np.random.default_rng(42)
        sizes = [65, 45, 64, 13, 10]
        fold_a = [rng.binomial(1, 0.4, size=s).astype(float) for s in sizes]
        fold_b = [rng.binomial(1, 0.6, size=s).astype(float) for s in sizes]
        start = time.perf_counter()
        paired_bootstrap_delta(fold_a, fold_b, n_resamples=10_000, seed=42)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Paired bootstrap took {elapsed:.2f}s — should be < 1s"


class TestBenjaminiHochberg:
    """Test BH FDR control."""

    def test_corrects_multiple_pvalues(self) -> None:
        from tract.training.evaluate import benjamini_hochberg

        p_values = [0.01, 0.03, 0.05, 0.20, 0.50]
        rejected, adjusted = benjamini_hochberg(p_values, q=0.10)
        assert isinstance(rejected, list)
        assert len(rejected) == len(p_values)
        assert all(isinstance(r, bool) for r in rejected)
        assert len(adjusted) == len(p_values)
        assert all(0.0 <= p <= 1.0 for p in adjusted)

    def test_no_rejections_for_high_pvalues(self) -> None:
        from tract.training.evaluate import benjamini_hochberg

        p_values = [0.50, 0.60, 0.70, 0.80, 0.90]
        rejected, adjusted = benjamini_hochberg(p_values, q=0.10)
        assert not any(rejected)

    def test_all_rejections_for_low_pvalues(self) -> None:
        from tract.training.evaluate import benjamini_hochberg

        p_values = [0.001, 0.002, 0.003]
        rejected, adjusted = benjamini_hochberg(p_values, q=0.10)
        assert all(rejected)

    def test_empty_input(self) -> None:
        from tract.training.evaluate import benjamini_hochberg

        rejected, adjusted = benjamini_hochberg([], q=0.10)
        assert rejected == []
        assert adjusted == []

    def test_adjusted_monotonic(self) -> None:
        """Adjusted p-values should be monotonically non-decreasing when sorted."""
        from tract.training.evaluate import benjamini_hochberg

        p_values = [0.04, 0.01, 0.08, 0.03, 0.15]
        _, adjusted = benjamini_hochberg(p_values, q=0.10)
        sorted_adj = sorted(adjusted)
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-10

    def test_single_pvalue(self) -> None:
        from tract.training.evaluate import benjamini_hochberg

        rejected, adjusted = benjamini_hochberg([0.05], q=0.10)
        assert rejected == [True]
        assert adjusted[0] == pytest.approx(0.05)


class TestSoftFloorCheck:
    """Test per-framework soft floor enforcement."""

    def test_passes_when_above_floor(self) -> None:
        from tract.training.evaluate import check_soft_floors

        per_fold_deltas = {
            "MITRE ATLAS": {"delta_mean": 0.05, "ci_low": -0.02, "ci_high": 0.12},
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 0

    def test_detects_violation(self) -> None:
        from tract.training.evaluate import check_soft_floors

        per_fold_deltas = {
            "MITRE ATLAS": {"delta_mean": -0.10, "ci_low": -0.15, "ci_high": -0.05},
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 1
        assert "MITRE ATLAS" in violations

    def test_nist_has_looser_floor(self) -> None:
        from tract.training.evaluate import check_soft_floors

        per_fold_deltas = {
            "NIST AI 100-2": {"delta_mean": -0.08, "ci_low": -0.09, "ci_high": -0.07},
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 0

    def test_nist_floor_violation(self) -> None:
        from tract.training.evaluate import check_soft_floors

        per_fold_deltas = {
            "NIST AI 100-2": {"delta_mean": -0.12, "ci_low": -0.15, "ci_high": -0.09},
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 1

    def test_ignores_small_folds(self) -> None:
        from tract.training.evaluate import check_soft_floors

        per_fold_deltas = {
            "OWASP Top10 for LLM": {
                "delta_mean": -0.50, "ci_low": -0.80, "ci_high": -0.20,
            },
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 0

    def test_multiple_violations(self) -> None:
        from tract.training.evaluate import check_soft_floors

        per_fold_deltas = {
            "MITRE ATLAS": {"delta_mean": -0.10, "ci_low": -0.15, "ci_high": -0.05},
            "OWASP AI Exchange": {"delta_mean": -0.10, "ci_low": -0.12, "ci_high": -0.08},
            "NIST AI 100-2": {"delta_mean": 0.05, "ci_low": 0.01, "ci_high": 0.09},
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 2
        assert "MITRE ATLAS" in violations
        assert "OWASP AI Exchange" in violations


class TestCoveredUncoveredSplit:
    """Test covered/uncovered hub split."""

    def test_splits_correctly(self) -> None:
        from tract.training.evaluate import compute_covered_uncovered_split

        items = [
            FakeEvalItem(control_text="a", ground_truth_hub_id="h1"),
            FakeEvalItem(control_text="b", ground_truth_hub_id="h2"),
            FakeEvalItem(control_text="c", ground_truth_hub_id="h3"),
        ]
        predictions = [["h1", "h2"], ["h2", "h1"], ["h1", "h3"]]
        training_hubs = {"h1", "h2"}

        result = compute_covered_uncovered_split(items, predictions, training_hubs)
        assert "covered" in result
        assert "uncovered" in result
        assert result["covered"]["n"] == 2.0
        assert result["uncovered"]["n"] == 1.0

    def test_all_covered(self) -> None:
        from tract.training.evaluate import compute_covered_uncovered_split

        items = [FakeEvalItem(control_text="a", ground_truth_hub_id="h1")]
        predictions = [["h1"]]
        result = compute_covered_uncovered_split(items, predictions, {"h1"})
        assert "covered" in result
        assert "uncovered" not in result
