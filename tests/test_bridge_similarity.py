"""Tests for tract.bridge.similarity — cosine matrix and top-K extraction."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def unit_embeddings():
    """11 unit-normalized 1024-dim vectors with fixed seed."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((11, 1024))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def hub_ids():
    return [
        "AI-1", "AI-2", "AI-3",
        "BOTH-1",
        "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5",
        "UNLINKED-1", "UNLINKED-2",
    ]


@pytest.fixture
def ai_only_ids():
    return ["AI-1", "AI-2", "AI-3"]


@pytest.fixture
def trad_only_ids():
    return ["TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5"]


class TestComputeBridgeSimilarities:

    def test_matrix_shape(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        matrix = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        assert matrix.shape == (3, 5)

    def test_values_bounded(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        matrix = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        assert matrix.min() >= -1.0 - 1e-6
        assert matrix.max() <= 1.0 + 1e-6

    def test_deterministic(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        m1 = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        m2 = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        np.testing.assert_array_equal(m1, m2)

    def test_dot_product_equals_cosine(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        matrix = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        ai_idx = [hub_ids.index(h) for h in ai_only_ids]
        trad_idx = [hub_ids.index(h) for h in trad_only_ids]
        for i, ai_i in enumerate(ai_idx):
            for j, trad_j in enumerate(trad_idx):
                expected = float(np.dot(unit_embeddings[ai_i], unit_embeddings[trad_j]))
                np.testing.assert_almost_equal(matrix[i, j], expected, decimal=6)


class TestExtractTopK:

    def test_returns_k_per_hub(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.5, 0.3], [0.2, 0.8, 0.6]])
        ai_ids = ["AI-1", "AI-2"]
        trad_ids = ["TRAD-1", "TRAD-2", "TRAD-3"]
        candidates = extract_top_k(matrix, ai_ids, trad_ids, k=2)
        assert len(candidates) == 4
        ai1_cands = [c for c in candidates if c["ai_hub_id"] == "AI-1"]
        assert len(ai1_cands) == 2

    def test_sorted_descending_per_hub(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.5, 0.3, 0.7, 0.1]])
        candidates = extract_top_k(matrix, ["AI-1"], ["T1", "T2", "T3", "T4", "T5"], k=3)
        sims = [c["cosine_similarity"] for c in candidates]
        assert sims == sorted(sims, reverse=True)

    def test_rank_numbering(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.5, 0.3]])
        candidates = extract_top_k(matrix, ["AI-1"], ["T1", "T2", "T3"], k=3)
        ranks = [c["rank_for_ai_hub"] for c in candidates]
        assert ranks == [1, 2, 3]

    def test_top_k_identity(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.1, 0.5]])
        candidates = extract_top_k(matrix, ["AI-1"], ["T1", "T2", "T3"], k=2)
        trad_ids = [c["trad_hub_id"] for c in candidates]
        assert trad_ids == ["T1", "T3"]


class TestExtractNegatives:

    def test_one_per_ai_hub(self) -> None:
        from tract.bridge.similarity import extract_negatives
        matrix = np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.3]])
        negatives = extract_negatives(matrix, ["AI-1", "AI-2"], ["T1", "T2", "T3"])
        assert len(negatives) == 2

    def test_picks_lowest_similarity(self) -> None:
        from tract.bridge.similarity import extract_negatives
        matrix = np.array([[0.9, 0.1, 0.5]])
        negatives = extract_negatives(matrix, ["AI-1"], ["T1", "T2", "T3"])
        assert negatives[0]["trad_hub_id"] == "T2"
        assert abs(negatives[0]["cosine_similarity"] - 0.1) < 1e-6


class TestComputeSimilarityStats:

    def test_stats_keys(self) -> None:
        from tract.bridge.similarity import compute_similarity_stats
        matrix = np.array([[0.9, 0.5], [0.3, 0.7]])
        stats = compute_similarity_stats(matrix)
        assert "matrix_shape" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "percentiles" in stats

    def test_matrix_shape_value(self) -> None:
        from tract.bridge.similarity import compute_similarity_stats
        matrix = np.array([[0.9, 0.5], [0.3, 0.7]])
        stats = compute_similarity_stats(matrix)
        assert stats["matrix_shape"] == [2, 2]
