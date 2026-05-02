"""Cosine similarity computation and top-K extraction for bridge analysis."""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_bridge_similarities(
    hub_embeddings: NDArray[np.floating],
    hub_ids: list[str],
    ai_only_ids: list[str],
    trad_only_ids: list[str],
) -> NDArray[np.floating]:
    """Compute cosine similarity matrix between AI-only and trad-only hubs.

    All hub embeddings are unit-normalized, so cosine = dot product.

    Returns:
        (n_ai, n_trad) float matrix of cosine similarities.
    """
    ai_indices = [hub_ids.index(h) for h in ai_only_ids]
    trad_indices = [hub_ids.index(h) for h in trad_only_ids]

    ai_emb = hub_embeddings[ai_indices]
    trad_emb = hub_embeddings[trad_indices]

    return ai_emb @ trad_emb.T


def extract_top_k(
    similarity_matrix: NDArray[np.floating],
    ai_hub_ids: list[str],
    trad_hub_ids: list[str],
    k: int = 3,
) -> list[dict]:
    """Extract top-K traditional matches per AI-only hub.

    Returns:
        List of candidate dicts sorted by (ai_hub_id, rank).
    """
    candidates: list[dict] = []
    for i, ai_id in enumerate(ai_hub_ids):
        row = similarity_matrix[i]
        top_k_indices = np.argsort(row)[-k:][::-1]
        for rank, j in enumerate(top_k_indices, 1):
            candidates.append({
                "ai_hub_id": ai_id,
                "trad_hub_id": trad_hub_ids[int(j)],
                "cosine_similarity": round(float(row[j]), 6),
                "rank_for_ai_hub": rank,
            })
    return candidates


def extract_negatives(
    similarity_matrix: NDArray[np.floating],
    ai_hub_ids: list[str],
    trad_hub_ids: list[str],
) -> list[dict]:
    """Extract bottom-1 traditional match per AI-only hub (negative controls).

    Returns:
        List of negative candidate dicts (one per AI hub).
    """
    negatives: list[dict] = []
    for i, ai_id in enumerate(ai_hub_ids):
        row = similarity_matrix[i]
        worst_idx = int(np.argmin(row))
        negatives.append({
            "ai_hub_id": ai_id,
            "trad_hub_id": trad_hub_ids[worst_idx],
            "cosine_similarity": round(float(row[worst_idx]), 6),
            "is_negative": True,
        })
    return negatives


def compute_similarity_stats(
    similarity_matrix: NDArray[np.floating],
) -> dict:
    """Compute summary statistics for the similarity matrix."""
    return {
        "matrix_shape": list(similarity_matrix.shape),
        "mean": round(float(np.mean(similarity_matrix)), 3),
        "std": round(float(np.std(similarity_matrix)), 3),
        "min": round(float(np.min(similarity_matrix)), 3),
        "max": round(float(np.max(similarity_matrix)), 3),
        "percentiles": {
            str(p): round(float(np.percentile(similarity_matrix, p)), 3)
            for p in [25, 50, 75, 90, 95, 99]
        },
    }
