"""Tests for experiment 1 scoring logic (not model inference)."""
from __future__ import annotations

import numpy as np
import pytest


def test_rank_by_cosine_similarity() -> None:
    from scripts.phase0.exp1_embedding_baseline import rank_by_cosine_similarity

    hub_ids = ["HUB-A", "HUB-B", "HUB-C"]
    hub_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    control_embedding = np.array([0.9, 0.1, 0.0])
    ranked = rank_by_cosine_similarity(control_embedding, hub_embeddings, hub_ids)
    assert ranked[0] == "HUB-A"
    assert len(ranked) == 3


def test_rank_by_nli_entailment() -> None:
    from scripts.phase0.exp1_embedding_baseline import rank_by_nli_scores

    hub_ids = ["HUB-A", "HUB-B", "HUB-C"]
    scores = np.array([0.2, 0.9, 0.5])
    ranked = rank_by_nli_scores(scores, hub_ids)
    assert ranked[0] == "HUB-B"
    assert ranked[1] == "HUB-C"
    assert ranked[2] == "HUB-A"
