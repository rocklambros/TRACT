"""Tests for HDBSCAN clustering on OOD control embeddings."""
from __future__ import annotations

import numpy as np
import pytest


class TestClusterOodControls:
    def test_empty_input_returns_empty(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls

        result = cluster_ood_controls(
            embeddings=np.empty((0, 1024)),
            control_ids=[],
        )
        assert result == []

    def test_insufficient_items_returns_empty(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls

        rng = np.random.default_rng(42)
        embs = rng.standard_normal((2, 1024)).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        result = cluster_ood_controls(
            embeddings=embs,
            control_ids=["c1", "c2"],
            min_cluster_size=3,
        )
        assert result == []

    def test_determinism(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls

        rng = np.random.default_rng(42)
        centers = rng.standard_normal((3, 1024)).astype(np.float32)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        embs_list = []
        ids_list = []
        for c_idx, center in enumerate(centers):
            for j in range(5):
                noise = rng.standard_normal(1024).astype(np.float32) * 0.01
                emb = center + noise
                emb = emb / np.linalg.norm(emb)
                embs_list.append(emb)
                ids_list.append(f"ctrl_{c_idx}_{j}")

        embs = np.array(embs_list)
        result1 = cluster_ood_controls(embs, ids_list, min_cluster_size=3)
        result2 = cluster_ood_controls(embs, ids_list, min_cluster_size=3)

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert sorted(c1.control_ids) == sorted(c2.control_ids)

    def test_cluster_fields(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls, Cluster

        rng = np.random.default_rng(42)
        center = rng.standard_normal(1024).astype(np.float32)
        center = center / np.linalg.norm(center)

        embs = []
        ids = []
        for j in range(5):
            noise = rng.standard_normal(1024).astype(np.float32) * 0.01
            emb = center + noise
            emb = emb / np.linalg.norm(emb)
            embs.append(emb)
            ids.append(f"ctrl_{j}")

        hub_embs = rng.standard_normal((10, 1024)).astype(np.float32)
        hub_embs = hub_embs / np.linalg.norm(hub_embs, axis=1, keepdims=True)
        hub_ids = [f"hub-{i}" for i in range(10)]

        result = cluster_ood_controls(
            np.array(embs), ids, min_cluster_size=3,
            hub_embeddings=hub_embs, hub_ids=hub_ids,
        )
        if result:
            c = result[0]
            assert isinstance(c.cluster_id, int)
            assert len(c.control_ids) >= 3
            assert c.centroid.shape == (1024,)
            assert c.nearest_hub_id in hub_ids
