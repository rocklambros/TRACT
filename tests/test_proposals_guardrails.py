"""Tests for hub proposal guardrails."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tract.proposals.cluster import Cluster


def _make_cluster(
    cluster_id: int,
    n_controls: int = 4,
    n_frameworks: int = 2,
    centroid_seed: int = 42,
) -> Cluster:
    rng = np.random.default_rng(centroid_seed)
    centroid = rng.standard_normal(1024).astype(np.float32)
    centroid = centroid / np.linalg.norm(centroid)
    fws = [f"fw_{i}" for i in range(n_frameworks)]
    cids = [f"{fws[i % n_frameworks]}::ctrl_{cluster_id}_{i}" for i in range(n_controls)]
    return Cluster(
        cluster_id=cluster_id,
        control_ids=cids,
        centroid=centroid,
        nearest_hub_id="hub-1",
        nearest_hub_similarity=0.4,
        member_frameworks=set(fws),
    )


class TestGuardrail1MinEvidence:
    def test_rejects_single_framework(self) -> None:
        from tract.proposals.guardrails import _check_min_evidence

        cluster = _make_cluster(0, n_controls=5, n_frameworks=1)
        assert not _check_min_evidence(cluster, min_controls=3, min_frameworks=2)

    def test_accepts_multi_framework(self) -> None:
        from tract.proposals.guardrails import _check_min_evidence

        cluster = _make_cluster(0, n_controls=5, n_frameworks=3)
        assert _check_min_evidence(cluster, min_controls=3, min_frameworks=2)


class TestGuardrail3InterClusterSeparation:
    def test_rejects_close_centroids(self) -> None:
        from tract.proposals.guardrails import _check_inter_cluster_separation

        rng = np.random.default_rng(42)
        base = rng.standard_normal(1024).astype(np.float32)
        base = base / np.linalg.norm(base)

        c1 = _make_cluster(0)
        c1 = Cluster(
            cluster_id=0, control_ids=c1.control_ids, centroid=base,
            nearest_hub_id="h1", nearest_hub_similarity=0.4,
            member_frameworks=c1.member_frameworks,
        )
        noise = rng.standard_normal(1024).astype(np.float32) * 0.01
        close_centroid = base + noise
        close_centroid = close_centroid / np.linalg.norm(close_centroid)
        c2 = Cluster(
            cluster_id=1, control_ids=["fw_0::c1"], centroid=close_centroid,
            nearest_hub_id="h2", nearest_hub_similarity=0.3,
            member_frameworks={"fw_0"},
        )

        result = _check_inter_cluster_separation([c1, c2], max_cosine=0.7)
        assert len(result) < 2

    def test_accepts_distant_centroids(self) -> None:
        from tract.proposals.guardrails import _check_inter_cluster_separation

        rng = np.random.default_rng(42)
        c1 = _make_cluster(0, centroid_seed=1)
        c2 = _make_cluster(1, centroid_seed=999)
        cos_sim = float(c1.centroid @ c2.centroid)
        if cos_sim < 0.7:
            result = _check_inter_cluster_separation([c1, c2], max_cosine=0.7)
            assert len(result) == 2


class TestApplyGuardrails:
    def test_budget_cap(self) -> None:
        from tract.proposals.guardrails import apply_guardrails

        clusters = [_make_cluster(i, n_controls=4, n_frameworks=2, centroid_seed=i * 100) for i in range(50)]
        hierarchy = MagicMock()
        hierarchy.hubs = {}
        hub_embs = np.random.default_rng(42).standard_normal((10, 1024)).astype(np.float32)
        hub_ids = [f"hub-{i}" for i in range(10)]

        results = apply_guardrails(clusters, hierarchy, hub_embs, hub_ids, {}, budget_cap=5)
        assert len(results) <= 55  # total results (passed + failed)
        passing = [r for r in results if r.passed]
        assert len(passing) <= 5
