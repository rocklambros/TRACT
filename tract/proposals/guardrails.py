"""Six-filter guardrail pipeline for hub proposals.

Guardrails:
1. Minimum evidence: 3+ controls from 2+ frameworks
2. Hierarchy constraint: parent hub identified via max cosine
3. Inter-cluster separation: pairwise centroid cosine < 0.7
4. Budget cap: top N by evidence strength
5. Candidate queue: proposals with evidence
6. Determinism: same inputs -> same outputs
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tract.config import (
    PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    PHASE1D_PROPOSAL_BUDGET_CAP,
    PHASE1D_PROPOSAL_INTER_CLUSTER_MAX_COSINE,
    PHASE1D_PROPOSAL_MIN_FRAMEWORKS,
    PHASE1D_PROPOSAL_UNCERTAIN_PLACEMENT_FLOOR,
)
from tract.hierarchy import CREHierarchy
from tract.proposals.cluster import Cluster

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    cluster: Cluster
    passed: bool
    rejection_reasons: list[str]
    parent_hub_id: str | None
    parent_similarity: float
    uncertain_placement: bool
    evidence_score: float


def _check_min_evidence(
    cluster: Cluster,
    min_controls: int = PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    min_frameworks: int = PHASE1D_PROPOSAL_MIN_FRAMEWORKS,
) -> bool:
    return (
        len(cluster.control_ids) >= min_controls
        and len(cluster.member_frameworks) >= min_frameworks
    )


def _check_inter_cluster_separation(
    clusters: list[Cluster],
    max_cosine: float = PHASE1D_PROPOSAL_INTER_CLUSTER_MAX_COSINE,
) -> list[Cluster]:
    """Remove clusters whose centroids are too close to each other.

    Keeps the cluster with more evidence when two are too close.
    """
    if len(clusters) <= 1:
        return list(clusters)

    to_remove: set[int] = set()

    for i in range(len(clusters)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(clusters)):
            if j in to_remove:
                continue
            cos_sim = float(clusters[i].centroid @ clusters[j].centroid)
            if cos_sim >= max_cosine:
                if len(clusters[i].control_ids) >= len(clusters[j].control_ids):
                    to_remove.add(j)
                    logger.info(
                        "Guardrail 3: removing cluster %d (too close to %d, cosine=%.3f)",
                        clusters[j].cluster_id, clusters[i].cluster_id, cos_sim,
                    )
                else:
                    to_remove.add(i)
                    logger.info(
                        "Guardrail 3: removing cluster %d (too close to %d, cosine=%.3f)",
                        clusters[i].cluster_id, clusters[j].cluster_id, cos_sim,
                    )

    return [c for idx, c in enumerate(clusters) if idx not in to_remove]


def apply_guardrails(
    clusters: list[Cluster],
    hierarchy: CREHierarchy,
    hub_embeddings: NDArray[np.floating],
    hub_ids: list[str],
    control_metadata: dict[str, dict],
    budget_cap: int = PHASE1D_PROPOSAL_BUDGET_CAP,
) -> list[GuardrailResult]:
    """Apply 6 guardrails to each cluster."""
    results: list[GuardrailResult] = []

    # Guardrail 1: Minimum evidence
    passing_g1: list[Cluster] = []
    for cluster in clusters:
        if _check_min_evidence(cluster):
            passing_g1.append(cluster)
        else:
            reasons = []
            if len(cluster.control_ids) < PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE:
                reasons.append(f"insufficient controls ({len(cluster.control_ids)})")
            if len(cluster.member_frameworks) < PHASE1D_PROPOSAL_MIN_FRAMEWORKS:
                reasons.append(f"insufficient frameworks ({len(cluster.member_frameworks)})")
            results.append(GuardrailResult(
                cluster=cluster, passed=False, rejection_reasons=reasons,
                parent_hub_id=None, parent_similarity=0.0,
                uncertain_placement=False, evidence_score=0.0,
            ))

    # Guardrail 3: Inter-cluster separation
    passing_g3 = _check_inter_cluster_separation(passing_g1)
    removed_g3 = set(c.cluster_id for c in passing_g1) - set(c.cluster_id for c in passing_g3)
    for cluster in passing_g1:
        if cluster.cluster_id in removed_g3:
            results.append(GuardrailResult(
                cluster=cluster, passed=False,
                rejection_reasons=["inter-cluster separation (cosine >= 0.7 to another cluster)"],
                parent_hub_id=None, parent_similarity=0.0,
                uncertain_placement=False, evidence_score=0.0,
            ))

    # Guardrail 2: Hierarchy constraint (parent hub identification)
    for cluster in passing_g3:
        if len(hub_embeddings) > 0 and len(hub_ids) > 0:
            sims = hub_embeddings @ cluster.centroid
            best_idx = int(np.argmax(sims))
            parent_hub_id = hub_ids[best_idx]
            parent_sim = float(sims[best_idx])
        else:
            parent_hub_id = None
            parent_sim = 0.0

        uncertain = parent_sim < PHASE1D_PROPOSAL_UNCERTAIN_PLACEMENT_FLOOR
        evidence = len(cluster.control_ids) * len(cluster.member_frameworks)

        results.append(GuardrailResult(
            cluster=cluster, passed=True, rejection_reasons=[],
            parent_hub_id=parent_hub_id, parent_similarity=parent_sim,
            uncertain_placement=uncertain,
            evidence_score=float(evidence),
        ))

    # Guardrail 4: Budget cap
    passing = [r for r in results if r.passed]
    passing.sort(key=lambda r: -r.evidence_score)
    if len(passing) > budget_cap:
        for r in passing[budget_cap:]:
            r.passed = False
            r.rejection_reasons.append(f"budget cap ({budget_cap})")

    logger.info(
        "Guardrails: %d/%d clusters passed",
        sum(1 for r in results if r.passed), len(clusters),
    )
    return results
