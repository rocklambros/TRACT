"""HDBSCAN clustering on OOD control embeddings.

Uses euclidean metric on L2-normalized vectors (equivalent to cosine distance).
Deterministic: same inputs -> same clusters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tract.config import (
    PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    PHASE1D_HDBSCAN_MIN_SAMPLES,
)

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    cluster_id: int
    control_ids: list[str]
    centroid: NDArray[np.floating]
    nearest_hub_id: str
    nearest_hub_similarity: float
    member_frameworks: set[str] = field(default_factory=set)


def cluster_ood_controls(
    embeddings: NDArray[np.floating],
    control_ids: list[str],
    min_cluster_size: int = PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = PHASE1D_HDBSCAN_MIN_SAMPLES,
    hub_embeddings: NDArray[np.floating] | None = None,
    hub_ids: list[str] | None = None,
) -> list[Cluster]:
    """HDBSCAN clustering on OOD control embeddings.

    Returns empty list if insufficient OOD items — expected behavior.
    """
    if len(embeddings) == 0 or len(control_ids) == 0:
        logger.info("No OOD items to cluster")
        return []

    if len(embeddings) < min_cluster_size:
        logger.info(
            "Insufficient OOD items (%d) for min_cluster_size=%d",
            len(embeddings), min_cluster_size,
        )
        return []

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if not unique_labels:
        logger.info("HDBSCAN found zero clusters in %d OOD items", len(embeddings))
        return []

    clusters: list[Cluster] = []
    for label in sorted(unique_labels):
        mask = labels == label
        member_ids = [control_ids[i] for i in range(len(control_ids)) if mask[i]]
        member_embs = embeddings[mask]
        centroid = member_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        nearest_hub = ""
        nearest_sim = 0.0
        if hub_embeddings is not None and hub_ids is not None:
            sims = hub_embeddings @ centroid
            best_idx = int(np.argmax(sims))
            nearest_hub = hub_ids[best_idx]
            nearest_sim = float(sims[best_idx])

        member_fws = set()
        for cid in member_ids:
            if "::" in cid:
                member_fws.add(cid.split("::")[0])

        clusters.append(Cluster(
            cluster_id=int(label),
            control_ids=member_ids,
            centroid=centroid,
            nearest_hub_id=nearest_hub,
            nearest_hub_similarity=nearest_sim,
            member_frameworks=member_fws,
        ))

    logger.info(
        "HDBSCAN found %d clusters from %d OOD items (%d noise)",
        len(clusters), len(embeddings), int((labels == -1).sum()),
    )
    return clusters
