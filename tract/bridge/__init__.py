"""TRACT bridge analysis — AI/traditional CRE hub bridge discovery."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tract.bridge.classify import classify_hubs
from tract.bridge.similarity import (
    compute_bridge_similarities,
    compute_similarity_stats,
    count_controls_for_hub,
    enrich_negatives,
    extract_negatives,
    extract_top_k,
)
from tract.bridge.types import BridgeCandidate
from tract.io import atomic_write_json

logger = logging.getLogger(__name__)


def run_bridge_analysis(
    *,
    artifacts_path: Path,
    hub_links_path: Path,
    hierarchy_path: Path,
    output_dir: Path,
    top_k: int = 3,
    skip_descriptions: bool = False,
) -> Path:
    """Run the full bridge analysis pipeline.

    Returns:
        Path to bridge_candidates.json.
    """
    from tract.hierarchy import CREHierarchy
    from tract.io import load_json

    data = np.load(str(artifacts_path), allow_pickle=False)
    hub_embeddings = data["hub_embeddings"]
    hub_ids = list(data["hub_ids"])

    hub_links = load_json(hub_links_path)
    hierarchy = CREHierarchy.load(hierarchy_path)

    classification = classify_hubs(hub_links_path, hub_ids)
    logger.info(
        "Classified %d AI-only, %d trad-only, %d bridged, %d unlinked",
        len(classification.ai_only), len(classification.trad_only),
        len(classification.naturally_bridged), len(classification.unlinked),
    )

    sim_matrix = compute_bridge_similarities(
        hub_embeddings, hub_ids,
        classification.ai_only, classification.trad_only,
    )
    stats = compute_similarity_stats(sim_matrix)

    raw_candidates = extract_top_k(
        sim_matrix, classification.ai_only, classification.trad_only, k=top_k,
    )

    # Build BridgeCandidate records rather than bolting keys onto the
    # RawCandidate dicts in place. description starts empty so the two paths
    # below (describe or skip) produce the same shape either way.
    candidates: list[BridgeCandidate] = []
    for raw in raw_candidates:
        ai_id = raw["ai_hub_id"]
        trad_id = raw["trad_hub_id"]
        candidates.append({
            "ai_hub_id": ai_id,
            "trad_hub_id": trad_id,
            "cosine_similarity": raw["cosine_similarity"],
            "rank_for_ai_hub": raw["rank_for_ai_hub"],
            "ai_hub_name": hierarchy.hubs[ai_id].name,
            "trad_hub_name": hierarchy.hubs[trad_id].name,
            "seed_evidence": {
                "ai_controls_linked": count_controls_for_hub(ai_id, hub_links),
                "trad_controls_linked": count_controls_for_hub(trad_id, hub_links),
            },
            "status": "pending",
            "reviewer_notes": "",
            "description": "",
        })

    negatives_raw = extract_negatives(
        sim_matrix, classification.ai_only, classification.trad_only,
    )
    # enrich_negatives lives in similarity.py, not describe.py, so this path
    # does not drag in anthropic when descriptions are skipped.
    negatives = enrich_negatives(negatives_raw, hierarchy)
    if not skip_descriptions:
        from tract.bridge.describe import generate_bridge_descriptions

        generate_bridge_descriptions(candidates, hierarchy, hub_links)
        generate_bridge_descriptions(negatives, hierarchy, hub_links)

    unclassified_leaves = [
        h for h in classification.unlinked
        if h in hierarchy.hubs and hierarchy.hubs[h].is_leaf
    ]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "top_k_per_ai_hub",
        "top_k": top_k,
        "similarity_stats": stats,
        "candidates": candidates,
        "negative_controls": negatives,
        "unclassified_leaf_hubs": unclassified_leaves,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "bridge_candidates.json"
    atomic_write_json(output, candidates_path)
    logger.info("Wrote %d candidates to %s", len(candidates), candidates_path)

    logger.info("AI-only hubs: %d", len(classification.ai_only))
    logger.info("Candidates generated: %d", len(candidates))
    logger.info("Output: %s", candidates_path)

    return candidates_path
