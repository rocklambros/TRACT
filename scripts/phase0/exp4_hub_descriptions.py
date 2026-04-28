"""Experiment 4: LLM hub description pilot.

Stage A: Generate 2-3 sentence descriptions for the top 50 AI-linked leaf hubs.
Stage B: Re-run best bi-encoder(s) with description-enriched hub text on the
         subset of evaluation items mapping to described hubs.

Hub descriptions are static features (no LOFO firewall on description generation).
Evaluation still uses LOFO for linked standard names in hub text.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter
from typing import Final

import numpy as np
import torch

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    CREHierarchy,
    EvalItem,
    HubStandardLink,
    LOFOFold,
    RESULTS_DIR,
    _ndcg_at_k,
    _reciprocal_rank,
    _sanitize_text,
    aggregate_lofo_metrics,
    bootstrap_ci,
    bootstrap_paired_delta,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    get_api_key,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
    score_predictions,
)
from scripts.phase0.exp1_embedding_baseline import (
    BIENCODER_MODELS,
    run_biencoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL: Final[str] = "claude-opus-4-20250514"
TOP_N_HUBS: Final[int] = 50
MAX_CONCURRENT_REQUESTS: Final[int] = 5
API_TIMEOUT_S: Final[int] = 120


# ── Hub Selection ───────────────────────────────────────────────────────────


def select_top_hubs(
    links: list[HubStandardLink],
    tree: CREHierarchy,
    n: int = TOP_N_HUBS,
) -> list[str]:
    """Select top N leaf hubs by AI framework link count."""
    leaf_ids = set(tree.leaf_hub_ids())
    ai_link_counts: Counter[str] = Counter()
    for link in links:
        if link.standard_name in AI_FRAMEWORK_NAMES and link.cre_id in leaf_ids:
            ai_link_counts[link.cre_id] += 1
    return [hub_id for hub_id, _ in ai_link_counts.most_common(n)]


# ── Description Generation ──────────────────────────────────────────────────


def build_description_prompt(
    hub_id: str,
    tree: CREHierarchy,
    links: list[HubStandardLink],
) -> str:
    """Build prompt for generating one hub description."""
    hub_name = tree.hubs[hub_id]
    path = tree.hierarchy_path(hub_id)

    linked_standards = sorted(set(
        link.section_name or link.section_id
        for link in links
        if link.cre_id == hub_id
    ))

    parent_id = tree.parent.get(hub_id)
    siblings: list[str] = []
    if parent_id:
        siblings = [
            tree.hubs[sid]
            for sid in tree.children.get(parent_id, [])
            if sid != hub_id
        ]

    return (
        "You are writing concise descriptions for CRE "
        "(Common Requirements Enumeration) taxonomy hubs.\n\n"
        f"HUB: {hub_name}\n"
        f"HIERARCHY PATH: {path}\n"
        f"LINKED STANDARDS: {', '.join(linked_standards) if linked_standards else '(none)'}\n"
        f"SIBLING HUBS: {', '.join(siblings[:10]) if siblings else '(none)'}\n\n"
        "Write a 2-3 sentence description covering:\n"
        "(a) What this hub covers\n"
        "(b) What distinguishes it from its siblings\n"
        "(c) Its scope boundary (what it does NOT cover)\n\n"
        "Be specific and technical. Use plain text, no markdown formatting."
    )


async def generate_descriptions(
    hub_ids: list[str],
    tree: CREHierarchy,
    links: list[HubStandardLink],
    max_concurrent: int,
) -> dict[str, str]:
    """Generate descriptions for selected hubs using Opus."""
    import anthropic

    api_key = get_api_key()
    client = anthropic.AsyncAnthropic(
        api_key=api_key, max_retries=3, timeout=API_TIMEOUT_S,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    descriptions: dict[str, str] = {}

    async def generate_one(hub_id: str) -> tuple[str, str]:
        prompt = build_description_prompt(hub_id, tree, links)
        async with semaphore:
            response = await asyncio.wait_for(
                client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=API_TIMEOUT_S,
                ),
                timeout=API_TIMEOUT_S + 10,
            )
            raw_text = response.content[0].text.strip()
            return hub_id, _sanitize_text(raw_text, max_length=2000)

    try:
        tasks = [generate_one(hid) for hid in hub_ids]
        results = await asyncio.gather(*tasks)

        for hub_id, description in results:
            descriptions[hub_id] = description
            logger.info(
                "Generated description for %s (%s): %d chars",
                hub_id, tree.hubs[hub_id], len(description),
            )
    finally:
        await client.close()

    return descriptions


# ── Evaluation Helpers ──────────────────────────────────────────────────────


def compute_subset_metrics(
    fold_results: list[dict[int, list[str]]],
    folds: list[LOFOFold],
    described_hub_ids: set[str],
) -> dict[str, np.ndarray]:
    """Compute per-item metric arrays filtered to described hubs only."""
    hit1_list: list[float] = []
    hit5_list: list[float] = []
    mrr_list: list[float] = []
    ndcg_list: list[float] = []

    for fold, results in zip(folds, fold_results):
        for i, item in enumerate(fold.eval_items):
            if item.ground_truth_hub_id not in described_hub_ids:
                continue
            pred = results.get(i, [])
            gt = item.ground_truth_hub_id
            hit1_list.append(1.0 if pred and pred[0] == gt else 0.0)
            hit5_list.append(1.0 if gt in pred[:5] else 0.0)
            mrr_list.append(_reciprocal_rank(pred, gt))
            ndcg_list.append(_ndcg_at_k(pred, gt))

    return {
        "hit_at_1": np.array(hit1_list),
        "hit_at_5": np.array(hit5_list),
        "mrr": np.array(mrr_list),
        "ndcg_at_10": np.array(ndcg_list),
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: Hub description pilot")
    parser.add_argument(
        "--model", choices=["bge", "gte", "all"], default="all",
        help="Which bi-encoder model to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS)
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip description generation, load from existing file",
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    top_hub_ids = select_top_hubs(links, tree, TOP_N_HUBS)
    logger.info("Selected %d hubs for description pilot", len(top_hub_ids))

    # ── Stage A: Description generation ─────────────────────────────────

    desc_file = RESULTS_DIR / "pilot_hub_descriptions.json"

    if args.skip_generation and desc_file.exists():
        logger.info("Loading existing descriptions from %s", desc_file)
        with open(desc_file, encoding="utf-8") as f:
            desc_data = json.load(f)
        descriptions: dict[str, str] = desc_data["descriptions"]
    else:
        logger.info("Generating descriptions for %d hubs...", len(top_hub_ids))
        descriptions = asyncio.run(
            generate_descriptions(top_hub_ids, tree, links, args.max_concurrent)
        )
        desc_data = {
            "model": MODEL,
            "n_hubs": len(descriptions),
            "descriptions": descriptions,
            "hub_details": {
                hid: {
                    "name": tree.hubs[hid],
                    "path": tree.hierarchy_path(hid),
                    "description": descriptions[hid],
                }
                for hid in descriptions
            },
        }
        save_results(desc_data, "pilot_hub_descriptions.json")

    described_hub_ids = set(descriptions.keys())
    eval_subset_count = sum(
        1 for item in corpus if item.ground_truth_hub_id in described_hub_ids
    )
    logger.info(
        "Evaluation subset: %d items (of %d total) map to %d described hubs",
        eval_subset_count, len(corpus), len(described_hub_ids),
    )

    # ── Stage B: Comparative evaluation ─────────────────────────────────

    folds_baseline = build_lofo_folds(
        tree, links, corpus, AI_FRAMEWORK_NAMES, template="default",
    )
    folds_desc = build_lofo_folds(
        tree, links, corpus, AI_FRAMEWORK_NAMES,
        template="description", descriptions=descriptions,
    )

    results: dict = {
        "experiment": "exp4_hub_descriptions",
        "n_described_hubs": len(described_hub_ids),
        "n_eval_subset": eval_subset_count,
        "models": {},
        "device": args.device,
    }

    models_to_run: list[dict[str, str]] = []
    if args.model in ("bge", "all"):
        models_to_run.append(BIENCODER_MODELS[0])
    if args.model in ("gte", "all"):
        models_to_run.append(BIENCODER_MODELS[1])

    for model_config in models_to_run:
        model_name = model_config["name"]
        logger.info("=" * 60)

        logger.info("Running %s BASELINE...", model_name)
        start = time.time()
        baseline_fold_results = run_biencoder(
            model_config, folds_baseline, corpus, args.device,
        )
        baseline_elapsed = time.time() - start

        logger.info("Running %s DESCRIPTION-ENRICHED...", model_name)
        start = time.time()
        desc_fold_results = run_biencoder(
            model_config, folds_desc, corpus, args.device,
        )
        desc_elapsed = time.time() - start

        baseline_items = compute_subset_metrics(
            baseline_fold_results, folds_baseline, described_hub_ids,
        )
        desc_items = compute_subset_metrics(
            desc_fold_results, folds_desc, described_hub_ids,
        )

        baseline_metrics_subset = {
            m: bootstrap_ci(baseline_items[m]) for m in baseline_items
        }
        desc_metrics_subset = {
            m: bootstrap_ci(desc_items[m]) for m in desc_items
        }

        deltas: dict[str, dict[str, float]] = {}
        for metric_name in baseline_items:
            deltas[metric_name] = bootstrap_paired_delta(
                baseline_items[metric_name],
                desc_items[metric_name],
            )

        results["models"][model_name] = {
            "hf_id": model_config["hf_id"],
            "baseline_subset": baseline_metrics_subset,
            "description_enriched_subset": desc_metrics_subset,
            "deltas_subset": deltas,
            "baseline_elapsed_seconds": round(baseline_elapsed, 1),
            "description_elapsed_seconds": round(desc_elapsed, 1),
        }

        logger.info(
            "%s described-hub subset: baseline hit@1=%.3f, desc hit@1=%.3f, delta=%.3f",
            model_name,
            baseline_metrics_subset["hit_at_1"]["mean"],
            desc_metrics_subset["hit_at_1"]["mean"],
            deltas["hit_at_1"]["delta_mean"],
        )

    save_results(results, "exp4_hub_descriptions.json")
    logger.info("Experiment 4 complete.")


if __name__ == "__main__":
    main()
