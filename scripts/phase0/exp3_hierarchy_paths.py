"""Experiment 3: Hierarchy path features for CRE hub assignment.

Re-runs experiment 1's bi-encoder pipeline (BGE and GTE) with path-enriched
hub text: "{hierarchy_path} | {hub_name}: {linked standard names}".

Reports side-by-side metrics and paired bootstrap deltas vs baseline.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Final

import numpy as np
import torch

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    LOFOFold,
    aggregate_lofo_metrics,
    bootstrap_paired_delta,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
    reciprocal_rank,
    ndcg_at_k,
)
from scripts.phase0.exp1_embedding_baseline import (
    BIENCODER_MODELS,
    run_biencoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRIC_NAMES: Final[list[str]] = ["hit_at_1", "hit_at_5", "mrr", "ndcg_at_10"]


def compute_per_item_metrics(
    fold_results: list[dict[int, list[str]]],
    folds: list[LOFOFold],
    track_filter: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-item metric arrays for paired bootstrap.

    Returns a dict mapping metric name to a numpy array with one value
    per evaluation item (across all folds, filtered by track).
    """
    hit1_list: list[float] = []
    hit5_list: list[float] = []
    mrr_list: list[float] = []
    ndcg_list: list[float] = []

    for fold, results in zip(folds, fold_results):
        for i, item in enumerate(fold.eval_items):
            if track_filter == "full-text" and item.track != "full-text":
                continue
            pred = results.get(i, [])
            gt = item.ground_truth_hub_id
            hit1_list.append(1.0 if pred and pred[0] == gt else 0.0)
            hit5_list.append(1.0 if gt in pred[:5] else 0.0)
            mrr_list.append(reciprocal_rank(pred, gt))
            ndcg_list.append(ndcg_at_k(pred, gt))

    return {
        "hit_at_1": np.array(hit1_list),
        "hit_at_5": np.array(hit5_list),
        "mrr": np.array(mrr_list),
        "ndcg_at_10": np.array(ndcg_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Hierarchy path features")
    parser.add_argument(
        "--model", choices=["bge", "gte", "all"], default="all",
        help="Which bi-encoder model to run (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output filename (e.g. '_bge' -> exp3_hierarchy_paths_bge.json)",
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    logger.info("Building LOFO folds (default + path templates)...")
    folds_baseline = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")
    folds_path = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="path")

    results: dict = {
        "experiment": "exp3_hierarchy_paths",
        "models": {},
        "device": args.device,
    }

    models_to_run: list[dict[str, str]] = []
    if args.model in ("bge", "all"):
        models_to_run.append(BIENCODER_MODELS[0])
    if args.model in ("gte", "all"):
        models_to_run.append(BIENCODER_MODELS[1])

    for model_config in models_to_run:
        model_name: str = model_config["name"]
        logger.info("=" * 60)

        logger.info("Running %s BASELINE...", model_name)
        start = time.time()
        baseline_fold_results = run_biencoder(model_config, folds_baseline, corpus, args.device)
        baseline_elapsed = time.time() - start

        logger.info("Running %s PATH-ENRICHED...", model_name)
        start = time.time()
        path_fold_results = run_biencoder(model_config, folds_path, corpus, args.device)
        path_elapsed = time.time() - start

        baseline_metrics_all = aggregate_lofo_metrics(baseline_fold_results, folds_baseline, track_filter=None)
        path_metrics_all = aggregate_lofo_metrics(path_fold_results, folds_path, track_filter=None)
        baseline_metrics_ft = aggregate_lofo_metrics(baseline_fold_results, folds_baseline, track_filter="full-text")
        path_metrics_ft = aggregate_lofo_metrics(path_fold_results, folds_path, track_filter="full-text")

        baseline_items_all = compute_per_item_metrics(baseline_fold_results, folds_baseline, track_filter=None)
        path_items_all = compute_per_item_metrics(path_fold_results, folds_path, track_filter=None)

        deltas_all: dict[str, dict[str, float]] = {}
        for metric_name in METRIC_NAMES:
            deltas_all[metric_name] = bootstrap_paired_delta(
                baseline_items_all[metric_name],
                path_items_all[metric_name],
            )

        results["models"][model_name] = {
            "hf_id": model_config["hf_id"],
            "baseline": {
                "all_198": baseline_metrics_all,
                "full_text": baseline_metrics_ft,
                "elapsed_seconds": round(baseline_elapsed, 1),
            },
            "path_enriched": {
                "all_198": path_metrics_all,
                "full_text": path_metrics_ft,
                "elapsed_seconds": round(path_elapsed, 1),
            },
            "deltas_all_198": deltas_all,
        }

        logger.info(
            "%s baseline all-198: hit@1=%.3f, hit@5=%.3f",
            model_name,
            baseline_metrics_all["hit_at_1"]["mean"],
            baseline_metrics_all["hit_at_5"]["mean"],
        )
        logger.info(
            "%s path all-198: hit@1=%.3f, hit@5=%.3f",
            model_name,
            path_metrics_all["hit_at_1"]["mean"],
            path_metrics_all["hit_at_5"]["mean"],
        )
        for metric_name in METRIC_NAMES:
            delta = deltas_all[metric_name]
            logger.info(
                "%s delta %s=%.3f [%.3f, %.3f]",
                model_name,
                metric_name,
                delta["delta_mean"],
                delta["ci_low"],
                delta["ci_high"],
            )

    output_name = f"exp3_hierarchy_paths{args.output_suffix}.json"
    save_results(results, output_name)
    logger.info("Experiment 3 complete.")


if __name__ == "__main__":
    main()
