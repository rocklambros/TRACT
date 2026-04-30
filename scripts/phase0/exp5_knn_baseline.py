"""Experiment 5: kNN retrieval baseline for CRE hub assignment.

For each held-out AI control, finds k nearest training controls by
embedding cosine similarity and transfers hub assignments via weighted
majority vote. Zero training cost — tests whether simple retrieval
already solves the task.

Usage:
    python -m scripts.phase0.exp5_knn_baseline
    python -m scripts.phase0.exp5_knn_baseline --model bge --k-values 5,10,20
    python -m scripts.phase0.exp5_knn_baseline --original  # use uncurated links
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from typing import Final

import numpy as np
import torch

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    HubStandardLink,
    LOFOFold,
    aggregate_lofo_metrics,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    finish_wandb,
    init_wandb,
    load_curated_links,
    load_opencre_cres,
    load_parsed_controls,
    log_aggregate_metrics,
    log_fold_metrics,
    save_results,
    score_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model Config ──────────────────────────────────────────────────────────

RETRIEVAL_MODELS: Final[dict[str, str]] = {
    "bge": "BAAI/bge-large-en-v1.5",
    "gte": "Alibaba-NLP/gte-large-en-v1.5",
}

DEFAULT_K_VALUES: Final[list[int]] = [5, 10, 20]
BATCH_SIZE: Final[int] = 64


# ── kNN Core ──────────────────────────────────────────────────────────────


def knn_weighted_vote(
    control_embedding: np.ndarray,
    pool_embeddings: np.ndarray,
    pool_hub_ids: list[str],
    k: int,
) -> list[str]:
    """Find k nearest training controls, return hubs ranked by weighted vote."""
    similarities = pool_embeddings @ control_embedding
    top_k_indices = np.argpartition(-similarities, k)[:k]
    top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]

    hub_weights: dict[str, float] = defaultdict(float)
    for idx in top_k_indices:
        hub_id = pool_hub_ids[idx]
        weight = float(similarities[idx])
        hub_weights[hub_id] += max(weight, 0.0)

    ranked = sorted(hub_weights.items(), key=lambda x: -x[1])
    return [hub_id for hub_id, _ in ranked[:20]]


def build_retrieval_pool(
    links: list[HubStandardLink],
    held_out_framework: str,
) -> tuple[list[str], list[str]]:
    """Build (texts, hub_ids) for all training links excluding held-out framework."""
    texts: list[str] = []
    hub_ids: list[str] = []
    for link in links:
        if link.standard_name == held_out_framework:
            continue
        text = link.section_name or link.section_id
        if text:
            texts.append(text)
            hub_ids.append(link.cre_id)
    return texts, hub_ids


# ── Main Pipeline ─────────────────────────────────────────────────────────


def run_knn(
    model_key: str,
    links: list[HubStandardLink],
    folds: list[LOFOFold],
    k_values: list[int],
    device: str,
) -> dict:
    """Run kNN retrieval baseline for all k values across all LOFO folds."""
    from sentence_transformers import SentenceTransformer

    hf_id = RETRIEVAL_MODELS[model_key]
    logger.info("Loading retrieval model: %s", hf_id)
    model = SentenceTransformer(hf_id, device=device, trust_remote_code=True)

    results_by_k: dict[int, list[dict[int, list[str]]]] = {k: [] for k in k_values}

    for fold in folds:
        logger.info("Fold: held out %s (%d items)", fold.held_out_framework, len(fold.eval_items))

        pool_texts, pool_hub_ids = build_retrieval_pool(links, fold.held_out_framework)
        logger.info("  Retrieval pool: %d training links", len(pool_texts))

        pool_embeddings = model.encode(
            pool_texts, batch_size=BATCH_SIZE,
            show_progress_bar=False, normalize_embeddings=True,
        )
        pool_embeddings = np.array(pool_embeddings)

        control_texts = [item.control_text for item in fold.eval_items]
        control_embeddings = model.encode(
            control_texts, batch_size=BATCH_SIZE,
            show_progress_bar=False, normalize_embeddings=True,
        )
        control_embeddings = np.array(control_embeddings)

        for k in k_values:
            predictions: dict[int, list[str]] = {}
            effective_k = min(k, len(pool_texts))
            for i, ctrl_emb in enumerate(control_embeddings):
                predictions[i] = knn_weighted_vote(ctrl_emb, pool_embeddings, pool_hub_ids, effective_k)
            results_by_k[k].append(predictions)

    return results_by_k


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 5: kNN retrieval baseline")
    parser.add_argument("--model", choices=list(RETRIEVAL_MODELS.keys()), default="bge")
    parser.add_argument("--k-values", default="5,10,20", help="Comma-separated k values")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--original", action="store_true", help="Use original uncurated links")
    parser.add_argument("--curated", action="store_true", help="Use curated links (default)")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    curated = not args.original

    logger.info("Loading data (curated=%s)...", curated)
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)

    if curated:
        links = load_curated_links()
    else:
        links = extract_hub_standard_links(cres)

    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    logger.info("Building LOFO folds...")
    folds = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")

    wandb_run = init_wandb(
        f"exp5_knn_{args.model}",
        config={
            "model": RETRIEVAL_MODELS[args.model],
            "k_values": k_values,
            "retrieval_pool": "curated" if curated else "original",
        },
        curated=curated,
        tags=["knn-baseline"],
    )

    start_time = time.time()
    results_by_k = run_knn(args.model, links, folds, k_values, args.device)
    total_elapsed = time.time() - start_time

    output: dict = {
        "experiment": "exp5_knn_baseline",
        "model": RETRIEVAL_MODELS[args.model],
        "model_key": args.model,
        "curated": curated,
        "elapsed_seconds": round(total_elapsed, 1),
        "k_values": {},
    }

    for k in k_values:
        fold_results = results_by_k[k]
        metrics_all = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
        metrics_ft = aggregate_lofo_metrics(fold_results, folds, track_filter="full-text")

        per_fold: list[dict] = []
        for fold, preds in zip(folds, fold_results):
            fold_preds_list = [preds[i] for i in range(len(fold.eval_items))]
            fold_gt = [item.ground_truth_hub_id for item in fold.eval_items]
            fold_metrics = score_predictions(fold_preds_list, fold_gt)
            per_fold.append({
                "framework": fold.held_out_framework,
                "n_items": len(fold.eval_items),
                "metrics": fold_metrics,
            })

            log_fold_metrics(wandb_run, f"knn_k{k}", fold.held_out_framework, fold_metrics)

        log_aggregate_metrics(wandb_run, f"knn_k{k}", metrics_all, prefix="all")
        log_aggregate_metrics(wandb_run, f"knn_k{k}", metrics_ft, prefix="full_text")

        output["k_values"][f"k{k}"] = {
            "all": metrics_all,
            "full_text": metrics_ft,
            "per_fold": per_fold,
        }

        logger.info(
            "k=%d all: hit@1=%.3f [%.3f, %.3f], hit@5=%.3f [%.3f, %.3f]",
            k,
            metrics_all["hit_at_1"]["mean"],
            metrics_all["hit_at_1"]["ci_low"],
            metrics_all["hit_at_1"]["ci_high"],
            metrics_all["hit_at_5"]["mean"],
            metrics_all["hit_at_5"]["ci_low"],
            metrics_all["hit_at_5"]["ci_high"],
        )

    finish_wandb(wandb_run)

    output_name = f"exp5_knn_baseline{args.output_suffix}.json"
    save_results(output, output_name)
    logger.info("Experiment 5 complete.")


if __name__ == "__main__":
    main()
