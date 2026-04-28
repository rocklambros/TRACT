"""Experiment 1: Multi-model embedding baseline for CRE hub assignment.

Evaluates three off-the-shelf encoders on the hub assignment task:
- BAAI/bge-large-en-v1.5 (bi-encoder, 335M params, 1024 dim)
- Alibaba-NLP/gte-large-en-v1.5 (bi-encoder, 434M params, 1024 dim)
- cross-encoder/nli-deberta-v3-large (NLI cross-encoder, 304M params)

Bi-encoders rank hubs by cosine similarity.
Cross-encoder ranks by NLI entailment probability.
LOFO cross-validation with hub firewall on 5 AI frameworks.
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
    EvalItem,
    LOFOFold,
    aggregate_lofo_metrics,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
    score_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model Config ────────────────────────────────────────────────────────────

BIENCODER_MODELS: Final[list[dict[str, str]]] = [
    {"name": "bge-large-v1.5", "hf_id": "BAAI/bge-large-en-v1.5"},
    {"name": "gte-large-v1.5", "hf_id": "Alibaba-NLP/gte-large-en-v1.5"},
]

CROSSENCODER_MODEL: Final[dict[str, str]] = {
    "name": "deberta-v3-nli",
    "hf_id": "cross-encoder/nli-deberta-v3-large",
}

BATCH_SIZE: Final[int] = 32


# ── Ranking Functions ───────────────────────────────────────────────────────


def rank_by_cosine_similarity(
    control_embedding: np.ndarray,
    hub_embeddings: np.ndarray,
    hub_ids: list[str],
) -> list[str]:
    """Rank hub IDs by cosine similarity to a single control embedding."""
    control_norm = control_embedding / (np.linalg.norm(control_embedding) + 1e-10)
    hub_norms = hub_embeddings / (np.linalg.norm(hub_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = hub_norms @ control_norm
    ranked_indices = np.argsort(-similarities)
    return [hub_ids[i] for i in ranked_indices]


def rank_by_nli_scores(
    scores: np.ndarray,
    hub_ids: list[str],
) -> list[str]:
    """Rank hub IDs by NLI entailment scores (descending)."""
    ranked_indices = np.argsort(-scores)
    return [hub_ids[i] for i in ranked_indices]


# ── Bi-Encoder Pipeline ────────────────────────────────────────────────────


def run_biencoder(
    model_config: dict[str, str],
    folds: list[LOFOFold],
    corpus: list[EvalItem],
    device: str,
) -> list[dict[int, list[str]]]:
    """Run bi-encoder evaluation across all LOFO folds.

    Returns list of fold result dicts, each mapping eval item index
    to a ranked list of hub IDs (top 20).
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading bi-encoder: %s", model_config["hf_id"])
    model = SentenceTransformer(model_config["hf_id"], device=device, trust_remote_code=True)

    fold_results: list[dict[int, list[str]]] = []

    for fold in folds:
        logger.info("Fold: held out %s (%d items)", fold.held_out_framework, len(fold.eval_items))

        hub_texts_ordered = [fold.hub_texts[hid] for hid in fold.hub_ids]
        hub_embeddings = model.encode(
            hub_texts_ordered,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        hub_embeddings = np.array(hub_embeddings)

        control_texts = [item.control_text for item in fold.eval_items]
        control_embeddings = model.encode(
            control_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        control_embeddings = np.array(control_embeddings)

        predictions: dict[int, list[str]] = {}
        for i, ctrl_emb in enumerate(control_embeddings):
            ranked = rank_by_cosine_similarity(ctrl_emb, hub_embeddings, fold.hub_ids)
            predictions[i] = ranked[:20]

        fold_results.append(predictions)

    return fold_results


# ── Cross-Encoder Pipeline ──────────────────────────────────────────────────


def run_crossencoder(
    model_config: dict[str, str],
    folds: list[LOFOFold],
    corpus: list[EvalItem],
    device: str,
) -> list[dict[int, list[str]]]:
    """Run NLI cross-encoder evaluation across all LOFO folds.

    Returns list of fold result dicts, each mapping eval item index
    to a ranked list of hub IDs (top 20).
    """
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder: %s", model_config["hf_id"])
    model = CrossEncoder(model_config["hf_id"], device=device)

    fold_results: list[dict[int, list[str]]] = []

    for fold in folds:
        logger.info("Fold: held out %s (%d items)", fold.held_out_framework, len(fold.eval_items))
        hub_texts_ordered = [fold.hub_texts[hid] for hid in fold.hub_ids]

        predictions: dict[int, list[str]] = {}

        for i, item in enumerate(fold.eval_items):
            pairs = [(item.control_text, ht) for ht in hub_texts_ordered]

            raw_scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
            scores_arr = np.array(raw_scores)

            if scores_arr.ndim == 2:
                entailment_scores = scores_arr[:, -1]
            else:
                entailment_scores = scores_arr

            ranked = rank_by_nli_scores(entailment_scores, fold.hub_ids)
            predictions[i] = ranked[:20]

            if (i + 1) % 10 == 0:
                logger.info("  Cross-encoder progress: %d/%d", i + 1, len(fold.eval_items))

        fold_results.append(predictions)

    return fold_results


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Multi-model embedding baseline")
    parser.add_argument(
        "--model",
        choices=["bge", "gte", "deberta", "all"],
        default="all",
        help="Which model to run (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output filename (e.g. '_bge' -> exp1_embedding_baseline_bge.json)",
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    logger.info("Building LOFO folds (default template)...")
    folds = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")

    results: dict = {
        "experiment": "exp1_embedding_baseline",
        "models": {},
        "device": args.device,
    }

    models_to_run: list[tuple[str, dict[str, str], str]] = []
    if args.model in ("bge", "all"):
        models_to_run.append(("bge-large-v1.5", BIENCODER_MODELS[0], "biencoder"))
    if args.model in ("gte", "all"):
        models_to_run.append(("gte-large-v1.5", BIENCODER_MODELS[1], "biencoder"))
    if args.model in ("deberta", "all"):
        models_to_run.append(("deberta-v3-nli", CROSSENCODER_MODEL, "crossencoder"))

    for model_name, model_config, model_type in models_to_run:
        logger.info("=" * 60)
        logger.info("Running %s (%s)", model_name, model_type)
        start_time = time.time()

        if model_type == "biencoder":
            fold_results = run_biencoder(model_config, folds, corpus, args.device)
        else:
            fold_results = run_crossencoder(model_config, folds, corpus, args.device)

        elapsed = time.time() - start_time
        logger.info("%s completed in %.1f seconds", model_name, elapsed)

        metrics_all198 = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
        metrics_fulltext = aggregate_lofo_metrics(fold_results, folds, track_filter="full-text")

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

        results["models"][model_name] = {
            "hf_id": model_config["hf_id"],
            "model_type": model_type,
            "elapsed_seconds": round(elapsed, 1),
            "all_198": metrics_all198,
            "full_text": metrics_fulltext,
            "per_fold": per_fold,
        }

        logger.info(
            "%s all-198: hit@1=%.3f [%.3f, %.3f], hit@5=%.3f [%.3f, %.3f]",
            model_name,
            metrics_all198["hit_at_1"]["mean"],
            metrics_all198["hit_at_1"]["ci_low"],
            metrics_all198["hit_at_1"]["ci_high"],
            metrics_all198["hit_at_5"]["mean"],
            metrics_all198["hit_at_5"]["ci_low"],
            metrics_all198["hit_at_5"]["ci_high"],
        )

    output_name = f"exp1_embedding_baseline{args.output_suffix}.json"
    save_results(results, output_name)
    logger.info("Experiment 1 complete.")


if __name__ == "__main__":
    main()
