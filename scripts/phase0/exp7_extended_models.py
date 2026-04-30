"""Experiment 7: Extended embedding models for Stage 0R base model selection.

Evaluates four larger / instruction-tuned encoders on the hub assignment task:
- Alibaba-NLP/gte-Qwen2-1.5B-instruct (1.5B params, 1536 dim)
- intfloat/e5-mistral-7b-instruct (7B params, 4096 dim)
- nvidia/NV-Embed-v2 (7B params, 4096 dim)
- Salesforce/SFR-Embedding-2_R (7B params, 4096 dim)

Uses curated (audit-corrected) training links by default.
LOFO cross-validation with hub firewall on 5 AI frameworks.
WandB tracking for all runs.
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
    finish_wandb,
    init_wandb,
    load_curated_links,
    load_opencre_cres,
    load_parsed_controls,
    log_aggregate_metrics,
    log_fold_metrics,
    log_wandb_summary_table,
    save_results,
    score_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model Config ────────────────────────────────────────────────────────────

INSTRUCTION_TASK: Final[str] = (
    "Given a security control description, retrieve the most relevant "
    "CRE (Common Requirements Enumeration) hub category"
)

EXTENDED_MODELS: Final[list[dict]] = [
    {
        "name": "gte-qwen2-1.5b",
        "hf_id": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "instruction": True,
        "trust_remote_code": True,
        "batch_size": 8,
    },
    {
        "name": "e5-mistral-7b",
        "hf_id": "intfloat/e5-mistral-7b-instruct",
        "instruction": True,
        "trust_remote_code": False,
        "batch_size": 4,
    },
    {
        "name": "nv-embed-v2",
        "hf_id": "nvidia/NV-Embed-v2",
        "instruction": True,
        "trust_remote_code": True,
        "batch_size": 4,
    },
    {
        "name": "sfr-embedding-2",
        "hf_id": "Salesforce/SFR-Embedding-2_R",
        "instruction": True,
        "trust_remote_code": False,
        "batch_size": 4,
    },
]

MODEL_NAME_TO_CONFIG: Final[dict[str, dict]] = {m["name"]: m for m in EXTENDED_MODELS}


# ── Ranking ─────────────────────────────────────────────────────────────────


def rank_by_cosine_similarity(
    control_embedding: np.ndarray,
    hub_embeddings: np.ndarray,
    hub_ids: list[str],
) -> list[str]:
    control_norm = control_embedding / (np.linalg.norm(control_embedding) + 1e-10)
    hub_norms = hub_embeddings / (np.linalg.norm(hub_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = hub_norms @ control_norm
    ranked_indices = np.argsort(-similarities)
    return [hub_ids[i] for i in ranked_indices]


# ── Bi-Encoder Pipeline (instruction-aware) ──────────────────────────────


def _prepare_query_texts(
    texts: list[str],
    model_config: dict,
) -> list[str]:
    """Prepend instruction prefix for instruction-tuned models.

    E5-Mistral uses 'Instruct: ...\\nQuery: ...' format.
    GTE-Qwen2 and NV-Embed use SentenceTransformer prompt kwargs.
    SFR-Embedding-2 uses E5-Mistral format.
    """
    hf_id = model_config["hf_id"]
    if "e5-mistral" in hf_id.lower() or "sfr-embedding" in hf_id.lower():
        return [f"Instruct: {INSTRUCTION_TASK}\nQuery: {t}" for t in texts]
    return texts


def _patch_dynamic_cache() -> None:
    """Patch DynamicCache for transformers>=4.45 compat with older custom models."""
    try:
        from transformers import DynamicCache
        if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
            DynamicCache.get_usable_length = DynamicCache.get_seq_length
    except ImportError:
        pass


def run_extended_biencoder(
    model_config: dict,
    folds: list[LOFOFold],
    corpus: list[EvalItem],
    device: str,
) -> list[dict[int, list[str]]]:
    """Run instruction-aware bi-encoder evaluation across LOFO folds."""
    from sentence_transformers import SentenceTransformer

    _patch_dynamic_cache()

    hf_id = model_config["hf_id"]
    batch_size = model_config["batch_size"]
    trust_remote = model_config.get("trust_remote_code", False)

    logger.info("Loading model: %s", hf_id)
    model = SentenceTransformer(
        hf_id, device=device, trust_remote_code=trust_remote,
    )

    prompt_kwargs: dict[str, str] = {}
    if model_config.get("instruction") and "gte-qwen2" in model_config["name"].lower():
        prompt_kwargs["prompt"] = f"Instruct: {INSTRUCTION_TASK}\nQuery: "
    elif model_config.get("instruction") and "nv-embed" in model_config["name"].lower():
        prompt_kwargs["prompt"] = f"Instruct: {INSTRUCTION_TASK}\nQuery: "

    fold_results: list[dict[int, list[str]]] = []

    for fold in folds:
        logger.info("Fold: held out %s (%d items)", fold.held_out_framework, len(fold.eval_items))

        hub_texts_ordered = [fold.hub_texts[hid] for hid in fold.hub_ids]
        hub_embeddings = model.encode(
            hub_texts_ordered,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        hub_embeddings = np.array(hub_embeddings)

        control_texts = [item.control_text for item in fold.eval_items]
        query_texts = _prepare_query_texts(control_texts, model_config)

        control_embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            **prompt_kwargs,
        )
        control_embeddings = np.array(control_embeddings)

        predictions: dict[int, list[str]] = {}
        for i, ctrl_emb in enumerate(control_embeddings):
            ranked = rank_by_cosine_similarity(ctrl_emb, hub_embeddings, fold.hub_ids)
            predictions[i] = ranked[:20]

        fold_results.append(predictions)

    return fold_results


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 7: Extended embedding models for Stage 0R",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_NAME_TO_CONFIG.keys()) + ["all"],
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
        help="Suffix for output filename",
    )
    parser.add_argument(
        "--template",
        choices=["default", "path", "both"],
        default="both",
        help="Hub text template (default: both)",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Use original uncurated links",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB tracking",
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)

    curated = not args.original
    if curated:
        links = load_curated_links()
        logger.info("Using CURATED links (audit-corrected)")
    else:
        links = extract_hub_standard_links(cres)
        logger.info("Using ORIGINAL links (no corrections)")

    templates = ["default", "path"] if args.template == "both" else [args.template]

    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    models_to_run: list[dict] = []
    if args.model == "all":
        models_to_run = list(EXTENDED_MODELS)
    else:
        models_to_run = [MODEL_NAME_TO_CONFIG[args.model]]

    wandb_run = None
    if not args.no_wandb:
        wandb_run = init_wandb(
            experiment_name=f"exp7_extended_models{args.output_suffix}",
            config={
                "models": [m["name"] for m in models_to_run],
                "templates": templates,
                "device": args.device,
                "n_eval_items": len(corpus),
            },
            tags=["exp7", "extended-models", "base-model-selection"],
            curated=curated,
        )

    results: dict = {
        "experiment": "exp7_extended_models",
        "curated": curated,
        "models": {},
        "device": args.device,
    }

    for model_config in models_to_run:
        model_name = model_config["name"]
        logger.info("=" * 60)
        logger.info("Running %s (%s)", model_name, model_config["hf_id"])
        start_time = time.time()

        try:
            model_results: dict = {
                "hf_id": model_config["hf_id"],
                "instruction_tuned": model_config.get("instruction", False),
                "status": "ok",
                "templates": {},
            }

            for template in templates:
                logger.info("  Template: %s", template)
                folds = build_lofo_folds(
                    tree, links, corpus, AI_FRAMEWORK_NAMES, template=template,
                )

                fold_results = run_extended_biencoder(model_config, folds, corpus, args.device)

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

                    log_fold_metrics(
                        wandb_run, f"{model_name}/{template}",
                        fold.held_out_framework, fold_metrics,
                    )

                log_aggregate_metrics(wandb_run, f"{model_name}/{template}", metrics_all, prefix="all")
                log_aggregate_metrics(wandb_run, f"{model_name}/{template}", metrics_ft, prefix="full_text")

                model_results["templates"][template] = {
                    "all": metrics_all,
                    "full_text": metrics_ft,
                    "per_fold": per_fold,
                }

                logger.info(
                    "  %s/%s all: hit@1=%.3f [%.3f, %.3f], hit@5=%.3f [%.3f, %.3f]",
                    model_name, template,
                    metrics_all["hit_at_1"]["mean"],
                    metrics_all["hit_at_1"]["ci_low"],
                    metrics_all["hit_at_1"]["ci_high"],
                    metrics_all["hit_at_5"]["mean"],
                    metrics_all["hit_at_5"]["ci_low"],
                    metrics_all["hit_at_5"]["ci_high"],
                )

            model_results["elapsed_seconds"] = round(time.time() - start_time, 1)
            results["models"][model_name] = model_results

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning("OOM for %s: %s — skipping", model_name, e)
                results["models"][model_name] = {
                    "hf_id": model_config["hf_id"],
                    "status": "oom",
                    "error": str(e),
                    "elapsed_seconds": round(time.time() - start_time, 1),
                }
                torch.cuda.empty_cache()
            else:
                raise

    summary_rows = []
    for mname, mdata in results["models"].items():
        if mdata.get("status") != "ok":
            continue
        best_template = list(mdata["templates"].keys())[0]
        best_metrics = mdata["templates"][best_template]["all"]
        summary_rows.append({
            "model": mname,
            "template": best_template,
            "hit_at_1": round(best_metrics["hit_at_1"]["mean"], 3),
            "hit_at_5": round(best_metrics["hit_at_5"]["mean"], 3),
            "mrr": round(best_metrics["mrr"]["mean"], 3),
            "ndcg_at_10": round(best_metrics["ndcg_at_10"]["mean"], 3),
            "elapsed_s": mdata["elapsed_seconds"],
        })
    log_wandb_summary_table(wandb_run, summary_rows, table_name="exp7_comparison")
    finish_wandb(wandb_run)

    output_name = f"exp7_extended_models{args.output_suffix}.json"
    save_results(results, output_name)
    logger.info("Experiment 7 complete.")


if __name__ == "__main__":
    main()
