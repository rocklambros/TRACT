"""Multi-fold, multi-config experiment runner.

Orchestrates the full Phase 1B pipeline:
1. Load and filter training data
2. For each LOFO fold:
   a. Build firewalled hub texts
   b. Generate training pairs with hard negatives
   c. Train LoRA model
   d. Evaluate on held-out framework
3. Aggregate across folds with fold-stratified bootstrap CIs
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    load_curated_links,
    load_opencre_cres,
    score_predictions,
)
from tract.config import (
    PHASE1B_RESULTS_DIR,
    PROCESSED_DIR,
)
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.config import TrainingConfig
from tract.training.data import (
    build_training_pairs,
    pairs_to_dataset,
)
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.evaluate import (
    evaluate_on_fold,
    fold_stratified_bootstrap_ci,
)
from tract.training.firewall import assert_firewall, build_all_hub_texts
from tract.training.loop import save_checkpoint, train_model

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_single_fold(
    config: TrainingConfig,
    held_out_framework: str,
    tiered_links: list,
    hierarchy: CREHierarchy,
    eval_items: list,
    hub_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate one LOFO fold. Returns fold result dict."""
    logger.info("=== FOLD: %s ===", held_out_framework)
    fold_start = time.time()

    include_desc = config.hub_rep_format == "path+name+desc"
    descriptions = None
    if include_desc:
        desc_data = load_json(PROCESSED_DIR / "hub_descriptions_reviewed.json")
        descriptions = {
            hid: d["description"]
            for hid, d in desc_data.get("descriptions", {}).items()
        }

    hub_texts = build_all_hub_texts(
        hierarchy,
        excluded_framework=held_out_framework,
        include_description=include_desc,
        descriptions=descriptions,
    )

    # Descriptions are CRE-authored and don't need firewalling.
    # Only build base_hub_texts when standards are included (framework-derived).
    include_standards = config.hub_rep_format == "path+name+standards"
    base_hub_texts = None
    if include_standards:
        base_hub_texts = build_all_hub_texts(
            hierarchy,
            excluded_framework=held_out_framework,
            include_description=include_desc,
            descriptions=descriptions,
        )
    assert_firewall(hub_texts, eval_items, held_out_framework, base_hub_texts)

    pairs = build_training_pairs(tiered_links, hub_texts, excluded_framework=held_out_framework)
    dataset = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=config.hard_negatives)

    fold_output = output_dir / f"fold_{held_out_framework.replace(' ', '_')}"
    fold_output.mkdir(parents=True, exist_ok=True)

    model = train_model(config, dataset, fold_output)

    metrics, predictions, hit1_indicators = evaluate_on_fold(
        model, eval_items, hub_ids, hub_texts,
    )
    logger.info("Fold %s: hit@1=%.3f, hit@5=%.3f, MRR=%.3f, NDCG@10=%.3f",
                held_out_framework, metrics["hit_at_1"], metrics["hit_at_5"],
                metrics["mrr"], metrics["ndcg_at_10"])

    save_checkpoint(model, config, metrics, fold_output / "model", _get_git_sha())

    pred_data = []
    for item, pred in zip(eval_items, predictions):
        pred_data.append({
            "control_text": item.control_text,
            "ground_truth_hub_id": item.ground_truth_hub_id,
            "predicted_top10": pred[:10],
            "framework": item.framework_name,
        })
    atomic_write_json(pred_data, fold_output / "predictions.json")
    atomic_write_json(metrics, fold_output / "metrics.json")

    elapsed = time.time() - fold_start
    logger.info("Fold %s complete in %.1fs", held_out_framework, elapsed)

    return {
        "held_out_framework": held_out_framework,
        "metrics": metrics,
        "predictions": predictions,
        "hit1_indicators": hit1_indicators,
        "n_eval_items": len(eval_items),
        "n_training_pairs": len(pairs),
        "elapsed_s": elapsed,
    }


def run_experiment(config: TrainingConfig) -> dict[str, Any]:
    """Run a full LOFO experiment with the given config."""
    logger.info("Starting experiment: %s", config.name)
    exp_start = time.time()

    tiered_links, raw_hash = load_and_filter_curated_links()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())

    cres = load_opencre_cres()
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    eval_by_fw: dict[str, list] = {}
    for item in corpus:
        eval_by_fw.setdefault(item.framework_name, []).append(item)

    output_dir = PHASE1B_RESULTS_DIR / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, Any]] = []
    for fw_name in sorted(AI_FRAMEWORK_NAMES):
        fw_items = eval_by_fw.get(fw_name, [])
        if not fw_items:
            logger.warning("No eval items for %s, skipping", fw_name)
            continue

        result = run_single_fold(
            config=config,
            held_out_framework=fw_name,
            tiered_links=tiered_links,
            hierarchy=hierarchy,
            eval_items=fw_items,
            hub_ids=hub_ids,
            output_dir=output_dir,
        )
        fold_results.append(result)

    fold_hit1s = [np.array(r["hit1_indicators"]) for r in fold_results]
    aggregate = fold_stratified_bootstrap_ci(fold_hit1s)
    logger.info("AGGREGATE hit@1: %.3f [%.3f, %.3f]",
                aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"])

    experiment_result = {
        "config": config.to_dict(),
        "aggregate_hit1": aggregate,
        "per_fold": {r["held_out_framework"]: r["metrics"] for r in fold_results},
        "raw_hash": raw_hash,
        "git_sha": _get_git_sha(),
        "total_elapsed_s": time.time() - exp_start,
    }
    atomic_write_json(experiment_result, output_dir / "aggregate_metrics.json")

    return experiment_result
