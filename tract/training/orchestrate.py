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
    EvalItem,
    build_evaluation_corpus,
    load_curated_links,
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
from tract.training.data_quality import TieredLink

logger = logging.getLogger(__name__)

# Written per fold, and the only artifact carrying the per-item hit@1
# indicators needed to micro-average across folds.
FOLD_RESULT_FILENAME = "fold_result.json"


def _free_gpu_memory() -> None:
    """Release the baseline model's VRAM before training allocates its own."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
    tiered_links: list[TieredLink],
    hierarchy: CREHierarchy,
    eval_items: list[EvalItem],
    hub_ids: list[str],
    output_dir: Path,
    standard_sections: dict[str, list[str]] | None = None,
    include_zero_shot: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one LOFO fold. Returns fold result dict.

    Args:
        include_zero_shot: Also evaluate the untrained base model on this fold,
            producing a per-item indicator array paired with the trained one.
    """
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

    include_standards = config.hub_rep_format == "path+name+standards"
    if include_standards and standard_sections is None:
        # Previously both builds below were called with an identical argument
        # list, so hub_texts == base_hub_texts, every appended slice was empty,
        # and the breach check was unreachable. Failing loudly is better than
        # silently producing a format the config asked for and never built.
        raise ValueError(
            "hub_rep_format='path+name+standards' requires standard_sections; "
            "none were supplied, so the standards format cannot be built and "
            "the firewall would have nothing to check"
        )

    hub_texts = build_all_hub_texts(
        hierarchy,
        excluded_framework=held_out_framework,
        include_description=include_desc,
        descriptions=descriptions,
        include_standards=include_standards,
        standard_sections=standard_sections,
    )

    # Any augmentation makes the appended slice the leakage surface, so build the
    # plain "path | name" base to subtract from it. Descriptions are CRE-authored
    # and were previously left unfirewalled, but passing base_hub_texts=None for
    # them does not merely skip the check -- assert_firewall then derives hub
    # names from the full text, so the description becomes part of the "hub name"
    # and a control text leaked into it is skipped as a hub-name match. The base
    # must be the base format, not the format minus one augmentation.
    base_hub_texts = None
    if include_standards or include_desc:
        base_hub_texts = build_all_hub_texts(
            hierarchy,
            excluded_framework=held_out_framework,
            include_description=False,
            include_standards=False,
        )
    assert_firewall(hub_texts, eval_items, held_out_framework, base_hub_texts)

    pairs = build_training_pairs(tiered_links, hub_texts, excluded_framework=held_out_framework)
    dataset = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=config.hard_negatives)

    fold_output = output_dir / f"fold_{held_out_framework.replace(' ', '_')}"
    fold_output.mkdir(parents=True, exist_ok=True)

    zero_shot: dict[str, Any] | None = None
    if include_zero_shot:
        # Measure the untrained base model on the SAME eval items, hub ids and
        # firewalled hub texts, in the same process, before training touches
        # anything. paired_bootstrap_delta requires the two indicator arrays to
        # be aligned item by item; a baseline computed in a separate job cannot
        # guarantee that alignment, and an unpaired delta is what put the
        # pre-registered gate inside its own confidence interval.
        from sentence_transformers import SentenceTransformer

        logger.info("Fold %s: measuring zero-shot baseline", held_out_framework)
        base_model = SentenceTransformer(config.base_model)
        base_model.max_seq_length = config.max_seq_length
        zs_metrics, _zs_predictions, zs_hit1 = evaluate_on_fold(
            base_model, eval_items, hub_ids, hub_texts,
        )
        zero_shot = {
            "metrics": zs_metrics,
            "hit1_indicators": [int(x) for x in zs_hit1],
        }
        logger.info("Fold %s zero-shot: hit@1=%.3f", held_out_framework,
                    zs_metrics["hit_at_1"])
        del base_model
        _free_gpu_memory()

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

    result: dict[str, Any] = {
        "held_out_framework": held_out_framework,
        "metrics": metrics,
        "predictions": predictions,
        "hit1_indicators": hit1_indicators,
        "n_eval_items": len(eval_items),
        "n_training_pairs": len(pairs),
        "elapsed_s": elapsed,
    }
    if zero_shot is not None:
        result["zero_shot"] = zero_shot

    # Persist the per-item indicators, not just the fold's summary metrics.
    # Averaging five fold summaries is a MACRO average: it weights a 6-item fold
    # the same as a 60-item one. The aggregate hit@1 TRACT reports is a MICRO
    # average, which needs the raw indicator array from every fold. Any path that
    # only keeps metrics.json cannot reconstruct it, and the model card now
    # refuses to build without it.
    fold_record = {k: v for k, v in result.items() if k != "predictions"}
    fold_record["hit1_indicators"] = [int(x) for x in hit1_indicators]
    fold_record["git_sha"] = _get_git_sha()
    atomic_write_json(fold_record, fold_output / FOLD_RESULT_FILENAME)

    return result


def load_fold_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load every persisted per-fold record under results_dir.

    Sorted by held-out framework so aggregation is order-independent and
    therefore reproducible regardless of the order folds finished in.
    """
    records = []
    for path in sorted(results_dir.glob(f"fold_*/{FOLD_RESULT_FILENAME}")):
        record = load_json(path)
        missing = {"held_out_framework", "hit1_indicators", "n_eval_items"} - set(record)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}. It was written by a version "
                "that dropped the per-item indicators, so the aggregate hit@1 "
                "cannot be micro-averaged from it."
            )
        if len(record["hit1_indicators"]) != record["n_eval_items"]:
            raise ValueError(
                f"{path}: {len(record['hit1_indicators'])} indicators for "
                f"{record['n_eval_items']} eval items. The fold record is "
                "internally inconsistent; refusing to aggregate it."
            )
        records.append(record)
    if not records:
        raise ValueError(
            f"No {FOLD_RESULT_FILENAME} files under {results_dir}. Nothing to "
            "aggregate: either no fold completed or the results were not collected."
        )
    return records


def aggregate_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average hit@1 across folds with a fold-stratified bootstrap CI.

    Pools per-item indicators rather than averaging per-fold rates, so each
    eval item carries equal weight. With TRACT's fold sizes the difference is
    not cosmetic: the smallest fold would otherwise count for 1/5 of the
    headline number instead of its share of the items.
    """
    fold_hit1s = [np.asarray(r["hit1_indicators"], dtype=float) for r in fold_results]
    aggregate: dict[str, Any] = dict(fold_stratified_bootstrap_ci(fold_hit1s))

    # Report the macro figure alongside it. They differ only through fold-size
    # imbalance, so a wide gap is a signal about the folds, not a second result.
    macro = float(np.mean([float(np.mean(f)) for f in fold_hit1s]))
    aggregate["macro_mean"] = macro
    aggregate["n_folds"] = len(fold_hit1s)
    aggregate["fold_sizes"] = {
        r["held_out_framework"]: int(r["n_eval_items"]) for r in fold_results
    }
    logger.info(
        "AGGREGATE hit@1 (micro): %.4f [%.4f, %.4f] over n=%d | macro: %.4f",
        aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"],
        aggregate["n_total"], macro,
    )
    return aggregate


def run_experiment(config: TrainingConfig) -> dict[str, Any]:
    """Run a full LOFO experiment with the given config."""
    logger.info("Starting experiment: %s", config.name)
    exp_start = time.time()

    tiered_links, raw_hash = load_and_filter_curated_links()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())

    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    eval_by_fw: dict[str, list[EvalItem]] = {}
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

    aggregate = aggregate_fold_results(fold_results)

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
