"""Re-score existing Phase 1B predictions with corrected evaluation.

Reads predictions.json from each fold, applies multi-label-aware scoring
with deduplicated evaluation corpus, and reports corrected metrics
without retraining.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    load_curated_links,
    score_predictions,
)
from tract.config import PHASE1B_RESULTS_DIR
from tract.io import atomic_write_json, load_json
from tract.training.evaluate import fold_stratified_bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def rescore_experiment(experiment_dir: Path) -> None:
    """Re-score one experiment's predictions with corrected methodology."""
    logger.info("Re-scoring: %s", experiment_dir.name)

    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    valid_hubs_by_text: dict[str, frozenset[str]] = {}
    for item in corpus:
        valid_hubs_by_text[item.control_text] = item.valid_hub_ids

    fold_results: list[dict] = []
    for fw_name in sorted(AI_FRAMEWORK_NAMES):
        fold_dir = experiment_dir / f"fold_{fw_name.replace(' ', '_')}"
        pred_path = fold_dir / "predictions.json"
        if not pred_path.exists():
            continue

        preds_data = load_json(pred_path)

        predictions = [p["predicted_top10"] for p in preds_data]
        ground_truth = [p["ground_truth_hub_id"] for p in preds_data]
        valid_sets = [
            valid_hubs_by_text.get(p["control_text"], frozenset({p["ground_truth_hub_id"]}))
            for p in preds_data
        ]

        old_hit1 = sum(
            1 for p, g in zip(predictions, ground_truth) if p and p[0] == g
        ) / len(predictions)
        metrics = score_predictions(predictions, ground_truth, valid_sets)
        hit1 = np.array([
            1.0 if pred and pred[0] in vs else 0.0
            for pred, vs in zip(predictions, valid_sets)
        ])

        logger.info(
            "  %s: hit@1=%.3f (was %.3f), n=%d",
            fw_name, metrics["hit_at_1"], old_hit1, len(predictions),
        )

        fold_results.append({
            "held_out_framework": fw_name,
            "metrics": metrics,
            "hit1_indicators": hit1.tolist(),
            "n_eval_items": len(predictions),
        })

    if fold_results:
        fold_hit1s = [np.array(r["hit1_indicators"]) for r in fold_results]
        aggregate = fold_stratified_bootstrap_ci(fold_hit1s)
        logger.info(
            "  CORRECTED AGGREGATE: hit@1=%.3f [%.3f, %.3f]",
            aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"],
        )

        corrected = {
            "scoring": "multi-label-aware",
            "aggregate_hit1": aggregate,
            "per_fold": {r["held_out_framework"]: r["metrics"] for r in fold_results},
        }
        atomic_write_json(corrected, experiment_dir / "corrected_metrics.json")


def main() -> None:
    for experiment in ["phase1b_primary", "phase1b_textaware", "ablation_a6_descriptions"]:
        exp_dir = PHASE1B_RESULTS_DIR / experiment
        if exp_dir.exists():
            rescore_experiment(exp_dir)
        else:
            logger.warning("Experiment dir not found: %s", exp_dir)

    print("\n" + "=" * 60)
    print("RE-SCORING COMPLETE")
    for experiment in ["phase1b_primary", "phase1b_textaware", "ablation_a6_descriptions"]:
        corrected_path = PHASE1B_RESULTS_DIR / experiment / "corrected_metrics.json"
        if corrected_path.exists():
            data = load_json(corrected_path)
            agg = data["aggregate_hit1"]
            print(f"\n  {experiment}:")
            print(f"    Corrected hit@1: {agg['mean']:.3f} [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}]")
            for fw, m in sorted(data["per_fold"].items()):
                print(f"    {fw}: hit@1={m['hit_at_1']:.3f} hit@5={m['hit_at_5']:.3f} MRR={m['mrr']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
