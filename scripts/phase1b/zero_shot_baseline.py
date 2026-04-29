"""Compute firewalled zero-shot BGE baseline with Phase 1B hub format.

Uses the same LOFO protocol, hub format ("{path} | {hub_name}"), and
corrected multi-label evaluation methodology as Phase 1B training runs,
but with no fine-tuning. Provides an apples-to-apples baseline for the
gate comparison.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    load_curated_links,
)
from tract.config import PHASE1B_RESULTS_DIR, PROCESSED_DIR
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.evaluate import (
    evaluate_on_fold,
    fold_stratified_bootstrap_ci,
)
from tract.training.firewall import build_all_hub_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    start = time.time()

    logger.info("Loading data...")
    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    eval_by_fw: dict[str, list] = {}
    for item in corpus:
        eval_by_fw.setdefault(item.framework_name, []).append(item)

    logger.info("Loading BGE-large-v1.5 (zero-shot, no fine-tuning)...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    output_dir = PHASE1B_RESULTS_DIR / "zero_shot_firewalled_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict] = []
    for fw_name in sorted(AI_FRAMEWORK_NAMES):
        fw_items = eval_by_fw.get(fw_name, [])
        if not fw_items:
            continue

        hub_texts = build_all_hub_texts(hierarchy, excluded_framework=fw_name)

        metrics, predictions, hit1_indicators = evaluate_on_fold(
            model, fw_items, hub_ids, hub_texts,
        )
        logger.info(
            "Fold %s: hit@1=%.3f, hit@5=%.3f, MRR=%.3f (n=%d)",
            fw_name, metrics["hit_at_1"], metrics["hit_at_5"],
            metrics["mrr"], len(fw_items),
        )

        fold_results.append({
            "held_out_framework": fw_name,
            "metrics": metrics,
            "hit1_indicators": hit1_indicators.tolist(),
            "n_eval_items": len(fw_items),
        })

        pred_data = []
        for item, pred in zip(fw_items, predictions):
            pred_data.append({
                "control_text": item.control_text,
                "ground_truth_hub_id": item.ground_truth_hub_id,
                "predicted_top10": pred[:10],
                "framework": item.framework_name,
            })
        fold_dir = output_dir / f"fold_{fw_name.replace(' ', '_')}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(pred_data, fold_dir / "predictions.json")
        atomic_write_json(metrics, fold_dir / "metrics.json")

    fold_hit1s = [np.array(r["hit1_indicators"]) for r in fold_results]
    aggregate = fold_stratified_bootstrap_ci(fold_hit1s)
    elapsed = time.time() - start
    logger.info(
        "AGGREGATE hit@1: %.3f [%.3f, %.3f] (%.1fs)",
        aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"], elapsed,
    )

    result = {
        "model": "BAAI/bge-large-en-v1.5",
        "hub_format": "path+name",
        "firewalled": True,
        "multi_label_aware": True,
        "aggregate_hit1": aggregate,
        "per_fold": {r["held_out_framework"]: r["metrics"] for r in fold_results},
        "elapsed_s": elapsed,
    }
    atomic_write_json(result, output_dir / "aggregate_metrics.json")

    print("\n" + "=" * 60)
    print("FIREWALLED ZERO-SHOT BASELINE COMPLETE")
    print(f"  Aggregate hit@1: {aggregate['mean']:.3f} [{aggregate['ci_low']:.3f}, {aggregate['ci_high']:.3f}]")
    for r in sorted(fold_results, key=lambda x: x["held_out_framework"]):
        m = r["metrics"]
        print(f"  {r['held_out_framework']}: hit@1={m['hit_at_1']:.3f} hit@5={m['hit_at_5']:.3f} MRR={m['mrr']:.3f}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
