"""Post-training diagnostic: compare model predictions vs Phase 0R Sonnet.

Categorizes each of the 197 eval items into:
- Both correct: High-confidence assignments
- Both wrong: Genuinely hard examples
- Model correct, Sonnet wrong: Model learned something Sonnet missed
- Model wrong, Sonnet correct: Model's specific failure modes

Usage:
    python -m scripts.phase1b.llm_comparison --model-results results/phase1b/phase1b_primary/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tract.io import load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model_predictions(results_dir: Path) -> dict[str, dict]:
    """Load per-item predictions from all fold directories."""
    predictions: dict[str, dict] = {}
    for fold_dir in sorted(results_dir.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        pred_file = fold_dir / "predictions.json"
        if not pred_file.exists():
            continue
        preds = load_json(pred_file)
        for item in preds:
            key = f"{item['framework']}:{item.get('control_text', '')[:50]}"
            predictions[key] = item
    return predictions


def compare_predictions(
    model_preds: dict[str, dict],
    sonnet_preds: dict[str, dict],
) -> dict[str, list[dict]]:
    """Compare model vs Sonnet predictions per item."""
    categories: dict[str, list[dict]] = {
        "both_correct": [],
        "both_wrong": [],
        "model_only_correct": [],
        "sonnet_only_correct": [],
    }

    for key, model_item in model_preds.items():
        gt = model_item["ground_truth_hub_id"]
        model_top1 = model_item["predicted_top10"][0] if model_item["predicted_top10"] else None

        sonnet_item = sonnet_preds.get(key)
        if sonnet_item is None:
            continue
        sonnet_top1 = sonnet_item.get("predicted_top10", [None])[0]

        model_correct = model_top1 == gt
        sonnet_correct = sonnet_top1 == gt

        entry = {
            "key": key,
            "ground_truth": gt,
            "model_prediction": model_top1,
            "sonnet_prediction": sonnet_top1,
        }

        if model_correct and sonnet_correct:
            categories["both_correct"].append(entry)
        elif not model_correct and not sonnet_correct:
            categories["both_wrong"].append(entry)
        elif model_correct and not sonnet_correct:
            categories["model_only_correct"].append(entry)
        else:
            categories["sonnet_only_correct"].append(entry)

    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Model vs Sonnet prediction comparison")
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--sonnet-results", type=Path, default=None)
    args = parser.parse_args()

    model_preds = load_model_predictions(args.model_results)
    logger.info("Loaded %d model predictions", len(model_preds))

    if args.sonnet_results and args.sonnet_results.exists():
        sonnet_preds = load_model_predictions(args.sonnet_results)
    else:
        logger.info("No Sonnet results provided — looking for Phase 0R results")
        sonnet_dir = Path("results/phase0r")
        if sonnet_dir.exists():
            sonnet_preds = load_model_predictions(sonnet_dir)
        else:
            logger.warning("No Sonnet predictions found. Skipping comparison.")
            return

    logger.info("Loaded %d Sonnet predictions", len(sonnet_preds))

    categories = compare_predictions(model_preds, sonnet_preds)
    total = sum(len(v) for v in categories.values())

    logger.info("=" * 60)
    logger.info("MODEL vs SONNET COMPARISON (%d items)", total)
    for cat, items in categories.items():
        logger.info("  %s: %d (%.1f%%)", cat, len(items), 100 * len(items) / max(total, 1))
    logger.info("=" * 60)

    if categories["model_only_correct"]:
        logger.info("\nModel-only correct (model learned what Sonnet missed):")
        for item in categories["model_only_correct"][:10]:
            logger.info("  %s -> model=%s (correct), sonnet=%s",
                        item["key"][:60], item["model_prediction"], item["sonnet_prediction"])

    if categories["sonnet_only_correct"]:
        logger.info("\nSonnet-only correct (model failure modes):")
        for item in categories["sonnet_only_correct"][:10]:
            logger.info("  %s -> model=%s, sonnet=%s (correct)",
                        item["key"][:60], item["model_prediction"], item["sonnet_prediction"])


if __name__ == "__main__":
    main()
