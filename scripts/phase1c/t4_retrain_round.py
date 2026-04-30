"""T4: Ingest expert reviews, evaluate stopping criteria, optionally retrain.

Reads:
  - results/phase1c/round_N/review.json (expert-reviewed)
  - data/canary_items_for_labeling.json
  - results/phase1c/deployment_model/ (current model)

Writes:
  - results/phase1c/round_N/round_summary.json
  - results/phase1c/round_{N+1}/review.json (if continuing)
  - results/phase1c/deployment_model_round_{N+1}/ (if retraining)

Usage:
  python -m scripts.phase1c.t4_retrain_round --round 1
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter

from tract.active_learning.canary import evaluate_canary_accuracy
from tract.active_learning.review import ingest_reviews
from tract.active_learning.stopping import evaluate_stopping_criteria
from tract.config import (
    PHASE1C_AL_MAX_ROUNDS,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1C_RESULTS_DIR,
    PHASE1C_UNMAPPED_FRAMEWORKS,
    PROJECT_ROOT,
)
from tract.crosswalk.snapshot import take_snapshot
from tract.io import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest expert reviews and evaluate stopping criteria")
    parser.add_argument("--round", type=int, required=True, help="Round number to process")
    args = parser.parse_args()

    round_num = args.round
    logger.info("=== T4: Process Round %d Reviews ===", round_num)
    t_start = time.time()

    round_dir = PHASE1C_RESULTS_DIR / f"round_{round_num}"
    review_path = round_dir / "review.json"
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_path}")

    review_data = load_json(review_path)
    items = review_data["items"]
    logger.info("Loaded %d review items from round %d", len(items), round_num)

    reviewed_items = [i for i in items if i.get("review") is not None]
    status_counts = Counter(i["review"]["status"] for i in reviewed_items)
    logger.info("Review status: %s (%d unreviewed)", dict(status_counts), len(items) - len(reviewed_items))

    if not reviewed_items:
        logger.error("No items have been reviewed. Cannot proceed.")
        return

    take_snapshot(PHASE1C_CROSSWALK_DB_PATH, round_num, f"Pre-ingest snapshot for round {round_num}")

    new_links = ingest_reviews(review_data)
    logger.info("Ingested %d accepted/corrected predictions", len(new_links))

    # Evaluate canary accuracy
    canary_data = load_json(PROJECT_ROOT / "data" / "canary_items_for_labeling.json")
    canary_labels: dict[str, frozenset[str]] = {}
    fw_id_map = {v: k for k, v in PHASE1C_UNMAPPED_FRAMEWORKS.items()}
    for c in canary_data["canaries"]:
        fw_raw = c["framework"]
        fid = fw_id_map.get(fw_raw, fw_raw)
        cid = f"{fid}:{c['control_id']}"
        canary_labels[cid] = frozenset(c["expert_hub_ids"])

    canary_accuracy = evaluate_canary_accuracy(canary_labels, items)

    total_reviewed = len(reviewed_items)
    accepted = status_counts.get("accepted", 0) + status_counts.get("corrected", 0)
    acceptance_rate = accepted / total_reviewed if total_reviewed > 0 else 0.0

    unique_hubs = set()
    for link in new_links:
        unique_hubs.add(link.link["cre_id"])

    stopping = evaluate_stopping_criteria(
        acceptance_rate=acceptance_rate,
        canary_accuracy=canary_accuracy,
        unique_hubs_accepted=len(unique_hubs),
    )

    summary = {
        "round": round_num,
        "total_items": len(items),
        "reviewed": total_reviewed,
        "status_counts": dict(status_counts),
        "acceptance_rate": acceptance_rate,
        "canary_accuracy": canary_accuracy,
        "unique_hubs_accepted": len(unique_hubs),
        "new_links_ingested": len(new_links),
        "stopping": stopping,
    }
    atomic_write_json(summary, round_dir / "round_summary.json")

    if stopping["should_stop"]:
        logger.info("=== STOPPING: All criteria met after round %d ===", round_num)
    elif round_num >= PHASE1C_AL_MAX_ROUNDS:
        logger.info("=== MAX ROUNDS REACHED (%d): Stopping ===", PHASE1C_AL_MAX_ROUNDS)
    else:
        logger.info("=== CONTINUE: Criteria not met, round %d needed ===", round_num + 1)
        logger.info("To retrain and generate next round, run T1 and T2 with updated training data.")

    logger.info("T4 complete in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
