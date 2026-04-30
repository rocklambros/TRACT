"""T1: Fit T_lofo diagnostic + train deployment model.

Reads:
  - results/phase1c/similarities/fold_*.npz (from T0)
  - data/training/hub_links_curated.jsonl (training links)
  - data/processed/cre_hierarchy.json
  - results/phase1c/round_*/review.json (if --round > 1, previous AL links)

Writes:
  - results/phase1c/calibration/t_lofo_result.json
  - results/phase1c/calibration/diagnostic_ece.json
  - results/phase1c/holdout/calibration_links.json
  - results/phase1c/holdout/canary_links.json
  - results/phase1c/deployment_model/ (trained model + metadata)

Usage:
  python -m scripts.phase1c.t1_calibrate_and_train [--round N]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tract.active_learning.deploy import select_holdout
from tract.active_learning.train_deploy import prepare_deployment_training_data
from tract.calibration.diagnostics import bootstrap_ece, expected_calibration_error
from tract.calibration.temperature import calibrate_similarities, fit_t_lofo
from tract.config import (
    PHASE1C_DEPLOYMENT_MODEL_DIR,
    PHASE1C_ECE_BOOTSTRAP_N,
    PHASE1C_ECE_N_BINS,
    PHASE1C_RESULTS_DIR,
    PHASE1C_SIMILARITIES_DIR,
    PROCESSED_DIR,
)
from tract.active_learning.review import ingest_reviews
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.config import TrainingConfig
from tract.training.data import build_training_pairs, pairs_to_dataset
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.firewall import build_all_hub_texts
from tract.training.loop import save_checkpoint, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_gt_json(gt_json: np.ndarray, hub_ids: list[str]) -> list[list[int]]:
    """Parse JSON-encoded ground truth into hub column indices."""
    hub_id_to_idx = {hid: i for i, hid in enumerate(hub_ids)}
    valid_indices = []
    for gt_str in gt_json:
        valid_hub_ids = json.loads(str(gt_str))
        indices = [hub_id_to_idx[hid] for hid in valid_hub_ids if hid in hub_id_to_idx]
        if not indices:
            raise ValueError(f"No valid hub indices for ground truth: {gt_str}")
        valid_indices.append(indices)
    return valid_indices


def _fit_diagnostic_temperature() -> dict:
    """Fit T_lofo on pooled LOFO fold similarities."""
    cal_dir = PHASE1C_RESULTS_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(PHASE1C_SIMILARITIES_DIR.glob("fold_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No fold NPZ files in {PHASE1C_SIMILARITIES_DIR}")

    fold_sims: dict[str, np.ndarray] = {}
    fold_valid_indices: dict[str, list[list[int]]] = {}

    for npz_path in npz_files:
        fold_name = npz_path.stem.replace("fold_", "").replace("_", " ")
        data = np.load(npz_path, allow_pickle=False)
        hub_ids = list(data["hub_ids"])
        sims = data["sims"]
        gt_json = data["gt_json"]

        valid_indices = _parse_gt_json(gt_json, hub_ids)
        fold_sims[fold_name] = sims
        fold_valid_indices[fold_name] = valid_indices
        logger.info("Loaded fold %s: %d items, %d hubs", fold_name, len(sims), len(hub_ids))

    result = fit_t_lofo(fold_sims, fold_valid_indices)

    serializable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            serializable[k] = v
    atomic_write_json(serializable, cal_dir / "t_lofo_result.json")

    all_sims = np.concatenate([fold_sims[n] for n in sorted(fold_sims.keys())])
    all_valid = []
    for n in sorted(fold_sims.keys()):
        all_valid.extend(fold_valid_indices[n])

    probs = calibrate_similarities(all_sims, result["temperature"])
    confidences = np.array([float(probs[i].max()) for i in range(len(probs))])
    accuracies = np.array([
        1.0 if int(probs[i].argmax()) in valid else 0.0
        for i, valid in enumerate(all_valid)
    ])

    ece = expected_calibration_error(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS)
    ece_ci = bootstrap_ece(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS, n_bootstrap=PHASE1C_ECE_BOOTSTRAP_N)

    diagnostic_ece = {
        "ece": float(ece),
        "ece_ci": ece_ci,
        "t_lofo": float(result["temperature"]),
        "n_items": len(all_sims),
    }
    atomic_write_json(diagnostic_ece, cal_dir / "diagnostic_ece.json")
    logger.info("Diagnostic ECE: %.4f [%.4f, %.4f]", ece, ece_ci["ci_low"], ece_ci["ci_high"])

    return result


def _select_and_save_holdout() -> tuple[list, list, list]:
    """Select 440 holdout, save link records, return (cal, canary, remaining)."""
    holdout_dir = PHASE1C_RESULTS_DIR / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    tiered_links, _ = load_and_filter_curated_links()
    cal_links, canary_links, remaining = select_holdout(tiered_links)

    cal_records = [{"link": l.link, "tier": l.tier.value} for l in cal_links]
    canary_records = [{"link": l.link, "tier": l.tier.value} for l in canary_links]
    atomic_write_json(cal_records, holdout_dir / "calibration_links.json")
    atomic_write_json(canary_records, holdout_dir / "canary_links.json")

    logger.info(
        "Holdout: %d calibration, %d canary, %d remaining for training",
        len(cal_links), len(canary_links), len(remaining),
    )
    return cal_links, canary_links, remaining


def _train_deployment_model(
    remaining_links: list,
    hierarchy: CREHierarchy,
) -> None:
    """Train deployment model on all data minus holdout."""
    PHASE1C_DEPLOYMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    hub_texts = build_all_hub_texts(hierarchy, excluded_framework=None)
    pairs = build_training_pairs(remaining_links, hub_texts, excluded_framework=None)
    dataset = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=3)

    config = TrainingConfig(name="phase1c_deployment")

    model = train_model(config, dataset, PHASE1C_DEPLOYMENT_MODEL_DIR)

    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"

    metrics = {"n_training_pairs": len(pairs), "n_links": len(remaining_links)}
    save_checkpoint(model, config, metrics, PHASE1C_DEPLOYMENT_MODEL_DIR / "model", git_sha)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Deployment model saved to %s", PHASE1C_DEPLOYMENT_MODEL_DIR)


def _load_al_links(up_to_round: int) -> list:
    """Load accepted/corrected links from all previous AL rounds."""
    from tract.training.data_quality import TieredLink
    al_links: list[TieredLink] = []
    for r in range(1, up_to_round):
        review_path = PHASE1C_RESULTS_DIR / f"round_{r}" / "review.json"
        if not review_path.exists():
            logger.warning("Round %d review.json not found at %s, skipping", r, review_path)
            continue
        review_data = load_json(review_path)
        round_links = ingest_reviews(review_data)
        al_links.extend(round_links)
        logger.info("Loaded %d AL links from round %d", len(round_links), r)
    return al_links


def main() -> None:
    parser = argparse.ArgumentParser(description="T1: Calibrate T_lofo + Train Deployment Model")
    parser.add_argument("--round", type=int, default=1, help="Current round (>1 incorporates previous AL links)")
    args = parser.parse_args()
    round_num = args.round

    logger.info("=== T1: Calibrate T_lofo + Train Deployment Model (round %d) ===", round_num)
    t0 = time.time()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))

    t_lofo_result = _fit_diagnostic_temperature()
    logger.info("T_lofo = %.4f", t_lofo_result["temperature"])

    cal_links, canary_links, remaining = _select_and_save_holdout()

    if round_num > 1:
        al_links = _load_al_links(round_num)
        logger.info("Adding %d AL links to %d training links", len(al_links), len(remaining))
        remaining = list(remaining) + al_links

    _train_deployment_model(remaining, hierarchy)

    logger.info("T1 complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
