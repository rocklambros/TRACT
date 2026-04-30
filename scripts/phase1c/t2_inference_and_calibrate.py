"""T2: Deployment model inference + production calibration + review.json generation.

Reads:
  - results/phase1c/deployment_model/model/model/ (trained deployment model)
  - results/phase1c/holdout/ (calibration and canary links)
  - data/processed/cre_hierarchy.json
  - data/processed/frameworks/*.json (unmapped controls)
  - data/canary_items_for_labeling.json (expert-labeled AI canaries)
  - tests/fixtures/ood_synthetic_texts.json

Writes:
  - results/phase1c/calibration/t_deploy_result.json
  - results/phase1c/calibration/ece_gate.json
  - results/phase1c/calibration/conformal.json
  - results/phase1c/calibration/ood.json
  - results/phase1c/calibration/ks_test.json
  - results/phase1c/calibration/global_threshold.json
  - results/phase1c/round_1/review.json
  - results/phase1c/similarities/deployment_*.npz

Usage:
  python -m scripts.phase1c.t2_inference_and_calibrate
"""
from __future__ import annotations

import gc
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tract.active_learning.deploy import holdout_to_eval
from tract.active_learning.model_io import load_deployment_model
from tract.active_learning.review import generate_review_json
from tract.active_learning.unmapped import load_unmapped_controls
from tract.calibration.conformal import build_prediction_sets, compute_conformal_coverage, compute_conformal_quantile
from tract.calibration.diagnostics import bootstrap_ece, expected_calibration_error, ks_test_similarity_distributions
from tract.calibration.ood import compute_ood_threshold, flag_ood_items, validate_ood_threshold
from tract.calibration.temperature import calibrate_similarities, find_global_threshold, fit_temperature
from tract.config import (
    PHASE1C_CONFORMAL_ALPHA,
    PHASE1C_CONFORMAL_COVERAGE_GATE,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1C_DEPLOYMENT_MODEL_DIR,
    PHASE1C_ECE_BOOTSTRAP_N,
    PHASE1C_ECE_N_BINS,
    PHASE1C_ECE_THRESHOLD,
    PHASE1C_OOD_SEPARATION_GATE,
    PHASE1C_RESULTS_DIR,
    PHASE1C_T_GAP_WARNING,
    PHASE1C_UNMAPPED_FRAMEWORKS,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from tract.crosswalk.store import insert_assignments
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.data_quality import QualityTier, TieredLink, load_and_filter_curated_links
from tract.training.firewall import build_all_hub_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _EvalItem:
    """Lightweight eval item for extract_similarity_matrix compatibility."""
    __slots__ = ("control_text", "framework_name", "valid_hub_ids")

    def __init__(self, control_text: str, framework_name: str, valid_hub_ids: frozenset[str]) -> None:
        self.control_text = control_text
        self.framework_name = framework_name
        self.valid_hub_ids = valid_hub_ids


def _dicts_to_eval_items(items: list[dict]) -> list[_EvalItem]:
    """Convert dicts to objects with attributes for extract_similarity_matrix."""
    return [
        _EvalItem(
            control_text=item.get("control_text", ""),
            framework_name=item.get("framework", ""),
            valid_hub_ids=frozenset(item.get("valid_hub_ids", [])),
        )
        for item in items
    ]


def _run_inference(
    model: object,
    eval_items: list[_EvalItem],
    hub_ids: list[str],
    hub_embs: np.ndarray,
    label: str,
) -> np.ndarray:
    """Run inference on a set of items, return similarity matrix."""
    from tract.training.evaluate import extract_similarity_matrix

    sim_data = extract_similarity_matrix(model, eval_items, hub_ids, hub_embs)
    logger.info("Inference %s: %d items × %d hubs", label, sim_data["sims"].shape[0], sim_data["sims"].shape[1])
    return sim_data["sims"]


def _load_holdout_links(holdout_dir: Path) -> tuple[list[TieredLink], list[TieredLink]]:
    """Load saved holdout links from JSON."""
    cal_data = load_json(holdout_dir / "calibration_links.json")
    canary_data = load_json(holdout_dir / "canary_links.json")
    cal_links = [TieredLink(link=d["link"], tier=QualityTier(d["tier"])) for d in cal_data]
    canary_links = [TieredLink(link=d["link"], tier=QualityTier(d["tier"])) for d in canary_data]
    return cal_links, canary_links


def main() -> None:
    logger.info("=== T2: Deployment Inference + Production Calibration ===")
    t_start = time.time()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())

    model = load_deployment_model(PHASE1C_DEPLOYMENT_MODEL_DIR / "model" / "model")

    hub_texts = build_all_hub_texts(hierarchy, excluded_framework=None)
    hub_texts_ordered = [hub_texts[hid] for hid in hub_ids]
    hub_embs = model.encode(
        hub_texts_ordered, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False, batch_size=128,
    )

    # === 1. Load all inference item sets ===
    holdout_dir = PHASE1C_RESULTS_DIR / "holdout"
    cal_links, canary_trad_links = _load_holdout_links(holdout_dir)
    cal_eval_dicts = [holdout_to_eval(l) for l in cal_links]
    cal_eval = _dicts_to_eval_items(cal_eval_dicts)

    unmapped_dicts = load_unmapped_controls(
        frameworks_dir=PROCESSED_DIR / "frameworks",
        framework_file_ids=list(PHASE1C_UNMAPPED_FRAMEWORKS.keys()),
        framework_display_names=PHASE1C_UNMAPPED_FRAMEWORKS,
    )
    unmapped_eval = _dicts_to_eval_items(unmapped_dicts)

    ood_data = load_json(PROJECT_ROOT / "tests" / "fixtures" / "ood_synthetic_texts.json")
    ood_eval = [_EvalItem(control_text=t, framework_name="synthetic_ood", valid_hub_ids=frozenset()) for t in ood_data]

    # === 2. Run inference ===
    sims_unmapped = _run_inference(model, unmapped_eval, hub_ids, hub_embs, "unmapped")
    sims_cal = _run_inference(model, cal_eval, hub_ids, hub_embs, "calibration_holdout")
    sims_ood = _run_inference(model, ood_eval, hub_ids, hub_embs, "ood_synthetic")

    sim_dir = PHASE1C_RESULTS_DIR / "similarities"
    sim_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sim_dir / "deployment_unmapped.npz", sims=sims_unmapped)
    np.savez_compressed(sim_dir / "deployment_calibration.npz", sims=sims_cal)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # === 3. Fit T_deploy on 420 holdout ===
    cal_dir = PHASE1C_RESULTS_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    hub_id_to_idx = {hid: i for i, hid in enumerate(hub_ids)}
    cal_valid_indices = []
    for item in cal_eval_dicts:
        indices = [hub_id_to_idx[hid] for hid in item["valid_hub_ids"] if hid in hub_id_to_idx]
        cal_valid_indices.append(indices)

    t_deploy_result = fit_temperature(sims_cal, cal_valid_indices)
    t_deploy = float(t_deploy_result["temperature"])
    atomic_write_json(t_deploy_result, cal_dir / "t_deploy_result.json")
    logger.info("T_deploy = %.4f", t_deploy)

    t_lofo_data = load_json(cal_dir / "t_lofo_result.json")
    t_gap = abs(t_deploy - t_lofo_data["temperature"])
    if t_gap > PHASE1C_T_GAP_WARNING:
        logger.warning("T_gap = %.4f > %.2f warning threshold", t_gap, PHASE1C_T_GAP_WARNING)

    # === 4. ECE gate ===
    probs_cal = calibrate_similarities(sims_cal, t_deploy)
    confidences = np.array([float(probs_cal[i].max()) for i in range(len(probs_cal))])
    accuracies = np.array([
        1.0 if int(probs_cal[i].argmax()) in valid else 0.0
        for i, valid in enumerate(cal_valid_indices)
    ])
    ece = float(expected_calibration_error(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS))
    ece_ci = bootstrap_ece(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS, n_bootstrap=PHASE1C_ECE_BOOTSTRAP_N)
    ece_passed = ece < PHASE1C_ECE_THRESHOLD
    ece_result = {"ece": ece, "ece_ci": ece_ci, "threshold": PHASE1C_ECE_THRESHOLD, "passed": ece_passed}
    atomic_write_json(ece_result, cal_dir / "ece_gate.json")
    logger.info("ECE gate: %.4f %s (threshold=%.2f)", ece, "PASS" if ece_passed else "FAIL", PHASE1C_ECE_THRESHOLD)

    # === 5. Conformal prediction (use calibration holdout as calibration set) ===
    conformal_quantile = float(compute_conformal_quantile(probs_cal, cal_valid_indices, alpha=PHASE1C_CONFORMAL_ALPHA))
    prediction_sets_cal = build_prediction_sets(probs_cal, hub_ids, conformal_quantile)
    valid_hub_sets_cal = [frozenset(item["valid_hub_ids"]) for item in cal_eval_dicts]
    coverage = float(compute_conformal_coverage(prediction_sets_cal, valid_hub_sets_cal))
    coverage_passed = coverage >= PHASE1C_CONFORMAL_COVERAGE_GATE

    conformal_result = {
        "quantile": conformal_quantile, "coverage": coverage,
        "coverage_gate": PHASE1C_CONFORMAL_COVERAGE_GATE, "passed": coverage_passed,
        "alpha": PHASE1C_CONFORMAL_ALPHA, "n_items": len(cal_eval),
        "mean_set_size": float(np.mean([len(s) for s in prediction_sets_cal])),
    }
    atomic_write_json(conformal_result, cal_dir / "conformal.json")
    logger.info("Conformal coverage: %.4f %s", coverage, "PASS" if coverage_passed else "FAIL")

    # === 6. OOD threshold ===
    max_sims_cal = np.array([float(sims_cal[i].max()) for i in range(len(sims_cal))])
    ood_threshold = float(compute_ood_threshold(max_sims_cal))
    max_sims_ood = np.array([float(sims_ood[i].max()) for i in range(len(sims_ood))])
    ood_result = validate_ood_threshold(max_sims_ood, ood_threshold)
    atomic_write_json(ood_result, cal_dir / "ood.json")
    logger.info("OOD gate: %s (separation=%.1f%%)", "PASS" if ood_result["gate_passed"] else "FAIL", ood_result["separation_rate"] * 100)

    # === 7. Global threshold (max-F1) ===
    threshold_result = find_global_threshold(sims_cal, cal_valid_indices, t_deploy)
    atomic_write_json(threshold_result, cal_dir / "global_threshold.json")
    logger.info("Global threshold: %.4f (F1=%.4f)", threshold_result["threshold"], threshold_result["f1"])

    # === 8. KS-test diagnostic ===
    max_sims_unmapped = np.array([float(sims_unmapped[i].max()) for i in range(len(sims_unmapped))])
    ks_result = ks_test_similarity_distributions(max_sims_cal, max_sims_unmapped)
    atomic_write_json(ks_result, cal_dir / "ks_test.json")
    logger.info("KS-test: statistic=%.4f, p=%.6f", ks_result["statistic"], ks_result["p_value"])

    # === 9. Generate review.json ===
    probs_unmapped = calibrate_similarities(sims_unmapped, t_deploy)
    conformal_sets_unmapped = build_prediction_sets(probs_unmapped, hub_ids, conformal_quantile)
    ood_flags_unmapped = flag_ood_items(max_sims_unmapped, ood_threshold)

    review_items = []
    for ctrl in unmapped_dicts:
        review_items.append({
            "control_id": ctrl["control_id"],
            "framework": ctrl["framework"],
            "control_text": ctrl["control_text"],
        })

    round_dir = PHASE1C_RESULTS_DIR / "round_1"
    round_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"

    generate_review_json(
        items=review_items,
        hub_ids=hub_ids,
        probs=probs_unmapped,
        conformal_sets=conformal_sets_unmapped,
        ood_flags=ood_flags_unmapped,
        threshold=threshold_result["threshold"],
        temperature=t_deploy,
        model_version=git_sha,
        round_number=1,
        output_path=round_dir / "review.json",
    )

    # === 10. Populate crosswalk.db with model prediction assignments ===
    db_path = PHASE1C_CROSSWALK_DB_PATH
    prediction_assignments = []
    for i, ctrl in enumerate(unmapped_dicts):
        top_idx = int(probs_unmapped[i].argmax())
        top_hub = hub_ids[top_idx]
        top_conf = float(probs_unmapped[i, top_idx])
        in_conformal = 1 if top_hub in conformal_sets_unmapped[i] else 0

        prediction_assignments.append({
            "control_id": ctrl["control_id"],
            "hub_id": top_hub,
            "confidence": round(top_conf, 4),
            "in_conformal_set": in_conformal,
            "is_ood": 1 if ood_flags_unmapped[i] else 0,
            "provenance": "active_learning_round_1",
            "source_link_id": None,
            "model_version": git_sha,
            "review_status": "pending",
        })

    insert_assignments(db_path, prediction_assignments)
    logger.info("Inserted %d prediction assignments into crosswalk.db", len(prediction_assignments))

    # === Summary ===
    logger.info("=== T2 QUALITY GATES ===")
    logger.info("ECE: %.4f %s", ece, "PASS" if ece_passed else "FAIL")
    logger.info("Conformal coverage: %.4f %s", coverage, "PASS" if coverage_passed else "FAIL")
    logger.info("OOD separation: %.1f%% %s", ood_result["separation_rate"] * 100, "PASS" if ood_result["gate_passed"] else "FAIL")
    logger.info("Global threshold: %.4f (F1=%.4f)", threshold_result["threshold"], threshold_result["f1"])
    logger.info("Review.json: %d items in %s", len(review_items), round_dir / "review.json")
    logger.info("T2 complete in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
