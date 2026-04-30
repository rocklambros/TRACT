"""T5: Finalize crosswalk.db and export crosswalk artifacts.

Rebuilds crosswalk.db from scratch with:
  (a) Ground truth: all curated CRE links with proper name normalization
  (b) Active learning: reviewed predictions from final AL round
Exports JSON, CSV, per-framework, and cross-framework equivalence matrix.

Reads:
  - data/processed/cre_hierarchy.json
  - data/processed/frameworks/*.json
  - data/training/hub_links_curated.jsonl
  - results/phase1c/round_2/review.json (final reviewed predictions)

Writes:
  - results/phase1c/crosswalk.db (rebuilt)
  - results/phase1c/exports/crosswalk_full.json
  - results/phase1c/exports/crosswalk_full.csv
  - results/phase1c/exports/cross_framework_matrix.json
  - results/phase1c/exports/per_framework/<framework_id>.json

Usage:
  python -m scripts.phase1c.t5_finalize_crosswalk
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tract.config import (
    OPENCRE_FRAMEWORK_ID_MAP,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1C_RESULTS_DIR,
    PROCESSED_DIR,
)
from tract.crosswalk.export import export_crosswalk
from tract.crosswalk.populate import build_control_records, build_framework_records, build_hub_records
from tract.crosswalk.schema import create_database, get_connection
from tract.crosswalk.snapshot import compute_db_hash
from tract.crosswalk.store import count_frameworks, count_hubs, insert_assignments, insert_controls, insert_frameworks, insert_hubs
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.data_quality import load_and_filter_curated_links

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_ground_truth_assignments(
    all_fw_data: list[dict],
    ctrl_records: list[dict],
) -> list[dict]:
    """Build ground truth assignments using OPENCRE_FRAMEWORK_ID_MAP for name normalization.

    Handles section_id format mismatches between OpenCRE training links and
    parsed framework controls via multi-key lookup.
    """
    tiered_links, _ = load_and_filter_curated_links()

    std_name_to_fw_id: dict[str, str] = {}
    for std_name, fw_id in OPENCRE_FRAMEWORK_ID_MAP.items():
        std_name_to_fw_id[std_name] = fw_id
    for fw in all_fw_data:
        std_name_to_fw_id[fw.get("framework_name", "")] = fw["framework_id"]
        std_name_to_fw_id[fw["framework_id"]] = fw["framework_id"]

    ctrl_lookup: dict[tuple[str, str], str] = {}
    for rec in ctrl_records:
        fw_id = rec["framework_id"]
        sid = rec["section_id"]
        ctrl_lookup[(fw_id, sid)] = rec["id"]

    records = []
    no_fw = 0
    no_ctrl = 0

    for link in tiered_links:
        std_name = link.link.get("standard_name", "")
        section_id = link.link.get("section_id", "")
        hub_id = link.link.get("cre_id", "")

        fw_id = std_name_to_fw_id.get(std_name)
        if fw_id is None:
            no_fw += 1
            continue

        composite = ctrl_lookup.get((fw_id, section_id))
        if composite is None:
            no_ctrl += 1
            continue

        records.append({
            "control_id": composite,
            "hub_id": hub_id,
            "confidence": None,
            "in_conformal_set": None,
            "is_ood": 0,
            "provenance": f"ground_truth_{link.tier.value}",
            "source_link_id": link.link.get("link_id"),
            "model_version": None,
            "review_status": "ground_truth",
        })

    if no_fw:
        logger.warning("Skipped %d links: standard_name not in framework map", no_fw)
    if no_ctrl:
        logger.info("Skipped %d links: section_id not in parsed controls (expected for traditional frameworks)", no_ctrl)
    logger.info("Built %d ground truth assignments from %d curated links", len(records), len(tiered_links))
    return records


def _build_al_assignments(review_path: Path, round_num: int) -> list[dict]:
    """Build assignments from reviewed AL predictions."""
    review_data = load_json(review_path)
    items = review_data["items"]

    records = []
    for item in items:
        review = item.get("review")
        if review is None:
            continue

        status = review["status"]
        if status == "rejected":
            continue

        if status == "corrected":
            hub_id = review["corrected_hub_id"]
        elif status == "accepted":
            preds = item.get("predictions", [])
            if not preds:
                continue
            hub_id = preds[0]["hub_id"]
        else:
            continue

        top_conf = None
        if item.get("predictions"):
            if status == "accepted":
                top_conf = item["predictions"][0]["confidence"]
            else:
                for p in item["predictions"]:
                    if p["hub_id"] == hub_id:
                        top_conf = p["confidence"]
                        break

        records.append({
            "control_id": item["control_id"],
            "hub_id": hub_id,
            "confidence": top_conf,
            "in_conformal_set": None,
            "is_ood": 1 if item.get("is_ood") else 0,
            "provenance": f"active_learning_round_{round_num}",
            "source_link_id": None,
            "model_version": review_data.get("model_version"),
            "review_status": "accepted",
        })

    logger.info("Built %d AL assignments from round %d (%d reviewed)", len(records), round_num, len(items))
    return records


def _export_cross_framework_matrix(db_path: Path, output_path: Path) -> Path:
    """Export cross-framework equivalence matrix via shared CRE hubs.

    Two controls are equivalent if they map to the same hub.
    Two controls are related if their hubs share a parent.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, a.hub_id, c.framework_id, c.title, "
            "h.name AS hub_name, h.parent_id "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN hubs h ON a.hub_id = h.id "
            "WHERE a.review_status IN ('accepted', 'ground_truth') "
            "ORDER BY a.hub_id"
        ).fetchall()
    finally:
        conn.close()

    hub_to_controls: dict[str, list[dict]] = defaultdict(list)
    parent_to_hubs: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        hub_to_controls[row["hub_id"]].append({
            "control_id": row["control_id"],
            "framework_id": row["framework_id"],
            "title": row["title"],
        })
        if row["parent_id"]:
            parent_to_hubs[row["parent_id"]].add(row["hub_id"])

    equivalences = []
    for hub_id, controls in hub_to_controls.items():
        frameworks = {c["framework_id"] for c in controls}
        if len(frameworks) >= 2:
            equivalences.append({
                "hub_id": hub_id,
                "hub_name": next(
                    (r["hub_name"] for r in rows if r["hub_id"] == hub_id), ""
                ),
                "controls": controls,
                "frameworks": sorted(frameworks),
            })

    equivalences.sort(key=lambda e: (-len(e["frameworks"]), e["hub_id"]))

    fw_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for eq in equivalences:
        fws = eq["frameworks"]
        for i, fw_a in enumerate(fws):
            for fw_b in fws[i + 1:]:
                pair = tuple(sorted([fw_a, fw_b]))
                fw_pair_counts[pair] += 1

    matrix = {
        "equivalences": equivalences,
        "framework_pair_overlap": [
            {"framework_a": a, "framework_b": b, "shared_hubs": n}
            for (a, b), n in sorted(fw_pair_counts.items(), key=lambda x: -x[1])
        ],
        "total_shared_hubs": len(equivalences),
        "total_cross_framework_controls": sum(len(e["controls"]) for e in equivalences),
    }

    atomic_write_json(matrix, output_path)
    logger.info(
        "Cross-framework matrix: %d shared hubs, %d framework pairs",
        len(equivalences), len(fw_pair_counts),
    )
    return output_path


def _export_per_framework(db_path: Path, output_dir: Path) -> None:
    """Export one JSON file per framework with its hub assignments."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, c.section_id, c.title, c.description, "
            "a.hub_id, h.name AS hub_name, h.path AS hub_path, "
            "a.confidence, a.provenance, a.review_status, "
            "f.id AS framework_id, f.name AS framework_name "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN hubs h ON a.hub_id = h.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            "WHERE a.review_status IN ('accepted', 'ground_truth') "
            "ORDER BY f.id, c.section_id"
        ).fetchall()
    finally:
        conn.close()

    by_framework: dict[str, dict] = defaultdict(lambda: {"framework_name": "", "mappings": []})
    for row in rows:
        fw_id = row["framework_id"]
        by_framework[fw_id]["framework_name"] = row["framework_name"]
        by_framework[fw_id]["mappings"].append({
            "control_id": row["section_id"],
            "title": row["title"],
            "hub_id": row["hub_id"],
            "hub_name": row["hub_name"],
            "hub_path": row["hub_path"],
            "confidence": row["confidence"],
            "provenance": row["provenance"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    for fw_id, data in sorted(by_framework.items()):
        atomic_write_json(data, output_dir / f"{fw_id}.json")

    logger.info("Exported %d per-framework files to %s", len(by_framework), output_dir)


def _generate_deployment_artifacts(
    model: Any,
    hub_ids: list[str],
    hub_texts: dict[str, str],
    control_ids: list[str],
    control_texts: dict[str, str],
    adapter_path: Path,
    output_path: Path,
) -> None:
    """Generate consolidated deployment NPZ with all cached embeddings.

    Hub IDs are stored in canonical sorted order. Embedding rows match
    the hub_ids/control_ids arrays for index consistency.
    """
    sorted_hub_ids = sorted(hub_ids)
    sorted_hub_texts = [hub_texts[hid] for hid in sorted_hub_ids]

    hub_embs = model.encode(
        sorted_hub_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128,
    )

    sorted_control_ids = sorted(control_ids)
    sorted_control_texts = [control_texts[cid] for cid in sorted_control_ids]

    ctrl_embs = model.encode(
        sorted_control_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128,
    )

    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        hub_embeddings=hub_embs.astype(np.float32),
        control_embeddings=ctrl_embs.astype(np.float32),
        hub_ids=np.array(sorted_hub_ids),
        control_ids=np.array(sorted_control_ids),
        model_adapter_hash=np.array(adapter_hash),
        generation_timestamp=np.array(timestamp),
    )
    logger.info(
        "Saved deployment artifacts: %d hubs, %d controls, adapter_hash=%s…",
        len(sorted_hub_ids), len(sorted_control_ids), adapter_hash[:12],
    )


def _generate_calibration_bundle(
    t_deploy: float,
    ood_threshold: float,
    conformal_quantile: float,
    global_threshold: float,
    hierarchy_path: Path,
    output_path: Path,
) -> None:
    """Bundle all calibration parameters into a single JSON file."""
    hierarchy_hash = hashlib.sha256(hierarchy_path.read_bytes()).hexdigest()

    bundle = {
        "t_deploy": t_deploy,
        "ood_threshold": ood_threshold,
        "conformal_quantile": conformal_quantile,
        "global_threshold": global_threshold,
        "hierarchy_hash": hierarchy_hash,
        "calibration_note": (
            "Calibrated on 420 traditional framework holdout items. "
            "Accuracy on AI framework text may differ."
        ),
    }

    atomic_write_json(bundle, output_path)
    logger.info("Saved calibration bundle to %s", output_path)


def main() -> None:
    logger.info("=== T5: Finalize Crosswalk Database ===")
    t_start = time.time()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))

    # === 1. Rebuild DB from scratch ===
    db_path = PHASE1C_CROSSWALK_DB_PATH
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing crosswalk.db")
    create_database(db_path)

    # === 2. Insert structural data ===
    hub_records = build_hub_records(hierarchy)
    insert_hubs(db_path, hub_records)
    logger.info("Inserted %d hubs", count_hubs(db_path))

    fw_dir = PROCESSED_DIR / "frameworks"
    all_fw_data = []
    for fw_file in sorted(fw_dir.glob("*.json")):
        with open(fw_file, encoding="utf-8") as f:
            all_fw_data.append(json.load(f))

    fw_records = build_framework_records(all_fw_data)
    insert_frameworks(db_path, fw_records)
    logger.info("Inserted %d frameworks", count_frameworks(db_path))

    ctrl_records = build_control_records(all_fw_data)
    insert_controls(db_path, ctrl_records)
    logger.info("Inserted %d controls", len(ctrl_records))

    # === 3. Ground truth assignments (with proper name mapping) ===
    gt_records = _build_ground_truth_assignments(all_fw_data, ctrl_records)
    if gt_records:
        insert_assignments(db_path, gt_records)

    # === 4. Active learning assignments (final round) ===
    review_path = PHASE1C_RESULTS_DIR / "round_2" / "review.json"
    if not review_path.exists():
        review_path = PHASE1C_RESULTS_DIR / "round_1" / "review.json"
    al_records = _build_al_assignments(review_path, round_num=2)
    if al_records:
        insert_assignments(db_path, al_records)

    # === 5. Summary ===
    conn = get_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        by_prov = conn.execute(
            "SELECT provenance, COUNT(*) as n FROM assignments GROUP BY provenance ORDER BY n DESC"
        ).fetchall()
        by_status = conn.execute(
            "SELECT review_status, COUNT(*) as n FROM assignments GROUP BY review_status ORDER BY n DESC"
        ).fetchall()
        n_frameworks_with_assignments = conn.execute(
            "SELECT COUNT(DISTINCT c.framework_id) FROM assignments a "
            "JOIN controls c ON a.control_id = c.id"
        ).fetchone()[0]
        n_unique_hubs = conn.execute(
            "SELECT COUNT(DISTINCT hub_id) FROM assignments"
        ).fetchone()[0]
    finally:
        conn.close()

    logger.info("=== CROSSWALK DB SUMMARY ===")
    logger.info("Total assignments: %d", total)
    logger.info("By provenance:")
    for row in by_prov:
        logger.info("  %s: %d", row["provenance"], row["n"])
    logger.info("By status:")
    for row in by_status:
        logger.info("  %s: %d", row["review_status"], row["n"])
    logger.info("Frameworks with assignments: %d", n_frameworks_with_assignments)
    logger.info("Unique hubs assigned: %d", n_unique_hubs)

    # === 6. Export ===
    export_dir = PHASE1C_RESULTS_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_crosswalk(db_path, export_dir / "crosswalk_full.json", fmt="json")
    export_crosswalk(db_path, export_dir / "crosswalk_full.csv", fmt="csv")
    _export_cross_framework_matrix(db_path, export_dir / "cross_framework_matrix.json")
    _export_per_framework(db_path, export_dir / "per_framework")

    # === 7. Generate deployment artifacts ===
    from tract.active_learning.model_io import load_deployment_model
    from tract.config import PHASE1D_ARTIFACTS_PATH, PHASE1D_CALIBRATION_PATH

    deploy_model_dir = PHASE1C_RESULTS_DIR / "deployment_model" / "model"
    if (deploy_model_dir / "model").exists():
        deploy_model_dir = deploy_model_dir / "model"
    if deploy_model_dir.exists():
        deploy_model = load_deployment_model(deploy_model_dir)

        hub_texts: dict[str, str] = {
            node.hub_id: f"{node.hierarchy_path}\n{node.name}"
            for node in hierarchy.hubs.values()
        }
        all_hub_ids = sorted(hierarchy.hubs.keys())

        ctrl_texts: dict[str, str] = {}
        ctrl_ids: list[str] = []
        for fw_data in all_fw_data:
            for ctrl in fw_data.get("controls", []):
                cid = f"{fw_data['framework_id']}::{ctrl['control_id']}"
                text_parts = [ctrl.get("title", ""), ctrl.get("description", "")]
                if ctrl.get("full_text"):
                    text_parts.append(ctrl["full_text"])
                ctrl_texts[cid] = " ".join(p for p in text_parts if p)
                ctrl_ids.append(cid)

        adapter_path = deploy_model_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            for p in deploy_model_dir.rglob("adapter_model.safetensors"):
                adapter_path = p
                break

        _generate_deployment_artifacts(
            model=deploy_model,
            hub_ids=all_hub_ids,
            hub_texts=hub_texts,
            control_ids=ctrl_ids,
            control_texts=ctrl_texts,
            adapter_path=adapter_path,
            output_path=PHASE1D_ARTIFACTS_PATH,
        )

        cal_results_dir = PHASE1C_RESULTS_DIR / "calibration"
        t_deploy = load_json(cal_results_dir / "t_deploy_result.json")["temperature"]
        ood_data = load_json(cal_results_dir / "ood.json")["threshold"]
        conformal_data = load_json(cal_results_dir / "conformal.json")["quantile"]
        global_data = load_json(cal_results_dir / "global_threshold.json")["threshold"]

        _generate_calibration_bundle(
            t_deploy=t_deploy,
            ood_threshold=ood_data,
            conformal_quantile=conformal_data,
            global_threshold=global_data,
            hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
            output_path=PHASE1D_CALIBRATION_PATH,
        )
    else:
        logger.warning("Deployment model not found at %s — skipping artifact generation", deploy_model_dir)

    db_hash = compute_db_hash(db_path)
    logger.info("DB hash: %s", db_hash)
    logger.info("T5 complete in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
