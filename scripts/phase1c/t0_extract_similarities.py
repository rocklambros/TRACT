"""T0: Extract similarity matrices from LOFO fold models + populate crosswalk.db.

Reads:
  - results/phase1b/phase1b_textaware/fold_*/model/model/ (5 fold models)
  - data/processed/cre_hierarchy.json (hub hierarchy)
  - data/processed/frameworks/*.json (all parsed frameworks)
  - data/training/hub_links_curated.jsonl (training links)

Writes:
  - results/phase1c/similarities/fold_{name}.npz (5 NPZ files)
  - results/phase1c/crosswalk.db (populated with hubs, frameworks, controls, training assignments)

Usage:
  python -m scripts.phase1c.t0_extract_similarities
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from scripts.phase0.common import AI_FRAMEWORK_NAMES, build_evaluation_corpus, load_curated_links
from tract.active_learning.model_io import load_fold_model
from tract.config import (
    PHASE1B_RESULTS_DIR,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1C_SIMILARITIES_DIR,
    PROCESSED_DIR,
)
from tract.crosswalk.populate import (
    build_control_records,
    build_framework_records,
    build_hub_records,
    build_training_assignments,
)
from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    count_frameworks,
    count_hubs,
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.evaluate import extract_similarity_matrix
from tract.training.firewall import build_all_hub_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FOLD_DIR = PHASE1B_RESULTS_DIR / "phase1b_textaware"


def _extract_fold_similarities(hierarchy: CREHierarchy) -> None:
    """Load each fold model, extract similarity matrices, save as NPZ."""
    PHASE1C_SIMILARITIES_DIR.mkdir(parents=True, exist_ok=True)

    hub_ids = sorted(hierarchy.hubs.keys())
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    eval_by_fw: dict[str, list] = {}
    for item in corpus:
        eval_by_fw.setdefault(item.framework_name, []).append(item)

    fold_dirs = sorted(d for d in FOLD_DIR.iterdir() if d.is_dir() and d.name.startswith("fold_"))

    for fold_dir in fold_dirs:
        fw_name = fold_dir.name.replace("fold_", "").replace("_", " ")
        eval_items = eval_by_fw.get(fw_name, [])
        if not eval_items:
            logger.warning("No eval items for fold %s, skipping", fw_name)
            continue

        logger.info("Extracting similarities for fold %s (%d items)", fw_name, len(eval_items))
        t0 = time.time()

        model = load_fold_model(fold_dir)

        hub_texts = build_all_hub_texts(hierarchy, excluded_framework=fw_name)
        hub_texts_ordered = [hub_texts[hid] for hid in hub_ids]
        hub_embs = model.encode(
            hub_texts_ordered,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=128,
        )

        sim_data = extract_similarity_matrix(model, eval_items, hub_ids, hub_embs)

        npz_path = PHASE1C_SIMILARITIES_DIR / f"fold_{fold_dir.name.replace('fold_', '')}.npz"
        np.savez_compressed(
            npz_path,
            sims=sim_data["sims"],
            hub_ids=np.array(sim_data["hub_ids"]),
            gt_json=np.array(sim_data["gt_json"]),
            frameworks=np.array(sim_data["frameworks"]),
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Fold %s: saved %s (%.1fs)", fw_name, npz_path.name, time.time() - t0)


def _populate_crosswalk_db(hierarchy: CREHierarchy) -> None:
    """Create and populate crosswalk.db with hubs, frameworks, controls, training assignments."""
    db_path = PHASE1C_CROSSWALK_DB_PATH
    create_database(db_path)

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

    tiered_links, _ = load_and_filter_curated_links()
    control_id_map: dict[tuple[str, str], str] = {}
    for rec in ctrl_records:
        fw_id = rec["framework_id"]
        fw_name = ""
        for fw in all_fw_data:
            if fw["framework_id"] == fw_id:
                fw_name = fw.get("framework_name", fw_id)
                break
        control_id_map[(fw_name, rec["section_id"])] = rec["id"]

    assignment_records = build_training_assignments(tiered_links, control_id_map)
    if assignment_records:
        insert_assignments(db_path, assignment_records)
    logger.info("Inserted %d training assignments", len(assignment_records))


def main() -> None:
    logger.info("=== T0: Extract Similarities + Populate Crosswalk DB ===")
    t0 = time.time()

    hierarchy_data = load_json(PROCESSED_DIR / "cre_hierarchy.json")
    hierarchy = CREHierarchy.model_validate(hierarchy_data)
    logger.info("Loaded hierarchy: %d hubs", len(hierarchy.hubs))

    _extract_fold_similarities(hierarchy)
    _populate_crosswalk_db(hierarchy)

    logger.info("T0 complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
