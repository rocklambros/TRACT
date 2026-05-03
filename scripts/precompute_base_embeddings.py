"""Pre-compute base BGE-large-v1.5 embeddings (before fine-tuning) for Figure 5.2.

Loads the same texts used in deployment_artifacts.npz, encodes them with the
un-fine-tuned base model, and saves the results. This is a one-time operation
(~2 min on Tegra ARM64, requires ~1.3 GB model download on first run).

Usage:
    python scripts/precompute_base_embeddings.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tract.config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "results" / "phase1b" / "base_bge_embeddings.npz"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64


def load_texts() -> tuple[list[str], list[str], list[str], list[str]]:
    """Load hub and control texts matching deployment_artifacts.npz order.

    Returns:
        (hub_texts, control_texts, hub_ids, control_ids)
    """
    artifacts = np.load(
        str(PROJECT_ROOT / "results" / "phase1c" / "deployment_model" / "deployment_artifacts.npz"),
        allow_pickle=False,
    )
    hub_ids = list(artifacts["hub_ids"])
    control_ids = list(artifacts["control_ids"])

    hierarchy = json.loads(
        (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
    )
    hubs = hierarchy["hubs"]

    hub_texts = []
    for hid in hub_ids:
        hub = hubs.get(str(hid), hubs.get(hid, {}))
        path = hub.get("hierarchy_path", "")
        name = hub.get("name", "")
        hub_texts.append(f"{path} {name}".strip())

    crosswalk_path = PROJECT_ROOT / "build" / "dataset" / "crosswalk_v1.0.jsonl"
    control_map: dict[str, str] = {}
    if crosswalk_path.exists():
        with crosswalk_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    cid = row.get("control_id", "")
                    title = row.get("control_title", "")
                    if cid and cid not in control_map:
                        control_map[cid] = title

    control_texts = [control_map.get(cid, str(cid)) for cid in control_ids]

    return hub_texts, control_texts, hub_ids, control_ids


def main() -> None:
    if OUTPUT_PATH.exists():
        logger.info("Output already exists: %s — skipping", OUTPUT_PATH)
        return

    logger.info("Loading texts from deployment artifacts...")
    hub_texts, control_texts, hub_ids, control_ids = load_texts()
    logger.info("Hub texts: %d, Control texts: %d", len(hub_texts), len(control_texts))

    logger.info("Loading base model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    logger.info("Encoding hub texts...")
    hub_embeddings = model.encode(
        hub_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    logger.info("Encoding control texts...")
    control_embeddings = model.encode(
        control_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    logger.info("Saving to %s", OUTPUT_PATH)
    np.savez(
        str(OUTPUT_PATH),
        hub_embeddings=hub_embeddings,
        control_embeddings=control_embeddings,
        hub_ids=np.array(hub_ids),
        control_ids=np.array(control_ids),
    )
    logger.info(
        "Done. hub_embeddings=%s, control_embeddings=%s",
        hub_embeddings.shape,
        control_embeddings.shape,
    )


if __name__ == "__main__":
    main()
