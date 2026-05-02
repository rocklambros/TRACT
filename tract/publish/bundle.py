"""Bundle inference data alongside the merged model for HuggingFace."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def bundle_inference_data(
    staging_dir: Path,
    *,
    hub_descriptions: Path,
    hierarchy: Path,
    calibration: Path,
    artifacts: Path,
    bridge_report: Path,
) -> None:
    """Copy and validate inference data files into the staging directory.

    Args:
        staging_dir: Target directory (must exist).
        hub_descriptions: Path to hub_descriptions_reviewed.json.
        hierarchy: Path to cre_hierarchy.json (post-bridge).
        calibration: Path to calibration.json.
        artifacts: Path to deployment_artifacts.npz (hub_ids + hub_embeddings extracted).
        bridge_report: Path to bridge_report.json.

    Raises:
        FileNotFoundError: If any source file is missing.
    """
    copies: list[tuple[Path, str]] = [
        (hub_descriptions, "hub_descriptions.json"),
        (hierarchy, "cre_hierarchy.json"),
        (calibration, "calibration.json"),
        (bridge_report, "bridge_report.json"),
    ]

    for src, dest_name in copies:
        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {src}")
        shutil.copy2(src, staging_dir / dest_name)
        logger.info("Copied %s -> %s", src.name, dest_name)

    if not artifacts.exists():
        raise FileNotFoundError(f"Artifacts not found: {artifacts}")

    data = np.load(str(artifacts), allow_pickle=False)
    hub_ids = list(data["hub_ids"])
    hub_embeddings = data["hub_embeddings"]

    hub_ids_path = staging_dir / "hub_ids.json"
    with open(hub_ids_path, "w", encoding="utf-8") as f:
        json.dump(hub_ids, f, indent=2)
        f.write("\n")
    logger.info("Extracted %d hub IDs to hub_ids.json", len(hub_ids))

    emb_path = staging_dir / "hub_embeddings.npy"
    np.save(str(emb_path), hub_embeddings)
    logger.info(
        "Saved hub embeddings (%s) to hub_embeddings.npy",
        hub_embeddings.shape,
    )
