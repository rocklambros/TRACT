"""Build and validate the production CRE hierarchy.

Reads OpenCRE data, constructs the CREHierarchy, validates integrity,
and writes to data/processed/cre_hierarchy.json.

Usage: python -m scripts.phase1a.build_hierarchy
"""
from __future__ import annotations

import hashlib
import logging

from tract.config import PROCESSED_DIR, RAW_OPENCRE_DIR
from tract.hierarchy import CREHierarchy
from tract.io import load_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    opencre_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    if not opencre_path.exists():
        raise FileNotFoundError(
            f"OpenCRE data not found at {opencre_path}. "
            "Run scripts/fetch_opencre.py first."
        )

    logger.info("Loading OpenCRE data from %s", opencre_path)
    data = load_json(opencre_path)
    cres = data["cres"]
    fetch_timestamp = data.get("fetch_timestamp", "unknown")

    raw_bytes = opencre_path.read_bytes()
    data_hash = hashlib.sha256(raw_bytes).hexdigest()
    logger.info("Data hash: %s", data_hash)

    hierarchy = CREHierarchy.from_opencre(cres, fetch_timestamp, data_hash)

    output_path = PROCESSED_DIR / "cre_hierarchy.json"
    hierarchy.save(output_path)

    logger.info(
        "Hierarchy saved: %d hubs, %d roots, %d leaves",
        len(hierarchy.hubs),
        len(hierarchy.roots),
        len(hierarchy.label_space),
    )

    # Print branch summary
    for root_id in hierarchy.roots:
        branch = hierarchy.get_branch_hub_ids(root_id)
        root_name = hierarchy.hubs[root_id].name
        logger.info("  Branch '%s': %d hubs", root_name, len(branch))


if __name__ == "__main__":
    main()
