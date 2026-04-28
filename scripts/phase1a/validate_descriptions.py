"""Validate hub description review status.

Reports counts of accepted, edited, rejected, and pending descriptions.
Validates that all reviewed_description values pass sanitization.

Usage: python -m scripts.phase1a.validate_descriptions
"""
from __future__ import annotations

import logging
import sys

from tract.config import PROCESSED_DIR
from tract.descriptions import HubDescriptionSet
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.sanitize import sanitize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
    descriptions_path = PROCESSED_DIR / "hub_descriptions.json"

    if not hierarchy_path.exists():
        logger.error("Hierarchy not found at %s", hierarchy_path)
        return 1
    if not descriptions_path.exists():
        logger.error("Descriptions not found at %s", descriptions_path)
        return 1

    hierarchy = CREHierarchy.load(hierarchy_path)
    data = load_json(descriptions_path)
    desc_set = HubDescriptionSet.model_validate(data)

    errors: list[str] = []

    # Check all leaf hubs present
    leaf_ids = set(hierarchy.label_space)
    desc_ids = set(desc_set.descriptions.keys())
    missing = leaf_ids - desc_ids
    extra = desc_ids - leaf_ids

    if missing:
        errors.append(f"Missing descriptions for {len(missing)} leaf hubs: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"Extra descriptions for {len(extra)} non-leaf hubs: {sorted(extra)[:5]}")

    # Check no empty descriptions
    empty = [hid for hid, d in desc_set.descriptions.items() if not d.description]
    if empty:
        errors.append(f"{len(empty)} hubs have empty descriptions: {empty[:5]}")

    # Validate reviewed_description sanitization
    sanitize_errors: list[str] = []
    for hub_id, desc in desc_set.descriptions.items():
        if desc.reviewed_description is not None:
            try:
                sanitize_text(desc.reviewed_description)
            except ValueError as e:
                sanitize_errors.append(f"{hub_id}: {e}")

    if sanitize_errors:
        errors.append(f"{len(sanitize_errors)} reviewed descriptions fail sanitization: {sanitize_errors[:3]}")

    # Count by status
    from collections import Counter
    status_counts = Counter(d.review_status for d in desc_set.descriptions.values())

    logger.info("=== Description Review Status ===")
    logger.info("  Total:    %d", len(desc_set.descriptions))
    logger.info("  Pending:  %d", status_counts.get("pending", 0))
    logger.info("  Accepted: %d", status_counts.get("accepted", 0))
    logger.info("  Edited:   %d", status_counts.get("edited", 0))
    logger.info("  Rejected: %d", status_counts.get("rejected", 0))

    if errors:
        for error in errors:
            logger.error("VALIDATION ERROR: %s", error)
        return 1

    logger.info("All validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
