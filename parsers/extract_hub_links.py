"""Extract standard-to-hub links from OpenCRE data for LOFO training splits."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from tract.config import (
    OPENCRE_FRAMEWORK_ID_MAP,
    RAW_OPENCRE_DIR,
    TRAINING_DIR,
)
from tract.io import atomic_write_json, load_json
from tract.schema import HubLink

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAINING_LINK_TYPES = {"LinkedTo", "Linked To", "AutomaticallyLinkedTo", "Automatically Linked To"}


def normalize_framework_id(standard_name: str) -> str:
    """Map OpenCRE standard name to canonical framework ID."""
    if standard_name in OPENCRE_FRAMEWORK_ID_MAP:
        return OPENCRE_FRAMEWORK_ID_MAP[standard_name]
    slug = standard_name.lower().replace(" ", "_").replace("-", "_")
    logger.debug("Unknown standard '%s' -> '%s'", standard_name, slug)
    return slug


def extract_links(opencre_path: Path) -> list[HubLink]:
    """Extract all standard-to-hub links from the OpenCRE dump."""
    data = load_json(opencre_path)
    cres = data.get("cres", data) if isinstance(data, dict) else data
    links: list[HubLink] = []

    for cre in cres:
        cre_id = cre.get("id", "")
        cre_name = cre.get("name", "")

        for link in cre.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") != "Standard":
                continue

            link_type = link.get("ltype", link.get("type", ""))
            if not link_type:
                continue
            normalized_ltype = link_type.replace(" ", "").lower()
            if normalized_ltype == "linkedto":
                normalized_ltype = "LinkedTo"
            elif normalized_ltype == "automaticallylinkedto":
                normalized_ltype = "AutomaticallyLinkedTo"
            else:
                continue

            standard_name = doc.get("name", "")
            if not standard_name:
                continue

            section_id = doc.get("sectionID") or doc.get("section", "")
            section_name = doc.get("section") or doc.get("sectionID", "")

            links.append(HubLink(
                cre_id=cre_id,
                cre_name=cre_name,
                standard_name=standard_name,
                section_id=str(section_id) if section_id else "",
                section_name=str(section_name) if section_name else "",
                link_type=normalized_ltype,
                framework_id=normalize_framework_id(standard_name),
            ))

    return links


def main() -> None:
    opencre_path = RAW_OPENCRE_DIR / "opencre_all_cres.json"
    if not opencre_path.exists():
        raise FileNotFoundError(f"OpenCRE data not found at {opencre_path}. Run fetch_opencre.py first.")

    links = extract_links(opencre_path)
    logger.info("Extracted %d standard-to-hub links", len(links))

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = TRAINING_DIR / "hub_links.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for link in links:
            f.write(json.dumps(link.model_dump(), sort_keys=True, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d links)", jsonl_path, len(links))

    by_framework: dict[str, list[dict[str, str]]] = defaultdict(list)
    for link in links:
        by_framework[link.framework_id].append(link.model_dump())

    grouped_path = TRAINING_DIR / "hub_links_by_framework.json"
    atomic_write_json(dict(by_framework), grouped_path)
    logger.info("Wrote %s (%d frameworks)", grouped_path, len(by_framework))

    for fw_id, fw_links in sorted(by_framework.items(), key=lambda x: -len(x[1])):
        logger.info("  %s: %d links", fw_id, len(fw_links))


if __name__ == "__main__":
    main()
