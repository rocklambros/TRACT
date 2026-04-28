"""Parser for MITRE ATLAS — Tier 1 structured JSON.

Extracts techniques (with sub-techniques) and mitigations from matrices[0].
"""
from __future__ import annotations

import logging

from tract.io import load_json
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MitreAtlasParser(BaseParser):
    framework_id = "mitre_atlas"
    framework_name = "MITRE ATLAS"
    version = "4.6.1"
    source_url = "https://atlas.mitre.org"
    mapping_unit_level = "technique"
    expected_count = 202

    def parse(self) -> list[Control]:
        data = load_json(self.raw_dir / "ATLAS_compiled.json")
        matrix = data["matrices"][0]
        controls: list[Control] = []

        for tech in matrix["techniques"]:
            tactic_names = [
                t.get("name", t) if isinstance(t, dict) else t
                for t in tech.get("tactics", [])
            ]
            controls.append(Control(
                control_id=tech["id"],
                title=tech["name"],
                description=tech["description"],
                hierarchy_level="technique",
                metadata={"tactics": tactic_names} if tactic_names else None,
            ))
            for sub in tech.get("subtechniques", []):
                controls.append(Control(
                    control_id=sub["id"],
                    title=sub["name"],
                    description=sub["description"],
                    hierarchy_level="sub-technique",
                    parent_id=tech["id"],
                    parent_name=tech["name"],
                ))

        for mit in matrix["mitigations"]:
            controls.append(Control(
                control_id=mit["id"],
                title=mit["name"],
                description=mit["description"],
                hierarchy_level="mitigation",
            ))

        return controls


if __name__ == "__main__":
    parser = MitreAtlasParser()
    parser.run()
