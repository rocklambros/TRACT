"""Parser for MITRE ATLAS — Tier 1 structured JSON.

Extracts techniques (with sub-techniques) and mitigations from matrices[0].
"""
from __future__ import annotations

import json
import logging

from typing import ClassVar

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
    fetched_date: ClassVar[str] = "2026-04-28"
    # All 202 units carry a statement and none equals its title, so the
    # attainable value is exactly 1.0 and the floor fires at 201/202 (0.9950).
    # [measured 2026-08-19]
    #
    # The margin is thin on one entry. AML.M0004's description is 62 characters
    # against a 60-character threshold, so an upstream copy edit that shortens
    # it trips this gate. That is the intended behaviour: read the failure,
    # confirm the shorter text is the real source, then lower the floor in the
    # same commit that moves version and fetched_date.
    min_prose_fraction: ClassVar[float] = 1.0

    def parse(self) -> list[Control]:
        data = json.loads(self.read_source("ATLAS_compiled.json"))
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
