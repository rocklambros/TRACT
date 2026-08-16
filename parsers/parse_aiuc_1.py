"""Parser for AIUC-1 Standard — Tier 1 structured JSON.

Mapping unit is activity (leaf nodes under controls).
"""
from __future__ import annotations

import json
import logging

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Aiuc1Parser(BaseParser):
    framework_id = "aiuc_1"
    framework_name = "AIUC-1 Standard"
    version = "1.0"
    source_url = "https://www.aiuc-1.com"
    mapping_unit_level = "activity"
    expected_count = 132
    fetched_date = "2026-04-28"

    def parse(self) -> list[Control]:
        data = json.loads(self.read_source("aiuc-1-standard.json"))
        controls: list[Control] = []

        for domain in data["domains"]:
            domain_name = domain["name"]
            for ctrl in domain["controls"]:
                ctrl_id = ctrl["id"]
                ctrl_title = ctrl["title"]
                for activity in ctrl.get("activities", []):
                    controls.append(Control(
                        control_id=activity["id"],
                        title=ctrl_title,
                        description=activity["description"],
                        hierarchy_level="activity",
                        parent_id=ctrl_id,
                        parent_name=ctrl_title,
                        metadata={
                            "category": activity.get("category", ""),
                            "domain": domain_name,
                            "evidence_types": activity.get("evidence_types", []),
                        },
                    ))

        return controls


if __name__ == "__main__":
    parser = Aiuc1Parser()
    parser.run()
