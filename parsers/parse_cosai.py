"""Parser for CoSAI Risk Map — Tier 2 YAML."""
from __future__ import annotations

import logging

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _flatten_yaml_text(value: object) -> str:
    """Recursively join YAML text fields that may be str, list[str], or nested lists."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_flatten_yaml_text(item) for item in value)
    return str(value)


class CosaiParser(BaseParser):
    framework_id = "cosai"
    framework_name = "CoSAI Landscape of AI Security Risk Map"
    version = "1.0"
    source_url = "https://cosai.dev"
    mapping_unit_level = "control"
    expected_count = 55
    fetched_date = "2026-04-28"

    def parse(self) -> list[Control]:
        controls: list[Control] = []

        # The CoSAI checkout keeps its taxonomy under risk-map/, matching the
        # upstream repo layout recorded in SOURCE_MANIFEST.md. Reading from the
        # framework root found nothing, so this parser could not run against a
        # faithful copy of its own source.
        risk_map = self.raw_dir / "risk-map"

        with open(risk_map / "controls.yaml", encoding="utf-8") as f:
            ctrl_data = yaml.safe_load(f)
        for ctrl in ctrl_data.get("controls", []):
            description = _flatten_yaml_text(ctrl.get("description", ""))
            controls.append(Control(
                control_id=ctrl["id"],
                title=ctrl["title"],
                description=description,
                hierarchy_level="control",
                metadata={
                    "category": ctrl.get("category", ""),
                    "personas": ctrl.get("personas", []),
                    "components": ctrl.get("components", []),
                },
            ))

        with open(risk_map / "risks.yaml", encoding="utf-8") as f:
            risk_data = yaml.safe_load(f)
        for risk in risk_data.get("risks", []):
            description = _flatten_yaml_text(risk.get("shortDescription", ""))
            long_desc = _flatten_yaml_text(risk.get("longDescription", ""))
            controls.append(Control(
                control_id=risk["id"],
                title=risk["title"],
                description=description,
                full_text=long_desc if long_desc else None,
                hierarchy_level="risk",
                metadata={
                    "category": risk.get("category", ""),
                    "controls": risk.get("controls", []),
                },
            ))

        return controls


if __name__ == "__main__":
    parser = CosaiParser()
    parser.run()
