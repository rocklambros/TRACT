"""Parser for CSA AI Controls Matrix (AICM) — Tier 1 structured JSON."""
from __future__ import annotations

import logging

from tract.io import load_json
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CsaAicmParser(BaseParser):
    framework_id = "csa_aicm"
    framework_name = "CSA AI Controls Matrix"
    version = "1.0.3"
    source_url = "https://cloudsecurityalliance.org/artifacts/ai-controls-matrix"
    mapping_unit_level = "control"
    expected_count = 243

    @staticmethod
    def _flatten_guidelines(field: str | dict[str, str] | None) -> str:
        if field is None:
            return ""
        if isinstance(field, str):
            return field
        return "\n".join(f"{k}: {v}" for k, v in field.items() if v)

    def parse(self) -> list[Control]:
        data = load_json(self.raw_dir / "csa_aicm.json")
        controls: list[Control] = []

        for raw_ctrl in data["controls"]:
            domain_id = raw_ctrl["domain_full"].split(" - ")[-1] if " - " in raw_ctrl.get("domain_full", "") else raw_ctrl.get("domain", "")

            full_parts = [raw_ctrl.get("specification", "")]
            impl = self._flatten_guidelines(raw_ctrl.get("implementation_guidelines"))
            audit = self._flatten_guidelines(raw_ctrl.get("auditing_guidelines"))
            if impl:
                full_parts.append(impl)
            if audit:
                full_parts.append(audit)
            full_text = "\n\n".join(p for p in full_parts if p) if len(full_parts) > 1 else None

            controls.append(Control(
                control_id=raw_ctrl["id"],
                title=raw_ctrl["title"],
                description=raw_ctrl["specification"],
                full_text=full_text,
                hierarchy_level="control",
                parent_id=domain_id,
                parent_name=raw_ctrl.get("domain", ""),
                metadata={
                    "control_type": raw_ctrl.get("control_type", ""),
                    "domain": raw_ctrl.get("domain", ""),
                },
            ))

        return controls


if __name__ == "__main__":
    parser = CsaAicmParser()
    parser.run()
