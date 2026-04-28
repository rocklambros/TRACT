"""Parser for NIST AI 600-1 GenAI Profile — Tier 3 markdown extraction."""
from __future__ import annotations

import logging
import re

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RISK_SECTION_RE = re.compile(
    r"\*\*2\.(?P<num>\d+)\.?\*\*\s+\*\*(?P<title>[^*]+)\*\*",
    re.MULTILINE,
)

GAI_RISK_IDS: dict[int, str] = {
    1: "GAI-CBRN",
    2: "GAI-CONFAB",
    3: "GAI-DANGEROUS",
    4: "GAI-PRIVACY",
    5: "GAI-ENVIRON",
    6: "GAI-BIAS",
    7: "GAI-HUMANAI",
    8: "GAI-INFOINTEG",
    9: "GAI-INFOSEC",
    10: "GAI-IP",
    11: "GAI-OBSCENE",
    12: "GAI-VALUECHAIN",
}


class NistAi600Parser(BaseParser):
    framework_id = "nist_ai_600_1"
    framework_name = "NIST AI 600-1 Generative AI Profile"
    version = "1.0"
    source_url = "https://doi.org/10.6028/NIST.AI.600-1"
    mapping_unit_level = "risk_category"
    expected_count = 12

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "nist_ai_600_1.md").read_text(encoding="utf-8")
        matches = list(RISK_SECTION_RE.finditer(text))
        controls: list[Control] = []

        for i, m in enumerate(matches):
            num = int(m.group("num"))
            title = m.group("title").strip()
            risk_id = GAI_RISK_IDS.get(num, f"GAI-{num:02d}")

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            rmf_refs = re.findall(r"(GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+", body)

            controls.append(Control(
                control_id=risk_id,
                title=title,
                description=body[:2000] if body else title,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="risk_category",
                metadata={"rmf_subcategories": sorted(set(rmf_refs))} if rmf_refs else None,
            ))

        return controls


if __name__ == "__main__":
    parser = NistAi600Parser()
    parser.run()
