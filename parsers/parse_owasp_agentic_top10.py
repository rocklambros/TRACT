"""Parser for OWASP Top 10 for Agentic Applications 2026 — Tier 3 markdown."""
from __future__ import annotations

import logging
import re

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASI_MAPPING: list[tuple[str, str]] = [
    ("ASI01", "Agent Goal Hijack"),
    ("ASI02", "Tool Misuse and Exploitation"),
    ("ASI03", "Identity and Privilege Abuse"),
    ("ASI04", "Agentic Supply Chain Vulnerabilities"),
    ("ASI05", "Unexpected Code Execution (RCE)"),
    ("ASI06", "Memory & Context Poisoning"),
    ("ASI07", "Insecure Inter-Agent Communication"),
    ("ASI08", "Cascading Failures"),
    ("ASI09", "Human-Agent Trust Exploitation"),
    ("ASI10", "Rogue Agents"),
]

RISK_SECTION_RE = re.compile(r"^#{1,6}\s*\*{0,2}Description\*{0,2}\s*$", re.MULTILINE)


class OwaspAgenticTop10Parser(BaseParser):
    framework_id = "owasp_agentic_top10"
    framework_name = "OWASP Top 10 for Agentic Applications 2026"
    version = "2026"
    source_url = "https://genai.owasp.org"
    mapping_unit_level = "risk"
    expected_count = 10

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "owasp_agentic_top10_2026.md").read_text(encoding="utf-8")
        desc_matches = list(RISK_SECTION_RE.finditer(text))
        controls: list[Control] = []

        for i, (asi_id, asi_name) in enumerate(ASI_MAPPING):
            if i >= len(desc_matches):
                logger.warning("Could not find Description section %d for %s", i + 1, asi_id)
                continue

            m = desc_matches[i]
            start = m.end()

            next_major = re.search(
                r"^#{1,4}\s+\*{0,2}(?:Common Examples|Prevention|Example Attack)",
                text[start:],
                re.MULTILINE,
            )
            if next_major:
                body = text[start:start + next_major.start()].strip()
            elif i + 1 < len(desc_matches):
                body = text[start:desc_matches[i + 1].start()].strip()
            else:
                body = text[start:start + 3000].strip()

            controls.append(Control(
                control_id=asi_id,
                title=asi_name,
                description=body[:2000] if body else asi_name,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="risk",
            ))

        return controls


if __name__ == "__main__":
    parser = OwaspAgenticTop10Parser()
    parser.run()
