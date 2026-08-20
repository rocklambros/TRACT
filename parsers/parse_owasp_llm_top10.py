"""Parser for OWASP Top 10 for LLM Applications 2025 — Tier 3 markdown."""
from __future__ import annotations

import logging
import re

from typing import ClassVar

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DESCRIPTION_RE = re.compile(r"^####\s+\*\*Description\*\*", re.MULTILINE)

LLM_ENTRIES: list[tuple[str, str]] = [
    ("LLM01:2025", "Prompt Injection"),
    ("LLM02:2025", "Sensitive Information Disclosure"),
    ("LLM03:2025", "Supply Chain Vulnerabilities"),
    ("LLM04:2025", "Data and Model Poisoning"),
    ("LLM05:2025", "Improper Output Handling"),
    ("LLM06:2025", "Excessive Agency"),
    ("LLM07:2025", "System Prompt Leakage"),
    ("LLM08:2025", "Vector and Embedding Weaknesses"),
    ("LLM09:2025", "Misinformation"),
    ("LLM10:2025", "Unbounded Consumption"),
]


class OwaspLlmTop10Parser(BaseParser):
    framework_id = "owasp_llm_top10"
    framework_name = "OWASP Top 10 for LLM Applications 2025"
    version = "2025"
    source_url = "https://genai.owasp.org"
    mapping_unit_level = "risk"
    expected_count = 10
    fetched_date: ClassVar[str] = "2026-04-28"
    # All 10 risks carry a statement and none equals its title. The shortest is
    # 1,970 characters, so the attainable value is exactly 1.0 and the floor
    # fires at 9/10 (0.9000) if one risk decays to its heading. [measured
    # 2026-08-19]
    min_prose_fraction: ClassVar[float] = 1.0

    def parse(self) -> list[Control]:
        text = self.read_source("owasp_llm_top_10_2025.md")
        desc_matches = list(DESCRIPTION_RE.finditer(text))

        if len(desc_matches) != 10:
            raise ValueError(
                f"Expected 10 Description sections, found {len(desc_matches)}"
            )

        controls: list[Control] = []
        for i, (control_id, title) in enumerate(LLM_ENTRIES):
            start = desc_matches[i].end()
            end = desc_matches[i + 1].start() if i + 1 < len(desc_matches) else len(text)
            body = text[start:end].strip()

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=body[:2000] if body else title,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="risk",
            ))

        return controls


if __name__ == "__main__":
    parser = OwaspLlmTop10Parser()
    parser.run()
