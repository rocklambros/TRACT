"""Parser for NIST AI RMF 1.0 — Tier 3 markdown regex extraction."""
from __future__ import annotations

import logging
import re

from typing import ClassVar

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUBCATEGORY_RE = re.compile(
    r"\*\*(?P<func>GOVERN|MAP|MEASURE|MANAGE)\s+(?P<cat>\d+)\.(?P<sub>\d+)[:.]?\*\*\s*(?P<title>[^\n]*)",
    re.MULTILINE,
)

FUNCTION_NAMES: dict[str, str] = {
    "GOVERN": "Govern",
    "MAP": "Map",
    "MEASURE": "Measure",
    "MANAGE": "Manage",
}


class NistAiRmfParser(BaseParser):
    framework_id = "nist_ai_rmf"
    framework_name = "NIST AI Risk Management Framework"
    version = "1.0"
    source_url = "https://doi.org/10.6028/NIST.AI.100-1"
    mapping_unit_level = "subcategory"
    expected_count = 72
    fetched_date: ClassVar[str] = "2026-04-28"
    # 55 of 72 subcategories clear HONEST_PROSE_MIN_CHARS, giving 0.7639, and
    # the floor fires at 54/72 (0.7500). [measured 2026-08-19]
    #
    # This is the lowest floor in the fleet, and a defect depresses it rather
    # than a terse source. SUBCATEGORY_RE captures the title as [^\n]*, which
    # stops at the first hard line wrap in the markdown. Every subcategory is a
    # single sentence, so the title takes the first line and the description
    # takes the remainder: 67 of the 72 descriptions open with a lowercase
    # continuation of their own title, and MEASURE 2.11 splits inside a markdown
    # token. The 17 that miss the threshold are the ones whose tail is short,
    # which makes 0.7639 a measure of where the converter wrapped lines rather
    # than of how much prose the source carries.
    #
    # The floor still earns its place, because it holds today's state against a
    # further regression. Repairing the split will raise the attainable value
    # toward 1.0, and that repair belongs in its own change with its own
    # re-measurement and a floor raised in the same commit.
    min_prose_fraction: ClassVar[float] = 0.76

    def parse(self) -> list[Control]:
        text = self.read_source("nist_ai_rmf_1.0.md")
        matches = list(SUBCATEGORY_RE.finditer(text))
        controls: list[Control] = []

        for i, m in enumerate(matches):
            func = m.group("func")
            cat = m.group("cat")
            sub = m.group("sub")
            title = m.group("title").strip().rstrip("*").strip()
            control_id = f"{func} {cat}.{sub}"

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            body = re.sub(r"\*\*[A-Z]+ \d+:\*\*[^\n]*\n?", "", body).strip()

            if not title and body:
                title = body[:80].split("\n")[0]

            controls.append(Control(
                control_id=control_id,
                title=title if title else control_id,
                description=body[:2000] if body else title,
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="subcategory",
                parent_id=f"{func} {cat}",
                parent_name=FUNCTION_NAMES.get(func, func),
                metadata={"function": func},
            ))

        return controls


if __name__ == "__main__":
    parser = NistAiRmfParser()
    parser.run()
