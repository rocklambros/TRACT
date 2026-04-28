"""Parser for NIST AI RMF 1.0 — Tier 3 markdown regex extraction."""
from __future__ import annotations

import logging
import re

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

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "nist_ai_rmf_1.0.md").read_text(encoding="utf-8")
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
