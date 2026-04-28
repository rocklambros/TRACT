"""Parser for EU GPAI Code of Practice — Tier 3 markdown extraction."""
from __future__ import annotations

import logging
import re

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHAPTER_RE = re.compile(r"^#\s+(?P<chapter>Transparency|Copyright|Safety and Security)\s+Chapter", re.MULTILINE)
COMMITMENT_RE = re.compile(r"^##\s+Commitment\s+(?P<num>\d+)\s+(?P<title>.+)$", re.MULTILINE)
MEASURE_RE = re.compile(r"^###\s+Measure\s+(?P<num>\d+\.\d+)\s+(?P<title>.+)$", re.MULTILINE)

CHAPTER_PREFIX: dict[str, str] = {
    "Transparency": "GPAI-T",
    "Copyright": "GPAI-C",
    "Safety and Security": "GPAI-SS",
}


class EuGpaiCopParser(BaseParser):
    framework_id = "eu_gpai_cop"
    framework_name = "EU GPAI Code of Practice"
    version = "2025"
    source_url = "https://digital-strategy.ec.europa.eu/en/policies/ai-pact"
    mapping_unit_level = "measure"
    expected_count = 40

    def parse(self) -> list[Control]:
        text = (self.raw_dir / "gpai_code_of_practice_combined.md").read_text(encoding="utf-8")
        controls: list[Control] = []

        chapters = list(CHAPTER_RE.finditer(text))
        chapter_ranges: list[tuple[str, int, int]] = []
        for i, cm in enumerate(chapters):
            ch_name = cm.group("chapter")
            start = cm.start()
            end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)
            chapter_ranges.append((ch_name, start, end))

        for ch_name, ch_start, ch_end in chapter_ranges:
            chapter_text = text[ch_start:ch_end]
            prefix = CHAPTER_PREFIX.get(ch_name, "GPAI")

            current_commitment = ""
            current_commitment_num = ""
            measures = list(MEASURE_RE.finditer(chapter_text))

            for i, m in enumerate(measures):
                measure_num = m.group("num")
                measure_title = m.group("title").strip()
                control_id = f"{prefix}-M{measure_num}"

                commitment_before = None
                for cm in COMMITMENT_RE.finditer(chapter_text[:m.start()]):
                    commitment_before = cm
                if commitment_before:
                    current_commitment = commitment_before.group("title").strip()
                    current_commitment_num = commitment_before.group("num")

                start = m.end()
                end = measures[i + 1].start() if i + 1 < len(measures) else ch_end - ch_start
                body = chapter_text[start:end].strip()

                next_commitment = COMMITMENT_RE.search(chapter_text[start:])
                if next_commitment and start + next_commitment.start() < end:
                    body = chapter_text[start:start + next_commitment.start()].strip()

                controls.append(Control(
                    control_id=control_id,
                    title=measure_title,
                    description=body[:2000] if body else measure_title,
                    full_text=body if len(body) > 2000 else None,
                    hierarchy_level="measure",
                    parent_id=f"{prefix}-C{current_commitment_num}" if current_commitment_num else None,
                    parent_name=current_commitment if current_commitment else None,
                    metadata={"chapter": ch_name},
                ))

        return controls


if __name__ == "__main__":
    parser = EuGpaiCopParser()
    parser.run()
