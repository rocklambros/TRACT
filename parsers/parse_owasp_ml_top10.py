"""Parser for the OWASP Machine Learning Security Top 10 (2023).

One of the two LOFO evaluation folds that reached the corpus through
parsers/fetch_opencre.py, which carries section names but not text. Its seven
linked entries were being evaluated as three-word titles while production hands
the model paragraphs, and at n=7 that fold moves 0.143 per item.

Source: OWASP/www-project-machine-learning-security-top-10 (docs/ML*.md)
"""
from __future__ import annotations

import logging
import re
import zipfile
from typing import Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "owasp_ml_top10.zip"
# docs/ML01_2023-Input_Manipulation_Attack.md ... ML10_2023-Model_Poisoning.md.
# Anchored on docs/ so the cheatsheets/ variants of the same names are excluded.
ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r"/docs/(ML(\d{2})_(\d{4})-[A-Za-z0-9_]+)\.md$"
)
FRONT_MATTER_RE: Final[re.Pattern[str]] = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
TITLE_RE: Final[re.Pattern[str]] = re.compile(r"^title:\s*(.+)$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
DESCRIPTION_CAP: Final[int] = 2000


def _clean(markdown: str) -> str:
    """Strip markdown decoration, keeping the prose."""
    text = re.sub(r"^#{1,6}\s*", "", markdown, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()


class OwaspMlTop10Parser(BaseParser):
    framework_id = "owasp_ml_top10"
    framework_name = "OWASP Top10 for ML"
    version = "2023"
    source_url = "https://owasp.org/www-project-machine-learning-security-top-10/"
    mapping_unit_level = "risk"
    expected_count = 10

    def parse(self) -> list[Control]:
        archive = self.raw_dir / ARCHIVE_NAME
        controls: list[Control] = []

        with zipfile.ZipFile(archive) as bundle:
            # Sort by archive path, not by the match object: re.Match has no
            # ordering and sorting tuples would compare them first.
            entries = sorted(
                (name, match)
                for name in bundle.namelist()
                if (match := ENTRY_RE.search(name))
            )
            for name, match in entries:
                body = bundle.read(name).decode("utf-8", errors="replace")

                heading = TITLE_RE.search(body)
                # "ML01:2023 Input Manipulation Attack" -> id and title.
                raw_title = heading.group(1).strip() if heading else match.group(1)
                control_id, _, title = raw_title.partition(" ")
                title = title.strip() or raw_title

                prose = _clean(FRONT_MATTER_RE.sub("", body))
                if not prose:
                    prose = title

                controls.append(Control(
                    control_id=control_id.strip(),
                    title=title,
                    description=prose[:DESCRIPTION_CAP],
                    full_text=prose if len(prose) > DESCRIPTION_CAP else None,
                    hierarchy_level="risk",
                    metadata={"slug": match.group(1), "year": match.group(3)},
                ))

        logger.info("Parsed %d OWASP ML Top 10 entries", len(controls))
        return controls


if __name__ == "__main__":
    OwaspMlTop10Parser().run()
