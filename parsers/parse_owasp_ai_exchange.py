"""Parser for the OWASP AI Exchange, keyed on its permalink slugs.

The previous version emitted programme-level items (AI_PROGRAM, SEC_PROGRAM)
while OpenCRE links this framework by anchor slug (ratelimit, monitoruse,
continuousvalidation). 26 of 64 links could not be joined, and those 26 were
exactly the items whose control text is character-identical to their
ground-truth hub name, in the fold carrying 43% of the evaluation weight.

Each control in the source is a heading followed by its own permalink:

    #### #MONITOR USE
    >Category: runtime information security control for input threats
    >Permalink: https://owaspai.org/go/monitoruse/

    **Description**
    Monitor use: observe, correlate, and log model usage ...

The slug in that permalink is the join key OpenCRE uses, so it becomes
control_id. Inline references of the form [#RATE LIMIT](/go/ratelimit/) point
at a definition elsewhere and are not themselves definitions, which is why a
heading must be followed by its own permalink to count.

Source: https://owaspai.org (owasp-ai-exchange repository, content/docs)
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_GLOB: Final[str] = "*.md"
# A markdown heading. The leading "#" inside the text ("#MONITOR USE") is the
# project's own convention for naming a control and is stripped from the title.
HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(#{2,6})\s*#?\s*(.+?)\s*$", re.MULTILINE,
)
PERMALINK_RE: Final[re.Pattern[str]] = re.compile(
    r"^>\s*Permalink:\s*https://owaspai\.org/go/([a-z0-9\-]+)/?\s*$", re.MULTILINE,
)
CATEGORY_RE: Final[re.Pattern[str]] = re.compile(
    r"^>\s*Category:\s*(.+?)\s*$", re.MULTILINE,
)
# A permalink belongs to the heading immediately above it, not to one further
# up the page. Two short metadata lines is the whole gap in this source.
MAX_HEADING_TO_PERMALINK_CHARS: Final[int] = 400
DESCRIPTION_CAP: Final[int] = 2000
MIN_BODY_CHARS: Final[int] = 40
_WHITESPACE = re.compile(r"\s+")


def _title_case(raw: str) -> str:
    """"#MONITOR USE" -> "Monitor use". Shouted headings are the convention."""
    text = raw.strip().lstrip("#").strip()
    if text.isupper():
        # "MONITOR USE" -> "Monitor use", but AI stays AI.
        text = text.capitalize().replace("Ai ", "AI ")
        if text.startswith("Ai"):
            text = "AI" + text[2:]
    return text


class OwaspAiExchangeParser(BaseParser):
    framework_id = "owasp_ai_exchange"
    framework_name = "OWASP AI Exchange"
    version = "2024"
    source_url = "https://owaspai.org"
    mapping_unit_level = "control"
    # The source defines more controls than OpenCRE links to (107 against 64
    # linked), so this tracks the source rather than the link set.
    expected_count = 107
    expected_count_is_floor: ClassVar[bool] = True
    fetched_date: ClassVar[str] = "2026-08-14"
    # All 107 controls carry a statement and none equals its title. The shortest
    # is 141 characters, so the attainable value is exactly 1.0 and the floor
    # fires at 106/107 (0.9907) if one control decays to its heading. [measured
    # 2026-08-19]
    min_prose_fraction: ClassVar[float] = 1.0

    def parse(self) -> list[Control]:
        controls: list[Control] = []
        seen: set[str] = set()

        # Globbed to find the names, read through the recording reader so
        # every file that contributed text lands in the manifest.
        for path in sorted(self.raw_dir.glob(SOURCE_GLOB)):
            name = path.relative_to(self.raw_dir).as_posix()
            text = self.read_source_bytes(name).decode("utf-8", errors="replace")
            headings = list(HEADING_RE.finditer(text))

            for index, heading in enumerate(headings):
                section_end = (
                    headings[index + 1].start()
                    if index + 1 < len(headings) else len(text)
                )
                section = text[heading.end():section_end]

                permalink = PERMALINK_RE.search(section)
                if not permalink or permalink.start() > MAX_HEADING_TO_PERMALINK_CHARS:
                    # A heading with no permalink of its own is prose structure,
                    # or the link is a reference to a control defined elsewhere.
                    continue

                slug = permalink.group(1)
                if slug in seen:
                    continue

                body = _WHITESPACE.sub(" ", section[permalink.end():]).strip()
                if len(body) < MIN_BODY_CHARS:
                    continue

                category = CATEGORY_RE.search(section)
                seen.add(slug)
                controls.append(Control(
                    control_id=slug,
                    title=_title_case(heading.group(2)),
                    description=body[:DESCRIPTION_CAP],
                    full_text=body if len(body) > DESCRIPTION_CAP else None,
                    hierarchy_level="control",
                    metadata={
                        "slug": slug,
                        "category": category.group(1).strip() if category else "",
                        "source_file": path.name,
                    },
                ))

        logger.info("Parsed %d OWASP AI Exchange controls", len(controls))
        return controls


if __name__ == "__main__":
    OwaspAiExchangeParser().run()
