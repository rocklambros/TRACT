"""Parser for NIST AI 100-2e2023, the adversarial machine learning taxonomy.

The second of the two LOFO evaluation folds that reached the corpus through
parsers/fetch_opencre.py with section names but no text. It is the larger of
the two: 45 training links and 28 eval items, 19% of the evaluation set, all
of them previously scored as three-word titles.

The document is a taxonomy, so OpenCRE links its numbered sections rather than
discrete controls. Each numbered heading becomes one mapping unit, with the
body text between it and the next heading as its description.

Source: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf
"""
from __future__ import annotations

import logging
import re
from typing import Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PDF_NAME: Final[str] = "nist_ai_100_2_e2023.pdf"

# "2.3.1 Availability Poisoning". Up to four levels; the trailing text must
# start capitalised and stay short, which excludes body sentences that happen
# to begin with a figure number.
HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][A-Za-z0-9 ,\-/()']{3,70})\s*$",
    re.MULTILINE,
)
# Front matter and back matter carry no taxonomy.
SKIP_TITLES: Final[frozenset[str]] = frozenset({
    "NIST AI 100-2e2023", "Introduction", "References", "Glossary",
    "Acknowledgments", "Abstract", "Appendix",
})
MIN_BODY_CHARS: Final[int] = 120
DESCRIPTION_CAP: Final[int] = 2000

# Mitigation techniques are enumerated inside the Mitigations subsections
# rather than given headings of their own: "1. Adversarial training: Introduced
# by Goodfellow et al. ...". OpenCRE links several of them by name, so they are
# mapping units even though the document does not number them as sections.
# The PDF wraps at roughly eighty characters, so the definition always spans
# several lines; the body is taken by offset after the match rather than by a
# greedy group, which would have to cross newlines to find anything.
TECHNIQUE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|\n)\s*\d+\.\s+([A-Z][A-Za-z0-9 ,\-/()']{3,60}):\s",
)
MIN_TECHNIQUE_CHARS: Final[int] = 80
TECHNIQUE_BODY_CHARS: Final[int] = 2500
_WHITESPACE = re.compile(r"\s+")
# Page furniture that pdfplumber interleaves with the prose.
_NOISE = re.compile(
    r"(NIST AI 100-2e2023|Adversarial Machine Learning:|"
    r"A Taxonomy and Terminology of Attacks and Mitigations|"
    r"This publication is available free of charge from:[^\n]*)",
)


# pdfplumber drops the fi/fl ligature glyphs in this document, yielding
# "classifcation", "verifcation", "identifed". That silently breaks the join:
# the parser emitted "Formal verifcation" while OpenCRE links "Formal
# verification", so the item could never resolve and fell back to its title.
_LIGATURE_REPAIRS: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(bad, re.IGNORECASE), good) for bad, good in (
        (r"\bclassifc", "classific"), (r"\bverifc", "verific"),
        (r"\bidentifed\b", "identified"), (r"\bspecifc", "specific"),
        (r"\bdefn", "defin"), (r"\bsignifcant", "significant"),
        (r"\bconfguration", "configuration"), (r"\bmodifcation", "modification"),
        (r"\bcertifcation", "certification"), (r"\bnotifcation", "notification"),
        (r"\bartifcial\b", "artificial"), (r"\bconfdence\b", "confidence"),
        (r"\bsufcient", "sufficient"), (r"\befcient", "efficient"),
        (r"\bdifcult", "difficult"), (r"\bfltering\b", "filtering"),
    )
)
# Words split across a line break: "sys- tems" -> "systems".
_HYPHEN_BREAK = re.compile(r"([a-z])-\s+([a-z])")


def _repair(text: str) -> str:
    for pattern, replacement in _LIGATURE_REPAIRS:
        text = pattern.sub(replacement, text)
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def _clean(text: str) -> str:
    return _repair(_WHITESPACE.sub(" ", _NOISE.sub(" ", text)).strip())


def _alt_titles(title: str) -> list[str]:
    """Names OpenCRE is likely to link this section under.

    Declared rather than fuzzy-matched at lookup time: the index attaches prose
    by exact name, and guessing there would risk binding the wrong text to a
    control.
    """
    # Strip a trailing "and Mitigations" only. The previous version also
    # appended " Attacks" to the stripped stem, generating "Evasion Attacks
    # Attacks" and the bare "Evasion", neither of which names anything, and
    # each of which is another chance to collide with a real title elsewhere
    # in the document.
    variants = set()
    for suffix in (" and Mitigations", " and Defenses"):
        if title.endswith(suffix):
            variants.add(title[: -len(suffix)].strip())
    return sorted(v for v in variants if v and v != title and len(v) > 3)


class NistAi1002Parser(BaseParser):
    framework_id = "nist_ai_100_2"
    framework_name = "NIST AI 100-2"
    version = "e2023"
    source_url = "https://doi.org/10.6028/NIST.AI.100-2e2023"
    mapping_unit_level = "taxonomy_section"
    expected_count = 20
    fetched_date = "2026-08-14"

    def parse(self) -> list[Control]:
        source = self.raw_dir / PDF_NAME
        with pdfplumber.open(source) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        logger.info("Extracted %d characters from %d pages", len(text), len(pdf.pages))

        # "17 NIST AI 100-2e2023" is a running page header, not a section. It
        # matched HEADING_RE, and although SKIP_TITLES stopped it being emitted
        # it still terminated the PRECEDING section's body, cutting five
        # sections mid-word ("...machine learning as a ser-"). Drop it from the
        # boundary list, not just from the output.
        matches = [
            m for m in HEADING_RE.finditer(text)
            if m.group(2).strip() not in SKIP_TITLES
        ]
        controls: list[Control] = []
        seen: set[str] = set()

        for i, match in enumerate(matches):
            number, title = match.group(1), _repair(match.group(2).strip())
            if title in SKIP_TITLES or number in seen:
                continue

            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = _clean(text[match.end():end])
            if len(body) < MIN_BODY_CHARS:
                # A heading repeated in the table of contents, or a section
                # whose body is a figure. Neither is worth more than its title.
                continue

            seen.add(number)
            controls.append(Control(
                control_id=number,
                title=title,
                description=body[:DESCRIPTION_CAP],
                full_text=body if len(body) > DESCRIPTION_CAP else None,
                hierarchy_level=f"level_{number.count('.') + 1}",
                metadata={"section": number, "alt_titles": _alt_titles(title)},
            ))

        controls.extend(self._parse_techniques(text, {c.title for c in controls}))
        logger.info("Parsed %d NIST AI 100-2 mapping units", len(controls))
        return controls

    def _parse_techniques(self, text: str, taken: set[str]) -> list[Control]:
        """Enumerated mitigation techniques, which have no section number."""
        found: list[Control] = []
        seen: set[str] = set()

        for match in TECHNIQUE_RE.finditer(text):
            title = _clean(match.group(1))
            if title in taken or title.lower() in seen or title in SKIP_TITLES:
                continue

            # Run to the next enumerated item or numbered heading, whichever
            # comes first.
            tail = text[match.end(): match.end() + TECHNIQUE_BODY_CHARS]
            stops = [m.start() for m in (TECHNIQUE_RE.search(tail),
                                         HEADING_RE.search(tail)) if m]
            body = _clean(tail[: min(stops)] if stops else tail)
            if len(body) < MIN_TECHNIQUE_CHARS:
                continue

            seen.add(title.lower())
            found.append(Control(
                control_id=f"technique:{title.lower().replace(' ', '_')}",
                title=title,
                description=body[:DESCRIPTION_CAP],
                full_text=body if len(body) > DESCRIPTION_CAP else None,
                hierarchy_level="technique",
                metadata={"kind": "mitigation_technique"},
            ))

        logger.info("Parsed %d enumerated mitigation techniques", len(found))
        return found


if __name__ == "__main__":
    NistAi1002Parser().run()
