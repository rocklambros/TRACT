"""Parser for NIST SP 800-63B, Authentication and Lifecycle Management.

The revision matters more than the parsing does. OpenCRE's 79 curated links
carry bare 800-63-3B section numbers (5.1.1.2, 6.1.2.3, A.3). Revision 4
restructured the document and renumbered it, so none of those numbers exists in
revision 4B and fetching it would leave every link unjoinable while looking
like a successful fetch. Measured on the two revisions: 4B matches 0 of the 25
distinct curated ids, 3B matches 24 of 25. The staged file is 3B. Its title
element says NIST Special Publication 800-63B and its headings carry the dotted
numbering the links use.

There is no digest gate, and the absence is deliberate rather than an
oversight. pages.nist.gov sits behind Cloudflare, which injects a per-response
bot-challenge nonce into the body, so two fetches of the identical document
hash differently and scripts/fetch_frameworks.py leaves expected_sha256 unset
for this source alone. Pinning a hash here would make --accept-new-hash routine
rather than an alert.

The substitute gate is structural. REQUIRED_SECTION_IDS holds every distinct
curated section_id the document can answer, and _check_structure refuses to
return unless all of them are present and unless the document yields at least
MIN_NUMBERED_SECTIONS numbered headings. A revision swap takes the dotted
numbering to zero, so it fails both halves rather than producing a corpus whose
join silently went to zero. tests/test_parse_nist_800_63.py derives the same
set from data/training/hub_links_curated.jsonl and holds the two equal, which
fails in both directions: an entry here that no link needs, and a link id that
this list does not require.

The mapping unit is the numbered section. Its id is the dotted number in the
heading text, its title is the rest of the heading, and its statement is
everything between that heading and the next heading at any level.

Measured on the staged file: 118 numbered headings, all distinct, none
duplicated. Three carry no body at all (5.1, 5.2 and 6.1.2, whose text lives
entirely in their subsections) and two carry only the 28-character sentence
that opens an informative chapter (8 and 11). Those five restate or under-run
their own titles, so ProseIndex excludes them and the corpus report counts them
under dropped_by_prose_rule. None of the five is linked.

description is capped by this parser, and the cap is load bearing rather than
cosmetic. BaseParser._sanitize_control calls sanitize_text(description,
return_full=True) and assigns whatever that returns to full_text, discarding
the parser's own value, whenever the description exceeds
DESCRIPTION_MAX_LENGTH. Twenty-two of the 118 section bodies run over that
limit, the longest at 6,866 characters, so without the cap the base class would
rewrite the anchor of 22 sections behind this parser's back. The cap keeps
every shipped description strictly under the limit and full_text carries the
complete statement, which is the field ProseIndex reads. Every cut writes a
before/after pair through write_repair_audit, because a count says a repair
fired and only the pair says what moved.

Nothing here is assembled, so text_origin stays unset. Every character in a
statement is NIST's, in NIST's order. The longest prefix shared by all indexed
anchors is 0 characters, measured, so there is no boilerplate head to strip and
none is stripped.

One curated link cannot be answered and no alternate is written for it. Its
section_id and section_name are both the five-character fragment "are g",
which is an extraction artifact in OpenCRE's own data. The fragment does occur
in this document, once, inside the Appendix A phrase "they are generated", but
that link's CRE is about the samesite attribute on session cookies and
Appendix A is about the entropy of memorized secrets. An alt_ids entry would
therefore resolve the link to demonstrably the wrong section, which is worse
than leaving it unresolved. The ceiling is 78 of 79.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import ClassVar, Final

from bs4 import BeautifulSoup, Tag

from tract.config import DESCRIPTION_MAX_LENGTH
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "sp800_63b.html"

HEADING_TAG: Final[re.Pattern[str]] = re.compile(r"^h[1-6]$")
# "5.1.1.2 Memorized Secret Verifiers" and "A.3 Complexity". Both forms appear
# among the curated section ids. The letter class is `[A-Z]` rather than the
# literal `A` so an errata that adds an Appendix B is parsed instead of
# dropped: measured on the staged file the two spellings find the identical 118
# headings, so the wider class costs nothing today.
NUMBERED: Final[re.Pattern[str]] = re.compile(
    r"^\s*((?:[A-Z]\.)?\d+(?:\.\d+)*)\.?\s+(\S.*)$"
)

# Every distinct curated section_id this document can answer, which is all 25
# minus the "are g" fragment. Checked against
# data/training/hub_links_curated.jsonl by
# tests/test_parse_nist_800_63.py::TestRequiredSectionIds, which derives the
# set from the link file rather than reading it back from here. [measured]
REQUIRED_SECTION_IDS: Final[tuple[str, ...]] = (
    "5.1.1.1", "5.1.1.2", "5.1.2.2", "5.1.3.2", "5.1.4.2", "5.1.5.2",
    "5.1.7.2", "5.2.1", "5.2.10", "5.2.2", "5.2.3", "5.2.5", "5.2.6",
    "5.2.8", "5.2.9", "6.1.2.3", "6.1.3", "6.1.4", "7.1", "7.1.1", "7.1.2",
    "7.2", "7.2.1", "A.3",
)
# The curated id this document cannot answer. Named so the required set above
# can be derived from the link file minus this one entry, and so a reader can
# see that the gap is declared rather than forgotten.
UNANSWERABLE_SECTION_ID: Final[str] = "are g"

# Well under the 118 measured, so ordinary errata cannot trip it, while a
# revision swap takes the dotted numbering to zero and does.
MIN_NUMBERED_SECTIONS: Final[int] = 100

# One below the length at which _sanitize_control takes full_text away from
# this parser, so a shipped description can never reach the limit even when the
# word-boundary search finds nothing to cut on.
DESCRIPTION_BUDGET: Final[int] = DESCRIPTION_MAX_LENGTH - 1

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class Nist80063Parser(BaseParser):
    framework_id: ClassVar[str] = "nist_800_63"
    framework_name: ClassVar[str] = "NIST 800-63"
    version: ClassVar[str] = "800-63B rev 3"
    source_url: ClassVar[str] = "https://pages.nist.gov/800-63-3/sp800-63b.html"
    mapping_unit_level: ClassVar[str] = "section"
    # 118 numbered headings in the staged document. [measured] Exact rather
    # than a floor: SP 800-63-3 is a finalised publication under errata-only
    # maintenance, and its successor is a separate document with its own URL,
    # so the section list does not grow. COUNT_TOLERANCE gives a band of 107 to
    # 129, which a revision swap clears in the wrong direction by 118.
    expected_count: ClassVar[int] = 118
    fetched_date: ClassVar[str] = "2026-08-15"
    # 113 of the 118 sections carry a statement longer than their own title.
    # [measured] The floor sits one section below that, so losing any single
    # section's body fails the gate, and the attainable value 0.9576 clears it.
    min_prose_fraction: ClassVar[float] = 0.95

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
        audit_dir: Path | None = None,
    ) -> None:
        # Instance attributes rather than ClassVars. A fixture-backed test
        # declares its own structural gate by assignment, and assigning to a
        # ClassVar through an instance is both a mypy error and a mutation of
        # every other parser instance in the process.
        super().__init__(raw_dir, output_dir, audit_dir)
        self.required_section_ids: tuple[str, ...] = REQUIRED_SECTION_IDS
        self.min_sections: int = MIN_NUMBERED_SECTIONS

    def parse(self) -> list[Control]:
        html = self.read_source(SOURCE_FILE)
        repairs: list[dict[str, object]] = []
        controls = self.sections_from_html(html, repairs)
        self._check_structure(controls)
        # Written after the structure check, so the file on disk always
        # describes a run against the revision the links key to.
        self.write_repair_audit(repairs)
        logger.info(
            "%s: %d numbered sections, %d of the %d required ids present, "
            "%d description(s) capped",
            self.framework_id, len(controls),
            len({c.control_id for c in controls} & set(self.required_section_ids)),
            len(self.required_section_ids), len(repairs),
        )
        return controls

    def _check_structure(self, controls: list[Control]) -> None:
        """Refuse a document that is not the revision the links key to.

        Raises:
            ValueError: If the numbering is too sparse, or a required section
                id is absent.
        """
        if len(controls) < self.min_sections:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} yields {len(controls)} "
                f"numbered sections, under the floor of {self.min_sections}. "
                f"Revision 4 restructured this document and renumbered it, and "
                f"fetching it would leave all 79 curated links unjoinable "
                f"while looking like a success. This source is not "
                f"digest-pinned, because Cloudflare injects a per-response "
                f"nonce, so this count is the pin."
            )
        found = {c.control_id for c in controls}
        missing = sorted(set(self.required_section_ids) - found)
        if missing:
            raise ValueError(
                f"{self.framework_id}: section id(s) {missing} are absent from "
                f"{SOURCE_FILE}. Every one of them is targeted by a curated "
                f"OpenCRE link, so their absence means this is not the "
                f"revision the links were written against."
            )

    @classmethod
    def sections_from_html(
        cls, html: str, repairs: list[dict[str, object]] | None = None,
    ) -> list[Control]:
        """One Control per numbered heading, in document order.

        `repairs` is an optional sink. Each capped description appends a
        before/after pair to it, which parse() hands to write_repair_audit.

        Raises:
            ValueError: If two headings claim the same section number, which
                would let one section's statement silently replace another's.
        """
        soup = BeautifulSoup(html, "lxml")
        found: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for heading in soup.find_all(HEADING_TAG):
            match = NUMBERED.match(
                _WHITESPACE.sub(" ", heading.get_text()).strip()
            )
            if match is None:
                continue
            number, title = match.group(1), match.group(2).strip()
            if number in seen:
                raise ValueError(
                    f"nist_800_63: section number {number!r} appears on more "
                    f"than one heading. Emitting both would give one "
                    f"control_id two statements and let whichever is written "
                    f"last answer every link to it."
                )
            seen.add(number)
            found.append((number, title, cls._body(heading)))

        return [
            cls._control(number, title, body, seen, repairs)
            for number, title, body in found
        ]

    @classmethod
    def _control(
        cls,
        number: str,
        title: str,
        body: str,
        numbers: set[str],
        repairs: list[dict[str, object]] | None,
    ) -> Control:
        """One section, with its statement split across description and full_text.

        full_text is set only when the cap fired, so it always carries text the
        description does not. A section whose whole statement fits stores it
        once, and a section with no statement at all stores its own title and is
        excluded from the prose index rather than answering a link with a
        heading.
        """
        if not body:
            statement, overflow = title, None
        else:
            statement, overflow = cls._cap_description(body)
            if overflow is not None and repairs is not None:
                repairs.append({
                    "control_id": number,
                    "repair": "description_capped_to_protect_full_text",
                    "field": "description",
                    "before": body,
                    "after": statement,
                })
        parent = number.rsplit(".", 1)[0] if "." in number else None
        return Control(
            control_id=number,
            title=title,
            description=statement,
            full_text=overflow,
            hierarchy_level="section",
            # Only when the parent is itself a numbered section. Appendix A's
            # heading carries no number, so A.3's nominal parent "A" exists in
            # no control and a dangling reference would be worse than none.
            parent_id=parent if parent in numbers else None,
        )

    @staticmethod
    def _cap_description(text: str) -> tuple[str, str | None]:
        """Split a statement into a capped description and its full form.

        Returns (description, full_text_or_None). full_text is None when the
        statement already fits, so a caller can tell a cut from a pass-through
        without comparing lengths.

        The budget is one character under DESCRIPTION_MAX_LENGTH, which is the
        length at which _sanitize_control replaces this parser's full_text with
        the sanitised description. Sanitisation only shrinks this corpus, since
        _body already collapsed whitespace and BeautifulSoup already resolved
        the entities, so a raw cut at the budget lands under the limit. The
        test suite asserts the shipped result rather than trusting that.
        """
        if len(text) <= DESCRIPTION_BUDGET:
            return text, None
        cut = text[:DESCRIPTION_BUDGET].rsplit(" ", 1)[0]
        # Mirrors sanitize_text's own guard: a statement with no space in its
        # first half would otherwise be cut to almost nothing.
        if len(cut) < DESCRIPTION_BUDGET // 2:
            cut = text[:DESCRIPTION_BUDGET]
        return cut, text

    @staticmethod
    def _body(heading: Tag) -> str:
        """Text between this heading and the next heading at any level."""
        parts: list[str] = []
        for sibling in heading.next_siblings:
            name = getattr(sibling, "name", None)
            if name and HEADING_TAG.match(name):
                break
            text = (
                sibling.get_text(" ") if hasattr(sibling, "get_text")
                else str(sibling)
            )
            cleaned = _WHITESPACE.sub(" ", text).strip()
            if cleaned:
                parts.append(cleaned)
        return " ".join(parts).strip()


def main() -> None:
    Nist80063Parser().run()


if __name__ == "__main__":
    main()
