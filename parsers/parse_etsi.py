"""Parser for ETSI GR SAI 005, Securing Artificial Intelligence Mitigation Strategy Report.

RESTRICTED. ETSI's copyright notification reserves reproduction in any medium
without written permission, so this framework sits in
tract.config.RESTRICTED_FRAMEWORK_IDS, data/processed/frameworks/etsi.json is
gitignored, and parsers/merge_all_controls.py routes its prose to the
gitignored licensed overlay. Nothing in this file, its tests, or its commit
messages quotes the source. Every assertion the test suite makes about the real
document is a count, a length, a digest or a negative.

The mapping unit is the numbered clause, chosen against a measured alternative.
OpenCRE's 36 curated links carry 24 distinct technique names over 16 section
ids. Only 2 of those 24 names are clause headings, and the rest occur as bullet
lead phrases or mid-sentence cross-references between 1 and 30 times each.
Segmenting a technique out of running prose would mean guessing sentence
boundaries around an ambiguous match, and a wrong guess attributes a mitigation
to the wrong attack class behind a provenance record that reads as correct.

Three kinds of page furniture look like clause headings or like clause text,
and all three were shipping before this parser existed.

The first is the running header. pdfplumber renders the top of every page as
"<page number> <document identifier>", and pages 5, 6 and 7 are the only ones
whose number falls in CLAUSE's range, so all three presented as top-level
clauses whose heading was the document identifier. Because the headers come
first in page order, the real headings for clauses 5, 6 and 7 lost the slot to
a silent duplicate skip, and clause 7 shipped 22,639 characters of contents
page, bibliography and clause 4 summary tables as one control statement.
Every gate passed: 25 distinct numbers matched either way, the garbage was long
enough to clear the prose floor, no curated link targets a bare 5, 6 or 7 so
the corpus report could not see it, and the output file is gitignored so no
reviewer saw it in a diff. [measured on the pinned bytes, pdfplumber 0.11.10]

The second is the same header, plus the footer, landing in the middle of a
clause. 32 furniture lines fall inside clause bodies, across 14 of the 25
clauses, putting 656 characters of the document identifier into the anchor the
encoder reads. A framework-identifying token in an anchor is a shortcut a
bi-encoder learns instead of the mapping. [measured]

The third is Annex A. Clause 7 is the last numbered clause, so its body ran to
the end of the file and swallowed the change-history table and the document
history, 1,889 of its 2,776 characters. That table also supplies a fourth false
clause match, since "6 June 2020 0.0.7 ..." has the shape CLAUSE matches.
[measured]

Three mechanisms answer them, and each one has work the others do not do. A
candidate heading that is the document identifier is refused, which is what
stops the header taking a clause number, and it fires 3 times. Header and
footer lines are dropped from every clause body, which is what keeps the
identifier out of the anchor, and that fires 32 times. The clause range stops
at the first annex heading, which removes the change-history table rather than
filtering the one row of it that looks like a heading, and that fires once. A
clause number that still matches twice raises, because a silent skip is how the
headers won three slots, and it is the backstop for whatever furniture shape
arrives next.

Almost nothing else is declared, and that is deliberate. control_id IS the
clause number, so 34 of the 36 curated links resolve through the id channel
with no alias at all. Registering all 24 technique names as alternate titles
would be harmful rather than generous: two of them name two clauses each, and
because ProseIndex.lookup tries the title first, a name registered on one
clause would answer the link that named the other. NAME_SECTION_IDS therefore
holds exactly the two rows whose section_id is a technique name rather than a
clause number. Both were verified twice against the pinned bytes, once on where
the name occurs and once on the CRE the row targets.

Seven clauses are headings whose text lives entirely in their subclauses, so a
clause with no body of its own takes the concatenation of its descendants. Only
leaf clauses contribute, because an empty parent contributes an empty string
and nothing is duplicated. That concatenation is assembled text rather than a
passage anyone wrote, so a rolled-up clause carries text_origin = synthetic and
every roll-up writes a before and after pair through write_repair_audit. Four
of the 36 links land on one.

description is capped by this parser, and the cap is load bearing. Ruling R14:
BaseParser._sanitize_control calls sanitize_text(description, return_full=True)
and assigns the result to full_text, discarding whatever the parser wrote,
whenever the description exceeds DESCRIPTION_MAX_LENGTH. Seventeen of the 25
clause statements run over that limit, the longest at 38,008 characters, so
without the cap the base class would rewrite 17 anchors behind this parser's
back. Every cut writes a before and after pair to the same audit file.

The cost of the clause ruling is stated rather than discovered. The 36 links
reach 24 short section-name fallbacks today and 14 clause statements after, at
2.57 links each, which the corpus report records as a named regression of 10 on
the anchor column against a gain on the text column. The longest prefix shared
by all 25 stored statements is 0 characters, measured, so there is no
boilerplate head to strip and none is stripped.

Two properties are worth knowing before reading the join report.

MAX_ANCHOR_CHARS cuts every anchor at 2,150 characters, and a rolled-up parent
opens with the whole of its first child. Clause 6 and clause 6.1 therefore
truncate to the same 2,150 characters, as do clause 6.2 and clause 6.2.1, so
the 25 statements present 23 distinct anchors to the encoder. No curated link
targets 6, 6.2 or 6.2.1, so distinct_anchors and its pre-truncation twin both
read 14 and neither can see this. [measured]

wrong_anchor_risk measures 32 of 36 checked, against the 1 pre-registered in
tract.corpus_report.JOIN_WRONG_ANCHOR_BUDGET. One of the 32 is detector A, the
6.3.1 row whose section_name is clause 6.3's heading. The other 31 are detector
B, which flags an id-channel link whose section_name appears nowhere in the
title of the clause its id reached. That is a fact about OpenCRE's ETSI rows
rather than a wrong anchor this parser can fix: the names sit one level FINER
than the ids, naming a technique inside the clause the id identifies. It is the
DSOMM shape of ruling R11 inverted, and COARSE_NAME_RATIO cannot route it to
DETECTOR_B_INAPPLICABLE because that test looks for ids outnumbering names
while ETSI has 16 ids against 24 names. The budget entry is not edited here.
The disagreement is reported and left to its owner.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import ClassVar, Final

import pdfplumber

from tract.config import DESCRIPTION_MAX_LENGTH, HONEST_PROSE_MIN_CHARS
from tract.corpus_report import SYNTHETIC_TEXT_ORIGIN, TEXT_ORIGIN_METADATA_KEY
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "etsi_gr_sai005_v010101p.pdf"
SOURCE_SHA256: Final[str] = (
    "46c2b6b880928ffe2e763fbd6e0d0660a0aa7de0ff0071f5e0694582d91d5622"
)

# Clauses 1 through 4 are scope, references, definitions and an overview. The
# mapping units are the attack and mitigation clauses in 5 through 7.
#
# The 81-character bound on the heading is what excludes the contents page, and
# it is load bearing rather than cosmetic. Every one of the 25 clauses appears
# there too, as "<number> <heading>" followed by dot leaders to the page
# number, and the contents page comes first in page order. Those rows run from
# 137 to 170 characters while the longest real heading measures 55, so the
# bound separates them by a wide margin. [measured] A contents row that ever
# lands inside the bound collides with the clause it names and the duplicate
# check below refuses the parse, which is the outcome to want.
CLAUSE: Final[re.Pattern[str]] = re.compile(r"^([5-7](?:\.\d+){0,3})\s+(\S.{2,80})$")

# A candidate heading that is the document identifier. This is the guard that
# stops a running header taking a clause number, and it is written against the
# heading rather than against the number so a real clause is never dropped for
# sharing a number with a page. Exported so the tests can assert on the real
# document without quoting it.
DOCUMENT_IDENTIFIER: Final[re.Pattern[str]] = re.compile(
    r"^ETSI\s+(?:GR|GS|TS|TR|EG|EN)\s", re.IGNORECASE
)

# The running header, "<page number> <document identifier>", and the running
# footer, the publisher's name alone on its own line. Dropped from every clause
# body, which is a separate job from the heading guard above: the guard keeps
# the header out of the clause LIST, and this keeps it out of the clause TEXT.
RUNNING_HEADER: Final[re.Pattern[str]] = re.compile(
    r"^\d+\s+ETSI\s+(?:GR|GS|TS|TR|EG|EN)\s", re.IGNORECASE
)
RUNNING_FOOTER: Final[re.Pattern[str]] = re.compile(r"^ETSI$")

# Where the numbered clauses stop. Clause 7 is the last one, so without this
# boundary its statement runs to the end of the file and carries the
# change-history table and the document history.
ANNEX_HEADING: Final[re.Pattern[str]] = re.compile(r"^Annex\s+[A-Z]\s*:?\s*$")

# The only two curated rows whose section_id is a technique name rather than a
# clause number, mapped to the clause that name belongs to.
#
# Two entries and not twenty-four, on purpose. See the module docstring: two
# technique names each name two clauses, and an alternate title answers every
# link carrying that name, including the link that named the other clause.
#
# Both were verified twice against the pinned bytes. "Data sanitisation" occurs
# once in the whole document, inside clause 5.2.2, and its row targets CRE
# 041-188, which the 5.2.2 row also targets. "Retraining" occurs seven times
# across four clauses and is a bullet lead phrase exactly once, in clause 5.3.2,
# and its row targets CRE 854-183, which the 5.3.2 row also targets. [measured]
NAME_SECTION_IDS: Final[dict[str, str]] = {
    "Data sanitisation": "5.2.2",
    "Retraining": "5.3.2",
}

# Numbered clauses in sections 5 through 7 of the pinned document. [measured]
EXPECTED_CLAUSES: Final[int] = 25

# One below the length at which _sanitize_control takes full_text away from
# this parser, so a shipped description can never reach the limit even when the
# word-boundary search finds nothing to cut on. Ruling R14.
DESCRIPTION_BUDGET: Final[int] = DESCRIPTION_MAX_LENGTH - 1

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True)
class Clause:
    """One numbered clause, with the provenance of its statement recorded.

    `assembled` is carried rather than inferred. Deriving it downstream by
    testing whether a child's text sits inside the parent's would be a guess
    about text this class already knows the answer for, and a guess is what
    puts the wrong provenance on a statement.
    """

    heading: str
    # What the clause carried itself. Empty for a heading whose text lives
    # entirely in its subclauses.
    own_body: str
    # What ships. Equal to own_body except on a roll-up.
    body: str

    @property
    def assembled(self) -> bool:
        """Whether `body` was concatenated from this clause's descendants."""
        return self.body != self.own_body


class EtsiParser(BaseParser):
    framework_id: ClassVar[str] = "etsi"
    framework_name: ClassVar[str] = "ETSI"
    version: ClassVar[str] = "1.1.1"
    source_url: ClassVar[str] = (
        "https://www.etsi.org/deliver/etsi_gr/SAI/001_099/005/"
        "01.01.01_60/gr_SAI005v010101p.pdf"
    )
    mapping_unit_level: ClassVar[str] = "clause"
    expected_count: ClassVar[int] = EXPECTED_CLAUSES
    fetched_date: ClassVar[str] = "2026-08-15"
    # All 25 statements clear HONEST_PROSE_MIN_CHARS after roll-up and none
    # equals its own heading, so the attainable value is exactly 1.0.
    # [measured] The shortest statement runs 452 characters.
    min_prose_fraction: ClassVar[float] = 1.0

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
        audit_dir: Path | None = None,
    ) -> None:
        # Instance attributes rather than ClassVars. A fixture-backed test
        # declares its own gates by assignment, and assigning to a ClassVar
        # through an instance is both a mypy error and a mutation of every
        # other parser instance in the process.
        super().__init__(raw_dir, output_dir, audit_dir)
        self.expected_clauses: int = EXPECTED_CLAUSES
        self.expected_sha256: str | None = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        clauses = self.clauses_from_text(text)
        # COUNT_TOLERANCE is 10%, so BaseParser's band around 25 runs from 23
        # to 27 and a parser that lost two clauses would write in silence.
        # This is the structural check that beats the band.
        if len(clauses) != self.expected_clauses:
            raise ValueError(
                f"{self.framework_id}: {len(clauses)} numbered clause(s) in "
                f"sections 5 through 7, expected {self.expected_clauses}. The "
                f"count band around {self.expected_clauses} would accept a "
                f"loss of two without a word."
            )

        repairs: list[dict[str, object]] = []
        controls = self.build_controls(clauses, NAME_SECTION_IDS, repairs)
        self.write_repair_audit(repairs)

        assembled = sorted(number for number, c in clauses.items() if c.assembled)
        logger.info(
            "%s: %d clauses, %d assembled from their subclauses %s, "
            "%d description(s) capped, %d name-shaped section id(s) declared "
            "as alternate titles: %s",
            self.framework_id, len(controls), len(assembled), assembled,
            sum(1 for c in controls if c.full_text is not None),
            len(NAME_SECTION_IDS), sorted(NAME_SECTION_IDS),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not "
                f"the pinned {self.expected_sha256}. Both NAME_SECTION_IDS "
                f"entries were verified against this revision's clause "
                f"numbering, the {EXPECTED_CLAUSES}-clause count was measured "
                f"on these bytes, and RUNNING_HEADER was written against this "
                f"revision's page furniture."
            )

    @staticmethod
    def _is_page_furniture(line: str) -> bool:
        """Whether a line is the running header or the running footer."""
        return bool(RUNNING_HEADER.match(line) or RUNNING_FOOTER.match(line))

    @classmethod
    def _clause_lines(cls, text: str) -> list[str]:
        """The lines that can hold a clause, blank lines gone and annexes cut.

        Page furniture is left in place here. The heading guard removes it from
        the clause list and the body filter removes it from the clause text,
        and dropping it this early would retire both.

        Raises:
            ValueError: If no annex heading is found, which would leave clause
                7 carrying the change-history table as its statement.
        """
        kept = [line.strip() for line in text.split("\n") if line.strip()]
        for position, line in enumerate(kept):
            if ANNEX_HEADING.match(line):
                return kept[:position]
        raise ValueError(
            f"etsi: no annex heading in {SOURCE_FILE}. Clause 7 is the last "
            f"numbered clause, so without that boundary its statement runs to "
            f"the end of the document and carries the change-history table, "
            f"1,889 of the 2,776 characters it used to ship. One row of that "
            f"table also has the shape of a top-level clause heading."
        )

    @classmethod
    def clauses_from_text(cls, text: str) -> dict[str, Clause]:
        """Clause number -> Clause, descendants rolled up where needed.

        Raises:
            ValueError: If no annex heading bounds the clause range, or if a
                clause number matches on more than one line.
        """
        lines = cls._clause_lines(text)

        starts: list[tuple[int, str, str]] = []
        seen: dict[str, str] = {}
        furniture = 0
        for index, line in enumerate(lines):
            match = CLAUSE.match(line)
            if match is None:
                continue
            number, heading = match.group(1), match.group(2).strip()
            if DOCUMENT_IDENTIFIER.match(heading):
                # The running header on the page numbered 5, 6 or 7. Skipped
                # rather than raised on, because the document has exactly one
                # per page and every one of them is expected. A shape this
                # pattern does not know reaches the duplicate check below.
                furniture += 1
                logger.debug(
                    "etsi: line %d has the shape of clause %s and is the "
                    "running header, not a heading", index, number,
                )
                continue
            if number in seen:
                raise ValueError(
                    f"etsi: clause {number} matches on two different lines, "
                    f"under headings of {len(seen[number])} and "
                    f"{len(heading)} characters. The released parser dropped "
                    f"the second with a silent continue, which is how the page "
                    f"headers on pages 5, 6 and 7 took the three top-level "
                    f"clause numbers while the real headings were discarded. "
                    f"Teach the furniture filter the new shape rather than "
                    f"restoring the silence."
                )
            seen[number] = heading
            starts.append((index, number, heading))

        dropped = 0
        own: dict[str, tuple[str, str]] = {}
        for position, (start, number, heading) in enumerate(starts):
            end = (
                starts[position + 1][0] if position + 1 < len(starts)
                else len(lines)
            )
            segment = [
                line for line in lines[start + 1:end]
                if not cls._is_page_furniture(line)
            ]
            dropped += (end - start - 1) - len(segment)
            own[number] = (heading, _WHITESPACE.sub(" ", " ".join(segment)).strip())

        logger.info(
            "etsi: %d clause heading(s), %d running header(s) refused as a "
            "heading, %d furniture line(s) removed from clause bodies",
            len(starts), furniture, dropped,
        )

        clauses: dict[str, Clause] = {}
        for number, (heading, body) in own.items():
            if len(body) >= HONEST_PROSE_MIN_CHARS:
                clauses[number] = Clause(heading, body, body)
                continue
            # The separator is part of the prefix. Without it clause 5.1 would
            # adopt a clause 5.10, which is a sibling rather than a child.
            #
            # Only leaf descendants carry text, because a parent with a body of
            # its own never reaches this branch, so nothing is duplicated. The
            # clause's own stub leads, so a heading with a one-line lead keeps
            # it rather than losing it to the roll-up. All seven roll-ups in
            # the pinned document have no lead at all. [measured]
            children = sorted(key for key in own if key.startswith(f"{number}."))
            merged = " ".join(
                [body] + [own[key][1] for key in children]
            ).strip()
            clauses[number] = Clause(heading, body, merged or body)
        return clauses

    @classmethod
    def build_controls(
        cls,
        clauses: dict[str, Clause],
        alternates_by_name: dict[str, str],
        repairs: list[dict[str, object]] | None = None,
    ) -> list[Control]:
        """One Control per clause, name-shaped section ids as alternate titles.

        `repairs` is an optional sink. Each roll-up and each capped description
        appends a before and after pair to it, which parse() hands to
        write_repair_audit. A count says a repair fired. Only the pair says
        what moved.

        Raises:
            ValueError: If a declared name points at a clause this parse did
                not produce, or a clause has no text at all after roll-up.
        """
        alternates: dict[str, list[str]] = {}
        for name, clause_id in sorted(alternates_by_name.items()):
            alternates.setdefault(clause_id, []).append(name)

        missing = sorted(set(alternates) - set(clauses))
        if missing:
            raise ValueError(
                f"etsi: NAME_SECTION_IDS points at clause(s) {missing} that "
                f"this parse did not produce. Two curated links carry a "
                f"technique name where a clause number belongs and reach their "
                f"clause only through this map, so a stale entry leaves them "
                f"resolving to nothing while the parser still writes."
            )

        controls: list[Control] = []
        for number in sorted(clauses, key=lambda n: [int(p) for p in n.split(".")]):
            clause = clauses[number]
            if not clause.body:
                raise ValueError(
                    f"etsi: clause {number} has no text of its own and no "
                    f"subclause to roll up. An empty statement is excluded "
                    f"from the prose index without any gate saying so."
                )
            metadata: dict[str, str | list[str]] = {}
            if clause.assembled:
                metadata[TEXT_ORIGIN_METADATA_KEY] = SYNTHETIC_TEXT_ORIGIN
                if repairs is not None:
                    repairs.append({
                        "control_id": number,
                        "repair": "statement_assembled_from_subclauses",
                        "field": "description",
                        "before": clause.own_body,
                        "after": clause.body,
                    })
            if number in alternates:
                metadata["alt_titles"] = alternates[number]

            statement, overflow = cls._cap_description(clause.body)
            if overflow is not None and repairs is not None:
                repairs.append({
                    "control_id": number,
                    "repair": "description_capped_to_protect_full_text",
                    "field": "description",
                    "before": clause.body,
                    "after": statement,
                })
            controls.append(Control(
                control_id=number,
                title=clause.heading,
                description=statement,
                full_text=overflow,
                hierarchy_level="clause",
                parent_id=number.rsplit(".", 1)[0] if "." in number else None,
                metadata=metadata or None,
            ))
        return controls

    @staticmethod
    def _cap_description(text: str) -> tuple[str, str | None]:
        """Split a statement into a capped description and its full form.

        Returns (description, full_text_or_None). full_text is None when the
        statement already fits, so a caller can tell a cut from a pass-through
        without comparing lengths.

        The budget is one character under DESCRIPTION_MAX_LENGTH, the length at
        which _sanitize_control replaces this parser's full_text with the
        sanitised description. Sanitisation only shrinks this document: its
        extracted text carries no ligature, no HTML entity, no zero-width
        character and no combining sequence NFC would recompose, so a raw cut
        at the budget lands under the limit. [measured] The test suite asserts
        the shipped result rather than trusting that.
        """
        if len(text) <= DESCRIPTION_BUDGET:
            return text, None
        cut = text[:DESCRIPTION_BUDGET].rsplit(" ", 1)[0]
        # Mirrors sanitize_text's own guard. A statement with no space in its
        # first half would otherwise be cut to almost nothing.
        if len(cut) < DESCRIPTION_BUDGET // 2:
            cut = text[:DESCRIPTION_BUDGET]
        return cut, text


def main() -> None:
    EtsiParser().run()


if __name__ == "__main__":
    main()
