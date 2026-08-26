"""Parser for ENISA's Securing Machine Learning Algorithms (December 2021).

There is no control identifier anywhere in this source, which is why OpenCRE's
own extraction degraded: 40 of the 68 curated links carry the literal string
"Table 5:" as their section_id and 18 carry "Table 3:". The join is therefore
the control NAME, and the name is what this parser has to get exactly right.

Two tables are read. Table 5 (pages 20 to 26 as printed, 19 to 25 zero-indexed)
gives 37 security controls with their definitions, which is the count the
document states in its own text. Table 3 (pages 15 to 16, 14 to 15
zero-indexed) gives 13 threats and sub-threats, and they are emitted too: 20 of
the 68 curated links target them, including Poisoning and Evasion, the two
most-linked entries in the framework. Annex C is NOT read. Every one of the 33
distinct curated names resolves against these two tables, so an Annex C pass
would add near-duplicate spellings of controls that already exist. [measured]

Extraction is per row, not per page. pdfplumber puts a definition in column 2
on some rows and column 3 on others, so a per-page densest-column heuristic
loses whichever rows sit in the other column. It also puts the NAME in column 0
on most rows and column 1 on others: Table 3 prints every sub-threat there, and
Table 5 prints two of its controls ("Ensure reliable sources are used" and "Use
methods to clean the training dataset from suspicious samples", six curated
links between them) plus all three of its category banners. So the name is the
first NAME_COLUMNS cells joined, the definition is columns NAME_COLUMNS through
DEFINITION_END_COLUMN joined with lone lifecycle "x" marks dropped, and a row
with no name is a continuation of the unit above it. Under that rule no unit
extracts with an empty definition. Read with one name column instead, the two
column-1 controls and the three banners all merge into the definition of
whichever control precedes them.

`banners` has an empty default and every caller passes one. Left at the
default, the table's own header row becomes a unit named "Security controls".
The default is kept empty rather than filled so a caller that forgets shows up
as an extra unit that _check_shape refuses by count, rather than as a silently
different filter.

Names are repaired before they are stored, not at lookup time, because
ProseIndex matches a link's section_name against the stored title verbatim. Two
defects are corrected, and both are recorded in the repair audit with the
before and after TEXT rather than as a count:

    footnote reference fused onto the last word    5 titles, 8 curated links
    typographic punctuation against ASCII          2 titles, 3 curated links

Naive matching resolves 57 of 68, the footnote table alone takes it to 65, the
punctuation fold alone takes it to 60, and both take it to 68. [measured]

The footnote names are DECLARED rather than found by a trailing-digit regex, so
a control name that legitimately ends in a digit cannot be damaged, and a
source refresh that moves the footnotes fails this parser instead of quietly
renaming a control.

Nothing here is assembled out of fragments. Every statement is the publisher's
own definition column, so no control carries a synthetic text origin.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from io import BytesIO
from typing import Any, ClassVar, Final

import pdfplumber

from tract.config import DESCRIPTION_MAX_LENGTH
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "enisa_securing_ml_algorithms.pdf"
SOURCE_SHA256: Final[str] = (
    "4de967bbdf92a01339ae449b7d305b8ff266d7f16ed0a7d92a711ca20e20f087"
)

# 0-indexed page ranges, verified against the pinned PDF. Table 4 begins on
# page 16 and Annex C on page 38, so both ranges stop short of them.
TABLE3_PAGES: Final[range] = range(14, 16)
TABLE5_PAGES: Final[range] = range(19, 26)

# The name spans the first two cells. Table 3 prints threats in column 0 and
# sub-threats in column 1; Table 5 prints most controls in column 0, two in
# column 1, and its category banners in column 1.
NAME_COLUMNS: Final[int] = 2

# Everything from NAME_COLUMNS to here is definition text. Four, not five.
# Across both tables the definition lands at column 2 on 167 cells and column 3
# on 211, and never past column 3. Column 4 carries four cells and none of them
# is a definition: three are Table 5's "Stages of the lifecycle" header, which
# the banner rule already drops, and the fourth is the rotated lifecycle header
# pdfplumber returns as reversed character runs on Table 3's second page. That
# one sits on a nameless row, so a span reaching column 4 appends it to the
# last unit of the page before, which corrupted "Model or data disclosure" with
# a trailing "a ta D". [measured]
#
# The stage columns still start at index 3 on Table 3's second page, inside
# this span, so the lone-"x" rule below stays load-bearing.
DEFINITION_END_COLUMN: Final[int] = 4

# Row names that are banners rather than units. Matched on a prefix because the
# header row reads "Security controls" on one page layout and "Security
# controls Definition" on the other. A future control name that opened with one
# of these would be dropped, which the unit-count gate in _check_shape refuses.
TABLE5_BANNERS: Final[tuple[str, ...]] = (
    "Security controls", "ORGANISATIONAL", "TECHNICAL", "SPECIFIC ML",
)
TABLE3_BANNERS: Final[tuple[str, ...]] = ("Threats",)

# Table 5 names carrying a footnote reference fused onto the last word.
# Declared, hand-verified against the pinned PDF, and checked at parse time by
# _check_declarations. A blanket trailing-digit strip would also damage a name
# that legitimately ends in a number.
FOOTNOTE_NAMES: Final[dict[str, str]] = {
    "Include ML applications into detection and response to security incident "
    "processes15":
        "Include ML applications into detection and response to security "
        "incident processes",
    "Add some adversarial examples to the training dataset16":
        "Add some adversarial examples to the training dataset",
    "Apply modifications on inputs17":
        "Apply modifications on inputs",
    "Reduce the information given by the model19":
        "Reduce the information given by the model",
    "Use less easily transferable models20":
        "Use less easily transferable models",
}

# The source's typographic forms against the ASCII OpenCRE stores. NFKD already
# resolves the ellipsis, and it leaves the quotes and dashes alone, so this map
# is what closes the remaining gap.
_PUNCTUATION: Final[dict[str, str]] = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_SLUG_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

# Metadata key carrying the source's own spelling of a repaired name, so the
# artifact states what was changed without a reader having to invert the
# repair. Present only where the stored title differs from the printed one.
SOURCE_NAME_METADATA_KEY: Final[str] = "source_name"

FOOTNOTE_REPAIR: Final[str] = "footnote_reference_removed"
PUNCTUATION_REPAIR: Final[str] = "punctuation_folded_to_ascii"


class EnisaParser(BaseParser):
    framework_id: ClassVar[str] = "enisa"
    # Matches the curated links' standard_name exactly, and FRAMEWORK_NAME_
    # ALIASES maps it to the framework_id, so the join needs no alias work.
    framework_name: ClassVar[str] = "ENISA"
    version: ClassVar[str] = "2021-12"
    source_url: ClassVar[str] = (
        "https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms"
    )
    mapping_unit_level: ClassVar[str] = "control"
    # 37 Table 5 controls, which is the number the document states in its own
    # text, plus 13 Table 3 threats. [measured]
    expected_count: ClassVar[int] = 50
    # COUNT_TOLERANCE is 10%, so the band around 50 runs from 45 to 55 and a
    # parser that lost five units would write in silence. These two are the
    # structural check that beats the band. Overridable on an instance so a
    # synthetic PDF can drive parse() in CI. [measured]
    expected_table5_units: ClassVar[int] = 37
    expected_table3_units: ClassVar[int] = 13
    fetched_date: ClassVar[str] = "2026-08-15"
    # All 50 statements clear HONEST_PROSE_MIN_CHARS, the shortest definition
    # is 80 characters, and none equals its own name, so the measured value is
    # exactly 1.0. A floor below it would let a unit decay to a bare title
    # without firing. The attainable range is 0.0 to 1.0 and the trigger sits
    # at the measured maximum, so any degradation fails. [measured end to end
    # on the pinned PDF]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            table5 = self._collect(pdf, TABLE5_PAGES, TABLE5_BANNERS)
            table3 = self._collect(pdf, TABLE3_PAGES, TABLE3_BANNERS)

        self._check_shape(table5, table3)
        self._check_declarations(table5)
        controls = self._build(table5, table3)
        self._check_description_budget(controls)
        self._check_unique_ids(controls)
        self.write_repair_audit(self.repair_records(controls))
        logger.info(
            "%s: %d controls and %d threats, %d repaired name(s)",
            self.framework_id,
            sum(1 for c in controls if c.hierarchy_level == "control"),
            sum(1 for c in controls if c.hierarchy_level == "threat"),
            sum(
                1 for c in controls
                if SOURCE_NAME_METADATA_KEY in (c.metadata or {})
            ),
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
                f"the pinned {self.expected_sha256}. FOOTNOTE_NAMES and both "
                f"page ranges quote this exact document."
            )

    def _check_shape(
        self, table5: list[tuple[str, str]], table3: list[tuple[str, str]],
    ) -> None:
        """Refuse a parse whose table sizes moved, before the band hides it.

        Raises:
            ValueError: If either table yields a different number of units
                than this parser declares.
        """
        if len(table5) != self.expected_table5_units:
            raise ValueError(
                f"{self.framework_id}: Table 5 yielded {len(table5)} unit(s), "
                f"expected {self.expected_table5_units}. COUNT_TOLERANCE puts "
                f"the band around 50 at 45 to 55, so a loss of five controls "
                f"would write without a word. Names extracted: "
                f"{sorted(name for name, _ in table5)}"
            )
        if len(table3) != self.expected_table3_units:
            raise ValueError(
                f"{self.framework_id}: Table 3 yielded {len(table3)} unit(s), "
                f"expected {self.expected_table3_units}. Twenty of the 68 "
                f"curated links target a Table 3 entry. Names extracted: "
                f"{sorted(name for name, _ in table3)}"
            )

    def _collect(
        self, pdf: Any, pages: range, banners: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        """(name, definition) for one table, across its pages.

        `pdf` is typed Any rather than pdfplumber.PDF. pyproject silences
        pdfplumber with ignore_missing_imports, which makes the module Any, so
        the attribute annotation raises `Name "pdfplumber.PDF" is not defined`
        under mypy --strict. [measured]

        Raises:
            ValueError: If no page in the range yields a table wide enough to
                hold a definition column.
        """
        rows: list[list[str | None]] = []
        for page_number in pages:
            if page_number >= len(pdf.pages):
                break
            tables = pdf.pages[page_number].extract_tables()
            if not tables:
                continue
            # The page carries several tables and the one that matters is the
            # widest: the others are one-cell artifacts of the rotated
            # lifecycle headers.
            widest = max(tables, key=lambda table: max(len(r) for r in table))
            if max(len(row) for row in widest) < DEFINITION_END_COLUMN:
                continue
            rows.extend(widest)
        units = self.rows_to_units(rows, NAME_COLUMNS, banners)
        if not units:
            raise ValueError(
                f"{self.framework_id}: no table rows on pages "
                f"{pages.start} to {pages.stop} of {SOURCE_FILE}. "
                f"extract_text() interleaves these tables and returns the "
                f"rotated lifecycle headers as reversed character runs, so "
                f"falling back to it would produce garbage rather than a "
                f"smaller result."
            )
        return units

    @classmethod
    def rows_to_units(
        cls,
        rows: list[list[str | None]],
        name_columns: int,
        banners: tuple[str, ...] = (),
    ) -> list[tuple[str, str]]:
        """(name, definition) per named row, continuations merged upward.

        Merging is per row rather than per page because pdfplumber places a
        definition in column 2 on some rows and column 3 on others.

        `banners` defaults to empty and every caller in this module passes one.
        A caller that forgets gets the table's header row back as a unit, which
        _check_shape then refuses on the count.
        """
        units: list[tuple[str, list[str]]] = []
        for row in rows:
            cells = [(cell or "").strip() for cell in row]
            padded = cells + [""] * DEFINITION_END_COLUMN
            name = _WHITESPACE.sub(
                " ", " ".join(c for c in padded[:name_columns] if c),
            ).strip()
            body = _WHITESPACE.sub(
                " ",
                " ".join(
                    c for c in padded[name_columns:DEFINITION_END_COLUMN]
                    # A lone "x" is a lifecycle mark from a stage column that
                    # starts before DEFINITION_END_COLUMN, not definition text.
                    if c and c.lower() != "x"
                ),
            ).strip()
            if name and any(name.startswith(banner) for banner in banners):
                continue
            if not name:
                if body and units:
                    units[-1][1].append(body)
                continue
            units.append((name, [body] if body else []))
        return [(name, " ".join(parts).strip()) for name, parts in units]

    @staticmethod
    def normalise_name(name: str) -> str:
        """The comparison key for a control or threat name.

        NFKD folds compatibility forms including the ellipsis, and the
        punctuation map turns the source's typographic quotes and dashes into
        the ASCII OpenCRE stores. A declared footnote reference is removed by
        clean() before this is called, not by a trailing-digit regex here.
        """
        folded = unicodedata.normalize("NFKD", name)
        for source, target in _PUNCTUATION.items():
            folded = folded.replace(source, target)
        return _WHITESPACE.sub(" ", folded).strip().lower()

    @staticmethod
    def clean(name: str) -> str:
        """A stored title: footnote reference removed, punctuation as ASCII."""
        stripped = FOOTNOTE_NAMES.get(name, name)
        folded = unicodedata.normalize("NFKD", stripped)
        for source, target in _PUNCTUATION.items():
            folded = folded.replace(source, target)
        return _WHITESPACE.sub(" ", folded).strip()

    def _check_declarations(self, table5: list[tuple[str, str]]) -> None:
        """Refuse declarations that no longer match the extracted table.

        Raises:
            ValueError: If a declared footnote name is absent from this parse.
        """
        extracted = {name for name, _ in table5}
        missing = sorted(set(FOOTNOTE_NAMES) - extracted)
        if missing:
            raise ValueError(
                f"{self.framework_id}: FOOTNOTE_NAMES declares {missing}, "
                f"which this parse did not produce. A stale entry means a "
                f"control name still carries a footnote digit, and every "
                f"curated link to it falls back to its own title."
            )

    @classmethod
    def _build(
        cls, table5: list[tuple[str, str]], table3: list[tuple[str, str]],
    ) -> list[Control]:
        """Controls from Table 5, threats from Table 3."""
        controls: list[Control] = []
        for table, level in ((table5, "control"), (table3, "threat")):
            for name, definition in table:
                title = cls.clean(name)
                metadata: dict[str, str | list[str]] = {
                    "table": "Table 5" if level == "control" else "Table 3",
                }
                if title != name:
                    metadata[SOURCE_NAME_METADATA_KEY] = name
                controls.append(Control(
                    control_id=cls.slug(title),
                    title=title,
                    description=definition,
                    hierarchy_level=level,
                    metadata=metadata,
                ))
        return controls

    @staticmethod
    def slug(title: str) -> str:
        """A synthetic control id.

        The source has no identifier of any kind, so this is generated. It is
        derived from the cleaned title, so it is stable across re-parses of the
        same bytes, and it is never used for the join: every curated link
        carries either "Table 5:" or "Table 3:" or the name itself. Not
        truncated, because two of these titles run past 90 characters and a cut
        would turn a near-duplicate name into a silent id collision.
        """
        return _SLUG_SEPARATOR.sub("-", title.lower()).strip("-")

    @staticmethod
    def _check_unique_ids(controls: list[Control]) -> None:
        """Refuse two units that generate the same slug.

        The id is derived from the name, and the source has 50 names that
        differ, so a collision means two units collapsed into one row of the
        artifact.

        Raises:
            ValueError: If any control id appears twice.
        """
        seen: dict[str, str] = {}
        for control in controls:
            if control.control_id in seen:
                raise ValueError(
                    f"enisa: {control.title!r} and {seen[control.control_id]!r} "
                    f"both generate the control id {control.control_id!r}. "
                    f"The id is derived from the name, so a collision means "
                    f"two units would occupy one row of the artifact."
                )
            seen[control.control_id] = control.title

    @staticmethod
    def _check_description_budget(controls: list[Control]) -> None:
        """Refuse a statement BaseParser._sanitize_control would rewrite.

        Ruling R14: that function truncates `description` past
        DESCRIPTION_MAX_LENGTH and writes the untruncated text into
        `full_text`, discarding whatever the parser put there. It does not fire
        on the pinned PDF, whose longest definition is 709 characters.
        [measured] This raises rather than pre-capping because these
        definitions are two to six sentences, so three times the measured
        maximum means the source or the column span moved, and silently
        shortening it would hide that.

        Raises:
            ValueError: If any statement reaches DESCRIPTION_MAX_LENGTH.
        """
        for control in controls:
            if len(control.description) < DESCRIPTION_MAX_LENGTH:
                continue
            raise ValueError(
                f"enisa: {control.control_id} has a statement of "
                f"{len(control.description)} characters, at or over the "
                f"{DESCRIPTION_MAX_LENGTH}-character limit where "
                f"BaseParser._sanitize_control truncates the description and "
                f"overwrites full_text with the untruncated text. The pinned "
                f"PDF's longest definition is 709 characters, so this means "
                f"the source or the column span moved."
            )

    @staticmethod
    def repair_records(controls: list[Control]) -> list[dict[str, str]]:
        """Before and after TEXT for every name this parser repaired.

        A count says a repair fired. It does not say what moved, and a title
        the join depends on is exactly the field a reviewer has to be able to
        check by eye. Sorted by control id so re-parsing the same bytes
        produces the same audit bytes.
        """
        records: list[dict[str, str]] = []
        for control in controls:
            source_name = (control.metadata or {}).get(
                SOURCE_NAME_METADATA_KEY
            )
            if not isinstance(source_name, str):
                continue
            records.append({
                "control_id": control.control_id,
                "repair": (
                    FOOTNOTE_REPAIR if source_name in FOOTNOTE_NAMES
                    else PUNCTUATION_REPAIR
                ),
                "before": source_name,
                "after": control.title,
            })
        return sorted(records, key=lambda record: record["control_id"])


def main() -> None:
    EnisaParser().run()


if __name__ == "__main__":
    main()
