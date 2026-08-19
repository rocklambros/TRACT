"""Parser for NIST SP 800-218, the Secure Software Development Framework.

The tasks live in one ruled table spanning printed pages 14 through 27 of the
PDF (0-indexed 13 through 26). Measured against the pinned bytes with
pdfplumber 0.11.4, `pdfplumber.extract_tables()` returns 47 task cells at
column index 3, all whole: the task statement arrives complete, with its own
newlines inside one cell. The rows below a task repeat wrapped fragments of the
PRACTICE cell in column 0 and hold nothing in column 3. So the practice column
needs a forward fill and nothing needs a rowspan merge. `extract_text()` is the
call that interleaves this table's columns; `extract_tables()` does not.
[measured 2026-08-19]

Seven pages carry a truncated second copy of a task at column index 4, for
PO.2.3, PS.2.1, PW.2.1, PW.4.4, PW.6.1, PW.9.1 and RV.2.2, each cut mid-phrase.
This parser reads column 3 only, so those copies are invisible to it, and a
premise check that scans every column sees 54 cells over columns [3, 4] with
seven duplicate task ids rather than 47 with none. [measured 2026-08-19]

Five task cells are redirects of the form "PW.3.1: Moved to PO.1.3". None is
targeted by a curated link. [measured] They are recorded as `retired_tasks`
metadata on the framework's first control and excluded from the emitted set: a
15-character statement is not a control, and emitting one would put a
non-statement anchor in the corpus and drag the prose floor for no join. That
leaves 42 real tasks over 19 practices, with none left parentless after the
forward fill. [measured]

The title is the task ID and this is not cosmetic. OpenCRE sets `section_name`
to a task statement for all 46 curated links, so a parser that used the
statement as its title would make description equal title, which is exactly the
case `ProseIndex` refuses to index. Using the parent practice name instead
costs five links, because five statements are shorter than their practice name
plus PROSE_MIN_EXTRA_CHARS. [measured]

The Notional Implementation Examples column does NOT reach the anchor, and this
departs from the task brief, which put it in `full_text`. `ProseIndex` prefers
`full_text` unconditionally, so storing the examples there makes them the
anchor for the 35 links that land on a task carrying them, and the examples run
up to 1,176 characters against a 333-character statement. [measured] They say
how an organisation might satisfy the task, which is the same kind of text
REMEDIATION_HEADINGS exists to cut and the same kind ruling R17 cut out of the
WSTG anchor. Cutting a column is the same decision as cutting a heading. The
examples are kept verbatim under `metadata["notional_examples"]`, where a
reviewer can read them and no encoder ever will, so nothing is discarded and
the anchor is the publisher's own task statement, unassembled. Nothing here is
therefore `text_origin: synthetic`: every stored statement is one source cell.

Two curated links carry a mid-sentence text fragment in `section_id` instead of
a task id. Both fragments appear verbatim inside a task statement, so they are
declared here as alternate ids and resolved through the `alt_ids` channel. They
are declared rather than derived: a substring search that ran at parse time
would silently re-attach a link to a different task after a source refresh.
Both strings are quoted from `data/training/hub_links_curated.jsonl` character
for character, en dashes included, and
`tests/test_parse_nist_ssdf.py::TestMalformedIdMap` re-derives the table from
that tracked file so a declared entry nothing spells and a malformed id nobody
declared both fail.

Ruling R14 was checked and does NOT fire here. The longest task statement is
333 characters against a DESCRIPTION_MAX_LENGTH of 2,000, so
`BaseParser._sanitize_control` never truncates a description or overwrites
`full_text` behind this parser. `_check_description_budget` raises rather than
pre-capping, matching ruling R14's Task 8 form: a single-sentence task
statement six times its measured maximum means the source or the column choice
moved, and silently shortening it would hide that.

Ruling R13 was checked and nothing is stripped. The shared leading prefix
across all 42 prepared anchors is 0 characters, because each statement opens on
its own verb. [measured]

The join reports `wrong_anchor_risk` 44 of 44 applicable, and every one of
those is a fact about the link file rather than a wrong anchor. Detector B
compares a link's `section_name` against the title of the control its id
reached. Here the name is the task STATEMENT and the title is the task ID, so
the detector holds a 160-character sentence against "PO.1.1" and can only fire.
`section_name` equals the resolved control's title for 0 of 46 links, which is
the same reading that retired detector B for dsomm under ruling R11.
`coarse_name_frameworks()` cannot derive it, because that criterion is
distinct(section_id) over distinct(section_name) and both are 44 here. The
exemption set is therefore left alone and the count is asserted in
tests/test_parse_nist_ssdf.py rather than silenced. Detectors A and C read 0.
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.config import DESCRIPTION_MAX_LENGTH
from tract.parsers.base import BaseParser
from tract.sanitize import sanitize_text
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "nist_sp_800_218.pdf"
SOURCE_SHA256: Final[str] = (
    "617746e553a9e2da49bfbd4eef0dfc3094758a39b869314e4173ac36605cde22"
)
# 0-indexed. The table starts on printed page 14 and ends on 27, and the range
# runs one page past it so a reprint that adds a page is still read.
TABLE_PAGES: Final[range] = range(13, 28)

PRACTICE_COLUMN: Final[int] = 0
TASK_COLUMN: Final[int] = 3
EXAMPLES_COLUMN: Final[int] = 6
# The source table is twelve columns wide, most of them the empty slivers
# between the four that carry text. Rows are padded to this before indexing so
# a short row cannot raise IndexError on a column that simply held nothing.
ROW_WIDTH: Final[int] = 12

# No re.S on either: every cell is whitespace-collapsed before it is matched,
# so no newline survives to need it, and a `.` that spanned lines would let a
# practice header swallow the task text below it.
TASK_ID: Final[re.Pattern[str]] = re.compile(r"^((?:P[OSW]|RV)\.\d+\.\d+):\s*(.+)$")
PRACTICE_ID: Final[re.Pattern[str]] = re.compile(
    r"^(.+?)\s*\(((?:P[OSW]|RV)\.\d+)\)\s*:\s*(.*)$"
)
REDIRECT: Final[re.Pattern[str]] = re.compile(r"^moved to\b", re.IGNORECASE)

# OpenCRE section_id values that are a sentence fragment rather than a task id,
# mapped to the task whose statement contains that fragment verbatim.
# Hand-verified against the pinned PDF and against the curated link file, and
# never derived at parse time. parse() checks both halves of each entry: the
# task exists, and the fragment is still inside its statement.
MALFORMED_SECTION_IDS: Final[dict[str, str]] = {
    "code, executable code, and configuration-as-code – based on the principle "
    "of least privilege so that only authorized personnel, tools, services, "
    "etc. have access.": "PS.1.1",
    "should be performed to find vulnerabilities not identified by previous "
    "reviews, analysis, or testing and, if so, which types of testing should "
    "be used.": "PW.8.1",
}

# The length at which BaseParser._sanitize_control truncates `description` and
# writes the untruncated text into `full_text`. The guard refuses at the limit
# rather than near it, so there is no second threshold to keep in step.
_DESCRIPTION_LIMIT: Final[int] = DESCRIPTION_MAX_LENGTH
# Generous, and it is a storage bound rather than an anchor bound: the examples
# never reach the encoder. The longest measured is 1,176 characters.
_EXAMPLES_MAX_CHARS: Final[int] = 20_000

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class NistSsdfParser(BaseParser):
    framework_id: ClassVar[str] = "nist_ssdf"
    # Matches the curated links' standard_name exactly, so no alias is needed.
    framework_name: ClassVar[str] = "NIST SSDF"
    version: ClassVar[str] = "1.1"
    source_url: ClassVar[str] = (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf"
    )
    mapping_unit_level: ClassVar[str] = "task"
    # 47 task cells minus 5 "Moved to" redirects. [measured]
    expected_count: ClassVar[int] = 42
    # COUNT_TOLERANCE is 10%, so the band around 42 is 38 to 46 and a parser
    # that lost four tasks would write without a word. These three are the
    # structural check that beats the band. Class-level so a synthetic PDF can
    # drive parse() in CI with its own shape instead of the real gate being
    # widened to accept two documents. [measured]
    expected_task_cells: ClassVar[int] = 47
    expected_redirects: ClassVar[int] = 5
    expected_practices: ClassVar[int] = 19
    fetched_date: ClassVar[str] = "2026-08-15"
    # 41 of 42 statements clear the 60-character bar; the shortest real task
    # statement is 54 characters, giving 0.9762. [measured]
    min_prose_fraction: ClassVar[float] = 0.97
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        rows = self._read_rows(payload)
        controls = self.rows_to_controls(rows, require_alternate_targets=True)
        self._check_shape(rows, controls)
        self._check_description_budget(controls)
        self.write_repair_audit(self.repair_records(rows))
        logger.info(
            "%s: %d tasks across %d practices",
            self.framework_id, len(controls),
            len({c.parent_id for c in controls}),
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
                f"the pinned {self.expected_sha256}. Both entries in "
                f"MALFORMED_SECTION_IDS quote this document's text verbatim, "
                f"so a changed source can attach a link to the wrong task."
            )

    def _check_shape(
        self, rows: list[list[str | None]], controls: list[Control],
    ) -> None:
        """Refuse a parse whose table shape moved, before the band hides it.

        Raises:
            ValueError: If the task-cell, redirect or practice count differs
                from the declared shape, or a task has no practice.
        """
        cells = sum(1 for _ in self._task_cells(rows))
        if cells != self.expected_task_cells:
            raise ValueError(
                f"{self.framework_id}: {cells} task cell(s) at column "
                f"{TASK_COLUMN}, expected {self.expected_task_cells}. "
                f"COUNT_TOLERANCE puts the band around 42 emitted tasks at 38 "
                f"to 46, so a loss of four would write in silence."
            )
        redirects = cells - len(controls)
        if redirects != self.expected_redirects:
            raise ValueError(
                f"{self.framework_id}: {redirects} redirect stub(s), expected "
                f"{self.expected_redirects}. A stub that stopped being "
                f"recognised becomes a 15-character anchor in the corpus."
            )
        practices = len({c.parent_id for c in controls})
        if practices != self.expected_practices:
            raise ValueError(
                f"{self.framework_id}: {practices} practice(s), expected "
                f"{self.expected_practices}. The practice is forward-filled "
                f"from the first task row of each group, so a wrong count "
                f"means tasks are attached to the wrong practice."
            )
        empty = [c.control_id for c in controls if not c.parent_id]
        if empty:
            raise ValueError(
                f"{self.framework_id}: task(s) {empty} have no practice after "
                f"the forward fill. The fill is the only thing that gives a "
                f"task its parent, so an empty one means the practice cell "
                f"moved out of column {PRACTICE_COLUMN}."
            )

    @staticmethod
    def _check_description_budget(controls: list[Control]) -> None:
        """Refuse a statement BaseParser._sanitize_control would rewrite.

        Ruling R14: that function truncates `description` past
        DESCRIPTION_MAX_LENGTH and writes the untruncated text into
        `full_text`, discarding whatever the parser put there. It does not fire
        on the pinned PDF, whose longest task statement is 333 characters.
        [measured] This raises rather than pre-capping because an SSDF task
        statement is one sentence, so six times its measured maximum means the
        source or the column choice moved, and silently shortening it would
        hide that.

        Raises:
            ValueError: If any statement reaches DESCRIPTION_MAX_LENGTH.
        """
        for control in controls:
            if len(control.description) < _DESCRIPTION_LIMIT:
                continue
            raise ValueError(
                f"nist_ssdf: {control.control_id} has a statement of "
                f"{len(control.description)} characters, at or over the "
                f"{_DESCRIPTION_LIMIT}-character limit where "
                f"BaseParser._sanitize_control truncates the description and "
                f"overwrites full_text with the untruncated text. The pinned "
                f"PDF's longest task statement is 333 characters, so this "
                f"means the source or the column choice moved."
            )

    def _read_rows(self, payload: bytes) -> list[list[str | None]]:
        """Every row of each table at least four columns wide, in page order.

        Raises:
            ValueError: If no page yields a table at least four columns wide.
        """
        rows: list[list[str | None]] = []
        with pdfplumber.open(BytesIO(payload)) as pdf:
            for page_number in TABLE_PAGES:
                if page_number >= len(pdf.pages):
                    break
                for table in pdf.pages[page_number].extract_tables():
                    # An empty table would make max() raise, and a narrow one
                    # is the page furniture rather than the task table.
                    if not table or max(len(row) for row in table) < 4:
                        continue
                    rows.extend(table)
        if not rows:
            raise ValueError(
                f"{self.framework_id}: no table of four or more columns on "
                f"pages {TABLE_PAGES.start}-{TABLE_PAGES.stop} of "
                f"{SOURCE_FILE}. extract_text() interleaves this table's "
                f"columns, so falling back to it would ship task statements "
                f"truncated by the adjacent Examples column."
            )
        return rows

    @staticmethod
    def _cells(row: list[str | None]) -> list[str]:
        """One row padded to ROW_WIDTH, each cell whitespace-collapsed."""
        return [
            _WHITESPACE.sub(" ", str(cell).strip()) if cell else ""
            for cell in (list(row) + [None] * ROW_WIDTH)[:ROW_WIDTH]
        ]

    @classmethod
    def _task_cells(
        cls, rows: list[list[str | None]],
    ) -> list[tuple[str, str]]:
        """(task id, statement) for every task cell at TASK_COLUMN.

        Redirect stubs included: this is the shape check's denominator, and it
        has to count what the table holds rather than what was emitted.
        """
        found: list[tuple[str, str]] = []
        for row in rows:
            match = TASK_ID.match(cls._cells(row)[TASK_COLUMN])
            if match is not None:
                found.append((match.group(1), match.group(2).strip()))
        return found

    @classmethod
    def _walk(
        cls, rows: list[list[str | None]],
    ) -> tuple[list[Control], dict[str, str]]:
        """One Control per real task, plus the redirect stubs that were not.

        The practice is forward-filled from the header row of each group,
        because the continuation rows under a task repeat wrapped fragments of
        the practice cell and carry nothing in the task column.
        """
        practice_id = ""
        practice_name = ""
        controls: list[Control] = []
        redirects: dict[str, str] = {}

        for row in rows:
            cells = cls._cells(row)
            practice = PRACTICE_ID.match(cells[PRACTICE_COLUMN])
            if practice is not None:
                practice_name = practice.group(1).strip()
                practice_id = practice.group(2)

            task = TASK_ID.match(cells[TASK_COLUMN])
            if task is None:
                continue
            task_id, statement = task.group(1), task.group(2).strip()
            if REDIRECT.match(statement):
                redirects[task_id] = statement
                continue

            metadata: dict[str, str | list[str]] = {"practice": practice_id}
            examples = cells[EXAMPLES_COLUMN]
            if examples:
                # Stored, not anchored. See the module docstring: the examples
                # are remediation guidance, and full_text would make them the
                # anchor. Sanitised here because run() sanitises the schema's
                # text fields and not metadata.
                metadata["notional_examples"] = sanitize_text(
                    examples, max_length=_EXAMPLES_MAX_CHARS,
                )
            alternates = sorted(
                fragment for fragment, target in MALFORMED_SECTION_IDS.items()
                if target == task_id
            )
            if alternates:
                metadata["alt_ids"] = alternates

            controls.append(Control(
                control_id=task_id,
                title=task_id,
                description=statement,
                hierarchy_level="task",
                parent_id=practice_id or None,
                parent_name=practice_name or None,
                metadata=metadata,
            ))

        return controls, redirects

    @classmethod
    def rows_to_controls(
        cls,
        rows: list[list[str | None]],
        require_alternate_targets: bool = False,
    ) -> list[Control]:
        """One Control per real task, practice forward-filled.

        Raises:
            ValueError: If require_alternate_targets is set and a declared
                malformed-id fragment names a task this parse did not produce,
                or no longer sits inside that task's statement.
        """
        controls, redirects = cls._walk(rows)
        cls._check_alternate_ids(controls, require_alternate_targets)
        if redirects and controls:
            cls._record_redirects(controls[0], redirects)
        logger.info("nist_ssdf: %d redirect stub(s) excluded: %s",
                    len(redirects), sorted(redirects))
        return controls

    @classmethod
    def repair_records(
        cls, rows: list[list[str | None]],
    ) -> list[dict[str, object]]:
        """Before and after, as text, for both things this parser moves.

        A count says a repair fired. It does not say what text a link now
        trains on, or what left the corpus, which is what a reviewer opens this
        file to check.
        """
        controls, redirects = cls._walk(rows)
        by_id = {c.control_id: c for c in controls}
        records: list[dict[str, object]] = []

        for task_id in sorted(redirects):
            records.append({
                "control_id": task_id,
                "repair": "redirect_stub_excluded",
                "before": f"{task_id}: {redirects[task_id]}",
                "after": "",
                "reason": (
                    "the source retired this task number and its cell holds a "
                    "pointer rather than a statement, so emitting it would put "
                    "a 15-character non-statement anchor in the corpus. No "
                    "curated link targets any of the five"
                ),
            })

        for fragment in sorted(MALFORMED_SECTION_IDS):
            target = MALFORMED_SECTION_IDS[fragment]
            control = by_id.get(target)
            if control is None:
                continue
            records.append({
                "control_id": target,
                "repair": "malformed_link_id_aliased_to_its_task",
                "before": fragment,
                "after": control.description,
                "reason": (
                    "OpenCRE wrote a mid-sentence fragment of the task where "
                    "the task id belongs, so without this alias the link falls "
                    "back to training on the fragment itself"
                ),
            })
        return records

    @staticmethod
    def _record_redirects(control: Control, redirects: dict[str, str]) -> None:
        """Attach the retired task numbering to the framework's first control.

        Recorded rather than dropped silently, and in the tracked artifact
        rather than only in the gitignored audit file. A reader who finds
        PW.3.2 cited in an older mapping needs somewhere in the published
        artifact that says where it went.
        """
        metadata = dict(control.metadata or {})
        metadata["retired_tasks"] = [
            f"{task}: {target}" for task, target in sorted(redirects.items())
        ]
        control.metadata = metadata

    @staticmethod
    def _check_alternate_ids(controls: list[Control], required: bool) -> None:
        """Refuse a declared malformed-id map that no longer matches the source.

        Both halves are checked. A target that vanished leaves the link
        unresolved, which the join floor would catch. A target that still
        exists but whose statement no longer holds the fragment is the worse
        case: the link resolves, the floor stays green, and the anchor belongs
        to a different task than the curator meant.

        Raises:
            ValueError: If a declared target is absent, or its statement no
                longer contains the declared fragment.
        """
        if not required:
            return
        by_id = {c.control_id: c for c in controls}
        missing = sorted({
            target for target in MALFORMED_SECTION_IDS.values()
            if target not in by_id
        })
        if missing:
            raise ValueError(
                f"nist_ssdf: MALFORMED_SECTION_IDS names task(s) {missing} "
                f"that this parse did not produce. Two curated links carry a "
                f"sentence fragment where a task id belongs and reach their "
                f"task only through this map; a stale entry leaves them "
                f"unresolved while every other gate stays green."
            )
        for fragment, target in sorted(MALFORMED_SECTION_IDS.items()):
            if fragment in by_id[target].description:
                continue
            raise ValueError(
                f"nist_ssdf: the declared fragment for task {target} is no "
                f"longer inside that task's statement. Declared: "
                f"{fragment!r}. Statement: {by_id[target].description!r}. The "
                f"link still resolves and the join floor stays green, so "
                f"nothing else would report that the curator's fragment now "
                f"belongs to a different task."
            )


def main() -> None:
    NistSsdfParser().run()


if __name__ == "__main__":
    main()
