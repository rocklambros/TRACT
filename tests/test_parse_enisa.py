"""ENISA has no control id, so the join is the name, and the name is damaged.

Measured on the pinned PDF with pdfplumber 0.11.10: naive exact matching of the
68 curated link names against the extracted titles resolves 57. Removing the
footnote digits fused onto five control names takes it to 65, folding the
source's typographic punctuation to the ASCII OpenCRE stores takes it to 60,
and doing both takes it to 68 of 68.

The definition also lands in column 2 on some rows and column 3 on others,
which is why the merge is per row rather than per page. The NAME lands in
column 0 on most rows and column 1 on others, which is why both columns are
read: Table 5 puts two of its controls and all three of its category banners in
column 1, and Table 3 puts every sub-threat there.

Every call to rows_to_units passes banners.
TestTheBannerArgumentIsLoadBearing states why the default stays empty.

TestSyntheticPdf drives parse() through pdfplumber against a PDF this file
builds, so the two-table extraction and all four gates run in CI, where
data/raw is absent. TestShippedArtifact reads the tracked JSON only, so the
join assertions run there too.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import pdfplumber
import pytest

from parsers.parse_enisa import (
    DEFINITION_END_COLUMN,
    FOOTNOTE_NAMES,
    NAME_COLUMNS,
    SOURCE_FILE,
    SOURCE_NAME_METADATA_KEY,
    TABLE3_BANNERS,
    TABLE3_PAGES,
    TABLE5_BANNERS,
    TABLE5_PAGES,
    EnisaParser,
)
from tests.synthetic_pdf import build_pdf
from tract.config import (
    DESCRIPTION_MAX_LENGTH,
    HONEST_PROSE_MIN_CHARS,
    MAX_ANCHOR_CHARS,
    PROCESSED_FRAMEWORKS_DIR,
)
from tract.corpus_report import (
    CURATED_LINKS_PATH,
    DETECTOR_B_INAPPLICABLE,
    JOIN_FLOORS,
    JOIN_WRONG_ANCHOR_BUDGET,
    SYNTHETIC_TEXT_ORIGIN,
    TEXT_ORIGIN_METADATA_KEY,
    CorpusReport,
    build_corpus_report,
    check_join_floors,
    name_level_mismatch_frameworks,
    wrong_anchor_applicable,
)
from tract.parsers.base import BaseParser
from tract.sanitize import sanitize_text
from tract.schema import Control
from tract.text_selection import prepare_anchor

# Table 5's shape, as pdfplumber returns it. The name is in column 0 on the
# first two units and in column 1 on the third, which is how the source prints
# "Ensure reliable sources are used". The definition is in column 2 on the
# first unit and column 3 on the others. The lone "x" at column 4 is a
# lifecycle mark on the pages where the stage columns start early.
# Invented wording throughout: a fixture that pasted the source's own
# sentences would put publisher text into a tracked CC0 file.
TABLE5: list[list[str | None]] = [
    ["Security controls", "", "Definition", "", "", "Stages of the lifecycle"],
    ["", "", "", "", "", ""],
    ["", "ORGANISATIONAL", "", "", "", ""],
    ["Apply modifications on inputs17", "",
     "Modify the model inputs so that an adversarial perturbation loses its",
     "", "", "x"],
    ["", "", "effect before the input reaches the model.", "", "", ""],
    ["Ensure ML applications comply with third parties’ security "
     "requirements", "", "",
     "Third-party components used by an ML application must meet the", "x",
     "x"],
    ["", "", "", "same security requirements as first-party components.", "",
     ""],
    ["", "Restrict the sources that feed a run", "",
     "Assess every upstream source before it is used, because open elements "
     "are widespread here.", "", ""],
]

# Table 3's shape. The threat is in column 0 and the sub-threat in column 1,
# and both are units: 20 of the 68 curated links target a Table 3 entry. The
# fourth row is what the source's second page opens with: the rotated lifecycle
# header, which pdfplumber returns as reversed character runs at column 4 on a
# row carrying no name. Both pages are collected into one row list, so a
# definition span that reached column 4 would append that fragment to the last
# unit of the page before.
TABLE3: list[list[str | None]] = [
    ["Threats | sub-\nthreats", "", "Definition", "", "", "Stage"],
    ["Evasion", "",
     "A type of attack in which the attacker works on the ML algorithm inputs",
     "", "", "x"],
    ["", "", "to find small perturbations leading to large output errors.", "",
     "", ""],
    ["", "", "", "", "a\nta\nD", ""],
    ["", "Data disclosure",
     "This threat refers to a leak of data manipulated by the ML application.",
     "x", "", ""],
]

# Six columns, so the widest table clears DEFINITION_END_COLUMN, with the last
# two narrow enough to hold a lifecycle mark and nothing else. Verified to
# round-trip through pdfplumber 0.11.10.
COLUMNS: list[float] = [15.0, 145.0, 275.0, 405.0, 535.0, 560.0, 590.0]

# A control statement long enough to clear HONEST_PROSE_MIN_CHARS and to beat
# every declared name by more than PROSE_MIN_EXTRA_CHARS, so the fixture
# exercises the floor rather than tripping it.
_STATEMENT = (
    "A statement that runs long enough to stand on its own as a control, "
    "rather than restating the name above it, number {index}."
)


def _wrap(text: str, width: int = 22) -> list[str]:
    """Break text into lines narrow enough to stay inside one fixture column."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _table(
    rows: list[dict[int, str]], split_rows: tuple[int, ...] = (),
) -> tuple[
    list[tuple[float, float, str]], list[tuple[float, float, float, float]],
]:
    """Column-indexed cells as a ruled table, with per-line bands where asked.

    A row named in split_rows keeps its whole name in the first band and gives
    every later definition line a band of its own, which is the shape the real
    PDF has: the name arrives once and the rest of the definition arrives as
    continuation rows carrying no name. Every other row is a single band, so
    the fixture covers both shapes.
    """
    runs: list[tuple[float, float, str]] = []
    bands = [50.0]
    for index, row in enumerate(rows):
        wrapped = {column: _wrap(text) for column, text in row.items()}
        name_lines = max(
            (len(v) for c, v in wrapped.items() if c < NAME_COLUMNS), default=0
        )
        body_lines = max(
            (len(v) for c, v in wrapped.items() if c >= NAME_COLUMNS), default=0
        )
        if index in split_rows and body_lines > 1:
            heights = [12 * max(name_lines, 1) + 10] + [12.0] * (body_lines - 1)
        else:
            heights = [12 * max(name_lines, body_lines, 1) + 10]
        start = bands[-1]
        for column, lines in wrapped.items():
            for offset, line in enumerate(lines):
                if column >= NAME_COLUMNS and offset < len(heights):
                    y = start + sum(heights[:offset]) + 10
                else:
                    y = start + 10 + offset * 12
                runs.append((COLUMNS[column] + 2, y, line))
        for height in heights:
            bands.append(bands[-1] + height)
    rules = [(x, bands[0], x, bands[-1]) for x in COLUMNS]
    rules += [(COLUMNS[0], y, COLUMNS[-1], y) for y in bands]
    return runs, rules


# Enough pages for the fixture to occupy the FIRST and the LAST page of
# TABLE5_PAGES, which is what makes the declared range load-bearing in a
# checkout with no data/raw: shrink the range by a page and a unit disappears.
_FIXTURE_PAGES = TABLE5_PAGES.stop

# A narrow ruled box drawn above the real table on one page. The source puts
# several tables on every page it uses, and the one that matters is the widest,
# so a fixture with a single table per page cannot tell "widest" from "first".
_DECOY_COLUMNS: list[float] = [15.0, 120.0, 225.0]


def _decoy() -> tuple[
    list[tuple[float, float, str]], list[tuple[float, float, float, float]],
]:
    """A two-column table narrower than DEFINITION_END_COLUMN, drawn first."""
    bands = [10.0, 26.0]
    rules = [(x, bands[0], x, bands[-1]) for x in _DECOY_COLUMNS]
    rules += [(_DECOY_COLUMNS[0], y, _DECOY_COLUMNS[-1], y) for y in bands]
    runs = [
        (_DECOY_COLUMNS[0] + 2, bands[0] + 11, "Figure"),
        (_DECOY_COLUMNS[1] + 2, bands[0] + 11, "caption"),
    ]
    return runs, rules


def _two_table_pdf() -> bytes:
    """Table 3 and Table 5 on the pages the parser reads.

    Every declared footnote name appears, because _check_declarations refuses a
    parse that does not produce all of FOOTNOTE_NAMES. A fixture that omitted
    one would fail the gate rather than test it.
    """
    footnotes = sorted(FOOTNOTE_NAMES)
    first_page: list[dict[int, str]] = [
        {0: "Security controls", 2: "Definition", 5: "Stage"},
        {1: "ORGANISATIONAL"},
    ]
    for index, name in enumerate(footnotes[:2]):
        first_page.append(
            {0: name, 2 + index % 2: _STATEMENT.format(index=index), 5: "x"}
        )
    second_page: list[dict[int, str]] = [
        {0: "Security controls", 2: "Definition", 5: "Stage"},
        {1: "TECHNICAL"},
    ]
    for index, name in enumerate(footnotes[2:], start=2):
        second_page.append(
            {0: name, 2 + index % 2: _STATEMENT.format(index=index), 4: "x"}
        )
    # The name in column 1 and a typographic apostrophe, both of which the real
    # Table 5 carries and neither of which the row-list tests drive through
    # pdfplumber. On the last page of the range, so the range is tested.
    last_page: list[dict[int, str]] = [
        {0: "Security controls", 1: "Definition", 4: "Stage"},
        {1: "Ensure third parties’ controls are checked",
         3: _STATEMENT.format(index=90)},
    ]
    third_table: list[dict[int, str]] = [
        {0: "Threats | sub-threats", 2: "Definition", 5: "Stage"},
        {0: "Evasion", 2: _STATEMENT.format(index=91), 5: "x"},
        {1: "Label modification", 2: _STATEMENT.format(index=92)},
    ]

    pages: list[list[tuple[float, float, str]]] = [
        [] for _ in range(_FIXTURE_PAGES)
    ]
    rules: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(_FIXTURE_PAGES)
    ]
    pages[TABLE3_PAGES.start], rules[TABLE3_PAGES.start] = _table(third_table)
    # split_rows on the first unit of page one, so the continuation-row merge
    # runs end to end and not only in the row-list tests above.
    pages[TABLE5_PAGES.start], rules[TABLE5_PAGES.start] = _table(
        first_page, split_rows=(2,),
    )
    decoy_runs, decoy_rules = _decoy()
    body_runs, body_rules = _table(second_page)
    pages[TABLE5_PAGES.start + 1] = decoy_runs + body_runs
    rules[TABLE5_PAGES.start + 1] = decoy_rules + body_rules
    pages[TABLE5_PAGES.stop - 1], rules[TABLE5_PAGES.stop - 1] = _table(
        last_page,
    )
    return build_pdf(pages, rules)


def _shipped() -> dict[str, dict[str, Any]]:
    """The tracked artifact, keyed by control id."""
    payload = json.loads(
        (PROCESSED_FRAMEWORKS_DIR / "enisa.json").read_text(encoding="utf-8")
    )
    controls: list[dict[str, Any]] = payload["controls"]
    return {str(c["control_id"]): c for c in controls}


def _curated() -> list[dict[str, Any]]:
    """Every curated ENISA link."""
    rows: list[dict[str, Any]] = []
    with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("framework_id") == "enisa":
                rows.append(row)
    return rows


def _shared_prefix(entries: list[str]) -> int:
    """How many leading characters every entry has in common. Ruling R13."""
    return len(os.path.commonprefix(entries)) if entries else 0


def _join(tmp_path: Path) -> CorpusReport:
    """The curated-link join against this framework's own tracked artifact.

    Not against data/processed/all_controls.json. Ruling R15 keeps that shared
    derived file out of a parser task's commit, so a test reading it would
    assert state this commit does not carry. Every column the join reports is
    computed per framework, so a one-framework corpus produces the same row.
    """
    corpus = tmp_path / "corpus.json"
    record = json.loads(
        (PROCESSED_FRAMEWORKS_DIR / "enisa.json").read_text(encoding="utf-8")
    )
    corpus.write_text(
        json.dumps({"frameworks": [record]}, sort_keys=True), encoding="utf-8",
    )
    return build_corpus_report(corpus_path=corpus)


class TestRowsToUnits:
    def test_a_definition_in_column_three_is_not_lost(self) -> None:
        units = dict(
            EnisaParser.rows_to_units(TABLE5, NAME_COLUMNS, TABLE5_BANNERS)
        )
        key = (
            "Ensure ML applications comply with third parties’ security "
            "requirements"
        )
        assert "Third-party components" in units[key]
        assert "first-party components" in units[key]

    def test_continuation_rows_join_the_unit_above(self) -> None:
        units = dict(
            EnisaParser.rows_to_units(TABLE5, NAME_COLUMNS, TABLE5_BANNERS)
        )
        assert units["Apply modifications on inputs17"].endswith(
            "before the input reaches the model."
        )

    def test_a_banner_in_the_second_name_column_is_not_a_unit(self) -> None:
        """Table 5 prints ORGANISATIONAL, TECHNICAL and SPECIFIC ML there.

        Read with one name column those three are nameless rows, so they merge
        into the definition of whatever control precedes them.
        """
        names = [
            n for n, _ in
            EnisaParser.rows_to_units(TABLE5, NAME_COLUMNS, TABLE5_BANNERS)
        ]
        assert "ORGANISATIONAL" not in names
        assert "Security controls" not in names
        bodies = [
            b for _, b in
            EnisaParser.rows_to_units(TABLE5, NAME_COLUMNS, TABLE5_BANNERS)
        ]
        assert not any("ORGANISATIONAL" in b for b in bodies)

    def test_a_name_in_the_second_column_is_its_own_unit(self) -> None:
        """Two Table 5 controls are printed there, and both carry links."""
        names = [
            n for n, _ in
            EnisaParser.rows_to_units(TABLE5, NAME_COLUMNS, TABLE5_BANNERS)
        ]
        assert "Restrict the sources that feed a run" in names

    def test_a_threat_and_a_sub_threat_are_both_units(self) -> None:
        names = [
            n for n, _ in
            EnisaParser.rows_to_units(TABLE3, NAME_COLUMNS, TABLE3_BANNERS)
        ]
        assert names == ["Evasion", "Data disclosure"]

    def test_a_lone_lifecycle_mark_stays_out_of_the_definition(self) -> None:
        """Table 3's second page starts its stage columns at index 3."""
        units = dict(
            EnisaParser.rows_to_units(TABLE3, NAME_COLUMNS, TABLE3_BANNERS)
        )
        assert units["Data disclosure"] == (
            "This threat refers to a leak of data manipulated by the ML "
            "application."
        )

    def test_a_rotated_lifecycle_header_never_joins_a_definition(self) -> None:
        """The defect a five-column definition span shipped. [measured]

        Both Table 3 pages are collected into one row list, so the rotated
        header that opens the second page attached itself to the last unit of
        the first, which is "Model or data disclosure" in the real source.
        """
        units = dict(
            EnisaParser.rows_to_units(TABLE3, NAME_COLUMNS, TABLE3_BANNERS)
        )
        assert units["Evasion"] == (
            "A type of attack in which the attacker works on the ML algorithm "
            "inputs to find small perturbations leading to large output "
            "errors."
        )

    def test_no_unit_extracts_with_an_empty_definition(self) -> None:
        for rows, banners in (
            (TABLE5, TABLE5_BANNERS), (TABLE3, TABLE3_BANNERS),
        ):
            for name, body in EnisaParser.rows_to_units(
                rows, NAME_COLUMNS, banners,
            ):
                assert body, name

    def test_the_banner_argument_is_load_bearing(self) -> None:
        """Left at its default, the header row becomes a unit.

        The default stays empty rather than being filled with Table 5's
        banners, so a caller that forgets gets one extra unit that
        _check_shape refuses by count instead of a silently different filter.
        """
        names = [n for n, _ in EnisaParser.rows_to_units(TABLE5, NAME_COLUMNS)]
        assert "Security controls" in names


class TestNameNormalisation:
    def test_a_fused_footnote_digit_is_removed(self) -> None:
        assert EnisaParser.normalise_name(
            EnisaParser.clean("Apply modifications on inputs17")
        ) == "apply modifications on inputs"

    def test_a_curly_apostrophe_matches_the_ascii_one(self) -> None:
        assert EnisaParser.normalise_name(
            "Ensure ML applications comply with third parties’ security "
            "requirements"
        ) == EnisaParser.normalise_name(
            "Ensure ML applications comply with third parties' security "
            "requirements"
        )

    def test_an_ellipsis_matches_three_periods(self) -> None:
        assert EnisaParser.normalise_name(
            "Use of adversarial examples crafted in white or grey box "
            "conditions (e.g. FGSM…)"
        ) == EnisaParser.normalise_name(
            "Use of adversarial examples crafted in white or grey box "
            "conditions (e.g. FGSM...)"
        )

    def test_a_name_that_genuinely_ends_in_a_digit_is_untouched(self) -> None:
        """Why FOOTNOTE_NAMES is declared instead of a trailing-digit regex."""
        assert EnisaParser.clean("Comply with ISO 27001") == (
            "Comply with ISO 27001"
        )


class TestSyntheticPdf:
    """parse() through pdfplumber, with no dependency on data/raw."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> EnisaParser:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(_two_table_pdf())
        instance = EnisaParser(
            raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path,
        )
        instance.expected_sha256 = None  # type: ignore[misc]
        instance.expected_table5_units = len(FOOTNOTE_NAMES) + 1  # type: ignore[misc]
        instance.expected_table3_units = 2  # type: ignore[misc]
        instance.expected_count = (  # type: ignore[misc]
            instance.expected_table5_units + instance.expected_table3_units
        )
        return instance

    def test_both_tables_are_read(self, parser: EnisaParser) -> None:
        controls = parser.parse()
        levels = [c.hierarchy_level for c in controls]
        assert levels.count("threat") == parser.expected_table3_units
        assert levels.count("control") == parser.expected_table5_units

    def test_a_continuation_row_reaches_the_stored_statement(
        self, parser: EnisaParser,
    ) -> None:
        """The first fixture unit is split into one ruled band per line."""
        controls = {c.title: c for c in parser.parse()}
        first = controls[FOOTNOTE_NAMES[sorted(FOOTNOTE_NAMES)[0]]]
        assert first.description.endswith("number 0.")

    def test_a_footnote_digit_is_gone_from_the_stored_title(
        self, parser: EnisaParser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert "Apply modifications on inputs" in titles
        assert "Apply modifications on inputs17" not in titles

    def test_a_typographic_apostrophe_is_folded_in_the_stored_title(
        self, parser: EnisaParser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert "Ensure third parties' controls are checked" in titles

    def test_nothing_is_marked_synthetic(self, parser: EnisaParser) -> None:
        """Every statement is the publisher's own definition column.

        Nothing here is assembled out of fragments, so no control carries the
        synthetic origin and the report's anchor_source_synthetic reads 0.
        """
        for control in parser.parse():
            assert TEXT_ORIGIN_METADATA_KEY not in (control.metadata or {})

    def test_the_repair_audit_records_before_and_after_text(
        self, parser: EnisaParser, tmp_path: Path,
    ) -> None:
        """A count says a repair fired, not what moved. Ruling R13."""
        (tmp_path / "out").mkdir()
        parser.run()
        lines = [
            json.loads(line)
            for line in (tmp_path / "enisa.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
        assert len(lines) == len(FOOTNOTE_NAMES) + 1
        for record in lines:
            assert record["before"] != record["after"]
            assert record["repair"] in {
                "footnote_reference_removed", "punctuation_folded_to_ascii",
            }
        repaired = {r["after"] for r in lines}
        assert "Apply modifications on inputs" in repaired
        assert "Ensure third parties' controls are checked" in repaired

    def test_an_unrepaired_title_writes_no_audit_record(
        self, parser: EnisaParser, tmp_path: Path,
    ) -> None:
        """Without this the count above would pass on a record per control."""
        (tmp_path / "out").mkdir()
        parser.run()
        lines = [
            json.loads(line)
            for line in (tmp_path / "enisa.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
        assert "Evasion" not in {r["after"] for r in lines}

    def test_a_short_table_five_is_refused(self, parser: EnisaParser) -> None:
        """The count band would accept the loss. The unit counts do not."""
        parser.expected_table5_units += 1  # type: ignore[misc]
        with pytest.raises(ValueError, match="Table 5"):
            parser.parse()

    def test_a_short_table_three_is_refused(self, parser: EnisaParser) -> None:
        parser.expected_table3_units += 1  # type: ignore[misc]
        with pytest.raises(ValueError, match="Table 3"):
            parser.parse()

    def test_a_stale_footnote_declaration_is_refused(
        self, parser: EnisaParser, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stale entry means a title still carries its footnote digit.

        The entry is added after the fixture has built its PDF, so the unit
        count does not move and _check_shape passes. Only the declaration gate
        can catch this.
        """
        monkeypatch.setitem(
            FOOTNOTE_NAMES, "Gone from the source99", "Gone from the source",
        )
        with pytest.raises(ValueError, match="FOOTNOTE_NAMES"):
            parser.parse()

    def test_a_pdf_with_no_wide_table_is_refused(
        self, parser: EnisaParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "raw" / SOURCE_FILE).write_bytes(
            build_pdf([[(72.0, 100.0, "Not in a table.")]] * _FIXTURE_PAGES)
        )
        with pytest.raises(ValueError, match="no table rows"):
            parser.parse()

    def test_run_writes(self, parser: EnisaParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == parser.expected_count
        assert [s.path for s in output.source_files] == [SOURCE_FILE]
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0

    def test_run_is_byte_identical_on_a_second_pass(
        self, parser: EnisaParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        first = (tmp_path / "out" / "enisa.json").read_bytes()
        parser.run()
        assert (tmp_path / "out" / "enisa.json").read_bytes() == first


class TestDescriptionBudget:
    """Ruling R14, and the positive control that proves the trap is real."""

    def test_the_trap_is_real(self) -> None:
        """Without this, the guard below asserts nothing.

        `_sanitize_control` calls `sanitize_text(description, return_full=True)`
        and assigns the second return value to `full_text`, discarding whatever
        the parser wrote there. A non-None second value is that condition.
        """
        _, overflow = sanitize_text(
            "word " * 500, max_length=DESCRIPTION_MAX_LENGTH, return_full=True,
        )
        assert overflow is not None

    def test_a_statement_at_the_limit_is_refused(self) -> None:
        control = Control(
            control_id="evasion",
            title="Evasion",
            description="w" * DESCRIPTION_MAX_LENGTH,
        )
        with pytest.raises(ValueError, match="BaseParser._sanitize_control"):
            EnisaParser._check_description_budget([control])

    def test_a_statement_one_character_under_the_limit_passes(self) -> None:
        """The guard refuses at the limit, so this is the boundary below it."""
        control = Control(
            control_id="evasion",
            title="Evasion",
            description="w" * (DESCRIPTION_MAX_LENGTH - 1),
        )
        EnisaParser._check_description_budget([control])


class TestUniqueIds:
    """The id is derived from the name, so a collision hides a whole unit.

    Nothing in the pinned source collides, so without a direct test the gate
    would be a branch no run ever takes and deleting it would cost nothing.
    """

    def test_two_names_that_slug_alike_are_refused(self) -> None:
        colliding = [
            Control(control_id=EnisaParser.slug(title), title=title,
                    description=f"A statement for {title}.")
            for title in ("Model disclosure", "Model  disclosure!")
        ]
        assert colliding[0].control_id == colliding[1].control_id
        with pytest.raises(ValueError, match="both generate the control id"):
            EnisaParser._check_unique_ids(colliding)

    def test_a_long_name_is_not_cut_short(self) -> None:
        """A cut slug is where two long names become one silent id.

        The longest ENISA title is 106 characters, which is what the tracked
        artifact carries, so a truncating slug would still ship 50 ids today
        and collide on the first source revision that lengthens a name.
        """
        longest = (
            "Ensure ML applications comply with protection policies and are "
            "integrated to security operations processes"
        )
        assert len(longest) == 106
        assert len(EnisaParser.slug(longest)) == 106

    def test_two_names_that_slug_apart_pass(self) -> None:
        distinct = [
            Control(control_id=EnisaParser.slug(title), title=title,
                    description=f"A statement for {title}.")
            for title in ("Model disclosure", "Model or data disclosure")
        ]
        EnisaParser._check_unique_ids(distinct)


class TestDigestGate:
    def test_a_different_pdf_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(_two_table_pdf())
        parser = EnisaParser(raw_dir=raw, output_dir=tmp_path)
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()

    def test_the_shipped_pin_is_the_one_the_fetcher_downloads(self) -> None:
        """Without this the class could ship `expected_sha256 = None`.

        Every synthetic-PDF test stands the gate down on purpose, so none of
        them can see a parser with no pin at all.
        """
        from scripts.fetch_frameworks import SOURCES

        pins = {
            source.expected_sha256
            for source in SOURCES
            if source.framework_id == "enisa"
        }
        assert pins == {EnisaParser.expected_sha256}


class TestSyntheticPdfShape:
    """The fixture has to carry the shapes the real source does."""

    def test_the_fixture_puts_a_definition_in_two_different_columns(
        self,
    ) -> None:
        payload = _two_table_pdf()
        columns: set[int] = set()
        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            for page_number in TABLE5_PAGES:
                tables = pdf.pages[page_number].extract_tables()
                if not tables:
                    continue
                widest = max(tables, key=lambda t: max(len(r) for r in t))
                for row in widest:
                    cells = [(c or "").strip() for c in row]
                    padded = cells + [""] * DEFINITION_END_COLUMN
                    for index in range(NAME_COLUMNS, DEFINITION_END_COLUMN):
                        if padded[index].startswith("A statement"):
                            columns.add(index)
        assert columns == {2, 3}

    def test_the_fixture_puts_a_narrower_table_first_on_one_page(self) -> None:
        """Otherwise "widest" and "first" pick the same table every time."""
        with pdfplumber.open(io.BytesIO(_two_table_pdf())) as pdf:
            tables = pdf.pages[TABLE5_PAGES.start + 1].extract_tables()
        widths = [max(len(row) for row in table) for table in tables]
        assert len(widths) >= 2
        assert widths[0] < DEFINITION_END_COLUMN
        assert max(widths) > DEFINITION_END_COLUMN

    def test_the_fixture_reaches_the_last_page_of_the_declared_range(
        self,
    ) -> None:
        """A page range shortened by one loses a unit and trips _check_shape."""
        with pdfplumber.open(io.BytesIO(_two_table_pdf())) as pdf:
            tables = pdf.pages[TABLE5_PAGES.stop - 1].extract_tables()
        assert tables
        assert max(len(row) for row in tables[0]) >= DEFINITION_END_COLUMN

    def test_the_declared_page_ranges_are_the_measured_ones(self) -> None:
        """A lock rather than a derivation, and it is one on purpose.

        The fixture lays its tables out from these two ranges, so it moves with
        them and cannot catch a change to either. Against the real PDF a
        shortened range loses a unit and _check_shape refuses, but CI has no
        data/raw, so without this the edit would be invisible there.

        Both ends are load-bearing. Table 3 ends where Table 4 begins on the
        next page, and Table 5 ends where the closing narrative begins, with
        Annex C further on. Widen either and a different table's rows arrive as
        units; narrow either and units disappear.
        """
        assert TABLE3_PAGES == range(14, 16)
        assert TABLE5_PAGES == range(19, 26)


class TestShippedArtifact:
    """Reads only tracked files, so it holds where data/raw is absent."""

    def test_the_shape_the_parser_declares(self) -> None:
        shipped = _shipped()
        assert len(shipped) == 50
        levels = [c["hierarchy_level"] for c in shipped.values()]
        assert levels.count("control") == 37
        assert levels.count("threat") == 13

    def test_the_control_ids_are_unique_and_untruncated(self) -> None:
        """A truncated slug is a silent id collision on two long names."""
        shipped = _shipped()
        assert len(shipped) == 50
        assert max(len(k) for k in shipped) == 106

    def test_no_description_reaches_the_length_that_rewrites_full_text(
        self,
    ) -> None:
        """Ruling R14. It does not fire here: the longest is 709."""
        lengths = [len(str(c["description"])) for c in _shipped().values()]
        assert max(lengths) == 709
        assert max(lengths) < DESCRIPTION_MAX_LENGTH
        assert not any("full_text" in c for c in _shipped().values())

    def test_the_prose_floor_is_cleared_by_every_control(self) -> None:
        shipped = _shipped()
        short = [
            control_id for control_id, c in shipped.items()
            if len(str(c["description"])) < HONEST_PROSE_MIN_CHARS
        ]
        assert short == []
        assert min(
            len(str(c["description"])) for c in shipped.values()
        ) == 80
        assert EnisaParser.min_prose_fraction == 1.0

    def test_no_rotated_lifecycle_header_reached_the_artifact(self) -> None:
        """The one anchor a five-column definition span corrupted. [measured]

        Table 3's second page opens with the rotated lifecycle header, which
        pdfplumber returns as reversed character runs at column 4 on a nameless
        row, so a span reaching that column appended "a ta D" to the last unit
        of the page before. This is the only control the change moved, so
        pinning its length and its tail fails in both directions: the leak
        coming back lengthens it, and any other edit to this statement fails
        too and gets read by a person.
        """
        disclosure = _shipped()["model-or-data-disclosure"]
        statement = str(disclosure["description"])
        assert len(statement) == 233
        assert statement.endswith("sensitive data)")

    def test_nothing_in_the_artifact_is_marked_synthetic(self) -> None:
        origins = {
            str((c.get("metadata") or {}).get(TEXT_ORIGIN_METADATA_KEY))
            for c in _shipped().values()
        }
        assert SYNTHETIC_TEXT_ORIGIN not in origins

    def test_the_anchors_share_no_leading_prefix(self) -> None:
        """Ruling R13. Zero, so nothing is stripped.

        Task 5's anchors opened on 364 byte-identical characters of a
        statistics table, 17% of the budget. A shared running header reaching
        the definition column fails here.
        """
        anchors = [
            prepare_anchor(str(c["description"]))[0]
            for c in _shipped().values()
        ]
        assert _shared_prefix(anchors) == 0
        assert len(set(anchors)) == 50
        assert max(len(a) for a in anchors) < MAX_ANCHOR_CHARS

    def test_the_join_is_sixty_eight_of_sixty_eight_through_the_title_channel(
        self, tmp_path: Path,
    ) -> None:
        """Every value is measured on the pinned PDF and every one can move.

        Built from this framework's own tracked artifact rather than from the
        merged corpus. Ruling R15 keeps data/processed/all_controls.json out of
        a parser task's commit. Every column below is per framework, so the two
        corpora produce the same row.
        """
        row = _join(tmp_path).by_id("enisa")
        assert row.links == 68
        assert row.by_title == 68
        assert row.by_id == 0
        assert row.unresolved == 0
        assert row.resolution_rate == 1.0
        assert row.resolution_rate >= JOIN_FLOORS["enisa"]
        assert row.distinct_anchors == 33
        assert row.distinct_anchors_pre_truncation == 33
        assert row.links_per_anchor == pytest.approx(68 / 33)
        assert row.truncated == 0
        assert row.nested_anchors == 0
        assert row.contained_anchors == 0
        assert row.dropped_by_prose_rule == 0
        assert row.fallback_anchors == 0
        assert row.distinct_hubs == 56
        assert row.links_per_hub == pytest.approx(68 / 56)
        assert row.anchor_source_full_text == 0
        assert row.anchor_source_description == 68
        assert row.anchor_source_title == 0
        assert row.anchor_source_synthetic == 0

    def test_the_floor_gate_passes_for_this_framework(
        self, tmp_path: Path,
    ) -> None:
        report = _join(tmp_path)
        assert check_join_floors(report, {"enisa": JOIN_FLOORS["enisa"]}) == []

    def test_detector_a_checks_every_link_and_flags_none(
        self, tmp_path: Path,
    ) -> None:
        """Zero over a denominator of 68, not zero over a denominator of 0.

        Every link resolves through the title channel and carries a non-empty
        section_id, so detector A runs on all 68. It cannot flag one, because
        no control id spells "Table 5:", "Table 3:", or any of the ten link
        names OpenCRE put in section_id, so the id channel never returns a
        second answer to disagree with. That is a property of this parser's id
        scheme rather than of the detector, and it fails here if a later change
        makes the id channel answer with different text.

        Ruling R21 declared enisa detector-B-inapplicable at 10 ids over 33
        names, and the two numbers above did not move, because detector B never
        ran for a framework whose links all resolve by title. Both memberships
        are asserted here so that the exemption stays visible next to the row it
        does not change.
        """
        report = _join(tmp_path)
        assert report.by_id("enisa").wrong_anchor_risk == 0
        assert wrong_anchor_applicable(report)["enisa"] == 68
        assert "enisa" not in JOIN_WRONG_ANCHOR_BUDGET
        assert "enisa" in DETECTOR_B_INAPPLICABLE
        assert "enisa" in name_level_mismatch_frameworks()

    def test_the_link_file_needs_no_alternate_title_table(self) -> None:
        """Derived from tracked files, so it fails in both directions.

        A curated name that stops matching a stored title fails here, which is
        what an OpenCRE refresh or a parser regression looks like. And the
        empty result is the claim: Annex C spells six Table 5 controls
        differently, no curated link uses any of those spellings, so declaring
        them as alternate titles would add six dead entries to a 33-anchor
        framework.
        """
        titles = {
            str(c["title"]).strip().lower() for c in _shipped().values()
        }
        assert len(titles) == 50
        unmatched = sorted({
            str(row.get("section_name") or "")
            for row in _curated()
            if str(row.get("section_name") or "").strip().lower() not in titles
        })
        assert unmatched == []

    def test_the_name_repairs_are_load_bearing_and_measurably_so(self) -> None:
        """Attainable range, both directions, from tracked files only.

        The artifact records each repaired name's printed spelling under
        `source_name`, so the unrepaired state is reconstructed rather than
        inverted. Matching the curated names against it resolves 57 of 68.
        Reversing only the footnote repair costs 8 links and reversing only the
        punctuation repair costs 3, so neither table can be dropped quietly.
        """
        curated = _curated()
        assert len(curated) == 68
        shipped = _shipped().values()

        def keys(reverse_footnotes: bool, reverse_punctuation: bool) -> set[str]:
            out: set[str] = set()
            for control in shipped:
                title = str(control["title"])
                printed = (control.get("metadata") or {}).get(
                    SOURCE_NAME_METADATA_KEY
                )
                is_footnote = printed in FOOTNOTE_NAMES
                reverse = (
                    reverse_footnotes if is_footnote else reverse_punctuation
                )
                out.add(
                    str(printed).strip().lower() if printed and reverse
                    else title.strip().lower()
                )
            return out

        def resolved(against: set[str]) -> int:
            return sum(
                1 for row in curated
                if str(row.get("section_name") or "").strip().lower() in against
            )

        assert resolved(keys(True, True)) == 57
        assert resolved(keys(True, False)) == 60
        assert resolved(keys(False, True)) == 65
        assert resolved(keys(False, False)) == 68


class TestRun:
    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = EnisaParser(output_dir=tmp_path, audit_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 50
        assert {c.hierarchy_level for c in output.controls} == {
            "control", "threat",
        }
        assert [s.path for s in output.source_files] == [SOURCE_FILE]
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0

    def test_the_two_column_one_controls_are_present(
        self, tmp_path: Path,
    ) -> None:
        """Table 5 prints these two in its sub-threat column, not column 0."""
        parser = EnisaParser(output_dir=tmp_path, audit_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        titles = {c.title for c in output.controls}
        assert "Ensure reliable sources are used" in titles
        assert (
            "Use methods to clean the training dataset from suspicious samples"
            in titles
        )

    def test_the_real_pdf_writes_seven_repair_records(
        self, tmp_path: Path,
    ) -> None:
        """Five footnote references and two punctuation folds. [measured]"""
        parser = EnisaParser(output_dir=tmp_path, audit_dir=tmp_path)
        try:
            parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        records = [
            json.loads(line)
            for line in (tmp_path / "enisa.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
        kinds = sorted(str(r["repair"]) for r in records)
        assert kinds.count("footnote_reference_removed") == 5
        assert kinds.count("punctuation_folded_to_ascii") == 2
