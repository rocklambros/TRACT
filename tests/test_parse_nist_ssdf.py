"""The SSDF task table is ruled, so extract_tables returns whole task cells.

Measured against the pinned PDF with pdfplumber 0.11.4: 47 task cells at column
index 3, no duplicate ids there, and 7 truncated second copies at column 4 on
seven pages. The parser reads column 3 only. The continuation rows below a task
repeat wrapped fragments of the practice cell in column 0 and hold nothing in
column 3, so the practice column needs a forward fill and nothing needs a
merge.

TestSyntheticPdf drives parse() through pdfplumber against a PDF this file
builds, so the extraction path runs in CI, where data/raw is absent.

Fixture text is invented rather than copied, with one deliberate exception:
both entries of MALFORMED_SECTION_IDS appear verbatim inside their task's
fixture statement, because the parser verifies exactly that and a paraphrase
would turn the check into a check of nothing. Those two strings are already
tracked in data/training/hub_links_curated.jsonl, and NIST SP 800-218 is a US
Government work in no licence tier.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pdfplumber
import pytest

from parsers.parse_nist_ssdf import (
    MALFORMED_SECTION_IDS,
    SOURCE_FILE,
    NistSsdfParser,
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
    CorpusReport,
    build_corpus_report,
    name_kind_mismatch_frameworks,
    name_level_mismatch_frameworks,
    wrong_anchor_applicable,
)
from tract.parsers.base import BaseParser
from tract.sanitize import sanitize_text
from tract.schema import Control
from tract.text_selection import prepare_anchor

# A practice header cell as the source writes it: the name, the id in
# parentheses, then the practice statement. Invented wording, real id shape.
PRACTICE = (
    "Record Security Requirements for Software\nDevelopment (PO.1): Say what "
    "the requirements are before anyone writes code."
)

# Twelve columns, as the source table has. Only 0, 3 and 6 carry text.
ROWS: list[list[str | None]] = [
    [None, "Practices", None, "Tasks", None, None, "Notional Implementation "
     "Examples", None, None, None, "References", None],
    [PRACTICE, "Record Security Requirements for Software", None,
     "PO.1.1: Write down every security requirement the delivery pipeline "
     "places on the teams that build software, and keep the list current.",
     None, None, "Example 1: Publish a written policy.", None, None, None,
     "BSAFSS: SM.3", None],
    [None, "Development (PO.1): Say what the requirements", None, None, None,
     None, None, None, None, None, "BSIMM: CP1.1", None],
    [None, None, None,
     "PO.1.2: Write down every security requirement the shipped product has "
     "to satisfy, and keep that list current as well.",
     None, None, None, None, None, None, "EO14028: 4e(ix)", None],
    # The seven pages that carry a truncated second copy of a task put it at
    # column 4. The parser must not see it. Column 3 is empty on this row.
    [None, None, None, None,
     "PO.1.9: Write down every security requirement the delivery pipe",
     None, None, None, None, None, None, None],
    [None, None, None, "PW.3.2: Moved to PW.4.4", None, None, None, None,
     None, None, None, None],
]

# Column and row boundaries for the synthetic table. Seven cells per row, so
# the task lands at index 3 and the examples at index 6, exactly as the source
# does. Verified to round-trip through pdfplumber 0.11.4.
COLUMNS = [40.0, 200.0, 215.0, 230.0, 430.0, 445.0, 460.0, 580.0]
BANDS = [60.0, 130.0, 230.0, 330.0, 440.0, 560.0, 610.0]
CELLS: list[dict[int, list[str]]] = [
    {0: ["Practices"], 3: ["Tasks"], 6: ["Examples"]},
    {0: ["Record Security", "Requirements for Software",
         "Development (PO.1): Say", "what the requirements are",
         "before anyone writes code."],
     3: ["PO.1.1: Write down every", "security requirement the",
         "delivery pipeline places", "on the teams that build",
         "software, and keep the", "list current."],
     6: ["Example 1: Publish a", "written policy."]},
    # A continuation row: column 0 holds a wrapped fragment of the practice
    # above, which is not a practice header, so the fill must survive it.
    {0: ["Requirements for", "Software Development"],
     3: ["PO.1.2: Write down every", "security requirement the",
         "shipped product has to", "satisfy, and keep that",
         "list current as well."]},
    {0: ["Guard Every Form of Code", "(PS.1): Keep code out of",
         "unauthorized hands."],
     3: ["PS.1.1: Hold source code,", "executable code, and",
         "configuration-as-code –", "based on the principle of",
         "least privilege so that", "only authorized personnel,",
         "tools, services, etc. have", "access."],
     6: ["Example 1: Restrict the", "repository."]},
    {0: ["Test Built Code for Flaws", "(PW.8): Run the checks",
         "that earlier reviews", "cannot."],
     3: ["PW.8.1: Decide whether a", "build-time scan should be",
         "performed to find", "vulnerabilities not",
         "identified by previous", "reviews, analysis, or",
         "testing and, if so, which", "types of testing should",
         "be used."]},
    {3: ["PW.3.2: Moved to PW.4.4"]},
]

# The synthetic table lands on page index 13, so TABLE_PAGES needs no override.
BLANK_PAGES = 13


def _table_pdf() -> bytes:
    """One ruled table on page 13, so TABLE_PAGES needs no override."""
    rules = [(x, BANDS[0], x, BANDS[-1]) for x in COLUMNS]
    rules += [(COLUMNS[0], y, COLUMNS[-1], y) for y in BANDS]
    runs: list[tuple[float, float, str]] = []
    for index, group in enumerate(CELLS):
        top = BANDS[index] + 11
        for column, lines in group.items():
            for offset, line in enumerate(lines):
                runs.append((COLUMNS[column] + 3, top + offset * 10, line))
    blank_text: list[list[tuple[float, float, str]]] = [
        [] for _ in range(BLANK_PAGES)
    ]
    blank_rules: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(BLANK_PAGES)
    ]
    return build_pdf(blank_text + [runs], blank_rules + [rules])


def _shipped() -> dict[str, dict[str, object]]:
    """The tracked artifact, keyed by control id."""
    payload = json.loads(
        (PROCESSED_FRAMEWORKS_DIR / "nist_ssdf.json").read_text(
            encoding="utf-8",
        )
    )
    return {c["control_id"]: c for c in payload["controls"]}


def _shared_prefix(entries: list[str]) -> int:
    """How many leading characters every entry has in common. Ruling R13."""
    return len(os.path.commonprefix(entries)) if entries else 0


def _join(tmp_path: Path) -> CorpusReport:
    """The curated-link join against this framework's own tracked artifact.

    Not against data/processed/all_controls.json. Ruling R15 keeps that shared
    derived file out of a parser task's commit, so a test reading it would
    assert state this commit does not carry, and it would be red in CI until
    some later task merges. Every column the join reports is computed per
    framework, so a one-framework corpus produces the same row.
    """
    corpus = tmp_path / "corpus.json"
    record = json.loads(
        (PROCESSED_FRAMEWORKS_DIR / "nist_ssdf.json").read_text(
            encoding="utf-8",
        )
    )
    corpus.write_text(
        json.dumps({"frameworks": [record]}, sort_keys=True),
        encoding="utf-8",
    )
    return build_corpus_report(corpus_path=corpus)


class TestRowsToControls:
    def test_only_task_cells_become_controls(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert [c.control_id for c in controls] == ["PO.1.1", "PO.1.2"]

    def test_a_task_shaped_cell_at_column_four_is_ignored(self) -> None:
        """Seven source pages carry a truncated second copy there. [measured]"""
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert "PO.1.9" not in {c.control_id for c in controls}

    def test_the_whole_task_statement_survives(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.description.endswith("keep the list current.")
        assert first.description.startswith("Write down every security")

    def test_title_is_the_task_id_not_the_statement(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.title == "PO.1.1"
        assert first.description != first.title

    def test_the_practice_is_forward_filled_onto_later_tasks(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert controls[0].parent_id == "PO.1"
        assert controls[1].parent_id == "PO.1"
        assert controls[1].parent_name is not None
        assert controls[1].parent_name.startswith(
            "Record Security Requirements for Software Development"
        )

    def test_a_moved_to_stub_is_not_emitted(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert "PW.3.2" not in {c.control_id for c in controls}

    def test_a_retired_task_is_still_findable_in_the_artifact(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.metadata is not None
        assert first.metadata["retired_tasks"] == ["PW.3.2: Moved to PW.4.4"]

    def test_notional_examples_are_kept_out_of_the_anchor(self) -> None:
        """ProseIndex prefers full_text, so full_text IS the anchor.

        The examples say how to satisfy the task, which is the class of text
        REMEDIATION_HEADINGS cuts. They are stored where a reviewer reads them
        and no encoder does.
        """
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.full_text is None
        assert "Example 1" not in first.description
        assert first.metadata is not None
        assert first.metadata["notional_examples"] == (
            "Example 1: Publish a written policy."
        )

    def test_a_task_without_examples_declares_none(self) -> None:
        second = NistSsdfParser.rows_to_controls(ROWS)[1]
        assert second.metadata is not None
        assert "notional_examples" not in second.metadata

    def test_the_practice_id_reaches_metadata(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert [(c.metadata or {})["practice"] for c in controls] == [
            "PO.1", "PO.1",
        ]

    def test_reparse_of_the_same_rows_is_identical(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)
        second = NistSsdfParser.rows_to_controls(ROWS)
        assert [c.model_dump() for c in first] == [
            c.model_dump() for c in second
        ]


class TestRepairRecords:
    """A count says a repair fired. These say what moved."""

    def test_an_excluded_stub_records_the_text_it_dropped(self) -> None:
        records = NistSsdfParser.repair_records(ROWS)
        stubs = [r for r in records if r["repair"] == "redirect_stub_excluded"]
        assert len(stubs) == 1
        assert stubs[0]["before"] == "PW.3.2: Moved to PW.4.4"
        assert stubs[0]["after"] == ""

    def test_an_alias_records_the_fragment_and_the_statement(self) -> None:
        records = NistSsdfParser.repair_records(_pdf_rows())
        aliases = [
            r for r in records
            if r["repair"] == "malformed_link_id_aliased_to_its_task"
        ]
        assert {r["control_id"] for r in aliases} == {"PS.1.1", "PW.8.1"}
        for record in aliases:
            assert record["before"] in str(record["after"])

    def test_an_absent_target_writes_no_alias_record(self) -> None:
        """ROWS carries neither PS.1.1 nor PW.8.1."""
        records = NistSsdfParser.repair_records(ROWS)
        assert not [
            r for r in records
            if r["repair"] == "malformed_link_id_aliased_to_its_task"
        ]


def _pdf_rows() -> list[list[str | None]]:
    """The synthetic table read back through pdfplumber."""
    rows: list[list[str | None]] = []
    with pdfplumber.open(io.BytesIO(_table_pdf())) as pdf:
        for table in pdf.pages[BLANK_PAGES].extract_tables():
            rows.extend(table)
    return rows


class TestMalformedIdMap:
    def test_an_alternate_whose_target_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names task"):
            NistSsdfParser.rows_to_controls(ROWS, require_alternate_targets=True)

    def test_an_alternate_absent_from_its_targets_statement_is_refused(
        self,
    ) -> None:
        """The worse half of the check. The link would still resolve.

        A target that vanished shows up as an unresolved link and the join
        floor catches it. A target that kept its id and lost the fragment
        resolves to a statement the curator never pointed at, with every gate
        green.
        """
        rows = [
            [None, None, None, "PS.1.1: A statement with none of it.",
             None, None, None, None, None, None, None, None],
            [None, None, None, "PW.8.1: Nor this one.",
             None, None, None, None, None, None, None, None],
        ]
        with pytest.raises(ValueError, match="no longer inside"):
            NistSsdfParser.rows_to_controls(rows, require_alternate_targets=True)

    def test_the_declared_map_is_exactly_what_the_link_file_malforms(
        self,
    ) -> None:
        """Derived in both directions, from two tracked files.

        A curated section_id that stops naming a real task and is not declared
        fails here, and a declared fragment the link file no longer spells
        fails here too. Neither side reads data/raw.
        """
        shipped = _shipped()
        assert len(shipped) == 42, sorted(shipped)

        observed: dict[str, str] = {}
        with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("framework_id") != "nist_ssdf":
                    continue
                section_id = str(row.get("section_id") or "")
                if section_id in shipped:
                    continue
                targets = [
                    control_id for control_id, control in shipped.items()
                    if section_id and section_id in str(control["description"])
                ]
                assert len(targets) == 1, (section_id, targets)
                observed[section_id] = targets[0]

        assert observed == dict(MALFORMED_SECTION_IDS)

    def test_both_declared_fragments_end_the_way_the_link_file_does(
        self,
    ) -> None:
        """An earlier draft ended the PW.8.1 fragment 'should be performed.'

        The curated link ends 'should be used.', so that entry matched
        nothing, the link stayed unresolved, and the declared 46/46 ceiling was
        unreachable while every other gate stayed green.
        """
        fragments = sorted(MALFORMED_SECTION_IDS)
        assert len(fragments) == 2
        assert any(f.endswith("which types of testing should be used.")
                   for f in fragments)
        assert not any(f.endswith("should be performed.") for f in fragments)


class TestSyntheticPdf:
    """parse() through pdfplumber, with no dependency on data/raw."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> NistSsdfParser:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(_table_pdf())
        instance = NistSsdfParser(
            raw_dir=raw,
            output_dir=tmp_path / "out",
            audit_dir=tmp_path / "audit",
        )
        instance.expected_sha256 = None  # type: ignore[misc]
        instance.expected_count = 4  # type: ignore[misc]
        instance.expected_task_cells = 5  # type: ignore[misc]
        instance.expected_redirects = 1  # type: ignore[misc]
        instance.expected_practices = 3  # type: ignore[misc]
        return instance

    def test_parse_reads_the_ruled_table(self, parser: NistSsdfParser) -> None:
        controls = parser.parse()
        assert [c.control_id for c in controls] == [
            "PO.1.1", "PO.1.2", "PS.1.1", "PW.8.1",
        ]

    def test_the_practice_is_forward_filled_across_the_real_extraction(
        self, parser: NistSsdfParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["PO.1.2"].parent_id == "PO.1"
        assert controls["PW.8.1"].parent_id == "PW.8"

    def test_both_declared_fragments_reach_their_task(
        self, parser: NistSsdfParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        for task_id, needle in (
            ("PS.1.1", "configuration-as-code"),
            ("PW.8.1", "which types of testing should be used."),
        ):
            metadata = controls[task_id].metadata
            assert metadata is not None
            assert any(needle in alt for alt in metadata["alt_ids"])

    def test_a_task_carrying_no_alternate_declares_none(
        self, parser: NistSsdfParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert "alt_ids" not in (controls["PO.1.1"].metadata or {})

    def test_a_short_table_is_refused(self, parser: NistSsdfParser) -> None:
        """The band would accept 38 of 42. The cell count does not."""
        parser.expected_task_cells = 6  # type: ignore[misc]
        with pytest.raises(ValueError, match="task cell"):
            parser.parse()

    def test_a_stub_that_stopped_being_recognised_is_refused(
        self, parser: NistSsdfParser,
    ) -> None:
        parser.expected_redirects = 2  # type: ignore[misc]
        with pytest.raises(ValueError, match="redirect stub"):
            parser.parse()

    def test_a_moved_practice_column_is_refused(
        self, parser: NistSsdfParser,
    ) -> None:
        parser.expected_practices = 4  # type: ignore[misc]
        with pytest.raises(ValueError, match="practice"):
            parser.parse()

    def test_a_pdf_with_no_wide_table_is_refused(
        self, parser: NistSsdfParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "raw" / SOURCE_FILE).write_bytes(
            build_pdf([[(72.0, 100.0, "PO.1.1: Not in a table.")]])
        )
        with pytest.raises(ValueError, match="no table of four or more"):
            parser.parse()

    def test_run_writes(self, parser: NistSsdfParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 4
        assert [s.path for s in output.source_files] == [SOURCE_FILE]
        assert (tmp_path / "audit" / "nist_ssdf.jsonl").exists()

    def test_run_is_byte_identical_on_a_second_pass(
        self, parser: NistSsdfParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        first = (tmp_path / "out" / "nist_ssdf.json").read_bytes()
        parser.run()
        assert (tmp_path / "out" / "nist_ssdf.json").read_bytes() == first


class TestDescriptionBudget:
    """Ruling R14, and the positive control that proves the trap is real."""

    def test_the_trap_is_real(self) -> None:
        """Without this, the guard below asserts nothing.

        `_sanitize_control` calls `sanitize_text(description, return_full=True)`
        and assigns the second return value to `full_text`, discarding whatever
        the parser wrote there. A non-None second value is exactly that
        condition.
        """
        _, overflow = sanitize_text(
            "word " * 500, max_length=DESCRIPTION_MAX_LENGTH, return_full=True,
        )
        assert overflow is not None

    def test_a_statement_at_the_limit_is_refused(self) -> None:
        control = Control(
            control_id="PO.1.1",
            title="PO.1.1",
            description="w" * DESCRIPTION_MAX_LENGTH,
        )
        with pytest.raises(ValueError, match="BaseParser._sanitize_control"):
            NistSsdfParser._check_description_budget([control])

    def test_a_statement_one_character_under_the_limit_passes(self) -> None:
        """The guard refuses at the limit, so this is the boundary below it."""
        control = Control(
            control_id="PO.1.1",
            title="PO.1.1",
            description="w" * (DESCRIPTION_MAX_LENGTH - 1),
        )
        NistSsdfParser._check_description_budget([control])


class TestDigestGate:
    def test_a_different_pdf_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(_table_pdf())
        parser = NistSsdfParser(raw_dir=raw, output_dir=tmp_path)
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
            if source.framework_id == "nist_ssdf"
        }
        assert pins == {NistSsdfParser.expected_sha256}


class TestSyntheticPdfHelper:
    """The fixture generator Tasks 11 and 13 also read."""

    def test_mismatched_page_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="index together"):
            build_pdf([[], []], [[]])

    def test_a_character_the_encoding_cannot_spell_is_refused(self) -> None:
        """Silently substituting it would make the fixture differ from the
        string the test declared."""
        with pytest.raises(ValueError, match="U\\+4E00"):
            build_pdf([[(72.0, 100.0, "a 一 b")]])

    def test_punctuation_specifications_use_round_trips(self) -> None:
        payload = build_pdf(
            [[(72.0, 100.0, "an en dash – and a quote ’ survive")]]
        )
        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            text = pdf.pages[0].extract_text()
        assert "–" in text
        assert "’" in text

    def test_parentheses_and_backslashes_survive(self) -> None:
        payload = build_pdf([[(72.0, 100.0, r"a (b) c \ d")]])
        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            text = pdf.pages[0].extract_text()
        assert "(b)" in text
        assert "\\" in text


class TestShippedArtifact:
    """Reads only tracked files, so it holds where data/raw is absent."""

    def test_the_shape_the_parser_declares(self) -> None:
        shipped = _shipped()
        assert len(shipped) == 42
        assert len({c["parent_id"] for c in shipped.values()}) == 19
        assert all(c["title"] == c["control_id"] for c in shipped.values())

    def test_no_description_reaches_the_length_that_rewrites_full_text(
        self,
    ) -> None:
        """Ruling R14. It does not fire here: the longest is 333."""
        lengths = [len(str(c["description"])) for c in _shipped().values()]
        assert max(lengths) == 333
        assert max(lengths) < DESCRIPTION_MAX_LENGTH
        assert not any("full_text" in c for c in _shipped().values())

    def test_the_prose_floor_is_cleared_with_one_short_statement(self) -> None:
        shipped = _shipped()
        short = [
            control_id for control_id, c in shipped.items()
            if len(str(c["description"])) < HONEST_PROSE_MIN_CHARS
        ]
        assert short == ["RV.2.2"]
        assert NistSsdfParser.min_prose_fraction <= (
            (len(shipped) - len(short)) / len(shipped)
        )

    def test_the_anchors_share_no_leading_prefix(self) -> None:
        """Ruling R13. Zero, so nothing is stripped.

        Task 5's anchors opened on 364 byte-identical characters of a
        statistics table, 17% of the budget. A shared header appearing upstream
        fails here.
        """
        anchors = [
            prepare_anchor(str(c["description"]))[0]
            for c in _shipped().values()
        ]
        assert _shared_prefix(anchors) == 0
        assert len(set(anchors)) == 42
        assert max(len(a) for a in anchors) < MAX_ANCHOR_CHARS

    def test_the_five_retired_task_numbers_are_recorded(self) -> None:
        first = _shipped()["PO.1.1"]
        retired = first["metadata"]["retired_tasks"]  # type: ignore[index]
        assert [str(entry).split(":")[0] for entry in retired] == [
            "PW.3.1", "PW.3.2", "PW.4.3", "PW.4.5", "PW.5.2",
        ]

    def test_the_join_is_forty_six_of_forty_six_through_the_id_channel(
        self, tmp_path: Path,
    ) -> None:
        """Every value is measured and every one can move.

        by_id is 46 only because MALFORMED_SECTION_IDS declares the two
        fragments. Drop the table and it is 44, resolution_rate 0.9565, under
        the derived floor of 1.00.

        Built from this framework's own tracked artifact rather than from the
        merged corpus. Ruling R15 keeps data/processed/all_controls.json out of
        a parser task's commit, so a test that read it would assert the state
        of a shared file this commit does not carry. Every column below is
        per-framework, so the two corpora produce the same row.
        """
        row = _join(tmp_path).by_id("nist_ssdf")
        assert row.links == 46
        assert row.by_title == 0
        assert row.by_id == 46
        assert row.unresolved == 0
        assert row.resolution_rate == 1.0
        assert row.resolution_rate >= JOIN_FLOORS["nist_ssdf"]
        assert row.distinct_anchors == 42
        assert row.distinct_anchors_pre_truncation == 42
        assert row.truncated == 0
        assert row.distinct_hubs == 28
        assert row.dropped_by_prose_rule == 0
        assert row.anchor_source_full_text == 0
        assert row.anchor_source_description == 46
        assert row.anchor_source_title == 0
        assert row.anchor_source_synthetic == 0
        assert row.fallback_anchors == 0

    def test_detector_b_is_inapplicable_here_and_the_cause_still_holds(
        self, tmp_path: Path,
    ) -> None:
        """0 of 0 under ruling R19, where this read 44 of 44 before it.

        The saturation was never a wrong anchor. The link's `section_name` is
        the task STATEMENT while the control's title is the task ID, so detector
        B compared a 156-character sentence against "PO.1.1" and could only ever
        fire. `section_name` equals the resolved control's title for 0 of 46
        links, never rarely, which is the exact reading that retired B for dsomm
        under ruling R11.

        `name_level_mismatch_frameworks()` still cannot see it, and that is
        asserted below rather than assumed. Its criterion is
        distinct(section_id) / distinct(section_name), here 1:1 at 44 and 44, so
        1.0 sits between COARSE_NAME_RATIO and FINE_NAME_RATIO and neither R11
        nor R21 reaches it. R19 added a third derived predicate on the KIND of
        label rather than its granularity, and that one does reach it.

        The zero denominator is honest rather than convenient. Detectors A and C
        both still run and neither has anything to reach: `by_title` is 0, so A
        never enters its branch, and the 44 normalised ids are leaf tasks at one
        depth with no ancestor pair between them, so C finds no candidate. This
        fails in both directions. A title that started carrying real prose drops
        nist_ssdf out of the kind predicate and the declared set stops matching.
        A roll-up id entering the link file gives C a candidate and the
        denominator stops being 0.
        """
        report = _join(tmp_path)
        assert report.by_id("nist_ssdf").wrong_anchor_risk == 0
        assert wrong_anchor_applicable(report)["nist_ssdf"] == 0
        assert report.by_id("nist_ssdf").by_title == 0
        assert "nist_ssdf" not in JOIN_WRONG_ANCHOR_BUDGET
        assert "nist_ssdf" in DETECTOR_B_INAPPLICABLE
        # The count-based predicate is still blind to it, which is why R19 was
        # needed. Asserting the absence keeps the two criteria distinguishable.
        assert "nist_ssdf" not in name_level_mismatch_frameworks()
        assert "nist_ssdf" in name_kind_mismatch_frameworks()


class TestRun:
    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = NistSsdfParser(output_dir=tmp_path, audit_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 42
        assert len({c.parent_id for c in output.controls}) == 19
        assert [s.path for s in output.source_files] == [SOURCE_FILE]
        assert BaseParser.honest_prose_fraction(output.controls) == pytest.approx(
            41 / 42
        )
