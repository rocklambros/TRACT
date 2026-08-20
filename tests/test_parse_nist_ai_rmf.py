"""Tests for parsers/parse_nist_ai_rmf.py.

The parser rejoins a sentence the source's PDF-to-markdown converter split at a
hard line wrap. These tests pin the repaired shape from both directions: the
split signature must be gone, and the scan that removes it must refuse to walk
past a cell boundary rather than assembling a statement out of page furniture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_nist_ai_rmf import NistAiRmfParser
from tract.config import HONEST_PROSE_MIN_CHARS
from tract.parsers.base import BaseParser
from tract.schema import Control

# Two table cells, wrapped the way the converter wraps them, followed by the
# page furniture it emits between table pages. Small enough to read, and it
# carries every structure the block scan has to tell apart.
SAMPLE = """Table 1: Categories and subcategories for the **GOVERN** function.


**Categories** **Subcategories**


**GOVERN 1:**
Policies, processes,
procedures, and
practices across the
organization.


**GOVERN 1.1:** Legal and regulatory requirements involving AI
are understood, managed, and documented.


**GOVERN 1.2:** The characteristics of trustworthy AI are integrated
into organizational policies, procedures, and

practices.


Continued on next page


Page 22


NIST AI 100-1 AI RMF 1.0
"""

# The same source with GOVERN 1.2's closing block removed, so the cell never
# closes on a sentence and the next block is page furniture.
UNTERMINATED = SAMPLE.replace("\n\npractices.\n", "\n")

# An unterminated cell whose next block ends on a period. Page furniture that
# happens not to end in a period stops the scan by running out of input, so it
# does not exercise BLOCK_STOP at all. A category cell does: without the guard
# the scan absorbs it, closes on its period and ships a statement assembled
# from two table columns with nothing to say so.
UNTERMINATED_BEFORE_CATEGORY = """**GOVERN 1.1:** Legal and regulatory requirements involving AI
are understood, managed, and documented.


**GOVERN 1.2:** The characteristics of trustworthy AI are integrated
into organizational policies, procedures, and


**GOVERN 2:**
Accountability structures are in place.
"""

# An unterminated cell separated from the next text by more than one blank
# line. One blank line inside a cell is a converter artifact the parser
# absorbs. Two is a gap wide enough that the parser refuses rather than
# assembling across it.
UNTERMINATED_ACROSS_A_WIDE_GAP = """**GOVERN 1.1:** Legal and regulatory requirements involving AI
are understood, managed, and documented.


**GOVERN 1.2:** The characteristics of trustworthy AI are integrated
into organizational policies, procedures, and




practices and documented operating procedures.
"""

# One cell whose statement fits on the marker's own line, so nothing is
# rejoined and the audit has to be able to stay empty.
UNWRAPPED = (
    "**GOVERN 1.1:** Legal and regulatory requirements involving AI are "
    "understood, managed, and documented.\n"
)


class SampleRmfParser(NistAiRmfParser):
    """The parser with the sample's count rather than the full source's.

    run()'s count gate stays real. The sample declares what it contains.
    """

    expected_count: ClassVar[int] = 2


class OneCellRmfParser(NistAiRmfParser):
    """The parser against a source holding a single unwrapped cell."""

    expected_count: ClassVar[int] = 1


def _build(
    tmp_path: Path,
    body: str,
    parser_cls: type[NistAiRmfParser] = SampleRmfParser,
) -> NistAiRmfParser:
    """A parser reading `body`, writing everything under tmp_path."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "nist_ai_rmf_1.0.md").write_text(body, encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return parser_cls(
        raw_dir=raw_dir, output_dir=out_dir, audit_dir=tmp_path / "audit",
    )


def _audit(tmp_path: Path) -> list[dict[str, object]]:
    text = (tmp_path / "audit" / "nist_ai_rmf.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


class TestTheWrappedSentenceIsRejoined:
    def test_a_cell_becomes_one_statement(self, tmp_path: Path) -> None:
        controls = _build(tmp_path, SAMPLE).run().controls

        assert [c.control_id for c in controls] == ["GOVERN 1.1", "GOVERN 1.2"]
        assert controls[0].description == (
            "Legal and regulatory requirements involving AI are understood, "
            "managed, and documented."
        )

    def test_a_blank_line_inside_a_cell_does_not_end_the_statement(
        self, tmp_path: Path,
    ) -> None:
        """GOVERN 1.2 in the sample carries the defect MEASURE 2.12 carries.

        The converter left a blank line mid-sentence. A block scan that stops
        at the first blank line loses the last word of the statement.
        """
        controls = _build(tmp_path, SAMPLE).run().controls

        assert controls[1].description == (
            "The characteristics of trustworthy AI are integrated into "
            "organizational policies, procedures, and practices."
        )

    def test_page_furniture_never_reaches_a_statement(
        self, tmp_path: Path,
    ) -> None:
        controls = _build(tmp_path, SAMPLE).run().controls

        for control in controls:
            for furniture in (
                "Continued on next page", "Page 22", "NIST AI 100-1",
                "Categories", "Table 1:",
            ):
                assert furniture not in control.description

    def test_the_category_cell_never_reaches_a_statement(
        self, tmp_path: Path,
    ) -> None:
        """The category is the table's left column, not part of any cell."""
        controls = _build(tmp_path, SAMPLE).run().controls

        assert "Policies, processes" not in controls[1].description

    def test_the_title_is_the_identifier(self, tmp_path: Path) -> None:
        """The source names no title, so a title here would be a truncation."""
        for control in _build(tmp_path, SAMPLE).run().controls:
            assert control.title == control.control_id

    def test_a_broken_word_rejoins_without_a_space(
        self, tmp_path: Path,
    ) -> None:
        """The converter can hyphenate at the wrap. A space would split it."""
        body = SAMPLE.replace(
            "requirements involving AI\nare understood",
            "requirements involv-\ning AI are understood",
        )

        controls = _build(tmp_path, body).run().controls

        assert "involving AI are understood" in controls[0].description


class TestTheScanRefusesWhatItCannotRead:
    def test_a_cell_that_never_closes_raises(self, tmp_path: Path) -> None:
        parser = _build(tmp_path, UNTERMINATED)

        with pytest.raises(ValueError, match=r"GOVERN 1\.2"):
            parser.run()

    def test_the_refusal_names_what_it_read(self, tmp_path: Path) -> None:
        """A message without the partial statement cannot be acted on."""
        parser = _build(tmp_path, UNTERMINATED)

        with pytest.raises(ValueError) as caught:
            parser.run()

        assert "The characteristics of trustworthy AI" in str(caught.value)

    def test_a_category_cell_is_never_absorbed_to_close_a_statement(
        self, tmp_path: Path,
    ) -> None:
        """The scan must refuse a new cell even when absorbing it would work.

        Absorbing this one produces a grammatical sentence that closes on a
        period and clears every count and prose gate, so nothing downstream
        would report it.
        """
        parser = _build(tmp_path, UNTERMINATED_BEFORE_CATEGORY)

        with pytest.raises(ValueError, match=r"GOVERN 1\.2"):
            parser.run()

    def test_a_statement_is_not_assembled_across_a_wide_gap(
        self, tmp_path: Path,
    ) -> None:
        """One blank line inside a cell is an artifact. Two is a boundary."""
        parser = _build(tmp_path, UNTERMINATED_ACROSS_A_WIDE_GAP)

        with pytest.raises(ValueError, match=r"GOVERN 1\.2"):
            parser.run()

    def test_a_source_with_no_marker_raises(self, tmp_path: Path) -> None:
        parser = _build(tmp_path, "Page 22\n\n\nNIST AI 100-1 AI RMF 1.0\n")

        with pytest.raises(ValueError, match="no subcategory marker"):
            parser.run()

    def test_a_mid_sentence_mention_is_not_a_marker(
        self, tmp_path: Path,
    ) -> None:
        """The marker opens a table cell. Narrative prose names them too."""
        body = SAMPLE.replace(
            "Page 22", "Users should read **GOVERN 1.9:** before proceeding.",
        )

        controls = _build(tmp_path, body).run().controls

        assert [c.control_id for c in controls] == ["GOVERN 1.1", "GOVERN 1.2"]


class TestTheRepairAuditCarriesText:
    def test_each_rejoin_records_both_halves_and_the_result(
        self, tmp_path: Path,
    ) -> None:
        _build(tmp_path, SAMPLE).run()
        records = _audit(tmp_path)

        assert [r["control_id"] for r in records] == ["GOVERN 1.1", "GOVERN 1.2"]
        for record in records:
            assert record["repair"] == "line_wrapped_statement_rejoined"
            # Text on both sides, not lengths. A count says a repair fired,
            # not which fragment landed on which control.
            after = record["after"]
            before = record["before"]
            assert isinstance(after, str)
            assert isinstance(before, list)
            assert all(isinstance(half, str) for half in before)
            assert len(before) == 2
            assert " ".join(" ".join(before).split()) == after

    def test_the_first_half_is_what_the_old_parser_stored_as_a_title(
        self, tmp_path: Path,
    ) -> None:
        _build(tmp_path, SAMPLE).run()

        assert _audit(tmp_path)[0]["before"] == [
            "Legal and regulatory requirements involving AI",
            "are understood, managed, and documented.",
        ]

    def test_the_record_says_how_many_blocks_the_cell_spanned(
        self, tmp_path: Path,
    ) -> None:
        """The count separates a plain wrap from a blank line inside a cell."""
        _build(tmp_path, SAMPLE).run()

        assert [r["source_blocks"] for r in _audit(tmp_path)] == [1, 2]

    def test_an_unwrapped_cell_writes_no_record(self, tmp_path: Path) -> None:
        """The audit must be able to stay empty, or it records nothing."""
        _build(tmp_path, UNWRAPPED, OneCellRmfParser).run()

        assert _audit(tmp_path) == []


class TestTheRealSource:
    @pytest.fixture()
    def controls(self, tmp_path: Path) -> list[Control]:
        try:
            raw_dir = NistAiRmfParser.resolve_raw_dir()
        except FileNotFoundError:
            pytest.skip("Raw data not available")
        if not (raw_dir / "nist_ai_rmf_1.0.md").exists():
            pytest.skip("Raw data not available")
        return NistAiRmfParser(
            raw_dir=raw_dir, audit_dir=tmp_path / "audit",
        ).parse()

    def test_every_subcategory_is_present(
        self, controls: list[Control],
    ) -> None:
        assert len(controls) == 72
        # The declaration too, not only the parse. run()'s count gate reads the
        # declaration, and nothing else in this module exercises it.
        assert NistAiRmfParser.expected_count == len(controls)
        assert {c.metadata["function"] for c in controls if c.metadata} == {
            "GOVERN", "MAP", "MEASURE", "MANAGE",
        }

    def test_no_statement_opens_on_a_lowercase_continuation(
        self, controls: list[Control],
    ) -> None:
        """The split signature. 67 of 72 carried it before the repair.

        [measured 2026-08-19]
        """
        opened_lowercase = [
            c.control_id for c in controls if c.description[:1].islower()
        ]

        assert opened_lowercase == []

    def test_every_statement_closes_on_a_sentence(
        self, controls: list[Control],
    ) -> None:
        unclosed = [
            c.control_id for c in controls if not c.description.endswith(".")
        ]

        assert unclosed == []

    def test_no_statement_carries_page_furniture(
        self, controls: list[Control],
    ) -> None:
        """GOVERN 1.4 shipped four furniture blocks before the repair."""
        for control in controls:
            for furniture in (
                "Continued on next page", "NIST AI 100-1", "**Categories**",
                "Table 1:", "Table 2:", "Table 3:", "Table 4:",
            ):
                assert furniture not in control.description

    def test_no_statement_carries_a_neighbouring_category_cell(
        self, controls: list[Control],
    ) -> None:
        """MEASURE 2.13 carried the MEASURE 3 category text before the repair."""
        found = {c.control_id: c.description for c in controls}

        assert found["MEASURE 2.13"] == (
            "Effectiveness of the employed TEVV metrics and processes in the "
            "**MEASURE** function are evaluated and documented."
        )

    def test_the_two_cells_the_converter_split_are_whole(
        self, controls: list[Control],
    ) -> None:
        """MAP 1.4 and MEASURE 2.12 need a second block. The rest need one."""
        found = {c.control_id: c.description for c in controls}

        assert found["MEASURE 2.12"].endswith(
            "– as identified in the **MAP** function – are assessed "
            "and documented."
        )
        assert found["MAP 1.4"].endswith("re-evaluated.")

    def test_the_marker_keeps_its_closing_asterisks(
        self, controls: list[Control],
    ) -> None:
        """MEASURE 2.11's title ended 'the **MAP' before the repair."""
        found = {c.control_id: c.description for c in controls}

        assert found["MEASURE 2.11"] == (
            "Fairness and bias – as identified in the **MAP** function "
            "– are evaluated and results are documented."
        )

    def test_the_prose_floor_is_attained_with_one_character_of_margin(
        self, controls: list[Control],
    ) -> None:
        """The floor is 1.0 because 1.0 is what the source supports.

        MAP 1.5 is the control that decides it. Pinned here so a change in
        sanitisation or in HONEST_PROSE_MIN_CHARS names the control instead of
        surfacing as an unexplained floor failure inside run().
        """
        found = {c.control_id: c.description for c in controls}

        assert BaseParser.honest_prose_fraction(controls) == 1.0
        assert NistAiRmfParser.min_prose_fraction == 1.0
        assert len(found["MAP 1.5"]) - HONEST_PROSE_MIN_CHARS == 1

    def test_every_title_is_its_identifier(
        self, controls: list[Control],
    ) -> None:
        assert [c.title for c in controls] == [c.control_id for c in controls]

    def test_re_parsing_the_same_bytes_gives_the_same_controls(
        self, controls: list[Control], tmp_path: Path,
    ) -> None:
        again = NistAiRmfParser(
            raw_dir=NistAiRmfParser.resolve_raw_dir(),
            audit_dir=tmp_path / "again",
        ).parse()

        assert [c.model_dump() for c in again] == [
            c.model_dump() for c in controls
        ]
