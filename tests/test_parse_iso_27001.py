"""Tests for the ISO 27001 Annex A parser.

The fixture carries real damage: a hyphenation break, a run-together token,
and the 5.6 to 5.7 cell-bleed pair. Repairs are asserted on exact output, not
only on counts, because this parser moves text across control ids.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from parsers.parse_iso_27001 import Iso27001Parser
from tract.schema import Control

FIXTURE = Path(__file__).parent / "fixtures" / "iso_27001_sample.md"


@pytest.fixture
def parser(tmp_path: Path) -> Iso27001Parser:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ISO_IEC_27001_2022_en.md").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    return Iso27001Parser(raw_dir=raw, output_dir=out)


class TestIso27001Parser:
    def test_extracts_only_numbered_control_rows(self, parser: Iso27001Parser) -> None:
        controls = parser.parse()
        ids = [c.control_id for c in controls]
        # "5" is a section header, not a control, and must not appear.
        assert "5" not in ids
        assert ids == sorted(ids, key=lambda s: [int(p) for p in s.split(".")])
        assert all("." in cid for cid in ids)

    def test_strips_the_control_keyword_from_the_statement(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert not controls["5.1"].description.startswith("Control")

    def test_repairs_the_hyphenation_break_in_the_title(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5.1"].title == "Policies for information security"

    def test_moves_the_bleed_fragment_back_to_5_6(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5.6"].description.rstrip().endswith("associations.")
        assert not controls["5.7"].description.startswith("associations")

    def test_splits_the_run_together_token_in_5_10(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert "Rulesfortheacceptableuse" not in controls["5.10"].description
        assert "Rules for the acceptable use" in controls["5.10"].description

    def test_every_control_carries_a_statement_not_a_title(
        self, parser: Iso27001Parser,
    ) -> None:
        from tract.parsers.base import BaseParser

        assert BaseParser.honest_prose_fraction(parser.parse()) == 1.0


class TestFindResidualRunTogether:
    """Coverage for the residual-damage visibility gap flagged in review.

    MAX_RUN_TOGETHER_REPAIRS only bounds successful splits; these tests cover
    the separate scan that flags rows still carrying an unsplit token, using
    synthetic text so the assertion is not tied to which real ISO rows happen
    to be repairable this run.
    """

    def test_flags_a_control_with_an_unbroken_long_token(self) -> None:
        controls = [
            Control(
                control_id="9.9",
                title="Example",
                description=(
                    "Thisisarunontokenthatstaysjoined and the rest of the "
                    "sentence is ordinary prose."
                ),
            ),
        ]
        assert Iso27001Parser._find_residual_run_together(controls) == ["9.9"]

    def test_does_not_flag_clean_prose(self) -> None:
        controls = [
            Control(
                control_id="9.10",
                title="Example",
                description=(
                    "This control statement is fully repaired prose without "
                    "any unbroken long token."
                ),
            ),
        ]
        assert Iso27001Parser._find_residual_run_together(controls) == []

    def test_logs_a_warning_with_the_control_ids_when_residual_damage_remains(
        self, parser: Iso27001Parser, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The real fixture rows are fully repairable at 93-row vocabulary
        # scale but not necessarily at the small fixture's scale; parse()
        # must not crash either way, and if residual damage is present it
        # must be visible in the log, not merely counted internally.
        with caplog.at_level(logging.WARNING, logger="parsers.parse_iso_27001"):
            controls = parser.parse()
        residual = Iso27001Parser._find_residual_run_together(controls)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        if residual:
            assert any(
                "unsplit run-together token" in r.getMessage() for r in warnings
            )
        else:
            assert not any(
                "unsplit run-together token" in r.getMessage() for r in warnings
            )
