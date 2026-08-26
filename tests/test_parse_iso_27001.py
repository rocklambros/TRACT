"""Tests for the ISO 27001 Annex A parser.

The fixture is a synthetic Annex-A-shaped table for a fictional ACME Secure
Systems Handbook, not the real ISO source. It reproduces the same damage
classes the real conversion carries: a hyphenation break, a run-together
token, and a cell-bleed pair. Repairs are asserted on exact output, not only
on counts, because this parser moves text across control ids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import pytest

from parsers.parse_iso_27001 import Iso27001Parser
from tract.schema import Control

FIXTURE = Path(__file__).parent / "fixtures" / "iso_27001_sample.md"


class SampleIso27001Parser(Iso27001Parser):
    """The parser with the fixture's repair counts rather than the source's.

    Repair expectations are exact and two-sided, so a synthetic 13-row table
    carries its own measured counts rather than the real source's. Declaring
    what this input contains keeps the real gate exact instead of loosening
    it to a range wide enough to cover both.
    """

    expected_repairs: ClassVar[dict[str, int]] = {
        "cell bleed": 4, "hyphen break": 2, "run-together": 1,
    }
    expected_residual_damage: ClassVar[int] = 1


@pytest.fixture
def parser(tmp_path: Path) -> Iso27001Parser:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ISO_IEC_27001_2022_en.md").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    # audit_dir is redirected into tmp_path so the test never writes into the
    # repository's data tree.
    return SampleIso27001Parser(
        raw_dir=raw, output_dir=out, audit_dir=tmp_path / "audit",
    )


class TestRepairExpectationsAreTwoSided:
    """A ceiling only catches a repair that runs away.

    The failure that ships bad text quietly is the opposite one: a source
    refresh moves the damage, the repair stops reaching it, and the output is
    truncated with every gate green. The run-together ceiling sat at 30
    against an actual 8 and could never have fired at all.
    """

    def test_a_repair_that_stops_firing_is_a_failure(
        self, tmp_path: Path,
    ) -> None:
        class Expecting(SampleIso27001Parser):
            expected_repairs: ClassVar[dict[str, int]] = {
                "cell bleed": 4, "hyphen break": 3, "run-together": 1,
            }

        with pytest.raises(ValueError, match="hyphen break"):
            self._parse_with(Expecting, tmp_path)

    def test_a_repair_that_fires_more_often_than_measured_is_a_failure(
        self, tmp_path: Path,
    ) -> None:
        class Expecting(SampleIso27001Parser):
            expected_repairs: ClassVar[dict[str, int]] = {
                "cell bleed": 4, "hyphen break": 1, "run-together": 1,
            }

        with pytest.raises(ValueError, match="hyphen break"):
            self._parse_with(Expecting, tmp_path)

    def test_residual_damage_is_declared_and_checked_too(
        self, tmp_path: Path,
    ) -> None:
        """Damage the repair never reached is a number, not a log line."""
        class Expecting(SampleIso27001Parser):
            expected_residual_damage: ClassVar[int] = 0

        with pytest.raises(ValueError, match="residual"):
            self._parse_with(Expecting, tmp_path)

    def test_the_documented_opt_out_permits_the_deviation(
        self, tmp_path: Path,
    ) -> None:
        class Drifting(SampleIso27001Parser):
            expected_repairs: ClassVar[dict[str, int]] = {
                "cell bleed": 4, "hyphen break": 3, "run-together": 1,
            }
            repair_deviation_reason: ClassVar[str] = (
                "the 2027 conversion fixed one hyphen break at source"
            )

        assert len(self._parse_with(Drifting, tmp_path)) == 12

    @staticmethod
    def _parse_with(
        parser_class: type[Iso27001Parser], tmp_path: Path,
    ) -> list[Control]:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "ISO_IEC_27001_2022_en.md").write_text(
            FIXTURE.read_text(encoding="utf-8"), encoding="utf-8",
        )
        out = tmp_path / "out"
        out.mkdir()
        return parser_class(
            raw_dir=raw, output_dir=out, audit_dir=tmp_path / "audit",
        ).parse()


class TestIso27001Parser:
    def test_extracts_only_numbered_control_rows(self, parser: Iso27001Parser) -> None:
        controls = parser.parse()
        ids = [c.control_id for c in controls]
        # "5" is a section header, not a control, and must not appear.
        assert "5" not in ids
        assert all("." in cid for cid in ids)
        # No ordering assertion. The parser emits rows in source order and
        # does not sort, so asserting sorted output would test the fixture's
        # layout rather than anything the parser does.

    def test_strips_the_control_keyword_from_the_statement(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert not controls["5.1"].description.startswith("Control")

    def test_repairs_the_hyphenation_break_in_the_title(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5.1"].title == "Badge issuance for facility security"

    def test_moves_the_bleed_fragment_back_to_5_6(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5.6"].description.rstrip().endswith("coordinators.")
        assert not controls["5.7"].description.startswith("coordinators")

    @pytest.mark.parametrize(("control_id", "expected"), [
        ("5.6", "Facility security staff shall keep an up to date roster of "
                "regional response teams and other specialist facility "
                "security coordinators."),
        ("5.7", "Hazard awareness information shall be gathered and reviewed "
                "to produce a facility hazard bulletin."),
        ("5.17", "Issuance and management of credential information shall be "
                 "controlled through a governance process, including "
                 "guidance to personnel on appropriate handling of "
                 "credential information."),
        ("5.18", "Facility access rights to premises and other associated "
                 "assets shall be granted, reviewed, modified and revoked in "
                 "accordance with the organization's role-based policy on "
                 "and rules for access control."),
        # The damaged pair. 7.5's expected output carries the elision marker
        # where the conversion lost a clause, NOT the fluent join that reads
        # "such as flood restoration infrastructure shall be designed and
        # implemented."
        ("7.5", "Protection against physical andenvironmental hazards, such "
                "as flood [...] restoration infrastructure shall be "
                "designed and implemented."),
        ("7.6", "Physical safeguards for restricted zones shall be designed "
                "and implemented."),
        ("7.8", "Devices shall be sited securely and protected."),
        ("7.9", "Offsite devices shall be protected."),
    ])
    def test_every_bleed_pair_matches_hand_checked_output(
        self, parser: Iso27001Parser, control_id: str, expected: str,
    ) -> None:
        """All four pairs, both sides, pinned to exact text.

        This repair reattributes a compliance statement from one control id to
        another, so a count proves nothing. Only one pair was pinned before,
        and the pair that was wrong was not the pinned one.
        """
        controls = {c.control_id: c for c in parser.parse()}
        assert controls[control_id].description == expected

    def test_splits_the_run_together_token_in_5_10(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert "Rulesfortheacceptableuse" not in controls["5.10"].description
        assert "Rules for the acceptable use" in controls["5.10"].description

    def test_every_control_carries_a_statement_not_a_title(
        self, parser: Iso27001Parser,
    ) -> None:
        """The floor exists to catch a regression to title-only extraction.

        Length alone is not the invariant. Three Annex A statements are
        genuinely one short sentence, and the fixture carries two of them, so
        a bare fraction assertion here would encode the fixture's shape rather
        than the parser's contract. What must hold for every row is that the
        statement is not a copy of the title, and that the only rows under the
        prose bar are the two known-short ones.
        """
        from tract.config import HONEST_PROSE_MIN_CHARS

        controls = parser.parse()
        for control in controls:
            assert control.description.strip()
            assert control.description.strip() != control.title.strip()

        short = {
            c.control_id for c in controls
            if len(c.description.strip()) < HONEST_PROSE_MIN_CHARS
        }
        assert short == {"7.8", "7.9"}


POISONING_FIXTURE = (
    Path(__file__).parent / "fixtures" / "vocabulary_poisoning_sample.md"
)


@pytest.fixture
def poisoning_parser(tmp_path: Path) -> Iso27001Parser:
    """Parser over a synthetic table carrying both vocabulary poisons.

    Synthetic rather than more ISO rows on purpose. The mechanic under test is
    how the parser builds its vocabulary, not anything specific to Annex A,
    and this repository is CC0 so every tracked line of the real standard is a
    rights claim the project would rather not make twice.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ISO_IEC_27001_2022_en.md").write_text(
        POISONING_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    return PoisoningIso27001Parser(
        raw_dir=raw, output_dir=out, audit_dir=tmp_path / "audit",
    )


class PoisoningIso27001Parser(Iso27001Parser):
    """The parser with the synthetic fixture's measured repair counts."""

    expected_repairs: ClassVar[dict[str, int]] = {
        "cell bleed": 0, "hyphen break": 1, "run-together": 3,
    }
    expected_residual_damage: ClassVar[int] = 0


class TestVocabularyIsNotPoisoned:
    """Two ways a broken word becomes the splitter's preferred answer."""

    def test_hyphen_fragments_do_not_enter_the_vocabulary(
        self, poisoning_parser: Iso27001Parser,
    ) -> None:
        """The vocabulary must be built after the hyphen repair, not before.

        "secu - rity" contributes "secu" and "rity" when the vocabulary is
        built first, and the splitter then prefers that pair over the whole
        word it never saw. The row that needed the repair most defeats it.
        """
        controls = {c.control_id: c for c in poisoning_parser.parse()}

        assert "information security policy" in controls["9.2"].description
        assert "secu rity" not in controls["9.2"].description

    def test_a_joined_title_is_split_too(
        self, poisoning_parser: Iso27001Parser,
    ) -> None:
        """The splitter used to run on bodies only.

        The title is what OpenCRE joins a link on, so a title shipping as
        "Addressinginformationsecurity" is a link that cannot resolve no
        matter how clean the body is.
        """
        controls = {c.control_id: c for c in poisoning_parser.parse()}

        assert controls["9.2"].title == "The information security policy"

    def test_a_joined_token_does_not_become_a_preferred_word(
        self, poisoning_parser: Iso27001Parser,
    ) -> None:
        """"andprocedures" is 13 characters, under the 20-char vocabulary cut.

        Left in, the fewest-segments search prefers it over "and" plus
        "procedures" because it is one segment rather than two, and the repair
        reports success while emitting corrupt text.
        """
        controls = {c.control_id: c for c in poisoning_parser.parse()}

        assert "acceptable use and procedures" in controls["9.4"].description
        assert "andprocedures" not in controls["9.4"].description


class TestDamagedControls:
    """7.5 lost a clause in PDF conversion and no transform can recover it.

    The fixture reads "such as flood" and stops; the next row opens with
    "restoration infrastructure shall be designed and implemented." The
    clause between them is absent from the source, so the unguarded join
    would produce a fluent, wrong requirement that cleared every gate.
    Writing the missing clause from memory would invent normative text,
    which is worse.
    """

    def test_the_damaged_control_is_marked_rather_than_silently_joined(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        damaged = controls["7.5"]

        assert damaged.metadata is not None
        assert damaged.metadata["damaged"] == "true"
        assert "unrecoverable" in damaged.metadata["damage_reason"]

    def test_the_damaged_statement_shows_the_gap_instead_of_reading_fluently(
        self, parser: Iso27001Parser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        description = controls["7.5"].description

        assert "[...]" in description
        # The fabricated join. Emitting it asserts a requirement no standard
        # contains, with nothing in the record to say so.
        assert "such as flood restoration" not in description

    def test_the_successor_keeps_only_its_own_statement(
        self, parser: Iso27001Parser,
    ) -> None:
        """Marking 7.5 damaged must not leave 7.6 carrying foreign text."""
        controls = {c.control_id: c for c in parser.parse()}

        assert controls["7.6"].description == (
            "Physical safeguards for restricted zones shall be designed "
            "and implemented."
        )

    def test_a_damaged_control_does_not_count_toward_the_prose_fraction(
        self, parser: Iso27001Parser,
    ) -> None:
        from tract.parsers.base import BaseParser

        controls = parser.parse()
        damaged = [c for c in controls if BaseParser.is_damaged(c)]

        assert [c.control_id for c in damaged] == ["7.5"]
        assert BaseParser.honest_prose_fraction(controls) == \
            BaseParser.honest_prose_fraction(
                [c for c in controls if not BaseParser.is_damaged(c)]
            )


class TestRepairAuditFile:
    """The audit file three documents promised and no function wrote."""

    def test_parse_writes_one_record_per_bleed_decision(
        self, parser: Iso27001Parser, tmp_path: Path,
    ) -> None:
        import json

        parser.parse()
        path = tmp_path / "audit" / "iso_27001.jsonl"
        records = [json.loads(line) for line in path.read_text(
            encoding="utf-8").splitlines()]

        pairs = [(r["predecessor_id"], r["successor_id"]) for r in records]
        assert pairs == [("5.6", "5.7"), ("5.17", "5.18"), ("7.5", "7.6"),
                         ("7.8", "7.9")]
        assert all("fragment" in r for r in records)
        assert all("predecessor_before" in r for r in records)
        assert all("predecessor_after" in r for r in records)

    def test_the_damaged_join_is_flagged_in_the_audit_record(
        self, parser: Iso27001Parser, tmp_path: Path,
    ) -> None:
        import json

        parser.parse()
        path = tmp_path / "audit" / "iso_27001.jsonl"
        records = {
            json.loads(line)["predecessor_id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        }

        assert records["7.5"]["known_damaged"] is True
        assert records["5.6"]["known_damaged"] is False


class TestFindResidualRunTogether:
    """The scan that counts damage no repair reached.

    split_run_together fails closed, so a row it cannot fully segment is left
    alone. The repair counts say how many rows it fixed and nothing about how
    many still need fixing, which is what this scan supplies.
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

    def test_the_remaining_control_ids_are_named_in_the_log(
        self, parser: Iso27001Parser, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A count alone does not say which rows to go and read."""
        with caplog.at_level(logging.INFO, logger="parsers.parse_iso_27001"):
            controls = parser.parse()

        residual = Iso27001Parser._find_residual_run_together(controls)
        assert residual == ["5.10"]
        assert any(
            "unsplit run-together token" in record.getMessage()
            and "5.10" in record.getMessage()
            for record in caplog.records
        )
