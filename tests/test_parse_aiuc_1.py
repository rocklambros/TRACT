"""Tests for parsers/parse_aiuc_1.py.

The Q1 2026 update retired two controls and left their records in place as
withdrawal notices. The parser drops them. These tests pin that from both
directions: the notices must not ship, and every way the drop could remove
something real must raise instead.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest

from parsers.parse_aiuc_1 import Aiuc1Parser
from tract.parsers.base import BaseParser
from tract.schema import Control

FIXTURE = Path("tests/fixtures/aiuc_1_sample.json")


class SampleAiuc1Parser(Aiuc1Parser):
    """The parser with the fixture's count rather than the full source's.

    The fixture holds 2 of the 130 controls. run()'s count gate is real and
    must stay real, so the test declares what this input contains instead of
    asking the gate to look the other way. count_deviation_reason exists for a
    source that genuinely changed, not for a test that feeds a sample.

    min_prose_fraction is deliberately NOT overridden. It states a property of
    the text rather than of the sample size, so the fixture has to carry
    activity statements as long as the real ones. Both fixture activities are
    verbatim source text for that reason. Shortening one puts the fixture below
    the 0.84 floor, and the right response is to restore the text rather than
    to relax the parser.
    """

    expected_count: ClassVar[int] = 2


class OneActivityParser(Aiuc1Parser):
    """The parser against a source that ships a single activity."""

    expected_count: ClassVar[int] = 1


def _source() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload


def _retired_control(
    control_id: str, statement: str, activity_statement: str,
) -> dict[str, Any]:
    """A withdrawal notice shaped the way the publisher writes them."""
    return {
        "id": control_id,
        "title": "Document system change approvals",
        "url": f"https://www.aiuc-1.com/controls/{control_id.lower()}",
        "classification": "Optional",
        "type": "Detective",
        "frequency": "Every 12 months",
        "description": statement,
        "applicable_capabilities": ["Universal"],
        "activities": [{
            "id": f"{control_id}.1",
            "description": activity_statement,
            "category": "Core",
            "evidence_types": [],
        }],
        "keywords": [],
        "framework_references": [],
    }


def _build(
    tmp_path: Path,
    payload: dict[str, Any],
    parser_cls: type[Aiuc1Parser] = SampleAiuc1Parser,
) -> Aiuc1Parser:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "aiuc-1-standard.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    return parser_cls(
        raw_dir=raw_dir, output_dir=out_dir, audit_dir=tmp_path / "audit",
    )


def _audit(tmp_path: Path) -> list[dict[str, Any]]:
    text = (tmp_path / "audit" / "aiuc_1.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_parses_sample_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(FIXTURE, raw_dir / "aiuc-1-standard.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = SampleAiuc1Parser(
        raw_dir=raw_dir, output_dir=out_dir, audit_dir=tmp_path / "audit",
    )
    result = parser.run()

    assert result.framework_id == "aiuc_1"
    assert len(result.controls) == 2
    assert result.controls[0].control_id == "A001.1"
    assert result.controls[0].parent_id == "A001"
    assert result.controls[0].parent_name == "Establish input data policy"
    assert result.controls[0].hierarchy_level == "activity"
    assert result.controls[0].metadata is not None
    assert result.controls[0].metadata["category"] == "Core"
    assert result.controls[0].metadata["domain"] == "Data & Privacy"


def test_the_artifact_records_the_bytes_it_was_built_from(tmp_path: Path) -> None:
    """The static coverage test proves the call exists; this proves it fires.

    A parser can import read_source and still take a different path at
    runtime, and the artifact would carry an empty source_files list with
    nothing to say so.
    """
    import hashlib

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(FIXTURE, raw_dir / "aiuc-1-standard.json")
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    result = SampleAiuc1Parser(
        raw_dir=raw_dir, output_dir=out_dir, audit_dir=tmp_path / "audit",
    ).run()

    assert [s.path for s in result.source_files] == ["aiuc-1-standard.json"]
    payload = (raw_dir / "aiuc-1-standard.json").read_bytes()
    assert result.source_files[0].sha256 == hashlib.sha256(payload).hexdigest()
    assert result.source_files[0].bytes == len(payload)


class TestAWithdrawalNoticeDoesNotShip:
    def test_a_retired_control_contributes_no_activities(
        self, tmp_path: Path,
    ) -> None:
        payload = _source()
        payload["domains"][0]["controls"].append(_retired_control(
            "E007",
            "RETIRED - Merged with A001 at Q1 2026 update.",
            "RETIRED - merged into A001.",
        ))

        controls = _build(tmp_path, payload).run().controls

        assert [c.control_id for c in controls] == ["A001.1", "A001.2"]

    def test_the_drop_records_the_notice_and_its_successor(
        self, tmp_path: Path,
    ) -> None:
        payload = _source()
        payload["domains"][0]["controls"].append(_retired_control(
            "E007",
            "RETIRED - Merged with A001 at Q1 2026 update.",
            "RETIRED - merged into A001.",
        ))

        _build(tmp_path, payload).run()
        records = _audit(tmp_path)

        assert len(records) == 1
        record = records[0]
        assert record["control_id"] == "E007.1"
        assert record["repair"] == "retired_activity_dropped"
        # Text on both sides, not lengths.
        assert record["before"] == "RETIRED - merged into A001."
        assert record["after"] == ""
        assert record["successor_id"] == "A001"
        assert record["successor_statement"] == (
            "Ensure policies for input data usage."
        )
        assert record["parent_statement"] == (
            "RETIRED - Merged with A001 at Q1 2026 update."
        )

    def test_a_source_with_no_withdrawals_writes_an_empty_audit(
        self, tmp_path: Path,
    ) -> None:
        """The audit must be able to stay empty, or it records nothing."""
        _build(tmp_path, _source()).run()

        assert _audit(tmp_path) == []

    def test_the_marker_is_anchored_at_the_head_of_the_statement(
        self, tmp_path: Path,
    ) -> None:
        """A control that discusses retiring a model is not a notice."""
        payload = _source()
        payload["domains"][0]["controls"][0]["description"] = (
            "Ensure models are RETIRED on a documented schedule."
        )

        controls = _build(tmp_path, payload).run().controls

        assert [c.control_id for c in controls] == ["A001.1", "A001.2"]


class TestTheDropRefusesToRemoveSomethingReal:
    def test_a_notice_on_a_live_control_raises(self, tmp_path: Path) -> None:
        """The control-level check does not cover this path."""
        payload = _source()
        payload["domains"][0]["controls"][0]["activities"][1]["description"] = (
            "RETIRED - merged into A001.1."
        )

        with pytest.raises(ValueError, match=r"A001\.2"):
            _build(tmp_path, payload, OneActivityParser).run()

    def test_a_notice_naming_no_successor_raises(
        self, tmp_path: Path,
    ) -> None:
        payload = _source()
        payload["domains"][0]["controls"].append(_retired_control(
            "E007",
            "RETIRED - withdrawn at Q1 2026 update.",
            "RETIRED - withdrawn.",
        ))

        with pytest.raises(ValueError, match="names no successor"):
            _build(tmp_path, payload).run()

    def test_a_notice_naming_an_unknown_successor_raises(
        self, tmp_path: Path,
    ) -> None:
        payload = _source()
        payload["domains"][0]["controls"].append(_retired_control(
            "E007",
            "RETIRED - Merged with Z999 at Q1 2026 update.",
            "RETIRED - merged into Z999.",
        ))

        with pytest.raises(ValueError, match="Z999"):
            _build(tmp_path, payload).run()

    def test_a_notice_naming_a_retired_successor_raises(
        self, tmp_path: Path,
    ) -> None:
        """One hop only. A chain has to be read and declared."""
        payload = _source()
        payload["domains"][0]["controls"].append(_retired_control(
            "E007",
            "RETIRED - Merged with E014 at Q1 2026 update.",
            "RETIRED - merged into E014.",
        ))
        payload["domains"][0]["controls"].append(_retired_control(
            "E014",
            "RETIRED - Merged into A001 at Q1 2026 update.",
            "RETIRED - merged into A001.",
        ))

        with pytest.raises(ValueError, match="itself retired"):
            _build(tmp_path, payload).run()

    def test_a_live_activity_under_a_retired_control_raises(
        self, tmp_path: Path,
    ) -> None:
        """Dropping the control would discard a real statement.

        The parser reads the whole notice before dropping anything, so an
        activity that survived the merge stops the run rather than vanishing.
        """
        payload = _source()
        retired = _retired_control(
            "E007",
            "RETIRED - Merged with A001 at Q1 2026 update.",
            "RETIRED - merged into A001.",
        )
        retired["activities"].append({
            "id": "E007.2",
            "description": (
                "Maintain an approval register recording who signed off each "
                "change to a deployed AI system and when."
            ),
            "category": "Core",
            "evidence_types": [],
        })
        payload["domains"][0]["controls"].append(retired)

        with pytest.raises(ValueError, match=r"E007\.2"):
            _build(tmp_path, payload).run()


class TestTheRealSource:
    @pytest.fixture()
    def controls(self, tmp_path: Path) -> list[Control]:
        try:
            raw_dir = Aiuc1Parser.resolve_raw_dir()
        except FileNotFoundError:
            pytest.skip("Raw data not available")
        if not (raw_dir / "aiuc-1-standard.json").exists():
            pytest.skip("Raw data not available")
        return Aiuc1Parser(
            raw_dir=raw_dir, audit_dir=tmp_path / "audit",
        ).parse()

    def test_the_two_tombstones_do_not_ship(
        self, controls: list[Control],
    ) -> None:
        shipped = {c.control_id for c in controls}

        assert len(controls) == 130
        # The declaration too, not only the parse. run()'s count gate reads the
        # declaration, and nothing else in this module exercises it against the
        # real source.
        assert Aiuc1Parser.expected_count == len(controls)
        assert "E007.1" not in shipped
        assert "E014.1" not in shipped

    def test_no_shipped_statement_is_a_redirect(
        self, controls: list[Control],
    ) -> None:
        redirects = [
            c.control_id for c in controls
            if c.description.upper().startswith("RETIRED")
        ]

        assert redirects == []

    def test_the_drop_is_recorded_with_both_notices(
        self, controls: list[Control], tmp_path: Path,
    ) -> None:
        records = _audit(tmp_path)

        assert [r["control_id"] for r in records] == ["E007.1", "E014.1"]
        assert [r["successor_id"] for r in records] == ["E004", "E017"]
        assert [r["before"] for r in records] == [
            "RETIRED - merged into E004.",
            "RETIRED - merged into E017.",
        ]

    def test_the_prose_floor_is_what_the_shipped_activities_measure(
        self, controls: list[Control],
    ) -> None:
        """110 of 130, against 110 of 132 while the tombstones shipped."""
        fraction = BaseParser.honest_prose_fraction(controls)

        assert round(fraction, 4) == 0.8462
        assert Aiuc1Parser.min_prose_fraction == 0.84
        assert fraction >= Aiuc1Parser.min_prose_fraction

    def test_re_parsing_the_same_bytes_gives_the_same_controls(
        self, controls: list[Control], tmp_path: Path,
    ) -> None:
        again = Aiuc1Parser(
            raw_dir=Aiuc1Parser.resolve_raw_dir(),
            audit_dir=tmp_path / "again",
        ).parse()

        assert [c.model_dump() for c in again] == [
            c.model_dump() for c in controls
        ]

    def test_no_activity_is_lost_beyond_the_two_notices(
        self, controls: list[Control],
    ) -> None:
        """The drop removes withdrawal notices and nothing else."""
        raw = json.loads(
            (Aiuc1Parser.resolve_raw_dir() / "aiuc-1-standard.json")
            .read_text(encoding="utf-8")
        )
        every_activity = {
            activity["id"]
            for domain in raw["domains"]
            for control in domain["controls"]
            for activity in control.get("activities", [])
        }

        assert every_activity - {c.control_id for c in controls} == {
            "E007.1", "E014.1",
        }


def test_the_fixture_is_not_mutated_by_the_helpers() -> None:
    """_source returns a fresh payload, so one test cannot poison the next."""
    first = _source()
    first["domains"][0]["controls"][0]["description"] = "changed"

    assert _source() != first
    assert copy.deepcopy(_source()) == _source()
