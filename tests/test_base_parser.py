"""Tests for tract.parsers.base — BaseParser ABC."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import ClassVar

import pytest

from tract.parsers.base import BaseParser
from tract.schema import Control, FrameworkOutput


class StubParser(BaseParser):
    """Concrete parser for testing — returns canned controls."""

    framework_id: ClassVar[str] = "stub_fw"
    framework_name: ClassVar[str] = "Stub Framework"
    version: ClassVar[str] = "1.0"
    source_url: ClassVar[str] = "https://example.com/stub"
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 2
    fetched_date: ClassVar[str] = "2026-01-01"
    manifest_exempt_reason: ClassVar[str] = (
        "returns canned controls for tests and reads no file"
    )

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
        controls: list[Control] | None = None,
        audit_dir: Path | None = None,
    ) -> None:
        super().__init__(
            raw_dir=raw_dir, output_dir=output_dir, audit_dir=audit_dir,
        )
        if controls is not None:
            self._controls = controls
        else:
            self._controls = [
                Control(
                    control_id="STUB-001",
                    title="First Control",
                    description="Description of the first control.",
                ),
                Control(
                    control_id="STUB-002",
                    title="Second Control",
                    description="Description of the second control.",
                ),
            ]

    def parse(self) -> list[Control]:
        return self._controls


class TestBaseParserRun:
    """Tests for BaseParser.run() via StubParser."""

    def test_produces_valid_output(self, tmp_path: Path) -> None:
        """run() returns a valid FrameworkOutput."""
        parser = StubParser(raw_dir=tmp_path, output_dir=tmp_path)
        result = parser.run()

        assert isinstance(result, FrameworkOutput)
        assert result.framework_id == "stub_fw"
        assert result.framework_name == "Stub Framework"
        assert len(result.controls) == 2

    def test_writes_json_file(self, tmp_path: Path) -> None:
        """run() writes a JSON file to output_dir."""
        parser = StubParser(raw_dir=tmp_path, output_dir=tmp_path)
        parser.run()

        output_file = tmp_path / "stub_fw.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert data["framework_id"] == "stub_fw"
        assert len(data["controls"]) == 2

    def test_output_has_sorted_keys(self, tmp_path: Path) -> None:
        """Output JSON has sorted keys (deterministic)."""
        parser = StubParser(raw_dir=tmp_path, output_dir=tmp_path)
        parser.run()

        output_file = tmp_path / "stub_fw.json"
        raw = output_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        top_keys = list(data.keys())
        assert top_keys == sorted(top_keys)

    def test_sanitizes_text_fields(self, tmp_path: Path) -> None:
        """run() sanitizes description and title text."""
        dirty_controls = [
            Control(
                control_id="DIRTY-001",
                title="<b>Bold Title</b>",
                description="  null\x00bytes  and   spaces  ",
            ),
        ]
        parser = StubParser(
            raw_dir=tmp_path,
            output_dir=tmp_path,
            controls=dirty_controls,
        )
        # Adjust expected count to match
        parser.expected_count = 1  # type: ignore[assignment]

        result = parser.run()
        ctrl = result.controls[0]
        assert "\x00" not in ctrl.description
        assert "null bytes and spaces" == ctrl.description
        assert ctrl.title == "Bold Title"

    def test_preserves_full_text_on_truncation(self, tmp_path: Path) -> None:
        """Long descriptions set full_text and truncate description."""
        long_desc = "a" * 3000
        controls = [
            Control(
                control_id="LONG-001",
                title="Long Control",
                description=long_desc,
            ),
        ]
        parser = StubParser(
            raw_dir=tmp_path,
            output_dir=tmp_path,
            controls=controls,
        )
        parser.expected_count = 1  # type: ignore[assignment]

        result = parser.run()
        ctrl = result.controls[0]
        assert len(ctrl.description) == 2000
        assert ctrl.full_text is not None
        assert len(ctrl.full_text) == 3000

    def test_count_mismatch_raises(
        self, tmp_path: Path
    ) -> None:
        """Deviation from expected_count raises ValueError."""
        single_control = [
            Control(
                control_id="ONLY-001",
                title="Only Control",
                description="The sole control.",
            ),
        ]
        parser = StubParser(
            raw_dir=tmp_path,
            output_dir=tmp_path,
            controls=single_control,
        )
        # expected_count is 2, but we provide 1 -> 50% deviation

        with pytest.raises(ValueError, match="deviation"):
            parser.run()

    def test_count_match_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When count matches expected, no WARNING is logged."""
        parser = StubParser(raw_dir=tmp_path, output_dir=tmp_path)

        with caplog.at_level(logging.WARNING):
            parser.run()

        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_records) == 0

    def test_raises_on_zero_controls(self, tmp_path: Path) -> None:
        """run() raises ValueError when parse() returns empty list."""
        parser = StubParser(
            raw_dir=tmp_path,
            output_dir=tmp_path,
            controls=[],
        )

        with pytest.raises(ValueError, match="zero controls"):
            parser.run()


class TestCountCheckRaises:
    def test_raises_when_count_is_outside_tolerance(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        only_one = [Control(
            control_id="STUB-001", title="First", description="Only one here.",
        )]
        parser = StubParser(raw_dir=tmp_path, output_dir=out, controls=only_one)

        with pytest.raises(ValueError, match="expected 2"):
            parser.run()

    def test_documented_opt_out_permits_the_deviation(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()

        class DriftingParser(StubParser):
            count_deviation_reason: ClassVar[str] = (
                "Upstream merged two controls in the 2026 revision."
            )

        only_one = [Control(
            control_id="STUB-001", title="First", description="Only one here.",
        )]
        result = DriftingParser(
            raw_dir=tmp_path, output_dir=out, controls=only_one,
        ).run()
        assert len(result.controls) == 1


class TestFloorCountSemantics:
    """A catalog parser declares a floor, and a floor is one-sided.

    CAPEC and CWE both state in comments that their counts are floors: the
    catalog holds more entries than OpenCRE links to, and emitting all the
    stable ones is correct. The two-sided band turned that into a refusal to
    write, so a parser working exactly as designed could not produce output.
    """

    def _controls(self, count: int) -> list[Control]:
        return [
            Control(
                control_id=f"F-{i}", title=f"Entry {i}",
                description=f"Statement number {i} of the catalog.",
            )
            for i in range(count)
        ]

    def test_a_floor_parser_may_overshoot_by_any_margin(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()

        class FloorParser(StubParser):
            expected_count: ClassVar[int] = 2
            expected_count_is_floor: ClassVar[bool] = True

        result = FloorParser(
            raw_dir=tmp_path, output_dir=out, controls=self._controls(20),
        ).run()
        assert len(result.controls) == 20

    def test_a_floor_parser_may_not_undershoot(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()

        class FloorParser(StubParser):
            expected_count: ClassVar[int] = 10
            expected_count_is_floor: ClassVar[bool] = True

        parser = FloorParser(
            raw_dir=tmp_path, output_dir=out, controls=self._controls(9),
        )
        with pytest.raises(ValueError, match="floor of 10"):
            parser.run()

    def test_a_fixed_count_parser_still_refuses_an_overshoot(
        self, tmp_path: Path,
    ) -> None:
        """Floor semantics are opt-in. Silence must keep the two-sided band."""
        out = tmp_path / "out"
        out.mkdir()

        class FixedParser(StubParser):
            expected_count: ClassVar[int] = 2

        parser = FixedParser(
            raw_dir=tmp_path, output_dir=out, controls=self._controls(20),
        )
        with pytest.raises(ValueError, match="deviation"):
            parser.run()


class TestCountIsMandatory:
    def test_a_parser_without_a_declared_count_refuses_to_write(
        self, tmp_path: Path,
    ) -> None:
        """Omission must not be the cheapest way past the gate.

        The old code skipped the check at DEBUG when no count was declared, so
        a new parser cleared it by saying nothing.
        """
        out = tmp_path / "out"
        out.mkdir()

        class CountlessParser(BaseParser):
            framework_id: ClassVar[str] = "countless_fw"
            framework_name: ClassVar[str] = "Countless Framework"
            version: ClassVar[str] = "1.0"
            source_url: ClassVar[str] = "https://example.com/countless"
            mapping_unit_level: ClassVar[str] = "control"
            fetched_date: ClassVar[str] = "2026-01-01"
            manifest_exempt_reason: ClassVar[str] = "reads no file"

            def parse(self) -> list[Control]:
                return [Control(
                    control_id="C-1", title="One",
                    description="A single control statement.",
                )]

        parser = CountlessParser(raw_dir=tmp_path, output_dir=out)
        with pytest.raises(ValueError, match="no expected_count"):
            parser.run()


class ReadingParser(BaseParser):
    """Parser that reads two real files through read_source()."""

    framework_id: ClassVar[str] = "reading_fw"
    framework_name: ClassVar[str] = "Reading Framework"
    version: ClassVar[str] = "1.0"
    source_url: ClassVar[str] = "https://example.com/reading"
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 1
    fetched_date: ClassVar[str] = "2026-01-01"

    def parse(self) -> list[Control]:
        first = self.read_source("a.txt")
        second = self.read_source("b.txt")
        return [Control(
            control_id="R-001",
            title="Read control",
            description=f"{first.strip()} and {second.strip()} together.",
        )]


class TestSourceManifest:
    def test_records_every_file_read(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.txt").write_text("alpha", encoding="utf-8")
        (raw / "b.txt").write_text("beta", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()

        result = ReadingParser(raw_dir=raw, output_dir=out).run()

        recorded = {s.path: s for s in result.source_files}
        assert set(recorded) == {"a.txt", "b.txt"}
        assert recorded["a.txt"].sha256 == hashlib.sha256(b"alpha").hexdigest()
        assert recorded["b.txt"].sha256 == hashlib.sha256(b"beta").hexdigest()
        assert recorded["a.txt"].bytes == 5
        assert recorded["b.txt"].bytes == 4

    def test_manifest_is_sorted_for_determinism(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "b.txt").write_text("beta", encoding="utf-8")
        (raw / "a.txt").write_text("alpha", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()

        result = ReadingParser(raw_dir=raw, output_dir=out).run()

        paths = [s.path for s in result.source_files]
        assert paths == sorted(paths)


class TestSourceManifestIsMandatory:
    """The manifest replaced a hand-maintained file that covered 7 of 19.

    It then covered 1 of 20, because the mandate lived in a docstring and
    nineteen parsers kept opening files directly. A file read outside
    read_source is invisible to the manifest, so run() wrote an empty
    source_files list and nothing said so.
    """

    class _SilentParser(BaseParser):
        framework_id: ClassVar[str] = "silent_fw"
        framework_name: ClassVar[str] = "Silent Framework"
        version: ClassVar[str] = "1.0"
        source_url: ClassVar[str] = "https://example.com/silent"
        mapping_unit_level: ClassVar[str] = "control"
        expected_count: ClassVar[int] = 1
        fetched_date: ClassVar[str] = "2026-01-01"

        def parse(self) -> list[Control]:
            # Deliberately bypasses read_source, the way the 19 parsers did.
            (self.raw_dir / "a.txt").read_text(encoding="utf-8")
            return [Control(
                control_id="S-1", title="Silent",
                description="A control whose source nothing recorded.",
            )]

    def _raw(self, tmp_path: Path) -> Path:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.txt").write_text("alpha", encoding="utf-8")
        return raw

    def test_a_parser_that_records_no_source_refuses_to_write(
        self, tmp_path: Path,
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        parser = self._SilentParser(raw_dir=self._raw(tmp_path), output_dir=out)

        with pytest.raises(ValueError, match="recorded no source files"):
            parser.run()

    def test_the_documented_exemption_permits_it(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()

        class ExemptParser(TestSourceManifestIsMandatory._SilentParser):
            framework_id: ClassVar[str] = "exempt_fw"
            manifest_exempt_reason: ClassVar[str] = (
                "synthesises controls from a constant table, reads no file"
            )

        result = ExemptParser(
            raw_dir=self._raw(tmp_path), output_dir=out,
        ).run()
        assert result.source_files == []


class TestDeterministicOutput:
    def test_two_runs_produce_identical_bytes(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.txt").write_text("alpha", encoding="utf-8")
        (raw / "b.txt").write_text("beta", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()

        ReadingParser(raw_dir=raw, output_dir=out).run()
        first = (out / "reading_fw.json").read_bytes()
        ReadingParser(raw_dir=raw, output_dir=out).run()
        second = (out / "reading_fw.json").read_bytes()

        assert first == second

    def test_fetched_date_is_declared_not_read_from_the_clock(self) -> None:
        import inspect

        from tract.parsers import base

        source = inspect.getsource(base)
        assert "datetime.now" not in source, (
            "BaseParser must not read the clock; fetched_date is declared "
            "per parser so re-parsing the same bytes gives the same bytes"
        )


class TestProseFloor:
    def _titles_only(self) -> list[Control]:
        return [
            Control(control_id=f"T-{i}", title=f"Title {i}", description=f"Title {i}")
            for i in range(4)
        ]

    def test_refuses_to_write_below_the_declared_floor(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()

        class ProseParser(StubParser):
            expected_count: ClassVar[int] = 4
            min_prose_fraction: ClassVar[float] = 1.0

        parser = ProseParser(
            raw_dir=tmp_path, output_dir=out, controls=self._titles_only(),
        )
        with pytest.raises(ValueError, match="prose fraction"):
            parser.run()

    def test_a_description_equal_to_its_title_is_not_prose(self) -> None:
        controls = [
            Control(control_id="A", title="Use of cryptography",
                    description="Use of cryptography"),
            Control(control_id="B", title="Access control",
                    description="Rules for access shall be defined, documented "
                                "and reviewed at planned intervals by the owner."),
        ]
        assert BaseParser.honest_prose_fraction(controls) == 0.5

    def test_a_long_restatement_of_the_title_is_not_prose(self) -> None:
        long_title = "Policies for information security and topic specific policies"
        controls = [
            Control(control_id="A", title=long_title, description=long_title),
        ]
        assert BaseParser.honest_prose_fraction(controls) == 0.0

    def test_a_damaged_control_is_excluded_from_the_measurement(self) -> None:
        """Damaged text must neither earn credit nor cost it.

        A control the parser has marked damaged is one whose source lost
        content. Counting it as prose lets a known-wrong statement clear the
        floor; counting it against the parser punishes an honest disclosure.
        """
        good = Control(
            control_id="A", title="Access control",
            description="Rules for access shall be defined, documented and "
                        "reviewed at planned intervals by the owner.",
        )
        damaged = Control(
            control_id="B", title="Protecting against threats",
            description="Protection against threats, such as natural [...] "
                        "infrastructure shall be designed and implemented.",
            metadata={"damaged": "true", "damage_reason": "clause lost in conversion"},
        )
        assert BaseParser.honest_prose_fraction([good, damaged]) == 1.0
        assert BaseParser.is_damaged(damaged)
        assert not BaseParser.is_damaged(good)

    def test_every_control_damaged_measures_zero_rather_than_dividing_by_zero(
        self,
    ) -> None:
        damaged = Control(
            control_id="B", title="T", description="D",
            metadata={"damaged": "true", "damage_reason": "r"},
        )
        assert BaseParser.honest_prose_fraction([damaged]) == 0.0


class TestRepairAudit:
    def test_writes_one_sorted_json_object_per_record(self, tmp_path: Path) -> None:
        """The audit file the repair-layer docstring promises must exist.

        The claim that repairs emit before/after pairs was in three documents
        and no function. A docstring asserting a control that does not exist
        is worse than no docstring.
        """
        audit = tmp_path / "audit"
        parser = StubParser(raw_dir=tmp_path, output_dir=tmp_path, audit_dir=audit)

        path = parser.write_repair_audit([
            {"successor_id": "5.7", "predecessor_id": "5.6", "applied": True},
            {"successor_id": "7.6", "predecessor_id": "7.5", "applied": False},
        ])

        lines = path.read_text(encoding="utf-8").splitlines()
        assert path == audit / "stub_fw.jsonl"
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["predecessor_id"] == "5.6"
        # Sorted keys, so re-running against the same source gives the same
        # bytes and a diff of the audit file shows real changes only.
        assert list(first) == sorted(first)

    def test_an_empty_record_list_still_writes_the_file(self, tmp_path: Path) -> None:
        """Absence of the file must mean the parser never ran, not zero repairs."""
        audit = tmp_path / "audit"
        parser = StubParser(raw_dir=tmp_path, output_dir=tmp_path, audit_dir=audit)

        path = parser.write_repair_audit([])

        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""
