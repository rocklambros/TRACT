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

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
        controls: list[Control] | None = None,
    ) -> None:
        super().__init__(raw_dir=raw_dir, output_dir=output_dir)
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
