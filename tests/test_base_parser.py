"""Tests for tract.parsers.base — BaseParser ABC."""

from __future__ import annotations

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

    def test_count_mismatch_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Deviation from expected_count logs a WARNING."""
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

        with caplog.at_level(logging.WARNING):
            parser.run()

        assert any("deviation" in r.message.lower() for r in caplog.records)

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

    def test_today_returns_iso_date(self) -> None:
        """_today() returns a YYYY-MM-DD string."""
        parser = StubParser()
        today = parser._today()
        # Validate format
        parts = today.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4
        assert len(parts[1]) == 2
        assert len(parts[2]) == 2
