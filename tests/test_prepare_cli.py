"""Tests for tract prepare and validate CLI subcommands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tract.cli import build_parser


class TestValidateArgParsing:
    def test_validate_file(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validate", "--file", "framework.json"])
        assert args.command == "validate"
        assert args.file == "framework.json"

    def test_validate_json_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validate", "--file", "f.json", "--json"])
        assert args.json is True

    def test_validate_help(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["validate", "--help"])
        assert exc_info.value.code == 0


class TestValidateCommand:
    def test_validate_valid_file_exits_0(self, tmp_path: Path) -> None:
        from tract.cli import _cmd_validate
        fw_data = {
            "framework_id": "test_fw",
            "framework_name": "Test Framework",
            "version": "1.0",
            "source_url": "https://example.com",
            "fetched_date": "2026-05-01",
            "mapping_unit_level": "control",
            "controls": [
                {"control_id": "TC-01", "title": "Access Control", "description": "Enforce access control policies for system components and users"},
            ],
        }
        file_path = tmp_path / "valid.json"
        file_path.write_text(json.dumps(fw_data), encoding="utf-8")
        import argparse
        args = argparse.Namespace(file=str(file_path), json=False)
        _cmd_validate(args)

    def test_validate_invalid_file_exits_1(self, tmp_path: Path) -> None:
        from tract.cli import _cmd_validate
        fw_data = {
            "framework_id": "BAD ID",
            "framework_name": "Test Framework",
            "version": "1.0",
            "source_url": "https://example.com",
            "fetched_date": "2026-05-01",
            "mapping_unit_level": "control",
            "controls": [
                {"control_id": "TC-01", "title": "Test", "description": "A valid description for the control test case"},
            ],
        }
        file_path = tmp_path / "invalid.json"
        file_path.write_text(json.dumps(fw_data), encoding="utf-8")
        import argparse
        args = argparse.Namespace(file=str(file_path), json=False)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_validate(args)
        assert exc_info.value.code == 1

    def test_validate_json_output(self, tmp_path: Path, capsys) -> None:
        from tract.cli import _cmd_validate
        fw_data = {
            "framework_id": "test_fw",
            "framework_name": "Test Framework",
            "version": "1.0",
            "source_url": "https://example.com",
            "fetched_date": "2026-05-01",
            "mapping_unit_level": "control",
            "controls": [
                {"control_id": "TC-01", "title": "Test", "description": "Short desc but still valid"},
            ],
        }
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(fw_data), encoding="utf-8")
        import argparse
        args = argparse.Namespace(file=str(file_path), json=True)
        _cmd_validate(args)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "errors" in result
        assert "warnings" in result
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)

    def test_validate_file_not_found(self, tmp_path: Path) -> None:
        from tract.cli import _cmd_validate
        import argparse
        args = argparse.Namespace(file=str(tmp_path / "nonexistent.json"), json=False)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_validate(args)
        assert exc_info.value.code == 1
