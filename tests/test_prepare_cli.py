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


class TestPrepareArgParsing:
    def test_prepare_required_args_only(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "framework.csv",
            "--framework-id", "test_fw",
            "--name", "Test Framework",
        ])
        assert args.command == "prepare"
        assert args.file == "framework.csv"
        assert args.framework_id == "test_fw"
        assert args.name == "Test Framework"
        assert args.version == "1.0"
        assert args.source_url == ""
        assert args.mapping_unit == "control"
        assert args.fetched_date is None
        assert args.expected_count is None

    def test_prepare_all_optional_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "framework.csv",
            "--framework-id", "test_fw",
            "--name", "Test Framework",
            "--version", "2.0",
            "--source-url", "https://example.com",
            "--mapping-unit", "technique",
            "--fetched-date", "2026-05-01",
            "--expected-count", "50",
        ])
        assert args.version == "2.0"
        assert args.source_url == "https://example.com"
        assert args.mapping_unit == "technique"
        assert args.fetched_date == "2026-05-01"
        assert args.expected_count == 50

    def test_prepare_csv_column_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "framework.csv",
            "--framework-id", "test_fw",
            "--name", "Test",
            "--id-column", "my_id",
            "--title-column", "my_title",
            "--description-column", "my_desc",
            "--fulltext-column", "my_text",
        ])
        assert args.id_column == "my_id"
        assert args.title_column == "my_title"
        assert args.description_column == "my_desc"
        assert args.fulltext_column == "my_text"

    def test_prepare_optional_llm_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "doc.pdf",
            "--framework-id", "test_fw",
            "--name", "Test",
            "--llm",
        ])
        assert args.llm is True

    def test_prepare_optional_format_override(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "weird.dat",
            "--framework-id", "test_fw",
            "--name", "Test",
            "--format", "csv",
        ])
        assert args.format == "csv"

    def test_prepare_optional_output(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "f.csv",
            "--framework-id", "test_fw",
            "--name", "Test",
            "--output", "custom_output.json",
        ])
        assert args.output == "custom_output.json"

    def test_prepare_heading_level(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "f.md",
            "--framework-id", "test_fw",
            "--name", "Test",
            "--heading-level", "3",
        ])
        assert args.heading_level == 3

    def test_prepare_help(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["prepare", "--help"])
        assert exc_info.value.code == 0


class TestPrepareCommand:
    def test_prepare_csv_end_to_end(self, tmp_path: Path) -> None:
        from tract.cli import _cmd_prepare

        csv_path = tmp_path / "fw.csv"
        csv_path.write_text(
            'control_id,title,description\n'
            'TC-01,Access Control,"Enforce access control policies for all system components and users"\n'
            'TC-02,Encryption,"Encrypt sensitive data at rest and in transit using standard algorithms"\n',
            encoding="utf-8",
        )
        output_path = tmp_path / "prepared.json"

        import argparse
        args = argparse.Namespace(
            file=str(csv_path),
            framework_id="test_fw",
            name="Test Framework",
            version="1.0",
            source_url="",
            mapping_unit="control",
            fetched_date=None,
            expected_count=None,
            id_column=None,
            title_column=None,
            description_column=None,
            fulltext_column=None,
            output=str(output_path),
            format=None,
            llm=False,
            heading_level=None,
            json=False,
        )
        _cmd_prepare(args)
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["framework_id"] == "test_fw"
        assert len(data["controls"]) == 2

    def test_prepare_file_not_found(self, tmp_path: Path) -> None:
        from tract.cli import _cmd_prepare

        import argparse
        args = argparse.Namespace(
            file=str(tmp_path / "nonexistent.csv"),
            framework_id="test_fw",
            name="Test",
            version="1.0",
            source_url="",
            mapping_unit="control",
            fetched_date=None,
            expected_count=None,
            id_column=None,
            title_column=None,
            description_column=None,
            fulltext_column=None,
            output=None,
            format=None,
            llm=False,
            heading_level=None,
            json=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_prepare(args)
        assert exc_info.value.code == 1
