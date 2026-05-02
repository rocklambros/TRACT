"""Tests for tract.prepare — prepare_framework() orchestrator."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tract.prepare import prepare_framework
from tract.schema import FrameworkOutput


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_framework.csv"
    path.write_text(
        'control_id,title,description\n'
        'TC-01,Access Control,"Enforce access control policies for all system components and users"\n'
        'TC-02,Data Encryption,"Encrypt sensitive data at rest and in transit using standard algorithms"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    content = (
        "# Framework\n\n"
        "## ASI01: Access Control\n\n"
        "Enforce access control policies for all system components "
        "and users to prevent unauthorized access.\n\n"
        "## ASI02: Data Encryption\n\n"
        "Encrypt sensitive data at rest and in transit using "
        "industry-standard cryptographic algorithms.\n"
    )
    path = tmp_path / "test_framework.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def json_array_file(tmp_path: Path) -> Path:
    data = [
        {"id": "TC-01", "name": "Access", "description": "Enforce access control policies for all system components"},
        {"id": "TC-02", "name": "Encrypt", "description": "Encrypt sensitive data at rest and in transit securely"},
    ]
    path = tmp_path / "test_framework.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def metadata() -> dict[str, str]:
    return {
        "framework_id": "test_fw",
        "name": "Test Framework",
        "version": "1.0",
        "source_url": "https://example.com",
        "mapping_unit": "control",
    }


class TestPrepareFramework:
    def test_prepare_csv(self, csv_file: Path, metadata: dict, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        result_path = prepare_framework(
            file_path=csv_file,
            framework_id=metadata["framework_id"],
            name=metadata["name"],
            version=metadata["version"],
            source_url=metadata["source_url"],
            mapping_unit=metadata["mapping_unit"],
            output_path=output_path,
        )
        assert result_path == output_path
        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        fw = FrameworkOutput.model_validate(data)
        assert fw.framework_id == "test_fw"
        assert len(fw.controls) == 2
        assert fw.controls[0].control_id == "TC-01"

    def test_prepare_markdown(self, md_file: Path, metadata: dict, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        prepare_framework(
            file_path=md_file,
            framework_id=metadata["framework_id"],
            name=metadata["name"],
            version=metadata["version"],
            source_url=metadata["source_url"],
            mapping_unit=metadata["mapping_unit"],
            output_path=output_path,
        )
        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data["controls"]) == 2

    def test_prepare_json(self, json_array_file: Path, metadata: dict, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        prepare_framework(
            file_path=json_array_file,
            framework_id=metadata["framework_id"],
            name=metadata["name"],
            version=metadata["version"],
            source_url=metadata["source_url"],
            mapping_unit=metadata["mapping_unit"],
            output_path=output_path,
        )
        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data["controls"]) == 2

    def test_default_output_path(self, csv_file: Path, metadata: dict) -> None:
        result_path = prepare_framework(
            file_path=csv_file,
            framework_id=metadata["framework_id"],
            name=metadata["name"],
            version=metadata["version"],
            source_url=metadata["source_url"],
            mapping_unit=metadata["mapping_unit"],
        )
        expected = csv_file.with_name("test_framework_prepared.json")
        assert result_path == expected
        assert expected.exists()
        expected.unlink()

    def test_format_override(self, csv_file: Path, metadata: dict, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        prepare_framework(
            file_path=csv_file,
            framework_id=metadata["framework_id"],
            name=metadata["name"],
            version=metadata["version"],
            source_url=metadata["source_url"],
            mapping_unit=metadata["mapping_unit"],
            output_path=output_path,
            format_override="csv",
        )
        assert output_path.exists()

    def test_sanitization_applied(self, tmp_path: Path, metadata: dict) -> None:
        csv_path = tmp_path / "dirty.csv"
        csv_path.write_text(
            'control_id,title,description\n'
            'TC-01,Null\x00Title,"Has null\x00bytes and  extra   spaces inside the description"\n',
            encoding="utf-8",
        )
        output_path = tmp_path / "output.json"
        prepare_framework(
            file_path=csv_path,
            framework_id=metadata["framework_id"],
            name=metadata["name"],
            version=metadata["version"],
            source_url=metadata["source_url"],
            mapping_unit=metadata["mapping_unit"],
            output_path=output_path,
        )
        data = json.loads(output_path.read_text(encoding="utf-8"))
        ctrl = data["controls"][0]
        assert "\x00" not in ctrl["title"]
        assert "\x00" not in ctrl["description"]
        assert "  " not in ctrl["description"]

    def test_output_is_valid_framework_output(self, csv_file: Path, metadata: dict, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        prepare_framework(
            file_path=csv_file,
            framework_id=metadata["framework_id"],
            name=metadata["name"],
            version=metadata["version"],
            source_url=metadata["source_url"],
            mapping_unit=metadata["mapping_unit"],
            output_path=output_path,
        )
        data = json.loads(output_path.read_text(encoding="utf-8"))
        fw = FrameworkOutput.model_validate(data)
        assert fw.framework_id == metadata["framework_id"]
        assert fw.framework_name == metadata["name"]
        assert fw.version == metadata["version"]
        assert fw.source_url == metadata["source_url"]
        assert fw.mapping_unit_level == metadata["mapping_unit"]

    def test_file_not_found_raises(self, metadata: dict, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prepare_framework(
                file_path=tmp_path / "nonexistent.csv",
                framework_id=metadata["framework_id"],
                name=metadata["name"],
                version=metadata["version"],
                source_url=metadata["source_url"],
                mapping_unit=metadata["mapping_unit"],
            )

    def test_unstructured_without_llm_raises(self, tmp_path: Path, metadata: dict) -> None:
        pdf_path = tmp_path / "document.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")
        with pytest.raises(ValueError, match="(?i)llm|unstructured"):
            prepare_framework(
                file_path=pdf_path,
                framework_id=metadata["framework_id"],
                name=metadata["name"],
                version=metadata["version"],
                source_url=metadata["source_url"],
                mapping_unit=metadata["mapping_unit"],
            )
