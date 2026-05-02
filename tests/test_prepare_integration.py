"""Integration test: prepare -> validate -> ingest round-trip.

Verifies that the full pipeline works end-to-end without the model
inference step (which requires GPU artifacts). Tests the data flow
from raw CSV through preparation, validation, and ingest validation gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.prepare import prepare_framework
from tract.schema import FrameworkOutput
from tract.validate import validate_framework


@pytest.fixture
def csv_framework(tmp_path: Path) -> Path:
    """A realistic multi-control CSV framework."""
    rows = [
        'control_id,title,description',
        'ISEC-01,AI Model Access Control,"Organizations shall implement role-based access controls for AI model endpoints and training infrastructure"',
        'ISEC-02,Training Data Encryption,"All AI training data shall be encrypted at rest using AES-256 and in transit using TLS 1.3"',
        'ISEC-03,Model Inference Logging,"Comprehensive logging shall be maintained for all model inference requests including input hashes and output metadata"',
        'ISEC-04,Adversarial Input Detection,"Systems shall implement input validation and adversarial example detection for production AI models"',
        'ISEC-05,Model Versioning,"All deployed AI models shall maintain cryptographic integrity verification and version tracking"',
    ]
    path = tmp_path / "ai_security_framework.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


@pytest.fixture
def md_framework(tmp_path: Path) -> Path:
    """A markdown framework document."""
    content = (
        "# AI Security Framework v2.0\n\n"
        "## CTRL-01: Access Management\n\n"
        "Implement comprehensive access management policies for all "
        "AI system components including model endpoints, training "
        "pipelines, and data storage systems.\n\n"
        "## CTRL-02: Data Protection\n\n"
        "Protect the confidentiality, integrity, and availability of "
        "AI training and inference data through encryption, access "
        "controls, and data loss prevention measures.\n\n"
        "## CTRL-03: Operational Monitoring\n\n"
        "Maintain continuous monitoring of AI system operations "
        "including model performance, data drift, adversarial "
        "attack detection, and resource utilization.\n"
    )
    path = tmp_path / "ai_security_v2.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestPrepareValidateRoundTrip:
    def test_csv_prepare_then_validate_clean(
        self, csv_framework: Path, tmp_path: Path,
    ) -> None:
        """Prepare a CSV, then validate the output — should be clean."""
        output_path = tmp_path / "prepared.json"

        result_path = prepare_framework(
            file_path=csv_framework,
            framework_id="ai_sec_fw",
            name="AI Security Framework",
            version="1.0",
            source_url="https://example.com/aisec",
            mapping_unit="control",
            output_path=output_path,
        )

        assert result_path.exists()
        data = json.loads(result_path.read_text(encoding="utf-8"))

        issues = validate_framework(data)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {[i.message for i in errors]}"

        fw = FrameworkOutput.model_validate(data)
        assert fw.framework_id == "ai_sec_fw"
        assert fw.framework_name == "AI Security Framework"
        assert len(fw.controls) == 5

    def test_markdown_prepare_then_validate_clean(
        self, md_framework: Path, tmp_path: Path,
    ) -> None:
        """Prepare a markdown file, then validate — should be clean."""
        output_path = tmp_path / "prepared.json"

        result_path = prepare_framework(
            file_path=md_framework,
            framework_id="ai_sec_v2",
            name="AI Security Framework v2",
            version="2.0",
            source_url="https://example.com/aisec-v2",
            mapping_unit="control",
            output_path=output_path,
        )

        assert result_path.exists()
        data = json.loads(result_path.read_text(encoding="utf-8"))

        issues = validate_framework(data)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {[i.message for i in errors]}"

        fw = FrameworkOutput.model_validate(data)
        assert len(fw.controls) == 3

    def test_prepared_output_passes_pydantic_schema(
        self, csv_framework: Path, tmp_path: Path,
    ) -> None:
        """FrameworkOutput.model_validate() on prepared JSON must succeed."""
        output_path = tmp_path / "prepared.json"
        prepare_framework(
            file_path=csv_framework,
            framework_id="schema_test",
            name="Schema Test Framework",
            version="1.0",
            source_url="https://example.com",
            mapping_unit="control",
            output_path=output_path,
        )

        data = json.loads(output_path.read_text(encoding="utf-8"))
        fw = FrameworkOutput.model_validate(data)
        assert fw.framework_id == "schema_test"

    def test_sanitization_removes_null_bytes(self, tmp_path: Path) -> None:
        """Null bytes in raw CSV should be stripped by sanitization."""
        csv_path = tmp_path / "dirty.csv"
        csv_path.write_text(
            'control_id,title,description\n'
            'TC-01,Null\x00Title,"Description with\x00null bytes that should be removed by sanitizer"\n',
            encoding="utf-8",
        )
        output_path = tmp_path / "prepared.json"
        prepare_framework(
            file_path=csv_path,
            framework_id="clean_test",
            name="Clean Test",
            version="1.0",
            source_url="https://example.com",
            mapping_unit="control",
            output_path=output_path,
        )

        data = json.loads(output_path.read_text(encoding="utf-8"))
        ctrl = data["controls"][0]
        assert "\x00" not in ctrl["title"]
        assert "\x00" not in ctrl["description"]

    def test_prepared_json_is_deterministic(
        self, csv_framework: Path, tmp_path: Path,
    ) -> None:
        """Running prepare twice on the same input produces identical output."""
        output_a = tmp_path / "a.json"
        output_b = tmp_path / "b.json"

        prepare_framework(
            file_path=csv_framework,
            framework_id="det_test",
            name="Determinism Test",
            version="1.0",
            source_url="https://example.com",
            mapping_unit="control",
            output_path=output_a,
        )
        prepare_framework(
            file_path=csv_framework,
            framework_id="det_test",
            name="Determinism Test",
            version="1.0",
            source_url="https://example.com",
            mapping_unit="control",
            output_path=output_b,
        )

        assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")

    def test_validation_gate_blocks_bad_prepared_file(self, tmp_path: Path) -> None:
        """Manually craft a bad prepared JSON and confirm validation catches it."""
        bad_data = {
            "framework_id": "BAD ID",
            "framework_name": "Bad Framework",
            "version": "1.0",
            "source_url": "https://example.com",
            "fetched_date": "2026-05-01",
            "mapping_unit_level": "control",
            "controls": [
                {
                    "control_id": "TC-01",
                    "title": "Test",
                    "description": "short",
                },
                {
                    "control_id": "TC-01",
                    "title": "Test",
                    "description": "Another description that is long enough for validation",
                },
            ],
        }
        issues = validate_framework(bad_data)
        errors = [i for i in issues if i.severity == "error"]

        error_rules = {i.rule for i in errors}
        assert "invalid_framework_id" in error_rules
        assert "empty_description" in error_rules
        assert "duplicate_control_id" in error_rules
