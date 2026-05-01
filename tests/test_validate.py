"""Tests for tract.validate — ValidationIssue and validate_framework()."""
from __future__ import annotations

import re
import unicodedata

import pytest

from tract.validate import ValidationIssue, validate_framework


def _make_framework(
    framework_id: str = "test_fw",
    framework_name: str = "Test Framework",
    version: str = "1.0",
    source_url: str = "https://example.com",
    fetched_date: str = "2026-05-01",
    mapping_unit_level: str = "control",
    controls: list[dict] | None = None,
) -> dict:
    """Build a minimal valid FrameworkOutput dict."""
    if controls is None:
        controls = [
            {
                "control_id": "TC-01",
                "title": "Access Control",
                "description": "Enforce access control policies for all system components and users",
            },
        ]
    return {
        "framework_id": framework_id,
        "framework_name": framework_name,
        "version": version,
        "source_url": source_url,
        "fetched_date": fetched_date,
        "mapping_unit_level": mapping_unit_level,
        "controls": controls,
    }


class TestValidationIssue:
    def test_dataclass_fields(self) -> None:
        issue = ValidationIssue(
            severity="error",
            control_id="TC-01",
            rule="empty_description",
            message="Control TC-01: description too short",
        )
        assert issue.severity == "error"
        assert issue.control_id == "TC-01"
        assert issue.rule == "empty_description"
        assert issue.message == "Control TC-01: description too short"

    def test_framework_level_issue_has_none_control_id(self) -> None:
        issue = ValidationIssue(
            severity="error",
            control_id=None,
            rule="zero_controls",
            message="No controls found",
        )
        assert issue.control_id is None


class TestValidateFrameworkErrors:
    def test_valid_framework_no_errors(self) -> None:
        issues = validate_framework(_make_framework())
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_schema_conformance_bad_data(self) -> None:
        issues = validate_framework({"bad": "data"})
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.rule == "schema_conformance" for i in errors)

    def test_empty_description(self) -> None:
        controls = [
            {"control_id": "TC-01", "title": "Test", "description": "short"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        errors = [i for i in issues if i.severity == "error"]
        assert any(
            i.rule == "empty_description" and i.control_id == "TC-01"
            for i in errors
        )

    def test_duplicate_control_id(self) -> None:
        controls = [
            {"control_id": "TC-01", "title": "Test A", "description": "Description of control A for testing purposes"},
            {"control_id": "TC-01", "title": "Test B", "description": "Description of control B for testing purposes"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.rule == "duplicate_control_id" for i in errors)

    def test_invalid_framework_id(self) -> None:
        issues = validate_framework(_make_framework(framework_id="BAD ID!"))
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.rule == "invalid_framework_id" for i in errors)

    def test_framework_id_starting_with_number(self) -> None:
        issues = validate_framework(_make_framework(framework_id="1bad"))
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.rule == "invalid_framework_id" for i in errors)

    def test_null_bytes_in_description(self) -> None:
        controls = [
            {"control_id": "TC-01", "title": "Test", "description": "Contains \x00 null byte and more text padding"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        errors = [i for i in issues if i.severity == "error"]
        assert any(
            i.rule == "null_bytes" and i.control_id == "TC-01"
            for i in errors
        )

    def test_null_bytes_in_title(self) -> None:
        controls = [
            {"control_id": "TC-01", "title": "Null\x00Title", "description": "A valid description for testing this control"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        errors = [i for i in issues if i.severity == "error"]
        assert any(
            i.rule == "null_bytes" and "title" in i.message
            for i in errors
        )

    def test_zero_controls(self) -> None:
        issues = validate_framework(_make_framework(controls=[]))
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.rule == "zero_controls" for i in errors)


class TestValidateFrameworkWarnings:
    def test_short_description_warning(self) -> None:
        controls = [
            {"control_id": "TC-01", "title": "Test", "description": "Short but valid desc"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(
            i.rule == "short_description" and i.control_id == "TC-01"
            for i in warnings
        )

    def test_long_description_without_full_text_warning(self) -> None:
        controls = [
            {
                "control_id": "TC-01",
                "title": "Test",
                "description": "A" * 2500,
            },
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(
            i.rule == "long_description_no_full_text" and i.control_id == "TC-01"
            for i in warnings
        )

    def test_long_description_with_full_text_no_warning(self) -> None:
        controls = [
            {
                "control_id": "TC-01",
                "title": "Test",
                "description": "A" * 2500,
                "full_text": "Full text version of the control",
            },
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert not any(
            i.rule == "long_description_no_full_text"
            for i in warnings
        )

    def test_problematic_control_id_chars(self) -> None:
        controls = [
            {"control_id": "TC:01 bad", "title": "Test", "description": "A valid description for the control test case"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(
            i.rule == "problematic_control_id_chars"
            for i in warnings
        )

    def test_low_control_count(self) -> None:
        controls = [
            {"control_id": "TC-01", "title": "Test", "description": "A valid description for the control test case"},
            {"control_id": "TC-02", "title": "Test", "description": "A valid description for the control test case"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(i.rule == "low_control_count" for i in warnings)

    def test_high_control_count(self) -> None:
        controls = [
            {"control_id": f"TC-{i:04d}", "title": f"Test {i}", "description": "A valid description for the control test case"}
            for i in range(2001)
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(i.rule == "high_control_count" for i in warnings)

    def test_missing_version_warning(self) -> None:
        issues = validate_framework(_make_framework(version=""))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(i.rule == "missing_optional_field" and "version" in i.message for i in warnings)

    def test_missing_source_url_warning(self) -> None:
        issues = validate_framework(_make_framework(source_url=""))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(i.rule == "missing_optional_field" and "source_url" in i.message for i in warnings)

    def test_non_nfc_unicode_warning(self) -> None:
        decomposed = unicodedata.normalize("NFD", "é")
        controls = [
            {"control_id": "TC-01", "title": "Test", "description": f"R{decomposed}sum{decomposed} of the control text for testing"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert any(
            i.rule == "non_nfc_unicode" and i.control_id == "TC-01"
            for i in warnings
        )

    def test_nfc_text_no_unicode_warning(self) -> None:
        controls = [
            {"control_id": "TC-01", "title": "Test", "description": "A normal ASCII description for testing the control"},
        ]
        issues = validate_framework(_make_framework(controls=controls))
        warnings = [i for i in issues if i.severity == "warning"]
        assert not any(i.rule == "non_nfc_unicode" for i in warnings)
