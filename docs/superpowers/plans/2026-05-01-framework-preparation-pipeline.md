# Framework Preparation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tract prepare` and `tract validate` CLI commands for converting raw framework documents to FrameworkOutput JSON and validating them before model inference.

**Architecture:** Modular extractors (CSV, markdown, JSON, LLM-assisted) behind a registry, deterministic validation as both standalone command and ingest gate, sanitization via existing pipeline.

**Tech Stack:** Python 3.12, Pydantic v2, csv module, anthropic SDK (for LLM extractor)

**Spec:** `docs/superpowers/specs/2026-05-01-framework-preparation-pipeline-design.md`

---

## File Structure

### New Files

```
tract/validate.py                        — ValidationIssue dataclass + validate_framework()
tract/prepare/__init__.py                — Public API: prepare_framework()
tract/prepare/extract.py                 — detect_format(), ExtractorRegistry
tract/prepare/csv_extractor.py           — CsvExtractor
tract/prepare/markdown_extractor.py      — MarkdownExtractor
tract/prepare/json_extractor.py          — JsonExtractor
tract/prepare/llm_extractor.py           — LlmExtractor (Claude API integration)

tests/test_validate.py                   — Validation rules: every error + warning
tests/test_csv_extractor.py              — CSV fixtures: column naming, empty rows, quoted fields
tests/test_markdown_extractor.py         — Markdown fixtures: heading levels, ID patterns
tests/test_json_extractor.py             — JSON fixtures: FrameworkOutput, array, nested
tests/test_llm_extractor.py             — Mock Claude API responses
tests/test_prepare.py                    — prepare_framework() orchestrator
tests/test_prepare_cli.py               — CLI subcommands (prepare + validate)
tests/test_prepare_integration.py        — Round-trip: prepare -> validate -> ingest

tests/fixtures/sample_framework.csv      — CSV test fixture
tests/fixtures/sample_framework.md       — Markdown test fixture
tests/fixtures/sample_framework_array.json — JSON array test fixture
```

### Modified Files

```
tract/config.py                          — Add PREPARE_* and VALIDATE_* constants
tract/cli.py                             — Add prepare + validate subcommands, validation gate in ingest
```

### Existing Files Referenced (read-only)

```
tract/schema.py                          — FrameworkOutput, Control (Pydantic models)
tract/sanitize.py                        — sanitize_text()
tract/io.py                              — atomic_write_json(), load_json()
tract/parsers/base.py                    — BaseParser._sanitize_control() pattern
```

---

### Task 1: ValidationIssue Dataclass + validate_framework() + Tests

**Files:**
- Create: `tract/validate.py`
- Create: `tests/test_validate.py`
- Modify: `tract/config.py`

- [ ] **Step 1: Add validation constants to config.py**

Add after the existing `PHASE1A_FRAMEWORK_SLUG_RE` constant:

```python
# ── Validation Constants ─────────────────────────────────────────────────

VALIDATE_FRAMEWORK_ID_RE: Final[str] = r"^[a-z][a-z0-9_]{1,49}$"
VALIDATE_MIN_DESCRIPTION_LENGTH: Final[int] = 10
VALIDATE_SHORT_DESCRIPTION_LENGTH: Final[int] = 50
VALIDATE_LONG_DESCRIPTION_LENGTH: Final[int] = 2000
VALIDATE_LOW_CONTROL_COUNT: Final[int] = 3
VALIDATE_HIGH_CONTROL_COUNT: Final[int] = 2000

# LLM extractor settings
PREPARE_LLM_MODEL: Final[str] = "claude-sonnet-4-20250514"
PREPARE_LLM_TEMPERATURE: Final[float] = 0.0
PREPARE_LLM_MAX_RETRIES: Final[int] = 3
PREPARE_LLM_RETRY_INITIAL_DELAY_S: Final[float] = 1.0
PREPARE_LLM_RETRY_BACKOFF_FACTOR: Final[float] = 2.0
PREPARE_LLM_CHUNK_TOKEN_LIMIT: Final[int] = 100_000
```

- [ ] **Step 2: Write the test file tests/test_validate.py**

```python
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
        # U+00E9 (e-acute) can be composed (NFC) or decomposed (NFD)
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
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_validate.py -q 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'tract.validate'`

- [ ] **Step 4: Create tract/validate.py**

```python
"""TRACT framework validation — rules engine for FrameworkOutput JSON.

Validates a FrameworkOutput dict (or already-loaded JSON) against a set of
error rules (block ingest) and warning rules (informational). Used by:
  - `tract validate --file prepared.json`
  - `tract ingest` validation gate (pre-inference)
  - `tract prepare` post-extraction quality check

Public API:
    validate_framework(data) -> list[ValidationIssue]
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from tract.config import (
    VALIDATE_FRAMEWORK_ID_RE,
    VALIDATE_HIGH_CONTROL_COUNT,
    VALIDATE_LONG_DESCRIPTION_LENGTH,
    VALIDATE_LOW_CONTROL_COUNT,
    VALIDATE_MIN_DESCRIPTION_LENGTH,
    VALIDATE_SHORT_DESCRIPTION_LENGTH,
)
from tract.schema import FrameworkOutput

logger = logging.getLogger(__name__)

_FRAMEWORK_ID_PATTERN: re.Pattern[str] = re.compile(VALIDATE_FRAMEWORK_ID_RE)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single validation finding (error or warning).

    Attributes:
        severity: "error" blocks ingest; "warning" is informational.
        control_id: The control that triggered this issue, or None for
            framework-level issues.
        rule: Machine-readable rule identifier (e.g., "empty_description").
        message: Human-readable description of the issue.
    """

    severity: Literal["error", "warning"]
    control_id: str | None
    rule: str
    message: str


def validate_framework(data: dict[str, Any]) -> list[ValidationIssue]:
    """Validate a FrameworkOutput dict against all error and warning rules.

    Args:
        data: A dict representing a FrameworkOutput JSON structure.

    Returns:
        List of ValidationIssue instances. Empty list means fully valid.
        Issues are ordered: errors first, then warnings.
    """
    issues: list[ValidationIssue] = []

    # ── Schema conformance via Pydantic ──────────────────────────────
    try:
        fw = FrameworkOutput.model_validate(data)
    except ValidationError as exc:
        issues.append(ValidationIssue(
            severity="error",
            control_id=None,
            rule="schema_conformance",
            message=f"Schema validation failed: {exc}",
        ))
        # If schema fails, we cannot reliably inspect controls — return early
        # but still check framework_id if present
        fw_id = data.get("framework_id", "")
        if fw_id and not _FRAMEWORK_ID_PATTERN.match(fw_id):
            issues.append(ValidationIssue(
                severity="error",
                control_id=None,
                rule="invalid_framework_id",
                message=f"Invalid framework_id: {fw_id!r}",
            ))
        return issues

    # ── Framework-level checks ───────────────────────────────────────
    if not _FRAMEWORK_ID_PATTERN.match(fw.framework_id):
        issues.append(ValidationIssue(
            severity="error",
            control_id=None,
            rule="invalid_framework_id",
            message=f"Invalid framework_id: {fw.framework_id!r}",
        ))

    if len(fw.controls) == 0:
        issues.append(ValidationIssue(
            severity="error",
            control_id=None,
            rule="zero_controls",
            message="No controls found",
        ))
        return issues

    # Duplicate control_id check
    seen_ids: set[str] = set()
    for ctrl in fw.controls:
        if ctrl.control_id in seen_ids:
            issues.append(ValidationIssue(
                severity="error",
                control_id=ctrl.control_id,
                rule="duplicate_control_id",
                message=f"Duplicate control_id: {ctrl.control_id}",
            ))
        seen_ids.add(ctrl.control_id)

    # ── Per-control checks ───────────────────────────────────────────
    for ctrl in fw.controls:
        cid = ctrl.control_id

        # Null bytes in any text field
        text_fields = {
            "description": ctrl.description,
            "title": ctrl.title,
        }
        if ctrl.full_text is not None:
            text_fields["full_text"] = ctrl.full_text

        for field_name, field_value in text_fields.items():
            if "\x00" in field_value:
                issues.append(ValidationIssue(
                    severity="error",
                    control_id=cid,
                    rule="null_bytes",
                    message=f"Control {cid}: null bytes in {field_name}",
                ))

        # Empty / too-short description
        desc_stripped = ctrl.description.strip()
        if len(desc_stripped) < VALIDATE_MIN_DESCRIPTION_LENGTH:
            issues.append(ValidationIssue(
                severity="error",
                control_id=cid,
                rule="empty_description",
                message=(
                    f"Control {cid}: description too short "
                    f"({len(desc_stripped)} chars, min {VALIDATE_MIN_DESCRIPTION_LENGTH})"
                ),
            ))
        elif len(desc_stripped) < VALIDATE_SHORT_DESCRIPTION_LENGTH:
            issues.append(ValidationIssue(
                severity="warning",
                control_id=cid,
                rule="short_description",
                message=(
                    f"Control {cid}: description only {len(desc_stripped)} chars "
                    "— may produce weak embeddings"
                ),
            ))

        # Long description without full_text
        if (
            len(ctrl.description) > VALIDATE_LONG_DESCRIPTION_LENGTH
            and not ctrl.full_text
        ):
            issues.append(ValidationIssue(
                severity="warning",
                control_id=cid,
                rule="long_description_no_full_text",
                message=(
                    f"Control {cid}: description {len(ctrl.description)} chars "
                    "without full_text split"
                ),
            ))

        # Problematic characters in control_id
        if ":" in cid or re.search(r"\s", cid):
            issues.append(ValidationIssue(
                severity="warning",
                control_id=cid,
                rule="problematic_control_id_chars",
                message=(
                    f"Control {cid}: contains characters that may cause "
                    "issues in DB keys"
                ),
            ))

        # Non-NFC unicode
        for field_name, field_value in text_fields.items():
            if field_value != unicodedata.normalize("NFC", field_value):
                issues.append(ValidationIssue(
                    severity="warning",
                    control_id=cid,
                    rule="non_nfc_unicode",
                    message=(
                        f"Control {cid}: text contains non-NFC unicode "
                        "(will be normalized)"
                    ),
                ))
                break  # One warning per control is enough

    # ── Framework-level warnings ─────────────────────────────────────
    n_controls = len(fw.controls)
    if n_controls < VALIDATE_LOW_CONTROL_COUNT:
        issues.append(ValidationIssue(
            severity="warning",
            control_id=None,
            rule="low_control_count",
            message=f"Only {n_controls} controls — unusually low",
        ))
    elif n_controls > VALIDATE_HIGH_CONTROL_COUNT:
        issues.append(ValidationIssue(
            severity="warning",
            control_id=None,
            rule="high_control_count",
            message=f"{n_controls} controls — unusually high, verify extraction",
        ))

    if not fw.version:
        issues.append(ValidationIssue(
            severity="warning",
            control_id=None,
            rule="missing_optional_field",
            message="Missing version — recommended for traceability",
        ))

    if not fw.source_url:
        issues.append(ValidationIssue(
            severity="warning",
            control_id=None,
            rule="missing_optional_field",
            message="Missing source_url — recommended for traceability",
        ))

    # Sort: errors first, then warnings
    issues.sort(key=lambda i: (0 if i.severity == "error" else 1, i.rule))

    logger.info(
        "Validation complete: %d errors, %d warnings",
        sum(1 for i in issues if i.severity == "error"),
        sum(1 for i in issues if i.severity == "warning"),
    )

    return issues
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_validate.py -v
```

All tests should pass.

- [ ] **Step 6: Commit**

```
feat: add ValidationIssue dataclass and validate_framework() engine

Implements validation rules from the framework preparation pipeline
design spec (Section 4). Covers 6 error rules (schema conformance,
empty description, duplicate control_id, invalid framework_id, null
bytes, zero controls) and 7 warning rules (short description, long
description without full_text, problematic control_id chars, low/high
control count, missing optional fields, non-NFC unicode).
```

---

### Task 2: `tract validate` CLI Subcommand + Tests

**Files:**
- Modify: `tract/cli.py`
- Create: `tests/test_prepare_cli.py`

- [ ] **Step 1: Write CLI argument parsing tests in tests/test_prepare_cli.py**

```python
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
                {
                    "control_id": "TC-01",
                    "title": "Access Control",
                    "description": "Enforce access control policies for system components and users",
                },
            ],
        }
        file_path = tmp_path / "valid.json"
        file_path.write_text(json.dumps(fw_data), encoding="utf-8")

        import argparse
        args = argparse.Namespace(file=str(file_path), json=False)
        # Should not raise or call sys.exit
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
                {
                    "control_id": "TC-01",
                    "title": "Test",
                    "description": "A valid description for the control test case",
                },
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
                {
                    "control_id": "TC-01",
                    "title": "Test",
                    "description": "Short desc but still valid",
                },
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py::TestValidateArgParsing -q 2>&1 | head -10
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py::TestValidateCommand -q 2>&1 | head -10
```

Expected: `AttributeError` for missing `validate` subcommand and `_cmd_validate`.

- [ ] **Step 3: Add the validate subcommand to tract/cli.py**

Add after the `p_accept` subparser definition (around line 89):

```python
    # ── validate ─────────────────────────────────────────────────────
    p_validate = subparsers.add_parser(
        "validate",
        help="Validate a prepared framework JSON file",
        epilog=(
            "Examples:\n"
            "  tract validate --file prepared.json\n"
            "  tract validate --file prepared.json --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_validate.add_argument("--file", required=True, help="Framework JSON file to validate")
    p_validate.add_argument("--json", action="store_true", help="Machine-readable JSON output")
```

- [ ] **Step 4: Add the _cmd_validate handler function**

Add after `_cmd_accept` (around line 459):

```python
def _cmd_validate(args: argparse.Namespace) -> None:
    from tract.io import load_json
    from tract.validate import validate_framework

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = load_json(file_path)
    except Exception as e:
        print(f"Error: Failed to load JSON: {e}", file=sys.stderr)
        sys.exit(1)

    issues = validate_framework(data)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if args.json:
        output = {
            "file": str(file_path),
            "errors": [
                {"control_id": i.control_id, "rule": i.rule, "message": i.message}
                for i in errors
            ],
            "warnings": [
                {"control_id": i.control_id, "rule": i.rule, "message": i.message}
                for i in warnings
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if errors:
            print(f"ERRORS ({len(errors)}):", file=sys.stderr)
            for i in errors:
                prefix = f"  [{i.control_id}] " if i.control_id else "  "
                print(f"{prefix}{i.message}", file=sys.stderr)

        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):", file=sys.stderr)
            for i in warnings:
                prefix = f"  [{i.control_id}] " if i.control_id else "  "
                print(f"{prefix}{i.message}", file=sys.stderr)

        if not errors and not warnings:
            print("Validation passed: no errors, no warnings.")
        elif not errors:
            print(f"\nValidation passed with {len(warnings)} warning(s).")

    if errors:
        sys.exit(1)
```

- [ ] **Step 5: Register the validate handler in the main() dispatch table**

In the `handlers` dict inside `main()`, add:

```python
        "validate": _cmd_validate,
```

- [ ] **Step 6: Update the test_all_commands_have_help test in tests/test_cli.py**

Add `"validate"` to the command list in the existing `test_all_commands_have_help`:

```python
        for cmd in ["assign", "compare", "ingest", "accept", "export", "hierarchy",
                     "propose-hubs", "review-proposals", "tutorial", "validate"]:
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py tests/test_cli.py::TestArgParsing::test_all_commands_have_help -v
```

- [ ] **Step 8: Commit**

```
feat: add tract validate CLI subcommand

Adds `tract validate --file <path>` command that runs all validation
rules against a FrameworkOutput JSON and reports errors/warnings.
Exits 0 on clean, 1 on any error. Supports --json for machine-readable
output.
```

---

### Task 3: Validation Gate in `tract ingest`

**Files:**
- Modify: `tract/cli.py` (`_cmd_ingest` function)
- Add to: `tests/test_prepare_cli.py`

- [ ] **Step 1: Write tests for the ingest validation gate**

Add to `tests/test_prepare_cli.py`:

```python
class TestIngestValidationGate:
    def test_ingest_rejects_invalid_framework(self, tmp_path: Path) -> None:
        """Ingest should exit 1 if validation finds errors."""
        from tract.cli import _cmd_ingest

        # Framework with invalid framework_id
        fw_data = {
            "framework_id": "BAD ID",
            "framework_name": "Test",
            "version": "1.0",
            "source_url": "https://example.com",
            "fetched_date": "2026-05-01",
            "mapping_unit_level": "control",
            "controls": [
                {
                    "control_id": "TC-01",
                    "title": "Test",
                    "description": "A valid description for the control test case",
                },
            ],
        }
        file_path = tmp_path / "bad_fw.json"
        file_path.write_text(json.dumps(fw_data), encoding="utf-8")

        import argparse
        args = argparse.Namespace(file=str(file_path), force=False, json=False)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_ingest(args)
        assert exc_info.value.code == 1

    def test_ingest_allows_warnings_only(self, tmp_path: Path) -> None:
        """Ingest should proceed (not exit 1 from validation) if only warnings."""
        # This test just verifies validation doesn't block on warnings;
        # the actual ingest will fail later due to missing model, which is fine.
        from tract.validate import validate_framework

        fw_data = {
            "framework_id": "test_fw",
            "framework_name": "Test Framework",
            "version": "",  # triggers warning, not error
            "source_url": "https://example.com",
            "fetched_date": "2026-05-01",
            "mapping_unit_level": "control",
            "controls": [
                {
                    "control_id": "TC-01",
                    "title": "Test",
                    "description": "Short desc but still valid",
                },
            ],
        }
        issues = validate_framework(fw_data)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0  # Only warnings, no errors
```

- [ ] **Step 2: Run tests — verify gate test fails (no validation in ingest yet)**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py::TestIngestValidationGate::test_ingest_rejects_invalid_framework -v 2>&1 | tail -10
```

Expected: The ingest handler currently does not check validation rules, so it will try Pydantic validation (which may or may not fail) but won't produce our ValidationIssue errors and exit 1.

- [ ] **Step 3: Add validation gate to _cmd_ingest in tract/cli.py**

In the `_cmd_ingest` function, after the file size check and before `raw_data = load_json(file_path)`, add the validation gate. Replace the existing code block from `raw_data = load_json(file_path)` through the `FrameworkOutput.model_validate(raw_data)` try/except with:

```python
    raw_data = load_json(file_path)

    # ── Validation gate — abort on errors, warn on warnings ──────
    from tract.validate import validate_framework

    validation_issues = validate_framework(raw_data)
    val_errors = [i for i in validation_issues if i.severity == "error"]
    val_warnings = [i for i in validation_issues if i.severity == "warning"]

    if val_errors:
        print(f"Validation failed ({len(val_errors)} error(s)):", file=sys.stderr)
        for issue in val_errors:
            prefix = f"  [{issue.control_id}] " if issue.control_id else "  "
            print(f"{prefix}{issue.message}", file=sys.stderr)
        if val_warnings:
            print(f"\n  ({len(val_warnings)} warning(s) also found)", file=sys.stderr)
        sys.exit(1)

    if val_warnings:
        print(f"Validation warnings ({len(val_warnings)}):", file=sys.stderr)
        for issue in val_warnings:
            prefix = f"  [{issue.control_id}] " if issue.control_id else "  "
            print(f"{prefix}{issue.message}", file=sys.stderr)
        print("", file=sys.stderr)

    # Schema is already validated by validate_framework, but we need the typed object
    try:
        fw = FrameworkOutput.model_validate(raw_data)
    except Exception as e:
        print(f"Error: Schema validation failed: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py::TestIngestValidationGate -v
```

- [ ] **Step 5: Commit**

```
feat: add validation gate to tract ingest

Runs validate_framework() before model loading in `tract ingest`.
Errors block ingest with exit code 1 and a detailed report. Warnings
are printed to stderr but do not block.
```

---

### Task 4: CSV Extractor + Tests

**Files:**
- Create: `tract/prepare/__init__.py`
- Create: `tract/prepare/extract.py`
- Create: `tract/prepare/csv_extractor.py`
- Create: `tests/fixtures/sample_framework.csv`
- Create: `tests/test_csv_extractor.py`

- [ ] **Step 1: Create test fixture tests/fixtures/sample_framework.csv**

```csv
control_id,title,description
TC-01,Access Control,"Enforce access control policies for all system components and users to prevent unauthorized access"
TC-02,Data Encryption,"Encrypt sensitive data at rest and in transit using industry-standard cryptographic algorithms"
TC-03,Audit Logging,"Maintain comprehensive audit logs of security-relevant events for forensic analysis"
```

- [ ] **Step 2: Write tests in tests/test_csv_extractor.py**

```python
"""Tests for tract.prepare.csv_extractor — CSV format extraction."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from tract.prepare.csv_extractor import CsvExtractor
from tract.schema import Control


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a sample CSV file and return its path."""
    path = tmp_path / "framework.csv"
    path.write_text(
        'control_id,title,description\n'
        'TC-01,Access Control,"Enforce access control policies for all system components and users"\n'
        'TC-02,Data Encryption,"Encrypt sensitive data at rest and in transit using standard algorithms"\n'
        'TC-03,Audit Logging,"Maintain comprehensive audit logs of security events for forensic analysis"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def alternate_columns_csv(tmp_path: Path) -> Path:
    """CSV with alternate column names (id, name, text)."""
    path = tmp_path / "alt_cols.csv"
    path.write_text(
        'id,name,text\n'
        'A1,Access Control,"Enforce access control policies for all system components"\n'
        'A2,Encryption,"Encrypt data at rest and in transit using industry standards"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def csv_with_empty_rows(tmp_path: Path) -> Path:
    """CSV with blank rows that should be skipped."""
    path = tmp_path / "blanks.csv"
    path.write_text(
        'control_id,title,description\n'
        'TC-01,Access Control,"Enforce access control policies for all system components and users"\n'
        ',,\n'
        '  , , \n'
        'TC-02,Data Encryption,"Encrypt sensitive data at rest and in transit using standard algorithms"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def csv_with_full_text(tmp_path: Path) -> Path:
    """CSV including an optional full_text column."""
    path = tmp_path / "full_text.csv"
    path.write_text(
        'control_id,title,description,full_text\n'
        'TC-01,Access Control,"Short desc for access","Full detailed text about access control policies"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def csv_missing_columns(tmp_path: Path) -> Path:
    """CSV without a recognizable description column."""
    path = tmp_path / "missing.csv"
    path.write_text(
        'foo,bar,baz\n'
        '1,2,3\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tsv_file(tmp_path: Path) -> Path:
    """Tab-separated file."""
    path = tmp_path / "framework.tsv"
    path.write_text(
        'control_id\ttitle\tdescription\n'
        'TC-01\tAccess Control\tEnforce access control policies for all system components and users\n',
        encoding="utf-8",
    )
    return path


class TestCsvExtractor:
    def test_extract_standard_columns(self, sample_csv: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(sample_csv)
        assert len(controls) == 3
        assert all(isinstance(c, Control) for c in controls)
        assert controls[0].control_id == "TC-01"
        assert controls[0].title == "Access Control"
        assert "access control" in controls[0].description.lower()

    def test_extract_alternate_columns(self, alternate_columns_csv: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(alternate_columns_csv)
        assert len(controls) == 2
        assert controls[0].control_id == "A1"
        assert controls[0].title == "Access Control"

    def test_skips_empty_rows(self, csv_with_empty_rows: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(csv_with_empty_rows)
        assert len(controls) == 2

    def test_full_text_column(self, csv_with_full_text: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(csv_with_full_text)
        assert len(controls) == 1
        assert controls[0].full_text == "Full detailed text about access control policies"

    def test_missing_columns_raises(self, csv_missing_columns: Path) -> None:
        extractor = CsvExtractor()
        with pytest.raises(ValueError, match="(?i)column"):
            extractor.extract(csv_missing_columns)

    def test_tsv_file(self, tsv_file: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(tsv_file)
        assert len(controls) == 1
        assert controls[0].control_id == "TC-01"

    def test_returns_list_of_control_objects(self, sample_csv: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(sample_csv)
        for ctrl in controls:
            assert isinstance(ctrl, Control)
            assert ctrl.control_id
            assert ctrl.description
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_csv_extractor.py -q 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'tract.prepare'`

- [ ] **Step 4: Create tract/prepare/__init__.py**

```python
"""TRACT framework preparation pipeline.

Converts raw framework documents (CSV, markdown, JSON, unstructured)
into validated FrameworkOutput JSON for model inference.

Public API:
    prepare_framework(file_path, metadata, ...) -> Path
"""
```

- [ ] **Step 5: Create tract/prepare/extract.py — format detection + registry**

```python
"""Format detection and extractor registry for framework preparation.

Public API:
    detect_format(path) -> str
    ExtractorRegistry — maps format strings to extractor classes
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from tract.schema import Control

logger = logging.getLogger(__name__)

# File extension to format mapping
_EXTENSION_FORMAT_MAP: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".pdf": "unstructured",
    ".html": "unstructured",
    ".htm": "unstructured",
    ".txt": "unstructured",
    ".docx": "unstructured",
}


class Extractor(Protocol):
    """Protocol for format-specific control extractors."""

    def extract(self, path: Path) -> list[Control]: ...


def detect_format(path: Path) -> str:
    """Detect the document format from file extension.

    Args:
        path: Path to the raw framework document.

    Returns:
        One of "csv", "markdown", "json", "unstructured".

    Raises:
        ValueError: If the extension is not recognized.
    """
    suffix = path.suffix.lower()
    fmt = _EXTENSION_FORMAT_MAP.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Unrecognized file extension: {suffix!r}. "
            f"Use --format to specify one of: csv, markdown, json, unstructured"
        )
    logger.info("Detected format %r for %s (extension: %s)", fmt, path.name, suffix)
    return fmt


class ExtractorRegistry:
    """Registry mapping format names to extractor instances."""

    def __init__(self) -> None:
        self._extractors: dict[str, Extractor] = {}

    def register(self, format_name: str, extractor: Extractor) -> None:
        """Register an extractor for a format name."""
        self._extractors[format_name] = extractor

    def get(self, format_name: str) -> Extractor:
        """Get the extractor for a format name.

        Raises:
            ValueError: If no extractor is registered for the format.
        """
        extractor = self._extractors.get(format_name)
        if extractor is None:
            registered = ", ".join(sorted(self._extractors.keys()))
            raise ValueError(
                f"No extractor registered for format {format_name!r}. "
                f"Available: {registered}"
            )
        return extractor

    @property
    def formats(self) -> list[str]:
        """Return sorted list of registered format names."""
        return sorted(self._extractors.keys())
```

- [ ] **Step 6: Create tract/prepare/csv_extractor.py**

```python
"""CSV/TSV extractor for framework preparation.

Reads a CSV or TSV file with a header row and maps columns to Control
fields using flexible, case-insensitive column name matching.

Public API:
    CsvExtractor.extract(path) -> list[Control]
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from tract.schema import Control

logger = logging.getLogger(__name__)

# Case-insensitive column name aliases -> canonical field name
_COLUMN_ALIASES: dict[str, str] = {
    "control_id": "control_id",
    "id": "control_id",
    "control id": "control_id",
    "section_id": "control_id",
    "section id": "control_id",
    "title": "title",
    "name": "title",
    "control_name": "title",
    "control name": "title",
    "description": "description",
    "desc": "description",
    "text": "description",
    "body": "description",
    "full_text": "full_text",
    "fulltext": "full_text",
    "full text": "full_text",
}


def _resolve_columns(
    header: list[str],
) -> dict[str, str]:
    """Map CSV header columns to canonical Control field names.

    Args:
        header: List of column names from the CSV header row.

    Returns:
        Dict mapping canonical field name -> CSV column name.

    Raises:
        ValueError: If required columns (control_id and description) cannot
            be identified.
    """
    resolved: dict[str, str] = {}  # canonical -> original CSV column name

    for col in header:
        normalized = col.strip().lower()
        canonical = _COLUMN_ALIASES.get(normalized)
        if canonical and canonical not in resolved:
            resolved[canonical] = col

    missing = []
    if "control_id" not in resolved:
        missing.append("control_id (or: id, section_id)")
    if "description" not in resolved:
        missing.append("description (or: desc, text, body)")

    if missing:
        raise ValueError(
            f"CSV missing required column(s): {', '.join(missing)}. "
            f"Found columns: {header}"
        )

    return resolved


class CsvExtractor:
    """Extract controls from a CSV or TSV file."""

    def extract(self, path: Path) -> list[Control]:
        """Read a CSV/TSV and convert rows to Control objects.

        Column mapping is case-insensitive and flexible (see _COLUMN_ALIASES).
        Rows where all mapped fields are empty are skipped.

        Args:
            path: Path to the CSV or TSV file.

        Returns:
            List of Control objects extracted from the file.

        Raises:
            ValueError: If required columns are missing.
            FileNotFoundError: If the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Detect delimiter: TSV if .tsv extension, otherwise comma
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

        text = path.read_text(encoding="utf-8")
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")

        col_map = _resolve_columns(list(reader.fieldnames))
        logger.info(
            "CSV column mapping for %s: %s",
            path.name,
            {k: v for k, v in col_map.items()},
        )

        controls: list[Control] = []
        for row_num, row in enumerate(reader, start=2):  # 2 because header is row 1
            # Extract mapped fields
            control_id = row.get(col_map["control_id"], "").strip()
            description = row.get(col_map["description"], "").strip()
            title = row.get(col_map.get("title", ""), "").strip() if "title" in col_map else ""
            full_text = (
                row.get(col_map.get("full_text", ""), "").strip()
                if "full_text" in col_map
                else None
            )

            # Skip rows where all mapped fields are empty
            if not control_id and not description and not title:
                logger.debug("Skipping empty row %d in %s", row_num, path.name)
                continue

            # Use row number as fallback control_id if empty
            if not control_id:
                control_id = f"ROW-{row_num}"
                logger.warning(
                    "Row %d in %s has no control_id, using %s",
                    row_num, path.name, control_id,
                )

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=description if description else title,
                full_text=full_text if full_text else None,
            ))

        logger.info("Extracted %d controls from CSV %s", len(controls), path.name)
        return controls
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_csv_extractor.py -v
```

- [ ] **Step 8: Commit**

```
feat: add CSV extractor with flexible column mapping

Implements CsvExtractor for the framework preparation pipeline.
Supports case-insensitive column name aliases (control_id/id,
title/name, description/desc/text), TSV files, empty row skipping,
and optional full_text column. Includes ExtractorRegistry and
detect_format() for the extraction framework.
```

---

### Task 5: Markdown Extractor + Tests

**Files:**
- Create: `tract/prepare/markdown_extractor.py`
- Create: `tests/fixtures/sample_framework.md`
- Create: `tests/test_markdown_extractor.py`

- [ ] **Step 1: Create test fixture tests/fixtures/sample_framework.md**

```markdown
# Security Framework

## ASI01: Access Control

Enforce access control policies for all system components to prevent unauthorized access to AI models and data.

## ASI02: Data Encryption

Encrypt sensitive AI training data at rest and in transit using industry-standard cryptographic algorithms.

## ASI03: Audit Logging

Maintain comprehensive audit logs of all security-relevant events for forensic analysis and compliance.
```

- [ ] **Step 2: Write tests in tests/test_markdown_extractor.py**

```python
"""Tests for tract.prepare.markdown_extractor — Markdown format extraction."""
from __future__ import annotations

from pathlib import Path

import pytest

from tract.prepare.markdown_extractor import MarkdownExtractor
from tract.schema import Control


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    """Markdown with h2 headings and ID prefixes."""
    content = (
        "# Security Framework\n\n"
        "## ASI01: Access Control\n\n"
        "Enforce access control policies for all system components "
        "to prevent unauthorized access to AI models and data.\n\n"
        "## ASI02: Data Encryption\n\n"
        "Encrypt sensitive AI training data at rest and in transit "
        "using industry-standard cryptographic algorithms.\n\n"
        "## ASI03: Audit Logging\n\n"
        "Maintain comprehensive audit logs of all security-relevant "
        "events for forensic analysis and compliance.\n"
    )
    path = tmp_path / "framework.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def h3_headings_md(tmp_path: Path) -> Path:
    """Markdown using h3 headings."""
    content = (
        "# Top Level\n\n"
        "## Category A\n\n"
        "### CTRL-01 - First Control\n\n"
        "Description of the first security control for access management.\n\n"
        "### CTRL-02 - Second Control\n\n"
        "Description of the second security control for data protection.\n\n"
        "## Category B\n\n"
        "### CTRL-03 - Third Control\n\n"
        "Description of the third security control for audit logging.\n"
    )
    path = tmp_path / "h3_framework.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def numbered_headings_md(tmp_path: Path) -> Path:
    """Markdown with 1.2.3 style numbering in headings."""
    content = (
        "# Framework\n\n"
        "## 1.1 Access Control\n\n"
        "Enforce access control policies for system components and users.\n\n"
        "## 1.2 Encryption\n\n"
        "Encrypt all sensitive data at rest and in transit securely.\n\n"
        "## 2.1 Logging\n\n"
        "Log all security events for monitoring and forensic analysis.\n"
    )
    path = tmp_path / "numbered.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def no_id_headings_md(tmp_path: Path) -> Path:
    """Markdown with headings that have no recognizable ID pattern."""
    content = (
        "# Framework\n\n"
        "## Access Control\n\n"
        "Enforce access control policies for system components and users.\n\n"
        "## Data Protection\n\n"
        "Protect data at rest and in transit using encryption.\n"
    )
    path = tmp_path / "no_id.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def no_headings_md(tmp_path: Path) -> Path:
    """Markdown with no extractable headings."""
    content = "Just a paragraph of text with no headings at all.\n"
    path = tmp_path / "no_headings.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestMarkdownExtractor:
    def test_extract_h2_with_id_prefix(self, sample_md: Path) -> None:
        extractor = MarkdownExtractor()
        controls = extractor.extract(sample_md)
        assert len(controls) == 3
        assert controls[0].control_id == "ASI01"
        assert controls[0].title == "Access Control"
        assert "access control" in controls[0].description.lower()

    def test_extract_h3_headings_with_level_override(self, h3_headings_md: Path) -> None:
        extractor = MarkdownExtractor(heading_level=3)
        controls = extractor.extract(h3_headings_md)
        assert len(controls) == 3
        assert controls[0].control_id == "CTRL-01"
        assert controls[0].title == "First Control"

    def test_extract_numbered_headings(self, numbered_headings_md: Path) -> None:
        extractor = MarkdownExtractor()
        controls = extractor.extract(numbered_headings_md)
        assert len(controls) == 3
        assert controls[0].control_id == "1.1"
        assert controls[0].title == "Access Control"

    def test_fallback_positional_id(self, no_id_headings_md: Path) -> None:
        extractor = MarkdownExtractor()
        controls = extractor.extract(no_id_headings_md)
        assert len(controls) == 2
        assert controls[0].control_id == "CTRL-01"
        assert controls[0].title == "Access Control"
        assert controls[1].control_id == "CTRL-02"

    def test_no_headings_raises(self, no_headings_md: Path) -> None:
        extractor = MarkdownExtractor()
        with pytest.raises(ValueError, match="(?i)no.*heading"):
            extractor.extract(no_headings_md)

    def test_autodetect_heading_level(self, h3_headings_md: Path) -> None:
        """Auto-detect should find h3 as the content-bearing level."""
        extractor = MarkdownExtractor()  # No explicit heading_level
        controls = extractor.extract(h3_headings_md)
        assert len(controls) == 3

    def test_returns_control_objects(self, sample_md: Path) -> None:
        extractor = MarkdownExtractor()
        controls = extractor.extract(sample_md)
        for ctrl in controls:
            assert isinstance(ctrl, Control)
            assert ctrl.control_id
            assert ctrl.description
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_markdown_extractor.py -q 2>&1 | head -10
```

Expected: `ModuleNotFoundError`

- [ ] **Step 4: Create tract/prepare/markdown_extractor.py**

```python
"""Markdown extractor for framework preparation.

Splits markdown documents on heading patterns. Each heading becomes a
control: heading text -> title, body text -> description, ID extracted
from heading prefix patterns or generated positionally.

Public API:
    MarkdownExtractor.extract(path) -> list[Control]
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.schema import Control

logger = logging.getLogger(__name__)

# Patterns for extracting control IDs from heading text.
# Order matters: try more specific patterns first.
_ID_PATTERNS: list[re.Pattern[str]] = [
    # "ASI01:" or "ASI01 -" style (letter prefix + digits + separator)
    re.compile(r"^([A-Za-z]{2,10}\d+(?:\.\d+)*)\s*[:\-–—]\s*(.+)$"),
    # "CTRL-01 -" or "CTRL_01:" style
    re.compile(r"^([A-Za-z]+-\d+(?:\.\d+)*)\s*[:\-–—]\s*(.+)$"),
    # "1.2.3 Title" style (numbered sections)
    re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$"),
    # "1.2 Title" style (two-level numbered)
    re.compile(r"^(\d+\.\d+)\s+(.+)$"),
]

# Heading pattern: matches lines like "## Heading text"
_HEADING_RE: re.Pattern[str] = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _parse_heading_id(text: str) -> tuple[str | None, str]:
    """Extract a control ID and clean title from heading text.

    Args:
        text: Raw heading text (without the # prefix).

    Returns:
        (control_id, title) — control_id is None if no pattern matched.
    """
    text = text.strip()
    for pattern in _ID_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1), match.group(2).strip()
    return None, text


def _detect_heading_level(text: str) -> int:
    """Auto-detect the heading level that has the most body text.

    Finds the heading level where the most headings are followed by
    non-empty body text. This heuristic picks the "content" level,
    ignoring top-level titles and category headings with no body.

    Args:
        text: Full markdown text.

    Returns:
        The heading depth (1-6) to extract controls from.
    """
    level_counts: dict[int, int] = {}
    lines = text.split("\n")

    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            # Check if there's body text below (non-empty, non-heading lines)
            has_body = False
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith("#"):
                    has_body = True
                    break
                if next_line.startswith("#"):
                    break
            if has_body:
                level_counts[level] = level_counts.get(level, 0) + 1

    if not level_counts:
        return 2  # Default to h2

    # Pick the level with the most body-bearing headings
    best_level = max(level_counts, key=lambda k: level_counts[k])
    logger.info(
        "Auto-detected heading level %d (counts: %s)",
        best_level,
        level_counts,
    )
    return best_level


class MarkdownExtractor:
    """Extract controls from a markdown document by splitting on headings."""

    def __init__(self, heading_level: int | None = None) -> None:
        """Initialize the markdown extractor.

        Args:
            heading_level: Heading depth to split on (1-6). If None,
                auto-detects the most common heading level with body text.
        """
        self._heading_level = heading_level

    def extract(self, path: Path) -> list[Control]:
        """Read a markdown file and extract controls from headings.

        Each heading at the target level becomes one control. The heading
        text is parsed for an ID prefix; if none is found, a positional
        ID (CTRL-01, CTRL-02, ...) is generated.

        Args:
            path: Path to the markdown file.

        Returns:
            List of Control objects extracted from the document.

        Raises:
            ValueError: If no headings are found at the target level.
            FileNotFoundError: If the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        text = path.read_text(encoding="utf-8")

        level = self._heading_level or _detect_heading_level(text)
        prefix = "#" * level

        # Find all headings at the target level
        sections: list[tuple[str, str]] = []  # (heading_text, body_text)
        lines = text.split("\n")
        current_heading: str | None = None
        body_lines: list[str] = []

        for line in lines:
            match = _HEADING_RE.match(line)
            if match and len(match.group(1)) == level:
                # Save previous section
                if current_heading is not None:
                    sections.append((current_heading, "\n".join(body_lines).strip()))
                current_heading = match.group(2).strip()
                body_lines = []
            elif match and len(match.group(1)) < level:
                # Higher-level heading: save current section and reset
                if current_heading is not None:
                    sections.append((current_heading, "\n".join(body_lines).strip()))
                current_heading = None
                body_lines = []
            elif current_heading is not None:
                body_lines.append(line)

        # Save final section
        if current_heading is not None:
            sections.append((current_heading, "\n".join(body_lines).strip()))

        if not sections:
            raise ValueError(
                f"No headings found at level {level} in {path.name}. "
                "Try using --heading-level to specify a different level, "
                "or use --llm for unstructured extraction."
            )

        controls: list[Control] = []
        positional_counter = 0

        for heading_text, body_text in sections:
            if not body_text:
                logger.debug("Skipping heading with no body: %r", heading_text)
                continue

            control_id, title = _parse_heading_id(heading_text)

            if control_id is None:
                positional_counter += 1
                control_id = f"CTRL-{positional_counter:02d}"

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=body_text,
            ))

        if not controls:
            raise ValueError(
                f"No controls with body text found at heading level {level} "
                f"in {path.name}."
            )

        logger.info(
            "Extracted %d controls from markdown %s (heading level %d)",
            len(controls), path.name, level,
        )
        return controls
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_markdown_extractor.py -v
```

- [ ] **Step 6: Commit**

```
feat: add Markdown extractor with heading-level auto-detection

Implements MarkdownExtractor for the framework preparation pipeline.
Splits markdown documents on heading patterns, extracts control IDs
from heading prefixes (ASI01:, CTRL-01 -, 1.2.3 style), falls back
to positional IDs. Auto-detects the content-bearing heading level
when --heading-level is not specified.
```

---

### Task 6: JSON Extractor + Tests

**Files:**
- Create: `tract/prepare/json_extractor.py`
- Create: `tests/fixtures/sample_framework_array.json`
- Create: `tests/test_json_extractor.py`

- [ ] **Step 1: Create test fixture tests/fixtures/sample_framework_array.json**

```json
[
  {"id": "TC-01", "name": "Access Control", "description": "Enforce access control policies for system components and users"},
  {"id": "TC-02", "name": "Encryption", "description": "Encrypt sensitive data at rest and in transit using standard algorithms"},
  {"id": "TC-03", "name": "Logging", "description": "Maintain audit logs of security-relevant events for forensic analysis"}
]
```

- [ ] **Step 2: Write tests in tests/test_json_extractor.py**

```python
"""Tests for tract.prepare.json_extractor — JSON format extraction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.prepare.json_extractor import JsonExtractor
from tract.schema import Control


@pytest.fixture
def framework_output_json(tmp_path: Path) -> Path:
    """A JSON file that is already a valid FrameworkOutput."""
    data = {
        "framework_id": "test_fw",
        "framework_name": "Test Framework",
        "version": "1.0",
        "source_url": "https://example.com",
        "fetched_date": "2026-05-01",
        "mapping_unit_level": "control",
        "controls": [
            {
                "control_id": "TC-01",
                "title": "Access Control",
                "description": "Enforce access control policies for system components and users",
            },
            {
                "control_id": "TC-02",
                "title": "Encryption",
                "description": "Encrypt sensitive data at rest and in transit using standard algorithms",
            },
        ],
    }
    path = tmp_path / "framework_output.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def array_json(tmp_path: Path) -> Path:
    """A JSON file that is an array of objects."""
    data = [
        {"id": "TC-01", "name": "Access Control", "description": "Enforce access control policies for components"},
        {"id": "TC-02", "name": "Encryption", "description": "Encrypt data at rest and in transit securely"},
    ]
    path = tmp_path / "array.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def nested_controls_json(tmp_path: Path) -> Path:
    """A JSON file with controls nested under a 'controls' key."""
    data = {
        "name": "Some Framework",
        "controls": [
            {"control_id": "TC-01", "title": "Access", "body": "Enforce access control policies for system users"},
            {"control_id": "TC-02", "title": "Encrypt", "text": "Encrypt data at rest and in transit using standards"},
        ],
    }
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def nested_items_json(tmp_path: Path) -> Path:
    """A JSON file with controls nested under an 'items' key."""
    data = {
        "items": [
            {"section_id": "S-01", "title": "First", "desc": "Description of first control for testing"},
        ],
    }
    path = tmp_path / "items.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def nested_data_json(tmp_path: Path) -> Path:
    """A JSON file with controls nested under a 'data' key."""
    data = {
        "data": [
            {"id": "D-01", "name": "Item", "text": "Description of the data item for extraction testing"},
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def unrecognizable_json(tmp_path: Path) -> Path:
    """A JSON file with no recognizable control structure."""
    data = {"config": {"setting": True}, "metadata": {"author": "test"}}
    path = tmp_path / "unrecognizable.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestJsonExtractor:
    def test_passthrough_framework_output(self, framework_output_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(framework_output_json)
        assert len(controls) == 2
        assert controls[0].control_id == "TC-01"
        assert controls[0].title == "Access Control"

    def test_array_of_objects(self, array_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(array_json)
        assert len(controls) == 2
        assert controls[0].control_id == "TC-01"
        assert controls[0].title == "Access Control"

    def test_nested_controls_key(self, nested_controls_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(nested_controls_json)
        assert len(controls) == 2
        assert controls[0].control_id == "TC-01"
        assert "access control" in controls[0].description.lower()

    def test_nested_items_key(self, nested_items_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(nested_items_json)
        assert len(controls) == 1
        assert controls[0].control_id == "S-01"

    def test_nested_data_key(self, nested_data_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(nested_data_json)
        assert len(controls) == 1
        assert controls[0].control_id == "D-01"

    def test_unrecognizable_raises(self, unrecognizable_json: Path) -> None:
        extractor = JsonExtractor()
        with pytest.raises(ValueError, match="(?i)no recognizable"):
            extractor.extract(unrecognizable_json)

    def test_returns_control_objects(self, array_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(array_json)
        for ctrl in controls:
            assert isinstance(ctrl, Control)
            assert ctrl.control_id
            assert ctrl.description
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_json_extractor.py -q 2>&1 | head -10
```

Expected: `ModuleNotFoundError`

- [ ] **Step 4: Create tract/prepare/json_extractor.py**

```python
"""JSON extractor for framework preparation.

Handles three cases:
1. Already a valid FrameworkOutput — passthrough (re-sanitize text fields).
2. Top-level JSON array of objects — map fields heuristically.
3. Object with a "controls"/"items"/"data" array — extract from that key.

Public API:
    JsonExtractor.extract(path) -> list[Control]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tract.schema import Control, FrameworkOutput

logger = logging.getLogger(__name__)

# Key aliases for heuristic field mapping (lowercase -> canonical)
_ID_KEYS: tuple[str, ...] = ("control_id", "id", "section_id", "control id")
_TITLE_KEYS: tuple[str, ...] = ("title", "name", "control_name", "control name")
_DESC_KEYS: tuple[str, ...] = ("description", "desc", "text", "body")
_FULL_TEXT_KEYS: tuple[str, ...] = ("full_text", "fulltext", "full text")

# Top-level keys that may contain the control array
_ARRAY_KEYS: tuple[str, ...] = ("controls", "items", "data")


def _find_key(obj: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    """Find the first matching key in obj (case-insensitive)."""
    lower_map = {k.lower(): k for k in obj.keys()}
    for candidate in candidates:
        real_key = lower_map.get(candidate.lower())
        if real_key is not None:
            return real_key
    return None


def _map_object_to_control(obj: dict[str, Any], index: int) -> Control:
    """Map a generic dict to a Control using heuristic key matching.

    Args:
        obj: A dict representing one control.
        index: 1-based position for fallback ID generation.

    Returns:
        A Control object.

    Raises:
        ValueError: If no description-like field is found.
    """
    id_key = _find_key(obj, _ID_KEYS)
    title_key = _find_key(obj, _TITLE_KEYS)
    desc_key = _find_key(obj, _DESC_KEYS)
    full_text_key = _find_key(obj, _FULL_TEXT_KEYS)

    control_id = str(obj[id_key]).strip() if id_key else f"ITEM-{index:03d}"
    title = str(obj[title_key]).strip() if title_key else ""

    if desc_key:
        description = str(obj[desc_key]).strip()
    elif title:
        description = title
    else:
        raise ValueError(
            f"Object at index {index} has no recognizable description field. "
            f"Keys found: {list(obj.keys())}"
        )

    full_text = str(obj[full_text_key]).strip() if full_text_key and obj.get(full_text_key) else None

    return Control(
        control_id=control_id,
        title=title,
        description=description,
        full_text=full_text,
    )


class JsonExtractor:
    """Extract controls from a JSON file."""

    def extract(self, path: Path) -> list[Control]:
        """Read a JSON file and extract controls.

        Handles FrameworkOutput passthrough, top-level arrays, and
        objects with nested control arrays.

        Args:
            path: Path to the JSON file.

        Returns:
            List of Control objects extracted from the file.

        Raises:
            ValueError: If no recognizable control structure is found.
            FileNotFoundError: If the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        text = path.read_text(encoding="utf-8")
        data = json.loads(text)

        # Case 1: Already a FrameworkOutput — passthrough
        if isinstance(data, dict):
            try:
                fw = FrameworkOutput.model_validate(data)
                logger.info(
                    "JSON is already FrameworkOutput (%d controls), using passthrough",
                    len(fw.controls),
                )
                return list(fw.controls)
            except ValidationError:
                pass  # Not a valid FrameworkOutput, try other cases

        # Case 2: Top-level array
        if isinstance(data, list):
            return self._extract_from_array(data)

        # Case 3: Object with a recognized array key
        if isinstance(data, dict):
            for key in _ARRAY_KEYS:
                lower_map = {k.lower(): k for k in data.keys()}
                real_key = lower_map.get(key)
                if real_key and isinstance(data[real_key], list):
                    logger.info("Found control array under key %r", real_key)
                    return self._extract_from_array(data[real_key])

        raise ValueError(
            f"No recognizable control structure found in {path.name}. "
            f"Expected: FrameworkOutput JSON, an array of objects, or an object "
            f"with a 'controls'/'items'/'data' array key. "
            f"Try --llm for complex nested formats."
        )

    def _extract_from_array(self, items: list[Any]) -> list[Control]:
        """Extract controls from a list of dicts.

        Args:
            items: List of dicts, each representing one control.

        Returns:
            List of Control objects.
        """
        controls: list[Control] = []
        for i, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict item at index %d: %r", i, type(item))
                continue
            controls.append(_map_object_to_control(item, i))

        logger.info("Extracted %d controls from JSON array", len(controls))
        return controls
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_json_extractor.py -v
```

- [ ] **Step 6: Commit**

```
feat: add JSON extractor with heuristic field mapping

Implements JsonExtractor for the framework preparation pipeline.
Handles three cases: FrameworkOutput passthrough, top-level arrays
of objects, and objects with nested control arrays (controls/items/data
keys). Uses case-insensitive heuristic key matching for control_id,
title, description, and full_text fields.
```

---

### Task 7: prepare_framework() Orchestrator + Tests

**Files:**
- Modify: `tract/prepare/__init__.py`
- Create: `tests/test_prepare.py`

- [ ] **Step 1: Write tests in tests/test_prepare.py**

```python
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
        result_path = prepare_framework(
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
        result_path = prepare_framework(
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
        # Cleanup
        expected.unlink()

    def test_format_override(self, csv_file: Path, metadata: dict, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        result_path = prepare_framework(
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
        """Verify that sanitize_text() is applied to extracted text fields."""
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
        # Whitespace should be collapsed
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
        # Must be a valid FrameworkOutput
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare.py -q 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'prepare_framework'`

- [ ] **Step 3: Implement prepare_framework() in tract/prepare/__init__.py**

```python
"""TRACT framework preparation pipeline.

Converts raw framework documents (CSV, markdown, JSON, unstructured)
into validated FrameworkOutput JSON for model inference.

Public API:
    prepare_framework(file_path, metadata, ...) -> Path
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from tract.config import DESCRIPTION_MAX_LENGTH
from tract.io import atomic_write_json
from tract.prepare.csv_extractor import CsvExtractor
from tract.prepare.extract import ExtractorRegistry, detect_format
from tract.prepare.json_extractor import JsonExtractor
from tract.prepare.markdown_extractor import MarkdownExtractor
from tract.sanitize import sanitize_text
from tract.schema import Control, FrameworkOutput
from tract.validate import validate_framework

logger = logging.getLogger(__name__)


def _build_registry(heading_level: int | None = None) -> ExtractorRegistry:
    """Build the default extractor registry.

    Args:
        heading_level: Optional heading level override for MarkdownExtractor.

    Returns:
        A populated ExtractorRegistry.
    """
    registry = ExtractorRegistry()
    registry.register("csv", CsvExtractor())
    registry.register("markdown", MarkdownExtractor(heading_level=heading_level))
    registry.register("json", JsonExtractor())
    return registry


def _sanitize_control(control: Control) -> Control:
    """Sanitize all text fields of a control using the TRACT pipeline.

    Mirrors BaseParser._sanitize_control() from tract/parsers/base.py.

    Args:
        control: A Control with potentially unsanitized text.

    Returns:
        A new Control with sanitized text fields.
    """
    sanitized_desc, full_text = sanitize_text(
        control.description,
        max_length=DESCRIPTION_MAX_LENGTH,
        return_full=True,
    )

    sanitized_full: str | None = full_text
    if control.full_text is not None and full_text is None:
        sanitized_full = sanitize_text(
            control.full_text,
            max_length=50_000,
        )

    sanitized_title: str = (
        sanitize_text(control.title, max_length=500)
        if control.title
        else ""
    )

    return Control(
        control_id=control.control_id,
        title=sanitized_title,
        description=sanitized_desc,
        full_text=sanitized_full,
    )


def prepare_framework(
    *,
    file_path: Path,
    framework_id: str,
    name: str,
    version: str,
    source_url: str,
    mapping_unit: str,
    output_path: Path | None = None,
    format_override: str | None = None,
    use_llm: bool = False,
    heading_level: int | None = None,
) -> Path:
    """Prepare a raw framework document into a validated FrameworkOutput JSON.

    This is the main orchestrator for the preparation pipeline:
    1. Detect or override format
    2. Extract controls using the appropriate extractor
    3. Sanitize all text fields
    4. Assemble FrameworkOutput
    5. Run validation (report issues to stderr, but still write output)
    6. Write output JSON atomically

    Args:
        file_path: Path to the raw framework document.
        framework_id: Canonical framework ID slug.
        name: Human-readable framework name.
        version: Framework version string.
        source_url: Official URL for the framework.
        mapping_unit: What each control represents (control, technique, etc.).
        output_path: Where to write the output JSON. Defaults to
            <input_stem>_prepared.json in the same directory.
        format_override: Override auto-detected format (csv, markdown, json,
            unstructured).
        use_llm: Use Claude API for LLM-assisted extraction.
        heading_level: Override heading level for markdown extraction.

    Returns:
        Path to the written output JSON file.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If format is unstructured and --llm is not set, or if
            extraction produces zero controls.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Determine format
    fmt = format_override or detect_format(file_path)
    logger.info("Preparing %s (format: %s)", file_path.name, fmt)

    # Handle unstructured format
    if fmt == "unstructured":
        if not use_llm:
            raise ValueError(
                f"File {file_path.name} detected as unstructured format. "
                "Use --llm for LLM-assisted extraction, or --format to "
                "override the detected format (csv, markdown, json)."
            )
        # LLM extraction is handled in Task 9
        from tract.prepare.llm_extractor import LlmExtractor
        extractor = LlmExtractor()
        raw_controls = extractor.extract(
            file_path,
            framework_id=framework_id,
            output_dir=output_path.parent if output_path else file_path.parent,
        )
    else:
        # Use registry for structured formats
        registry = _build_registry(heading_level=heading_level)
        extractor = registry.get(fmt)
        raw_controls = extractor.extract(file_path)

    if not raw_controls:
        raise ValueError(
            f"Extraction produced zero controls from {file_path.name}."
        )

    # Sanitize all text fields
    sanitized_controls = [_sanitize_control(c) for c in raw_controls]

    # Assemble FrameworkOutput
    fw = FrameworkOutput(
        framework_id=framework_id,
        framework_name=name,
        version=version,
        source_url=source_url,
        fetched_date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        mapping_unit_level=mapping_unit,
        controls=sanitized_controls,
    )

    # Run validation — report but always write
    issues = validate_framework(fw.model_dump(mode="json", exclude_none=True))
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        logger.warning(
            "Prepared output has %d validation error(s) — writing anyway for inspection",
            len(errors),
        )
        for issue in errors:
            prefix = f"[{issue.control_id}] " if issue.control_id else ""
            print(f"  ERROR: {prefix}{issue.message}", file=sys.stderr)

    if warnings:
        for issue in warnings:
            prefix = f"[{issue.control_id}] " if issue.control_id else ""
            print(f"  WARNING: {prefix}{issue.message}", file=sys.stderr)

    # Determine output path
    if output_path is None:
        output_path = file_path.with_name(f"{file_path.stem}_prepared.json")

    # Write atomically
    atomic_write_json(
        fw.model_dump(mode="json", exclude_none=True),
        output_path,
    )

    logger.info(
        "Wrote %d controls to %s (%d errors, %d warnings)",
        len(sanitized_controls),
        output_path,
        len(errors),
        len(warnings),
    )

    return output_path
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare.py -v
```

- [ ] **Step 5: Commit**

```
feat: add prepare_framework() orchestrator

Implements the main preparation pipeline: format detection, extraction
via registered extractors, sanitization of all text fields, assembly
into FrameworkOutput, validation with reporting, and atomic JSON write.
Writes output even if validation errors exist so users can inspect and
fix manually.
```

---

### Task 8: `tract prepare` CLI Subcommand + Tests

**Files:**
- Modify: `tract/cli.py`
- Add to: `tests/test_prepare_cli.py`

- [ ] **Step 1: Write CLI tests for the prepare subcommand**

Add to `tests/test_prepare_cli.py`:

```python
class TestPrepareArgParsing:
    def test_prepare_required_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "framework.csv",
            "--framework-id", "test_fw",
            "--name", "Test Framework",
            "--version", "1.0",
            "--source-url", "https://example.com",
            "--mapping-unit", "control",
        ])
        assert args.command == "prepare"
        assert args.file == "framework.csv"
        assert args.framework_id == "test_fw"
        assert args.name == "Test Framework"
        assert args.version == "1.0"
        assert args.source_url == "https://example.com"
        assert args.mapping_unit == "control"

    def test_prepare_optional_llm_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "prepare",
            "--file", "doc.pdf",
            "--framework-id", "test_fw",
            "--name", "Test",
            "--version", "1.0",
            "--source-url", "https://example.com",
            "--mapping-unit", "control",
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
            "--version", "1.0",
            "--source-url", "https://example.com",
            "--mapping-unit", "control",
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
            "--version", "1.0",
            "--source-url", "https://example.com",
            "--mapping-unit", "control",
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
            "--version", "1.0",
            "--source-url", "https://example.com",
            "--mapping-unit", "control",
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
            source_url="https://example.com",
            mapping_unit="control",
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
            source_url="https://example.com",
            mapping_unit="control",
            output=None,
            format=None,
            llm=False,
            heading_level=None,
            json=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_prepare(args)
        assert exc_info.value.code == 1
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py::TestPrepareArgParsing -q 2>&1 | head -10
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py::TestPrepareCommand -q 2>&1 | head -10
```

Expected: failures because `prepare` subcommand and `_cmd_prepare` do not exist yet.

- [ ] **Step 3: Add the prepare subcommand to build_parser() in tract/cli.py**

Add after the `p_validate` subparser definition:

```python
    # ── prepare ──────────────────────────────────────────────────────
    p_prepare = subparsers.add_parser(
        "prepare",
        help="Prepare a raw framework document for ingestion",
        epilog=(
            "Examples:\n"
            "  tract prepare --file controls.csv --framework-id new_fw \\\n"
            "    --name 'New Framework' --version '1.0' \\\n"
            "    --source-url 'https://example.com' --mapping-unit control\n"
            "\n"
            "  tract prepare --file doc.pdf --llm --framework-id new_fw \\\n"
            "    --name 'New Framework' --version '1.0' \\\n"
            "    --source-url 'https://example.com' --mapping-unit control\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_prepare.add_argument("--file", required=True, help="Input file path (CSV, markdown, JSON, or unstructured)")
    p_prepare.add_argument("--framework-id", required=True, help="Framework ID slug (lowercase, underscores)")
    p_prepare.add_argument("--name", required=True, help="Human-readable framework name")
    p_prepare.add_argument("--version", required=True, help="Framework version string")
    p_prepare.add_argument("--source-url", required=True, help="Official framework URL")
    p_prepare.add_argument("--mapping-unit", required=True, help="What each control represents (control, technique, risk, article)")
    p_prepare.add_argument("--llm", action="store_true", help="Use Claude API for LLM-assisted extraction")
    p_prepare.add_argument("--format", choices=["csv", "markdown", "json", "unstructured"], help="Override auto-detected format")
    p_prepare.add_argument("--output", help="Output file path (default: <input_stem>_prepared.json)")
    p_prepare.add_argument("--heading-level", type=int, help="Markdown heading depth to split on (default: auto-detect)")
    p_prepare.add_argument("--json", action="store_true", help="Output summary as JSON")
```

- [ ] **Step 4: Add the _cmd_prepare handler function**

Add after `_cmd_validate` in `tract/cli.py`:

```python
def _cmd_prepare(args: argparse.Namespace) -> None:
    from tract.prepare import prepare_framework

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    try:
        result_path = prepare_framework(
            file_path=file_path,
            framework_id=args.framework_id,
            name=args.name,
            version=args.version,
            source_url=args.source_url,
            mapping_unit=args.mapping_unit,
            output_path=output_path,
            format_override=args.format,
            use_llm=args.llm,
            heading_level=args.heading_level,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        from tract.io import load_json
        data = load_json(result_path)
        summary = {
            "output_path": str(result_path),
            "framework_id": data["framework_id"],
            "controls": len(data["controls"]),
        }
        print(json.dumps(summary, indent=2))
    else:
        from tract.io import load_json
        data = load_json(result_path)
        print(f"Prepared: {data['framework_name']} ({data['framework_id']})")
        print(f"  Controls: {len(data['controls'])}")
        print(f"  Output: {result_path}")
```

- [ ] **Step 5: Register the prepare handler in the main() dispatch table**

In the `handlers` dict inside `main()`, add:

```python
        "prepare": _cmd_prepare,
```

- [ ] **Step 6: Update test_all_commands_have_help in tests/test_cli.py**

Add `"prepare"` to the command list:

```python
        for cmd in ["assign", "compare", "ingest", "accept", "export", "hierarchy",
                     "propose-hubs", "review-proposals", "tutorial", "validate", "prepare"]:
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_cli.py tests/test_cli.py::TestArgParsing::test_all_commands_have_help -v
```

- [ ] **Step 8: Commit**

```
feat: add tract prepare CLI subcommand

Adds `tract prepare --file <path> --framework-id <id> ...` command
for converting raw framework documents (CSV, markdown, JSON) into
validated FrameworkOutput JSON. Supports --llm for unstructured docs,
--format override, --heading-level for markdown, and --output for
custom output paths.
```

---

### Task 9: LLM Extractor + Tests (Mock API Calls)

**Files:**
- Create: `tract/prepare/llm_extractor.py`
- Create: `tests/test_llm_extractor.py`

- [ ] **Step 1: Write tests in tests/test_llm_extractor.py**

```python
"""Tests for tract.prepare.llm_extractor — LLM-assisted extraction with mocked API."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tract.prepare.llm_extractor import LlmExtractor, _build_tool_schema
from tract.schema import Control


def _mock_api_response(controls: list[dict]) -> MagicMock:
    """Build a mock Anthropic API response with tool_use content."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "extract_controls"
    tool_use_block.input = {"controls": controls}

    response = MagicMock()
    response.content = [tool_use_block]
    response.stop_reason = "tool_use"
    return response


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """A plain text file for LLM extraction."""
    content = (
        "AI Security Framework v1.0\n\n"
        "1. Access Control\n"
        "Organizations must enforce strict access control policies "
        "for AI model endpoints and training pipelines.\n\n"
        "2. Data Protection\n"
        "Sensitive training data must be encrypted at rest and in "
        "transit using industry-standard cryptographic methods.\n\n"
        "3. Model Monitoring\n"
        "Continuously monitor AI model behavior for drift, adversarial "
        "inputs, and unexpected output patterns.\n"
    )
    path = tmp_path / "framework.txt"
    path.write_text(content, encoding="utf-8")
    return path


class TestBuildToolSchema:
    def test_schema_has_required_fields(self) -> None:
        schema = _build_tool_schema()
        assert schema["name"] == "extract_controls"
        props = schema["input_schema"]["properties"]["controls"]["items"]["properties"]
        assert "control_id" in props
        assert "title" in props
        assert "description" in props
        assert "full_text" in props


class TestLlmExtractor:
    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_extract_returns_controls(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([
            {"control_id": "AC-01", "title": "Access Control", "description": "Enforce strict access control policies for AI model endpoints"},
            {"control_id": "DP-01", "title": "Data Protection", "description": "Encrypt sensitive training data at rest and in transit"},
            {"control_id": "MM-01", "title": "Model Monitoring", "description": "Monitor AI model behavior for drift and adversarial inputs"},
        ])

        extractor = LlmExtractor()
        controls = extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )

        assert len(controls) == 3
        assert all(isinstance(c, Control) for c in controls)
        assert controls[0].control_id == "AC-01"
        assert controls[0].title == "Access Control"

        # Verify API was called
        mock_client.messages.create.assert_called_once()

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_saves_raw_llm_response(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([
            {"control_id": "AC-01", "title": "Access", "description": "Access control policies for systems"},
        ])

        extractor = LlmExtractor()
        extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )

        raw_path = tmp_path / "test_fw_llm_raw.json"
        assert raw_path.exists()
        raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
        assert "controls" in raw_data

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_zero_controls_raises(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([])

        extractor = LlmExtractor()
        with pytest.raises(ValueError, match="(?i)no controls"):
            extractor.extract(
                sample_text_file,
                framework_id="test_fw",
                output_dir=tmp_path,
            )

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_api_failure_retries(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Fail twice, succeed on third attempt
        mock_client.messages.create.side_effect = [
            Exception("API temporarily unavailable"),
            Exception("Rate limited"),
            _mock_api_response([
                {"control_id": "AC-01", "title": "Access", "description": "Access control policies for system components and users"},
            ]),
        ]

        extractor = LlmExtractor()
        controls = extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )
        assert len(controls) == 1
        assert mock_client.messages.create.call_count == 3

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_deduplicates_by_control_id(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        """When chunking produces duplicates, keep the one with longer description."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([
            {"control_id": "AC-01", "title": "Access", "description": "Short desc"},
            {"control_id": "AC-01", "title": "Access Control", "description": "Longer description that provides more context about access control"},
        ])

        extractor = LlmExtractor()
        controls = extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )
        assert len(controls) == 1
        assert "Longer description" in controls[0].description
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_llm_extractor.py -q 2>&1 | head -10
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create tract/prepare/llm_extractor.py**

```python
"""LLM-assisted extractor for framework preparation.

Uses Claude API with tool_use structured output to extract controls
from unstructured documents (PDF, HTML, plain text).

Public API:
    LlmExtractor.extract(path, framework_id, output_dir) -> list[Control]
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tract.config import (
    PREPARE_LLM_CHUNK_TOKEN_LIMIT,
    PREPARE_LLM_MAX_RETRIES,
    PREPARE_LLM_MODEL,
    PREPARE_LLM_RETRY_BACKOFF_FACTOR,
    PREPARE_LLM_RETRY_INITIAL_DELAY_S,
    PREPARE_LLM_TEMPERATURE,
)
from tract.io import atomic_write_json
from tract.schema import Control

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a security framework analyst. Your task is to extract EVERY "
    "security control, requirement, technique, or actionable item from "
    "the provided document. Be exhaustive — do not skip any controls. "
    "Each control must have a unique ID (use the document's numbering if "
    "available, otherwise generate sequential IDs like CTRL-001), a title, "
    "and a description that captures the full requirement text."
)


def _build_tool_schema() -> dict[str, Any]:
    """Build the tool_use schema for structured control extraction.

    Returns:
        Tool definition dict for the Anthropic API.
    """
    return {
        "name": "extract_controls",
        "description": "Extract security framework controls from the document",
        "input_schema": {
            "type": "object",
            "properties": {
                "controls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "control_id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "full_text": {"type": ["string", "null"]},
                        },
                        "required": ["control_id", "title", "description"],
                    },
                },
            },
            "required": ["controls"],
        },
    }


def _get_anthropic_client() -> Any:
    """Get an Anthropic client, loading the API key from the pass store.

    Returns:
        An anthropic.Anthropic client instance.

    Raises:
        RuntimeError: If the API key cannot be retrieved or the
            anthropic package is not installed.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The 'anthropic' package is required for LLM extraction. "
            "Install it with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            result = subprocess.run(
                ["pass", "anthropic/api-key"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            api_key = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not api_key:
        raise RuntimeError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
            "variable or configure via 'pass anthropic/api-key'."
        )

    return anthropic.Anthropic(api_key=api_key)


def _deduplicate_controls(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate controls by control_id, keeping the longer description.

    Args:
        controls: List of control dicts from LLM extraction.

    Returns:
        Deduplicated list of control dicts.
    """
    seen: dict[str, dict[str, Any]] = {}
    for ctrl in controls:
        cid = ctrl.get("control_id", "")
        if cid in seen:
            existing_desc = seen[cid].get("description", "")
            new_desc = ctrl.get("description", "")
            if len(new_desc) > len(existing_desc):
                seen[cid] = ctrl
        else:
            seen[cid] = ctrl
    return list(seen.values())


class LlmExtractor:
    """Extract controls from unstructured documents using Claude API."""

    def extract(
        self,
        path: Path,
        *,
        framework_id: str,
        output_dir: Path,
    ) -> list[Control]:
        """Read a document and extract controls via Claude API tool_use.

        Args:
            path: Path to the unstructured document.
            framework_id: Framework ID for naming the raw response file.
            output_dir: Directory to save the raw LLM response JSON.

        Returns:
            List of Control objects extracted by the LLM.

        Raises:
            ValueError: If the LLM returns zero controls.
            RuntimeError: If all API retries are exhausted.
            FileNotFoundError: If the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = path.read_text(encoding="utf-8")
        logger.info("Read %d chars from %s for LLM extraction", len(text), path.name)

        client = _get_anthropic_client()
        tool_schema = _build_tool_schema()

        # Call the API with retry
        raw_controls = self._call_with_retry(client, text, tool_schema)

        # Deduplicate
        deduped = _deduplicate_controls(raw_controls)

        # Save raw response
        raw_path = output_dir / f"{framework_id}_llm_raw.json"
        atomic_write_json({"controls": deduped}, raw_path)
        logger.info("Saved raw LLM response to %s", raw_path)

        if not deduped:
            raise ValueError(
                "LLM extraction returned no controls. The document may not "
                "contain a structured framework."
            )

        # Convert to Control objects
        controls: list[Control] = []
        for item in deduped:
            controls.append(Control(
                control_id=item["control_id"],
                title=item.get("title", ""),
                description=item["description"],
                full_text=item.get("full_text"),
            ))

        logger.info(
            "LLM extracted %d controls (%d before dedup)",
            len(controls), len(raw_controls),
        )
        return controls

    def _call_with_retry(
        self,
        client: Any,
        text: str,
        tool_schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Call the Anthropic API with exponential backoff retry.

        Args:
            client: Anthropic client instance.
            text: Document text to send.
            tool_schema: Tool definition for structured output.

        Returns:
            List of control dicts from the API response.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        delay = PREPARE_LLM_RETRY_INITIAL_DELAY_S
        last_error: Exception | None = None

        for attempt in range(1, PREPARE_LLM_MAX_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=PREPARE_LLM_MODEL,
                    max_tokens=4096,
                    temperature=PREPARE_LLM_TEMPERATURE,
                    system=_SYSTEM_PROMPT,
                    tools=[tool_schema],
                    tool_choice={"type": "tool", "name": "extract_controls"},
                    messages=[
                        {"role": "user", "content": text},
                    ],
                )

                # Extract tool_use result
                for block in response.content:
                    if getattr(block, "type", None) == "tool_use":
                        return block.input.get("controls", [])

                logger.warning("API response had no tool_use block on attempt %d", attempt)
                return []

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM API attempt %d/%d failed: %s",
                    attempt, PREPARE_LLM_MAX_RETRIES, e,
                )
                if attempt < PREPARE_LLM_MAX_RETRIES:
                    time.sleep(delay)
                    delay *= PREPARE_LLM_RETRY_BACKOFF_FACTOR

        raise RuntimeError(
            f"LLM extraction failed after {PREPARE_LLM_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_llm_extractor.py -v
```

- [ ] **Step 5: Commit**

```
feat: add LLM extractor with Claude API tool_use

Implements LlmExtractor for extracting controls from unstructured
documents via Claude API. Uses tool_use structured output for reliable
extraction, temperature=0 for consistency, exponential backoff retry,
control deduplication, and raw response audit logging.
```

---

### Task 10: Integration Test — prepare -> validate -> ingest Round-Trip

**Files:**
- Create: `tests/test_prepare_integration.py`

- [ ] **Step 1: Write the integration test**

```python
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

        # Validate: should have no errors
        issues = validate_framework(data)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {[i.message for i in errors]}"

        # Verify structure
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
        fw = FrameworkOutput.model_validate(data)  # Must not raise
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
        # Note: fetched_date will be the same within a single test run
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
                    "description": "short",  # too short
                },
                {
                    "control_id": "TC-01",  # duplicate
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
```

- [ ] **Step 2: Run the integration tests**

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_prepare_integration.py -v
```

All tests should pass — they depend only on the validation and extraction modules, not on the trained model.

- [ ] **Step 3: Commit**

```
feat: add integration tests for prepare-validate round-trip

Tests the full pipeline from raw CSV/markdown through prepare_framework(),
validate_framework(), and FrameworkOutput.model_validate(). Verifies
sanitization, deterministic output, and validation gate behavior.
```

---

### Task 11: Create Test Fixtures as Static Files

**Files:**
- Create: `tests/fixtures/sample_framework.csv`
- Create: `tests/fixtures/sample_framework.md`
- Create: `tests/fixtures/sample_framework_array.json`

- [ ] **Step 1: Write static fixtures**

`tests/fixtures/sample_framework.csv`:
```csv
control_id,title,description
TC-01,Access Control,"Enforce access control policies for all system components and users to prevent unauthorized access"
TC-02,Data Encryption,"Encrypt sensitive data at rest and in transit using industry-standard cryptographic algorithms"
TC-03,Audit Logging,"Maintain comprehensive audit logs of security-relevant events for forensic analysis and compliance"
```

`tests/fixtures/sample_framework.md`:
```markdown
# Security Framework

## ASI01: Access Control

Enforce access control policies for all system components to prevent unauthorized access to AI models and data.

## ASI02: Data Encryption

Encrypt sensitive AI training data at rest and in transit using industry-standard cryptographic algorithms.

## ASI03: Audit Logging

Maintain comprehensive audit logs of all security-relevant events for forensic analysis and compliance.
```

`tests/fixtures/sample_framework_array.json`:
```json
[
  {"id": "TC-01", "name": "Access Control", "description": "Enforce access control policies for system components and users to prevent unauthorized access"},
  {"id": "TC-02", "name": "Encryption", "description": "Encrypt sensitive data at rest and in transit using standard cryptographic algorithms"},
  {"id": "TC-03", "name": "Logging", "description": "Maintain comprehensive audit logs of security events for forensic analysis and compliance"}
]
```

- [ ] **Step 2: Verify fixtures are loadable**

```bash
cd /home/rock/github_projects/TRACT && python -c "
from pathlib import Path
import csv, json

csv_path = Path('tests/fixtures/sample_framework.csv')
with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f'CSV: {len(rows)} rows, columns: {list(rows[0].keys())}')

md_path = Path('tests/fixtures/sample_framework.md')
print(f'Markdown: {len(md_path.read_text())} chars')

json_path = Path('tests/fixtures/sample_framework_array.json')
data = json.loads(json_path.read_text())
print(f'JSON: {len(data)} items')
"
```

- [ ] **Step 3: Commit**

```
feat: add static test fixtures for framework preparation

Adds CSV, markdown, and JSON array fixtures representing small
sample frameworks for use by extractor and integration tests.
```

---

## Post-Implementation Checklist

After all tasks are complete, run the full test suite:

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/test_validate.py tests/test_csv_extractor.py tests/test_markdown_extractor.py tests/test_json_extractor.py tests/test_llm_extractor.py tests/test_prepare.py tests/test_prepare_cli.py tests/test_prepare_integration.py -v
```

Verify no regressions in existing tests:

```bash
cd /home/rock/github_projects/TRACT && python -m pytest tests/ -q --ignore=tests/test_assign_e2e.py --ignore=tests/test_ingest_e2e.py --ignore=tests/test_opencre_export_e2e.py --ignore=tests/test_propose_e2e.py -x
```

Run type checking:

```bash
cd /home/rock/github_projects/TRACT && mypy tract/validate.py tract/prepare/ --strict
```

---

## Spec Coverage Verification

| Spec Section | Task(s) | Status |
|---|---|---|
| Section 2: Architecture overview | All tasks | Covered |
| Section 3: `tract prepare` CLI | Task 8 | Covered |
| Section 3: `tract validate` CLI | Task 2 | Covered |
| Section 3: `tract ingest` validation gate | Task 3 | Covered |
| Section 4: Error rules (6 rules) | Task 1 | All 6 rules implemented |
| Section 4: Warning rules (7 rules) | Task 1 | All 7 rules implemented |
| Section 5: CSV extractor | Task 4 | Covered |
| Section 5: Markdown extractor | Task 5 | Covered |
| Section 5: JSON extractor (3 cases) | Task 6 | All 3 cases covered |
| Section 5: LLM extractor | Task 9 | Covered |
| Section 6: Module structure | Tasks 1, 4-9 | Matches spec layout |
| Section 7: Sanitization | Tasks 7, 10 | Via existing sanitize_text() |
| Section 8: Reproducibility | Task 10 | Determinism test included |
| Section 9: Error handling | Tasks 4-9 | All error cases have tests |
| Section 10: Testing strategy | Tasks 1-10 | All strategies covered |
