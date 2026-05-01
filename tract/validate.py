"""TRACT framework validation — rules engine for FrameworkOutput JSON.

Validates a FrameworkOutput dict against error rules (block ingest) and
warning rules (informational).

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
_REFERENCE_RE: re.Pattern[str] = re.compile(
    r"^(see|refer to|as defined in|per)\b", re.IGNORECASE,
)


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


def validate_framework(data: dict[str, Any], *, expected_count: int | None = None) -> list[ValidationIssue]:
    """Validate a FrameworkOutput dict against all error and warning rules.

    Args:
        data: A dict representing a FrameworkOutput JSON structure.

    Returns:
        List of ValidationIssue instances. Empty list means fully valid.
        Issues are ordered: errors first, then warnings.
    """
    issues: list[ValidationIssue] = []

    # Pre-check: zero controls before Pydantic (which rejects empty list
    # via min_length=1, masking the specific zero_controls error)
    controls_raw = data.get("controls")
    if isinstance(controls_raw, list) and len(controls_raw) == 0:
        issues.append(ValidationIssue(
            severity="error",
            control_id=None,
            rule="zero_controls",
            message="No controls found",
        ))
        fw_id = data.get("framework_id", "")
        if fw_id and not _FRAMEWORK_ID_PATTERN.match(fw_id):
            issues.append(ValidationIssue(
                severity="error",
                control_id=None,
                rule="invalid_framework_id",
                message=f"Invalid framework_id: {fw_id!r}",
            ))
        return issues

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

        if _REFERENCE_RE.match(desc_stripped) and len(desc_stripped) < 100:
            issues.append(ValidationIssue(
                severity="warning",
                control_id=cid,
                rule="reference_only_description",
                message=(
                    f"Control {cid}: description appears to be a reference, "
                    "not substantive text — will produce weak embeddings"
                ),
            ))

        if ctrl.title.strip().lower() == desc_stripped.lower():
            issues.append(ValidationIssue(
                severity="warning",
                control_id=cid,
                rule="title_description_redundancy",
                message=(
                    f"Control {cid}: description duplicates title — "
                    "add substantive text for better embeddings"
                ),
            ))

        ascii_letters = sum(1 for c in desc_stripped if c.isascii() and c.isalpha())
        total_letters = sum(1 for c in desc_stripped if c.isalpha())
        if total_letters > 10 and ascii_letters / total_letters < 0.7:
            issues.append(ValidationIssue(
                severity="warning",
                control_id=cid,
                rule="non_english_text",
                message=(
                    f"Control {cid}: description may not be English — "
                    "model was trained on English text"
                ),
            ))

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

    if expected_count is not None and n_controls > 0:
        deviation = abs(n_controls - expected_count) / expected_count
        if deviation > 0.5:
            issues.append(ValidationIssue(
                severity="warning",
                control_id=None,
                rule="expected_count_mismatch",
                message=(
                    f"Expected ~{expected_count} controls, got {n_controls} "
                    "— verify extraction captured everything"
                ),
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
