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
    """Build the default extractor registry."""
    registry = ExtractorRegistry()
    registry.register("csv", CsvExtractor())
    registry.register("markdown", MarkdownExtractor(heading_level=heading_level))
    registry.register("json", JsonExtractor())
    return registry


def _sanitize_control(control: Control) -> Control:
    """Sanitize all text fields of a control using the TRACT pipeline."""
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
        hierarchy_level=control.hierarchy_level,
        parent_id=control.parent_id,
        parent_name=control.parent_name,
        metadata=control.metadata,
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
    fetched_date: str | None = None,
    expected_count: int | None = None,
    column_overrides: dict[str, str] | None = None,
) -> Path:
    """Prepare a raw framework document into a validated FrameworkOutput JSON.

    Pipeline: format detection → extraction → sanitization → assembly →
    validation → atomic write.

    Args:
        file_path: Path to the raw framework document.
        framework_id: Canonical framework ID slug.
        name: Human-readable framework name.
        version: Framework version string.
        source_url: Official URL for the framework.
        mapping_unit: What each control represents (control, technique, etc.).
        output_path: Where to write output JSON. Defaults to
            <input_stem>_prepared.json in the same directory.
        format_override: Override auto-detected format.
        use_llm: Use Claude API for LLM-assisted extraction.
        heading_level: Override heading level for markdown extraction.
        fetched_date: Override auto-generated fetch date (YYYY-MM-DD).
        expected_count: Expected number of controls; warns on mismatch.
        column_overrides: CSV column name overrides mapping canonical
            names (control_id, title, description, full_text) to actual
            column names in the file.

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

    fmt = format_override or detect_format(file_path)
    logger.info("Preparing %s (format: %s)", file_path.name, fmt)

    if fmt == "unstructured":
        if not use_llm:
            raise ValueError(
                f"File {file_path.name} detected as unstructured format. "
                "Use --llm for LLM-assisted extraction, or --format to "
                "override the detected format (csv, markdown, json)."
            )
        from tract.prepare.llm_extractor import LlmExtractor
        extractor = LlmExtractor()
        raw_controls = extractor.extract(
            file_path,
            framework_id=framework_id,
            output_dir=output_path.parent if output_path else file_path.parent,
        )
    else:
        registry = _build_registry(heading_level=heading_level)
        structured_extractor = registry.get(fmt)
        raw_controls = structured_extractor.extract(file_path)

    if not raw_controls:
        raise ValueError(
            f"Extraction produced zero controls from {file_path.name}."
        )

    if expected_count is not None and len(raw_controls) != expected_count:
        logger.warning(
            "Expected %d controls but extracted %d from %s",
            expected_count,
            len(raw_controls),
            file_path.name,
        )

    sanitized_controls = [_sanitize_control(c) for c in raw_controls]

    resolved_date = fetched_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    fw = FrameworkOutput(
        framework_id=framework_id,
        framework_name=name,
        version=version,
        source_url=source_url,
        fetched_date=resolved_date,
        mapping_unit_level=mapping_unit,
        controls=sanitized_controls,
    )

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

    if output_path is None:
        output_path = file_path.with_name(f"{file_path.stem}_prepared.json")

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
