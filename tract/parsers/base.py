"""TRACT BaseParser — abstract base class for all framework parsers.

Every parser (parsers/parse_*.py) subclasses BaseParser and implements
parse() -> list[Control]. The concrete run() method handles sanitization,
validation, count-checking, and atomic output writing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from tract.config import (
    COUNT_TOLERANCE,
    DESCRIPTION_MAX_LENGTH,
    EXPECTED_COUNTS,
    PROCESSED_FRAMEWORKS_DIR,
    RAW_FRAMEWORKS_DIR,
)
from tract.io import atomic_write_json
from tract.sanitize import sanitize_text
from tract.schema import Control, FrameworkOutput

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base for TRACT framework parsers.

    Subclasses must set the class-level attributes and implement parse().

    Class Attributes:
        framework_id: Canonical ID (e.g., "csa_aicm").
        framework_name: Human-readable name (e.g., "CSA AI Controls Matrix").
        version: Framework version string.
        source_url: Official URL for the framework.
        mapping_unit_level: Granularity level (e.g., "control", "technique").
        expected_count: Expected number of mapping units after parsing.
    """

    framework_id: ClassVar[str]
    framework_name: ClassVar[str]
    version: ClassVar[str]
    source_url: ClassVar[str]
    mapping_unit_level: ClassVar[str]
    expected_count: ClassVar[int]

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize the parser with input/output directories.

        Args:
            raw_dir: Directory containing raw framework files.
                Defaults to DATA_DIR/raw/frameworks/<framework_id>.
            output_dir: Directory for processed output.
                Defaults to DATA_DIR/processed/frameworks.
        """
        self.raw_dir = raw_dir or RAW_FRAMEWORKS_DIR / self.framework_id
        self.output_dir = output_dir or PROCESSED_FRAMEWORKS_DIR

    @abstractmethod
    def parse(self) -> list[Control]:
        """Parse raw framework data into a list of Control objects.

        Subclasses implement the framework-specific extraction logic here.
        Do NOT sanitize text in parse() — run() handles that.

        Returns:
            List of Control objects with raw (unsanitized) text fields.
        """
        ...

    def run(self) -> FrameworkOutput:
        """Execute the full parser pipeline: parse -> sanitize -> validate -> write.

        Returns:
            The validated FrameworkOutput that was written to disk.

        Raises:
            ValueError: If no controls are produced or validation fails.
        """
        logger.info(
            "Parsing %s (%s) from %s",
            self.framework_name,
            self.framework_id,
            self.raw_dir,
        )

        raw_controls = self.parse()
        if not raw_controls:
            raise ValueError(
                f"Parser {self.framework_id} produced zero controls"
            )

        sanitized_controls = [
            self._sanitize_control(c) for c in raw_controls
        ]

        self._check_expected_count(len(sanitized_controls))

        output = FrameworkOutput(
            framework_id=self.framework_id,
            framework_name=self.framework_name,
            version=self.version,
            source_url=self.source_url,
            fetched_date=self._today(),
            mapping_unit_level=self.mapping_unit_level,
            controls=sanitized_controls,
        )

        output_path = self.output_dir / f"{self.framework_id}.json"
        atomic_write_json(
            output.model_dump(mode="json", exclude_none=True),
            output_path,
        )

        logger.info(
            "Wrote %d controls to %s",
            len(sanitized_controls),
            output_path,
        )
        return output

    def _sanitize_control(self, control: Control) -> Control:
        """Sanitize text fields of a control using the TRACT pipeline.

        Uses sanitize_text with return_full=True so that if description
        exceeds DESCRIPTION_MAX_LENGTH, the full text is preserved.

        Args:
            control: A Control with potentially raw/unsanitized text.

        Returns:
            A new Control with sanitized text fields.
        """
        sanitized_desc, full_text = sanitize_text(
            control.description,
            max_length=DESCRIPTION_MAX_LENGTH,
            return_full=True,
        )

        # If the control already had full_text, sanitize that too
        sanitized_full: str | None = full_text
        if control.full_text is not None and full_text is None:
            sanitized_full = sanitize_text(
                control.full_text,
                max_length=50_000,  # generous limit for full text
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

    def _check_expected_count(self, actual: int) -> None:
        """Log a warning if the parsed count deviates from expected.

        Uses COUNT_TOLERANCE (default 10%) to determine deviation threshold.
        Falls back to the class attribute expected_count, then EXPECTED_COUNTS.

        Args:
            actual: Number of controls actually parsed.
        """
        expected = getattr(self, "expected_count", None)
        if expected is None:
            expected = EXPECTED_COUNTS.get(self.framework_id)

        if expected is None or expected == 0:
            logger.debug(
                "%s: no expected count configured, skipping count check",
                self.framework_id,
            )
            return

        deviation = abs(actual - expected) / expected
        if deviation > COUNT_TOLERANCE:
            logger.warning(
                "%s: parsed %d controls, expected %d (%.1f%% deviation)",
                self.framework_id,
                actual,
                expected,
                deviation * 100,
            )
        else:
            logger.info(
                "%s: parsed %d controls (expected %d, within tolerance)",
                self.framework_id,
                actual,
                expected,
            )

    @staticmethod
    def _today() -> str:
        """Return today's date as an ISO 8601 string (YYYY-MM-DD, UTC).

        Returns:
            Date string like "2026-04-27".
        """
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
