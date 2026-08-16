"""TRACT BaseParser — abstract base class for all framework parsers.

Every parser (parsers/parse_*.py) subclasses BaseParser and implements
parse() -> list[Control]. The concrete run() method handles sanitization,
validation, count-checking, and atomic output writing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from tract.config import (
    CONTROL_DAMAGED_METADATA_KEY,
    CONTROL_DAMAGED_METADATA_VALUE,
    COUNT_TOLERANCE,
    DESCRIPTION_MAX_LENGTH,
    EXPECTED_COUNTS,
    HONEST_PROSE_MIN_CHARS,
    PROCESSED_FRAMEWORKS_DIR,
    PROCESSED_REPAIR_AUDIT_DIR,
    RAW_FRAMEWORKS_DIR,
)
from tract.io import atomic_write_json
from tract.sanitize import sanitize_text
from tract.schema import Control, FrameworkOutput, SourceFile

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
    # The date the raw bytes were fetched, not the date they were parsed.
    # Declared rather than stamped: re-parsing the same input must produce the
    # same output bytes, and a clock read makes that impossible.
    fetched_date: ClassVar[str]
    # Set only when the raw directory is not the framework_id in either
    # underscore or hyphen form. Kept explicit rather than guessed.
    raw_dir_name: ClassVar[str | None] = None
    # Set to a written reason when a parser legitimately deviates from its
    # expected count. Unset means a deviation is a bug and run() refuses to
    # write. A warning that nobody reads is not a gate.
    count_deviation_reason: ClassVar[str | None] = None
    # The floor this parser's output must clear. Measured on stored text, not
    # on the join-path prose_fraction telemetry, which records whether a
    # lookup hit rather than whether the text is prose.
    min_prose_fraction: ClassVar[float] = 0.0

    @classmethod
    def resolve_raw_dir(cls) -> Path:
        """Locate this framework's raw directory.

        framework_id uses underscores; the raw tree, which is copied from the
        upstream project, uses hyphens. Eleven of the twelve parsers could not
        find their own input after a restore because of that alone. Try the
        declared name, then both separator conventions, and fail with the list
        of what was tried rather than with a bare FileNotFoundError from
        whichever read happens first inside parse().
        """
        candidates = []
        if cls.raw_dir_name:
            candidates.append(cls.raw_dir_name)
        candidates += [cls.framework_id, cls.framework_id.replace("_", "-")]

        seen: list[Path] = []
        for name in candidates:
            path = RAW_FRAMEWORKS_DIR / name
            if path not in seen:
                seen.append(path)
            if path.is_dir():
                return path

        raise FileNotFoundError(
            f"No raw directory for {cls.framework_id}. Tried: "
            f"{[str(p) for p in seen]}. data/raw/ is gitignored, so a fresh "
            f"checkout starts empty; repopulate it from the source recorded in "
            f"data/raw/PROVENANCE.txt."
        )

    @staticmethod
    def is_damaged(control: Control) -> bool:
        """Whether a parser marked this control's source text incomplete."""
        if not control.metadata:
            return False
        return (
            control.metadata.get(CONTROL_DAMAGED_METADATA_KEY)
            == CONTROL_DAMAGED_METADATA_VALUE
        )

    @staticmethod
    def honest_prose_fraction(controls: list[Control]) -> float:
        """Fraction of controls whose description is more than their title.

        Both conditions must hold. A byte-copy of the title is not prose no
        matter how long the title is: nist_ssdf has a 156-character median
        description and a 0% real-prose rate because its descriptions are long
        titles.

        Controls marked damaged are excluded from both sides of the ratio.
        Counting one as prose lets a statement with a known hole in it clear
        the floor, and counting it against the parser penalises the disclosure
        rather than the damage.
        """
        measurable = [c for c in controls if not BaseParser.is_damaged(c)]
        if not measurable:
            return 0.0
        honest = sum(
            1 for c in measurable
            if len(c.description.strip()) >= HONEST_PROSE_MIN_CHARS
            and c.description.strip() != c.title.strip()
        )
        return honest / len(measurable)

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
        audit_dir: Path | None = None,
    ) -> None:
        """Initialize the parser with input/output directories.

        Args:
            raw_dir: Directory containing raw framework files.
                Defaults to the resolved DATA_DIR/raw/frameworks/<framework>.
            output_dir: Directory for processed output.
                Defaults to DATA_DIR/processed/frameworks.
            audit_dir: Directory for repair audit files.
                Defaults to DATA_DIR/processed/repair_audit, which is
                gitignored because audit records quote source text verbatim.
        """
        self._raw_dir: Path | None = raw_dir
        self.output_dir = output_dir or PROCESSED_FRAMEWORKS_DIR
        self.audit_dir = audit_dir or PROCESSED_REPAIR_AUDIT_DIR

        # Populated by read_source*, drained into FrameworkOutput by run().
        self._source_files: dict[str, SourceFile] = {}

    @property
    def raw_dir(self) -> Path:
        """The framework's raw directory, resolved on first use.

        Lazy on purpose. Constructing a parser must not require the raw tree to
        be present, since data/raw/ is gitignored and plenty of callers only
        want the class metadata.
        """
        if self._raw_dir is None:
            self._raw_dir = self.resolve_raw_dir()
        return self._raw_dir

    @abstractmethod
    def parse(self) -> list[Control]:
        """Parse raw framework data into a list of Control objects.

        Subclasses implement the framework-specific extraction logic here.
        Do NOT sanitize text in parse() — run() handles that.

        Returns:
            List of Control objects with raw (unsanitized) text fields.
        """
        ...

    def read_source_bytes(self, name: str) -> bytes:
        """Read one raw input file and record its digest.

        Parsers must read through this rather than opening files directly.
        A file read outside it is invisible to the manifest, which defeats
        the point of recording one.
        """
        path = self.raw_dir / name
        payload = path.read_bytes()
        self._source_files[name] = SourceFile(
            path=name,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
        return payload

    def read_source(self, name: str, encoding: str = "utf-8") -> str:
        """Read one raw input file as text and record its digest."""
        return self.read_source_bytes(name).decode(encoding)

    def write_repair_audit(
        self, records: Sequence[Mapping[str, object]],
    ) -> Path:
        """Persist before/after pairs for repairs that move text across ids.

        A count says a repair fired. It does not say what moved, or where to,
        and a fragment attributed to the wrong control is a wrong compliance
        assertion carrying a plausible-looking provenance record. This is the
        file a reviewer reads to check one.

        Written unconditionally, empty list included, so a missing file means
        the parser never ran rather than the repair never fired. Keys are
        sorted and no clock is read, so re-parsing the same bytes produces the
        same audit bytes and a diff shows real changes only.

        Returns the path written.
        """
        path = self.audit_dir / f"{self.framework_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        )

        # Same atomic pattern as tract.io.atomic_write_json. JSONL is not JSON,
        # so it cannot go through that helper, and a half-written audit file is
        # worse than none: it reads as a complete record of what moved.
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, path)
        except OSError:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        logger.info(
            "%s: wrote %d repair audit record(s) to %s",
            self.framework_id, len(records), path,
        )
        return path

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

        fraction = self.honest_prose_fraction(sanitized_controls)
        if fraction < self.min_prose_fraction:
            raise ValueError(
                f"{self.framework_id}: honest prose fraction {fraction:.3f} is "
                f"below the declared floor {self.min_prose_fraction:.3f}. The "
                f"parser is emitting titles where control statements were "
                f"expected. This is the check that would have caught the 568 "
                f"title-only controls already in the corpus."
            )
        logger.info(
            "%s: honest prose fraction %.3f (floor %.3f)",
            self.framework_id, fraction, self.min_prose_fraction,
        )

        output = FrameworkOutput(
            framework_id=self.framework_id,
            framework_name=self.framework_name,
            version=self.version,
            source_url=self.source_url,
            fetched_date=self.fetched_date,
            mapping_unit_level=self.mapping_unit_level,
            controls=sanitized_controls,
            source_files=[
                self._source_files[k] for k in sorted(self._source_files)
            ],
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
        """Raise if the parsed count deviates from expected without documented reason.

        Uses COUNT_TOLERANCE (default 10%) to determine deviation threshold.
        Falls back to the class attribute expected_count, then EXPECTED_COUNTS.

        Args:
            actual: Number of controls actually parsed.

        Raises:
            ValueError: If actual count deviates beyond tolerance and no
                count_deviation_reason is set.
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
        if deviation <= COUNT_TOLERANCE:
            logger.info(
                "%s: parsed %d controls (expected %d, within tolerance)",
                self.framework_id, actual, expected,
            )
            return

        if self.count_deviation_reason:
            logger.warning(
                "%s: parsed %d controls, expected %d (%.1f%% deviation). "
                "Permitted: %s",
                self.framework_id, actual, expected, deviation * 100,
                self.count_deviation_reason,
            )
            return

        raise ValueError(
            f"{self.framework_id}: parsed {actual} controls, expected "
            f"{expected} ({deviation * 100:.1f}% deviation, tolerance "
            f"{COUNT_TOLERANCE * 100:.0f}%). Either the source changed or the "
            f"parser is wrong. If the deviation is correct, set "
            f"count_deviation_reason on the parser with the reason."
        )
