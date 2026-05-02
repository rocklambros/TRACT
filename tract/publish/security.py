"""Context-aware security scan for HuggingFace upload staging directory."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tract.config import HF_SCAN_EXTENSIONS, HF_SECRET_PATTERNS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecurityFinding:
    file_path: str
    line_number: int
    pattern_name: str
    matched_text: str


def _scan_file_contents(file_path: Path, findings: list[SecurityFinding]) -> None:
    """Scan a text file against all secret patterns."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return

    for line_num, line in enumerate(content.splitlines(), 1):
        for pattern in HF_SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append(SecurityFinding(
                    file_path=str(file_path),
                    line_number=line_num,
                    pattern_name=pattern.pattern,
                    matched_text=match.group()[:50],
                ))


def _scan_bridge_report_notes(file_path: Path, findings: list[SecurityFinding]) -> None:
    """Scan reviewer_notes fields in bridge_report.json for PII."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    for i, candidate in enumerate(data.get("candidates", [])):
        notes = candidate.get("reviewer_notes", "")
        if not notes:
            continue
        for pattern in HF_SECRET_PATTERNS:
            match = pattern.search(notes)
            if match:
                findings.append(SecurityFinding(
                    file_path=str(file_path),
                    line_number=i,
                    pattern_name=f"reviewer_notes: {pattern.pattern}",
                    matched_text=match.group()[:50],
                ))


def scan_for_secrets(staging_dir: Path) -> list[SecurityFinding]:
    """Scan staging directory for secrets before HuggingFace upload.

    Scans:
    - .py, .md, .txt, .yaml, .yml files against all secret patterns
    - bridge_report.json reviewer_notes fields for PII
    - Structural: no .git/ directory, no adapter_config.json

    Returns:
        List of findings. Any non-empty list = hard failure.
    """
    findings: list[SecurityFinding] = []

    if (staging_dir / ".git").exists():
        findings.append(SecurityFinding(
            file_path=str(staging_dir / ".git"),
            line_number=0,
            pattern_name="structural",
            matched_text=".git directory present in staging area",
        ))

    if (staging_dir / "adapter_config.json").exists():
        findings.append(SecurityFinding(
            file_path=str(staging_dir / "adapter_config.json"),
            line_number=0,
            pattern_name="structural",
            matched_text="adapter_config.json present — merge incomplete",
        ))

    for path in staging_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in HF_SCAN_EXTENSIONS:
            _scan_file_contents(path, findings)

    bridge_report = staging_dir / "bridge_report.json"
    if bridge_report.exists():
        _scan_bridge_report_notes(bridge_report, findings)

    if findings:
        logger.error("Security scan found %d issues:", len(findings))
        for f in findings:
            logger.error("  %s:%d — %s", f.file_path, f.line_number, f.pattern_name)
    else:
        logger.info("Security scan passed — no secrets detected")

    return findings
