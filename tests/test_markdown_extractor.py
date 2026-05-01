"""Tests for tract.prepare.markdown_extractor — Markdown format extraction."""
from __future__ import annotations

from pathlib import Path

import pytest

from tract.prepare.markdown_extractor import MarkdownExtractor
from tract.schema import Control


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
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
        extractor = MarkdownExtractor()
        controls = extractor.extract(h3_headings_md)
        assert len(controls) == 3

    def test_returns_control_objects(self, sample_md: Path) -> None:
        extractor = MarkdownExtractor()
        controls = extractor.extract(sample_md)
        for ctrl in controls:
            assert isinstance(ctrl, Control)
            assert ctrl.control_id
            assert ctrl.description
