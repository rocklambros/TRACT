"""Tests for parsers/parse_nist_ai_100_2.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR

RAW_DIR = RAW_FRAMEWORKS_DIR / "nist_ai_100_2"


def _parse() -> object:
    pytest.importorskip("pdfplumber")
    from parsers.parse_nist_ai_100_2 import NistAi1002Parser

    if not any(RAW_DIR.glob("*.pdf")):
        pytest.skip("Raw data not available")
    with tempfile.TemporaryDirectory() as td:
        return NistAi1002Parser(raw_dir=RAW_DIR, output_dir=Path(td)).run()


def test_parses_real_data() -> None:
    result = _parse()

    assert result.framework_id == "nist_ai_100_2"
    assert len(result.controls) == 66

    ids = {c.control_id for c in result.controls}
    assert "2.1" in ids
    assert all(c.description for c in result.controls)


def test_named_techniques_are_addressable() -> None:
    """OpenCRE links NIST AI 100-2 by technique name, not by section number.

    The numbered sections alone cannot carry those links, so named techniques
    get their own controls under a "technique:" prefix that cannot collide
    with a section id.
    """
    result = _parse()

    techniques = [c for c in result.controls if c.control_id.startswith("technique:")]
    assert len(techniques) == 21
    for control in techniques:
        assert control.description
        assert ":" not in control.control_id.split("technique:", 1)[1]


def test_pdf_ligatures_are_repaired() -> None:
    """pdfplumber emits U+FB01 for "fi", which no framework text ever contains.

    Left in place it splits "identification" from "identiﬁcation" for both the
    tokenizer and the stop word list.
    """
    result = _parse()

    for control in result.controls:
        body = f"{control.title} {control.description}"
        assert "ﬁ" not in body, control.control_id
        assert "ﬂ" not in body, control.control_id


def test_page_furniture_is_excluded() -> None:
    """Running headers and the NIST DOI footer repeat on all 100-odd pages."""
    result = _parse()

    for control in result.controls:
        assert "doi.org/10.6028" not in control.description, control.control_id
        assert "This publication is available free of charge" not in control.description
