"""Tests for parsers/parse_nist_800_53.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR

RAW_DIR = RAW_FRAMEWORKS_DIR / "nist_800_53"


def _parse() -> object:
    from parsers.parse_nist_800_53 import Nist80053Parser

    if not any(RAW_DIR.glob("*.json")):
        pytest.skip("Raw data not available")
    with tempfile.TemporaryDirectory() as td:
        return Nist80053Parser(raw_dir=RAW_DIR, output_dir=Path(td)).run()


def test_parses_real_data() -> None:
    result = _parse()

    assert result.framework_id == "nist_800_53"
    assert len(result.controls) == 300

    ids = {c.control_id for c in result.controls}
    # First and last family anchors, plus the most heavily CRE-linked control.
    assert "AC-1" in ids
    assert "SR-12" in ids
    assert "SC-8" in ids
    assert all(c.description for c in result.controls)


def test_control_id_is_the_uppercase_opencre_form() -> None:
    """OpenCRE links "AC-1"; OSCAL stores "ac-1". The join needs the former."""
    result = _parse()

    for control in result.controls:
        assert control.control_id == control.control_id.upper()
        assert control.metadata["oscal_id"] == control.control_id.lower()
        assert control.metadata["family"] == control.control_id.split("-")[0]


def test_title_carries_the_id_prefix() -> None:
    """The bare title ("Policy and Procedures") repeats across 20 families.

    Prefixing it with the control id is what keeps those 20 anchors distinct
    when the text-selection layer falls back to the title.
    """
    result = _parse()

    ac1 = next(c for c in result.controls if c.control_id == "AC-1")
    assert ac1.title == "AC-1 Policy and Procedures"
    assert ac1.metadata["control_title"] == "Policy and Procedures"

    titles = [c.title for c in result.controls]
    assert len(set(titles)) == len(titles)
