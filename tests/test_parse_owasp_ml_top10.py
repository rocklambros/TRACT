"""Tests for parsers/parse_owasp_ml_top10.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR

RAW_DIR = RAW_FRAMEWORKS_DIR / "owasp_ml_top10"


def _parse() -> object:
    from parsers.parse_owasp_ml_top10 import OwaspMlTop10Parser

    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        pytest.skip("Raw data not available")
    with tempfile.TemporaryDirectory() as td:
        return OwaspMlTop10Parser(raw_dir=RAW_DIR, output_dir=Path(td)).run()


def test_parses_real_data() -> None:
    result = _parse()

    assert result.framework_id == "owasp_ml_top10"
    assert len(result.controls) == 10
    assert all(c.description for c in result.controls)


def test_ids_are_the_full_ml_top_ten() -> None:
    """OpenCRE links "ML01:2023" through "ML10:2023"; a gap loses a link."""
    result = _parse()

    ids = [c.control_id for c in result.controls]
    assert ids == [f"ML{n:02d}:2023" for n in range(1, 11)]


def test_titles_match_the_published_risks() -> None:
    result = _parse()

    by_id = {c.control_id: c for c in result.controls}
    assert by_id["ML01:2023"].title == "Input Manipulation Attack"
    assert by_id["ML10:2023"].title == "Model Poisoning"
    assert by_id["ML01:2023"].metadata["year"] == "2023"
