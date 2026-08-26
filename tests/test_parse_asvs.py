"""Tests for parsers/parse_asvs.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR

RAW_DIR = RAW_FRAMEWORKS_DIR / "asvs"


def _parse() -> object:
    from parsers.parse_asvs import AsvsParser

    if not any(RAW_DIR.iterdir()) if RAW_DIR.exists() else True:
        pytest.skip("Raw data not available")
    with tempfile.TemporaryDirectory() as td:
        return AsvsParser(raw_dir=RAW_DIR, output_dir=Path(td)).run()


def test_parses_real_data() -> None:
    result = _parse()

    assert result.framework_id == "asvs"
    assert len(result.controls) == 278

    ids = {c.control_id for c in result.controls}
    assert "V1.1.1" in ids
    assert "V14.5.4" in ids
    assert all(c.description for c in result.controls)


def test_control_id_is_the_dotted_requirement_number() -> None:
    """OpenCRE links ASVS by "V1.1.1", chapter.section.requirement."""
    result = _parse()

    for control in result.controls:
        assert control.control_id.startswith("V"), control.control_id
        parts = control.control_id[1:].split(".")
        assert len(parts) == 3, control.control_id
        assert all(p.isdigit() for p in parts), control.control_id


def test_chapter_and_section_context_is_retained() -> None:
    """A requirement reads as a bare "Verify that..." sentence on its own.

    The chapter and section names are the only thing that says which layer of
    the application it verifies, so they have to survive into metadata.
    """
    result = _parse()

    v111 = next(c for c in result.controls if c.control_id == "V1.1.1")
    assert v111.metadata["chapter"].startswith("V1 ")
    assert v111.metadata["section"].startswith("V1.1 ")
    assert all(c.metadata["chapter"] for c in result.controls)
    assert all(c.metadata["section"] for c in result.controls)


def test_verification_levels_are_recorded() -> None:
    """L1/L2/L3 is what tells a reviewer whether a requirement applies."""
    result = _parse()

    seen: set[str] = set()
    for control in result.controls:
        levels = control.metadata["levels"]
        assert isinstance(levels, list)
        seen.update(levels)
    assert seen <= {"L1", "L2", "L3"}
    assert "L1" in seen
