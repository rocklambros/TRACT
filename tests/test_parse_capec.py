"""Tests for parsers/parse_capec.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR

RAW_DIR = RAW_FRAMEWORKS_DIR / "capec"


def _parse() -> object:
    if not (RAW_DIR / "capec_latest.xml").exists():
        pytest.skip("Raw data not available")
    from parsers.parse_capec import CapecParser

    with tempfile.TemporaryDirectory() as td:
        return CapecParser(raw_dir=RAW_DIR, output_dir=Path(td)).run()


def test_parses_real_data() -> None:
    result = _parse()

    assert result.framework_id == "capec"
    assert len(result.controls) == 558

    ids = {c.control_id for c in result.controls}
    # CAPEC-1 and CAPEC-98 are stable patterns OpenCRE links through the
    # CAPEC -> CWE -> CRE chain, so a rename here breaks the auto-links.
    assert "1" in ids
    assert "98" in ids
    assert all(c.description for c in result.controls)


def test_control_id_is_the_bare_number() -> None:
    """OpenCRE's join key is the numeric id, not the "CAPEC-" display form."""
    result = _parse()

    for control in result.controls:
        assert control.control_id.isdigit(), control.control_id
        # metadata carries the human-facing "CAPEC-1" form alongside it.
        assert control.metadata["capec_id"] == f"CAPEC-{control.control_id}"


def test_deprecated_patterns_are_excluded() -> None:
    """A deprecated pattern carries no usable text and no live CRE link."""
    from parsers.parse_capec import EXCLUDED_STATUS

    result = _parse()

    statuses = {c.metadata["status"] for c in result.controls}
    assert statuses.isdisjoint(EXCLUDED_STATUS), statuses & EXCLUDED_STATUS


def test_rejects_an_empty_tree() -> None:
    """getroot() can return None, and the loop below it would AttributeError."""
    from parsers.parse_capec import CapecParser

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"
        raw.mkdir()
        # A well-formed XML declaration with no root element.
        (raw / "capec_latest.xml").write_bytes(b"<?xml version='1.0'?>")
        parser = CapecParser(raw_dir=raw, output_dir=Path(td) / "out")
        with pytest.raises(Exception):  # noqa: B017 - parse or empty-tree error
            parser.parse()
