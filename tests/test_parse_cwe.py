"""Tests for parsers/parse_cwe.py."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR

RAW_DIR = RAW_FRAMEWORKS_DIR / "cwe"


def _parse() -> object:
    from parsers.parse_cwe import CweParser

    if not any(RAW_DIR.glob("*.zip")):
        pytest.skip("Raw data not available")
    with tempfile.TemporaryDirectory() as td:
        return CweParser(raw_dir=RAW_DIR, output_dir=Path(td)).run()


def test_parses_real_data() -> None:
    result = _parse()

    assert result.framework_id == "cwe"
    assert len(result.controls) == 1331

    ids = {c.control_id for c in result.controls}
    # CWE-79 and CWE-89 are the two most heavily CRE-linked weaknesses; they
    # anchor the CAPEC -> CWE -> CRE auto-link chain.
    assert "79" in ids
    assert "89" in ids
    assert all(c.description for c in result.controls)


def test_control_id_is_the_bare_number() -> None:
    """OpenCRE joins on the numeric id; "CWE-79" is the display form."""
    result = _parse()

    for control in result.controls:
        assert control.control_id.isdigit(), control.control_id
        assert control.metadata["cwe_id"] == f"CWE-{control.control_id}"


def test_weaknesses_and_categories_are_distinguishable() -> None:
    """A category is an organizing bucket, not a weakness. Keep them labelled.

    Both carry CRE links, so both are parsed, but a consumer that wants only
    real weaknesses must be able to tell them apart without re-reading the XML.
    """
    result = _parse()

    types = {c.metadata["entry_type"] for c in result.controls}
    assert types == {"weakness", "category"}
    weaknesses = [c for c in result.controls if c.metadata["entry_type"] == "weakness"]
    assert len(weaknesses) == 944


def test_refuses_an_oversized_member() -> None:
    """The archive is untrusted input, so a zip bomb must not be read in."""
    from parsers.parse_cwe import MAX_UNCOMPRESSED_BYTES, CweParser

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"
        raw.mkdir()
        # Highly compressible payload one byte over the declared ceiling.
        with zipfile.ZipFile(raw / "cwec_latest.xml.zip", "w",
                             zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("cwec.xml", b"\0" * (MAX_UNCOMPRESSED_BYTES + 1))

        parser = CweParser(raw_dir=raw, output_dir=Path(td) / "out")
        with pytest.raises(ValueError, match="ceiling"):
            parser.parse()
