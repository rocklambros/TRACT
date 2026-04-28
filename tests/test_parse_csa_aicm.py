"""Tests for parsers/parse_csa_aicm.py."""
from __future__ import annotations

import shutil
from pathlib import Path

from parsers.parse_csa_aicm import CsaAicmParser


def test_parses_sample_fixture(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/csa_aicm_sample.json")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(fixture, raw_dir / "csa_aicm.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = CsaAicmParser(raw_dir=raw_dir, output_dir=out_dir)
    result = parser.run()

    assert result.framework_id == "csa_aicm"
    assert len(result.controls) == 2
    assert result.controls[0].control_id == "A&A-01"
    assert result.controls[0].title == "Audit and Assurance Policy"
    assert "audit and assurance policies" in result.controls[0].description.lower()
    assert result.controls[0].parent_id == "A&A"
    assert result.controls[0].metadata is not None
    assert result.controls[0].metadata["control_type"] == "Cloud & AI Related"

    output_file = out_dir / "csa_aicm.json"
    assert output_file.exists()
