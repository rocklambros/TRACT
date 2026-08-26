"""Tests for parsers/parse_csa_aicm.py."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

from parsers.parse_csa_aicm import CsaAicmParser


class SampleCsaAicmParser(CsaAicmParser):
    """The parser with the fixture's count rather than the full source's.

    The fixture holds 2 of the 243 controls. run()'s count gate is real and
    must stay real, so the test declares what this input contains.

    min_prose_fraction is deliberately NOT overridden. It states a property of
    the text rather than of the sample size, so both fixture specifications
    carry their full source wording and clear the 0.97 floor.
    """

    expected_count: ClassVar[int] = 2


def test_parses_sample_fixture(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/csa_aicm_sample.json")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(fixture, raw_dir / "csa_aicm.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = SampleCsaAicmParser(raw_dir=raw_dir, output_dir=out_dir)
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
