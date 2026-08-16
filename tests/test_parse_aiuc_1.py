"""Tests for parsers/parse_aiuc_1.py."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

from parsers.parse_aiuc_1 import Aiuc1Parser


class SampleAiuc1Parser(Aiuc1Parser):
    """The parser with the fixture's count rather than the full source's.

    The fixture holds 2 of the 132 controls. run()'s count gate is real and
    must stay real, so the test declares what this input contains instead of
    asking the gate to look the other way. count_deviation_reason exists for a
    source that genuinely changed, not for a test that feeds a sample.
    """

    expected_count: ClassVar[int] = 2


def test_parses_sample_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy("tests/fixtures/aiuc_1_sample.json", raw_dir / "aiuc-1-standard.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = SampleAiuc1Parser(raw_dir=raw_dir, output_dir=out_dir)
    result = parser.run()

    assert result.framework_id == "aiuc_1"
    assert len(result.controls) == 2
    assert result.controls[0].control_id == "A001.1"
    assert result.controls[0].parent_id == "A001"
    assert result.controls[0].parent_name == "Establish input data policy"
    assert result.controls[0].hierarchy_level == "activity"
    assert result.controls[0].metadata is not None
    assert result.controls[0].metadata["category"] == "Core"
    assert result.controls[0].metadata["domain"] == "Data & Privacy"
