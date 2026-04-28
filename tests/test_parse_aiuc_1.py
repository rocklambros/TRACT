"""Tests for parsers/parse_aiuc_1.py."""
from __future__ import annotations

import shutil
from pathlib import Path

from parsers.parse_aiuc_1 import Aiuc1Parser


def test_parses_sample_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy("tests/fixtures/aiuc_1_sample.json", raw_dir / "aiuc-1-standard.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = Aiuc1Parser(raw_dir=raw_dir, output_dir=out_dir)
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
