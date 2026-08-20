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

    min_prose_fraction is deliberately NOT overridden. It states a property of
    the text rather than of the sample size, so the fixture has to carry
    activity statements as long as the real ones. Both fixture activities are
    verbatim source text for that reason. Shortening one puts the fixture below
    the 0.83 floor, and the right response is to restore the text rather than
    to relax the parser.
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


def test_the_artifact_records_the_bytes_it_was_built_from(tmp_path: Path) -> None:
    """The static coverage test proves the call exists; this proves it fires.

    A parser can import read_source and still take a different path at
    runtime, and the artifact would carry an empty source_files list with
    nothing to say so.
    """
    import hashlib

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy("tests/fixtures/aiuc_1_sample.json", raw_dir / "aiuc-1-standard.json")
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    result = SampleAiuc1Parser(raw_dir=raw_dir, output_dir=out_dir).run()

    assert [s.path for s in result.source_files] == ["aiuc-1-standard.json"]
    payload = (raw_dir / "aiuc-1-standard.json").read_bytes()
    assert result.source_files[0].sha256 == hashlib.sha256(payload).hexdigest()
    assert result.source_files[0].bytes == len(payload)
