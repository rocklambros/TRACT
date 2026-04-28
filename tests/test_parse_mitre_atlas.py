"""Tests for parsers/parse_mitre_atlas.py."""
from __future__ import annotations

import shutil
from pathlib import Path

from parsers.parse_mitre_atlas import MitreAtlasParser


def test_parses_sample_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy("tests/fixtures/mitre_atlas_sample.json", raw_dir / "ATLAS_compiled.json")

    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    parser = MitreAtlasParser(raw_dir=raw_dir, output_dir=out_dir)
    result = parser.run()

    assert result.framework_id == "mitre_atlas"
    ids = [c.control_id for c in result.controls]
    assert "AML.T0000" in ids
    assert "AML.T0001" in ids
    assert "AML.T0001.001" in ids
    assert "AML.M0000" in ids
    assert len(result.controls) == 4

    tech = next(c for c in result.controls if c.control_id == "AML.T0000")
    assert tech.hierarchy_level == "technique"

    mit = next(c for c in result.controls if c.control_id == "AML.M0000")
    assert mit.hierarchy_level == "mitigation"

    sub = next(c for c in result.controls if c.control_id == "AML.T0001.001")
    assert sub.hierarchy_level == "sub-technique"
    assert sub.parent_id == "AML.T0001"
