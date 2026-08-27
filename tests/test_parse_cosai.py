"""Tests for parsers/parse_cosai.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "cosai"
    # Guard on the path the parser actually reads. Until 2026-08-26 this checked
    # `controls.yaml` at the framework root -- the layout parse_cosai used before
    # 731b167 moved it under risk-map/ to match the upstream CoSAI checkout. The
    # test file was never updated with the parser, so on a faithful risk-map/
    # tree the guard was False and this test SKIPPED SILENTLY, while on a
    # flattened copy it passed the guard and then died in the parser on the
    # nested path. It had real coverage on neither layout. A guard that names a
    # path the code under test never opens can only report on the wrong thing.
    if not (raw_dir / "risk-map" / "controls.yaml").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_cosai import CosaiParser

    with tempfile.TemporaryDirectory() as td:
        parser = CosaiParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "cosai"
        assert len(result.controls) >= 20
        levels = {c.hierarchy_level for c in result.controls}
        assert "control" in levels
        assert "risk" in levels
        assert all(c.control_id for c in result.controls)
