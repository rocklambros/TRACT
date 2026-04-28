"""Tests for parsers/parse_cosai.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "cosai"
    if not (raw_dir / "controls.yaml").exists():
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
