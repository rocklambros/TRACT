"""Tests for parsers/parse_owasp_agentic_top10.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_agentic_top10"
    if not (raw_dir / "owasp_agentic_top10_2026.md").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_owasp_agentic_top10 import OwaspAgenticTop10Parser

    with tempfile.TemporaryDirectory() as td:
        parser = OwaspAgenticTop10Parser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_agentic_top10"
        assert len(result.controls) == 10
        ids = [c.control_id for c in result.controls]
        assert ids == ["ASI01", "ASI02", "ASI03", "ASI04", "ASI05",
                       "ASI06", "ASI07", "ASI08", "ASI09", "ASI10"]
        assert all(c.description for c in result.controls)
