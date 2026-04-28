"""Tests for parsers/parse_owasp_dsgai.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_dsgai"
    if not (raw_dir / "OWASP-GenAI-Data-Security-Risks-and-Mitigations-2026-v1.0.txt").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_owasp_dsgai import OwaspDsgaiParser

    with tempfile.TemporaryDirectory() as td:
        parser = OwaspDsgaiParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_dsgai"
        assert len(result.controls) == 21

        ids = [c.control_id for c in result.controls]
        assert "DSGAI01" in ids
        assert "DSGAI21" in ids

        # All controls must have non-empty descriptions
        assert all(c.description for c in result.controls)

        # Verify first and last entries have correct titles
        first = result.controls[0]
        assert first.control_id == "DSGAI01"
        assert "Sensitive Data Leakage" in first.title

        last = result.controls[-1]
        assert last.control_id == "DSGAI21"
        assert "Disinformation" in last.title

        # Spot-check wrapped-title entries to confirm they are assembled correctly
        dsgai07 = next(c for c in result.controls if c.control_id == "DSGAI07")
        assert "Systems" in dsgai07.title

        dsgai12 = next(c for c in result.controls if c.control_id == "DSGAI12")
        assert "SQL" in dsgai12.title

        dsgai15 = next(c for c in result.controls if c.control_id == "DSGAI15")
        assert "Sharing" in dsgai15.title
        # Title should not contain a spurious space before "Sharing"
        assert "Over- Sharing" not in dsgai15.title
