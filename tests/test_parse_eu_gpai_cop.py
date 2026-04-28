"""Tests for parsers/parse_eu_gpai_cop.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "eu_gpai_cop"
    if not (raw_dir / "gpai_code_of_practice_combined.md").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_eu_gpai_cop import EuGpaiCopParser

    with tempfile.TemporaryDirectory() as td:
        parser = EuGpaiCopParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "eu_gpai_cop"
        assert len(result.controls) >= 35
        chapters = {c.metadata["chapter"] for c in result.controls if c.metadata}
        assert "Transparency" in chapters
        assert "Copyright" in chapters
        assert "Safety and Security" in chapters
        assert all(c.description for c in result.controls)
