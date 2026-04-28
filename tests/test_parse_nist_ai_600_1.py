"""Tests for parsers/parse_nist_ai_600_1.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "nist_ai_600_1"
    if not (raw_dir / "nist_ai_600_1.md").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_nist_ai_600_1 import NistAi600Parser

    with tempfile.TemporaryDirectory() as td:
        parser = NistAi600Parser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "nist_ai_600_1"
        assert len(result.controls) == 12
        ids = {c.control_id for c in result.controls}
        assert "GAI-CBRN" in ids
        assert "GAI-CONFAB" in ids
        assert "GAI-VALUECHAIN" in ids
