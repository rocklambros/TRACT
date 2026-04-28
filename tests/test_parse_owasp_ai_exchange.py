"""Tests for parsers/parse_owasp_ai_exchange.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_ai_exchange"
    if not (raw_dir / "src_1_general_controls.md").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_owasp_ai_exchange import OwaspAiExchangeParser

    with tempfile.TemporaryDirectory() as td:
        parser = OwaspAiExchangeParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_ai_exchange"
        assert len(result.controls) >= 50
        ids = [c.control_id for c in result.controls]
        assert all(c.description for c in result.controls)
