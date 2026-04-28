"""Tests for parsers/parse_owasp_llm_top10.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "owasp_llm_top10"
    if not (raw_dir / "owasp_llm_top_10_2025.md").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_owasp_llm_top10 import OwaspLlmTop10Parser

    with tempfile.TemporaryDirectory() as td:
        parser = OwaspLlmTop10Parser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "owasp_llm_top10"
        assert len(result.controls) == 10
        ids = [c.control_id for c in result.controls]
        assert ids == [
            "LLM01:2025", "LLM02:2025", "LLM03:2025", "LLM04:2025",
            "LLM05:2025", "LLM06:2025", "LLM07:2025", "LLM08:2025",
            "LLM09:2025", "LLM10:2025",
        ]
        assert all(c.description for c in result.controls)
