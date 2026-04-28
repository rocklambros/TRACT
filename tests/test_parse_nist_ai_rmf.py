"""Tests for parsers/parse_nist_ai_rmf.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "nist_ai_rmf"
    if not (raw_dir / "nist_ai_rmf_1.0.md").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_nist_ai_rmf import NistAiRmfParser

    with tempfile.TemporaryDirectory() as td:
        parser = NistAiRmfParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "nist_ai_rmf"
        assert len(result.controls) >= 65
        funcs = {c.metadata["function"] for c in result.controls if c.metadata}
        assert funcs == {"GOVERN", "MAP", "MEASURE", "MANAGE"}
        assert all(c.description for c in result.controls)
