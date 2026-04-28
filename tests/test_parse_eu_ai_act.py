"""Tests for parsers/parse_eu_ai_act.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR


def test_parses_real_data() -> None:
    raw_dir = RAW_FRAMEWORKS_DIR / "eu_ai_act"
    if not (raw_dir / "eu_ai_act_2024_1689.html").exists():
        pytest.skip("Raw data not available")

    from parsers.parse_eu_ai_act import EuAiActParser

    with tempfile.TemporaryDirectory() as td:
        parser = EuAiActParser(raw_dir=raw_dir, output_dir=Path(td))
        result = parser.run()

        assert result.framework_id == "eu_ai_act"
        assert len(result.controls) >= 90

        ids = {c.control_id for c in result.controls}
        assert "AIA-Art1" in ids, "Missing Article 1"
        assert "AIA-Art113" in ids, "Missing Article 113"

        annex_ids = {c.control_id for c in result.controls if c.hierarchy_level == "annex"}
        assert len(annex_ids) >= 1, "No annex controls found"
        assert "AIA-AnnexI" in annex_ids, "Missing Annex I"

        assert all(c.description for c in result.controls), (
            "Some controls have empty descriptions"
        )

        article_controls = [c for c in result.controls if c.hierarchy_level == "article"]
        annex_controls = [c for c in result.controls if c.hierarchy_level == "annex"]
        assert len(article_controls) >= 100, (
            f"Expected >=100 articles, got {len(article_controls)}"
        )
        assert len(annex_controls) >= 10, (
            f"Expected >=10 annexes, got {len(annex_controls)}"
        )

        for c in result.controls:
            assert c.metadata is not None, f"{c.control_id} missing metadata"
