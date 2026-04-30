"""Tests for deployment model utilities."""
from __future__ import annotations

import numpy as np
import pytest

from tract.training.data_quality import QualityTier, TieredLink


class TestSelectHoldout:
    def test_returns_correct_counts(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(500)
        ]

        cal, canary, remaining = select_holdout(links, n_cal=420, n_canary=20, seed=42)
        assert len(cal) == 420
        assert len(canary) == 20
        assert len(remaining) == 60

    def test_deterministic_with_seed(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "CWE", "section_name": f"s{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(500)
        ]

        cal1, _, _ = select_holdout(links, seed=42)
        cal2, _, _ = select_holdout(links, seed=42)
        assert [l.link["cre_id"] for l in cal1] == [l.link["cre_id"] for l in cal2]

    def test_no_overlap(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(500)
        ]

        cal, canary, remaining = select_holdout(links, seed=42)
        cal_ids = {l.link["cre_id"] for l in cal}
        canary_ids = {l.link["cre_id"] for l in canary}
        remaining_ids = {l.link["cre_id"] for l in remaining}

        assert cal_ids & canary_ids == set()
        assert cal_ids & remaining_ids == set()
        assert canary_ids & remaining_ids == set()

    def test_excludes_ai_frameworks(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": "h1", "standard_name": "MITRE ATLAS", "section_name": "s1", "link_type": "LinkedTo"}, tier=QualityTier.T1_AI),
            TieredLink(link={"cre_id": "h2", "standard_name": "ASVS", "section_name": "s2", "link_type": "LinkedTo"}, tier=QualityTier.T1),
        ]

        cal, canary, remaining = select_holdout(links, n_cal=1, n_canary=0, seed=42)
        assert all(l.link["standard_name"] not in {"MITRE ATLAS"} for l in cal)


class TestHoldoutToEval:
    def test_converts_to_eval_dict(self) -> None:
        from tract.active_learning.deploy import holdout_to_eval

        link = TieredLink(
            link={"cre_id": "236-712", "standard_name": "ASVS", "section_name": "Validate all input", "section_id": "V5.1.1", "link_type": "LinkedTo"},
            tier=QualityTier.T1,
        )

        result = holdout_to_eval(link)
        assert result["valid_hub_ids"] == frozenset({"236-712"})
        assert "Validate all input" in result["control_text"]
        assert result["framework"] == "ASVS"
