"""Tests for deployment model training wrapper."""
from __future__ import annotations

from tract.training.data_quality import TieredLink, QualityTier


class TestPrepareDeploymentTrainingData:
    def test_removes_holdout_from_training(self) -> None:
        from tract.active_learning.train_deploy import prepare_deployment_training_data

        all_links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "section_id": f"id{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(100)
        ]
        holdout = all_links[:10]
        remaining = prepare_deployment_training_data(all_links, holdout)
        assert len(remaining) == 90
        holdout_ids = {(l.link["section_name"], l.link["cre_id"]) for l in holdout}
        for link in remaining:
            assert (link.link["section_name"], link.link["cre_id"]) not in holdout_ids

    def test_preserves_ai_links(self) -> None:
        from tract.active_learning.train_deploy import prepare_deployment_training_data

        trad_links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "section_id": f"id{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(50)
        ]
        ai_links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "MITRE ATLAS", "section_name": f"a{i}", "section_id": f"aid{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1_AI)
            for i in range(10)
        ]
        all_links = trad_links + ai_links
        holdout = trad_links[:5]
        remaining = prepare_deployment_training_data(all_links, holdout)
        ai_count = sum(1 for l in remaining if l.tier == QualityTier.T1_AI)
        assert ai_count == 10
