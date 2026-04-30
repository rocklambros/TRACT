"""Tests for training data quality pipeline."""
from __future__ import annotations

import pytest

from tract.training.data_quality import (
    QualityTier,
    TieredLink,
    assign_quality_tier,
    compute_data_hash,
    filter_training_links,
)


class TestQualityTierAssignment:
    """Test quality tier assignment logic."""

    def test_human_linked_traditional_is_t1(self) -> None:
        link = {
            "cre_id": "760-764",
            "cre_name": "Injection protection",
            "standard_name": "OWASP Top 10 2021",
            "link_type": "LinkedTo",
            "section_id": "A03",
            "section_name": "Injection prevention and input validation",
            "framework_id": "owasp_top10_2021",
        }
        assert assign_quality_tier(link) == QualityTier.T1

    def test_auto_linked_is_t3(self) -> None:
        link = {
            "cre_id": "760-764",
            "cre_name": "Injection protection",
            "standard_name": "CAPEC",
            "link_type": "AutomaticallyLinkedTo",
            "section_id": "CAPEC-66",
            "section_name": "SQL Injection",
            "framework_id": "capec",
        }
        assert assign_quality_tier(link) == QualityTier.T3

    def test_ai_framework_is_t1_ai(self) -> None:
        link = {
            "cre_id": "123-456",
            "cre_name": "Test",
            "standard_name": "MITRE ATLAS",
            "link_type": "LinkedTo",
            "section_id": "AML.T0001",
            "section_name": "Adversarial Perturbation",
            "framework_id": "mitre_atlas",
        }
        assert assign_quality_tier(link) == QualityTier.T1_AI

    def test_bare_id_short_text_is_dropped(self) -> None:
        link = {
            "cre_id": "111-222",
            "cre_name": "Test",
            "standard_name": "NIST 800-63",
            "link_type": "LinkedTo",
            "section_id": "5.1.4.2",
            "section_name": "5.1.4.2",
            "framework_id": "nist_800_63",
        }
        assert assign_quality_tier(link) == QualityTier.DROPPED

    def test_short_section_name_is_dropped(self) -> None:
        link = {
            "cre_id": "333-444",
            "cre_name": "Test",
            "standard_name": "DSOMM",
            "link_type": "LinkedTo",
            "section_id": "D1",
            "section_name": "Process",
            "framework_id": "dsomm",
        }
        assert assign_quality_tier(link) == QualityTier.DROPPED

    def test_dropped_framework_regardless_of_text(self) -> None:
        link = {
            "cre_id": "555-666",
            "cre_name": "Test",
            "standard_name": "OWASP Proactive Controls",
            "link_type": "LinkedTo",
            "section_id": "C1",
            "section_name": "A very descriptive section name that is long enough",
            "framework_id": "owasp_proactive_controls",
        }
        assert assign_quality_tier(link) == QualityTier.DROPPED

    def test_all_five_ai_frameworks_are_t1_ai(self) -> None:
        for name in [
            "MITRE ATLAS",
            "NIST AI 100-2",
            "OWASP AI Exchange",
            "OWASP Top10 for LLM",
            "OWASP Top10 for ML",
        ]:
            link = {
                "cre_id": "100-200",
                "cre_name": "Test",
                "standard_name": name,
                "link_type": "LinkedTo",
                "section_id": "X1",
                "section_name": "A descriptive section text",
                "framework_id": "test_fw",
            }
            assert assign_quality_tier(link) == QualityTier.T1_AI, f"Failed for {name}"


class TestFilterTrainingLinks:
    """Test end-to-end link filtering."""

    def test_filters_dropped_links(self) -> None:
        links = [
            {
                "cre_id": "1",
                "cre_name": "A",
                "standard_name": "ASVS",
                "link_type": "LinkedTo",
                "section_id": "V1",
                "section_name": "Architecture Assessment",
                "framework_id": "asvs",
            },
            {
                "cre_id": "2",
                "cre_name": "B",
                "standard_name": "NIST 800-63",
                "link_type": "LinkedTo",
                "section_id": "5.1.4.2",
                "section_name": "5.1.4.2",
                "framework_id": "nist_800_63",
            },
        ]
        result = filter_training_links(links)
        assert len(result) == 1
        assert result[0].link["cre_id"] == "1"
        assert result[0].tier == QualityTier.T1

    def test_preserves_all_ai_links(self) -> None:
        links = [
            {
                "cre_id": "1",
                "cre_name": "A",
                "standard_name": "MITRE ATLAS",
                "link_type": "LinkedTo",
                "section_id": "AML.T0001",
                "section_name": "Adversarial Perturbation",
                "framework_id": "mitre_atlas",
            },
        ]
        result = filter_training_links(links)
        assert len(result) == 1
        assert result[0].tier == QualityTier.T1_AI

    def test_mixed_tiers_in_result(self) -> None:
        links = [
            {
                "cre_id": "1",
                "cre_name": "A",
                "standard_name": "ASVS",
                "link_type": "LinkedTo",
                "section_id": "V1",
                "section_name": "Architecture Assessment",
                "framework_id": "asvs",
            },
            {
                "cre_id": "2",
                "cre_name": "B",
                "standard_name": "CAPEC",
                "link_type": "AutomaticallyLinkedTo",
                "section_id": "CAPEC-66",
                "section_name": "SQL Injection Attack",
                "framework_id": "capec",
            },
            {
                "cre_id": "3",
                "cre_name": "C",
                "standard_name": "OWASP AI Exchange",
                "link_type": "LinkedTo",
                "section_id": "OAI-1",
                "section_name": "AI Supply Chain Integrity",
                "framework_id": "owasp_ai_exchange",
            },
        ]
        result = filter_training_links(links)
        assert len(result) == 3
        tiers = {r.tier for r in result}
        assert tiers == {QualityTier.T1, QualityTier.T3, QualityTier.T1_AI}


class TestDataHash:
    """Test deterministic data hashing."""

    def test_hash_is_deterministic(self) -> None:
        data = [{"a": 1}, {"b": 2}]
        h1 = compute_data_hash(data)
        h2 = compute_data_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_data(self) -> None:
        h1 = compute_data_hash([{"a": 1}])
        h2 = compute_data_hash([{"a": 2}])
        assert h1 != h2

    def test_hash_is_order_independent_for_keys(self) -> None:
        h1 = compute_data_hash([{"a": 1, "b": 2}])
        h2 = compute_data_hash([{"b": 2, "a": 1}])
        assert h1 == h2  # sort_keys=True makes this deterministic


def test_quality_tier_al_exists() -> None:
    from tract.training.data_quality import QualityTier
    assert QualityTier.AL.value == "AL"
