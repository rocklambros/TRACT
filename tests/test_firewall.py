"""Tests for hub representation firewall."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.hierarchy import CREHierarchy

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase1a_mini_cres.json"


@pytest.fixture
def hierarchy() -> CREHierarchy:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return CREHierarchy.from_opencre(
        cres=data["cres"],
        fetch_timestamp=data["fetch_timestamp"],
        data_hash="abc123",
    )


class TestBuildFirewalledHubText:
    """Test firewalled hub text construction."""

    def test_returns_path_pipe_name(self, hierarchy: CREHierarchy) -> None:
        from tract.training.firewall import build_firewalled_hub_text

        hub_id = list(hierarchy.hubs.keys())[0]
        text = build_firewalled_hub_text(hub_id, hierarchy)
        assert " | " in text
        assert hierarchy.hubs[hub_id].name in text
        assert hierarchy.hubs[hub_id].hierarchy_path in text

    def test_excluded_framework_does_not_affect_primary_rep(
        self, hierarchy: CREHierarchy
    ) -> None:
        from tract.training.firewall import build_firewalled_hub_text

        hub_id = list(hierarchy.hubs.keys())[0]
        text_a = build_firewalled_hub_text(
            hub_id, hierarchy, excluded_framework="MITRE ATLAS"
        )
        text_b = build_firewalled_hub_text(
            hub_id, hierarchy, excluded_framework="OWASP AI Exchange"
        )
        assert text_a == text_b

    def test_format_is_path_pipe_name(self, hierarchy: CREHierarchy) -> None:
        from tract.training.firewall import build_firewalled_hub_text

        for hub_id, node in hierarchy.hubs.items():
            text = build_firewalled_hub_text(hub_id, hierarchy)
            expected = f"{node.hierarchy_path} | {node.name}"
            assert text == expected

    def test_description_appended_when_requested(
        self, hierarchy: CREHierarchy
    ) -> None:
        from tract.training.firewall import build_firewalled_hub_text

        hub_id = list(hierarchy.hubs.keys())[0]
        desc = "A security control for preventing attacks"
        text = build_firewalled_hub_text(
            hub_id,
            hierarchy,
            include_description=True,
            descriptions={hub_id: desc},
        )
        assert desc in text

    def test_standards_appended_excluding_held_out(
        self, hierarchy: CREHierarchy
    ) -> None:
        from tract.training.firewall import build_firewalled_hub_text

        hub_id = list(hierarchy.hubs.keys())[0]
        sections = {hub_id: ["ASVS: V1.1", "MITRE ATLAS: AML.T0001", "CWE: CWE-79"]}
        text = build_firewalled_hub_text(
            hub_id,
            hierarchy,
            excluded_framework="MITRE ATLAS",
            include_standards=True,
            standard_sections=sections,
        )
        assert "ASVS" in text
        assert "CWE" in text
        assert "MITRE ATLAS" not in text


class TestBuildAllHubTexts:
    """Test bulk hub text construction."""

    def test_builds_text_for_all_hubs(self, hierarchy: CREHierarchy) -> None:
        from tract.training.firewall import build_all_hub_texts

        texts = build_all_hub_texts(hierarchy)
        assert len(texts) == len(hierarchy.hubs)
        for hub_id, text in texts.items():
            assert hub_id in hierarchy.hubs
            assert len(text) > 0


class TestFirewallAssertion:
    """Test firewall breach detection."""

    def test_passes_when_no_leakage(self) -> None:
        from tract.training.firewall import assert_firewall

        hub_texts = {
            "hub-1": "Root > Security | Security",
            "hub-2": "Root > Privacy | Privacy",
        }

        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw

        items = [MockItem("SQL Injection attacks", "ATLAS")]
        assert_firewall(hub_texts, items, "ATLAS")

    def test_fails_when_control_text_in_appended_text(self) -> None:
        from tract.training.firewall import assert_firewall

        base_texts = {"hub-1": "Security | Security"}
        hub_texts = {"hub-1": "Security | Security: SQL Injection attacks"}

        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw

        items = [MockItem("SQL Injection attacks", "ATLAS")]
        with pytest.raises(AssertionError, match="Firewall breach"):
            assert_firewall(hub_texts, items, "ATLAS", base_hub_texts=base_texts)

    def test_passes_base_format_with_matching_hub_name(self) -> None:
        from tract.training.firewall import assert_firewall

        hub_texts = {"hub-1": "Root > AI > Adversarial training | Adversarial training"}

        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw

        items = [MockItem("Adversarial training", "NIST AI 100-2")]
        assert_firewall(hub_texts, items, "NIST AI 100-2")

    def test_passes_when_control_matches_hub_name_in_description(self) -> None:
        from tract.training.firewall import assert_firewall

        base_texts = {
            "hub-1": "Root > AI > Adversarial training | Adversarial training",
        }
        hub_texts = {
            "hub-1": "Root > AI > Adversarial training | Adversarial training: "
            "This hub covers adversarial training techniques for ML models",
        }

        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw

        items = [MockItem("Adversarial training", "NIST AI 100-2")]
        assert_firewall(hub_texts, items, "NIST AI 100-2", base_hub_texts=base_texts)

    def test_passes_when_control_is_substring_of_another_hub_name(self) -> None:
        from tract.training.firewall import assert_firewall

        base_texts = {
            "hub-1": "Root > Input > Anomalous handling | Anomalous handling",
            "hub-2": "Root > Input > Rate limiting | Rate limiting",
        }
        hub_texts = {
            "hub-1": "Root > Input > Anomalous handling | Anomalous handling: "
            "distinguishing it from Rate limiting which controls volume",
            "hub-2": "Root > Input > Rate limiting | Rate limiting: "
            "Controls input volume",
        }

        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw

        items = [MockItem("Rate limit", "OWASP AI Exchange")]
        assert_firewall(hub_texts, items, "OWASP AI Exchange", base_hub_texts=base_texts)

    def test_skips_very_short_control_text(self) -> None:
        from tract.training.firewall import assert_firewall

        hub_texts = {"hub-1": "XSS | XSS"}

        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw

        items = [MockItem("XSS", "ATLAS")]
        assert_firewall(hub_texts, items, "ATLAS")
