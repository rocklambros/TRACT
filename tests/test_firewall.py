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


class TestFirewallCanActuallyFail:
    """Regression tests for commit 6548703.

    assert_firewall could not raise on any code path before that commit. With
    the default hub_rep_format ("path+name") base_hub_texts stayed None, and the
    function logged a pass and returned before checking anything. Every LOFO
    number TRACT has published rested on an assertion that was decorative.

    An untested firewall fix is not a firewall, so these tests poison a hub with
    a held-out framework's control text and require the breach to be raised.
    """

    class Item:
        def __init__(self, text: str, fw: str) -> None:
            self.control_text = text
            self.framework = fw

    def test_base_format_leak_raises(self) -> None:
        """The default path must be able to fail.

        The leak is in the hierarchy path rather than the appended slice, which
        is the only place framework text can reach a base-format hub text. Before
        6548703 this returned a logged pass.
        """
        from tract.training.firewall import assert_firewall

        hub_texts = {
            "hub-1": "Root > Governance > Establish an AI incident response plan | Oversight",
        }
        items = [self.Item("Establish an AI incident response plan", "CSA AICM")]

        with pytest.raises(AssertionError, match="Firewall breach"):
            assert_firewall(hub_texts, items, "CSA AICM")

    def test_standards_leak_through_real_builders_raises(
        self, hierarchy: CREHierarchy
    ) -> None:
        """End-to-end leak through build_all_hub_texts, not a hand-built base.

        build_firewalled_hub_text drops a section only when the excluded
        framework name is a substring of it. A section labelled "AICM: ..." does
        not contain "CSA AICM", so it survives the filter and carries the
        held-out framework's control text into the hub representation. This is
        the realistic leakage shape, and the firewall has to catch it.
        """
        from tract.training.firewall import assert_firewall, build_all_hub_texts

        hub_id = next(iter(hierarchy.hubs))
        leaked = "Model provenance must be recorded for every training run"
        sections = {hub_id: [f"AICM: {leaked}"]}

        hub_texts = build_all_hub_texts(
            hierarchy,
            excluded_framework="CSA AICM",
            include_standards=True,
            standard_sections=sections,
        )
        base_hub_texts = build_all_hub_texts(
            hierarchy,
            excluded_framework="CSA AICM",
            include_standards=False,
        )
        assert hub_texts[hub_id] != base_hub_texts[hub_id], (
            "the two builds must differ or the appended slice is empty and "
            "nothing is ever checked"
        )

        items = [self.Item(leaked, "CSA AICM")]
        with pytest.raises(AssertionError, match="Firewall breach"):
            assert_firewall(hub_texts, items, "CSA AICM", base_hub_texts=base_hub_texts)

    def test_standards_format_without_sections_is_refused(
        self, hierarchy: CREHierarchy
    ) -> None:
        """orchestrate must refuse to build a format it cannot firewall.

        Both build_all_hub_texts calls used to be made with an identical argument
        list, so hub_texts == base_hub_texts, every appended slice was empty and
        the breach check was unreachable. Asking for the standards format without
        the sections that define it now fails loudly instead.
        """
        from tract.training.config import TrainingConfig
        from tract.training.orchestrate import run_single_fold

        config = TrainingConfig(name="fw-test", hub_rep_format="path+name+standards")

        with pytest.raises(ValueError, match="requires standard_sections"):
            run_single_fold(
                config=config,
                held_out_framework="CSA AICM",
                tiered_links=[],
                hierarchy=hierarchy,
                eval_items=[],
                hub_ids=[],
                output_dir=Path("/nonexistent-should-not-be-reached"),
                standard_sections=None,
            )

    def test_hub_name_mention_does_not_exempt_the_whole_control(self) -> None:
        """A shared word must not disable the check.

        The skip rule used to read `control_lower in name or name in
        control_lower`. Real CRE hub names include short generic nouns, so the
        second half exempted any control sentence mentioning one -- against every
        hub, not just the hub whose name matched. Here the leak sits in hub-1
        while the control merely mentions hub-2's name.
        """
        from tract.training.firewall import assert_firewall

        leaked = "Establish an AI incident response plan for Data"
        hub_texts = {
            "hub-1": f"Root > Governance > {leaked} | Oversight",
            "hub-2": "Root > Data | Data",
        }
        items = [self.Item(leaked, "CSA AICM")]

        with pytest.raises(AssertionError, match="Firewall breach"):
            assert_firewall(hub_texts, items, "CSA AICM")

    def test_orchestrate_firewalls_a_leaked_description(
        self, hierarchy: CREHierarchy, monkeypatch
    ) -> None:
        """Drive the real orchestrate path, not a hand-paired assert_firewall.

        orchestrate built base_hub_texts only for the standards format, so the
        description format reached assert_firewall with base_hub_texts=None. The
        allowlist is then derived from the hub texts being checked, and
        everything after ' | ' counts as the hub name -- so a control leaked into
        a description matched a "hub name" and was skipped. The poison immunised
        itself. The base has to be plain "path | name", not the requested format
        minus one augmentation.

        The firewall runs before any training, so this needs no model.
        """
        from tract.training import orchestrate as orch
        from tract.training.config import TrainingConfig

        hub_id = next(iter(hierarchy.hubs))
        leaked = "Rotate model signing keys every ninety days"
        monkeypatch.setattr(
            orch, "load_json",
            lambda _path: {"descriptions": {hub_id: {"description": leaked}}},
        )

        config = TrainingConfig(name="fw-desc", hub_rep_format="path+name+desc")

        with pytest.raises(AssertionError, match="Firewall breach"):
            orch.run_single_fold(
                config=config,
                held_out_framework="CSA AICM",
                tiered_links=[],
                hierarchy=hierarchy,
                eval_items=[self.Item(leaked, "CSA AICM")],
                hub_ids=[],
                output_dir=Path("/nonexistent-should-not-be-reached"),
            )
