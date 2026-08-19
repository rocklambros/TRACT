"""Tests for tract.publish.model_card — AIBOM-compliant model card generation."""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_FOLD_RESULTS = [
    {"fold": "MITRE ATLAS", "hit1": 0.279, "zs_hit1": 0.273, "n": 43, "hit_any": 0.35},
    {"fold": "NIST AI 100-2", "hit1": 0.429, "zs_hit1": 0.107, "n": 28, "hit_any": 0.50},
    {"fold": "OWASP AI Exchange", "hit1": 0.762, "zs_hit1": 0.619, "n": 63, "hit_any": 0.82},
    {"fold": "OWASP Top10 for LLM", "hit1": 0.333, "zs_hit1": 0.333, "n": 6, "hit_any": 0.50},
    {"fold": "OWASP Top10 for ML", "hit1": 0.714, "zs_hit1": 0.429, "n": 7, "hit_any": 0.86},
]

# The card computes the aggregate hit@1 interval by fold-stratified bootstrap,
# so each fold must carry its per-item indicators. Derived from the fold's own
# hit1 and n to keep the fixture self-consistent.
for _f in SAMPLE_FOLD_RESULTS:
    _hits = round(_f["hit1"] * _f["n"])
    _f["hit1_indicators"] = [1.0] * _hits + [0.0] * (_f["n"] - _hits)

SAMPLE_CALIBRATION = {
    "t_deploy": 0.074,
    "ood_threshold": 0.568,
    "conformal_quantile": 0.997,
}

SAMPLE_ECE = {"ece": 0.079, "ece_ci": {"ci_low": 0.049, "ci_high": 0.111}}

SAMPLE_BRIDGE = {"counts": {"accepted": 5, "rejected": 58, "total": 63}}


class TestGenerateModelCard:

    def test_creates_readme(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path,
            fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION,
            ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE,
            gpu_hours=2.5,
        )
        assert (tmp_path / "README.md").exists()

    def test_contains_model_description(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "TRACT" in content
        assert "CRE" in content
        assert "bi-encoder" in content.lower() or "bi_encoder" in content.lower()

    def test_contains_lofo_table(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "MITRE ATLAS" in content
        assert "0.279" in content
        assert "hit@any" in content.lower() or "hit_any" in content.lower()

    def test_contains_calibration(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "0.074" in content or "0.0738" in content
        assert "0.079" in content

    def test_contains_limitations(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "limitation" in content.lower()

    def test_contains_license(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "MIT" in content

    def test_contains_bridge_summary(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "bridge" in content.lower()
        assert "5" in content  # accepted count

    def test_no_secrets_in_output(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "/home/rock" not in content
        assert "sk-" not in content
        assert "hf_" not in content


class TestErratumSurvivesRegeneration:
    """README.md links to #erratum-2026-08-15 on the published model card.

    The erratum lived only in the uploaded artifact, not in this generator, so
    the next publish would have dropped the section and left the repository
    pointing at an anchor that no longer resolved. These tests pin the erratum
    to the generator and pin the heading text that produces the anchor.
    """

    def _card(self, tmp_path) -> str:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path,
            fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION,
            ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE,
            gpu_hours=2.5,
        )
        return (tmp_path / "README.md").read_text(encoding="utf-8")

    def test_heading_produces_the_anchor_the_readme_links_to(self, tmp_path) -> None:
        # GitHub and HuggingFace slugify "## Erratum 2026-08-15" to
        # "#erratum-2026-08-15". Changing this heading breaks README.md:48.
        assert "## Erratum 2026-08-15" in self._card(tmp_path)

    def test_states_the_figures_are_withdrawn(self, tmp_path) -> None:
        card = self._card(tmp_path).lower()
        assert "withdrawn" in card
        assert "pre-registered gate" in card

    def test_names_the_specific_audit_failures(self, tmp_path) -> None:
        card = self._card(tmp_path)
        assert "arithmetic on the point estimate" in card
        assert "-0.0004" in card
        assert "1,265" in card

    def test_scopes_the_review_claim(self, tmp_path) -> None:
        # The card is hard-wrapped, so claims span newlines. Collapse whitespace
        # before matching or the assertion depends on where the wrap happens to
        # fall, which is not what is being tested.
        card = " ".join(self._card(tmp_path).split())
        assert "single reviewer" in card
        assert "13 of 20" in card
        assert "Inter-rater reliability is not measured" in card
        assert "imported rather than reviewed here" in card
