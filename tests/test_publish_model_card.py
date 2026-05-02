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
