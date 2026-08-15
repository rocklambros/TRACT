"""Tests for the encoder registry in tract/encoders.py.

BGE-large's 512-token ceiling is architectural, so testing alternatives means
swapping the encoder. Every swap has to carry a pinned revision, the right
LoRA target module names, and an honest activation cost -- each of which has
already failed silently once in this project.
"""
from __future__ import annotations

import pytest

from tract.encoders import ENCODERS, resolve


class TestRegistryIntegrity:

    def test_every_encoder_pins_a_full_commit(self) -> None:
        """A model fetched at 'main' cannot be tied to the number it made."""
        for name, spec in ENCODERS.items():
            assert len(spec.revision) == 40, name
            assert all(c in "0123456789abcdef" for c in spec.revision), name

    def test_every_encoder_declares_lora_targets(self) -> None:
        for name, spec in ENCODERS.items():
            assert spec.lora_target_modules, name

    def test_target_modules_match_the_architecture(self) -> None:
        """["query","key","value"] matches neither ModernBERT nor Qwen3, and
        PEFT only raises after the zero-shot GPU pass has been paid for."""
        expected = {
            "bert": {"query", "key", "value"},
            "xlm-roberta": {"query", "key", "value"},
            "modernbert": {"Wqkv", "Wo"},
            "qwen3": {"q_proj", "k_proj", "v_proj", "o_proj"},
        }
        for name, spec in ENCODERS.items():
            assert set(spec.lora_target_modules) == expected[spec.model_type], name

    def test_no_encoder_needs_remote_code(self) -> None:
        """trust_remote_code executes repo code on every training pod."""
        assert "Alibaba-NLP/gte-large-en-v1.5" not in ENCODERS

    def test_the_incumbent_is_present_and_still_512(self) -> None:
        bge = resolve("BAAI/bge-large-en-v1.5")
        assert bge.max_seq_length == 512
        assert bge.activation_cost_ratio == pytest.approx(1.0)


class TestActivationCost:
    """The memory guard was calibrated on BGE's shape; wider and deeper
    encoders must scale it or it gives false confidence."""

    def test_a_bigger_encoder_costs_more(self) -> None:
        big = resolve("Qwen/Qwen3-Embedding-4B")
        assert big.activation_cost_ratio > 3.0

    def test_a_smaller_encoder_costs_less(self) -> None:
        small = resolve("Alibaba-NLP/gte-modernbert-base")
        assert small.activation_cost_ratio < 1.0

    def test_cost_tracks_width_times_depth(self) -> None:
        spec = resolve("Qwen/Qwen3-Embedding-0.6B")
        assert spec.activation_cost_ratio == pytest.approx(
            (spec.hidden_size * spec.num_layers) / (1024 * 24)
        )


class TestResolve:

    def test_an_unknown_encoder_is_refused(self) -> None:
        """An allowlist, so an unpinned model cannot reach a pod by typo."""
        with pytest.raises(ValueError, match="Unknown encoder"):
            resolve("some-org/not-a-real-model")

    def test_the_error_lists_what_is_available(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            resolve("nope")
        assert "BAAI/bge-large-en-v1.5" in str(excinfo.value)


class TestConfigIntegration:

    def test_config_resolves_targets_from_the_encoder(self) -> None:
        from tract.training.config import TrainingConfig

        cfg = TrainingConfig(name="t", base_model="Qwen/Qwen3-Embedding-0.6B")
        assert cfg.resolved_lora_targets() == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_an_explicit_setting_wins(self) -> None:
        from tract.training.config import TrainingConfig

        cfg = TrainingConfig(name="t", lora_target_modules=["custom"])
        assert cfg.resolved_lora_targets() == ["custom"]

    def test_the_revision_reaches_the_fold_record(self) -> None:
        """The record must name the weights, not just the repo."""
        from tract.training.config import TrainingConfig

        cfg = TrainingConfig(name="t")
        assert cfg.to_dict()["base_model_revision"] == resolve(cfg.base_model).revision
