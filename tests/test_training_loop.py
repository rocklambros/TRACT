"""Tests for the LoRA training loop.

These tests run on CPU with minimal data to verify the training
pipeline wiring. They do NOT test model quality.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.training.config import TrainingConfig


class TestTrainingConfig:

    def test_defaults(self) -> None:
        config = TrainingConfig(name="test")
        assert config.base_model == "BAAI/bge-large-en-v1.5"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.batch_size == 32
        assert config.seed == 42

    def test_frozen(self) -> None:
        config = TrainingConfig(name="test")
        with pytest.raises(AttributeError):
            config.name = "changed"  # type: ignore[misc]

    def test_to_dict_roundtrip(self) -> None:
        config = TrainingConfig(name="test-run", data_hash="abc123")
        d = config.to_dict()
        assert d["name"] == "test-run"
        assert d["data_hash"] == "abc123"
        assert d["lora_rank"] == 16
        assert isinstance(d["lora_target_modules"], list)

    def test_custom_overrides(self) -> None:
        config = TrainingConfig(
            name="custom",
            lora_rank=8,
            batch_size=32,
            learning_rate=1e-3,
        )
        assert config.lora_rank == 8
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3


class TestLoadBaseModel:

    @pytest.mark.slow
    def test_loads_bge_with_lora(self) -> None:
        from tract.training.loop import load_model_with_lora

        config = TrainingConfig(name="test", lora_rank=4, max_seq_length=64)
        model = load_model_with_lora(config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total
        assert trainable > 0

    @pytest.mark.slow
    def test_full_finetune_when_rank_zero(self) -> None:
        from tract.training.loop import load_model_with_lora

        config = TrainingConfig(name="test", lora_rank=0, max_seq_length=64)
        model = load_model_with_lora(config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total


class TestTrainStep:

    @pytest.mark.slow
    def test_single_epoch_smoke(self) -> None:
        from datasets import Dataset

        from tract.training.loop import train_model

        config = TrainingConfig(
            name="smoke-test",
            lora_rank=4,
            batch_size=2,
            max_epochs=1,
            max_seq_length=32,
            learning_rate=1e-4,
        )
        train_data = Dataset.from_list([
            {"anchor": "SQL injection attack", "positive": "Root > Security | Injection"},
            {"anchor": "Cross-site scripting", "positive": "Root > Security | XSS"},
            {"anchor": "Buffer overflow", "positive": "Root > Security | Memory Safety"},
            {"anchor": "Broken authentication", "positive": "Root > Security | Auth"},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_model(config, train_data, output_dir=Path(tmpdir))
            assert model is not None


class TestSaveCheckpoint:

    @pytest.mark.slow
    def test_saves_metadata(self) -> None:
        import json

        from tract.training.loop import load_model_with_lora, save_checkpoint

        config = TrainingConfig(name="ckpt-test", lora_rank=4, max_seq_length=32)
        model = load_model_with_lora(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_checkpoint(
                model, config,
                metrics={"hit_at_1": 0.5},
                output_dir=Path(tmpdir) / "checkpoint",
                git_sha="abc123",
            )
            meta_path = out / "metadata.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["name"] == "ckpt-test"
            assert meta["metrics"]["hit_at_1"] == 0.5
            assert meta["git_sha"] == "abc123"
