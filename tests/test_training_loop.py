"""Tests for the LoRA training loop.

These tests run on CPU with minimal data to verify the training
pipeline wiring. They do NOT test model quality.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# These need the optional `phase0` extra (and matplotlib for the notebook
# helpers), which the default CI test job does not install. Skip visibly
# rather than failing collection; run them with
# `pip install -e '.[phase0]' matplotlib`.
pytest.importorskip("torch", reason="needs the phase0 extra")
pytest.importorskip("sentence_transformers", reason="needs the phase0 extra")

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


class TestLoRACheckpointPersistence:
    """Regression tests for the sentence-transformers 5.x auto_model defect.

    `model[0].auto_model = get_peft_model(...)` is silently discarded on
    sentence-transformers 5.7: auto_model became a read-only property, and
    nn.Module.__setattr__ files the value into _modules where property reads
    never look. Training still ran and loss still fell, but save_pretrained
    executed against the unwrapped backbone, so the checkpoint reloaded with
    randomly initialised q/k/v. Measured min cosine on reload was 0.38.

    A loss-only smoke test does not catch this -- one passed while the defect
    was live. Only save-then-reload does.
    """

    # Small BERT-architecture model: same query/key/value target modules as
    # BGE-large, a fraction of the download.
    SMALL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    PROBES = [
        "Implement access controls for AI model training pipelines",
        "Data encryption at rest using AES-256",
        "Regularly audit AI system outputs for bias and fairness",
    ]

    @staticmethod
    def _perturb_adapter(model) -> int:
        """Push lora_B off its zero initialisation.

        LoRA initialises lora_B to zero, so a freshly attached adapter is an
        identity map on the embeddings. A round-trip test against an unperturbed
        adapter passes whether or not the adapter was saved, which is exactly the
        vacuous result this defect hid behind. Perturbing makes the adapter
        observable in the output.
        """
        import torch

        generator = torch.Generator().manual_seed(0)
        count = 0
        for name, param in model.named_parameters():
            if "lora_B" in name:
                with torch.no_grad():
                    param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
                count += 1
        return count

    def _adapted_model(self):
        from tract.training.config import TrainingConfig
        from tract.training.loop import load_model_with_lora

        config = TrainingConfig(
            name="lora-persistence", base_model=self.SMALL_MODEL,
            lora_rank=8, max_seq_length=64,
        )
        model = load_model_with_lora(config)
        assert self._perturb_adapter(model) > 0, "no lora_B parameters: adapter never attached"
        return model

    def _embed(self, model):
        return model.encode(self.PROBES, normalize_embeddings=True, show_progress_bar=False)

    @pytest.mark.slow
    def test_probe_detects_a_missing_adapter(self) -> None:
        """The probe must be able to tell an adapted model from the base model.

        Without this, a passing round-trip test proves nothing.
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer

        adapted = self._adapted_model()
        base = SentenceTransformer(self.SMALL_MODEL)
        base.max_seq_length = 64

        cosines = np.sum(self._embed(adapted) * self._embed(base), axis=1)
        assert float(np.min(cosines)) < 0.999, (
            "perturbed adapter does not change the embeddings, so a round-trip "
            "assertion on these probes would pass vacuously"
        )

    @pytest.mark.slow
    def test_checkpoint_directory_is_self_describing(self) -> None:
        """The saved directory must name its own architecture.

        transformers omits config.json for an adapter save, and
        sentence-transformers 5.7 needs it to resolve a processor. Without it the
        adapter weights are all correct and nothing can open the directory.
        """
        import json

        from tract.training.checkpoint import save_sentence_transformer

        model = self._adapted_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = Path(tmpdir) / "model"
            save_sentence_transformer(model, saved)

            assert (saved / "adapter_config.json").is_file(), (
                "the checkpoint stopped being adapter-only"
            )
            config_path = saved / "config.json"
            assert config_path.is_file(), (
                "no config.json beside the adapter, so SentenceTransformer will "
                "raise 'Unrecognized model' on this directory"
            )
            assert json.loads(config_path.read_text(encoding="utf-8"))["model_type"] == "bert"

    @pytest.mark.slow
    def test_adapter_survives_save_and_reload(self) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        from tract.training.checkpoint import save_sentence_transformer

        model = self._adapted_model()
        before = self._embed(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = Path(tmpdir) / "model"
            save_sentence_transformer(model, saved)

            # Plain load, the way load_fold_model and every downstream consumer
            # opens a checkpoint. Anything cleverer here would verify a path no
            # consumer takes.
            reloaded = SentenceTransformer(str(saved))
            reloaded.max_seq_length = 64
            after = reloaded.encode(
                self.PROBES, normalize_embeddings=True, show_progress_bar=False,
            )

        cosines = np.sum(before * after, axis=1)
        assert float(np.min(cosines)) >= 0.999, (
            f"LoRA adapter did not survive save/reload: per-probe cosines "
            f"{cosines.tolist()}. The saved checkpoint is not the trained model."
        )

    @pytest.mark.slow
    def test_verify_checkpoint_roundtrip_raises_on_mismatch(self) -> None:
        """The runtime guard must fire when the artifact is not the live model.

        Saves the unadapted base model, then asks the guard to verify it against
        an adapted in-memory model. That is precisely the shape of the original
        defect: a healthy live model beside a checkpoint that lost its adapter.
        """
        from sentence_transformers import SentenceTransformer

        from tract.training.checkpoint import save_sentence_transformer
        from tract.training.loop import verify_checkpoint_roundtrip

        adapted = self._adapted_model()
        base = SentenceTransformer(self.SMALL_MODEL)
        base.max_seq_length = 64

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = Path(tmpdir) / "base-model"
            save_sentence_transformer(base, saved)

            with pytest.raises(RuntimeError, match="does not reproduce the in-memory model"):
                verify_checkpoint_roundtrip(adapted, saved)

    @pytest.mark.slow
    def test_verify_checkpoint_roundtrip_passes_on_match(self) -> None:
        from tract.training.checkpoint import save_sentence_transformer
        from tract.training.loop import verify_checkpoint_roundtrip

        model = self._adapted_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = Path(tmpdir) / "model"
            save_sentence_transformer(model, saved)
            assert verify_checkpoint_roundtrip(model, saved) >= 0.999

    @pytest.mark.slow
    def test_verify_checkpoint_roundtrip_rejects_an_adapter_only_directory(self) -> None:
        """A checkpoint written by a bare model.save must not pass verification.

        This is the artifact the defect produced. The guard used to reattach the
        adapter by hand and report a healthy cosine for a directory that no
        consumer could open.
        """
        from tract.training.loop import verify_checkpoint_roundtrip

        model = self._adapted_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = Path(tmpdir) / "model"
            model.save(str(saved))
            assert not (saved / "config.json").exists(), (
                "sentence-transformers now writes a base config for an adapter "
                "save, so this test no longer builds the artifact it describes"
            )

            with pytest.raises(RuntimeError, match="adapter-only checkpoint"):
                verify_checkpoint_roundtrip(model, saved)


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
