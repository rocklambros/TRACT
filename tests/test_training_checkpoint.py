"""Tests for tract.training.checkpoint.

These drive the checkpoint-completion logic with stand-in objects rather than a
real SentenceTransformer, so they run anywhere. The end-to-end proof that a
completed checkpoint actually reloads lives in tests/test_training_loop.py and
needs the pinned training stack.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tract.training.checkpoint import (
    ADAPTER_CONFIG_NAME,
    HF_CONFIG_NAME,
    assert_loadable_checkpoint,
    save_sentence_transformer,
)


class _FakeConfig:
    """Stands in for a transformers PretrainedConfig."""

    def __init__(self, payload: dict[str, Any], writes_file: bool = True) -> None:
        self.payload = payload
        self.writes_file = writes_file

    def save_pretrained(self, output_path: str) -> None:
        if not self.writes_file:
            return
        target = Path(output_path) / HF_CONFIG_NAME
        target.write_text(json.dumps(self.payload, sort_keys=True), encoding="utf-8")


class _FakeBackbone:
    """Stands in for the transformer backbone held by ST module 0."""

    def __init__(self, config: _FakeConfig | None) -> None:
        self.config = config


class _FakeModule:
    """Stands in for a sentence_transformers Transformer module."""

    def __init__(self, backbone: _FakeBackbone | None) -> None:
        if backbone is not None:
            self.auto_model = backbone


class _FakeModel:
    """Stands in for a SentenceTransformer.

    ``save`` writes whichever files the scenario needs, mirroring what
    transformers does for an adapter-carrying model versus a plain one.
    """

    def __init__(self, module: _FakeModule, files: dict[str, str]) -> None:
        self._module = module
        self.files = files
        self.save_calls: list[str] = []

    def __getitem__(self, index: int) -> _FakeModule:
        if index != 0:
            raise IndexError(index)
        return self._module

    def save(self, output_path: str) -> None:
        self.save_calls.append(output_path)
        for name, body in self.files.items():
            (Path(output_path) / name).write_text(body, encoding="utf-8")


def _adapter_only_model(model_type: str = "bert") -> _FakeModel:
    """A model whose save leaves out config.json, as an adapter save does."""
    config = _FakeConfig({"model_type": model_type, "architectures": ["BertModel"]})
    module = _FakeModule(_FakeBackbone(config))
    return _FakeModel(module, {
        ADAPTER_CONFIG_NAME: json.dumps({"base_model_name_or_path": "BAAI/bge-large-en-v1.5"}),
        "tokenizer_config.json": "{}",
    })


class TestSaveSentenceTransformer:

    def test_completes_an_adapter_only_save(self, tmp_path: Path) -> None:
        """The whole point: the saved directory ends up self-describing."""
        model = _adapter_only_model()
        out = tmp_path / "checkpoint"

        result = save_sentence_transformer(model, out)  # type: ignore[arg-type]

        assert result == out
        written = json.loads((out / HF_CONFIG_NAME).read_text(encoding="utf-8"))
        assert written["model_type"] == "bert"
        # The adapter must still be there. A "fix" that merged or dropped it
        # would satisfy the config assertion above and destroy the checkpoint.
        assert (out / ADAPTER_CONFIG_NAME).is_file()

    def test_creates_the_output_directory(self, tmp_path: Path) -> None:
        model = _adapter_only_model()
        out = tmp_path / "nested" / "checkpoint"

        save_sentence_transformer(model, out)  # type: ignore[arg-type]

        assert out.is_dir()
        assert model.save_calls == [str(out)]

    def test_leaves_an_existing_config_untouched(self, tmp_path: Path) -> None:
        """A full fine-tune's config is written by transformers, not by us.

        Rewriting it from the live config would undo the generation-parameter
        migration transformers performs on the way out.
        """
        config = _FakeConfig({"model_type": "from-the-live-model"})
        module = _FakeModule(_FakeBackbone(config))
        model = _FakeModel(module, {
            HF_CONFIG_NAME: json.dumps({"model_type": "from-save-pretrained"}),
        })
        out = tmp_path / "checkpoint"

        save_sentence_transformer(model, out)  # type: ignore[arg-type]

        written = json.loads((out / HF_CONFIG_NAME).read_text(encoding="utf-8"))
        assert written["model_type"] == "from-save-pretrained"

    def test_raises_when_the_module_exposes_no_backbone(self, tmp_path: Path) -> None:
        model = _FakeModel(_FakeModule(None), {ADAPTER_CONFIG_NAME: "{}"})

        with pytest.raises(RuntimeError, match="exposes no auto_model"):
            save_sentence_transformer(model, tmp_path / "checkpoint")  # type: ignore[arg-type]

    def test_raises_when_the_backbone_carries_no_config(self, tmp_path: Path) -> None:
        model = _FakeModel(_FakeModule(_FakeBackbone(None)), {ADAPTER_CONFIG_NAME: "{}"})

        with pytest.raises(RuntimeError, match="carries no config attribute"):
            save_sentence_transformer(model, tmp_path / "checkpoint")  # type: ignore[arg-type]

    def test_raises_when_the_config_write_produces_nothing(self, tmp_path: Path) -> None:
        """save_pretrained returning quietly without writing must not pass."""
        config = _FakeConfig({"model_type": "bert"}, writes_file=False)
        model = _FakeModel(
            _FakeModule(_FakeBackbone(config)), {ADAPTER_CONFIG_NAME: "{}"},
        )

        with pytest.raises(RuntimeError, match="still absent"):
            save_sentence_transformer(model, tmp_path / "checkpoint")  # type: ignore[arg-type]


class TestAssertLoadableCheckpoint:

    def test_accepts_a_directory_with_a_base_config(self, tmp_path: Path) -> None:
        (tmp_path / HF_CONFIG_NAME).write_text('{"model_type": "bert"}', encoding="utf-8")

        assert_loadable_checkpoint(tmp_path)

    def test_accepts_a_completed_adapter_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / HF_CONFIG_NAME).write_text('{"model_type": "bert"}', encoding="utf-8")
        (tmp_path / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")

        assert_loadable_checkpoint(tmp_path)

    def test_rejects_an_adapter_only_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")

        with pytest.raises(RuntimeError, match="adapter-only checkpoint"):
            assert_loadable_checkpoint(tmp_path)

    def test_rejects_a_directory_that_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / "modules.json").write_text("[]", encoding="utf-8")

        with pytest.raises(RuntimeError, match="not a model checkpoint"):
            assert_loadable_checkpoint(tmp_path)

    def test_a_directory_named_config_json_is_not_a_config(self, tmp_path: Path) -> None:
        """is_file, not exists: a directory of that name must not satisfy the guard."""
        (tmp_path / HF_CONFIG_NAME).mkdir()
        (tmp_path / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")

        with pytest.raises(RuntimeError, match="adapter-only checkpoint"):
            assert_loadable_checkpoint(tmp_path)


class TestRepairAdapterOnlyCheckpoint:
    """D2(b), answered 2026-08-26: make the 98 adapter-only checkpoints loadable.

    They were written before save_sentence_transformer existed, so every one
    carries correct weights that no consumer can open. The repair is the second
    half of assert_loadable_checkpoint's own error message: copy the base
    model's config.json in beside the adapter. It is a file operation, which
    matters because it must run on a machine that never allocates a model.

    The guard that earns its place here is the base-model match. These are 98
    artifacts whose provenance nobody has audited, and 95 of them name
    BAAI/bge-large-en-v1.5 while 3 name Qwen/Qwen3-Embedding-0.6B. Writing the
    wrong backbone's config produces a checkpoint that opens and is wrong,
    which is worse than one that refuses to open.

    The match is against the repo id the caller fetched the config FOR, never
    against `_name_or_path` inside the config. See the build-path test below:
    that field records wherever the config was last saved from, which for a
    published model is a path on the publisher's build machine.
    """

    BGE = "BAAI/bge-large-en-v1.5"
    QWEN = "Qwen/Qwen3-Embedding-0.6B"

    def _adapter_dir(self, tmp_path: Path, base_model: str) -> Path:
        d = tmp_path / "checkpoint-1234"
        d.mkdir()
        (d / ADAPTER_CONFIG_NAME).write_text(
            json.dumps({"base_model_name_or_path": base_model, "peft_type": "LORA"}),
            encoding="utf-8",
        )
        return d

    def _base_config(
        self, tmp_path: Path, name_or_path: str, model_type: str = "bert"
    ) -> Path:
        p = tmp_path / "base_config.json"
        p.write_text(
            json.dumps({"_name_or_path": name_or_path, "model_type": model_type}),
            encoding="utf-8",
        )
        return p

    def test_the_repaired_checkpoint_becomes_loadable(self, tmp_path: Path) -> None:
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        cfg = self._base_config(tmp_path, self.BGE)

        with pytest.raises(RuntimeError):
            assert_loadable_checkpoint(d)

        assert repair_adapter_only_checkpoint(d, cfg, self.BGE) is True
        assert_loadable_checkpoint(d)

    def test_the_written_config_is_the_base_config(self, tmp_path: Path) -> None:
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        cfg = self._base_config(tmp_path, self.BGE)

        repair_adapter_only_checkpoint(d, cfg, self.BGE)

        written = json.loads((d / HF_CONFIG_NAME).read_text(encoding="utf-8"))
        assert written["model_type"] == "bert"

    def test_a_publisher_build_path_is_not_treated_as_a_mismatch(
        self, tmp_path: Path
    ) -> None:
        """BAAI shipped bge-large-en-v1.5 with a build-machine path in the config.

        `_name_or_path` reads
        `/root/.cache/torch/sentence_transformers/BAAI_bge-large-en/` in the
        published file. Ninety-five of the 98 checkpoints are this backbone, so
        a guard reading that field refuses the entire real workload.
        """
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        cfg = self._base_config(
            tmp_path, "/root/.cache/torch/sentence_transformers/BAAI_bge-large-en/"
        )

        assert repair_adapter_only_checkpoint(d, cfg, self.BGE) is True
        assert_loadable_checkpoint(d)

    def test_a_mismatched_backbone_is_refused(self, tmp_path: Path) -> None:
        """A Qwen checkpoint must not receive the config fetched for BGE."""
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.QWEN)
        cfg = self._base_config(tmp_path, self.BGE)

        with pytest.raises(ValueError, match="names Qwen/Qwen3-Embedding-0.6B"):
            repair_adapter_only_checkpoint(d, cfg, self.BGE)

        assert not (d / HF_CONFIG_NAME).exists()

    def test_an_already_complete_checkpoint_is_left_alone(self, tmp_path: Path) -> None:
        """Idempotent, and it must not rewrite a config transformers wrote."""
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        original = json.dumps({"model_type": "bert", "written_by": "transformers"})
        (d / HF_CONFIG_NAME).write_text(original, encoding="utf-8")
        cfg = self._base_config(tmp_path, self.BGE)

        assert repair_adapter_only_checkpoint(d, cfg, self.BGE) is False
        assert (d / HF_CONFIG_NAME).read_text(encoding="utf-8") == original

    def test_a_directory_that_is_not_a_checkpoint_raises(self, tmp_path: Path) -> None:
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = tmp_path / "not-a-checkpoint"
        d.mkdir()
        cfg = self._base_config(tmp_path, self.BGE)

        with pytest.raises(ValueError, match="no adapter_config.json"):
            repair_adapter_only_checkpoint(d, cfg, self.BGE)
