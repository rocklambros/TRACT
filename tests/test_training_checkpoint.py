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
