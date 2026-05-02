"""Tests for tract.publish.merge — LoRA adapter merge into base model."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _create_merged_output(output_dir: Path) -> None:
    """Create the expected post-merge directory structure."""
    (output_dir / "0_Transformer").mkdir(parents=True)
    (output_dir / "0_Transformer" / "model.safetensors").write_bytes(b"fake-weights")
    (output_dir / "0_Transformer" / "config.json").write_text("{}")
    (output_dir / "1_Pooling").mkdir()
    (output_dir / "1_Pooling" / "config.json").write_text("{}")
    (output_dir / "2_Normalize").mkdir()
    (output_dir / "modules.json").write_text("[]")
    (output_dir / "sentence_bert_config.json").write_text("{}")
    (output_dir / "config_sentence_transformers.json").write_text("{}")


def _make_mock_model(fake_save_dir: Path | None = None) -> MagicMock:
    """Create a mock SentenceTransformer with encode() returning unit vectors."""
    mock_peft_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_peft_model

    mock_transformer_module = MagicMock()
    mock_transformer_module.auto_model = mock_peft_model

    mock_model = MagicMock()
    mock_model.__getitem__ = MagicMock(return_value=mock_transformer_module)

    fake_emb = np.ones((3, 1024), dtype=np.float32)
    fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)
    mock_model.encode.return_value = fake_emb

    if fake_save_dir is not None:
        def fake_save(path: str) -> None:
            _create_merged_output(Path(path))
        mock_model.save = fake_save

    return mock_model


class TestValidateMergedOutput:

    def test_rejects_leftover_adapter(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        _create_merged_output(output_dir)
        (output_dir / "0_Transformer" / "adapter_config.json").write_text("{}")
        with pytest.raises(RuntimeError, match="adapter_config.json"):
            validate_merged_output(output_dir)

    def test_rejects_missing_weights(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        output_dir.mkdir(parents=True)
        (output_dir / "0_Transformer").mkdir()
        (output_dir / "modules.json").write_text("[]")
        with pytest.raises(RuntimeError, match="model.safetensors"):
            validate_merged_output(output_dir)

    def test_accepts_clean_output(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        _create_merged_output(output_dir)
        validate_merged_output(output_dir)


class TestMergeLoraAdapters:

    def test_calls_merge_and_unload(self, tmp_path) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = tmp_path / "input"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        with patch("tract.publish.merge.SentenceTransformer", return_value=mock_model):
            merge_lora_adapters(model_dir, output_dir)

        mock_model[0].auto_model.merge_and_unload.assert_called_once()
        assert mock_model.encode.call_count == 2  # pre-merge + post-merge

    def test_output_directory_created(self, tmp_path) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = tmp_path / "input"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        with patch("tract.publish.merge.SentenceTransformer", return_value=mock_model):
            result = merge_lora_adapters(model_dir, output_dir)
        assert result == output_dir
        assert (output_dir / "0_Transformer" / "model.safetensors").exists()

    def test_fails_on_cosine_mismatch(self, tmp_path) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = tmp_path / "input"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        pre_emb = np.ones((3, 1024), dtype=np.float32)
        pre_emb /= np.linalg.norm(pre_emb, axis=1, keepdims=True)
        post_emb = np.random.default_rng(42).standard_normal((3, 1024)).astype(np.float32)
        post_emb /= np.linalg.norm(post_emb, axis=1, keepdims=True)
        mock_model.encode.side_effect = [pre_emb, post_emb]

        with patch("tract.publish.merge.SentenceTransformer", return_value=mock_model):
            with pytest.raises(RuntimeError, match="Merge verification failed"):
                merge_lora_adapters(model_dir, output_dir)
