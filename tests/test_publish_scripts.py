"""Tests for tract.publish.scripts — standalone predict.py and train.py."""
from __future__ import annotations

from pathlib import Path

import pytest


class TestWritePredictScript:

    def test_creates_file(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        assert (tmp_path / "predict.py").exists()

    def test_contains_imports(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "sentence_transformers" in content
        assert "numpy" in content

    def test_contains_sanitize(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "sanitize_text" in content
        assert "html.unescape" in content

    def test_contains_temperature_scaling(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "temperature" in content.lower() or "softmax" in content.lower()

    def test_contains_ood_flag(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "ood" in content.lower()

    def test_is_valid_python(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        compile(content, "predict.py", "exec")

    def test_no_tract_import(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "from tract" not in content
        assert "import tract" not in content


class TestWriteTrainScript:

    def test_creates_file(self, tmp_path) -> None:
        from tract.publish.scripts import write_train_script
        write_train_script(tmp_path)
        assert (tmp_path / "train.py").exists()

    def test_contains_reproduction_note(self, tmp_path) -> None:
        from tract.publish.scripts import write_train_script
        write_train_script(tmp_path)
        content = (tmp_path / "train.py").read_text()
        assert "TRACT" in content
        assert "clone" in content.lower() or "repository" in content.lower()

    def test_contains_hyperparameters(self, tmp_path) -> None:
        from tract.publish.scripts import write_train_script
        write_train_script(tmp_path)
        content = (tmp_path / "train.py").read_text()
        assert "seed" in content.lower()
        assert "lora" in content.lower() or "LoRA" in content
