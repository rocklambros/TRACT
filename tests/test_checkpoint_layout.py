"""Tests for the saved-checkpoint layout in tract/training/loop.py.

With a PEFT adapter attached, SentenceTransformer.save(path) writes the whole
tree to path/model/ rather than to path. Reloading path then fails with
"Unrecognized model ... should have a model_type key", which is what ended the
canary fold after twenty epochs of training had already completed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from tract.training.loop import _flatten_saved_model_dir  # noqa: E402


def _tree(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text("{}", encoding="utf-8")


def test_a_nested_tree_is_moved_up(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _tree(model_dir / "model", ["modules.json", "adapter_config.json", "vocab.txt"])
    (model_dir / "model" / "1_Pooling").mkdir()

    _flatten_saved_model_dir(model_dir)

    assert (model_dir / "modules.json").is_file()
    assert (model_dir / "adapter_config.json").is_file()
    assert (model_dir / "1_Pooling").is_dir()
    assert not (model_dir / "model").exists()


def test_a_correct_layout_is_untouched(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _tree(model_dir, ["modules.json", "config.json"])

    _flatten_saved_model_dir(model_dir)

    assert (model_dir / "modules.json").is_file()
    assert (model_dir / "config.json").is_file()


def test_an_unrelated_subdirectory_named_model_is_left_alone(
    tmp_path: Path
) -> None:
    """Only a subdirectory that is itself an ST root counts as nesting."""
    model_dir = tmp_path / "model"
    _tree(model_dir, ["modules.json"])
    (model_dir / "model").mkdir()
    (model_dir / "model" / "weights.bin").write_text("x", encoding="utf-8")

    _flatten_saved_model_dir(model_dir)

    assert (model_dir / "model" / "weights.bin").is_file()


def test_an_ambiguous_layout_raises_rather_than_guessing(tmp_path: Path) -> None:
    """Both levels holding a root means the checkpoint is not understood."""
    model_dir = tmp_path / "model"
    _tree(model_dir, ["modules.json", "vocab.txt"])
    _tree(model_dir / "model", ["modules.json", "vocab.txt"])

    # Root already present, so this is a no-op rather than a destructive move.
    _flatten_saved_model_dir(model_dir)
    assert (model_dir / "model" / "modules.json").is_file()


def test_a_collision_during_flattening_raises(tmp_path: Path) -> None:
    """Overwriting on the way up would silently destroy an artifact."""
    model_dir = tmp_path / "model"
    _tree(model_dir, ["vocab.txt"])          # no modules.json here
    _tree(model_dir / "model", ["modules.json", "vocab.txt"])

    with pytest.raises(RuntimeError, match="already exists"):
        _flatten_saved_model_dir(model_dir)


def test_a_missing_directory_is_a_no_op(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _flatten_saved_model_dir(model_dir)
    assert list(model_dir.iterdir()) == []
