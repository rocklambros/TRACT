"""Tests for tract.io — atomic JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path

from tract.io import atomic_write_json, load_json


class TestAtomicWriteJson:
    """Tests for atomic_write_json."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        data = {"key": "value", "count": 42}
        atomic_write_json(data, target)

        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == data

    def test_sorted_keys(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        data = {"zebra": 1, "alpha": 2, "mid": 3}
        atomic_write_json(data, target)

        raw = target.read_text(encoding="utf-8")
        keys_in_order = list(json.loads(raw).keys())
        assert keys_in_order == ["alpha", "mid", "zebra"]

    def test_trailing_newline(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        atomic_write_json({"a": 1}, target)

        raw = target.read_bytes()
        assert raw.endswith(b"\n")
        # Ensure it's exactly one trailing newline (not two)
        assert not raw.endswith(b"\n\n")

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        data = {"text": "café — naïve"}
        atomic_write_json(data, target)

        raw = target.read_text(encoding="utf-8")
        assert "café" in raw
        assert "\\u" not in raw  # ensure_ascii=False

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        atomic_write_json({"version": 1}, target)
        atomic_write_json({"version": 2}, target)

        loaded = load_json(target)
        assert loaded["version"] == 2

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "dir" / "test.json"
        atomic_write_json({"nested": True}, target)

        assert target.exists()
        loaded = load_json(target)
        assert loaded["nested"] is True

    def test_indent_two_spaces(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        atomic_write_json({"a": {"b": 1}}, target)

        raw = target.read_text(encoding="utf-8")
        # The nested key should be indented by 4 spaces (2 per level)
        assert '    "b": 1' in raw


class TestLoadJson:
    """Tests for load_json."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        target.write_text('{"hello": "world"}\n', encoding="utf-8")

        data = load_json(target)
        assert data == {"hello": "world"}

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        import pytest

        target = tmp_path / "bad.json"
        target.write_text("not json!", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_json(target)
