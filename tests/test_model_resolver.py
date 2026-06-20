# tests/test_model_resolver.py
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
import tract.model_resolver as mr
from tract import config


def _make_snapshot(d: Path, hashes: dict[str, bytes]):
    d.mkdir(parents=True, exist_ok=True)
    for name in ("modules.json", "config_sentence_transformers.json"):
        (d / name).write_text("{}", encoding="utf-8")
    for name, content in hashes.items():
        (d / name).write_bytes(content)
    return d


def _good_files():
    # Build files whose sha256 match the recorded constants by monkeypatching the
    # expected-hash map to the bytes we write.
    return {
        "model.safetensors": b"WEIGHTS",
        "deployment_artifacts.npz": b"NPZ",
        "calibration.json": b"{}",
        "cre_hierarchy.json": b"{}",
    }


def test_local_complete_returns_local_without_download(tmp_path, monkeypatch):
    local = tmp_path / "deployment_model"
    _make_snapshot(local / "model", {})
    (local / "deployment_artifacts.npz").write_bytes(b"x")
    (local / "calibration.json").write_text("{}", encoding="utf-8")
    (local / "cre_hierarchy.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    with patch.object(mr, "snapshot_download") as sd:
        result = mr.ensure_deployment_model()
        sd.assert_not_called()
    assert result.path == local and result.source == "local"


def test_local_absent_triggers_pinned_download(tmp_path, monkeypatch):
    local = tmp_path / "deployment_model"  # does not exist
    snap = tmp_path / "snap"
    files = _good_files()
    _make_snapshot(snap, files)
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    monkeypatch.setattr(config, "TRACT_MODEL_PINNED_FILE_HASHES",
                        {n: hashlib.sha256(c).hexdigest() for n, c in files.items()})
    with patch.object(mr, "snapshot_download", return_value=str(snap)) as sd:
        result = mr.ensure_deployment_model()
        _, kwargs = sd.call_args
        assert kwargs["revision"] == config.TRACT_MODEL_PINNED_REVISION
        assert "cre_hierarchy.json" in kwargs["allow_patterns"]
    # Bug in brief (line 68): `snap / ".tract-verified-" + revision` is a TypeError
    # because Path.__truediv__ returns Path, and Path + str is unsupported.
    # Correct form uses an f-string.
    assert result.source == "download" and (snap / f".tract-verified-{config.TRACT_MODEL_PINNED_REVISION}").exists()


def test_download_integrity_mismatch_raises(tmp_path, monkeypatch):
    local = tmp_path / "deployment_model"
    snap = tmp_path / "snap"
    _make_snapshot(snap, _good_files())
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    monkeypatch.setattr(config, "TRACT_MODEL_PINNED_FILE_HASHES",
                        {"model.safetensors": "0" * 64})  # wrong
    with patch.object(mr, "snapshot_download", return_value=str(snap)):
        with pytest.raises(mr.ModelIntegrityError):
            mr.ensure_deployment_model()


def test_offline_cold_cache_raises_offline(tmp_path, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", tmp_path / "nope")
    with patch.object(mr, "snapshot_download", side_effect=LocalEntryNotFoundError("x")):
        with pytest.raises(mr.OfflineModelError) as e:
            mr.ensure_deployment_model()
    assert "HF_HUB_OFFLINE" in str(e.value) and config.HF_DEFAULT_REPO_ID in str(e.value)
