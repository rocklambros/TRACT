# tests/test_model_resolver.py
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
import tract.model_resolver as mr
from tract import config


ST_MARKERS = ("modules.json", "config_sentence_transformers.json")


def _make_snapshot(d: Path, hashes: dict[str, bytes]):
    d.mkdir(parents=True, exist_ok=True)
    for name in ST_MARKERS:
        (d / name).write_text("{}", encoding="utf-8")
    for name, content in hashes.items():
        (d / name).write_bytes(content)
    return d


def _hashes_for(snapshot: Path) -> dict[str, str]:
    """Recorded hashes covering EVERY file in the snapshot.

    _verify_pinned now walks the snapshot and refuses a file it has no hash for,
    so a fixture that writes six files and records four is asserting the old
    permissive contract. Deriving the map from what was actually written keeps
    the fixture honest as the marker set changes.
    """
    return {
        p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.rglob("*")) if p.is_file()
    }


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
    monkeypatch.setattr(config, "TRACT_MODEL_PINNED_FILE_HASHES", _hashes_for(snap))
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


def test_an_unhashed_file_in_the_snapshot_is_refused(tmp_path, monkeypatch):
    """Fail closed: a downloaded file with no recorded hash stops the load.

    The old check iterated the hash MAP, so a file present in the snapshot but
    absent from the map was never looked at -- which is how nine of the thirteen
    published files, including modules.json and config.json, were consumed
    unverified while the CLI reported a clean integrity check. Walking the
    snapshot instead is only a real control if an unaccounted-for file is an
    error rather than a shrug.
    """
    local = tmp_path / "deployment_model"
    snap = tmp_path / "snap"
    _make_snapshot(snap, _good_files())
    recorded = _hashes_for(snap)
    (snap / "surprise.json").write_text('{"added": "after the pins were taken"}',
                                        encoding="utf-8")
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    monkeypatch.setattr(config, "TRACT_MODEL_PINNED_FILE_HASHES", recorded)
    with patch.object(mr, "snapshot_download", return_value=str(snap)):
        with pytest.raises(mr.ModelIntegrityError, match="no recorded sha256"):
            mr.ensure_deployment_model()


def test_our_own_verify_sentinel_does_not_trip_the_sweep(tmp_path, monkeypatch):
    """The carve-out, pinned. Without it the second run refuses our own model.

    ensure_deployment_model writes .tract-verified-<revision> into the snapshot
    AFTER _verify_pinned returns, so a sweep that fails closed on anything
    unhashed passes once and then refuses forever. Found by running the new gate
    against the real published snapshot; a guard that rejects the artifact it
    protects gets deleted rather than fixed.
    """
    local = tmp_path / "deployment_model"
    snap = tmp_path / "snap"
    _make_snapshot(snap, _good_files())
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    monkeypatch.setattr(config, "TRACT_MODEL_PINNED_FILE_HASHES", _hashes_for(snap))
    with patch.object(mr, "snapshot_download", return_value=str(snap)):
        mr.ensure_deployment_model()          # writes the sentinel
        mr.ensure_deployment_model()          # must not refuse it on the way back
    assert (snap / f"{mr.SENTINEL_PREFIX}{config.TRACT_MODEL_PINNED_REVISION}").exists()
