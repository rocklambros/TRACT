"""Resolve a deployment-model directory, downloading a pinned HF snapshot lazily.

Resolution order: a complete local dir (dev checkout / prior `tract download`) ->
a pinned HuggingFace snapshot in the HF cache -> an actionable offline error.
Integrity (recorded sha256) is verified once per download, gated by a sentinel.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

from tract import config
from tract.inference import find_st_model_root, resolve_hierarchy_path

logger = logging.getLogger(__name__)


class OfflineModelError(RuntimeError):
    """Model not cached and the Hub is unreachable / offline."""


class ModelIntegrityError(RuntimeError):
    """A downloaded artifact's sha256 did not match the recorded constant."""


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    revision: str
    source: str  # "local" | "download"


def _local_is_complete(model_dir: Path) -> bool:
    if not (model_dir / "deployment_artifacts.npz").exists():
        return False
    if not (model_dir / "calibration.json").exists():
        return False
    try:
        find_st_model_root(model_dir)
        resolve_hierarchy_path(model_dir, source="local")
    except FileNotFoundError:
        return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_pinned(snapshot: Path) -> None:
    for name, expected in config.TRACT_MODEL_PINNED_FILE_HASHES.items():
        actual = _sha256(snapshot / name)
        if actual != expected:
            raise ModelIntegrityError(
                f"Integrity check failed for {name}: expected {expected}, got {actual}. "
                f"Clear the HF cache for {config.HF_DEFAULT_REPO_ID} and re-run."
            )


def ensure_deployment_model() -> ResolvedModel:
    local = config.PHASE1D_DEPLOYMENT_MODEL_DIR
    if _local_is_complete(local):
        logger.info("Using local deployment model at %s", local)
        return ResolvedModel(path=local, revision="local", source="local")

    repo_id = os.environ.get("TRACT_MODEL_REPO_ID", config.HF_DEFAULT_REPO_ID)
    revision = os.environ.get("TRACT_MODEL_REVISION", config.TRACT_MODEL_PINNED_REVISION)
    is_pinned_default = (
        repo_id == config.HF_DEFAULT_REPO_ID
        and revision == config.TRACT_MODEL_PINNED_REVISION
    )
    if not is_pinned_default:
        logger.warning(
            "Using non-default model repo/revision (%s@%s); recorded-hash integrity "
            "is skipped (revision-trust only).", repo_id, revision)

    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(config.TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS),
        ))
    except (LocalEntryNotFoundError, OfflineModeIsEnabled) as exc:
        raise OfflineModelError(
            f"Model not in cache and the Hub is offline. Repo {repo_id}@{revision}. "
            f"Unset HF_HUB_OFFLINE or run `tract download` while online."
        ) from exc

    sentinel = snapshot / f".tract-verified-{revision}"
    if is_pinned_default and not sentinel.exists():
        _verify_pinned(snapshot)
        try:
            sentinel.touch()
        except OSError:
            logger.debug("Could not write verify sentinel in %s", snapshot)
    return ResolvedModel(path=snapshot, revision=revision, source="download")
