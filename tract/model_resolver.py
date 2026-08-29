"""Resolve a deployment-model directory, downloading a pinned HF snapshot lazily.

Resolution order: a complete local dir (dev checkout / prior `tract download`) ->
a pinned HuggingFace snapshot in the HF cache -> an actionable offline error.
Integrity (recorded sha256) is verified once per download, gated by a sentinel.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

from tract import config
from tract.inference import find_st_model_root, resolve_hierarchy_path
from tract.io import sha256_file

logger = logging.getLogger(__name__)

# Written into the snapshot by ensure_deployment_model once the recorded hashes
# have verified, so a second run can skip a 1.3 GB rehash. It is our file, not
# the publisher's, and must not be mistaken for unaccounted-for content.
SENTINEL_PREFIX: Final[str] = ".tract-verified-"


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


def _verify_pinned(snapshot: Path) -> None:
    """Every file in the snapshot must match a recorded hash, or nothing loads.

    Fails CLOSED on an unrecorded file rather than ignoring it. The previous
    version iterated the hash map, so a file present in the snapshot but absent
    from the map was simply never looked at -- and for a long time that was nine
    of the thirteen, including modules.json, which names the classes
    SentenceTransformer imports, and config.json, which it builds the model
    from. Verifying the weights while skipping the file that decides which code
    loads them is a check that reports clean for the wrong reason.

    Iterating the SNAPSHOT rather than the map is what makes a newly published
    file fail loudly instead of arriving unverified.
    """
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        name = path.relative_to(snapshot).as_posix()
        # Our own markers are not downloaded content. The verify sentinel
        # (.tract-verified-<revision>, written by the caller AFTER this returns)
        # and huggingface_hub's cache bookkeeping both live inside the snapshot
        # directory, so a sweep that fails closed on anything unhashed refuses
        # the model on its second run -- the first pass writes the sentinel, the
        # next pass finds a file it has no hash for. Found by running this
        # against the real published snapshot rather than reasoning about it.
        if name.startswith(SENTINEL_PREFIX) or name.startswith("."):
            continue
        expected = config.TRACT_MODEL_PINNED_FILE_HASHES.get(name)
        if expected is None:
            raise ModelIntegrityError(
                f"{name} is present in the downloaded snapshot but has no "
                f"recorded sha256, so nothing checked it. Add it to "
                f"TRACT_MODEL_PINNED_FILE_HASHES (scripts/recompute_model_pins.py "
                f"emits the full map) or remove it from "
                f"TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS. Refusing to load a model "
                f"from a snapshot this build cannot account for."
            )
        actual = sha256_file(path)
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
