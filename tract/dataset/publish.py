"""TRACT dataset publication — upload staging directory to HuggingFace Hub."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def publish_dataset(
    repo_id: str,
    staging_dir: Path,
    *,
    dry_run: bool = False,
    skip_upload: bool = False,
) -> None:
    """Upload dataset staging directory to HuggingFace Hub.

    Uses repo_type="dataset" (not "model").
    Token retrieved via ``pass huggingface/token``.

    Args:
        repo_id: HuggingFace repository ID (e.g. "rockCO78/tract-crosswalk-dataset").
        staging_dir: Path to the assembled staging directory.
        dry_run: If True, log but skip the upload.
        skip_upload: If True, log but skip the upload (staging dir left intact).
    """
    if dry_run:
        logger.info("Dry run — skipping upload to %s", repo_id)
        return
    if skip_upload:
        logger.info("Skip upload — staging dir ready at %s", staging_dir)
        return

    from huggingface_hub import HfApi

    token = subprocess.check_output(
        ["pass", "huggingface/token"], text=True,
    ).strip()

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("Dataset uploaded to %s", repo_id)
    finally:
        del token
