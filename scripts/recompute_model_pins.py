"""Print the pinned constants for a given HF revision.

Usage:
    python scripts/recompute_model_pins.py <model_commit_sha>
    python scripts/recompute_model_pins.py --dataset <dataset_commit_sha>

The dataset mode exists because tract/config.py ships
TRACT_DATASET_PINNED_REVISION and TRACT_CROSSWALK_DB_SHA256 as UNSET, and
`tract download` refuses to fetch crosswalk.db until they are recorded. The
digest MUST come from a fetch of the published artifact, which is what this
does; the crosswalk.db sitting in results/phase1c/ on a developer's machine
has never been confirmed to equal it. When a local copy is present this prints
its digest too, as a cross-check -- a divergence there is worth understanding
BEFORE a pin is committed, not after.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from tract import config
from tract.io import sha256_file

FILES = ("model.safetensors", "deployment_artifacts.npz",
         "calibration.json", "cre_hierarchy.json")

MODEL_CONSTANTS = {
    "model.safetensors": "TRACT_MODEL_SAFETENSORS_SHA256",
    "deployment_artifacts.npz": "TRACT_DEPLOYMENT_ARTIFACTS_SHA256",
    "calibration.json": "TRACT_CALIBRATION_SHA256",
    "cre_hierarchy.json": "TRACT_HIERARCHY_SHA256",
}

CROSSWALK_DB_NAME = "crosswalk.db"


def main(revision: str) -> None:
    print(f'TRACT_MODEL_PINNED_REVISION = "{revision}"')
    for name in FILES:
        path = hf_hub_download(
            repo_id=config.HF_DEFAULT_REPO_ID, revision=revision, filename=name)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        print(f'{MODEL_CONSTANTS[name]} = "{digest}"')


def main_dataset(revision: str) -> None:
    """Print the two dataset constants, fetched from the Hub at *revision*."""
    path = Path(hf_hub_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        revision=revision,
        filename=CROSSWALK_DB_NAME,
    ))
    published = sha256_file(path)
    print(f'TRACT_DATASET_PINNED_REVISION = "{revision}"')
    print(f'TRACT_CROSSWALK_DB_SHA256 = "{published}"')

    local = config.PHASE1C_CROSSWALK_DB_PATH
    if not local.is_file():
        return
    here = sha256_file(local)
    if here == published:
        print(f"# {local} matches this revision.", file=sys.stderr)
        return
    print(
        f"# WARNING: {local} hashes to {here}, which is NOT what "
        f"{config.HF_DATASET_REPO_ID}@{revision} serves. One of the two is "
        "stale. Do not pin until you know which.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dataset":
        main_dataset(sys.argv[2])
    elif len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        main(sys.argv[1])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
