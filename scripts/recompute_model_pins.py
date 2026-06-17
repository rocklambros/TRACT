"""Print the pinned-model constants for a given HF revision.

Usage: python scripts/recompute_model_pins.py <full_commit_sha>
"""
import hashlib
import sys

from huggingface_hub import hf_hub_download

from tract import config

FILES = ("model.safetensors", "deployment_artifacts.npz",
         "calibration.json", "cre_hierarchy.json")


def main(revision: str) -> None:
    print(f'TRACT_MODEL_PINNED_REVISION = "{revision}"')
    for name in FILES:
        path = hf_hub_download(
            repo_id=config.HF_DEFAULT_REPO_ID, revision=revision, filename=name)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        const = {
            "model.safetensors": "TRACT_MODEL_SAFETENSORS_SHA256",
            "deployment_artifacts.npz": "TRACT_DEPLOYMENT_ARTIFACTS_SHA256",
            "calibration.json": "TRACT_CALIBRATION_SHA256",
            "cre_hierarchy.json": "TRACT_HIERARCHY_SHA256",
        }[name]
        print(f'{const} = "{digest}"')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
