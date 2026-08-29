# tests/test_model_pins_consistency.py
import hashlib
import pytest
from tract import config

pytestmark = pytest.mark.integration  # network; excluded from default suite

def test_recorded_hashes_match_pinned_revision():
    from huggingface_hub import hf_hub_download
    for name, expected in config.TRACT_MODEL_PINNED_FILE_HASHES.items():
        path = hf_hub_download(
            repo_id=config.HF_DEFAULT_REPO_ID,
            revision=config.TRACT_MODEL_PINNED_REVISION,
            filename=name,
        )
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert actual == expected, f"{name}: {actual} != {expected}"


def test_recorded_dataset_hash_matches_pinned_revision():
    """The dataset pin gets the same drift job the model pin has.

    Skipped while it is UNSET -- `tract download` refuses in that state, so
    nothing unverified ships -- and enforced from the moment it is recorded.
    Without this the dataset pin would rot exactly the way the original
    deferral did.
    """
    from huggingface_hub import hf_hub_download

    if config.TRACT_DATASET_PINNED_REVISION == config.TRACT_PIN_UNSET:
        pytest.skip("dataset pin not recorded yet; tract download refuses")

    for name, expected in config.TRACT_DATASET_PINNED_FILE_HASHES.items():
        path = hf_hub_download(
            repo_id=config.HF_DATASET_REPO_ID,
            repo_type="dataset",
            revision=config.TRACT_DATASET_PINNED_REVISION,
            filename=name,
        )
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert actual == expected, f"{name}: {actual} != {expected}"
