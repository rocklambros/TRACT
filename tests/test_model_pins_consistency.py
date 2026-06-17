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
