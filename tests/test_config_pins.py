import re
import tomllib
from pathlib import Path
from tract import config


def test_pin_and_hashes_well_formed():
    assert re.fullmatch(r"[0-9a-f]{40}", config.TRACT_MODEL_PINNED_REVISION)
    for h in (config.TRACT_MODEL_SAFETENSORS_SHA256, config.TRACT_DEPLOYMENT_ARTIFACTS_SHA256,
              config.TRACT_CALIBRATION_SHA256, config.TRACT_HIERARCHY_SHA256):
        assert re.fullmatch(r"[0-9a-f]{64}", h)


def test_allow_patterns_include_hierarchy_exclude_scripts():
    ap = config.TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "cre_hierarchy.json" in ap
    assert "deployment_artifacts.npz" in ap and "calibration.json" in ap
    assert "model.safetensors" in ap and "modules.json" in ap
    assert "predict.py" not in ap and "train.py" not in ap


def test_exit_codes_distinct():
    codes = {config.EXIT_USER_ERROR, config.EXIT_OFFLINE,
             config.EXIT_INTEGRITY, config.EXIT_MISSING_RUNTIME}
    assert codes == {2, 3, 4, 5}


def test_huggingface_hub_is_a_default_dependency():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert any(d.startswith("huggingface_hub") or d.startswith("huggingface-hub") for d in deps), deps


def test_dataset_pin_is_unset_or_well_formed():
    """UNSET is the honest state until a maintainer records a verified fetch.

    What must never happen is a half-pin: a revision with no digest, or a
    digest that is not one. Either would make the download path believe it is
    verifying something.
    """
    revision = config.TRACT_DATASET_PINNED_REVISION
    digest = config.TRACT_CROSSWALK_DB_SHA256
    unset = config.TRACT_PIN_UNSET

    if revision == unset or digest == unset:
        assert revision == unset and digest == unset, (
            "half-pinned: record both constants or neither")
        return

    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert set(config.HF_DATABASE_FILES) <= set(
        config.TRACT_DATASET_PINNED_FILE_HASHES), (
        "a file is downloaded that no recorded digest covers")
