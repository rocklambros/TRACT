"""TRACT HuggingFace publication — model merge, bundling, and upload."""
from __future__ import annotations

import logging
import re
from typing import Any
from pathlib import Path

from tract.config import HIERARCHY_BRIDGE_VERSION
from tract.io import load_json

logger = logging.getLogger(__name__)

AIBOM_REPO = "https://github.com/GenAI-Security-Project/aibom-generator.git"
# Empty on purpose, and checked before anything runs. This constant read "main"
# and was handed to `git clone --branch`, so the revision that executed as the
# publishing user was whatever that external branch happened to hold at publish
# time — on the one host that holds the pass store, where huggingface/token has
# write scope. A pin is a 40-hex commit somebody has actually read, and nobody
# here has read one, so there is no default. An operator who has reviewed a
# revision passes it as --aibom-commit.
AIBOM_COMMIT_SHA = ""
AIBOM_SHA_PATTERN = re.compile(r"\A[0-9a-f]{40}\Z")
AIBOM_CLONE_TIMEOUT_S = 120
AIBOM_RUN_TIMEOUT_S = 120
# The validator is third-party code executed as the publishing user, so it gets
# what an interpreter needs to start and nothing else. The publishing shell
# carries HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY, and none of those are
# the validator's business. Note this strips variables only: the child still
# reads the same home directory, so ~/.cache/huggingface/token, ~/.ssh and the
# pass store on disk are untouched by it. This is not a sandbox.
AIBOM_ENV_ALLOWLIST = (
    "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT",
    # The validator fetches its required-field list over HTTPS. Without these a
    # host behind a proxy or a private CA fails the step on a network error
    # that reads like a bug in the pin.
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
)


def check_publication_gate(
    bridge_report_path: Path,
    hierarchy_path: Path | None = None,
) -> None:
    """Verify bridge analysis is complete before publication.

    Raises ValueError if:
    - bridge_report.json does not exist
    - Any candidate has status 'pending'
    - Accepted bridges exist but hierarchy not updated
    """
    if not bridge_report_path.exists():
        raise ValueError(
            f"bridge_report.json not found at {bridge_report_path}. "
            "Run 'tract bridge' and 'tract bridge --commit' first."
        )

    report = load_json(bridge_report_path)
    pending = [
        c for c in report.get("candidates", [])
        if c.get("status") == "pending"
    ]
    if pending:
        raise ValueError(
            f"{len(pending)} candidates still have 'pending' status in "
            f"{bridge_report_path}. Review all candidates before publishing."
        )

    accepted = [
        c for c in report.get("candidates", [])
        if c.get("status") == "accepted"
    ]
    if accepted and hierarchy_path:
        hier = load_json(hierarchy_path)
        if hier.get("version") != HIERARCHY_BRIDGE_VERSION:
            raise ValueError(
                f"Bridge report has {len(accepted)} accepted bridges but "
                f"hierarchy version is '{hier.get('version')}', not "
                f"'{HIERARCHY_BRIDGE_VERSION}'. "
                "Run 'tract bridge --commit' to update the hierarchy."
            )
        for bridge in accepted:
            ai_id = bridge["ai_hub_id"]
            trad_id = bridge["trad_hub_id"]
            ai_related = hier.get("hubs", {}).get(ai_id, {}).get("related_hub_ids", [])
            if trad_id not in ai_related:
                raise ValueError(
                    f"Accepted bridge {ai_id}↔{trad_id} not found in "
                    f"hierarchy related_hub_ids. Run 'tract bridge --commit'."
                )


def publish_to_huggingface(
    *,
    repo_id: str,
    staging_dir: Path,
    model_dir: Path,
    artifacts_path: Path,
    hierarchy_path: Path,
    hub_descriptions_path: Path,
    calibration_path: Path,
    ece_gate_path: Path,
    bridge_report_path: Path,
    fold_results: list[dict[str, Any]],
    gpu_hours: float,
    dry_run: bool = False,
    skip_upload: bool = False,
    validate_aibom: bool = False,
    aibom_commit: str = AIBOM_COMMIT_SHA,
) -> None:
    """Full HuggingFace publication pipeline.

    Steps: gate check → merge → bundle → model card → scripts → security scan → upload.

    ``validate_aibom`` opts into cloning and executing the third-party AIBOM
    validator at ``aibom_commit`` on this host. It defaults off because the step
    used to run above the ``dry_run`` return below, so `publish-hf --dry-run`
    executed whatever that external repository held at the time, on the machine
    that holds the pass store.
    """
    import shutil

    from tract.licensing import copy_licensing_files
    from tract.publish.bundle import bundle_inference_data
    from tract.publish.merge import merge_lora_adapters
    from tract.publish.model_card import generate_model_card
    from tract.publish.scripts import write_predict_script, write_train_script
    from tract.publish.security import scan_for_secrets

    # Step 1 merges the LoRA adapters, which loads the weights. A missing or
    # mistyped --aibom-commit discovered at step 5 would cost all of that for
    # nothing, so the pin is checked before any work happens.
    if validate_aibom:
        _require_aibom_pin(aibom_commit)

    check_publication_gate(bridge_report_path, hierarchy_path=hierarchy_path)
    logger.info("Publication gate passed")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    logger.info("Step 1/7: Merging LoRA adapters...")
    merge_lora_adapters(model_dir, staging_dir)

    logger.info("Step 2/7: Bundling inference data...")
    calibration = load_json(calibration_path)
    ece_data = load_json(ece_gate_path)
    bridge_summary = load_json(bridge_report_path)

    bundle_inference_data(
        staging_dir,
        hub_descriptions=hub_descriptions_path,
        hierarchy=hierarchy_path,
        calibration=calibration_path,
        artifacts=artifacts_path,
        bridge_report=bridge_report_path,
    )

    logger.info("Step 3/7: Generating model card...")
    generate_model_card(
        staging_dir,
        fold_results=fold_results,
        calibration=calibration,
        ece_data=ece_data,
        bridge_summary=bridge_summary,
        gpu_hours=gpu_hours,
    )

    logger.info("Step 4/7: Writing standalone scripts...")
    write_predict_script(staging_dir)
    write_train_script(staging_dir)

    # The licence record travels with the weights. The card's license_link
    # points at NOTICE inside this directory, so a consumer who downloads the
    # model and never visits the source repository can still read the terms of
    # every framework the weights were trained on.
    copy_licensing_files(staging_dir)

    if validate_aibom:
        logger.info("Step 5/7: AIBOM validation...")
        _validate_aibom(staging_dir, aibom_commit)
    else:
        logger.info(
            "Step 5/7: AIBOM validation skipped — not requested. Pass "
            "--validate-aibom with a reviewed --aibom-commit to run it."
        )

    logger.info("Step 6/7: Running security scan...")
    findings = scan_for_secrets(staging_dir)
    if findings:
        for f in findings:
            logger.warning("ALERT: %s:%d — %s", f.file_path, f.line_number, f.pattern_name)
        raise ValueError(
            f"Security scan found {len(findings)} issues. Fix and re-run."
        )
    logger.info("Security scan passed")

    if dry_run:
        logger.info("Dry run complete. Staging directory: %s", staging_dir)
        logger.info("Run without --dry-run to upload.")
        return

    if skip_upload:
        logger.info("Build complete. Staging directory: %s", staging_dir)
        logger.info("Run without --skip-upload to upload.")
        return

    logger.info("Step 7/7: Uploading to HuggingFace...")
    _upload_to_hub(repo_id, staging_dir)
    logger.info("Published to https://huggingface.co/%s", repo_id)


def _require_aibom_pin(commit_sha: str) -> None:
    """Reject anything but a full 40-hex commit for the AIBOM validator.

    Split out of _validate_aibom so publish_to_huggingface can refuse a missing
    pin before step 1 merges the adapters, instead of after.
    """
    if not AIBOM_SHA_PATTERN.match(commit_sha):
        raise ValueError(
            f"AIBOM validation needs a full 40-character commit SHA, got "
            f"{commit_sha!r}. Review a revision of {AIBOM_REPO} and pass it as "
            "--aibom-commit: a branch name resolves to whatever that repository "
            "holds at publish time, and the clone is executed on this host."
        )


def _stderr_text(stderr: object) -> str:
    """Decode a captured stderr, which is bytes or str depending on text=.

    CalledProcessError stringifies to the exit status alone, so without this a
    failed checkout reports "returned non-zero exit status 128" and never says
    that the SHA could not be found.
    """
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace").strip()
    if isinstance(stderr, str):
        return stderr.strip()
    return ""


def _validate_aibom(staging_dir: Path, commit_sha: str) -> None:
    """Run the third-party AIBOM validator against the generated model card.

    Clones AIBOM_REPO at ``commit_sha`` and executes it, so calling this is
    opting into running that exact revision as the publishing user.

    Raises ValueError if the pin is not a full commit SHA, if the checkout does
    not land on it, if the clone cannot be fetched, or if the validator reports
    failure. The previous version downgraded every one of those to a warning,
    which is the worst of both: it could never block a bad card, and it ran the
    foreign code anyway.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    _require_aibom_pin(commit_sha)

    readme = staging_dir / "README.md"
    if not readme.is_file():
        raise ValueError(
            f"AIBOM validation was requested but no README.md exists in "
            f"{staging_dir} — there is no model card to validate."
        )

    with tempfile.TemporaryDirectory() as tmp:
        checkout = Path(tmp) / "aibom-generator"
        workdir = Path(tmp) / "work"
        workdir.mkdir()

        try:
            subprocess.run(
                ["git", "clone", "--no-checkout", AIBOM_REPO, str(checkout)],
                check=True, capture_output=True, timeout=AIBOM_CLONE_TIMEOUT_S,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "checkout", "--detach", commit_sha],
                check=True, capture_output=True, timeout=AIBOM_CLONE_TIMEOUT_S,
            )
            head = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
                timeout=AIBOM_CLONE_TIMEOUT_S,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            detail = _stderr_text(e.stderr)
            raise ValueError(
                f"Could not check out {AIBOM_REPO} at {commit_sha}: {e}"
                + (f": {detail}" if detail else "")
            ) from e
        except OSError as e:
            # FileNotFoundError when git is not installed, but a PermissionError
            # or a fork failure lands here too, and none of them should leave
            # this function as anything other than ValueError.
            raise ValueError(
                f"Could not run git to check out {AIBOM_REPO} at {commit_sha}: {e}"
            ) from e

        # A rev can be resolved through a ref, and refs move. The run is only
        # pinned if HEAD is byte-for-byte the commit that was reviewed.
        if head != commit_sha:
            raise ValueError(
                f"AIBOM checkout landed on {head}, not the pinned {commit_sha}. "
                "Refusing to execute an unreviewed revision."
            )

        # sys.executable rather than "python", because PATH decides what
        # "python" means and PATH is not part of anyone's review. The separate
        # workdir is housekeeping, not containment: PYTHONPATH puts the clone on
        # sys.path ahead of site-packages exactly as cwd did, so the pinned
        # revision still runs with the operator's ambient access either way.
        readme_copy = workdir / "README_to_validate.md"
        shutil.copy2(readme, readme_copy)

        env = {k: v for k, v in os.environ.items() if k in AIBOM_ENV_ALLOWLIST}
        env["PYTHONPATH"] = str(checkout)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "aibom_generator", str(readme_copy)],
                capture_output=True, text=True, timeout=AIBOM_RUN_TIMEOUT_S,
                cwd=workdir, env=env, check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            # The timeout bounds this call, not the foreign code: anything the
            # validator forked into the background outlives both the timeout and
            # the TemporaryDirectory cleanup.
            raise ValueError(
                f"AIBOM validator at {commit_sha} could not be run: {e}"
            ) from e

        logger.info("AIBOM output:\n%s", result.stdout)
        if result.returncode != 0:
            raise ValueError(
                f"AIBOM validation failed (exit {result.returncode}) against "
                f"{AIBOM_REPO}@{commit_sha}: {result.stderr}"
            )


def _upload_to_hub(repo_id: str, staging_dir: Path) -> None:
    """Upload staging directory to HuggingFace Hub."""
    import subprocess

    from huggingface_hub import HfApi

    token = subprocess.check_output(
        ["pass", "huggingface/token"], text=True
    ).strip()

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("Uploaded to %s", repo_id)
    finally:
        del token
