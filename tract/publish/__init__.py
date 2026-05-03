"""TRACT HuggingFace publication — model merge, bundling, and upload."""
from __future__ import annotations

import logging
from pathlib import Path

from tract.config import HIERARCHY_BRIDGE_VERSION
from tract.io import load_json

logger = logging.getLogger(__name__)


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
    fold_results: list[dict],
    gpu_hours: float,
    dry_run: bool = False,
    skip_upload: bool = False,
) -> None:
    """Full HuggingFace publication pipeline.

    Steps: gate check → merge → bundle → model card → scripts → security scan → upload.
    """
    import shutil

    from tract.publish.bundle import bundle_inference_data
    from tract.publish.merge import merge_lora_adapters
    from tract.publish.model_card import generate_model_card
    from tract.publish.scripts import write_predict_script, write_train_script
    from tract.publish.security import scan_for_secrets

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

    logger.info("Step 5/7: AIBOM validation...")
    _validate_aibom(staging_dir)

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


AIBOM_REPO = "https://github.com/GenAI-Security-Project/aibom-generator.git"
AIBOM_COMMIT_SHA = "main"


def _validate_aibom(staging_dir: Path) -> None:
    """Run AIBOM validator against the generated model card.

    Non-blocking: logs a warning if the tool is unavailable or broken.
    """
    import shutil
    import subprocess
    import tempfile

    readme = staging_dir / "README.md"
    if not readme.exists():
        logger.warning("No README.md found in staging dir — skipping AIBOM validation")
        return

    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "clone", "--depth=1", "--branch", AIBOM_COMMIT_SHA,
                 AIBOM_REPO, tmp],
                check=True, capture_output=True, timeout=60,
            )
            readme_copy = Path(tmp) / "README_to_validate.md"
            shutil.copy2(readme, readme_copy)
            result = subprocess.run(
                ["python", "-m", "aibom_generator", str(readme_copy)],
                capture_output=True, text=True, timeout=120, cwd=tmp,
            )
            logger.info("AIBOM output:\n%s", result.stdout)
            if result.returncode != 0:
                logger.warning("AIBOM validation returned non-zero: %s", result.stderr)
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning("AIBOM validation skipped — tool unavailable: %s", e)


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
