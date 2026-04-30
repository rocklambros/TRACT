"""Phase 1C RunPod retraining: provision H100, run T1+T2, collect results, teardown.

Single-pod execution for deployment model retraining with AL round data.

Usage:
    python -m scripts.phase1c.runpod_retrain --round 2              # full pipeline
    python -m scripts.phase1c.runpod_retrain --round 2 provision    # create pod only
    python -m scripts.phase1c.runpod_retrain --round 2 run          # bootstrap + run on existing pod
    python -m scripts.phase1c.runpod_retrain --round 2 collect      # rsync results back
    python -m scripts.phase1c.runpod_retrain --round 2 teardown     # terminate pod
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Final

from scripts.phase0.runpod_provision import (
    create_pod,
    find_fastest_available,
    terminate_pod,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1c"
POD_STATE_FILE: Final[Path] = PROJECT_ROOT / "scripts" / "phase1c" / ".pod_state_retrain.json"

SSH_KEY: Final[str] = os.path.expanduser("~/.ssh/id_ed25519")
SSH_OPTS: Final[str] = (
    f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    f"-o LogLevel=ERROR -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "
    f"-i {SSH_KEY}"
)

DOCKER_IMAGE: Final[str] = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def _get_credential(name: str) -> str:
    result = subprocess.run(
        ["pass", name], capture_output=True, text=True, check=True, timeout=10,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"pass returned empty value for {name}")
    return value


def _get_pod_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for cred, var in [("wandb/api-key", "WANDB_API_KEY"), ("huggingface/token", "HF_TOKEN")]:
        try:
            env[var] = _get_credential(cred)
        except Exception as e:
            logger.warning("Could not get %s: %s", cred, e)
    return env


def _ssh(
    ip: str, port: int, cmd: str,
    check: bool = True,
    env: dict[str, str] | None = None,
    timeout: int = 7200,
) -> subprocess.CompletedProcess:
    env_lines = ""
    if env:
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env.items()) + "\n"
    script = env_lines + cmd
    ssh_cmd = f"ssh {SSH_OPTS} -p {port} root@{ip} bash -s"
    logger.info("[ssh %s:%d] %s", ip, port, cmd[:200])
    result = subprocess.run(
        ssh_cmd, shell=True, input=script, text=True,
        capture_output=True, timeout=timeout,
    )
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-20:]:
            logger.info("  stdout: %s", line)
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-10:]:
            if "WARNING" not in line and "UserWarning" not in line:
                logger.warning("  stderr: %s", line)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (exit {result.returncode}): {cmd[:200]}\n"
            f"stderr: {result.stderr[-1000:]}"
        )
    return result


def _rsync_to(ip: str, port: int, local_path: str, remote_path: str) -> None:
    cmd = (
        f"rsync -rltz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' "
        f"--exclude='.mypy_cache' --exclude='models' "
        f"--exclude='wandb' --exclude='.wandb' "
        f"--exclude='results/phase0' --exclude='results/phase1b' "
        f"-e 'ssh {SSH_OPTS} -p {port}' {local_path} root@{ip}:{remote_path}"
    )
    logger.info("[rsync to] %s:%d %s", ip, port, remote_path)
    subprocess.run(cmd, shell=True, check=True, timeout=600)


def _rsync_from(ip: str, port: int, remote_path: str, local_path: str) -> None:
    cmd = f"rsync -rltz -e 'ssh {SSH_OPTS} -p {port}' root@{ip}:{remote_path} {local_path}"
    logger.info("[rsync from] %s:%d %s", ip, port, remote_path)
    subprocess.run(cmd, shell=True, check=True, timeout=600)


def _save_pod_state(pod: dict) -> None:
    POD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    POD_STATE_FILE.write_text(json.dumps(pod, indent=2, sort_keys=True))
    logger.info("Pod state saved to %s", POD_STATE_FILE)


def _load_pod_state() -> dict:
    if not POD_STATE_FILE.exists():
        raise FileNotFoundError(f"No pod state file at {POD_STATE_FILE} — run 'provision' first")
    return json.loads(POD_STATE_FILE.read_text())


def provision() -> dict:
    logger.info("Finding fastest available GPU (>= 48GB VRAM)...")
    gpu_type = find_fastest_available(min_vram_gb=48)
    logger.info("Selected GPU: %s", gpu_type)

    pod = create_pod(
        gpu_type, name="tract-p1c-retrain",
        image=DOCKER_IMAGE, volume_gb=50, container_disk_gb=20,
    )

    _save_pod_state(pod)
    logger.info("Pod provisioned and SSH-ready: %s:%d", pod["ip"], pod["port"])
    return pod


def _bootstrap(pod: dict) -> None:
    ip, port = pod["ip"], pod["port"]
    logger.info("Bootstrapping pod %s:%d...", ip, port)

    _ssh(ip, port, "apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1", check=False)

    _rsync_to(ip, port, f"{PROJECT_ROOT}/", "/workspace/tract/")

    _ssh(ip, port, (
        "cd /workspace/tract && "
        "pip install --quiet -e '.[phase0]' && "
        "pip install --quiet sentence-transformers==5.3.0 peft datasets accelerate"
    ))

    _ssh(ip, port, (
        "python --version && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader "
        "&& python -c 'import torch; import sentence_transformers; import peft; "
        "print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        "st={sentence_transformers.__version__} peft={peft.__version__}\")'"
    ), check=False)

    logger.info("Bootstrap complete.")


def run_retrain(round_num: int) -> None:
    pod = _load_pod_state()
    ip, port = pod["ip"], pod["port"]

    _bootstrap(pod)

    env = _get_pod_env()

    logger.info("=" * 60)
    logger.info("RUNNING T1 (round %d) on RunPod", round_num)
    logger.info("=" * 60)

    t1_start = time.time()
    _ssh(ip, port, (
        f"cd /workspace/tract && "
        f"python -m scripts.phase1c.t1_calibrate_and_train --round {round_num} "
        f"2>&1 | tee results/phase1c/t1_round{round_num}_log.txt"
    ), env=env, timeout=7200)
    t1_elapsed = time.time() - t1_start
    logger.info("T1 complete in %.1fm", t1_elapsed / 60)

    logger.info("=" * 60)
    logger.info("RUNNING T2 (round %d) on RunPod", round_num)
    logger.info("=" * 60)

    t2_start = time.time()
    _ssh(ip, port, (
        f"cd /workspace/tract && "
        f"python -m scripts.phase1c.t2_inference_and_calibrate "
        f"2>&1 | tee results/phase1c/t2_round{round_num}_log.txt"
    ), env=env, timeout=3600)
    t2_elapsed = time.time() - t2_start
    logger.info("T2 complete in %.1fm", t2_elapsed / 60)

    logger.info("=" * 60)
    logger.info("TOTAL: T1+T2 in %.1fm (T1: %.1fm, T2: %.1fm)",
                (t1_elapsed + t2_elapsed) / 60, t1_elapsed / 60, t2_elapsed / 60)
    logger.info("=" * 60)


def collect(round_num: int) -> None:
    pod = _load_pod_state()
    ip, port = pod["ip"], pod["port"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for subdir in ["calibration", "deployment_model", "holdout", "similarities", f"round_1"]:
        local = RESULTS_DIR / subdir
        local.mkdir(parents=True, exist_ok=True)
        try:
            _rsync_from(ip, port, f"/workspace/tract/results/phase1c/{subdir}/", f"{local}/")
            logger.info("Collected %s", subdir)
        except Exception as e:
            logger.warning("Failed to collect %s: %s", subdir, e)

    for log_name in [f"t1_round{round_num}_log.txt", f"t2_round{round_num}_log.txt"]:
        try:
            _rsync_from(ip, port, f"/workspace/tract/results/phase1c/{log_name}", f"{RESULTS_DIR}/{log_name}")
        except Exception as e:
            logger.warning("Failed to collect %s: %s", log_name, e)

    logger.info("Results collected to %s", RESULTS_DIR)


def teardown() -> None:
    pod = _load_pod_state()
    logger.info("Terminating pod %s...", pod["pod_id"])
    terminate_pod(pod["pod_id"])
    if POD_STATE_FILE.exists():
        POD_STATE_FILE.unlink()
    logger.info("Pod terminated.")


def full_pipeline(round_num: int) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1C RETRAIN ROUND %d (RunPod)", round_num)
    logger.info("=" * 60)
    start = time.time()

    provision()
    run_retrain(round_num)
    collect(round_num)
    teardown()

    elapsed = time.time() - start
    logger.info("Total pipeline time: %.1fm", elapsed / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1C RunPod retraining")
    parser.add_argument("--round", type=int, required=True, help="Round number (2+)")
    parser.add_argument("action", nargs="?", default="full",
                        choices=["full", "provision", "run", "collect", "teardown"])
    args = parser.parse_args()

    if args.action == "full":
        full_pipeline(args.round)
    elif args.action == "provision":
        provision()
    elif args.action == "run":
        run_retrain(args.round)
    elif args.action == "collect":
        collect(args.round)
    elif args.action == "teardown":
        teardown()


if __name__ == "__main__":
    main()
