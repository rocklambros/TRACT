"""Phase 1B RunPod parallel fold executor.

Provisions 5 H100 pods (one per LOFO fold), bootstraps them in parallel,
runs training + evaluation simultaneously, collects results, tears down.

Usage:
    python -m scripts.phase1b.runpod_parallel                    # full pipeline
    python -m scripts.phase1b.runpod_parallel provision          # create pods only
    python -m scripts.phase1b.runpod_parallel run                # bootstrap + run on existing pods
    python -m scripts.phase1b.runpod_parallel collect            # rsync results back
    python -m scripts.phase1b.runpod_parallel teardown           # terminate all pods
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

from scripts.phase0.runpod_provision import (
    create_pods_parallel,
    find_fastest_available,
    get_running_pods,
    terminate_all,
    terminate_pod,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1b"
POD_STATE_FILE: Final[Path] = PROJECT_ROOT / "scripts" / "phase1b" / ".pod_state.json"

SSH_KEY: Final[str] = os.path.expanduser("~/.ssh/id_ed25519")
SSH_OPTS: Final[str] = (
    f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    f"-o LogLevel=ERROR -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "
    f"-i {SSH_KEY}"
)

DOCKER_IMAGE: Final[str] = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

FOLD_FRAMEWORKS: Final[list[str]] = [
    "MITRE ATLAS",
    "NIST AI 100-2",
    "OWASP AI Exchange",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
]

POD_CONFIGS: Final[list[dict[str, str]]] = [
    {"name": f"tract-p1b-fold{i}", "role": fw}
    for i, fw in enumerate(FOLD_FRAMEWORKS)
]


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
    try:
        env["WANDB_API_KEY"] = _get_credential("wandb/api-key")
    except Exception as e:
        logger.warning("Could not get WandB API key: %s", e)
    try:
        env["HF_TOKEN"] = _get_credential("huggingface/token")
    except Exception as e:
        logger.warning("Could not get HuggingFace token: %s", e)
    return env


def _ssh(
    ip: str, port: int, cmd: str,
    check: bool = True,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess:
    env_lines = ""
    if env:
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env.items()) + "\n"
    script = env_lines + cmd
    ssh_cmd = f"ssh {SSH_OPTS} -p {port} root@{ip} bash -s"
    logger.info("[ssh %s:%d] %s", ip, port, cmd[:120])
    result = subprocess.run(
        ssh_cmd, shell=True, input=script, text=True,
        capture_output=True, timeout=timeout,
    )
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            logger.info("  stdout: %s", line)
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-5:]:
            if "WARNING" not in line and "UserWarning" not in line:
                logger.warning("  stderr: %s", line)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (exit {result.returncode}): {cmd[:120]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
    return result


def _rsync_to(ip: str, port: int, local_path: str, remote_path: str) -> None:
    cmd = (
        f"rsync -rltz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' "
        f"--exclude='results' --exclude='.mypy_cache' --exclude='models' "
        f"--exclude='wandb' --exclude='.wandb' "
        f"-e 'ssh {SSH_OPTS} -p {port}' {local_path} root@{ip}:{remote_path}"
    )
    logger.info("[rsync to] %s:%d %s", ip, port, remote_path)
    subprocess.run(cmd, shell=True, check=True, timeout=300)


def _rsync_from(ip: str, port: int, remote_path: str, local_path: str) -> None:
    cmd = f"rsync -rltz -e 'ssh {SSH_OPTS} -p {port}' root@{ip}:{remote_path} {local_path}"
    logger.info("[rsync from] %s:%d %s", ip, port, remote_path)
    subprocess.run(cmd, shell=True, check=True, timeout=300)


def _save_pod_state(pods: list[dict]) -> None:
    POD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    POD_STATE_FILE.write_text(json.dumps(pods, indent=2, sort_keys=True))
    logger.info("Pod state saved to %s", POD_STATE_FILE)


def _load_pod_state() -> list[dict]:
    if not POD_STATE_FILE.exists():
        raise FileNotFoundError(f"No pod state file at {POD_STATE_FILE} — run 'provision' first")
    return json.loads(POD_STATE_FILE.read_text())


def provision() -> list[dict]:
    logger.info("Finding fastest available GPU (>= 48GB VRAM)...")
    gpu_type = find_fastest_available(min_vram_gb=48)
    logger.info("Selected GPU: %s", gpu_type)

    pods = create_pods_parallel(
        POD_CONFIGS, gpu_type, image=DOCKER_IMAGE,
        volume_gb=50, container_disk_gb=20,
    )

    _save_pod_state(pods)
    logger.info("All %d pods provisioned and SSH-ready.", len(pods))
    return pods


def _bootstrap_pod(pod: dict) -> None:
    ip, port, role = pod["ip"], pod["port"], pod["role"]
    logger.info("Bootstrapping pod for fold '%s' (%s:%d)...", role, ip, port)

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

    logger.info("Bootstrap complete for fold '%s'", role)


def _run_fold_on_pod(pod: dict, config_name: str) -> dict:
    ip, port = pod["ip"], pod["port"]
    framework = pod["role"]

    # A module invocation, not an inline program. run_fold.py persists the full
    # per-fold record, including the per-item hit@1 indicators the aggregate CI
    # needs; the string this replaced kept only the summary metrics, which can
    # be averaged into a macro figure but never a micro one.
    fold_cmd = (
        f"cd /workspace/tract && python -m scripts.phase1b.run_fold "
        f"--framework {shlex.quote(framework)} "
        f"--config-name {shlex.quote(config_name)} "
        f"--zero-shot"
    )

    logger.info("[%s] Starting fold training...", framework)
    start = time.time()
    env = _get_pod_env()
    try:
        result = _ssh(ip, port, fold_cmd, env=env, timeout=3600)
        elapsed = time.time() - start
        logger.info("[%s] COMPLETE in %.1fm", framework, elapsed / 60)
        return {"fold": framework, "status": "ok", "elapsed_s": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] FAILED after %.1fm: %s", framework, elapsed / 60, e)
        return {"fold": framework, "status": "failed", "error": str(e), "elapsed_s": elapsed}


def run_folds(config_name: str = "phase1b_primary") -> None:
    pods = _load_pod_state()

    logger.info("Bootstrapping %d pods in parallel...", len(pods))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pods)) as ex:
        list(ex.map(_bootstrap_pod, pods))
    logger.info("All pods bootstrapped.")

    logger.info("=" * 60)
    logger.info("RUNNING 5 FOLDS IN PARALLEL")
    logger.info("=" * 60)

    start = time.time()
    fold_results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pods)) as ex:
        futures = {
            ex.submit(_run_fold_on_pod, pod, config_name): pod["role"]
            for pod in pods
        }
        for f in concurrent.futures.as_completed(futures):
            role = futures[f]
            result = f.result()
            fold_results.append(result)
            logger.info("  [%s] %s (%.1fm)", role, result["status"], result["elapsed_s"] / 60)

    elapsed = time.time() - start
    ok = sum(1 for r in fold_results if r["status"] == "ok")
    failed = sum(1 for r in fold_results if r["status"] == "failed")
    logger.info("=" * 60)
    logger.info("ALL FOLDS COMPLETE: %d OK, %d FAILED in %.1fm", ok, failed, elapsed / 60)
    for r in fold_results:
        if r["status"] == "failed":
            logger.error("  FAILED: %s — %s", r["fold"], r.get("error", "")[:200])
    logger.info("=" * 60)


def collect(config_name: str = "phase1b_primary") -> list[str]:
    """Retrieve every fold's results. Returns the roles that failed to collect.

    Collection failure is not cosmetic: the fold ran, the GPU hour was paid for,
    and the per-item indicators live only on the pod. teardown() must not
    terminate a pod whose results are still on it.
    """
    pods = _load_pod_state()
    local_results = RESULTS_DIR / config_name
    local_results.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for pod in pods:
        logger.info("Collecting fold '%s' from %s:%d...", pod["role"], pod["ip"], pod["port"])
        try:
            _rsync_from(
                pod["ip"], pod["port"],
                f"/workspace/tract/results/phase1b/{config_name}/",
                f"{local_results}/",
            )
        except Exception as e:
            logger.error("Collection from %s FAILED: %s", pod["role"], e)
            failed.append(pod["role"])

    logger.info("Results collected to %s", local_results)
    if failed:
        logger.error("Collection failed for %d fold(s): %s", len(failed), failed)
    return failed


def aggregate(config_name: str = "phase1b_primary") -> dict:
    """Micro-average the collected folds and write the experiment record.

    The RunPod path had no aggregation step at all. Averaging the five fold
    summaries by hand gives a macro average, which weights every fold equally
    regardless of how many eval items it holds. This pools the per-item
    indicators instead, and reports the paired delta against the zero-shot
    baseline measured on the same items.
    """
    from tract.io import atomic_write_json
    from tract.training.orchestrate import (
        aggregate_fold_results,
        gate_decision,
        load_fold_results,
    )

    local_results = RESULTS_DIR / config_name
    fold_results = load_fold_results(local_results)
    logger.info("Aggregating %d folds from %s", len(fold_results), local_results)

    record = {
        "config_name": config_name,
        "aggregate_hit1": aggregate_fold_results(fold_results),
        "gate": gate_decision(fold_results),
        "per_fold": {
            r["held_out_framework"]: r["metrics"] for r in fold_results
        },
    }
    atomic_write_json(record, local_results / "aggregate_metrics.json")
    logger.info("Wrote %s", local_results / "aggregate_metrics.json")
    return record


def teardown() -> None:
    logger.info("Terminating all running pods...")
    terminate_all()
    if POD_STATE_FILE.exists():
        POD_STATE_FILE.unlink()
    logger.info("All pods terminated.")


def full_pipeline(config_name: str = "phase1b_primary") -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1B PARALLEL FOLD EXECUTION")
    logger.info("=" * 60)
    start = time.time()

    pods = provision()
    run_folds(config_name)
    uncollected = collect(config_name)
    if uncollected:
        # Every fold that did not come back is a paid-for GPU hour whose only
        # copy of the per-item indicators is still on the pod. Terminating now
        # destroys it. Leave the pods up and make the operator decide.
        raise RuntimeError(
            f"Results were not collected from {uncollected}. Pods are still "
            f"running and have NOT been terminated so the results can be "
            f"retrieved: python -m scripts.phase1b.runpod_parallel collect. "
            f"Tear down with 'teardown' once they are safe."
        )
    teardown()

    elapsed = time.time() - start
    logger.info("Total pipeline time: %.1fm", elapsed / 60)

    aggregate(config_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1B RunPod parallel fold executor")
    parser.add_argument("action", nargs="?", default="full",
                        choices=["full", "provision", "run", "collect", "aggregate",
                                 "teardown"])
    parser.add_argument("--config-name", type=str, default="phase1b_primary",
                        help="Experiment config name")
    args = parser.parse_args()

    if args.action == "full":
        full_pipeline(args.config_name)
    elif args.action == "provision":
        provision()
    elif args.action == "run":
        run_folds(args.config_name)
    elif args.action == "collect":
        collect(args.config_name)
    elif args.action == "aggregate":
        aggregate(args.config_name)
    elif args.action == "teardown":
        teardown()


if __name__ == "__main__":
    main()
