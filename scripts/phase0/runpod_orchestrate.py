"""Phase 0 RunPod orchestrator: provision, bootstrap, run experiments, collect, teardown.

Replaces runpod_setup.sh with pure Python — no runpodctl CLI dependency.
Uses direct SSH/rsync via subprocess (proven pattern from old project).

Usage:
    python -m scripts.phase0.runpod_orchestrate [provision|run|collect|teardown|all]
    python -m scripts.phase0.runpod_orchestrate all          # full pipeline
    python -m scripts.phase0.runpod_orchestrate --terminate-all  # emergency cleanup
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

from scripts.phase0.runpod_provision import (
    create_pod,
    find_fastest_available,
    get_running_pods,
    terminate_all,
    terminate_pod,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase0"
POD_STATE_FILE: Final[Path] = PROJECT_ROOT / "scripts" / "phase0" / ".pod_state.json"

SSH_KEY: Final[str] = os.path.expanduser("~/.ssh/id_ed25519")
SSH_OPTS: Final[str] = (
    f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    f"-o LogLevel=ERROR -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "
    f"-i {SSH_KEY}"
)

POD_CONFIGS: Final[list[dict[str, str]]] = [
    {"name": "tract-phase0-bge", "role": "bge"},
    {"name": "tract-phase0-gte", "role": "gte"},
    {"name": "tract-phase0-deberta", "role": "deberta"},
]

DOCKER_IMAGE: Final[str] = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"


# ── SSH / rsync helpers ────────────────────────────────────────────────────


def _ssh(ip: str, port: int, cmd: str, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env_lines = ""
    if env:
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env.items()) + "\n"
    script = env_lines + cmd
    ssh_cmd = f"ssh {SSH_OPTS} -p {port} root@{ip} bash -s"
    logger.info("[ssh %s:%d] %s", ip, port, cmd[:100])
    result = subprocess.run(
        ssh_cmd, shell=True, input=script, text=True,
        capture_output=True, timeout=1800,
    )
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-5:]:
            logger.info("  stdout: %s", line)
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-3:]:
            logger.warning("  stderr: %s", line)
    if check and result.returncode != 0:
        raise RuntimeError(f"SSH command failed (exit {result.returncode}): {cmd[:120]}")
    return result


def _rsync_to(ip: str, port: int, local_path: str, remote_path: str) -> None:
    cmd = (
        f"rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' "
        f"--exclude='data/raw' --exclude='results' --exclude='.mypy_cache' "
        f"-e 'ssh {SSH_OPTS} -p {port}' {local_path} root@{ip}:{remote_path}"
    )
    logger.info("[rsync] %s -> %s:%s", local_path, ip, remote_path)
    subprocess.run(cmd, shell=True, check=True, timeout=300)


def _rsync_from(ip: str, port: int, remote_path: str, local_path: str) -> None:
    cmd = f"rsync -avz -e 'ssh {SSH_OPTS} -p {port}' root@{ip}:{remote_path} {local_path}"
    logger.info("[rsync] %s:%s -> %s", ip, remote_path, local_path)
    subprocess.run(cmd, shell=True, check=True, timeout=300)


# ── Pod state persistence ──────────────────────────────────────────────────


def _save_pod_state(pods: list[dict]) -> None:
    POD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    POD_STATE_FILE.write_text(json.dumps(pods, indent=2, sort_keys=True))
    logger.info("Pod state saved to %s", POD_STATE_FILE)


def _load_pod_state() -> list[dict]:
    if not POD_STATE_FILE.exists():
        raise FileNotFoundError(f"No pod state file at {POD_STATE_FILE} — run 'provision' first")
    return json.loads(POD_STATE_FILE.read_text())


# ── Pipeline stages ────────────────────────────────────────────────────────


def provision() -> list[dict]:
    logger.info("Finding fastest available GPU...")
    gpu_type = find_fastest_available(min_vram_gb=48)
    logger.info("Selected GPU: %s", gpu_type)

    pods: list[dict] = []
    for config in POD_CONFIGS:
        logger.info("Creating pod: %s", config["name"])
        pod = create_pod(gpu_type, name=config["name"], image=DOCKER_IMAGE)
        pod["role"] = config["role"]
        pods.append(pod)
        logger.info("Ready: %s @ %s:%d", config["name"], pod["ip"], pod["port"])

    _save_pod_state(pods)
    return pods


def _bootstrap_pod(pod: dict) -> None:
    ip, port, role = pod["ip"], pod["port"], pod["role"]
    logger.info("Bootstrapping %s (%s:%d)...", role, ip, port)

    _ssh(ip, port, "apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1", check=False)

    _rsync_to(ip, port, f"{PROJECT_ROOT}/", "/workspace/tract/")

    _ssh(ip, port, (
        "cd /workspace/tract && pip install --quiet -e '.[phase0]'"
    ))

    _ssh(ip, port, (
        "python --version && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader "
        "&& python -c 'import torch; print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()}\")'"
    ), check=False)

    logger.info("Bootstrap complete for %s", role)


def _run_on_pod(pod: dict, experiment_cmd: str) -> dict:
    ip, port, role = pod["ip"], pod["port"], pod["role"]
    logger.info("[%s] Starting: %s", role, experiment_cmd)
    start = time.time()
    try:
        anthropic_key = subprocess.run(
            ["pass", "anthropic/api-key"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        _ssh(ip, port, f"cd /workspace/tract && {experiment_cmd}",
             env={"ANTHROPIC_API_KEY": anthropic_key})
        elapsed = time.time() - start
        logger.info("[%s] COMPLETE in %.1fm", role, elapsed / 60)
        return {"role": role, "cmd": experiment_cmd, "status": "ok", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] FAILED after %.1fm: %s", role, elapsed / 60, e)
        return {"role": role, "cmd": experiment_cmd, "status": "failed", "error": str(e), "elapsed": elapsed}


def run_experiments() -> None:
    pods = _load_pod_state()
    pod_by_role = {p["role"]: p for p in pods}

    logger.info("Bootstrapping %d pods in parallel...", len(pods))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pods)) as ex:
        list(ex.map(_bootstrap_pod, pods))
    logger.info("All pods bootstrapped.")

    # Phase A: Embedding baselines (3 pods in parallel)
    logger.info("=" * 60)
    logger.info("PHASE A: Embedding baselines (3 pods parallel)")
    phase_a_cmds = {
        "bge": "python -m scripts.phase0.exp1_embedding_baseline --model bge --output-suffix _bge",
        "gte": "python -m scripts.phase0.exp1_embedding_baseline --model gte --output-suffix _gte",
        "deberta": "python -m scripts.phase0.exp1_embedding_baseline --model deberta --output-suffix _deberta",
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(_run_on_pod, pod_by_role[role], cmd): role
            for role, cmd in phase_a_cmds.items()
        }
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            logger.info("Phase A [%s]: %s", result["role"], result["status"])
    logger.info("PHASE A complete.")

    # Phase B: Hierarchy paths (BGE + GTE only)
    logger.info("=" * 60)
    logger.info("PHASE B: Hierarchy path experiments (BGE + GTE parallel)")
    phase_b_cmds = {
        "bge": "python -m scripts.phase0.exp3_hierarchy_paths --model bge --output-suffix _bge",
        "gte": "python -m scripts.phase0.exp3_hierarchy_paths --model gte --output-suffix _gte",
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_run_on_pod, pod_by_role[role], cmd): role
            for role, cmd in phase_b_cmds.items()
        }
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            logger.info("Phase B [%s]: %s", result["role"], result["status"])
    logger.info("PHASE B complete.")

    # Phase C: Hub descriptions (single pod — needs Anthropic API)
    logger.info("=" * 60)
    logger.info("PHASE C: Hub description pilot (single pod)")
    result = _run_on_pod(
        pod_by_role["bge"],
        "python -m scripts.phase0.exp4_hub_descriptions --model all",
    )
    logger.info("Phase C [%s]: %s", result["role"], result["status"])
    logger.info("PHASE C complete.")


def collect() -> None:
    pods = _load_pod_state()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for pod in pods:
        logger.info("Collecting from %s (%s:%d)...", pod["role"], pod["ip"], pod["port"])
        try:
            _rsync_from(pod["ip"], pod["port"], "/workspace/tract/results/phase0/", f"{RESULTS_DIR}/")
        except Exception as e:
            logger.warning("Collection from %s failed: %s", pod["role"], e)

    logger.info("Merging per-model result files...")
    _merge_results()

    logger.info("Results in %s:", RESULTS_DIR)
    for f in sorted(RESULTS_DIR.iterdir()):
        logger.info("  %s (%d bytes)", f.name, f.stat().st_size)


def _merge_results() -> None:
    for prefix in ["exp1_embedding_baseline", "exp3_hierarchy_paths"]:
        merged: dict = {"models": {}}
        for part_file in sorted(RESULTS_DIR.glob(f"{prefix}_*.json")):
            with open(part_file, encoding="utf-8") as f:
                part = json.load(f)
            for k, v in part.items():
                if k == "models":
                    merged["models"].update(v)
                elif k not in merged:
                    merged[k] = v
        if merged["models"]:
            out_path = RESULTS_DIR / f"{prefix}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, sort_keys=True, indent=2)
            logger.info("Merged %d models into %s", len(merged["models"]), out_path.name)


def teardown() -> None:
    try:
        pods = _load_pod_state()
        for pod in pods:
            logger.info("Terminating %s (%s)...", pod["role"], pod["pod_id"])
            try:
                terminate_pod(pod["pod_id"])
            except Exception as e:
                logger.warning("Failed to terminate %s: %s", pod["pod_id"], e)
        POD_STATE_FILE.unlink(missing_ok=True)
    except FileNotFoundError:
        logger.info("No pod state file — checking for running pods...")
        terminate_all()
    logger.info("Teardown complete.")


# ── Exp2 (LLM probe) runs locally — needs Anthropic API, no GPU ──────────

def run_exp2_local() -> None:
    logger.info("Running exp2 (LLM probe) locally — API-only, no GPU needed...")
    subprocess.run(
        [sys.executable, "-m", "scripts.phase0.exp2_llm_probe"],
        cwd=str(PROJECT_ROOT), check=True,
    )
    logger.info("Exp2 complete.")


def run_summary() -> None:
    logger.info("Running summary and gate evaluation...")
    subprocess.run(
        [sys.executable, "-m", "scripts.phase0.run_summary"],
        cwd=str(PROJECT_ROOT), check=True,
    )
    logger.info("Summary complete.")


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 RunPod orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stages:\n"
            "  provision     Create 3 GPU pods\n"
            "  run           Bootstrap pods and run exp1/exp3/exp4\n"
            "  collect       Rsync results back and merge\n"
            "  teardown      Terminate all pods\n"
            "  exp2          Run Opus LLM probe locally (API-only)\n"
            "  summary       Run summary and gate evaluation\n"
            "  all           provision → run → collect → teardown\n"
            "  full          all + exp2 + summary\n"
        ),
    )
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["provision", "run", "collect", "teardown",
                                 "exp2", "summary", "all", "full"])
    parser.add_argument("--terminate-all", action="store_true",
                        help="Emergency: terminate all running pods")
    args = parser.parse_args()

    if args.terminate_all:
        terminate_all()
        return

    if args.stage == "provision":
        provision()
    elif args.stage == "run":
        run_experiments()
    elif args.stage == "collect":
        collect()
    elif args.stage == "teardown":
        teardown()
    elif args.stage == "exp2":
        run_exp2_local()
    elif args.stage == "summary":
        run_summary()
    elif args.stage == "all":
        try:
            provision()
            run_experiments()
            collect()
        finally:
            teardown()
    elif args.stage == "full":
        try:
            provision()
            run_experiments()
            collect()
        finally:
            teardown()
        run_exp2_local()
        run_summary()


if __name__ == "__main__":
    main()
