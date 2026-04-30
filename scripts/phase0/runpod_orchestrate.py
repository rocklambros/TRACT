"""Phase 0R RunPod orchestrator: provision, bootstrap, run experiments, collect, teardown.

Provisions pods IN PARALLEL, bootstraps them in parallel, then runs the
Phase 0R experiment schedule across 4 GPU pods + local API experiments.

Usage:
    python -m scripts.phase0.runpod_orchestrate phase0r       # full Phase 0R pipeline
    python -m scripts.phase0.runpod_orchestrate provision      # create pods only
    python -m scripts.phase0.runpod_orchestrate run            # bootstrap + run on existing pods
    python -m scripts.phase0.runpod_orchestrate collect        # rsync results back
    python -m scripts.phase0.runpod_orchestrate teardown       # terminate all pods
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
    create_pods_parallel,
    find_fastest_available,
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
    {"name": "tract-p0r-a", "role": "small-a"},
    {"name": "tract-p0r-b", "role": "small-b"},
    {"name": "tract-p0r-c", "role": "large-1"},
    {"name": "tract-p0r-d", "role": "large-2"},
]

DOCKER_IMAGE: Final[str] = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


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
        f"rsync -rltz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' "
        f"--exclude='results' --exclude='.mypy_cache' --exclude='models' "
        f"-e 'ssh {SSH_OPTS} -p {port}' {local_path} root@{ip}:{remote_path}"
    )
    logger.info("[rsync] %s -> %s:%s", ip, port, remote_path)
    subprocess.run(cmd, shell=True, check=True, timeout=300)


def _rsync_from(ip: str, port: int, remote_path: str, local_path: str) -> None:
    cmd = f"rsync -rltz -e 'ssh {SSH_OPTS} -p {port}' root@{ip}:{remote_path} {local_path}"
    logger.info("[rsync] %s:%s -> %s", ip, remote_path, local_path)
    subprocess.run(cmd, shell=True, check=True, timeout=300)


# ── Credential helpers ────────────────────────────────────────────────────


def _get_credential(name: str) -> str:
    result = subprocess.run(
        ["pass", name], capture_output=True, text=True, check=True, timeout=10,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"pass returned empty value for {name}")
    return value


def _get_pod_env() -> dict[str, str]:
    """Get environment variables to forward to pods."""
    env: dict[str, str] = {}
    try:
        env["ANTHROPIC_API_KEY"] = _get_credential("anthropic/api-key")
    except Exception as e:
        logger.warning("Could not get Anthropic API key: %s", e)
    try:
        env["WANDB_API_KEY"] = _get_credential("wandb/api-key")
    except Exception as e:
        logger.warning("Could not get WandB API key: %s", e)
    return env


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
    """Provision all pods IN PARALLEL."""
    logger.info("Finding fastest available GPU...")
    gpu_type = find_fastest_available(min_vram_gb=48)
    logger.info("Selected GPU: %s", gpu_type)

    pods = create_pods_parallel(POD_CONFIGS, gpu_type, image=DOCKER_IMAGE)

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
    env = _get_pod_env()
    try:
        _ssh(ip, port, f"cd /workspace/tract && {experiment_cmd}", env=env)
        elapsed = time.time() - start
        logger.info("[%s] COMPLETE in %.1fm", role, elapsed / 60)
        return {"role": role, "cmd": experiment_cmd, "status": "ok", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] FAILED after %.1fm: %s", role, elapsed / 60, e)
        return {"role": role, "cmd": experiment_cmd, "status": "failed", "error": str(e), "elapsed": elapsed}


def _run_cmds_sequential(pod: dict, cmds: list[str]) -> list[dict]:
    """Run multiple commands sequentially on a single pod."""
    results: list[dict] = []
    for cmd in cmds:
        result = _run_on_pod(pod, cmd)
        results.append(result)
        if result["status"] == "failed":
            logger.warning("Stopping sequential run on %s after failure", pod["role"])
            break
    return results


def _run_phase_parallel(
    pod_commands: dict[str, list[str]],
    pod_by_role: dict[str, dict],
    phase_name: str,
) -> list[dict]:
    """Run experiment commands across pods in parallel."""
    logger.info("=" * 60)
    logger.info("PHASE %s", phase_name)

    all_results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pod_commands)) as ex:
        futures = {
            ex.submit(_run_cmds_sequential, pod_by_role[role], cmds): role
            for role, cmds in pod_commands.items()
        }
        for f in concurrent.futures.as_completed(futures):
            role = futures[f]
            results = f.result()
            for r in results:
                logger.info("  [%s] %s: %s", role, r["cmd"][:60], r["status"])
            all_results.extend(results)

    logger.info("PHASE %s complete.", phase_name)
    return all_results


def run_experiments() -> None:
    """Bootstrap pods and run the full Phase 0R experiment schedule."""
    pods = _load_pod_state()
    pod_by_role = {p["role"]: p for p in pods}

    logger.info("Bootstrapping %d pods in parallel...", len(pods))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pods)) as ex:
        list(ex.map(_bootstrap_pod, pods))
    logger.info("All pods bootstrapped.")

    all_run_results: list[dict] = []

    # Phase A: Small model baselines + kNN (3 pods parallel, ~15 min)
    phase_a = {
        "small-a": [
            "python -m scripts.phase0.exp1_embedding_baseline --model bge --curated --output-suffix _bge_curated",
            "python -m scripts.phase0.exp1_embedding_baseline --model gte --curated --output-suffix _gte_curated",
        ],
        "small-b": [
            "python -m scripts.phase0.exp3_hierarchy_paths --model bge --curated --output-suffix _bge_curated",
            "python -m scripts.phase0.exp3_hierarchy_paths --model gte --curated --output-suffix _gte_curated",
        ],
        "large-1": [
            "python -m scripts.phase0.exp5_knn_baseline --model bge",
        ],
    }
    all_run_results.extend(_run_phase_parallel(phase_a, pod_by_role, "A: Small models + kNN"))

    # Phase B: Large embedding models (2 pods parallel, ~30 min each)
    phase_b = {
        "large-1": [
            "python -m scripts.phase0.exp7_extended_models --model gte-qwen2-1.5b --output-suffix _gte-qwen2",
            "python -m scripts.phase0.exp7_extended_models --model e5-mistral-7b --output-suffix _e5-mistral",
        ],
        "large-2": [
            "python -m scripts.phase0.exp7_extended_models --model nv-embed-v2 --output-suffix _nv-embed",
            "python -m scripts.phase0.exp7_extended_models --model sfr-embedding-2 --output-suffix _sfr",
        ],
    }
    all_run_results.extend(_run_phase_parallel(phase_b, pod_by_role, "B: Large models"))

    # Phase C: Hub descriptions (single pod — uses Anthropic API)
    phase_c = {
        "small-a": [
            "python -m scripts.phase0.exp4_hub_descriptions --model all --curated",
        ],
    }
    all_run_results.extend(_run_phase_parallel(phase_c, pod_by_role, "C: Hub descriptions"))

    # Summary
    logger.info("=" * 60)
    logger.info("ALL REMOTE EXPERIMENTS COMPLETE")
    ok = sum(1 for r in all_run_results if r["status"] == "ok")
    failed = sum(1 for r in all_run_results if r["status"] == "failed")
    logger.info("  OK: %d, FAILED: %d", ok, failed)
    for r in all_run_results:
        if r["status"] == "failed":
            logger.error("  FAILED: [%s] %s — %s", r["role"], r["cmd"][:60], r.get("error", ""))


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
    for prefix in [
        "exp1_embedding_baseline",
        "exp3_hierarchy_paths",
        "exp7_extended_models",
    ]:
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


# ── Local experiments (API-only, no GPU) ──────────────────────────────────


def run_exp2_local() -> None:
    logger.info("Running exp2 (Opus LLM probe) locally — API-only, no GPU needed...")
    subprocess.run(
        [sys.executable, "-m", "scripts.phase0.exp2_llm_probe"],
        cwd=str(PROJECT_ROOT), check=True,
    )
    logger.info("Exp2 complete.")


def run_exp6_local() -> None:
    logger.info("Running exp6 (few-shot Sonnet) locally — API-only, no GPU needed...")
    subprocess.run(
        [sys.executable, "-m", "scripts.phase0.exp6_fewshot_sonnet"],
        cwd=str(PROJECT_ROOT), check=True,
    )
    logger.info("Exp6 complete.")


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
        description="Phase 0R RunPod orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stages:\n"
            "  provision     Create 4 GPU pods (parallel)\n"
            "  run           Bootstrap pods and run GPU experiments\n"
            "  collect       Rsync results back and merge\n"
            "  teardown      Terminate all pods\n"
            "  exp2          Run Opus LLM probe locally (API-only)\n"
            "  exp6          Run few-shot Sonnet locally (API-only)\n"
            "  summary       Run summary and gate evaluation\n"
            "  all           provision → run → collect → teardown\n"
            "  phase0r       all + exp6 + summary (full Phase 0R)\n"
            "  full          all + exp2 + exp6 + summary\n"
        ),
    )
    parser.add_argument("stage", nargs="?", default="phase0r",
                        choices=["provision", "run", "collect", "teardown",
                                 "exp2", "exp6", "summary",
                                 "all", "phase0r", "full"])
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
    elif args.stage == "exp6":
        run_exp6_local()
    elif args.stage == "summary":
        run_summary()
    elif args.stage == "all":
        try:
            provision()
            run_experiments()
            collect()
        finally:
            teardown()
    elif args.stage == "phase0r":
        try:
            provision()
            run_experiments()
            collect()
        finally:
            teardown()
        run_exp6_local()
        run_summary()
    elif args.stage == "full":
        try:
            provision()
            run_experiments()
            collect()
        finally:
            teardown()
        run_exp2_local()
        run_exp6_local()
        run_summary()


if __name__ == "__main__":
    main()
