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

Before the first run:
    ssh-keygen -t ed25519 -f ~/.ssh/tract_runpod -N ''
and register ~/.ssh/tract_runpod.pub with the RunPod account; TRACT_RUNPOD_SSH_KEY
overrides the path. Phase C (exp4_hub_descriptions) is the only stage that needs a
credential on the pod: export TRACT_PHASE0_SHIP_ANTHROPIC_KEY=1 to forward it, and
provision the fleet on the priced tier or the preflight will refuse.
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
from typing import Any, Final

from scripts.phase0.runpod_provision import (
    PRICE_CLOUD_TYPE,
    create_pods_parallel,
    find_fastest_available,
    terminate_all,
    terminate_pod,
    validate_ssh_endpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase0"
POD_STATE_FILE: Final[Path] = PROJECT_ROOT / "scripts" / "phase0" / ".pod_state.json"

# A key for this channel, not the operator's general-purpose identity. The
# default here was ~/.ssh/id_ed25519 -- the key that opens everything else this
# account reaches -- offered on a handshake with a host nothing authenticated.
# Create it with `ssh-keygen -t ed25519 -f ~/.ssh/tract_runpod -N ""` and
# register the public half with RunPod. TRACT_RUNPOD_SSH_KEY overrides the path
# and is not validated beyond existence, so pointing it back at the general
# identity undoes this; the separation is a default, not an enforcement.
SSH_KEY: Final[str] = os.environ.get(
    "TRACT_RUNPOD_SSH_KEY", os.path.expanduser("~/.ssh/tract_runpod")
)
# Host keys are recorded per run rather than discarded to /dev/null. Every
# endpoint on this path arrives from the RunPod API or from a .pod_state.json
# that may be hours stale, and RunPod hands out IP:port pairs from a recycled
# pool, so StrictHostKeyChecking=no meant each of them was trusted sight unseen
# and a substituted host got the root session and whatever it carried.
# accept-new is the strongest posture available while RunPod publishes no host
# key in advance: it trusts first contact and makes a key that changes mid-run a
# hard failure. provision() discards the file each round -- see the note there.
KNOWN_HOSTS_FILE: Final[Path] = (
    Path(__file__).resolve().parent / ".runpod_known_hosts"
)
# A tuple, not a string, because nothing on this path goes through a local shell
# any more: these are argv elements.
SSH_OPTS: Final[tuple[str, ...]] = (
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={KNOWN_HOSTS_FILE}",
    "-o", "IdentitiesOnly=yes",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=10",
    "-i", SSH_KEY,
)

# The two subprocess walls, named because they were bare literals at the call
# sites. An experiment step is the long one; a transfer is not.
SSH_TIMEOUT_S: Final[int] = 1800
RSYNC_TIMEOUT_S: Final[int] = 300

POD_CONFIGS: Final[list[dict[str, str]]] = [
    {"name": "tract-p0r-a", "role": "small-a"},
    {"name": "tract-p0r-b", "role": "small-b"},
    {"name": "tract-p0r-c", "role": "large-1"},
    {"name": "tract-p0r-d", "role": "large-2"},
]

DOCKER_IMAGE: Final[str] = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


# ── SSH / rsync helpers ────────────────────────────────────────────────────


def _require_ssh_key() -> None:
    """Refuse to touch the channel without the dedicated key.

    Failing here is the point of moving off ~/.ssh/id_ed25519: a missing
    dedicated key has to be an error that names the fix, not a silent fallback
    to the identity that opens everything else. provision() calls this before it
    buys anything, so the usual discovery is free rather than four billing H100s
    in.

    Raises:
        FileNotFoundError: No key at SSH_KEY. The message carries the
            ssh-keygen line that creates one.
    """
    if not os.path.exists(SSH_KEY):
        raise FileNotFoundError(
            f"No SSH key at {SSH_KEY}. Phase 0 pods are reached with a "
            f"dedicated key rather than the operator's general-purpose "
            f"identity. Create one:\n"
            f"  ssh-keygen -t ed25519 -f {SSH_KEY} -N ''\n"
            f"then add {SSH_KEY}.pub to the RunPod account's SSH keys. Set "
            f"TRACT_RUNPOD_SSH_KEY to use a different path."
        )


def _ssh_transport(port: int) -> str:
    """The -e string rsync starts ssh with.

    rsync splits this itself, so it is the one place the options still have to
    be a string rather than an argv. shlex.join keeps a key or known-hosts path
    that contains a space from silently becoming two options.
    """
    return shlex.join(["ssh", *SSH_OPTS, "-p", str(port)])


def _ssh(
    ip: str, port: int, cmd: str, check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _require_ssh_key()
    # ip and port come from the RunPod API or from a .pod_state.json an earlier
    # run wrote. Parse them before they are addressed, so a value carrying shell
    # metacharacters is a refusal rather than an argument.
    ip, port = validate_ssh_endpoint(ip, port)
    env_lines = ""
    if env:
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env.items()) + "\n"
    script = env_lines + cmd
    # argv, not a shell string. `cmd` still travels on stdin to `bash -s` on the
    # pod, which is the design; what is gone is the LOCAL shell that used to
    # re-interpret the endpoint on the way out.
    ssh_argv = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{ip}", "bash", "-s"]
    logger.info("[ssh %s:%d] %s", ip, port, cmd[:100])
    result = subprocess.run(
        ssh_argv, input=script, text=True,
        capture_output=True, timeout=SSH_TIMEOUT_S,
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
    """Ship the working tree to a pod."""
    _require_ssh_key()
    ip, port = validate_ssh_endpoint(ip, port)
    # The exclude list is what stands between the operator's working tree and
    # four rented hosts, so it has to cover what .gitignore does. It covered six
    # patterns and none of the sensitive ones.
    #
    # .pod_state.json holds every pod's live IP and SSH port, so copying it to
    # all of them handed each pod the address of the rest; '.pod_state.json.*'
    # and '*.tmp' close the same hole for the partial copies that can sit beside
    # it, and .runpod_known_hosts is this run's host-key record, which belongs to
    # the operator and to nobody it connects to.
    #
    # .env is this repo's conventional secret file (.gitignore line 63) and *.db
    # is crosswalk.db at the root; both were being copied to four rented hosts,
    # which is the same disclosure the credential exports were, arriving through
    # the filesystem instead of stdin and outliving the process. data/raw is the
    # immutable licensed corpus and no phase 0 experiment reads it.
    excludes = [f"--exclude={pat}" for pat in (
        "__pycache__", "*.pyc", ".git", "results", ".mypy_cache", "models",
        ".env", "*.db", "data/raw", "venv", ".venv", ".claude",
        ".pod_state.json", ".pod_state.json.*", "*.tmp", ".runpod_known_hosts",
    )]
    argv = [
        "rsync", "-rltz", *excludes,
        "-e", _ssh_transport(port),
        local_path, f"root@{ip}:{remote_path}",
    ]
    logger.info("[rsync to] %s:%d %s", ip, port, remote_path)
    subprocess.run(argv, check=True, timeout=RSYNC_TIMEOUT_S)


def _rsync_from(ip: str, port: int, remote_path: str, local_path: str) -> None:
    """Retrieve results from a pod.

    --safe-links drops any symlink pointing outside the transfer. Without it a
    compromised pod ships `x -> /home/<op>/.ssh` and a later pass writes through
    it, which turns collecting results into an arbitrary write on the operator's
    machine. -l is kept because the tree may hold internal links.
    """
    _require_ssh_key()
    ip, port = validate_ssh_endpoint(ip, port)
    argv = [
        "rsync", "-rltz", "--safe-links",
        "-e", _ssh_transport(port),
        f"root@{ip}:{remote_path}", local_path,
    ]
    logger.info("[rsync from] %s:%d %s -> %s", ip, port, remote_path, local_path)
    subprocess.run(argv, check=True, timeout=RSYNC_TIMEOUT_S)


# ── Credential helpers ────────────────────────────────────────────────────


ANTHROPIC_KEY_ENTRY: Final[str] = "anthropic/api-key"
# Forwarding an account-wide API key to a rented host is a decision the operator
# makes each time, not a default. "1" allows it; anything else does not.
SHIP_ANTHROPIC_KEY_ENV: Final[str] = "TRACT_PHASE0_SHIP_ANTHROPIC_KEY"
# The one scheduled command that calls the Anthropic API. Every other command on
# these pods is a local embedding run, and the ten-command schedule is in
# run_experiments() below if this ever grows a second. Gating on the RUN rather
# than the COMMAND was the earlier mistake: it left the key exported on all four
# pods for all of phases A and B, which is verbatim the exposure being fixed.
ANTHROPIC_EXPERIMENT_MODULE: Final[str] = "scripts.phase0.exp4_hub_descriptions"


def _get_credential(name: str) -> str:
    result = subprocess.run(
        ["pass", name], capture_output=True, text=True, check=True, timeout=10,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"pass returned empty value for {name}")
    return value


def _get_pod_env(pod: dict[str, Any], cmd: str) -> dict[str, str]:
    """Environment exported on a pod before one experiment command runs.

    Empty unless THIS command is the one that needs a credential and the
    operator asked for it. This used to read ANTHROPIC_API_KEY and
    WANDB_API_KEY out of `pass` and export both before every command on all four
    pods -- account-wide keys, one able to spend money and one able to rewrite
    the experiment record -- over a channel that checked no host key, onto hosts
    that may have landed on the COMMUNITY tier. Nine of the ten scheduled
    commands never call either service.

    The WandB key is gone rather than gated. Phase 0 is finished; `collect`
    brings the result JSON back and `run_summary` runs on the operator's
    machine, which is where that key already lives. init_wandb returns None when
    it cannot find a key, so a pod simply runs untracked.

    The Anthropic key is the one a pod experiment genuinely needs --
    exp4_hub_descriptions, phase C -- so it is gated rather than removed. It is
    still an account-wide, spend-capable key on a rented host reached over a
    trust-on-first-use channel; the gate narrows who gets it, it does not make
    it safe.

    Raises:
        RuntimeError: This pod's recorded tier is not the one the fleet was
            priced on. _require_priced_tier already refused the whole fleet
            before the bootstrap push, so reaching this is a last-line assertion
            that the preflight ran -- not the control that prevents the
            exposure, which has to precede the push.
        Whatever _get_credential raises when `pass` cannot supply the key:
            subprocess.CalledProcessError, subprocess.TimeoutExpired after 10s,
            or RuntimeError on an empty value. The old code logged those and
            continued, so a missing credential surfaced an hour later as a
            failed experiment instead of here.
    """
    if ANTHROPIC_EXPERIMENT_MODULE not in cmd:
        return {}
    if os.environ.get(SHIP_ANTHROPIC_KEY_ENV, "").strip() != "1":
        logger.info(
            "No credential forwarded to %s for %s. Export %s=1 to allow it.",
            pod.get("role", "?"), ANTHROPIC_EXPERIMENT_MODULE,
            SHIP_ANTHROPIC_KEY_ENV,
        )
        return {}
    tier = pod.get("cloud_type") or "unrecorded"
    if tier != PRICE_CLOUD_TYPE:
        raise RuntimeError(
            f"Refusing to forward the Anthropic key to pod {pod.get('role', '?')}: "
            f"it is on cloud tier {tier}, not {PRICE_CLOUD_TYPE}. "
            f"_require_priced_tier should already have refused this fleet before "
            f"the bootstrap push; reaching here means it did not run."
        )
    return {"ANTHROPIC_API_KEY": _get_credential(ANTHROPIC_KEY_ENTRY)}


def _require_priced_tier(pods: list[dict[str, Any]]) -> None:
    """Refuse a credential-carrying run whose fleet is not on the priced tier.

    This has to run BEFORE the bootstrap push, which is why it is a fleet-level
    preflight and not a per-pod check inside _get_pod_env: by the time an
    experiment command is being assembled, _bootstrap_pod has already rsynced
    the working tree to every pod, and create_pod's own comment states the stake
    -- the tier is a statement about where licensed corpus went. A refusal that
    arrives then is a report, not a control.

    create_pod falls back a tier silently when capacity is short, so a fleet is
    not uniform and a pod dict from an older state file carries no cloud_type at
    all; both read as off-tier, which is the fail-closed direction.

    Raises:
        RuntimeError: The operator opted in to forwarding the Anthropic key and
            at least one pod is not on PRICE_CLOUD_TYPE.
    """
    if os.environ.get(SHIP_ANTHROPIC_KEY_ENV, "").strip() != "1":
        return
    off_tier = [
        (str(p.get("role", "?")), str(p.get("cloud_type") or "unrecorded"))
        for p in pods
        if (p.get("cloud_type") or "unrecorded") != PRICE_CLOUD_TYPE
    ]
    if off_tier:
        detail = ", ".join(f"{role} on {tier}" for role, tier in off_tier)
        raise RuntimeError(
            f"{SHIP_ANTHROPIC_KEY_ENV}=1 asks for an account-wide Anthropic key "
            f"to be forwarded, but this fleet did not all land on "
            f"{PRICE_CLOUD_TYPE}: {detail}. Re-provision, or unset "
            f"{SHIP_ANTHROPIC_KEY_ENV} and skip phase C."
        )


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
    # Refuse before anything is bought. _ssh and the rsync helpers each call
    # this too, but by then four H100s are provisioned and billing at roughly
    # $3.29/hr apiece -- and the first run after this change is the likely one
    # to be missing the key, since the operator has only ever registered
    # ~/.ssh/id_ed25519.pub. A fleet that cannot be reached must not be bought.
    _require_ssh_key()

    # Start each round with no recorded host keys. The file only accumulates,
    # and RunPod draws IP:port from a pool, so an endpoint eventually returns
    # attached to a different machine and ssh answers "Host key verification
    # failed" against a fleet that is already billing. Discarding it costs
    # nothing: every pod below is created minutes from now, so an entry from an
    # earlier round describes a host that no longer exists. Within a round the
    # file still does its job -- accept-new records each pod on first contact,
    # and a key that changes mid-run stays a hard failure.
    if KNOWN_HOSTS_FILE.exists():
        logger.info(
            "Discarding %s: its entries describe pods that no longer exist, and "
            "a recycled endpoint would read as a host-key failure.",
            KNOWN_HOSTS_FILE,
        )
    KNOWN_HOSTS_FILE.unlink(missing_ok=True)

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


def _run_on_pod(pod: dict[str, Any], experiment_cmd: str) -> dict[str, Any]:
    ip, port, role = pod["ip"], pod["port"], pod["role"]
    logger.info("[%s] Starting: %s", role, experiment_cmd)
    start = time.time()
    # Per pod AND per command: whether a credential may be forwarded depends on
    # the tier this pod landed on, and only one of the ten scheduled commands
    # calls the API at all.
    env = _get_pod_env(pod, experiment_cmd)
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

    # Before the push, not at the first experiment command. _bootstrap_pod below
    # rsyncs the working tree to every pod in the fleet, so a tier refusal raised
    # once an experiment is being assembled arrives after the thing it exists to
    # prevent has already happened.
    _require_priced_tier(pods)

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
