"""Phase 1B RunPod parallel fold executor.

Provisions one pod per LOFO fold, bootstraps them in parallel, runs training
and evaluation simultaneously, collects results, aggregates, tears down.

Usage:
    python -m scripts.phase1b.runpod_parallel                 # full pipeline
    python -m scripts.phase1b.runpod_parallel price           # budget check only, creates nothing
    python -m scripts.phase1b.runpod_parallel provision       # create pods only
    python -m scripts.phase1b.runpod_parallel run             # bootstrap + run on existing pods
    python -m scripts.phase1b.runpod_parallel collect         # rsync results back
    python -m scripts.phase1b.runpod_parallel aggregate       # micro-average + gate decision
    python -m scripts.phase1b.runpod_parallel teardown        # terminate THIS RUN's pods
    python -m scripts.phase1b.runpod_parallel reap --confirm  # recover from a dead orchestrator

Before the first run:
    ssh-keygen -t ed25519 -f ~/.ssh/tract_runpod -N ''
and register ~/.ssh/tract_runpod.pub with the RunPod account.

Environment overrides: TRACT_RUNPOD_BUDGET_USD (default 1000),
TRACT_RUNPOD_MAX_HOURLY (12), TRACT_RUNPOD_MAX_HOURS (6),
TRACT_RUNPOD_SSH_KEY.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

from scripts.phase0.runpod_provision import (
    is_capacity_error,
    rank_available_gpus,
    create_pods_parallel,
    find_fastest_available,
    get_gpu_price,
    get_running_pods,
    terminate_pods,
    validate_ssh_endpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1b"
POD_STATE_FILE: Final[Path] = PROJECT_ROOT / "scripts" / "phase1b" / ".pod_state.json"

# A run-scoped key, not the operator's general-purpose identity. The default
# used to be ~/.ssh/id_ed25519, offered on every handshake to a host whose key
# was never checked. Override for a different path; provision it with
# `ssh-keygen -t ed25519 -f ~/.ssh/tract_runpod -N ""` and register the public
# half with RunPod.
SSH_KEY: Final[str] = os.environ.get(
    "TRACT_RUNPOD_SSH_KEY", os.path.expanduser("~/.ssh/tract_runpod")
)
# Host keys are recorded per run rather than discarded to /dev/null, so a pod
# whose key changes mid-run is a hard failure instead of a silent reconnect to
# something else. accept-new trusts first contact, which is the best available
# posture given RunPod publishes no host key in advance.
KNOWN_HOSTS_FILE: Final[Path] = (
    Path(__file__).resolve().parent / ".runpod_known_hosts"
)
SSH_OPTS: Final[str] = (
    f"-o StrictHostKeyChecking=accept-new "
    f"-o UserKnownHostsFile={KNOWN_HOSTS_FILE} "
    f"-o IdentitiesOnly=yes "
    f"-o LogLevel=ERROR -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "
    f"-i {SSH_KEY}"
)

# Digest-pinned. A mutable tag let the image change under the run: the previous
# tag shipped torch 2.4.0, which cannot run the pinned stack (torch 2.13
# requires CUDA 13). See the ml-stack pin for how these digests were resolved.
DOCKER_IMAGE: Final[str] = (
    "runpod/pytorch@sha256:"
    "bf9f4b90f4a8cd55d902b74003859fd6bce06255bb135acd964a7b71bf31fa05"
)

# Budget controls. The $1000 ceiling was prose; these make it a gate.
BUDGET_USD: Final[float] = float(os.environ.get("TRACT_RUNPOD_BUDGET_USD", "2000"))
# Refuse a part whose rate would burn the budget faster than the run can finish.
MAX_USD_PER_HOUR_PER_POD: Final[float] = float(
    os.environ.get("TRACT_RUNPOD_MAX_HOURLY", "12")
)
# Folds are expected in well under this; it is the wall the watchdog enforces.
MAX_RUN_HOURS: Final[float] = float(os.environ.get("TRACT_RUNPOD_MAX_HOURS", "6"))

# A fold's results run to gigabytes; the previous 300s was optimistic. Retries
# because a failed collection destroys work that has already been paid for.
RSYNC_TIMEOUT_S: Final[int] = 1800
RSYNC_ATTEMPTS: Final[int] = 3

# One fold: LoRA training plus a paired zero-shot pass.
FOLD_TIMEOUT_S: Final[int] = 7200
# The fold runs detached and the orchestrator polls for its exit sentinel.
FOLD_POLL_INTERVAL_S: Final[int] = 60
# Named apart from runpod_provision.SSH_POLL_TIMEOUT_S, which is the much
# longer ceiling for a pod's SSH coming up in the first place.
FOLD_POLL_SSH_TIMEOUT_S: Final[int] = 120
# A poll failure means the network blinked, not that the fold died; the fold
# is detached and still running. Only a sustained outage is treated as fatal.
MAX_CONSECUTIVE_POLL_ERRORS: Final[int] = 10

# Default per-command SSH ceiling; bootstrap issues several.
SSH_DEFAULT_TIMEOUT_S: Final[int] = 3600

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


def select_pod_configs(folds: list[str] | None = None) -> list[dict[str, str]]:
    """Pod configs for a subset of folds, preserving the canonical names.

    Exists so a canary is a supported operation rather than a hand-edit of
    POD_CONFIGS. Before this, provision() always created all five pods, so the
    only way to validate the machinery end to end was to pay for the whole
    fleet -- which is the opposite of what a canary is for.

    Pod names stay tied to the framework's index in FOLD_FRAMEWORKS, not to
    its position in the filtered list, so a canary pod and the same fold in a
    later full run carry the same name and `reap` recognises both.
    """
    if not folds:
        return list(POD_CONFIGS)
    unknown = [f for f in folds if f not in FOLD_FRAMEWORKS]
    if unknown:
        raise ValueError(
            f"Unknown fold(s) {unknown}. Expected any of {FOLD_FRAMEWORKS}."
        )
    wanted = set(folds)
    return [c for c in POD_CONFIGS if c["role"] in wanted]


HF_READ_TOKEN_ENTRY: Final[str] = "huggingface/read-token"


def _get_hf_read_token() -> str:
    """A READ-ONLY HuggingFace token for the pods, or raise.

    The base model was fetched anonymously until HuggingFace rate-limited the
    datacenter IP mid-canary (HTTP 429, "create a HF account ... and make sure
    you pass a HF_TOKEN"). Anonymous fetching is therefore not a property of
    the repo being public; it is a quota that a fleet of pods exhausts.

    This deliberately does NOT use `pass huggingface/token`. That entry is a
    fine-grained token carrying repo.write, inference.*.write and job.write,
    which means write access to the published model and dataset repos. Putting
    it on five rented hosts to download a public model would trade a
    catastrophic credential for a convenience. A separate read-only entry
    keeps the worst case at "someone reads public models".
    """
    try:
        result = subprocess.run(
            ["pass", HF_READ_TOKEN_ENTRY],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"No `pass {HF_READ_TOKEN_ENTRY}`. The pods need a HuggingFace "
            f"token to fetch the base model without hitting the anonymous "
            f"rate limit.\n"
            f"Create a READ-ONLY one at "
            f"https://huggingface.co/settings/tokens (type: Fine-grained, "
            f"tick only 'Read access to contents of all public gated repos'), "
            f"then:\n"
            f"  pass insert -f -e {HF_READ_TOKEN_ENTRY}\n"
            f"Do NOT reuse `pass huggingface/token`: it carries repo.write to "
            f"the published model and dataset."
        ) from exc
    token = result.stdout.strip()
    if not token:
        raise RuntimeError(f"`pass {HF_READ_TOKEN_ENTRY}` is empty.")
    return token


def _get_pod_env() -> dict[str, str]:
    """Environment exported on the pod before the fold command runs.

    Carries exactly one credential: a read-only HuggingFace token, because
    anonymous model fetching is rate-limited per IP and a fleet exhausts it.
    No WandB key goes here -- logging happens on the operator's machine after
    collection -- and no HuggingFace WRITE token, which would put push access
    to the published model on a host the operator does not control.

    HF_HOME is redirected onto the 50GB volume; the container disk is not
    where a multi-gigabyte model cache belongs.
    """
    return {
        "HF_TOKEN": _get_hf_read_token(),
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Determinism, per PRD 6.4.2.
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        # Anchor lengths vary from a few tokens to the full 512-token window,
        # so the allocator sees a wide spread of block sizes and fragments.
        # The OOM that killed the canary reported 621 MiB reserved but
        # unallocated and recommended exactly this.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # _rsync_to excludes .git, so `git rev-parse` on the pod has no
        # repository to read and every fold recorded git_sha="unknown". A
        # fleet that unanimously says "unknown" passes load_fold_results'
        # stale-fold check as a warning, so the check was dead on the only
        # path that spends money. This is the SHA of the tree being shipped.
        "TRACT_GIT_SHA": _local_git_sha(),
    }


def _local_git_sha() -> str:
    """Short SHA of the working tree this orchestrator is shipping."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    logger.warning(
        "Could not resolve the local git SHA; folds will record 'unknown' and "
        "the aggregate cannot be tied to a commit."
    )
    return "unknown"


def _require_ssh_key() -> None:
    if not os.path.exists(SSH_KEY):
        raise FileNotFoundError(
            f"No SSH key at {SSH_KEY}. This run uses a dedicated key rather "
            f"than the operator's general-purpose identity. Create one:\n"
            f"  ssh-keygen -t ed25519 -f {SSH_KEY} -N ''\n"
            f"then add {SSH_KEY}.pub to the RunPod account's SSH keys. Set "
            f"TRACT_RUNPOD_SSH_KEY to use a different path."
        )


def _ssh(
    ip: str, port: int, cmd: str,
    check: bool = True,
    env: dict[str, str] | None = None,
    timeout: int = SSH_DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    _require_ssh_key()
    ip, port = validate_ssh_endpoint(ip, port)
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
    ip, port = validate_ssh_endpoint(ip, port)
    # The exclude list is what stands between the operator's working tree and
    # five rented hosts, so it has to cover everything .gitignore does. The
    # sharpest omission was .pod_state.json: it is chmod 600 locally precisely
    # because it holds every pod's live IP and SSH port, and it was being copied
    # to all of them, so owning one pod disclosed the address of the rest.
    excludes = " ".join(f"--exclude={pat!r}" for pat in (
        "__pycache__", "*.pyc", ".git", "results", ".mypy_cache", "models",
        "wandb", ".wandb", ".env", "*.db", "data/raw", ".claude", "venv",
        ".venv", ".pod_state.json", ".runpod_known_hosts", "build",
        ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache", "*.egg-info",
        ".DS_Store",
    ))
    cmd = (
        f"rsync -rltz {excludes} "
        f"-e 'ssh {SSH_OPTS} -p {port}' {local_path} root@{ip}:{remote_path}"
    )
    logger.info("[rsync to] %s:%d %s", ip, port, remote_path)
    subprocess.run(cmd, shell=True, check=True, timeout=RSYNC_TIMEOUT_S)


def _rsync_from(ip: str, port: int, remote_path: str, local_path: str) -> None:
    """Retrieve results, with retries.

    A fold's output runs to gigabytes and the old 300s timeout was optimistic
    for it. Collection is also the one step whose failure destroys work that
    has already been paid for, so it retries rather than giving up first time.
    """
    ip, port = validate_ssh_endpoint(ip, port)
    # --safe-links drops any symlink pointing outside the transfer. Without it a
    # compromised pod can ship `x -> /Users/<op>/.ssh` and a later pass writes
    # through it, which turns retrieving results into an arbitrary write on the
    # operator's machine. -l is kept because the tree may hold internal links.
    cmd = (
        f"rsync -rltz --safe-links --partial --timeout=120 "
        f"-e 'ssh {SSH_OPTS} -p {port}' root@{ip}:{remote_path} {local_path}"
    )
    last_error: Exception | None = None
    for attempt in range(1, RSYNC_ATTEMPTS + 1):
        logger.info("[rsync from] %s:%d %s (attempt %d/%d)",
                    ip, port, remote_path, attempt, RSYNC_ATTEMPTS)
        try:
            subprocess.run(cmd, shell=True, check=True, timeout=RSYNC_TIMEOUT_S)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            logger.warning("  rsync attempt %d failed: %s", attempt, exc)
            if attempt < RSYNC_ATTEMPTS:
                time.sleep(10 * attempt)
    raise RuntimeError(
        f"Failed to collect {remote_path} from {ip}:{port} after "
        f"{RSYNC_ATTEMPTS} attempts: {last_error}"
    )


def _save_pod_state(
    pods: list[dict[str, Any]], meta: dict[str, Any] | None = None,
) -> None:
    POD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pods": pods, "meta": meta or {}}
    POD_STATE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))
    # Contains live pod IPs and SSH ports. .gitignore covers it; chmod keeps it
    # off other local accounts too.
    POD_STATE_FILE.chmod(0o600)
    logger.info("Pod state saved to %s (%d pods)", POD_STATE_FILE, len(pods))


def _read_pod_state() -> dict[str, Any]:
    if not POD_STATE_FILE.exists():
        raise FileNotFoundError(
            f"No pod state file at {POD_STATE_FILE} — run 'provision' first"
        )
    raw: Any = json.loads(POD_STATE_FILE.read_text())
    # Tolerate the previous bare-list format so a state file written by an
    # older run can still be read to tear its pods down.
    if isinstance(raw, list):
        return {"pods": raw, "meta": {}}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{POD_STATE_FILE} holds {type(raw).__name__}, not a pod-state "
            f"object. Refusing to read pod identity from it."
        )
    state: dict[str, Any] = raw
    return state


def _load_pod_state() -> list[dict[str, Any]]:
    pods: list[dict[str, Any]] = _read_pod_state()["pods"]
    for pod in pods:
        # These came from a remote API and are about to be interpolated into
        # shell commands. Re-validate on the way out of storage as well.
        pod["ip"], pod["port"] = validate_ssh_endpoint(pod["ip"], int(pod["port"]))
    return pods


def _check_deadline() -> None:
    """Abort if the run has outlived its budgeted wall time."""
    deadline = _read_pod_state().get("meta", {}).get("deadline")
    if deadline and time.time() > deadline:
        raise RuntimeError(
            f"Run exceeded its {MAX_RUN_HOURS}h budget window. Pods are still "
            "up. Collect what finished, then tear down: "
            "python -m scripts.phase1b.runpod_parallel collect && "
            "python -m scripts.phase1b.runpod_parallel teardown"
        )


def _check_budget(gpu_type: str, n_pods: int) -> dict[str, Any]:
    """Refuse to provision a fleet whose worst case exceeds the budget.

    There was no price query, no spend poll and no abort anywhere in this
    module; the ceiling existed only in prose. This converts it into a
    precondition that runs before any pod is created.
    """
    price_per_pod = get_gpu_price(gpu_type)
    fleet_hourly = price_per_pod * n_pods

    # Price the wall the code can actually reach, not the one it intends to.
    # The bounding timeouts are: bootstrap (three _ssh at SSH_DEFAULT_TIMEOUT_S
    # plus one rsync), the fold itself, and a SERIAL collect across pods. A
    # budget check against MAX_RUN_HOURS alone understated the permitted spend
    # by more than a factor of two, which made the gate unreachable: the $12/hr
    # part filter already caps worst case below the $1000 budget, so the check
    # could never fire on any input the caller could produce.
    bootstrap_h = (3 * SSH_DEFAULT_TIMEOUT_S + RSYNC_TIMEOUT_S) / 3600
    fold_h = FOLD_TIMEOUT_S / 3600
    collect_h = n_pods * RSYNC_ATTEMPTS * RSYNC_TIMEOUT_S / 3600
    reachable_h = bootstrap_h + fold_h + collect_h
    worst_case = fleet_hourly * reachable_h

    logger.info("Budget check:")
    logger.info("  %s at $%.2f/hr x %d pods = $%.2f/hr",
                gpu_type, price_per_pod, n_pods, fleet_hourly)
    logger.info("  reachable wall time = %.1fh (bootstrap %.1f + fold %.1f + "
                "serial collect %.1f), declared cap %.1fh",
                reachable_h, bootstrap_h, fold_h, collect_h, MAX_RUN_HOURS)
    logger.info("  worst case = $%.2f against budget $%.2f",
                worst_case, BUDGET_USD)

    if reachable_h > MAX_RUN_HOURS:
        logger.warning(
            "The timeouts permit %.1fh but MAX_RUN_HOURS is %.1fh. The deadline "
            "is the intent; the timeouts are what actually bound spend.",
            reachable_h, MAX_RUN_HOURS,
        )

    if worst_case > BUDGET_USD:
        raise RuntimeError(
            f"Refusing to provision: {n_pods} x {gpu_type} at "
            f"${price_per_pod:.2f}/hr could reach ${worst_case:.2f} over the "
            f"{reachable_h:.1f}h the configured timeouts permit, above the "
            f"${BUDGET_USD:.2f} budget. Raise TRACT_RUNPOD_BUDGET_USD, lower "
            f"the timeouts, or pick a cheaper part."
        )
    return {
        "gpu_type": gpu_type,
        "usd_per_hour_per_pod": price_per_pod,
        "fleet_usd_per_hour": fleet_hourly,
        "worst_case_usd": worst_case,
        "budget_usd": BUDGET_USD,
        "max_run_hours": MAX_RUN_HOURS,
        "reachable_hours": reachable_h,
    }


def _preflight_tracking() -> None:
    """Fail before any pod exists if the campaign could not be tracked.

    Tracking runs at the very end, after collect and aggregate, and its
    failure is deliberately non-fatal there because the results are already
    safe by that point. The consequence is that an unusable WandB key costs a
    full campaign of GPU time before anyone learns the runs were never going
    to appear. Checking the credential here moves that discovery to before the
    first dollar. TRACT_SKIP_WANDB=1 opts out for a deliberately untracked
    run.
    """
    if os.environ.get("TRACT_SKIP_WANDB", "").strip() == "1":
        logger.warning(
            "TRACT_SKIP_WANDB=1: this campaign will not be tracked in WandB."
        )
        return
    from tract.config import LOFO_WANDB_PROJECT
    from tract.training.tracking import verify_credential

    try:
        viewer = verify_credential()
    except RuntimeError as exc:
        raise RuntimeError(
            f"WandB credential preflight failed, so this campaign could not be "
            f"tracked: {exc}\nFix the key, or set TRACT_SKIP_WANDB=1 to run "
            f"untracked on purpose."
        ) from exc
    # Which workspace the runs land in is worth stating before the spend
    # rather than discovering afterwards in someone else's project.
    logger.info(
        "WandB preflight passed: user=%s entity=%s -> project %s",
        viewer["username"], viewer["entity"], LOFO_WANDB_PROJECT,
    )


def provision(folds: list[str] | None = None) -> list[dict[str, Any]]:
    _preflight_tracking()
    configs = select_pod_configs(folds)
    logger.info("Ranking available GPUs (>= 48GB VRAM, <= $%.2f/hr)...",
                MAX_USD_PER_HOUR_PER_POD)
    # A list, not a single choice. list_available_gpus reports the types that
    # EXIST, not the types with free instances, so the best candidate can fail
    # at create time with "no instances currently available". That happened
    # two pods into a five-pod fleet and ended the campaign; it is a transient
    # supply condition and should cost a different GPU, not the run.
    candidates = rank_available_gpus(
        min_vram_gb=48, max_usd_per_hour=MAX_USD_PER_HOUR_PER_POD,
    )
    logger.info("Candidates: %s",
                ", ".join(f"{g} (${p:.2f}/hr)" for g, p in candidates[:5]))
    gpu_type = candidates[0][0]
    # Priced on the most expensive candidate that could actually be used, so
    # the budget check cannot be passed by a cheap first choice and then
    # silently exceeded by the fallback.
    budget = _check_budget(max(candidates, key=lambda c: c[1])[0], len(configs))

    # Record the intent BEFORE creating anything. create_pods_parallel returns
    # only once every pod is up, so a partial failure previously left running
    # pods with no local record and nothing to terminate them by.
    _save_pod_state([], meta={
        "budget": budget,
        "started_at": time.time(),
        "state": "provisioning",
        "requested": [c["name"] for c in configs],
    })

    pods = None
    exhausted: list[str] = []
    for gpu_type, price in candidates:
        try:
            logger.info("Creating %d pods on %s ($%.2f/hr)...",
                        len(configs), gpu_type, price)
            pods = create_pods_parallel(
                configs, gpu_type, image=DOCKER_IMAGE,
                volume_gb=50,
                # ~12GB of site-packages plus a pip cache and HF downloads.
                # 20GB left no headroom and the failure mode is a fold dying
                # late.
                container_disk_gb=60,
            )
            break
        except Exception as exc:
            if is_capacity_error(exc):
                # create_pods_parallel has already terminated whatever came
                # up, so there is nothing billing and the next type is free
                # to try.
                logger.warning("%s has no free capacity; trying the next type.",
                               gpu_type)
                exhausted.append(gpu_type)
                continue
            logger.error(
                "Provisioning failed for a reason other than capacity. Any "
                "pods that were created are ORPHANED and still billing. Reap "
                "them: python -m scripts.phase1b.runpod_parallel reap --confirm"
            )
            raise

    if pods is None:
        raise RuntimeError(
            f"Every acceptable GPU type is out of capacity: {exhausted}. "
            f"Nothing was left running. Retry later, or raise "
            f"TRACT_RUNPOD_MAX_HOURLY to widen the candidate set."
        )

    _save_pod_state(pods, meta={
        "budget": budget,
        "started_at": time.time(),
        "state": "running",
        "deadline": time.time() + MAX_RUN_HOURS * 3600,
    })
    logger.info("All %d pods provisioned and SSH-ready.", len(pods))
    return pods


def _bootstrap_pod(pod: dict[str, Any]) -> None:
    ip, port, role = pod["ip"], pod["port"], pod["role"]
    logger.info("Bootstrapping pod for fold '%s' (%s:%d)...", role, ip, port)

    _ssh(ip, port, "apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1", check=False)

    _rsync_to(ip, port, f"{PROJECT_ROOT}/", "/workspace/tract/")

    # --break-system-packages is required, not sloppy. The pod image ships a
    # Debian-packaged Python 3.12, which is PEP 668 "externally managed", so a
    # plain `pip install` refuses outright and the bootstrap fails before any
    # training starts. On a rented single-purpose container the system
    # interpreter IS the environment and the host is destroyed at teardown, so
    # there is no system to protect; building a venv would only add a layer
    # and a PATH to get wrong. Passed via PIP_BREAK_SYSTEM_PACKAGES so it
    # applies to the transitive pip calls too, and pinned installs still come
    # from requirements-train.txt.
    _ssh(ip, port, (
        "export PIP_BREAK_SYSTEM_PACKAGES=1 && "
        "mkdir -p /workspace/.cache/huggingface && "
        "cd /workspace/tract && "
        "pip install --quiet -e '.[phase0]' && "
        "pip install --quiet -r requirements-train.txt"
    ))

    # Fetch the base model once, here, rather than inside the fold. A 429 at
    # this point costs a bootstrap; the same 429 twenty minutes into training
    # costs the fold. It also proves the read-only token works before any
    # GPU time is spent on it.
    _ssh(ip, port, (
        "cd /workspace/tract && python -c "
        "'from huggingface_hub import snapshot_download; "
        "p = snapshot_download(\"BAAI/bge-large-en-v1.5\"); "
        "print(f\"base model cached at {p}\")'"
    ), env=_get_pod_env())

    # Fatal, not advisory. This probe used to run with check=False while
    # tract/training/loop.py sets fp16=torch.cuda.is_available(): a driver
    # mismatch therefore degraded silently to CPU, and the first symptom would
    # have been five GPUs billing for an hour while every fold timed out.
    _ssh(ip, port, (
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
        "cd /workspace/tract && python -c "
        "'import torch, sentence_transformers, peft, transformers; "
        "assert torch.cuda.is_available(), \"CUDA unavailable: this pod would "
        "train on CPU\"; "
        "print(f\"torch={torch.__version__} cuda={torch.version.cuda} "
        "gpu={torch.cuda.get_device_name(0)} tf={transformers.__version__} "
        "st={sentence_transformers.__version__} peft={peft.__version__}\")'"
    ))

    logger.info("Bootstrap complete for fold '%s'", role)


def _run_fold_on_pod(
    pod: dict[str, Any], config_name: str, arm_flags: tuple[str, ...] = (),
) -> dict[str, Any]:
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
        + "".join(f" {flag}" for flag in arm_flags)
    )

    # Detach the fold from the SSH session that starts it. Previously the
    # training ran in the foreground of a single long-lived SSH connection, so
    # a laptop sleeping, a wifi change, or any transient network drop killed
    # the process on the pod and lost an hour of paid GPU time that had
    # already produced most of a result. setsid plus nohup means the fold
    # outlives its shell, and the orchestrator polls for the sentinel instead
    # of holding a connection open.
    remote_dir = "/workspace/tract"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", framework)
    log_path = f"{remote_dir}/fold_{slug}.log"
    exit_path = f"{remote_dir}/fold_{slug}.exit"
    launch = (
        f"cd {remote_dir} && rm -f {shlex.quote(exit_path)} && "
        f"setsid nohup bash -c {shlex.quote(fold_cmd + f'; echo $? > {exit_path}')} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo started"
    )

    logger.info("[%s] Launching fold (detached)...", framework)
    start = time.time()
    env = _get_pod_env()
    try:
        _ssh(ip, port, launch, env=env, timeout=SSH_DEFAULT_TIMEOUT_S)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] LAUNCH FAILED after %.1fm: %s", framework, elapsed / 60, e)
        return {"fold": framework, "status": "failed",
                "error": f"launch: {e}", "elapsed_s": elapsed}

    # Poll for the sentinel. A dropped poll is retried rather than fatal: the
    # fold is still running on the pod either way, and treating a transient
    # SSH error as a fold failure is what this change exists to prevent.
    consecutive_poll_errors = 0
    while True:
        elapsed = time.time() - start
        if elapsed > FOLD_TIMEOUT_S:
            logger.error("[%s] TIMEOUT after %.1fm", framework, elapsed / 60)
            return {"fold": framework, "status": "failed",
                    "error": f"exceeded FOLD_TIMEOUT_S={FOLD_TIMEOUT_S}s",
                    "elapsed_s": elapsed}
        time.sleep(FOLD_POLL_INTERVAL_S)
        try:
            probe = _ssh(
                ip, port,
                f"cat {shlex.quote(exit_path)} 2>/dev/null || echo RUNNING",
                check=False, timeout=FOLD_POLL_SSH_TIMEOUT_S,
            )
            consecutive_poll_errors = 0
        except Exception as e:  # noqa: BLE001 - transient network, keep polling
            consecutive_poll_errors += 1
            if consecutive_poll_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                logger.error("[%s] UNREACHABLE after %d polls: %s",
                             framework, consecutive_poll_errors, e)
                return {"fold": framework, "status": "failed",
                        "error": f"pod unreachable: {e}", "elapsed_s": elapsed}
            logger.warning("[%s] poll %d/%d failed (%s); fold continues on the pod",
                           framework, consecutive_poll_errors,
                           MAX_CONSECUTIVE_POLL_ERRORS, e)
            continue

        marker = (probe.stdout or "").strip().splitlines()
        status = marker[-1] if marker else "RUNNING"
        if status == "RUNNING":
            continue

        elapsed = time.time() - start
        if status == "0":
            logger.info("[%s] COMPLETE in %.1fm", framework, elapsed / 60)
            return {"fold": framework, "status": "ok", "elapsed_s": elapsed}

        # Bring back the tail of the remote log; without it the operator has
        # only an exit code and has to SSH in to find out what happened.
        tail = ""
        try:
            tail = (_ssh(ip, port, f"tail -n 40 {shlex.quote(log_path)}",
                         check=False, timeout=FOLD_POLL_SSH_TIMEOUT_S).stdout or "")
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        logger.error("[%s] FAILED (exit %s) after %.1fm:\n%s",
                     framework, status, elapsed / 60, tail[-2000:])
        return {"fold": framework, "status": "failed",
                "error": f"exit={status}: {tail[-500:]}", "elapsed_s": elapsed}


def run_folds(
    config_name: str = "phase1b_primary",
    arm_flags: tuple[str, ...] = (),
) -> list[str]:
    """Run every fold on its pod. Returns the roles that FAILED.

    The return value used to be None and fold failures were logged and
    dropped. full_pipeline then collected whatever existed, set
    results_are_safe, and tore the fleet down -- so two failed folds out of
    five destroyed the pods and the failure surfaced later, at aggregation,
    with nothing left to retry on.
    """
    pods = _load_pod_state()
    _check_deadline()

    logger.info("Bootstrapping %d pods in parallel...", len(pods))
    # Isolate failures. list(ex.map(...)) re-raises the first exception and
    # abandons the rest, so one bad pod aborted the whole fleet while the other
    # four kept billing.
    bootstrap_errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pods)) as ex:
        bootstrap_futures = {
            ex.submit(_bootstrap_pod, pod): pod["role"] for pod in pods
        }
        for future in concurrent.futures.as_completed(bootstrap_futures):
            role = bootstrap_futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - collect all, decide after
                logger.error("Bootstrap FAILED for '%s': %s", role, exc)
                bootstrap_errors[role] = str(exc)

    if bootstrap_errors:
        # Every fold is needed for the aggregate, so a partial fleet cannot
        # produce the number. Stop before paying for training on it.
        raise RuntimeError(
            f"{len(bootstrap_errors)} of {len(pods)} pods failed to bootstrap: "
            f"{sorted(bootstrap_errors)}. Every fold is required for the "
            f"aggregate, so training on the rest would buy nothing. Pods are "
            f"still up; tear down with 'teardown' or investigate first. "
            f"Errors: {bootstrap_errors}"
        )
    logger.info("All %d pods bootstrapped.", len(pods))

    logger.info("=" * 60)
    logger.info("RUNNING 5 FOLDS IN PARALLEL")
    logger.info("=" * 60)

    start = time.time()
    fold_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pods)) as ex:
        fold_futures = {
            ex.submit(_run_fold_on_pod, pod, config_name, arm_flags): pod["role"]
            for pod in pods
        }
        for f in concurrent.futures.as_completed(fold_futures):
            role = fold_futures[f]
            result = f.result()
            fold_results.append(result)
            logger.info("  [%s] %s (%.1fm)", role, result["status"], result["elapsed_s"] / 60)
            # The deadline was checked once, before the first fold started,
            # and never again, so MAX_RUN_HOURS bounded nothing once the
            # fleet was running. Report every time a fold lands; the in-flight
            # folds are already paid for, so this warns rather than aborts.
            try:
                _check_deadline()
            except RuntimeError as exc:
                logger.error("DEADLINE EXCEEDED mid-run: %s", exc)

    elapsed = time.time() - start
    ok = sum(1 for r in fold_results if r["status"] == "ok")
    failed_roles = [r["fold"] for r in fold_results if r["status"] == "failed"]
    logger.info("=" * 60)
    logger.info("ALL FOLDS COMPLETE: %d OK, %d FAILED in %.1fm",
                ok, len(failed_roles), elapsed / 60)
    for r in fold_results:
        if r["status"] == "failed":
            logger.error("  FAILED: %s — %s", r["fold"], r.get("error", "")[:200])
    logger.info("=" * 60)
    return failed_roles


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


def aggregate(
    config_name: str = "phase1b_primary",
    folds: list[str] | None = None,
) -> dict[str, Any]:
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
    # An explicitly scoped run declares its own fold set; anything else must
    # match the full LOFO set, so a partial cross-validation cannot be
    # aggregated into a headline number by omission.
    fold_results = load_fold_results(
        local_results, expected_frameworks=set(folds) if folds else None,
    )
    if folds:
        logger.warning(
            "Aggregating a SCOPED run over %s. This is not a LOFO result and "
            "must not be reported as one.", sorted(folds),
        )
    logger.info("Aggregating %d folds from %s", len(fold_results), local_results)

    record = {
        "config_name": config_name,
        # Marks the record itself, so a scoped run cannot be mistaken for a
        # full cross-validation once it is read back out of the file.
        "scoped_to_folds": sorted(folds) if folds else None,
        "aggregate_hit1": aggregate_fold_results(fold_results),
        "gate": gate_decision(fold_results),
        "per_fold": {
            r["held_out_framework"]: r["metrics"] for r in fold_results
        },
    }
    atomic_write_json(record, local_results / "aggregate_metrics.json")
    logger.info("Wrote %s", local_results / "aggregate_metrics.json")
    return record


def track(config_name: str = "phase1b_primary") -> int:
    """Log every collected fold to WandB from this machine.

    Tracking runs here rather than on the pods on purpose. _get_pod_env ships
    no credentials, and a WandB API key is account-wide: handing one to five
    rented hosts to get live loss curves puts a reusable credential on machines
    the operator does not control, and buys telemetry that the collected fold
    records already contain. The trade is live streaming for credential
    containment, and containment wins.

    Idempotent by run id. WandB names are not unique keys -- wandb.init makes
    a fresh run on every call whatever the name -- so this keys each run on
    (config_name, arm, framework). Re-running after a partial collect updates
    the folds already logged and adds the missing ones, instead of leaving two
    populations of the same experiment in the project.
    """
    from tract.config import LOFO_WANDB_ENTITY, LOFO_WANDB_PROJECT
    from tract.io import load_json
    from tract.training.orchestrate import FOLD_RESULT_FILENAME
    from tract.training.tracking import (
        finish_run,
        init_run,
        log_fold,
        stable_run_id,
    )

    local_results = RESULTS_DIR / config_name
    paths = sorted(local_results.glob(f"fold_*/{FOLD_RESULT_FILENAME}"))
    if not paths:
        logger.error(
            "No %s under %s. Run 'collect' first; there is nothing to track.",
            FOLD_RESULT_FILENAME, local_results,
        )
        return 1

    logged = 0
    for path in paths:
        record = load_json(path)
        config = record.get("config") or {}
        arm = _arm_from_config(config)
        framework = record.get("held_out_framework", path.parent.name)
        run = init_run(
            project=LOFO_WANDB_PROJECT,
            entity=LOFO_WANDB_ENTITY,
            name=f"{arm}/{framework}",
            config={
                **config,
                "held_out_framework": framework,
                "arm": arm,
                "config_name": config_name,
                "git_sha": record.get("git_sha"),
                **{f"inputs/{k}": v for k, v in (record.get("inputs") or {}).items()},
            },
            tags=[arm, framework, "lofo"],
            run_id=stable_run_id(config_name, arm, framework),
        )
        log_fold(run, record)
        finish_run(run)
        logged += 1

    logger.info("Logged %d folds to WandB project %s", logged, LOFO_WANDB_PROJECT)
    return 0


def _arm_from_config(config: dict[str, Any]) -> str:
    """Recover the arm label from a fold record's config block.

    Kept in step with run_fold._arm_label by a test rather than by discipline:
    two labels for one arm would split a campaign across two names in the UI.
    """
    if not config.get("use_prose", True):
        return "title-only"
    parts = ["prose"]
    if config.get("use_description_only"):
        parts.append("desconly")
    if config.get("use_stopword_filter"):
        parts.append("stopwords")
    return "-".join(parts)


def teardown() -> None:
    """Terminate only the pods this run created.

    Previously called terminate_all(), which kills every running pod on the
    account including work belonging to someone else. It also deleted the state
    file unconditionally, so a pod that failed to terminate lost its only local
    record and kept billing unnoticed.
    """
    state = _read_pod_state()
    pod_ids = [p["pod_id"] for p in state["pods"] if p.get("pod_id")]
    if not pod_ids:
        logger.warning("No pod ids in %s; nothing scoped to terminate", POD_STATE_FILE)
        return

    logger.info("Terminating %d pods from this run...", len(pod_ids))
    failed = terminate_pods(pod_ids)

    if failed:
        # Keep the record. It is the only thing that names what is still up.
        state["meta"]["state"] = "teardown_failed"
        state["meta"]["still_running"] = failed
        _save_pod_state(state["pods"], state["meta"])
        raise RuntimeError(
            f"{len(failed)} pod(s) did not terminate and are still billing: "
            f"{failed}. State kept at {POD_STATE_FILE}."
        )

    POD_STATE_FILE.unlink()
    logger.info("All %d pods terminated.", len(pod_ids))


def reap(confirm: bool = False) -> None:
    """Terminate every pod named by this run's state file.

    Nothing reaps pods if the orchestrating process dies: the RunPod create
    payload carries no server-side TTL, so a killed orchestrator leaves the
    fleet billing until someone notices. This is the recovery path, and it
    works from the state file alone.
    """
    try:
        state = _read_pod_state()
    except FileNotFoundError:
        logger.info("No state file; nothing to reap from this run.")
        return

    pods = state["pods"]
    meta = state.get("meta", {})
    logger.info("State file records %d pod(s), state=%s, started_at=%s",
                len(pods), meta.get("state"), meta.get("started_at"))
    for pod in pods:
        logger.info("  %s (%s) role=%s",
                    pod.get("pod_id"), pod.get("name"), pod.get("role"))

    # The state file records pods=[] between "intent to provision" and "all pods
    # up". That is exactly the window a crash during provisioning lands in, and
    # exactly when reap is reached for. Terminating an empty list and reporting
    # "reaped cleanly" would hand back a false all-clear in the one case this
    # command exists for. Fall back to matching the account's running pods by
    # the deterministic names in POD_CONFIGS.
    known_ids = {p["pod_id"] for p in pods if p.get("pod_id")}
    expected_names = {c["name"] for c in POD_CONFIGS}
    orphans = [
        p for p in get_running_pods()
        if p.get("name") in expected_names and p.get("id") not in known_ids
    ]
    if orphans:
        logger.warning(
            "Found %d running pod(s) matching this run's names but absent from "
            "the state file: %s", len(orphans),
            [(p.get("id"), p.get("name")) for p in orphans],
        )

    targets = sorted(known_ids | {p["id"] for p in orphans})
    if not targets:
        logger.info("No pods from this run are running. Nothing to reap.")
        POD_STATE_FILE.unlink(missing_ok=True)
        return

    if not confirm:
        logger.warning("Dry run: would terminate %s. Re-run with --confirm.", targets)
        return

    failed = terminate_pods(targets)
    if failed:
        raise RuntimeError(
            f"{len(failed)} pod(s) did not terminate and are still billing: {failed}"
        )
    POD_STATE_FILE.unlink(missing_ok=True)
    logger.info("Reaped %d pod(s) cleanly.", len(targets))


def full_pipeline(
    config_name: str = "phase1b_primary",
    arm_flags: tuple[str, ...] = (),
    folds: list[str] | None = None,
) -> None:
    logger.info("=" * 60)
    logger.info("PHASE 1B PARALLEL FOLD EXECUTION")
    logger.info("=" * 60)
    start = time.time()

    # provision(); run_folds(); collect(); teardown() used to be a bare
    # sequence. Any exception between the first and last call orphaned the
    # whole fleet, which then billed until someone noticed. The sibling
    # scripts/phase0/runpod_orchestrate.py already did this correctly.
    results_are_safe = False
    try:
        # provision() was outside this try, so the stage with the highest
        # orphan rate was the one the finally never covered.
        provision(folds)
        failed_folds = run_folds(config_name, arm_flags)
        # Collect BEFORE reacting to failures: the folds that did succeed are
        # already paid for and their indicators live only on the pods.
        uncollected = collect(config_name)
        if uncollected:
            # Every fold that did not come back is a paid-for GPU hour whose
            # only copy of the per-item indicators is still on the pod.
            raise RuntimeError(
                f"Results were not collected from {uncollected}. Retry with "
                f"'collect' before tearing anything down."
            )
        if failed_folds:
            # A partial fold set cannot produce a LOFO number, and tearing the
            # fleet down here would mean re-provisioning and re-bootstrapping
            # to retry a fold whose pod is still warm and still has the model
            # cached.
            raise RuntimeError(
                f"{len(failed_folds)} fold(s) failed: {sorted(failed_folds)}. "
                f"The successful folds have been collected. Pods are still up "
                f"so the failures can be retried on them:\n"
                f"  retry : python -m scripts.phase1b.runpod_parallel run "
                f"--config-name {config_name}\n"
                f"  or end: python -m scripts.phase1b.runpod_parallel teardown"
            )
        results_are_safe = True
    finally:
        if results_are_safe:
            teardown()
        else:
            # Do not destroy results that were paid for and never retrieved.
            # Leaving pods up costs money; terminating them costs the run.
            logger.error(
                "Pipeline did not reach a state where results are safely "
                "collected. Pods have been LEFT RUNNING and are still billing "
                "so the results can be recovered.\n"
                "  retrieve : python -m scripts.phase1b.runpod_parallel collect\n"
                "  then     : python -m scripts.phase1b.runpod_parallel teardown\n"
                "  give up  : python -m scripts.phase1b.runpod_parallel reap --confirm"
            )

    elapsed = time.time() - start
    logger.info("Total pipeline time: %.1fm", elapsed / 60)

    aggregate(config_name, folds)
    # Tracking is last and non-fatal. The fold records are on disk and
    # aggregated by this point, so a WandB outage must not fail a run whose
    # results are already safe. `track` re-runs cleanly on its own.
    try:
        track(config_name)
    except Exception as exc:
        logger.error(
            "WandB logging failed: %s. Results are collected and aggregated; "
            "re-run `python -m scripts.phase1b.runpod_parallel track "
            "--config-name %s` once it is resolved.", exc, config_name,
        )


def main() -> int:
    """Return a non-zero exit code on failure.

    Every action used to return 0 regardless. The recovery runbook this module
    prints tells the operator to run `collect` and then `teardown`; with a
    silent exit, a partially failed collect looked identical to a clean one and
    the next command destroyed the pods holding the missing results.
    """
    parser = argparse.ArgumentParser(description="Phase 1B RunPod parallel fold executor")
    parser.add_argument("action", nargs="?", default="full",
                        choices=["full", "provision", "run", "collect", "aggregate",
                                 "track", "teardown", "reap", "price"])
    parser.add_argument("--config-name", type=str, default="phase1b_primary",
                        help="Experiment config name")
    parser.add_argument("--confirm", action="store_true",
                        help="Required by 'reap' to actually terminate pods")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated subset of folds to provision and "
                             "run, for a canary. Omit for the full LOFO set. "
                             "A scoped aggregate is labelled as such and is "
                             "not a cross-validation result.")
    # The arm has to reach the pod. Without these the orchestrator ran the
    # prose arm whatever --config-name said, so a three-arm ablation would have
    # been three identical runs in three differently named directories.
    parser.add_argument("--no-prose", action="store_true",
                        help="Baseline arm: anchor on section titles")
    parser.add_argument("--stopwords", action="store_true",
                        help="Ablation arm: filter corpus-derived boilerplate")
    parser.add_argument("--description-only", action="store_true",
                        help="Ablation arm: cut each control at its first "
                             "remediation heading")
    args = parser.parse_args()

    folds = (
        [f.strip() for f in args.folds.split(",") if f.strip()]
        if args.folds else None
    )

    arm_flags = tuple(
        flag for flag, on in (("--no-prose", args.no_prose),
                              ("--stopwords", args.stopwords),
                              ("--description-only", args.description_only)) if on
    )
    if arm_flags and args.config_name == "phase1b_primary":
        # Arms must not share a results directory: fold records carry no arm,
        # so a mixed directory aggregates into a number describing no single
        # configuration.
        raise SystemExit(
            f"Refusing to run arm {arm_flags} into the default results "
            f"directory. Pass a distinct --config-name."
        )

    if args.action == "full":
        full_pipeline(args.config_name, arm_flags, folds)
    elif args.action == "provision":
        provision(folds)
    elif args.action == "run":
        # The canary exposed this: the fold failed on a HuggingFace 429 and
        # `run` still exited 0, so an unattended wrapper checking $? would
        # have moved on to collect and teardown on a fleet that produced
        # nothing. Same class as the failure full_pipeline already guards.
        failed = run_folds(args.config_name, arm_flags)
        if failed:
            logger.error(
                "%d fold(s) failed: %s. Pods are still up; fix and re-run "
                "'run', or 'teardown' to stop billing.",
                len(failed), sorted(failed),
            )
            return 1
    elif args.action == "collect":
        uncollected = collect(args.config_name)
        if uncollected:
            logger.error(
                "Collection incomplete for %s. Do NOT tear down: those pods hold "
                "the only copy of their fold.", uncollected,
            )
            return 1
    elif args.action == "aggregate":
        aggregate(args.config_name, folds)
    elif args.action == "track":
        return track(args.config_name)
    elif args.action == "teardown":
        teardown()
    elif args.action == "reap":
        reap(confirm=args.confirm)
    elif args.action == "price":
        # Dry-run the budget gate without creating anything.
        gpu_type = find_fastest_available(
            min_vram_gb=48, max_usd_per_hour=MAX_USD_PER_HOUR_PER_POD,
        )
        _check_budget(gpu_type, len(select_pod_configs(folds)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Anything that reaches here left pods running or results uncollected.
        logger.exception("FAILED. Check for running pods: reap --confirm")
        sys.exit(1)
