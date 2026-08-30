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
TRACT_RUNPOD_SSH_KEY. Set the budget explicitly for any real campaign; the
default is a backstop, not a plan.
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

from tract.config import FOLD_RESULT_FILENAME, PHASE1B_BASE_MODEL, PROCESSED_DIR
# Lightweight on purpose: data_quality pulls neither torch nor datasets, so the
# operator's machine can enforce the corpus gate without a training stack.
from tract.training.data_quality import assert_corpus_matches_training_links
from tract.io import atomic_write_json, load_json
from scripts.phase0.runpod_provision import (
    PRICE_CLOUD_TYPE,
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
# Owner-only. The file holds every pod's live IP and SSH port, so a mode any
# wider hands the fleet's address list to every account on this machine.
POD_STATE_MODE: Final[int] = 0o600
# Where the bytes go when the state file will not parse. reap unlinks the state
# file when it finishes, and the unreadable bytes may still be the only record
# of a pod its name sweep cannot see.
POD_STATE_CORRUPT_SUFFIX: Final[str] = ".corrupt"

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

# The interpreter inside DOCKER_IMAGE, which is what every pin in
# requirements-train.txt has to install on. The digest above is
# 1.1.0-cu1300-torch291-ubuntu2404-cluster, and Ubuntu 24.04's system python3
# is 3.12; the 2026-08-14 canary confirmed it by tripping PEP 668's
# externally-managed guard, which is why _bootstrap_pod exports
# PIP_BREAK_SYSTEM_PACKAGES. Kept beside the digest on purpose: the two facts
# change together, because the Python is a property of the image.
POD_PYTHON_VERSION: Final[str] = "3.12"

# Budget controls. The $1000 ceiling was prose; these make it a gate.
# Owner-set authorization ceiling, lowered from 2000 to 1000 on 2026-08-26.
# This is the BACKSTOP, not the plan: it bounds what the code permits when
# nobody exported anything. Campaign 2 exports 200, which is the tight
# per-run bound P5 asks for, and an export always wins over this default.
BUDGET_USD: Final[float] = float(os.environ.get("TRACT_RUNPOD_BUDGET_USD", "1000"))
# Refuse a part whose rate would burn the budget faster than the run can finish.
MAX_USD_PER_HOUR_PER_POD: Final[float] = float(
    os.environ.get("TRACT_RUNPOD_MAX_HOURLY", "12")
)
# Folds are expected in well under this; it is the wall the watchdog enforces,
# and _check_deadline ABORTS the run when it passes. It must therefore exceed
# bootstrap + one fold, or a fleet is provisioned that cannot possibly finish --
# see the coherence check in _check_budget. Raised from 6 alongside
# FOLD_TIMEOUT_S for the long-context rebaseline; at a 4h fold ceiling, 6 left
# only 1.57h of slack over bootstrap plus one fold, which a single retry
# consumes.
MAX_RUN_HOURS: Final[float] = float(os.environ.get("TRACT_RUNPOD_MAX_HOURS", "8"))
# A campaign runs several arms on one fleet, each getting its own window. The
# cap bounds the total: without it, "extend per arm" is no bound at all.
MAX_DEADLINE_EXTENSIONS: Final[int] = int(
    os.environ.get("TRACT_RUNPOD_MAX_ARMS", "12")
)

# rsync's --timeout is an I/O-IDLE timer, not a wall clock: it fires when no
# bytes have moved for this long, whatever the process is nominally doing. Both
# directions carry it. The two walls below bound the process itself, and they
# are separate because the two directions move payloads three orders of
# magnitude apart.
RSYNC_IDLE_TIMEOUT_S: Final[int] = 120

# The PUSH is the working tree: 61,327,794 bytes across 560 files, measured on
# 2026-08-28 with the exclude list _rsync_to builds. 300s is a 200 KB/s floor
# for that payload. It used to borrow the pull's 1800s wall, which on 61MB
# tolerates 34 KB/s -- around 250x slower than a healthy link -- and it carried
# no idle timer at all, so a stalled-but-alive transfer was bounded by nothing
# shorter than that wall. On 2026-08-27 the two omissions together cost 90
# minutes and a fleet: a pod sat with 82MB present and moved ZERO bytes across
# a sampled 45-second window while the push ran its full wall, three times, and
# the four other pods sat at run_folds' bootstrap barrier until the campaign
# aborted without launching a single fold.
#
# Two attempts, not the pull's three. Re-sending the tree is idempotent and a
# retry is cheap, but a third wall buys a third case of the same wedge.
RSYNC_PUSH_TIMEOUT_S: Final[int] = 300
RSYNC_PUSH_ATTEMPTS: Final[int] = 2
PUSH_PAYLOAD_BYTES: Final[int] = 61_327_794

# The PULL is where the gigabytes argument belongs, and it is the one step
# whose failure destroys work that has already been paid for: a fold's per-item
# indicators exist only on the pod until this succeeds. Hence the long wall and
# the third attempt.
RSYNC_PULL_TIMEOUT_S: Final[int] = 1800
RSYNC_PULL_ATTEMPTS: Final[int] = 3
RSYNC_PULL_BACKOFF_S: Final[int] = 10

# One fold: LoRA training plus a paired zero-shot pass.
#
# Raised from 7200 for the anchor-budget rebaseline. 7200 was calibrated on
# Campaign 2's 60-78 minute folds at max_seq_length=512. Attention is quadratic
# in sequence length and padding is not free: at 512 a batch of 32 occupies
# 16,384 slots, at 1,024 it occupies 32,768, so a 1,024-token arm runs roughly
# twice the compute per step and four of the five test folds would cross the old
# ceiling. The failure mode is the expensive one -- exceeding this abandons the
# fold while the DETACHED trainer keeps running and keeps billing, so the run
# loses its result and pays for it anyway.
#
# 14400 (4h) covers a 1,024-token fold -- roughly twice Campaign 2's 60-78
# minutes -- with better than 50% margin. It is deliberately NOT sized for
# 2,048: this constant is priced by _check_budget and multiplied by the fleet,
# so an 8h ceiling put the worst case at $958 against the $600 budget
# tests/test_runpod_safety.py holds it to. A 2,048-token arm needs its own
# raise and its own budget conversation, which is the right place for that
# trade to be visible.
FOLD_TIMEOUT_S: Final[int] = 14400
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
# Bootstrap's longest step is the pip install plus a 1.3GB model fetch, which
# takes 2-4 minutes on a healthy pod. The default hour was the wrong ceiling
# for it: a pod whose SSH session hung silently -- reachable for new
# connections, but with the original one a zombie -- blocked the whole fleet's
# bootstrap barrier for an hour before anything noticed, because the retry
# only fires on a returncode and a hang never produces one. Fifteen minutes is
# generous for the work and short enough that a hang becomes a retry.
SSH_BOOTSTRAP_TIMEOUT_S: Final[int] = 900

# The fold launch detaches a trainer and echoes. It should return in under a
# second; a minute is already pathological. It used to be given
# SSH_DEFAULT_TIMEOUT_S -- 3600 -- and _ssh retries on TimeoutExpired, so on
# 2026-08-28 a fold that outran the hour got a SECOND trainer on the same GPU.
# A fire-and-forget command must never share a wall clock sized for work.
SSH_LAUNCH_TIMEOUT_S: Final[int] = 120
SSH_CONNECT_ATTEMPTS: Final[int] = 4
SSH_RETRY_BACKOFF_S: Final[int] = 15

# One pod's bootstrap, end to end, as a wall clock. The work under it is
# apt-get, a 61MB push, a pip install, a 1.3GB model fetch and a CUDA probe:
# two to four minutes on a healthy pod. Twenty-five minutes is roughly five
# times that, and still an order of magnitude tighter than the ladder it
# replaces -- four SSH steps at four connect attempts each, plus the push, is
# 4.27h per pod, and was 5.61h before the push got its own wall. Every minute
# of that is fleet time, not pod time: run_folds bootstraps inside a
# ThreadPoolExecutor context manager, so the whole fleet waits at the barrier
# for the slowest pod.
#
# Enforced COOPERATIVELY, by _clamp_to_deadline shortening each step until a
# thread that has run out of budget ends itself. Nothing outside the thread
# can end it: the executor's context manager joins on exit, cancel() cannot
# touch a future that has already started -- and max_workers == len(pods), so
# every future has -- and shutdown(wait=False, cancel_futures=True) leaves the
# process hanging at interpreter exit instead. The premortem tested all three.
BOOTSTRAP_DEADLINE_S: Final[int] = 1500
# Cooperative means the deadline can be overrun by a wait that began just
# before it expired. The longest of those is the last SSH backoff.
BOOTSTRAP_DEADLINE_SLACK_S: Final[int] = (
    SSH_RETRY_BACKOFF_S * (SSH_CONNECT_ATTEMPTS - 1)
)
# How many _ssh calls _bootstrap_pod issues. A test pins this against the
# function's own source, because _check_budget priced three of them for as
# long as there have been four.
BOOTSTRAP_SSH_STEPS: Final[int] = 4

# The validation split holds out traditional-security frameworks instead of
# AI ones, so arm selection happens on 1,265 items rather than the test set's
# 147 -- where the minimum detectable effect is 11.4 hit@1 points against
# effects of 1-3. The five largest, so one fleet covers the split.
VALIDATION_FOLD_FRAMEWORKS: Final[list[str]] = [
    "CAPEC",
    "NIST 800-53 v5",
    "ASVS",
    "CWE",
    "ISO 27001",
]

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


def fold_roster(split: str = "test") -> list[str]:
    """Which frameworks are held out for this split."""
    return (
        list(FOLD_FRAMEWORKS) if split == "test"
        else list(VALIDATION_FOLD_FRAMEWORKS)
    )


def select_pod_configs(
    folds: list[str] | None = None, split: str = "test",
) -> list[dict[str, str]]:
    """Pod configs for a subset of folds, preserving the canonical names.

    Exists so a canary is a supported operation rather than a hand-edit of
    POD_CONFIGS. Before this, provision() always created all five pods, so the
    only way to validate the machinery end to end was to pay for the whole
    fleet -- which is the opposite of what a canary is for.

    Pod names stay tied to the framework's index in FOLD_FRAMEWORKS, not to
    its position in the filtered list, so a canary pod and the same fold in a
    later full run carry the same name and `reap` recognises both.
    """
    roster = fold_roster(split)
    # Pod names stay tied to position in the roster so reap's orphan sweep
    # recognises them, and the two splits never share a name.
    prefix = "tract-p1b" if split == "test" else "tract-p1b-val"
    configs = [
        {"name": f"{prefix}-fold{i}", "role": fw} for i, fw in enumerate(roster)
    ]
    if not folds:
        return configs
    unknown = [f for f in folds if f not in roster]
    if unknown:
        raise ValueError(
            f"Unknown fold(s) {unknown} for split {split!r}. "
            f"Expected any of {roster}."
        )
    wanted = set(folds)
    return [c for c in configs if c["role"] in wanted]


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


# ssh exits 255 for every transport-level failure, so the message decides
# whether retrying is sensible. Anything here is the far end refusing or
# dropping the connection rather than rejecting the caller.
TRANSIENT_SSH_MARKERS: Final[tuple[str, ...]] = (
    "kex_exchange_identification",
    "connection reset by peer",
    "connection closed by remote host",
    "connection refused",
    "connection timed out",
    "operation timed out",
    "broken pipe",
    "no route to host",
)


def _is_transient_ssh_failure(stderr: str) -> bool:
    """Whether an SSH failure is worth retrying.

    Excludes authentication and host-key failures on purpose: those are
    configuration errors that will fail identically every time, and retrying
    them only delays a clear message.
    """
    text = stderr.lower()
    if "permission denied" in text or "host key verification failed" in text:
        return False
    return any(marker in text for marker in TRANSIENT_SSH_MARKERS)


def _clamp_to_deadline(deadline: float, step_timeout: int, step: str) -> int:
    """How long this step may run before the pod's bootstrap deadline bites.

    The cooperative half of BOOTSTRAP_DEADLINE_S. A caller holding an absolute
    monotonic deadline asks this for every timeout it is about to pass to a
    subprocess, so a step that starts late is given only what is left rather
    than its nominal ceiling, and a step with nothing left never starts.

    Args:
        deadline: Absolute time.monotonic() value the bootstrap must end by.
        step_timeout: What this step would be allowed if it were alone.
        step: Named in the failure, so an operator reading a bootstrap error
            knows which stage the pod ran out of budget in.

    Returns:
        Seconds, never below one. int() truncation turns a sub-second remainder
        into zero, and a zero-second timeout is an artifact of rounding rather
        than a decision anyone made.

    Raises:
        TimeoutError: The deadline has already passed. Raised rather than
            returning, because the alternative is issuing a doomed subprocess
            call and then paying the retry ladder's backoff on top of it.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            f"Pod bootstrap deadline exceeded by {-remaining:.0f}s before "
            f"'{step}' could start (budget was {BOOTSTRAP_DEADLINE_S}s). This "
            f"pod is abandoned so the rest of the fleet stops waiting on it."
        )
    return max(1, min(step_timeout, int(remaining)))


def _ssh(
    ip: str, port: int, cmd: str,
    check: bool = True,
    env: dict[str, str] | None = None,
    timeout: int = SSH_DEFAULT_TIMEOUT_S,
    deadline: float | None = None,
) -> subprocess.CompletedProcess[str]:
    _require_ssh_key()
    ip, port = validate_ssh_endpoint(ip, port)
    env_lines = ""
    if env:
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env.items()) + "\n"
    script = env_lines + cmd
    ssh_cmd = f"ssh {SSH_OPTS} -p {port} root@{ip} bash -s"
    logger.info("[ssh %s:%d] %s", ip, port, cmd[:120])

    # Retry connection-level failures. All five pods of a running fleet lost
    # SSH simultaneously with "kex_exchange_identification: Connection reset
    # by peer" -- the transport was refused before authentication, which is a
    # transient condition at the far end (proxy hiccup, sshd MaxStartups under
    # concurrent polling), not a failed command. Without a retry that ended a
    # campaign that had already paid for its pods. Only transport failures are
    # retried: a command that ran and exited non-zero is reported as-is,
    # because re-running it could repeat a side effect.
    #
    # `deadline` clamps EVERY attempt, not just the first. Clamping only the
    # first would leak the whole ladder: four attempts at the caller's ceiling
    # plus 90s of backoff is 4x the budget a caller thought it had handed out.
    for attempt in range(1, SSH_CONNECT_ATTEMPTS + 1):
        attempt_timeout = timeout if deadline is None else _clamp_to_deadline(
            deadline, timeout, f"ssh {cmd[:40]}",
        )
        try:
            result = subprocess.run(
                ssh_cmd, shell=True, input=script, text=True,
                capture_output=True, timeout=attempt_timeout,
            )
        except subprocess.TimeoutExpired:
            # A hung session produces no returncode at all, so the marker
            # check below never sees it. This is the shape that blocked a
            # fleet's bootstrap barrier for an hour.
            if attempt == SSH_CONNECT_ATTEMPTS:
                raise
            logger.warning(
                "[ssh %s:%d] hung for %ds (attempt %d/%d); retrying.",
                ip, port, attempt_timeout, attempt, SSH_CONNECT_ATTEMPTS,
            )
            time.sleep(SSH_RETRY_BACKOFF_S * attempt)
            continue
        if result.returncode != 255 or attempt == SSH_CONNECT_ATTEMPTS:
            break
        if not _is_transient_ssh_failure(result.stderr or ""):
            break
        backoff = SSH_RETRY_BACKOFF_S * attempt
        logger.warning(
            "[ssh %s:%d] transport failed (attempt %d/%d): %s. Retrying in %ds.",
            ip, port, attempt, SSH_CONNECT_ATTEMPTS,
            (result.stderr or "").strip()[-120:], backoff,
        )
        time.sleep(backoff)
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


def _rsync_to(
    ip: str, port: int, local_path: str, remote_path: str,
    deadline: float | None = None,
) -> None:
    """Ship the working tree to a pod, with an idle timer and a short wall.

    Args:
        deadline: Absolute time.monotonic() value the caller's bootstrap must
            end by, if it has one. Each attempt is clamped to what remains, so
            a wedged push cannot spend the budget the steps after it need.
    """
    ip, port = validate_ssh_endpoint(ip, port)
    # The exclude list is what stands between the operator's working tree and
    # five rented hosts, so it has to cover everything .gitignore does. The
    # sharpest omission was .pod_state.json: it is chmod 600 locally precisely
    # because it holds every pod's live IP and SSH port, and it was being copied
    # to all of them, so owning one pod disclosed the address of the rest.
    # '*.tmp' and '.pod_state.json.*' close the same hole for the two files that
    # can now sit BESIDE it and hold the same addresses: the atomic write leaves
    # '..pod_state.json.<rand>.tmp' if the orchestrator is killed mid-write, and
    # reap parks unparseable bytes at '.pod_state.json.corrupt'. Neither name is
    # matched by the exclude that exists for the file they are copies of.
    excludes = " ".join(f"--exclude={pat!r}" for pat in (
        "__pycache__", "*.pyc", ".git", "results", ".mypy_cache", "models",
        "wandb", ".wandb", ".env", "*.db", "data/raw", ".claude", "venv",
        ".venv", ".pod_state.json", ".pod_state.json.*", "*.tmp",
        ".runpod_known_hosts", "build",
        ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache", "*.egg-info",
        ".DS_Store",
    ))
    # --timeout and --partial are the twenty characters this direction was
    # missing while its sibling had them. The idle timer abandons a transfer
    # that has stopped moving bytes instead of waiting out the process wall,
    # and --partial keeps the file that was in flight when it fired, so the
    # retry deltas against what arrived instead of starting that file again.
    # See RSYNC_PUSH_TIMEOUT_S for what the omission cost.
    cmd = (
        f"rsync -rltz --partial --timeout={RSYNC_IDLE_TIMEOUT_S} {excludes} "
        f"-e 'ssh {SSH_OPTS} -p {port}' {local_path} root@{ip}:{remote_path}"
    )
    # Retried for the same reason _rsync_from is. This direction had no
    # retry, and a single transient rsync failure during bootstrap took down
    # two pods of a five-pod fleet mid-campaign -- the same connection-level
    # flakiness the SSH retry already handles, arriving through a different
    # subprocess. Sending the tree again is idempotent, so a retry is free.
    for attempt in range(1, RSYNC_PUSH_ATTEMPTS + 1):
        wall = RSYNC_PUSH_TIMEOUT_S if deadline is None else _clamp_to_deadline(
            deadline, RSYNC_PUSH_TIMEOUT_S, "rsync push",
        )
        # The payload size is in the line because a slow push is the failure
        # this direction has, and an operator watching it needs to know what
        # the wall was sized for.
        logger.info("[rsync to] %s:%d %s (attempt %d/%d, %.1fMB, %ds wall)",
                    ip, port, remote_path, attempt, RSYNC_PUSH_ATTEMPTS,
                    PUSH_PAYLOAD_BYTES / 1e6, wall)
        try:
            subprocess.run(cmd, shell=True, check=True, timeout=wall)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt == RSYNC_PUSH_ATTEMPTS:
                raise
            backoff = SSH_RETRY_BACKOFF_S * attempt
            logger.warning("rsync to %s:%d failed (%s); retrying in %ds.",
                           ip, port, exc, backoff)
            time.sleep(backoff)


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
        f"rsync -rltz --safe-links --partial "
        f"--timeout={RSYNC_IDLE_TIMEOUT_S} "
        f"-e 'ssh {SSH_OPTS} -p {port}' root@{ip}:{remote_path} {local_path}"
    )
    last_error: Exception | None = None
    for attempt in range(1, RSYNC_PULL_ATTEMPTS + 1):
        logger.info("[rsync from] %s:%d %s (attempt %d/%d)",
                    ip, port, remote_path, attempt, RSYNC_PULL_ATTEMPTS)
        try:
            subprocess.run(cmd, shell=True, check=True,
                           timeout=RSYNC_PULL_TIMEOUT_S)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            logger.warning("  rsync attempt %d failed: %s", attempt, exc)
            if attempt < RSYNC_PULL_ATTEMPTS:
                time.sleep(RSYNC_PULL_BACKOFF_S * attempt)
    raise RuntimeError(
        f"Failed to collect {remote_path} from {ip}:{port} after "
        f"{RSYNC_PULL_ATTEMPTS} attempts: {last_error}"
    )


def _assert_pod_state_is_private(path: Path) -> None:
    """Refuse to leave the pod roster readable by other local accounts.

    Split out from the write so it can be exercised directly: the mode is the
    kind of property that is correct until some future change to the write path
    quietly widens it, and nothing else here would notice a world-readable list
    of live pod addresses and SSH ports.
    """
    mode = path.stat().st_mode & 0o777
    if mode != POD_STATE_MODE:
        raise RuntimeError(
            f"{path} is mode {mode:#o}, not {POD_STATE_MODE:#o}. It holds every "
            f"pod's live IP and SSH port, so it must not be readable by other "
            f"accounts on this machine."
        )


def _save_pod_state(
    pods: list[dict[str, Any]], meta: dict[str, Any] | None = None,
) -> None:
    POD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pods": pods, "meta": meta or {}}
    # Atomic, because this file is the ONLY local record of which pods are
    # billing. write_text truncates in place, so a crash, a full disk or a
    # killed orchestrator part-way through left a file that exists and does not
    # parse -- and every reader of it then dies on a JSONDecodeError: teardown,
    # reap, and the scheduled reaper guard that is the last bound on spend once
    # the orchestrator is gone. A temp file and os.replace() means a reader
    # sees the previous roster or the new one, never half of either.
    atomic_write_json(payload, POD_STATE_FILE)
    # .gitignore covers this path; the mode keeps it off other local accounts.
    # mkstemp creates the temp at 0600 and os.replace carries that mode across
    # rather than the target's, so this restates the intent instead of closing
    # a window -- and the check below is what actually holds it.
    POD_STATE_FILE.chmod(POD_STATE_MODE)
    _assert_pod_state_is_private(POD_STATE_FILE)
    logger.info("Pod state saved to %s (%d pods)", POD_STATE_FILE, len(pods))


def _read_pod_state() -> dict[str, Any]:
    """The recorded roster, or raise.

    Deliberately fatal on a file that will not parse. Callers that terminate
    pods BY ID cannot do anything sensible without the ids, and guessing there
    would be a silent partial teardown. `reap` is the one caller that can work
    without them -- it sweeps by name -- and it is the one that catches.
    """
    if not POD_STATE_FILE.exists():
        raise FileNotFoundError(
            f"No pod state file at {POD_STATE_FILE} — run 'provision' first"
        )
    raw: Any = json.loads(POD_STATE_FILE.read_text(encoding="utf-8"))
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


def _preserve_corrupt_pod_state() -> None:
    """Move an unparseable state file aside instead of letting reap delete it.

    reap unlinks the state file on its way out, which is right when the file
    was understood and its pods dealt with. It is wrong for bytes nothing could
    read: a truncated record can still name a pod the name sweep cannot see --
    validation pods carry the `tract-p1b-val` prefix and reap's orphan sweep
    matches POD_CONFIGS, which holds the test names only -- and those bytes are
    then the last evidence that pod ever existed.

    A failure to move them is logged and not raised. This runs inside the
    recovery command, and refusing to reap a billing fleet because a rename
    failed would be the wrong trade.
    """
    destination = POD_STATE_FILE.with_name(
        POD_STATE_FILE.name + POD_STATE_CORRUPT_SUFFIX
    )
    try:
        size = POD_STATE_FILE.stat().st_size
        POD_STATE_FILE.replace(destination)
    except OSError as exc:
        logger.error(
            "Could not move the unreadable %s aside (%s). reap will unlink it, "
            "so copy it now if you want the bytes.", POD_STATE_FILE, exc,
        )
        return
    logger.warning(
        "Kept the %d unreadable byte(s) at %s. They may still name a pod the "
        "name sweep cannot match.", size, destination,
    )


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


def _extend_deadline() -> None:
    """Give the current arm its own MAX_RUN_HOURS window."""
    state = _read_pod_state()
    meta = dict(state.get("meta") or {})
    meta["deadline"] = time.time() + MAX_RUN_HOURS * 3600
    meta["deadline_extensions"] = int(meta.get("deadline_extensions", 0)) + 1
    if meta["deadline_extensions"] > MAX_DEADLINE_EXTENSIONS:
        raise RuntimeError(
            f"Refusing to extend the run window past "
            f"{MAX_DEADLINE_EXTENSIONS} arms ({MAX_DEADLINE_EXTENSIONS * MAX_RUN_HOURS}h "
            f"of fleet time). Collect what finished and tear down."
        )
    _save_pod_state(state.get("pods") or [], meta=meta)
    logger.info("Run window extended to %.1fh from now (extension %d/%d).",
                MAX_RUN_HOURS, meta["deadline_extensions"], MAX_DEADLINE_EXTENSIONS)


def _bootstrap_ladder_s() -> int:
    """Bootstrap's worst case if only the retry ladder bounded it.

    Not what the code permits any more -- BOOTSTRAP_DEADLINE_S is -- but the
    number that deadline exists to defend against, and the one _check_budget
    was trying to state. Kept as an expression over the constants rather than
    a literal so it cannot fall out of step with them: every term below is a
    knob someone may turn, and turning one used to leave the budget model
    describing a bootstrap that no longer existed.
    """
    # Backoff between attempts is SSH_RETRY_BACKOFF_S * attempt, summed over
    # every attempt but the last: 15 + 30 + 45.
    ssh_backoff = (
        SSH_RETRY_BACKOFF_S * SSH_CONNECT_ATTEMPTS * (SSH_CONNECT_ATTEMPTS - 1) // 2
    )
    per_ssh = SSH_CONNECT_ATTEMPTS * SSH_BOOTSTRAP_TIMEOUT_S + ssh_backoff
    push_backoff = (
        SSH_RETRY_BACKOFF_S * RSYNC_PUSH_ATTEMPTS * (RSYNC_PUSH_ATTEMPTS - 1) // 2
    )
    push = RSYNC_PUSH_ATTEMPTS * RSYNC_PUSH_TIMEOUT_S + push_backoff
    return BOOTSTRAP_SSH_STEPS * per_ssh + push


def _check_budget(gpu_type: str, n_pods: int) -> dict[str, Any]:
    """Refuse to provision a fleet whose worst case exceeds the budget.

    There was no price query, no spend poll and no abort anywhere in this
    module; the ceiling existed only in prose. This converts it into a
    precondition that runs before any pod is created.
    """
    price_per_pod = get_gpu_price(gpu_type)
    fleet_hourly = price_per_pod * n_pods

    # Price the wall the code can actually reach, not the one it intends to.
    # The bounding stages are bootstrap, the fold itself, and a SERIAL collect
    # across pods. A budget check against MAX_RUN_HOURS alone understated the
    # permitted spend by more than a factor of two, which made the gate
    # unreachable: the $12/hr part filter already caps worst case below the
    # $1000 budget, so the check could never fire on any input the caller
    # could produce.
    #
    # The bootstrap term was (3 * SSH_DEFAULT_TIMEOUT_S + RSYNC_TIMEOUT_S), or
    # 3.50h, and was wrong three ways at once: _bootstrap_pod issues FOUR _ssh
    # calls, they run at SSH_BOOTSTRAP_TIMEOUT_S rather than the hour-long
    # default, and neither SSH_CONNECT_ATTEMPTS nor the rsync attempts appeared
    # in it at all. Two of those errors happened to cancel the third, which is
    # the least durable way for a number to be approximately right.
    #
    # What is priced now is BOOTSTRAP_DEADLINE_S, because that is the bound
    # _bootstrap_pod enforces on itself; the retry ladder underneath it, still
    # reported below, is 4.27h and would price a wall no pod can reach. The
    # slack is the one backoff sleep that can begin just before the deadline
    # expires.
    bootstrap_h = (BOOTSTRAP_DEADLINE_S + BOOTSTRAP_DEADLINE_SLACK_S) / 3600
    ladder_h = _bootstrap_ladder_s() / 3600
    fold_h = FOLD_TIMEOUT_S / 3600
    # Attempts and their backoff, per pod, because collect walks the roster
    # serially and every pod can pay the full ladder.
    collect_per_pod_s = (
        RSYNC_PULL_ATTEMPTS * RSYNC_PULL_TIMEOUT_S
        + RSYNC_PULL_BACKOFF_S * RSYNC_PULL_ATTEMPTS * (RSYNC_PULL_ATTEMPTS - 1) // 2
    )
    collect_h = n_pods * collect_per_pod_s / 3600
    reachable_h = bootstrap_h + fold_h + collect_h
    worst_case = fleet_hourly * reachable_h

    logger.info("Budget check:")
    logger.info("  %s at $%.2f/hr x %d pods = $%.2f/hr",
                gpu_type, price_per_pod, n_pods, fleet_hourly)
    logger.info("  reachable wall time = %.2fh (bootstrap %.2f + fold %.2f + "
                "serial collect %.2f), declared cap %.1fh",
                reachable_h, bootstrap_h, fold_h, collect_h, MAX_RUN_HOURS)
    logger.info("  bootstrap retry ladder would permit %.2fh/pod; the "
                "cooperative deadline holds it to %.2fh",
                ladder_h, bootstrap_h)
    logger.info("  worst case = $%.2f against budget $%.2f",
                worst_case, BUDGET_USD)

    # Coherence, not thrift. _check_deadline ABORTS the run when MAX_RUN_HOURS
    # passes, so a window that cannot fit bootstrap plus a single fold buys a
    # fleet that is guaranteed to be torn down mid-fold with nothing collected
    # and every pod billed for the attempt. Raising FOLD_TIMEOUT_S without
    # raising this produced exactly that: 0.43h + 8.00h against a 6h window.
    # Refuse at provision time, where it costs nothing.
    minimum_window_h = bootstrap_h + fold_h
    if minimum_window_h > MAX_RUN_HOURS:
        raise RuntimeError(
            f"Refusing to provision: bootstrap ({bootstrap_h:.2f}h) plus one "
            f"fold ({fold_h:.2f}h) needs a {minimum_window_h:.2f}h window, but "
            f"MAX_RUN_HOURS is {MAX_RUN_HOURS:.1f}h and _check_deadline aborts "
            "the run when it passes. This fleet could not finish a single fold. "
            "Raise TRACT_RUNPOD_MAX_HOURS or lower FOLD_TIMEOUT_S."
        )

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
        # Per stage, because the total is a sum of three very different
        # arguments and the state file is where an operator reconstructs which
        # one moved. bootstrap_ladder_hours is what the retry ladder would
        # permit without the deadline: recorded so its removal shows up as a
        # number, not as a silently cheaper estimate.
        "bootstrap_hours": bootstrap_h,
        "bootstrap_ladder_hours": ladder_h,
        "fold_hours": fold_h,
        "collect_hours": collect_h,
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


def _preflight_training_stack() -> None:
    """Refuse to provision until the pinned encoder stack can supply its symbols.

    The pods install requirements-train.txt and then import the training
    modules. sentence-transformers moves symbols between minor versions, so a
    pin bump can turn every fold into a ModuleNotFoundError seconds after the
    fleet starts billing. That is the most expensive place to learn about an
    import path.

    Three checks, because none alone is sufficient.

    The INSTALL check runs first and is the widest. Every earlier check here
    asked a question about imports, and an import check assumes the stack
    installed. scipy==1.18.0 is the case that proves the gap: it declares
    `Requires-Python: >=3.12`, so on any 3.11 interpreter `pip install -r
    requirements-train.txt` dies during resolution, the symbols never exist to
    be imported, and the fleet is already billing. So resolve the pinned set
    against the interpreter inside DOCKER_IMAGE, and refuse when any pin's
    declared floor excludes it.

    The PIN check reads the sentence-transformers version the PODS will install
    and refuses when that version's package layout was never read from its
    wheel. The local resolve is a weaker cross-check, because the provisioning
    host usually carries the serving pin rather than the training pin, so it
    proves the shim works for the layout it can see and nothing more.
    """
    from tract import supply_chain
    from tract.supply_chain import (
        find_python_incompatible_pins,
        parse_exact_pins,
    )
    from tract.training.st_compat import (
        SYMBOL_PATHS,
        installed_version,
        pinned_st_version,
        require_tested_version,
        resolve_symbol,
    )

    requirements = PROJECT_ROOT / "requirements-train.txt"

    pins = parse_exact_pins(requirements)
    # Read off the module rather than imported by name so a test can substitute
    # a metadata table for the PyPI round trip.
    violations = find_python_incompatible_pins(
        pins, POD_PYTHON_VERSION, supply_chain.fetch_requires_python,
    )
    if violations:
        detail = "; ".join(v.message() for v in violations)
        raise RuntimeError(
            f"Refusing to provision: {len(violations)} of {len(pins)} pins in "
            f"{requirements.name} cannot install on the pod interpreter "
            f"(Python {POD_PYTHON_VERSION} in {DOCKER_IMAGE}). {detail}. The "
            f"pods install this file after they exist, so shipping it would "
            f"buy a fleet that dies at dependency resolution."
        )
    logger.info(
        "Training-stack preflight: all %d pins in %s admit Python %s.",
        len(pins), requirements.name, POD_PYTHON_VERSION,
    )

    pinned = pinned_st_version(requirements)
    require_tested_version(
        pinned, f"Refusing to provision: {requirements.name} pins a stack TRACT "
        f"cannot vouch for",
    )
    logger.info(
        "Training-stack preflight: pods will install sentence-transformers==%s, "
        "whose layout is covered.", pinned,
    )

    # Resolving every symbol rather than the first one, so a partial move is
    # reported in full instead of one name at a time across successive runs.
    local = installed_version()
    for symbol in sorted(SYMBOL_PATHS):
        resolve_symbol(symbol)
    logger.info(
        "Training-stack preflight: %d symbols resolved locally under "
        "sentence-transformers==%s.", len(SYMBOL_PATHS), local,
    )


def _preflight_corpus() -> str:
    """Refuse to spend a GPU hour on a corpus the training links were not built from.

    `assert_corpus_matches_training_links` was written as a refusal and called
    from nothing. It lived in the Jetson briefing as a checklist row and in its
    own tests, which meant the control existed and never fired. A checklist row
    depends on a person; this does not.

    Called from both `provision` and `run_folds` on purpose. provision is where
    it saves money, and run_folds is where it holds for anyone driving the
    subcommands by hand or resuming onto an existing fleet.

    Returns:
        The corpus digest this run reads.

    Raises:
        CorpusMismatchError: The corpus differs from the recorded one.
        FileNotFoundError: The metadata sidecar is absent, so there is nothing
            to check against. Not a pass.
    """
    digest = assert_corpus_matches_training_links()
    logger.info("Corpus matches the training links it was built from (%s).",
                digest[:12])
    return digest


def provision(
    folds: list[str] | None = None, split: str = "test",
) -> list[dict[str, Any]]:
    # Ordered cheapest-check-first. All three run before anything is created,
    # so none of them can cost a dollar to fail.
    #
    # The corpus check sits between the other two deliberately. It hashes a
    # multi-megabyte file, so it is not as cheap as resolving a version pin,
    # and it is local, so it is far cheaper than the tracking check's network
    # round trip. Putting it first also made a stack-pin failure surface as a
    # corpus error on any fresh clone, which is how CI caught it.
    _preflight_training_stack()
    _preflight_corpus()
    _preflight_tracking()

    # Start every round with no recorded host keys. The file only ever
    # accumulates, and across a campaign's thirty pod-runs drawn from RunPod's
    # IP and port pool an endpoint eventually comes back attached to a
    # different machine. ssh then answers "Host key verification failed", which
    # _is_transient_ssh_failure refuses to retry -- correctly, because a changed
    # key normally IS a configuration error rather than a blip. Here it is a
    # false positive that hard-aborts the bootstrap of a fleet that is already
    # billing.
    #
    # Keeping the file across rounds buys no security to weigh against that.
    # Every pod is created fresh minutes ago, so there is no legitimate earlier
    # key for a new one to be compared against: an entry from a previous round
    # describes a host that no longer exists. Within a round the file still
    # does its job -- accept-new records each pod's key on first contact, and a
    # key that changes mid-run is still a hard failure, which is the case this
    # file was introduced for.
    if KNOWN_HOSTS_FILE.exists():
        logger.info(
            "Discarding %s before this round. Its entries describe pods that "
            "no longer exist, and a reused IP and port would read as a host-key "
            "failure -- which is not retried.", KNOWN_HOSTS_FILE,
        )
    KNOWN_HOSTS_FILE.unlink(missing_ok=True)

    configs = select_pod_configs(folds, split)
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
    # create_pod carries the tier each pod landed on into the state file, so
    # the record now says WHERE every fold ran and not merely that it did.
    # Called out here as well because the fallback is silent per pod and the
    # thing that follows provisioning is _rsync_to, which ships the working
    # tree -- data/processed/licensed included -- to whichever hosts answered.
    elsewhere = sorted(
        p["role"] for p in pods if p.get("cloud_type") != PRICE_CLOUD_TYPE
    )
    if elsewhere:
        logger.warning(
            "%d of %d fold(s) are on a cloud tier other than %s (or recorded "
            "none): %s. The licensed corpus is rsynced to those hosts, and the "
            "budget was priced on %s.",
            len(elsewhere), len(pods), PRICE_CLOUD_TYPE, elsewhere,
            PRICE_CLOUD_TYPE,
        )
    logger.info("All %d pods provisioned and SSH-ready.", len(pods))
    return pods


def _bootstrap_pod(
    pod: dict[str, Any], base_model: str = PHASE1B_BASE_MODEL,
    env: dict[str, str] | None = None,
    deadline: float | None = None,
) -> None:
    """Install the stack on one pod, or give up inside BOOTSTRAP_DEADLINE_S.

    Args:
        deadline: Absolute time.monotonic() value this pod's bootstrap must
            finish by. Defaults to BOOTSTRAP_DEADLINE_S from now, so a caller
            that knows nothing about deadlines still gets one.

    Raises:
        TimeoutError: The deadline passed with steps still to run. The pod is
            left as it is -- it is still billing and still named in the state
            file -- and the fleet stops waiting on it.

    Every step is clamped to what remains of the deadline instead of running
    to its own ceiling. That is the only mechanism that works here: this runs
    on a worker thread inside run_folds' ThreadPoolExecutor, whose context
    manager joins every thread on the way out, so a wedged step is not
    something the orchestrator can cancel from outside -- the thread has to
    end itself. On 2026-08-27 one did not, and four healthy pods waited 90
    minutes at the barrier for a fifth that was moving no bytes.
    """
    ip, port, role = pod["ip"], pod["port"], pod["role"]
    if deadline is None:
        deadline = time.monotonic() + BOOTSTRAP_DEADLINE_S
    # The caller passes the env so the credential is read once on the main
    # thread. Falling back to _get_pod_env() here keeps a direct call working,
    # but the fleet path must not take that branch: see run_folds.
    pod_env = _get_pod_env() if env is None else env
    logger.info("Bootstrapping pod for fold '%s' (%s:%d), %ds budget...",
                role, ip, port, BOOTSTRAP_DEADLINE_S)

    _ssh(ip, port, "apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1",
         check=False, timeout=SSH_BOOTSTRAP_TIMEOUT_S, deadline=deadline)

    _rsync_to(ip, port, f"{PROJECT_ROOT}/", "/workspace/tract/", deadline=deadline)

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
    ), timeout=SSH_BOOTSTRAP_TIMEOUT_S, deadline=deadline)

    # Fetch the base model once, here, rather than inside the fold. A 429 at
    # this point costs a bootstrap; the same 429 twenty minutes into training
    # costs the fold. It also proves the read-only token works before any
    # GPU time is spent on it.
    _ssh(ip, port, (
        "cd /workspace/tract && python -c "
        "'import os; from huggingface_hub import snapshot_download; "
        "from tract.encoders import resolve; "
        # The model name arrives through the environment, not interpolated
        # into the program text. _ssh wraps the whole thing in single quotes,
        # so a repr() here closes that quoting and the shell splits the name
        # into separate words -- which is how "name = Qwen/Qwen3-Embedding-0.6B"
        # reached the remote interpreter unquoted and raised SyntaxError.
        "name = os.environ[\"TRACT_BASE_MODEL\"]; s = resolve(name); "
        "p = snapshot_download(name, revision=s.revision); "
        "print(f\"cached {name} at {s.revision[:12]} -> {p}\")'"
    ), env={**pod_env, "TRACT_BASE_MODEL": base_model},
       timeout=SSH_BOOTSTRAP_TIMEOUT_S, deadline=deadline)

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
    ), timeout=SSH_BOOTSTRAP_TIMEOUT_S, deadline=deadline)

    logger.info("Bootstrap complete for fold '%s' with %ds of budget to spare",
                role, max(0, int(deadline - time.monotonic())))


def _run_fold_on_pod(
    pod: dict[str, Any], config_name: str, arm_flags: tuple[str, ...] = (),
    split: str = "test", env: dict[str, str] | None = None,
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
        f"--split {shlex.quote(split)} "
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
    # IDEMPOTENT, because this command is sent through a transport that retries.
    #
    # _ssh retries on TimeoutExpired, and the launch used to be given
    # SSH_DEFAULT_TIMEOUT_S -- one hour -- for a command whose whole job is to
    # detach a process and echo. On 2026-08-28 arm A3's folds ran ~89 minutes
    # each; the launch session hit its one-hour wall, _ssh retried, and a SECOND
    # detached trainer started on the same GPU. `setsid` guarantees the first
    # survives, so both then trained the same fold at half speed, into a log
    # they interleaved, and the fold could not finish inside FOLD_TIMEOUT_S.
    # Confirmed on the pod: pids 1248 and 1720, both `run_fold --framework
    # 'NIST 800-53 v5'`, 6636 MiB and 6576 MiB on one A100, started exactly
    # sixty minutes apart.
    #
    # The bug was invisible until a fold outran the launch timeout. Arm A1's
    # 34-minute folds finished before the wall was ever reached, so A1 passed
    # not because this was correct but because it was fast enough.
    #
    # `mkdir` is the guard because it is atomic on POSIX and needs no process
    # matching: exactly one caller can create the directory, so a retry takes
    # the else branch and reports the launch it did not perform. A pgrep guard
    # was the obvious alternative and is the wrong tool -- the shell running the
    # pgrep carries the pattern in its own command line, which is a defect this
    # repository has now written three times.
    lock_path = f"{remote_dir}/fold_{slug}.launched"
    launch = (
        f"cd {remote_dir} && "
        f"if mkdir {shlex.quote(lock_path)} 2>/dev/null; then "
        f"rm -f {shlex.quote(exit_path)}; "
        f"setsid nohup bash -c {shlex.quote(fold_cmd + f'; echo $? > {exit_path}')} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"echo started; "
        f"else echo already-launched; fi"
    )

    logger.info("[%s] Launching fold (detached)...", framework)
    start = time.time()
    # Read on the caller's thread, not here. This line used to sit above the
    # try below, so a `pass` timeout raised out of this function instead of
    # returning a status dict, and run_folds had no guard: one credential
    # hiccup ended the fleet. See run_folds for the other half of the fix.
    pod_env = _get_pod_env() if env is None else env
    try:
        launched = _ssh(ip, port, launch, env=pod_env,
                        timeout=SSH_LAUNCH_TIMEOUT_S)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] LAUNCH FAILED after %.1fm: %s", framework, elapsed / 60, e)
        return {"fold": framework, "status": "failed",
                "error": f"launch: {e}", "elapsed_s": elapsed}

    # The lock's two branches mean opposite things and the exit status cannot
    # tell them apart -- both are 0. Reading the word it echoed is the whole
    # point of echoing it, and this was written without a reader: a retry took
    # the else branch, satisfied check=True, and the poller below then read the
    # PREVIOUS run's exit file within one poll interval. With a prior exit of 0
    # that reports COMPLETE in seconds having trained nothing, and `collect`
    # then rsyncs the old results into a number nobody re-earned. The runbook
    # and the orchestrator's own failure message both tell an operator to
    # re-run `run`, so this is a documented path, not an exotic one.
    # Exact match on the last line, not a substring scan of the whole stream.
    # The launch echoes exactly one of two words, so `in` was both looser than
    # needed and the same shape as the argv-substring defect the hygiene test
    # exists to catch: any path or log line containing "already-launched"
    # anywhere in stdout would have satisfied it.
    launch_lines = (launched.stdout or "").strip().splitlines()
    if launch_lines and launch_lines[-1].strip() == "already-launched":
        alive = _ssh(
            ip, port,
            # Same probe shape as reaper_guard.pod_training_state, and for the
            # same reason: a pgrep carries its own pattern in its own command
            # line, so the probing shell answers about itself. $$ is excluded
            # explicitly and the process is identified by its exe being python
            # plus the run_fold MODULE path in argv, not by a loose substring.
            "for p in /proc/[0-9]*; do "
            '  pid=${p##*/}; [ "$pid" = "$$" ] && continue; '
            "  exe=$(readlink -f $p/exe 2>/dev/null) || continue; "
            '  case "${exe##*/}" in python*) ;; *) continue ;; esac; '
            "  if tr '\\0' ' ' < $p/cmdline 2>/dev/null "
            "     | grep -q 'scripts.phase1b.run_fold'; then "
            "    echo BUSY; exit 0; fi; "
            "done; echo IDLE",
            check=False, timeout=SSH_LAUNCH_TIMEOUT_S,
        )
        answer = (alive.stdout or "").strip().splitlines()
        if answer and answer[-1] == "BUSY":
            logger.info("[%s] A trainer is already running; attaching to it",
                        framework)
        else:
            elapsed = time.time() - start
            logger.error(
                "[%s] REFUSING: %s exists but no trainer is running, so this "
                "pod holds a FINISHED fold. Polling would report that run's "
                "stale exit code as though this launch produced it. To re-run "
                "deliberately: ssh in and `rm -rf %s`.",
                framework, lock_path, lock_path,
            )
            return {"fold": framework, "status": "failed",
                    "error": f"stale launch lock at {lock_path}; refusing to "
                             "report a prior run's exit code as this one's",
                    "elapsed_s": elapsed}

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
    split: str = "test",
) -> list[str]:
    """Run every fold on its pod. Returns the roles that FAILED.

    The return value used to be None and fold failures were logged and
    dropped. full_pipeline then collected whatever existed, set
    results_are_safe, and tore the fleet down -- so two failed folds out of
    five destroyed the pods and the failure surfaced later, at aggregation,
    with nothing left to retry on.
    """
    # Before anything else, including before the pod roster is read. A fleet
    # is already billing by the time this runs, so the cost of stopping here
    # is minutes; the cost of not stopping is a full arm measured against 370
    # missing links that reports normally.
    _preflight_corpus()
    pods = _load_pod_state()
    # Refresh the window for this arm. The deadline was stamped once at
    # provision and never renewed, so a campaign of several arms tripped it
    # partway through -- and tripping it stopped the work while leaving five
    # GPUs billing, which is exactly backwards for a budget control.
    _check_deadline()
    # The pod roster is fixed at provision. Running a split whose folds are
    # not the roles the pods carry fails per-fold, deep inside training, on
    # every pod at once -- which is how a validation campaign launched on a
    # test fleet burned a bootstrap before saying so.
    roster = set(fold_roster(split))
    have = {p["role"] for p in pods}
    if have != roster:
        raise RuntimeError(
            f"Fleet was provisioned for a different split: pods hold "
            f"{sorted(have)} but split {split!r} needs {sorted(roster)}. "
            f"Tear down and re-provision with --split {split}."
        )
    _extend_deadline()

    # Read the credential once, here, on the main thread. It used to be read
    # inside each worker: five threads shelled out to `pass` at the same
    # instant, the GPG agent serialised decryption and could not run pinentry
    # from a non-tty thread, so an expired agent cache raced all five against
    # the same ten-second timeout. One `pass` invocation cannot race itself.
    pod_env = _get_pod_env()

    logger.info("Bootstrapping %d pods in parallel...", len(pods))
    # Isolate failures. list(ex.map(...)) re-raises the first exception and
    # abandons the rest, so one bad pod aborted the whole fleet while the other
    # four kept billing.
    bootstrap_errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pods)) as ex:
        # The arm's encoder, so the pod prefetches the model it will train
        # rather than always BGE-large. A mismatch here means the 429-safe
        # prefetch warms the wrong weights and the fold downloads inside
        # training, where a rate limit costs the fold instead of a bootstrap.
        arm_model = PHASE1B_BASE_MODEL
        for i, flag in enumerate(arm_flags):
            if flag == "--base-model" and i + 1 < len(arm_flags):
                arm_model = arm_flags[i + 1]
        bootstrap_futures = {
            ex.submit(_bootstrap_pod, pod, arm_model, pod_env): pod["role"]
            for pod in pods
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
            ex.submit(
                _run_fold_on_pod, pod, config_name, arm_flags, split, pod_env,
            ): pod["role"]
            for pod in pods
        }
        for f in concurrent.futures.as_completed(fold_futures):
            role = fold_futures[f]
            # Same treatment the bootstrap loop above already had. Without it,
            # one raise ended the as_completed loop, discarded the other
            # futures' results, and left full_pipeline's finally with
            # results_are_safe False -- five GPUs billing and four folds that
            # were about to succeed abandoned.
            try:
                result = f.result()
            except Exception as exc:  # noqa: BLE001 - one fold, not the fleet
                logger.error("[%s] FOLD RAISED: %s", role, exc)
                result = {
                    "fold": role, "status": "failed",
                    "error": f"raised: {exc}", "elapsed_s": 0.0,
                }
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


def _fold_result_landed(local_results: Path, role: str) -> bool:
    """Did this role's fold record arrive, and does it parse?

    Two slug conventions are live in this codebase: run_fold.py builds the
    directory with `role.replace(' ', '_')` while this module's log and exit
    paths use `re.sub(r"[^A-Za-z0-9]+", "_", role)`. They agree on 'MITRE
    ATLAS' and disagree on 'NIST 800-53 v5'. Accept either, because a false
    "not collected" here tears down nothing but does block a fleet that
    actually succeeded.
    """
    candidates = {
        role.replace(" ", "_"),
        re.sub(r"[^A-Za-z0-9]+", "_", role),
    }
    for slug in candidates:
        path = local_results / f"fold_{slug}" / FOLD_RESULT_FILENAME
        if not path.is_file():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A truncated transfer leaves a file that exists and is not JSON.
            logger.error("%s exists but does not parse: %s", path, exc)
            continue
        return True
    return False


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
            continue

        # Verify the payload, not the transport. rsync against a directory
        # that exists and holds no fold record exits 0, so a fold that died
        # without writing its result was counted as collected: failed_folds
        # empty, uncollected empty, results_are_safe True, teardown destroys
        # the pods, and aggregate then fails with nothing left to re-run and
        # the GPU hours already spent.
        if not _fold_result_landed(local_results, pod["role"]):
            logger.error(
                "Collection from %s moved no usable %s. The rsync succeeded, "
                "so the fold ran and wrote nothing readable. Do NOT tear this "
                "pod down: its logs are the only record of why.",
                pod["role"], FOLD_RESULT_FILENAME,
            )
            failed.append(pod["role"])

    logger.info("Results collected to %s", local_results)
    if failed:
        logger.error("Collection failed for %d fold(s): %s", len(failed), failed)
    return failed


def _assert_results_are_current(local_results: Path, allow_stale: bool) -> None:
    """Refuse to turn stale folds into a headline number.

    `load_fold_results` already refuses a partial fold set, mixed arms, mixed
    input digests and mixed git SHAs. Every one of those compares folds
    against their siblings, and none asks whether the corpus those digests
    describe is the corpus on disk. Five uniformly stale folds pass all of
    them and aggregate into a number that reads as current -- which is the
    state A1 and A2 are in right now.

    `allow_stale` exists because the briefing's rule has two halves: a stale
    result MAY be compared against its own recorded inputs, and MAY NOT be
    quoted as a current measurement. The flag is the first half, and it has to
    be typed on purpose.

    Raises:
        RuntimeError: A fold's recorded inputs no longer match the files.
    """
    from tract.staleness import check_result

    statuses = [
        check_result(p)
        for p in sorted(local_results.glob(f"fold_*/{FOLD_RESULT_FILENAME}"))
    ]
    stale = [s for s in statuses if s.is_stale]
    if not stale:
        return

    moved = sorted({si.field for s in stale for si in s.stale})
    detail = "; ".join(
        f"{s.result_path}: " + ", ".join(si.field for si in s.stale) for s in stale
    )
    if allow_stale:
        logger.warning(
            "AGGREGATING STALE FOLDS on purpose (--allow-stale). %d of %d fold(s) "
            "were measured against different inputs (%s). This number describes "
            "the corpus recorded in those folds, NOT the one on disk. Do not "
            "quote it as a current measurement.",
            len(stale), len(statuses), ", ".join(moved),
        )
        return

    raise RuntimeError(
        f"{len(stale)} of {len(statuses)} fold(s) under {local_results} were "
        f"measured against inputs that have since changed ({', '.join(moved)}). "
        f"They agree with each other, which is why every existing check passes, "
        f"and they do not agree with the corpus on disk. Re-run the arm, or "
        f"pass --allow-stale to compare it against its own recorded inputs and "
        f"accept that the result cannot be quoted as current. Details: {detail}"
    )


def aggregate(
    config_name: str = "phase1b_primary",
    folds: list[str] | None = None,
    split: str = "test",
    n_configurations: int = 1,
    allow_stale: bool = False,
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
    # Currency first. Everything below compares folds to their siblings; this
    # is the only check that compares them to the corpus on disk.
    _assert_results_are_current(local_results, allow_stale)
    # An explicitly scoped run declares its own fold set; anything else must
    # match the full LOFO set, so a partial cross-validation cannot be
    # aggregated into a headline number by omission.
    fold_results = load_fold_results(
        local_results,
        expected_frameworks=set(folds) if folds else set(fold_roster(split)),
    )
    if folds:
        logger.warning(
            "Aggregating a SCOPED run over %s. This is not a LOFO result and "
            "must not be reported as one.", sorted(folds),
        )
    logger.info("Aggregating %d folds from %s", len(fold_results), local_results)

    from tract.hierarchy import CREHierarchy
    from tract.training.orchestrate import lexical_overlap_diagnostic

    hierarchy = CREHierarchy.model_validate(
        load_json(PROCESSED_DIR / "cre_hierarchy.json")
    )
    lexical = lexical_overlap_diagnostic(local_results, hierarchy)
    logger.info(
        "  lexical echo    : %d/%d items (%.1f%%) already contain their answer",
        lexical["n_lexical_echo"], lexical["n_total"],
        100 * lexical["echo_fraction"],
    )
    if lexical["hit_at_1_non_echo"] is not None:
        logger.info(
            "  hit@1 non-echo  : %.4f  <- semantic mapping, not string match",
            lexical["hit_at_1_non_echo"],
        )

    record = {
        "config_name": config_name,
        "lexical_overlap": lexical,
        # Marks the record itself, so a scoped run cannot be mistaken for a
        # full cross-validation once it is read back out of the file.
        "scoped_to_folds": sorted(folds) if folds else None,
        "aggregate_hit1": aggregate_fold_results(fold_results),
        "gate": gate_decision(fold_results, n_configurations),
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


def _arm_from_config(config: dict[str, Any]) -> str:  # noqa: D401
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
    if config.get("use_framework_identity_filter"):
        parts.append("fwid")
    label = "-".join(parts)
    # Mirrors run_fold._campaign_label: the encoder and the branch balance
    # are part of a configuration's identity, not just its anchor arm.
    bal = config.get("branch_balance_temperature") or 0
    if bal:
        label += f"-bal{bal:g}"
    hub_rep = config.get("hub_rep_format") or "path+name"
    if hub_rep != "path+name":
        label += "-" + hub_rep.replace("path+name+", "")
    base = config.get("base_model") or ""
    if base and base != "BAAI/bge-large-en-v1.5":
        label += "-" + base.split("/")[-1]
    return label


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
    # The sweep runs whether or not a state file survives. teardown() unlinks
    # it on success and provision() overwrites it, so the exact situation that
    # strands pods -- a failed campaign followed by a second provision, or a
    # killed orchestrator -- is the one where this file is gone or stale.
    # Returning early here made the single recovery command report all-clear
    # on a fleet that was still billing.
    #
    # A file that exists and does not parse is that same situation wearing a
    # different exception, and only FileNotFoundError was caught: a truncated
    # state file therefore raised JSONDecodeError out of the ONE command that
    # recovers a fleet -- and out of reaper_guard, which calls reap(confirm=True)
    # and is the only automatic bound on spend once the orchestrator has died.
    # ValueError is the right net: json.JSONDecodeError and UnicodeDecodeError
    # both derive from it, as does _read_pod_state's own refusal of a payload
    # that is neither a list nor an object. All three mean the same thing here
    # -- there are no pod ids to work from -- and the name sweep below is what
    # this command does when there are none.
    try:
        state = _read_pod_state()
    except FileNotFoundError:
        logger.warning(
            "No state file. Sweeping the account for pods matching this "
            "run's names instead."
        )
        state = {"pods": [], "meta": {}}
    except ValueError as exc:
        logger.error(
            "%s exists but did not parse: %s. Treating it as absent and "
            "sweeping the account for pods matching this run's names. That "
            "sweep is blind to any pod whose name is not in POD_CONFIGS, so "
            "check the RunPod console before calling this fleet dead.",
            POD_STATE_FILE, exc,
        )
        _preserve_corrupt_pod_state()
        state = {"pods": [], "meta": {}}

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
    # the deterministic names both splits can produce.
    #
    # BOTH splits, and that is not a detail. This matched POD_CONFIGS until
    # 2026-08-27, and POD_CONFIGS is built from FOLD_FRAMEWORKS -- the TEST
    # roster -- so it holds tract-p1b-fold0..4 only. select_pod_configs names
    # validation pods tract-p1b-val-fold0..4 under a different prefix, and the
    # two sets are disjoint. The sweep was therefore blind to every validation
    # pod, which is four of Campaign 2's rounds.
    #
    # It cost a manual recovery within minutes of the first provision. A
    # capacity error killed fold0 while four validation pods came up billing;
    # the operator interrupted inside the pods=[] window this comment describes,
    # so teardown reported "nothing scoped to terminate" and this fallback --
    # the one path built for exactly that window -- would have swept past all
    # four because their names were not in POD_CONFIGS. They had to be
    # terminated by hand.
    known_ids = {p["pod_id"] for p in pods if p.get("pod_id")}
    expected_names = {
        config["name"]
        for split in ("test", "validation")
        for config in select_pod_configs(None, split)
    }
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
    split: str = "test",
    n_configurations: int = 1,
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
        provision(folds, split)
        failed_folds = run_folds(config_name, arm_flags, split)
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

    aggregate(config_name, folds, split, n_configurations)
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
    parser.add_argument("--framework-identity", action="store_true",
                        help="Ablation arm: strip the acronyms that name a "
                             "framework (OWASP, CWE, CAPEC, CCM)")
    parser.add_argument("--description-only", action="store_true",
                        help="Ablation arm: cut each control at its first "
                             "remediation heading")
    # These reached run_fold and the aggregator but never this layer, so an
    # encoder or rebalance arm either died on argparse -- stranding the fleet,
    # because SystemExit does not inherit from Exception and never reaches the
    # __main__ handler -- or ran as plain BGE-large under a name claiming
    # otherwise. The aggregator cannot catch that second case: every fold
    # agrees, so the arm check passes and the null result looks measured.
    parser.add_argument("--split", choices=("test", "validation"),
                        default="test",
                        help="validation selects arms on 1,265 non-AI items; "
                             "test reports on the pre-registered 147")
    # No default, deliberately. This used to default to 1, and a forgotten flag
    # then produced an UNCORRECTED nominal interval that is indistinguishable
    # from a correct one -- no error, no warning, just a gate that priced a
    # three-arm raffle as though one arm had run. Campaign 2 needs 3 on the
    # validation aggregates where selection happens and 1 on the single
    # uncontaminated test round, so neither value is safe as a silent default.
    # Forcing the operator to state it converts a silent wrong number into a
    # loud missing argument. See results/phase1b/CAMPAIGN2.md reporting rule 2.
    parser.add_argument("--n-configurations", type=int, default=None,
                        help="REQUIRED for `aggregate`. How many arms competed. "
                             "Sidak-corrects the gate so a winner is not "
                             "mistaken for a result. Campaign 2: 3 on "
                             "validation, 1 on the test round.")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Encoder arm: fine-tune this model instead of "
                             "the pinned BGE-large")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Encoder token budget; the anchor character cut "
                             "is derived from it")
    parser.add_argument("--hub-rep", type=str, default=None,
                        help="Hub representation arm. PRD:372 requires "
                             "path+name+desc be measured and 32/32 folds have "
                             "used the default bare label.")
    # Never set this on a campaign run. It exists so a stale result can be
    # read against its own recorded inputs, which the briefing permits, and
    # not so a refusal can be argued with.
    parser.add_argument("--allow-stale", action="store_true",
                        help="aggregate folds whose recorded inputs no longer "
                             "match the corpus on disk. The number describes "
                             "the recorded corpus and cannot be quoted as a "
                             "current measurement.")
    parser.add_argument("--branch-balance", type=float, default=None,
                        help="Rebalance arm: temperature flattening the "
                             "CRE-branch distribution")
    args = parser.parse_args()

    folds = (
        [f.strip() for f in args.folds.split(",") if f.strip()]
        if args.folds else None
    )

    arm_flags = tuple(
        flag for flag, on in (("--no-prose", args.no_prose),
                              ("--stopwords", args.stopwords),
                              ("--framework-identity", args.framework_identity),
                              ("--description-only", args.description_only)) if on
    )
    # Valued flags, passed through with their argument.
    for flag, value in (("--base-model", args.base_model),
                        ("--max-seq-length", args.max_seq_length),
                        ("--hub-rep", args.hub_rep),
                        ("--branch-balance", args.branch_balance)):
        if value is not None:
            arm_flags += (flag, str(value))

    if arm_flags and args.config_name == "phase1b_primary":
        # Arms must not share a results directory: fold records carry no arm,
        # so a mixed directory aggregates into a number describing no single
        # configuration.
        raise SystemExit(
            f"Refusing to run arm {arm_flags} into the default results "
            f"directory. Pass a distinct --config-name."
        )

    # Checked here rather than by argparse's `required=`, because the flag is
    # only meaningful for the two actions that reach gate_decision. Making it
    # globally required would force a meaningless number onto `price`, `reap`
    # and `teardown`, which is how required flags get reflexively set to 1.
    if args.action in ("full", "aggregate") and args.n_configurations is None:
        raise SystemExit(
            "--n-configurations is required for "
            f"`{args.action}` and has no default. It Sidak-corrects the gate "
            "for the number of arms that competed, and a wrong value produces "
            "an interval that looks correct. Campaign 2 pre-registers 3 for "
            "the validation aggregates where arm selection happens, and 1 for "
            "the single test round, because selection already happened on a "
            "disjoint split. See results/phase1b/CAMPAIGN2.md, reporting "
            "rule 2."
        )

    if args.action == "full":
        full_pipeline(args.config_name, arm_flags, folds, args.split,
                      args.n_configurations)
    elif args.action == "provision":
        provision(folds, args.split)
    elif args.action == "run":
        # The canary exposed this: the fold failed on a HuggingFace 429 and
        # `run` still exited 0, so an unattended wrapper checking $? would
        # have moved on to collect and teardown on a fleet that produced
        # nothing. Same class as the failure full_pipeline already guards.
        failed = run_folds(args.config_name, arm_flags, args.split)
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
        aggregate(args.config_name, folds, args.split, args.n_configurations,
                  allow_stale=args.allow_stale)
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
        _check_budget(gpu_type, len(select_pod_configs(folds, args.split)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Anything that reaches here left pods running or results uncollected.
        logger.exception("FAILED. Check for running pods: reap --confirm")
        sys.exit(1)
