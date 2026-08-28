"""Wait for RunPod capacity, then run one Campaign 2 arm end to end.

    python -m scripts.phase1b.await_capacity --arm A1              # dry, polls only
    python -m scripts.phase1b.await_capacity --arm A1 --confirm    # provisions and runs

WHY POLLING `price` WOULD NOT WORK, which is the whole reason this file exists.
At 11:47 on 2026-08-27 `price` reported "Selected GPU NVIDIA H100 80GB HBM3 at
$3.29/hr" and provisioning failed sixty seconds later with "There are no
instances currently available". `price` reads `list_available_gpus`, which
filters on the `secureCloud` / `communityCloud` booleans -- those say a GPU type
is OFFERED on a tier, not that anyone can have one. A watcher built on it would
have fired immediately, every time, and failed identically.

`lowestPrice { stockStatus }` is the real signal. Surveyed the same afternoon:
every H100- and A100-class part read `Low`, which is precisely the state that
produced four pods and one capacity error. `Low` means provisionable but scarce,
so it is not a green light for a five-pod fleet -- this waits for better.

WHAT IT WILL AND WILL NOT DO UNATTENDED. It provisions, checks the two gates the
runbook puts between `provision` and `run`, trains, collects, and tears down. It
does NOT aggregate: that produces the number, it needs --n-configurations, and a
human should read it. It stops after one arm.

Every failure path tears the fleet down before returning. The one incident this
apparatus already had was an interrupt landing in the window where
.pod_state.json records pods=[], which left `teardown` reporting "nothing scoped
to terminate" while four pods billed -- so teardown here is followed by an
independent sweep against the account, by name, across both splits.

Owner: TRACT.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scripts.phase0.runpod_provision import _gql, get_running_pods, terminate_pods
from scripts.phase1b.runpod_parallel import (
    PROJECT_ROOT,
    select_pod_configs,
)

# NOT in the repository. This runs for up to twelve hours and the operator will
# want to read it afterwards, so it cannot go under XDG_RUNTIME_DIR, which is
# wiped at logout -- but a log file in PROJECT_ROOT shows up as untracked and
# breaks the clean-tree check the runbook's Gate 0 makes before every provision,
# which is the check that keeps git_sha from being confidently wrong.
LOG_DIR: Final[Path] = Path.home() / "tract-campaign2-logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Attached to the root logger explicitly rather than through basicConfig.
# basicConfig is a NO-OP when the root logger already has handlers, and
# runpod_parallel.py calls it at import -- which this module does at line 45,
# before reaching here. The first run therefore logged to the terminal and
# wrote a ZERO-BYTE file, so the only record of a three-attempt provisioning
# session was tmux scrollback, which is finite and scrolls away.
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_file_handler = logging.FileHandler(LOG_DIR / "await_capacity.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_root.addHandler(_file_handler)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
           for h in _root.handlers):
    _stream = logging.StreamHandler(sys.stdout)
    _stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _root.addHandler(_stream)
logger = logging.getLogger(__name__)

EXIT_OK: Final[int] = 0
EXIT_GAVE_UP: Final[int] = 1
EXIT_REFUSED: Final[int] = 2

# Gate A of the runbook. A five-fold arm on a slower part is not a cheaper run,
# it is a dead one: a 57-minute Qwen fold at 0.47x throughput becomes ~120
# minutes, which is exactly FOLD_TIMEOUT_S, so all five folds die at the wall.
# H200 is included because it is strictly faster than an H100, not because it
# was measured here.
ACCEPTABLE_GPUS: Final[tuple[str, ...]] = (
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H100 PCIe",
    "NVIDIA H200",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
)

# `Low` is the state that produced four pods and one capacity error on
# 2026-08-27. It is not a lie -- pods really were created -- but it does not
# carry a five-pod fleet, and a fleet short one fold cannot produce a LOFO
# number. Anything better is worth an attempt.
SUFFICIENT_STOCK: Final[frozenset[str]] = frozenset({"High", "Medium"})

POLL_SECONDS: Final[int] = 10 * 60
# Twelve hours of polling at ten-minute intervals. Past that the operator should
# decide again rather than have a script decide for them, and the reaper guard's
# own quiet streak will have disarmed long before, so an unattended launch after
# that point would run with no independent spend bound.
MAX_WALL_SECONDS: Final[int] = 12 * 60 * 60
# Capacity that appears and evaporates between the poll and the create call is
# normal. Three failed attempts means the signal is not predicting anything and
# a human should look.
MAX_PROVISION_ATTEMPTS: Final[int] = 3

# How long a freshly created pod may sit without a published SSH endpoint before
# that counts as broken rather than pending. Measured on 2026-08-27: pods sat at
# ip=pending for 156s and were still healthy, so a minute is far too short.
ENDPOINT_WAIT_S: Final[int] = 6 * 60
ENDPOINT_POLL_S: Final[int] = 15


@dataclass(frozen=True)
class Arm:
    """One pre-registered Campaign 2 arm, exactly as CAMPAIGN2.md binds it."""

    key: str
    config_name: str
    split: str
    flags: tuple[str, ...]


# Bound by results/phase1b/CAMPAIGN2.md. A2 and A4 were dropped 2026-08-27; the
# names carry the c2r_ prefix because the c2_ directories hold stale folds and
# `collect` rsyncs without --delete.
ARMS: Final[dict[str, Arm]] = {
    "A1": Arm("A1", "c2r_A1_prose_sw_bge", "validation", ("--stopwords",)),
    "A3": Arm("A3", "c2r_A3_prose_sw_qwen06b", "validation",
              ("--stopwords", "--base-model", "Qwen/Qwen3-Embedding-0.6B")),
    "A5": Arm("A5", "c2r_A5_title_bge", "validation", ("--no-prose",)),
}


def stock_status(gpu_id: str) -> str:
    """RunPod's own word for how much of this part is left, or 'unknown'.

    Never raises. A watcher that dies because one GraphQL call hiccuped is a
    watcher that silently stops watching, and the failure it exists to catch
    happens while nobody is looking.
    """
    try:
        data = _gql(
            'query Q($input: GpuLowestPriceInput) { gpuTypes(input: {id: "'
            + gpu_id
            + '"}) { id lowestPrice(input: $input) { stockStatus } } }',
            {"input": {"gpuCount": 1, "secureCloud": True}},
        )
        types = data.get("gpuTypes") or []
        if not types:
            return "unknown"
        return ((types[0].get("lowestPrice") or {}).get("stockStatus")) or "none"
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("stock query failed for %s: %s", gpu_id, exc)
        return "unknown"


def survey() -> list[tuple[str, str]]:
    return [(g, stock_status(g)) for g in ACCEPTABLE_GPUS]


def capacity_is_sufficient(readings: list[tuple[str, str]]) -> bool:
    return any(status in SUFFICIENT_STOCK for _, status in readings)


def sweep_account() -> int:
    """Terminate every pod either split can name. Returns how many died.

    Independent of .pod_state.json ON PURPOSE. `teardown` reads that file, and
    the file records pods=[] between "intent to provision" and "all pods up" --
    the window an interrupted provision lands in, and the window that left four
    pods billing on 2026-08-27 while teardown reported nothing to do.
    """
    expected = {
        config["name"]
        for split in ("test", "validation")
        for config in select_pod_configs(None, split)
    }
    mine = [p for p in get_running_pods() if p.get("name") in expected]
    if not mine:
        return 0
    logger.warning("Sweeping %d pod(s) the state file did not account for: %s",
                   len(mine), [p.get("name") for p in mine])
    failed = terminate_pods([p["id"] for p in mine])
    if failed:
        raise RuntimeError(f"{len(failed)} pod(s) would not terminate and are still billing: {failed}")
    return len(mine)


def arm_the_reaper() -> None:
    """Ensure a scheduled reaper exists before a fleet does.

    Never raises. A watcher that dies because systemd-run was unhappy is worse
    than one running without the backstop, because the backstop is not the only
    control -- this module tears down on every path -- whereas a dead watcher
    provisions nothing and reports nothing.
    """
    unit = f"tract-reaper-await-{int(time.time())}"
    try:
        subprocess.run(
            ["systemd-run", "--user", "--collect", "--on-active=8h",
             f"--unit={unit}", f"--working-directory={PROJECT_ROOT}",
             "--setenv=USE_TF=0",
             sys.executable, "-m", "scripts.phase1b.reaper_guard", "--confirm"],
            check=True, capture_output=True, timeout=30,
        )
        logger.info("Reaper armed as %s (T+8h) before provisioning.", unit)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Could not arm the reaper (%s). Proceeding: this module "
                       "tears down on every path, but the independent backstop "
                       "is absent if it dies.", exc)


def orchestrator(*args: str) -> int:
    """Run one runpod_parallel subcommand, streaming its output. Returns rc."""
    cmd = [sys.executable, "-m", "scripts.phase1b.runpod_parallel", *args]
    logger.info("$ %s", " ".join(cmd[2:]))
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


def gate_a_gpu_is_fast_enough() -> bool:
    """Every pod landed on a part that can finish a fold inside the timeout."""
    import json

    state_path = PROJECT_ROOT / "scripts" / "phase1b" / ".pod_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pods = state.get("pods") or []
    if not pods:
        logger.error("Gate A: state file records no pods.")
        return False
    ok = True
    for pod in pods:
        gpu = pod.get("gpu_type")
        cloud = pod.get("cloud_type")
        verdict = "OK" if gpu in ACCEPTABLE_GPUS else "TOO SLOW"
        logger.info("  Gate A: %s -> %s (%s) %s", pod.get("name"), gpu, cloud, verdict)
        ok = ok and gpu in ACCEPTABLE_GPUS
    expected = len(select_pod_configs(None, "validation"))
    if len(pods) != expected:
        logger.error("Gate A: %d pods, expected %d. A LOFO arm short one fold "
                     "cannot produce a number.", len(pods), expected)
        ok = False
    return ok


def gate_b_ssh_actually_authenticates() -> bool:
    """One real handshake. `provision` proves port 22 answered, nothing more.

    _wait_for_ssh returns on a bare socket connect, so "SSH-ready" in the
    provision log is not evidence that our key works. The first real
    authentication happens inside bootstrap with check=False, where a
    `Permission denied (publickey)` is one warning among many and then surfaces
    as an opaque rsync failure minutes later, on a billing fleet.
    """
    import json

    state_path = PROJECT_ROOT / "scripts" / "phase1b" / ".pod_state.json"
    pods = (json.loads(state_path.read_text(encoding="utf-8")).get("pods") or [])
    if not pods:
        return False
    # WAIT for the endpoint rather than failing on its absence.
    #
    # This cost a fleet. On 2026-08-27 at 16:50 all five pods created cleanly on
    # A100-SXM4-80GB, Gate A passed, and this check ran the instant provision
    # returned -- while RunPod had not yet published an IP. It read "no ip/port
    # recorded" as a failed handshake and tore down five good pods during the
    # only capacity window of the afternoon. A pod sits at ip=pending for a
    # minute or two after creation; that is normal, not broken. Absence of an
    # endpoint is "not yet", and only its continued absence is a failure.
    deadline = time.monotonic() + ENDPOINT_WAIT_S
    pod = pods[0]
    ip = pod.get("ip")
    port = pod.get("port")
    while (not ip or not port) and time.monotonic() < deadline:
        waited = int(ENDPOINT_WAIT_S - (deadline - time.monotonic()))
        logger.info("  Gate B: %s has no endpoint yet (%ds); waiting.",
                    pod.get("name"), waited)
        time.sleep(ENDPOINT_POLL_S)
        pods = (json.loads(state_path.read_text(encoding="utf-8")).get("pods") or [])
        pod = next((p for p in pods if p.get("name") == pod.get("name")), pod)
        ip, port = pod.get("ip"), pod.get("port")
    if not ip or not port:
        logger.error("Gate B: pod %s still has no ip/port after %ds. That is no "
                     "longer 'not yet'. Recorded keys on that pod: %s",
                     pod.get("name"), ENDPOINT_WAIT_S, sorted(pod))
        return False
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=accept-new",
         "-o", f"UserKnownHostsFile={PROJECT_ROOT}/scripts/phase1b/.runpod_known_hosts",
         "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=30",
         "-i", str(Path.home() / ".ssh" / "tract_runpod"),
         "-p", str(port), f"root@{ip}", "id"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    out = (result.stdout or "").strip()
    logger.info("  Gate B: %s -> rc=%d %s", pod.get("name"), result.returncode,
                out or (result.stderr or "").strip()[:120])
    return result.returncode == 0 and "uid=0" in out


def attempt_arm(arm: Arm) -> bool:
    """Provision, gate, train, collect, tear down. True only on a complete arm."""
    base = ("--config-name", arm.config_name, "--split", arm.split, *arm.flags)

    # The same distinction full_pipeline draws, and for the same reason:
    # "leaving pods up costs money; terminating them costs the run." Teardown is
    # safe only when there is nothing on those pods worth more than the hourly
    # rate. Before training, that is always true -- nothing has been computed.
    # After training and before a successful collect, it is never true, because
    # the per-item indicators exist ONLY on the pods and a fold cannot be
    # rebuilt from the fleet once it is gone.
    nothing_to_lose = True
    results_are_safe = False
    try:
        # Re-arm the reaper before creating anything. This watcher can poll for
        # twelve hours; the guard disarms after three consecutive quiet checks,
        # which is about six. So by the time capacity appears the independent
        # spend bound may be long gone -- and the fleet it would have to bound is
        # the one we are about to create, unattended, possibly at 3am. Arming is
        # idempotent enough: a live unit name collides and is skipped, which is
        # the harmless direction.
        arm_the_reaper()

        if orchestrator("provision", *base) != 0:
            logger.error("provision failed. Nothing has been computed, so the "
                         "fleet is torn down.")
            return False
        if not gate_a_gpu_is_fast_enough():
            logger.error("Gate A failed. Not spending training hours on a fleet "
                         "whose folds would die at FOLD_TIMEOUT_S.")
            return False
        if not gate_b_ssh_actually_authenticates():
            logger.error("Gate B failed: SSH did not authenticate. Tearing down "
                         "now rather than discovering it inside bootstrap, "
                         "minutes later, with five pods billing.")
            return False

        # From here a pod may hold the only copy of a fold's indicators.
        nothing_to_lose = False

        run_rc = orchestrator("run", *base)
        collect_rc = orchestrator("collect", *base)

        if collect_rc != 0:
            logger.error(
                "collect FAILED. Pods are being LEFT RUNNING and are still "
                "billing, deliberately: a fold's per-item indicators exist only "
                "on them and teardown would destroy paid-for work. Recover with "
                "`runpod_parallel collect --config-name %s --split %s`, then "
                "teardown by hand. The reaper guard is the backstop if nobody "
                "does.", arm.config_name, arm.split,
            )
            return False

        if run_rc != 0:
            # Collected, so the fleet is expendable -- but the arm is incomplete
            # and a partial fold set cannot produce a LOFO number.
            results_are_safe = True
            logger.error("run reported failed folds. What landed has been "
                         "collected; the arm is incomplete and needs a human.")
            return False

        results_are_safe = True
        return True
    finally:
        if nothing_to_lose or results_are_safe:
            orchestrator("teardown")
            swept = sweep_account()
            if swept:
                logger.warning("Swept %d pod(s) teardown did not reach -- the "
                               "pods=[] window again.", swept)
        else:
            logger.error("NOT tearing down. Pods hold uncollected results.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--confirm", action="store_true",
                        help="actually provision. Without it, polls and reports only.")
    args = parser.parse_args(argv)
    arm = ARMS[args.arm]

    logger.info("Waiting for capacity to run arm %s (%s, split=%s, flags=%s).",
                arm.key, arm.config_name, arm.split, " ".join(arm.flags) or "(none)")
    logger.info("Acceptable parts: %s", ", ".join(ACCEPTABLE_GPUS))
    logger.info("Trigger: any of them at %s. Polling every %d min, giving up after %d h.",
                " or ".join(sorted(SUFFICIENT_STOCK)), POLL_SECONDS // 60,
                MAX_WALL_SECONDS // 3600)
    if not args.confirm:
        logger.warning("DRY RUN: --confirm not given, nothing will be provisioned.")

    started = time.monotonic()
    attempts = 0
    while time.monotonic() - started < MAX_WALL_SECONDS:
        readings = survey()
        good = [(g, s) for g, s in readings if s in SUFFICIENT_STOCK]
        elapsed = int((time.monotonic() - started) // 60)
        logger.info("[%3d min] %s", elapsed,
                    "  ".join(f"{g.split()[1]}={s}" for g, s in readings))
        if good:
            logger.warning("Capacity: %s", ", ".join(f"{g}={s}" for g, s in good))
            if not args.confirm:
                logger.warning("Would provision now, but this is a dry run.")
                return EXIT_OK
            attempts += 1
            logger.warning("Provision attempt %d of %d.", attempts, MAX_PROVISION_ATTEMPTS)
            if attempt_arm(arm):
                logger.warning("Arm %s COMPLETE and collected. Pods are down. "
                               "Aggregate is deliberately NOT run -- it needs "
                               "--n-configurations 3 and a human to read it.", arm.key)
                return EXIT_OK
            if attempts >= MAX_PROVISION_ATTEMPTS:
                logger.error("Gave up after %d attempts. Capacity appears and "
                             "evaporates faster than the signal predicts; a "
                             "human should look.", attempts)
                return EXIT_GAVE_UP
        time.sleep(POLL_SECONDS)

    logger.error("Gave up after %d h without sufficient capacity.", MAX_WALL_SECONDS // 3600)
    return EXIT_GAVE_UP


if __name__ == "__main__":
    sys.exit(main())
