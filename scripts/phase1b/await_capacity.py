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

Every failure path either tears the fleet down or STOPS, loudly, naming the pods
it deliberately left up; it never returns to polling with a fleet running. The
one incident this apparatus already had was an interrupt landing in the window
where .pod_state.json records pods=[], which left `teardown` reporting "nothing
scoped to terminate" while four pods billed -- so teardown here is followed by
an independent sweep against the account, by name, across both splits.

WHAT $? MEANS, because the two failures that leave money running have to be
distinguishable by a wrapper and by a human reading a log at 3am:
    0  the arm is complete and collected, and the pods are down.
    1  gave up -- capacity never appeared, or every attempt failed. Nothing is
       running.
    2  refused before creating anything. The reaper could not be armed.
    3  STOPPED WITH PODS UP. A fleet holds uncollected folds and is billing
       until someone collects it and tears it down.

Owner: TRACT.
"""

from __future__ import annotations

import argparse
import enum
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scripts.phase0.runpod_provision import (
    SSH_POLL_TIMEOUT_S,
    _gql,
    get_running_pods,
    terminate_pods,
)
# One number, not two. The arm command below used to ask for --on-active=8h
# while the guard's own cadence was 2h, so the two halves of the same control
# disagreed about how often anything gets looked at.
from scripts.phase1b.reaper_guard import REARM_SECONDS
from scripts.phase1b.runpod_parallel import (
    BOOTSTRAP_DEADLINE_S,
    BOOTSTRAP_DEADLINE_SLACK_S,
    FOLD_TIMEOUT_S,
    PROJECT_ROOT,
    RSYNC_PULL_ATTEMPTS,
    RSYNC_PULL_BACKOFF_S,
    RSYNC_PULL_TIMEOUT_S,
    select_pod_configs,
)

# NOT in the repository. This runs for up to twelve hours and the operator will
# want to read it afterwards, so it cannot go under XDG_RUNTIME_DIR, which is
# wiped at logout -- but a log file in PROJECT_ROOT shows up as untracked and
# breaks the clean-tree check the runbook's Gate 0 makes before every provision,
# which is the check that keeps git_sha from being confidently wrong.
#
# Overridable because importing this module OPENS the log, and the test suite
# imports it: without the override, `pytest` during a live campaign interleaves
# fixture output into the one file an incident review reads afterwards.
LOG_DIR_ENV: Final[str] = "TRACT_CAMPAIGN2_LOG_DIR"
LOG_DIR: Final[Path] = Path(
    os.environ.get(LOG_DIR_ENV) or Path.home() / "tract-campaign2-logs"
)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Attached to the root logger explicitly rather than through basicConfig.
# basicConfig is a NO-OP when the root logger already has handlers, and
# runpod_parallel.py calls it at import -- which the import block above does,
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
# Distinct from EXIT_GAVE_UP because the two demand opposite responses from
# whoever reads $?: "gave up, nothing is running" can wait until morning, and
# "stopped, five pods are still billing" cannot.
EXIT_FLEET_HELD: Final[int] = 3

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

# WHEN THE REAPER SHOULD FIRST LOOK, for the window this module opens.
#
# Never later than the guard's own steady-state cadence, which is why
# REARM_SECONDS is imported rather than restated: a first look scheduled after
# the interval the guard then re-arms itself on is a number that can only be
# wrong. Shorter than it, deliberately, because of what the window contains.
#
# Between provision and the first fold there is nothing to protect: no fold has
# run, no indicator exists, and a fleet nobody is driving is pure loss. The
# guard already knows the difference -- it stands down and re-arms whenever a
# runpod_parallel process is alive, and asks every pod whether a fold is running
# before it touches anything -- so an early check costs a log line while the arm
# is healthy and bounds an ORPHANED pre-training fleet to under an hour. At the
# 2026-08-27 rate of $3.29/hr x 5 pods that is ~$12 of exposure against the ~$130
# the old --on-active=8h permitted, three times that whole night's loss.
#
# The objection to answer is what an early check might destroy. Two windows have
# pods up with no runpod_parallel process alive: Gate B's endpoint wait, where
# nothing has been computed and being reaped costs a re-provision rather than a
# fold; and a fleet held for uncollected results, which cannot exist inside the
# first 45 minutes -- a bootstrap plus a fold is longer than that. By then the
# guard is on its own REARM cadence and this number no longer applies, so
# shortening it does not shorten anyone's window to rescue paid-for work.
FIRST_REAPER_CHECK_SECONDS: Final[int] = min(45 * 60, REARM_SECONDS)
FIRST_REAPER_CHECK: Final[str] = f"{FIRST_REAPER_CHECK_SECONDS // 60}min"
# systemd-run only registers a timer unit; it does not run the job. Thirty
# seconds is a hang, not a slow start, and hanging here would hold up the
# provision this is a precondition for.
REAPER_ARM_TIMEOUT_S: Final[int] = 30

# WHY EVERY STAGE NEEDS ITS OWN WALL.
#
# orchestrator() called subprocess.run with no timeout, so a wedged child owned
# the watcher: MAX_WALL_SECONDS is re-read at the top of the polling loop, which
# a running stage never reaches. On 2026-08-27 one pod sat in rsync inside the
# bootstrap barrier for ninety minutes and the watcher waited out every minute
# of it, with four other pods billing and no way to reconsider.
#
# Sized UP from the orchestrator's own ceilings rather than guessed, because the
# two failures are not symmetric: a wall that fires early kills training that
# has already been paid for, while one that fires late costs the hourly rate.
# These are ceilings-of-ceilings, and their only job is to guarantee the watcher
# gets control back. `run` and `collect` reuse the same reachable-wall model
# _check_budget prices, so no stage can outlive the spend the budget gate
# already authorised.
STAGE_TIMEOUT_MARGIN: Final[float] = 1.25
# Sized on the LARGER roster, not on "validation". That comment used to read
# "every Campaign 2 arm is a validation arm", which stopped being true the
# moment the TEST arm was registered here. Both rosters hold five frameworks
# today, so the old constant was right by coincidence; adding or dropping one
# validation framework would have sized the test round's timeouts against a
# fleet it never provisions.
FLEET_SIZE: Final[int] = max(
    len(select_pod_configs(None, split)) for split in ("validation", "test")
)
# provision walks a fallback ladder of GPU types when one is out of capacity,
# and every rung waits SSH_POLL_TIMEOUT_S for endpoints before giving up.
PROVISION_TIMEOUT_S: Final[int] = int(
    STAGE_TIMEOUT_MARGIN * len(ACCEPTABLE_GPUS) * SSH_POLL_TIMEOUT_S
)
# The bootstrap barrier plus one fold, the same two terms _check_budget prices.
# Imported rather than restated so that shortening a bootstrap deadline over
# there shortens this wall here, instead of leaving a stale number that quietly
# permits more than the budget gate was told about.
RUN_TIMEOUT_S: Final[int] = int(
    STAGE_TIMEOUT_MARGIN
    * (BOOTSTRAP_DEADLINE_S + BOOTSTRAP_DEADLINE_SLACK_S + FOLD_TIMEOUT_S)
)
# A SERIAL rsync per pod, each pod able to pay the full pull ladder including
# its backoffs -- _check_budget's collect term exactly. This is the largest wall
# here by a wide margin; the ladder itself is the orchestrator's to shorten.
COLLECT_TIMEOUT_S: Final[int] = int(
    STAGE_TIMEOUT_MARGIN * FLEET_SIZE * (
        RSYNC_PULL_ATTEMPTS * RSYNC_PULL_TIMEOUT_S
        + RSYNC_PULL_BACKOFF_S * RSYNC_PULL_ATTEMPTS * (RSYNC_PULL_ATTEMPTS - 1) // 2
    )
)
# Terminate calls against an API, with no SSH and no data movement. A teardown
# that cannot finish in fifteen minutes is wedged rather than slow, and the
# account sweep that follows it is the real backstop.
TEARDOWN_TIMEOUT_S: Final[int] = 15 * 60
STAGE_TIMEOUT_S: Final[dict[str, int]] = {
    "provision": PROVISION_TIMEOUT_S,
    "run": RUN_TIMEOUT_S,
    "collect": COLLECT_TIMEOUT_S,
    "teardown": TEARDOWN_TIMEOUT_S,
}
# timeout(1)'s convention, so a wedged stage reads unambiguously in the log and
# is never mistaken for a subcommand's own exit code. Non-zero above all: every
# caller here tests the return value, and a wedge is a failure.
STAGE_TIMEOUT_RC: Final[int] = 124


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
    # The single test round. A3 advanced by the tie-break at CAMPAIGN2.md:168-169
    # -- both non-primary arms cleared A1 by more than the MDE, so the higher
    # absolute validation hit@1 advances. Runs ONCE on the 147-item AI split and
    # is unrecoverable: a second run would contaminate the split that the whole
    # validation/test separation exists to protect.
    "TEST": Arm("TEST", "c2r_TEST_A3_prose_sw_qwen06b", "test",
                ("--stopwords", "--base-model", "Qwen/Qwen3-Embedding-0.6B")),
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
    """Ensure a scheduled reaper exists before a fleet does, or RAISE.

    This used to log a warning and carry on, reasoning that the backstop is not
    the only control because this module tears down on every path. The premortem
    that followed the 2026-08-27 night rejected that: the paths this module
    tears down on are the ones it survives to reach. A SIGKILL, an OOM, a closed
    tmux, or the stranding bug fixed in attempt_arm below all end with pods up
    and no code left running to notice -- which is precisely the scenario the
    reaper exists for, and precisely the scenario in which "proceeding without
    it" means creating a billing fleet with no bound on it at all.

    So it is a precondition now. Refusing costs a capacity window; proceeding
    costs whatever the fleet bills until a human happens to look.

    Raises:
        RuntimeError: if no timer could be armed, for any reason.
    """
    unit = f"tract-reaper-await-{int(time.time())}"
    try:
        subprocess.run(
            ["systemd-run", "--user", "--collect",
             f"--on-active={FIRST_REAPER_CHECK}",
             f"--unit={unit}", f"--working-directory={PROJECT_ROOT}",
             "--setenv=USE_TF=0",
             sys.executable, "-m", "scripts.phase1b.reaper_guard", "--confirm"],
            check=True, capture_output=True, timeout=REAPER_ARM_TIMEOUT_S,
        )
    # OSError covers the host that has no systemd-run at all, which is the one
    # form of this failure that is a property of the machine rather than a hiccup.
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise RuntimeError(
            f"could not arm the reaper as {unit} ({exc}: "
            f"{detail.decode('utf-8', 'replace').strip()}). A fleet created "
            f"now would have no bound on it except this process staying alive. "
            f"Check `systemctl --user list-timers 'tract-reaper*'` and whether "
            f"the user manager is running (loginctl enable-linger)."
        ) from exc
    logger.info("Reaper armed as %s (first check T+%s) before provisioning.",
                unit, FIRST_REAPER_CHECK)


def orchestrator(*args: str) -> int:
    """Run one runpod_parallel subcommand, streaming its output. Returns rc.

    A stage that outruns its wall is reported as FAILED rather than waited on.
    The child is killed; the ssh and rsync processes it spawned are its own
    children and outlive it, and whatever the pods were doing continues -- the
    teardown path in attempt_arm and the reaper guard are what answer for the
    remote side. What this buys is the watcher's own control back.

    Raises:
        ValueError: for a stage with no declared wall. Every caller here is
            inside this file, so that is a typo, and running an unbounded child
            against a billing fleet is not the way to discover it.
    """
    stage = args[0]
    timeout = STAGE_TIMEOUT_S.get(stage)
    if timeout is None:
        raise ValueError(
            f"No wall is declared for orchestrator stage {stage!r}. Known "
            f"stages: {sorted(STAGE_TIMEOUT_S)}."
        )
    cmd = [sys.executable, "-m", "scripts.phase1b.runpod_parallel", *args]
    logger.info("$ %s   [wall %dm]", " ".join(cmd[2:]), timeout // 60)
    try:
        return subprocess.run(
            cmd, cwd=PROJECT_ROOT, check=False, timeout=timeout,
        ).returncode
    except subprocess.TimeoutExpired:
        logger.error(
            "'%s' exceeded its %dm wall and was killed. It is treated as a "
            "failed stage: the pods may still be working, but nothing here is "
            "watching them any more.", stage, timeout // 60,
        )
        return STAGE_TIMEOUT_RC


def gate_a_gpu_is_fast_enough(split: str = "validation") -> bool:
    """Every pod landed on a part that can finish a fold inside the timeout.

    The expected pod count follows the ARM's split. It was hardcoded to
    "validation" while ARMS["TEST"] provisions the test roster: both hold five
    frameworks today, so the gate passed by coincidence. Change either roster
    and a fleet short one fold passes the gate, or a healthy fleet fails it and
    the single unrepeatable test round loses its capacity window.
    """
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
    expected = len(select_pod_configs(None, split))
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


class ArmOutcome(enum.Enum):
    """What an attempt left behind, which is what the caller has to know.

    A bool could not carry the distinction that cost the money. "Failed, the
    pods are down" and "failed, the pods are deliberately still up" were both
    False, and main() retried both -- and provisioning again rewrites
    .pod_state.json with the new fleet's ids, under pod names identical to the
    old fleet's. One retry over a held fleet makes the held fleet invisible to
    `teardown` and leaves only the name-based sweep able to find it at all.
    """

    COMPLETE = "complete"
    #: Failed, and nothing is billing. Safe to try again.
    FLEET_DOWN = "fleet_down"
    #: Failed with pods UP on purpose, because they hold uncollected folds.
    FLEET_HELD = "fleet_held"
    #: Nothing was created: a precondition said no.
    REFUSED = "refused"


def attempt_arm(arm: Arm) -> ArmOutcome:
    """Provision, gate, train, collect, tear down. One arm, one fleet."""
    base = ("--config-name", arm.config_name, "--split", arm.split, *arm.flags)

    # Re-arm the reaper before creating anything, and before the try, because a
    # refusal here must not take the teardown path: no fleet exists yet, and the
    # sweep would be an API call answering a question nobody asked.
    #
    # This watcher can poll for twelve hours; the guard disarms after three
    # consecutive quiet checks. So by the time capacity appears the independent
    # spend bound may be long gone -- and the fleet it would have to bound is
    # the one about to be created, unattended, possibly at 3am. Arming is
    # idempotent enough: a live unit name collides and is skipped, which is the
    # harmless direction.
    try:
        arm_the_reaper()
    except RuntimeError as exc:
        logger.error("REFUSING to provision: %s", exc)
        return ArmOutcome.REFUSED

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
        if orchestrator("provision", *base) != 0:
            logger.error("provision failed. Nothing has been computed, so the "
                         "fleet is torn down.")
            return ArmOutcome.FLEET_DOWN
        if not gate_a_gpu_is_fast_enough(arm.split):
            logger.error("Gate A failed. Not spending training hours on a fleet "
                         "whose folds would die at FOLD_TIMEOUT_S.")
            return ArmOutcome.FLEET_DOWN
        if not gate_b_ssh_actually_authenticates():
            logger.error("Gate B failed: SSH did not authenticate. Tearing down "
                         "now rather than discovering it inside bootstrap, "
                         "minutes later, with five pods billing.")
            return ArmOutcome.FLEET_DOWN

        run_rc = orchestrator("run", *base)
        if run_rc != 0:
            # THE STRANDING BUG OF 2026-08-27, and the reason this branch exists
            # at all. `collect` used to run here unconditionally, and its exit
            # code alone decided whether the fleet lived. A run that aborts in
            # the bootstrap barrier computes nothing and never creates the
            # remote results directory, so rsync exits 23 -- and the failure
            # that PROVED the pods were empty was read as proof that they were
            # full. Five pods spent the night billing on a fleet holding
            # nothing, and were terminated by hand the next morning.
            #
            # Nothing here can tell a bootstrap abort from two dead folds: both
            # are exit 1 out of `run`. Teardown is the answer to both. A partial
            # fold set cannot produce a LOFO number, so the alternative is
            # paying the hourly rate while a human reaches the same conclusion.
            # The recoverable case is the one below, where `run` SUCCEEDED and
            # there is therefore something on those pods worth holding them for.
            logger.error(
                "run failed (rc=%d) and collect is deliberately NOT being "
                "attempted: a run that aborted before its folds has nothing to "
                "collect, and it was collect's failure on an empty fleet that "
                "stranded five pods on 2026-08-27. Any fold that did finish "
                "goes with the fleet -- a partial arm cannot produce a number. "
                "The fold-by-fold detail is above, in `run`'s own output.",
                run_rc,
            )
            return ArmOutcome.FLEET_DOWN

        # From here a pod may hold the only copy of a fold's indicators: `run`
        # returned 0, which the orchestrator emits only when every fold in the
        # roster reported ok.
        nothing_to_lose = False

        if orchestrator("collect", *base) != 0:
            logger.error(
                "collect FAILED after a clean run. Pods are being LEFT RUNNING "
                "and are still billing, deliberately: a fold's per-item "
                "indicators exist only on them and teardown would destroy "
                "paid-for work. Recover with `runpod_parallel collect "
                "--config-name %s --split %s`, then teardown by hand. The "
                "reaper guard is the backstop if nobody does.",
                arm.config_name, arm.split,
            )
            return ArmOutcome.FLEET_HELD

        results_are_safe = True
        return ArmOutcome.COMPLETE
    finally:
        if nothing_to_lose or results_are_safe:
            orchestrator("teardown")
            # sweep_account raises when a pod refuses to terminate, and
            # get_running_pods raises on any API or network error. Raising from
            # a `finally` REPLACES the return value: the ArmOutcome is
            # discarded, the exception escapes main(), and the process exits 1
            # -- which this module documents as "gave up ... Nothing is
            # running". That is the precise inverse of the truth in the case
            # that matters, a pod that would not die and is still billing. The
            # one distinction the exit codes exist to draw is the one the
            # unguarded raise destroys.
            try:
                swept = sweep_account()
            except Exception as exc:  # noqa: BLE001 - must not mask the outcome
                logger.error(
                    "SWEEP FAILED after teardown: %s. Pods may still be "
                    "billing. Check the RunPod console; do not read this run's "
                    "exit code as evidence the account is clear.", exc,
                )
            else:
                if swept:
                    logger.warning("Swept %d pod(s) teardown did not reach -- "
                                   "the pods=[] window again.", swept)
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
            outcome = attempt_arm(arm)
            if outcome is ArmOutcome.COMPLETE:
                logger.warning("Arm %s COMPLETE and collected. Pods are down. "
                               "Aggregate is deliberately NOT run -- it needs "
                               "--n-configurations 3 and a human to read it.", arm.key)
                return EXIT_OK
            if outcome is ArmOutcome.REFUSED:
                # Nothing was created, and nothing will be: whatever refused
                # will refuse the next attempt identically. Retrying would only
                # spend the capacity window.
                logger.error("Attempt %d was REFUSED before anything was "
                             "created. Not retrying: the precondition that said "
                             "no will say no again.", attempts)
                return EXIT_REFUSED
            if outcome is ArmOutcome.FLEET_HELD:
                # The one state in which polling again would be actively
                # destructive: provision() rewrites .pod_state.json, both fleets
                # carry the same pod names, and the held fleet -- the one with
                # uncollected folds on it -- would stop being reachable by
                # `teardown` at all.
                logger.error(
                    "STOPPING with pods UP. Attempt %d left a fleet holding "
                    "uncollected folds and the watcher will NOT provision over "
                    "it: that would overwrite .pod_state.json and make this "
                    "fleet unreachable by teardown. Collect it, then tear it "
                    "down. Until you do, it is billing.", attempts,
                )
                return EXIT_FLEET_HELD
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
