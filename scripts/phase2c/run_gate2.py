"""Phase 2C Gate 2: four arms, one pod, in sequence.

`runpod_retrain` provisions and tears down safely but runs
`t1_calibrate_and_train` and `t2_inference_and_calibrate` -- the Phase 1C
active-learning scripts, not `run_fold`. It cannot run a Gate 2 arm. This module
reuses its hardened pod lifecycle (price ceiling, wall clock, try/finally
teardown, a pod name the reaper sweeps) and runs the right command.

ONE TRAINING RUN PER ARM, scoring all 74 items. Not one per framework: Gate 2's
primary estimand is a difference between exposed and unexposed items, and those
strata must come from the SAME model. Training per framework would put
training-draw variance BETWEEN the strata, which is precisely the variance the
difference-in-differences exists to cancel.

THE ARMS

    A0    bridge-free                       the comparator
    A0R   placebo, size- and hub-matched    isolates judgement from "any positive"
    A1    round-2 union, confidence >= 2    the treatment
    A0P   bridge-free, seed+1               the noise floor

A0P is what makes a number readable. Two same-arm runs in this repository differ
by 19% per-item discordance, so without a measured floor a delta of +0.08 cannot
be told from a differently-shuffled fine-tune. It runs LAST: if capacity or
budget fails mid-sequence, the arms that answer the question are already home.

Each arm gets its own `--config-name`, hence its own output directory and WandB
run id. Seed is deliberately not arm-defining, so A0 and A0P under one name
would share a directory and overwrite each other.

Every arm is pod work. CLAUDE.md: all inference and training runs on RunPod,
never locally.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from scripts.phase1c.runpod_retrain import (
    MAX_RUN_HOURS,
    MAX_USD_PER_HOUR,
    _bootstrap,
    _get_pod_env,
    _load_pod_state,
    _rsync_from,
    _ssh,
    provision,
    teardown,
)
from tract.config import (
    PHASE1B_SEED,
    PHASE2C_GATE2_EVAL_FRAMEWORKS,
    PROJECT_ROOT,
    TRAINING_DIR,
)

logger = logging.getLogger(__name__)

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1b"
ARM_TIMEOUT_S: Final[int] = 10800


@dataclass(frozen=True)
class Arm:
    """One trained configuration, and what makes it that one."""

    name: str
    config_name: str
    bridge_corpus: Path | None
    seed: int
    purpose: str


def gate2_arms(round_label: str = "r2") -> list[Arm]:
    """The four arms, in the order they run.

    A1 before A0P deliberately: the treatment and its two comparators are what
    the gate needs, and the noise floor -- while it is what makes a null
    readable -- is the one that can be re-run later on its own.
    """
    return [
        Arm("A0", "gate2_A0", None, PHASE1B_SEED,
            "bridge-free comparator"),
        Arm("A1", "gate2_A1",
            TRAINING_DIR / f"hub_links_bridge.{round_label}.jsonl",
            PHASE1B_SEED, "the treatment"),
        Arm("A0R", "gate2_A0R",
            TRAINING_DIR / "hub_links_bridge.placebo.jsonl",
            PHASE1B_SEED, "placebo: judgement vs any positive"),
        Arm("A0P", "gate2_A0P", None, PHASE1B_SEED + 1,
            "noise floor: the comparator at another seed"),
    ]


def _remote_command(arm: Arm) -> str:
    """The remote invocation, with two things the first run got wrong.

    `mkdir -p results/phase1b` because POD_RSYNC_EXCLUDES excludes `results`
    wholesale -- correctly, since it holds the Tier-3 quarantined export and the
    ceiling study's LLM-written descriptions, and nothing on a pod reads any of
    it. But that means /workspace/tract/results does not exist, so `tee` failed
    to open its log the instant the arm started. The first Gate 2 attempt
    trained for 44 minutes and then died on that, having evaluated nothing.

    `set -o pipefail` because without it a pipeline returns the exit code of
    `tee`, not of python. The first attempt failed loudly only because tee was
    the thing that broke; the reverse -- training dies, tee writes an empty log
    and exits 0 -- would have reported SUCCESS and collected nothing, which is
    the far more expensive direction.
    """
    bridge = (
        f"--bridge-links {arm.bridge_corpus.relative_to(PROJECT_ROOT)} "
        if arm.bridge_corpus else ""
    )
    return (
        "set -o pipefail && "
        "cd /workspace/tract && "
        f"mkdir -p results/phase1b && "
        "python -m scripts.phase1b.run_fold "
        "--split gate2 --framework ALL "
        f"--config-name {arm.config_name} "
        f"--seed {arm.seed} "
        f"{bridge}"
        f"2>&1 | tee results/phase1b/{arm.config_name}_log.txt"
    )


def preflight(arms: list[Arm]) -> None:
    """Refuse to rent anything until every input exists.

    A missing corpus is not a crash on a pod -- `load_bridge_links` raises, the
    arm dies, and the pod bills through the debugging. Checking here costs
    nothing and it is the difference between a typo and an hour of H100.
    """
    for arm in arms:
        if arm.bridge_corpus is not None and not arm.bridge_corpus.is_file():
            raise FileNotFoundError(
                f"{arm.name}: {arm.bridge_corpus} does not exist. Build it "
                "with scripts.build_gate2_corpus (treatment) or "
                "scripts.build_placebo_corpus (placebo) before provisioning."
            )
    strata = PROJECT_ROOT / "results" / "phase2c" / "gate2_strata.json"
    if not strata.is_file():
        raise FileNotFoundError(
            f"{strata} is missing. The exposure partition is pre-registered and "
            "committed BEFORE the run -- computing it afterwards makes it the "
            "post-hoc stratum this project has withdrawn headlines over. Run "
            "scripts.analysis.gate2_strata first."
        )
    seeds = {(a.config_name, a.seed) for a in arms}
    if len({c for c, _ in seeds}) != len(arms):
        raise ValueError(
            "two arms share a --config-name, so they would share an output "
            "directory and overwrite each other"
        )
    logger.info(
        "Preflight OK: %d arms, eval=%s",
        len(arms), sorted(PHASE2C_GATE2_EVAL_FRAMEWORKS),
    )


def remote_preflight(pod: dict[str, Any], arms: list[Arm]) -> None:
    """Prove the pod can run an arm before spending 45 minutes finding out.

    The first Gate 2 attempt trained for 44 minutes and died writing its log,
    because `results/` is excluded from the rsync and so the directory did not
    exist. Every check here costs about ten seconds and would have caught it:
    the point is that an arm's FIRST failure should be cheap, and a training
    step is the most expensive place to discover a missing directory.
    """
    ip, port = str(pod["ip"]), int(pod["port"])
    corpora = " ".join(
        str(arm.bridge_corpus.relative_to(PROJECT_ROOT))
        for arm in arms if arm.bridge_corpus is not None
    )
    _ssh(ip, port, (
        "set -o pipefail && cd /workspace/tract && "
        # The directory the rsync deliberately does not ship.
        "mkdir -p results/phase1b && test -d results/phase1b && "
        # The corpora each treatment arm reads. load_bridge_links raises on a
        # missing file, which on a pod means a dead arm and a billing GPU.
        f"for f in {corpora}; do test -s \"$f\" || "
        "{ echo \"MISSING CORPUS: $f\"; exit 1; }; done && "
        # The module imports, so a syntax or import error is not discovered
        # after the model is loaded.
        "python -c 'import scripts.phase1b.run_fold' && "
        "echo REMOTE_PREFLIGHT_OK"
    ), timeout=300)
    logger.info("Remote preflight OK: results dir, %d corpora, imports clean",
                len([a for a in arms if a.bridge_corpus]))


def run_arm(arm: Arm) -> None:
    """Train and score one arm on the provisioned pod, then collect it."""
    pod = _load_pod_state()
    ip, port = pod["ip"], pod["port"]

    logger.info("=" * 66)
    logger.info("ARM %s (%s)", arm.name, arm.purpose)
    logger.info("  config : %s", arm.config_name)
    logger.info("  bridge : %s", arm.bridge_corpus or "none")
    logger.info("  seed   : %d", arm.seed)
    logger.info("=" * 66)

    started = time.time()
    _ssh(ip, port, _remote_command(arm),
         env=_get_pod_env(), timeout=ARM_TIMEOUT_S)
    logger.info("Arm %s trained in %.1fm", arm.name, (time.time() - started) / 60)

    # Collected immediately, not at the end. An arm whose results are still on
    # the pod when the next one starts is an arm that a teardown loses, and the
    # money is already spent.
    local = RESULTS_DIR / arm.config_name
    local.mkdir(parents=True, exist_ok=True)
    _rsync_from(ip, port, f"/workspace/tract/results/phase1b/{arm.config_name}/",
                f"{local}/")
    if not any(local.glob("fold_*/fold_result.json")):
        raise RuntimeError(
            f"Arm {arm.name} produced no fold result under {local}. The arm "
            "cost money and its output is not here; do not continue to the "
            "next arm, which would overwrite the pod-side directory."
        )
    logger.info("Arm %s collected to %s", arm.name, local)


def full_pipeline(round_label: str) -> None:
    arms = gate2_arms(round_label)
    preflight(arms)

    logger.info("=" * 66)
    logger.info("PHASE 2C GATE 2 -- %d arms on one pod", len(arms))
    logger.info("  price ceiling : $%.2f/hr", MAX_USD_PER_HOUR)
    logger.info("  wall clock    : %.1f h", MAX_RUN_HOURS)
    for arm in arms:
        logger.info("    %-4s %s", arm.name, arm.purpose)
    logger.info("=" * 66)

    started = time.time()
    deadline = started + MAX_RUN_HOURS * 3600
    completed: list[str] = []
    try:
        provision()
        pod = _load_pod_state()
        _bootstrap(pod)
        # Belt as well as braces: every arm's command creates this too, but a
        # directory the rsync deliberately does not ship should exist before
        # anything tries to write into it.
        remote_preflight(pod, arms)
        for arm in arms:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Wall clock of {MAX_RUN_HOURS}h reached with "
                    f"{len(arms) - len(completed)} arm(s) unrun. Tearing down "
                    "rather than billing on; completed arms are collected and "
                    "the rest can resume on a fresh pod."
                )
            run_arm(arm)
            completed.append(arm.name)
    finally:
        if len(completed) < len(arms):
            logger.error(
                "Completed %d of %d arms (%s). Tearing the pod down anyway -- "
                "an unfinished sequence is cheaper than an unattended GPU.",
                len(completed), len(arms), ", ".join(completed) or "none",
            )
        teardown()

    logger.info("All %d arms complete in %.1fm", len(arms),
                (time.time() - started) / 60)
    logger.info("")
    logger.info("Score with:")
    logger.info(
        "  python -m scripts.analysis.gate2_delta "
        "--comparator results/phase1b/gate2_A0 "
        "--treatment results/phase1b/gate2_A1 "
        "--strata results/phase2c/gate2_strata.json "
        "--out results/phase2c/gate2_primary.json --label 'A1 vs A0 (PRIMARY)'"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="preflight",
                        choices=["preflight", "full", "provision", "teardown"],
                        help="Defaults to preflight, which rents nothing.")
    parser.add_argument("--round-label", default="r2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.action == "preflight":
        preflight(gate2_arms(args.round_label))
        for arm in gate2_arms(args.round_label):
            logger.info("  %-4s %s", arm.name, _remote_command(arm))
    elif args.action == "full":
        full_pipeline(args.round_label)
    elif args.action == "provision":
        provision()
    elif args.action == "teardown":
        teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
