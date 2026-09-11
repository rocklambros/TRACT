"""Phase 2C Gate 2: the arm-vs-arm comparison, which had no implementation.

`tract.training.orchestrate.gate_decision` is the only function in this
repository that emits a gate verdict, and it is hardwired to
`baseline = r["zero_shot"]["hit1_indicators"]` -- the comparator
`docs/phase2c-preregistration.md` retires in writing, at a threshold of 0.10
that Gate 2 does not use, and one that 73% of BRIDGE-FREE Campaign 2 arms
already satisfy. Nothing paired two trained arms.

Meanwhile every arm writes an `aggregate_metrics.json` carrying
`gate.preregistered_pass`, `gate.ci_low_pass` and `gate.familywise_pass`. Those
fields sit exactly where a reader looks for a verdict. Quoting one of them as
Gate 2 is the likeliest way this round reports a number nobody computed, and it
is why this is a separate script with its own threshold constant rather than a
new mode of `gate_decision`.

WHAT IT COMPUTES

The primary estimand is a difference-in-differences over a pre-registered
exposure partition:

    DiD = mean(delta | exposed) - mean(delta | unexposed)

An item is EXPOSED when at least one hub in its `valid_hub_ids` receives a
positive from the bridge corpus. Under the strict all-AI firewall only 27 of 74
eval items are exposed; the other 47 cannot respond to the treatment at all and
can only contribute noise to a pooled contrast.

The DiD is not a refinement, it is the calibration. `paired_bootstrap_delta`
resamples ITEMS and nothing else, so its interval omits training-draw variance
-- and two fine-tunes differing by 2% of their corpus differ by ~19% per-item
discordance in this repository's own committed artifacts. Global drift moves
both strata together and cancels in the difference. Simulated on this design, a
pooled contrast false-passes at 0.086 under 10pp fold drift where the DiD
false-passes at 0.016, and the DiD is also the more powerful of the two.

It is also the mechanism test the deliverable needs: a positive pooled delta
with a FLAT DiD says the gain was churn or a global training-mix effect, not the
bridge links.

Read-only over two results directories. Loads no model.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from tract.config import (
    PHASE1B_BOOTSTRAP_CI_LEVEL,
    PHASE1B_BOOTSTRAP_N_RESAMPLES,
    PHASE1B_BOOTSTRAP_SEED,
    PHASE2C_GATE2_N_CONFIGURATIONS,
    PHASE2C_GATE2_THRESHOLD,
)
from tract.io import atomic_write_json, repo_relative
from tract.training.evaluate import _fold_rng

logger = logging.getLogger(__name__)

FOLD_RESULT_NAME: Final[str] = "fold_result.json"
PREDICTIONS_NAME: Final[str] = "predictions.json"


@dataclass(frozen=True)
class FoldRecord:
    """One scored population within one arm.

    `item_frameworks` is per ITEM, not per fold. Gate 2 trains ONE model per arm
    and scores all 74 items with it, because the difference-in-differences needs
    the exposed and unexposed strata to come from the SAME model -- scoring each
    framework with its own separately-trained model would put training-draw
    variance BETWEEN the strata, which is precisely the variance the DiD exists
    to cancel. So one fold here carries three frameworks, and the stratification
    has to read each item's own.
    """

    framework: str
    indicators: NDArray[np.floating[Any]]
    item_keys: tuple[str, ...]
    item_frameworks: tuple[str, ...]
    excluded_frameworks: tuple[str, ...]


@dataclass(frozen=True)
class ArmRecord:
    """One trained arm: its folds, and what made it that arm."""

    path: str
    bridge_links_path: str | None
    bridge_links_sha256: str | None
    folds: dict[str, FoldRecord]

    @property
    def frameworks(self) -> tuple[str, ...]:
        return tuple(sorted(self.folds))


@dataclass(frozen=True)
class DiDResult:
    """The Gate 2 statistic and everything needed to read it honestly."""

    did: float
    ci_low: float
    ci_high: float
    p_did_le_threshold: float
    delta_exposed: float
    delta_unexposed: float
    pooled_delta: float
    n_exposed: int
    n_unexposed: int
    per_framework: dict[str, float] = field(default_factory=dict)
    n_resamples: int = PHASE1B_BOOTSTRAP_N_RESAMPLES
    threshold: float = PHASE2C_GATE2_THRESHOLD


def load_arm(arm_dir: Path) -> ArmRecord:
    """Read one arm's fold records and per-item keys.

    Raises on anything partial. An arm loaded with one fold missing produces an
    interval over a different population than the arm it is compared against,
    and nothing downstream would notice.
    """
    folds: dict[str, FoldRecord] = {}
    bridge_path: str | None = None
    bridge_sha: str | None = None
    seen_config = False

    for result_path in sorted(arm_dir.glob(f"fold_*/{FOLD_RESULT_NAME}")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        framework = payload["held_out_framework"]
        predictions_path = result_path.parent / PREDICTIONS_NAME
        if not predictions_path.is_file():
            raise ValueError(
                f"{predictions_path} is missing, so this arm's item order "
                "cannot be verified against the other arm's."
            )
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        keys = tuple(_item_key(row, i) for i, row in enumerate(predictions))
        item_frameworks = tuple(
            str(row.get("framework") or framework) for row in predictions
        )

        indicators = np.asarray(payload["hit1_indicators"], dtype=float)
        if len(indicators) != len(keys):
            raise ValueError(
                f"{result_path}: {len(indicators)} indicators against "
                f"{len(keys)} predictions. The two describe the same items and "
                "disagreeing means one of them is not this fold's."
            )
        folds[framework] = FoldRecord(
            framework=framework,
            indicators=indicators,
            item_keys=keys,
            item_frameworks=item_frameworks,
            excluded_frameworks=tuple(
                sorted(payload.get("excluded_frameworks") or [])
            ),
        )
        config = payload.get("config") or {}
        if not seen_config:
            bridge_path = config.get("bridge_links_path")
            bridge_sha = (payload.get("inputs") or {}).get(
                "bridge_links_sha256"
            )
            seen_config = True
        elif config.get("bridge_links_path") != bridge_path:
            raise ValueError(
                f"{arm_dir} holds folds trained on different bridge corpora "
                f"({bridge_path!r} and {config.get('bridge_links_path')!r}). "
                "That is two arms in one directory, not one arm."
            )

    if not folds:
        raise ValueError(f"{arm_dir} holds no fold results")
    return ArmRecord(
        path=repo_relative(arm_dir),
        bridge_links_path=bridge_path,
        bridge_links_sha256=bridge_sha,
        folds=folds,
    )


def _item_key(row: dict[str, Any], index: int) -> str:
    """Stable per-item identity across two separately-executed arms.

    `control_text_sha256` rather than `control_text`: ETSI is a restricted
    framework whose verbatim text is withheld from `predictions.json`, and a
    digest compares exactly as well without being readable.
    """
    key = row.get("control_text_sha256")
    if not key:
        raise ValueError(
            f"predictions row {index} has no control_text_sha256, so the two "
            "arms cannot be proven to have scored the same items. Re-run the "
            "fold: this field has been written since the Gate 2 work."
        )
    return str(key)


def verify_alignment(a: ArmRecord, b: ArmRecord) -> None:
    """Refuse any comparison that is not actually paired.

    `paired_bootstrap_delta` validates fold COUNT and fold SIZE and nothing
    else, so two 74-item arms scored in different orders produce a paired
    interval they have not earned. It is a silent failure: the number looks
    entirely ordinary.
    """
    if a.frameworks != b.frameworks:
        raise ValueError(
            f"The arms scored different folds: {a.frameworks} vs "
            f"{b.frameworks}. There is no paired comparison between them."
        )
    for framework in a.frameworks:
        fa, fb = a.folds[framework], b.folds[framework]
        if fa.item_keys != fb.item_keys:
            raise ValueError(
                f"Fold {framework}: the two arms scored a different item order "
                f"({len(fa.item_keys)} vs {len(fb.item_keys)} items, "
                f"{sum(1 for x, y in zip(fa.item_keys, fb.item_keys) if x != y)}"
                " positions differ). A paired interval over unpaired items is "
                "not a measurement."
            )
        if fa.excluded_frameworks != fb.excluded_frameworks:
            raise ValueError(
                f"Fold {framework}: the arms were firewalled differently "
                f"({list(fa.excluded_frameworks)} vs "
                f"{list(fb.excluded_frameworks)}). Arms that held out different "
                "frameworks are not each other's counterfactual."
            )
    if a.bridge_links_sha256 == b.bridge_links_sha256:
        raise ValueError(
            "Both arms read the same bridge corpus "
            f"({a.bridge_links_sha256!r}). The treatment and its comparator "
            "must differ in exactly the bridge links and nothing else; here "
            "they differ in nothing."
        )


def compute_did(
    a: ArmRecord,
    b: ArmRecord,
    exposed_keys: set[str],
    *,
    n_resamples: int = PHASE1B_BOOTSTRAP_N_RESAMPLES,
    seed: int = PHASE1B_BOOTSTRAP_SEED,
    ci_level: float = PHASE1B_BOOTSTRAP_CI_LEVEL,
    threshold: float = PHASE2C_GATE2_THRESHOLD,
) -> DiDResult:
    """Difference-in-differences on hit@1, b minus a, exposed minus unexposed.

    Resampling is stratified on (framework, exposure) cells, so both the fold
    structure the pre-registration names and the exposure partition the primary
    estimand rests on are preserved in every bootstrap replicate. Each cell
    draws from a stream keyed to its own contents, so the interval does not
    depend on the order the cells were assembled in.
    """
    verify_alignment(a, b)

    # Stratified on each ITEM's framework and its exposure, not on the fold it
    # was written under. Gate 2 puts all three frameworks in one fold, because
    # both strata must come from one trained model.
    grouped: dict[tuple[str, bool], list[float]] = {}
    by_framework: dict[str, list[float]] = {}
    for fold_name in a.frameworks:
        fa, fb = a.folds[fold_name], b.folds[fold_name]
        deltas = fb.indicators - fa.indicators
        for delta, key, framework in zip(
            deltas, fa.item_keys, fa.item_frameworks
        ):
            grouped.setdefault(
                (framework, key in exposed_keys), []
            ).append(float(delta))
            by_framework.setdefault(framework, []).append(float(delta))

    cells: dict[tuple[str, bool], NDArray[np.floating[Any]]] = {
        k: np.asarray(v, dtype=float) for k, v in sorted(grouped.items())
    }
    per_framework = {
        k: float(np.mean(v)) for k, v in sorted(by_framework.items())
    }

    exposed_cells = [v for (_, e), v in cells.items() if e]
    unexposed_cells = [v for (_, e), v in cells.items() if not e]
    n_exposed = sum(len(v) for v in exposed_cells)
    n_unexposed = sum(len(v) for v in unexposed_cells)
    if n_exposed == 0:
        raise ValueError(
            "No eval item is exposed to the bridge corpus, so there is no "
            "treatment stratum and the DiD is undefined. Check the corpus and "
            "the strata file rather than reporting a pooled delta instead."
        )
    if n_unexposed == 0:
        raise ValueError(
            "Every eval item is exposed, so there is no unexposed control "
            "stratum and the DiD is undefined. Only a pooled delta is "
            "available, and it carries the training-draw variance the DiD "
            "exists to cancel."
        )

    def _mean(groups: list[NDArray[np.floating[Any]]]) -> float:
        return float(np.concatenate(groups).mean())

    delta_exposed = _mean(exposed_cells)
    delta_unexposed = _mean(unexposed_cells)
    pooled = _mean(exposed_cells + unexposed_cells)

    # One resampled draw per cell, keyed on the cell's own contents.
    boot_exposed = np.zeros(n_resamples)
    boot_unexposed = np.zeros(n_resamples)
    for (_, is_exposed), values in cells.items():
        idx = _fold_rng(values, seed).integers(
            0, len(values), size=(n_resamples, len(values))
        )
        totals = values[idx].sum(axis=1)
        if is_exposed:
            boot_exposed += totals
        else:
            boot_unexposed += totals
    boot_did = boot_exposed / n_exposed - boot_unexposed / n_unexposed

    alpha = (1 - ci_level) / 2
    return DiDResult(
        did=delta_exposed - delta_unexposed,
        ci_low=float(np.percentile(boot_did, 100 * alpha)),
        ci_high=float(np.percentile(boot_did, 100 * (1 - alpha))),
        p_did_le_threshold=float(np.mean(boot_did <= threshold)),
        delta_exposed=delta_exposed,
        delta_unexposed=delta_unexposed,
        pooled_delta=pooled,
        n_exposed=n_exposed,
        n_unexposed=n_unexposed,
        per_framework=per_framework,
        n_resamples=n_resamples,
        threshold=threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator", type=Path, required=True,
                        help="A0: the bridge-free arm's results directory.")
    parser.add_argument("--treatment", type=Path, required=True,
                        help="A1: the bridge-trained arm's results directory.")
    parser.add_argument("--strata", type=Path, required=True,
                        help="results/phase2c/gate2_strata.json, committed "
                             "BEFORE the run.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label", default="A1 vs A0",
                        help="Which contrast this is, for the report.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    strata = json.loads(args.strata.read_text(encoding="utf-8"))
    exposed = set(strata["exposed_item_keys"])
    comparator = load_arm(args.comparator)
    treatment = load_arm(args.treatment)
    result = compute_did(comparator, treatment, exposed)

    logger.info("=" * 70)
    logger.info("PHASE 2C GATE 2 -- %s", args.label)
    logger.info("  comparator : %s  (bridge=%s)",
                comparator.path, comparator.bridge_links_path)
    logger.info("  treatment  : %s  (bridge=%s)",
                treatment.path, treatment.bridge_links_path)
    logger.info("")
    logger.info("  delta on   exposed items (n=%3d): %+.4f",
                result.n_exposed, result.delta_exposed)
    logger.info("  delta on unexposed items (n=%3d): %+.4f",
                result.n_unexposed, result.delta_unexposed)
    logger.info("  PRIMARY  DiD = %+.4f  [%+.4f, %+.4f]",
                result.did, result.ci_low, result.ci_high)
    logger.info("  pooled delta (descriptive, NOT the verdict): %+.4f",
                result.pooled_delta)
    logger.info("")
    for framework, delta in sorted(result.per_framework.items()):
        logger.info("    %-8s %+.4f", framework, delta)
    logger.info("")
    verdict = "PASS" if result.ci_low > result.threshold else "FAIL"
    logger.info("  ci_low > %.2f  ->  %s", result.threshold, verdict)
    logger.info("")
    logger.info(
        "  A null here is weak evidence of absence at this exposure. The "
        "pooled delta is descriptive and is not the criterion; "
        "docs/phase2c-gate2-plan.md section 5 holds the outcome table."
    )
    logger.info("=" * 70)

    if args.out is not None:
        atomic_write_json(
            {
                "label": args.label,
                "comparator": comparator.path,
                "treatment": treatment.path,
                "comparator_bridge_sha256": comparator.bridge_links_sha256,
                "treatment_bridge_sha256": treatment.bridge_links_sha256,
                "n_configurations": PHASE2C_GATE2_N_CONFIGURATIONS,
                "verdict": verdict,
                **{k: v for k, v in vars(result).items()},
            },
            args.out,
        )
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
