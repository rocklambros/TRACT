"""Phase 2C Gate 1, computed: the orphan reduction AND the four conditions.

`docs/phase2c-preregistration.md` §2 binds five things. Until checkpoint 2 one
of them was computed and four were prose:

    orphan rate 78/78 -> <= 55/78     computed by scripts/analysis/orphan_rate.py
    Q1  >= 40 distinct controls        prose
    Q2  <= 6 AI hubs per control       prose
    Q3  confidence >= 2 to count       prose (the constant had no reader)
    Q4  >= 15% double-annotated        prose

The cost of that gap is one command. A sheet mapping ONE control onto all 78
hub ids -- copied from the first column of the packet the volunteer is handed,
confidence 1, rationale "." -- imports cleanly and reports 78/78 -> 0/78. It
violates Q1, Q2 and Q3 at once. Gate 1 reads PASS.

So this module computes all five and returns a single verdict that is their
conjunction. It refuses to print a verdict when a condition cannot be computed,
because a missing condition must never read as a satisfied one.

Q3 is applied BEFORE the orphan count, not after. That is the whole point of a
counting floor: a link the document calls "data, not evidence" must not
de-orphan a hub. Applying it afterwards, as a report line beside an orphan rate
computed over everything, is the same defect wearing a label.

Read-only. Loads no model, writes nothing unless asked for JSON output.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

from scripts.analysis.orphan_rate import (
    bridge_link_pairs,
    load_framework_hub_links,
    strict_firewall_orphans,
)
from tract.bridge.links import BridgeLink, load_bridge_links
from tract.config import (
    PHASE2C_GATE1_MAX_ORPHANS,
    PHASE2C_GATE1_MIN_DEORPHANED,
    PHASE2C_Q1_MIN_DISTINCT_CONTROLS,
    PHASE2C_Q2_MAX_HUBS_PER_CONTROL,
    PHASE2C_Q3_CONFIDENCE_FLOOR,
    PHASE2C_Q4_MIN_DOUBLE_ANNOTATED,
)
from tract.io import atomic_write_json

logger = logging.getLogger(__name__)


class Condition(TypedDict):
    """One pre-registered quality condition and its verdict.

    `value` is measured over the COUNTING subset (Q3 applied); `submitted` is
    the same statistic over every imported link. Both are reported because
    they answer different questions. A sheet whose links all fall below the
    confidence floor has an empty counting subset, so Q2 would measure zero
    hubs per control and pass vacuously -- while the sheet that produced it put
    78 hubs on a single control. The operator diagnosing a FAIL needs to see
    that, and a report showing only `value` hides it.
    """

    value: float
    submitted: float
    threshold: float
    passed: bool
    # Q4 only: the human-human agreement rate, or None when fewer than two
    # annotators worked the same control. None is not zero.
    agreement: float | None


def _q4(counting: list[BridgeLink], submitted: list[BridgeLink]) -> Condition:
    """Double-annotation rate, and the agreement among the overlap.

    Agreement is over the SET of hubs two annotators gave the same control --
    Jaccard, so partial overlap is partial credit. It is `None` when no control
    was worked by two people, because one person cannot agree with themselves
    and a rate of 1.0 there would be a fabricated number in the one place the
    pre-registration calls this project's first human-human measurement.
    """
    by_control: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for link in counting:
        by_control[link.section_id][link.annotator_id].add(link.cre_id)

    doubled = {
        control: annotators
        for control, annotators in by_control.items()
        if len(annotators) >= 2
    }
    rate = len(doubled) / len(by_control) if by_control else 0.0

    agreement: float | None = None
    if doubled:
        scores: list[float] = []
        for annotators in doubled.values():
            sets = sorted(annotators.values(), key=len, reverse=True)[:2]
            union = sets[0] | sets[1]
            scores.append(len(sets[0] & sets[1]) / len(union) if union else 0.0)
        agreement = sum(scores) / len(scores)

    submitted_controls: dict[str, set[str]] = defaultdict(set)
    for link in submitted:
        submitted_controls[link.section_id].add(link.annotator_id)
    submitted_rate = (
        sum(1 for a in submitted_controls.values() if len(a) >= 2)
        / len(submitted_controls)
        if submitted_controls else 0.0
    )

    return Condition(
        value=rate,
        submitted=submitted_rate,
        threshold=PHASE2C_Q4_MIN_DOUBLE_ANNOTATED,
        passed=rate >= PHASE2C_Q4_MIN_DOUBLE_ANNOTATED,
        agreement=agreement,
    )


def gate1_report(bridge_path: Path) -> dict[str, Any]:
    """Compute Gate 1 in full. Returns the report; raises rather than guessing."""
    links = load_bridge_links(bridge_path)
    if not links:
        raise ValueError(
            f"{bridge_path} holds no links. A Gate 1 verdict over an empty "
            "corpus would report the unchanged orphan rate as a measurement."
        )

    # Q3 first. Everything downstream is computed over the counting subset,
    # because a link below the floor is data and not evidence.
    counting = [
        link for link in links if link.confidence >= PHASE2C_Q3_CONFIDENCE_FLOOR
    ]

    def _shape(subset: list[BridgeLink]) -> tuple[int, int]:
        by_control: dict[str, set[str]] = defaultdict(set)
        for link in subset:
            by_control[link.section_id].add(link.cre_id)
        return len(by_control), max((len(v) for v in by_control.values()), default=0)

    distinct_controls, max_hubs = _shape(counting)
    submitted_controls, submitted_max_hubs = _shape(links)

    base = load_framework_hub_links()
    orphans_before, total = strict_firewall_orphans(base)
    orphans_after, total_after = strict_firewall_orphans(
        base + bridge_link_pairs(counting)
    )
    deorphaned = orphans_before - orphans_after

    conditions: dict[str, Condition] = {
        "Q1_distinct_controls": Condition(
            value=distinct_controls,
            submitted=submitted_controls,
            threshold=PHASE2C_Q1_MIN_DISTINCT_CONTROLS,
            passed=distinct_controls >= PHASE2C_Q1_MIN_DISTINCT_CONTROLS,
            agreement=None,
        ),
        "Q2_max_hubs_per_control": Condition(
            value=max_hubs,
            submitted=submitted_max_hubs,
            # Judged on what was SUBMITTED, not on what counts. Q2 asks whether
            # the annotator made a judgement about the control or about the
            # region, and dropping their low-confidence links does not unask
            # it. Measuring the counting subset alone lets a sheet that put 78
            # hubs on one control pass by having every link filtered out.
            threshold=PHASE2C_Q2_MAX_HUBS_PER_CONTROL,
            passed=submitted_max_hubs <= PHASE2C_Q2_MAX_HUBS_PER_CONTROL,
            agreement=None,
        ),
        "Q3_confidence_floor": Condition(
            value=len(counting),
            submitted=len(links),
            threshold=PHASE2C_Q3_CONFIDENCE_FLOOR,
            # Q3 is a filter, not a threshold on a count: it passes as long as
            # something survived it. A round where every link is below the
            # floor has no evidence in it at all.
            passed=bool(counting),
            agreement=None,
        ),
        "Q4_double_annotated": _q4(counting, links),
    }

    orphan_reduction_passed = (
        orphans_after <= PHASE2C_GATE1_MAX_ORPHANS
        and deorphaned >= PHASE2C_GATE1_MIN_DEORPHANED
    )

    return {
        "bridge_path": str(bridge_path),
        "n_links_total": len(links),
        "n_links_counting": len(counting),
        "n_annotators": len({link.annotator_id for link in counting}),
        "orphans_before": orphans_before,
        "orphans_after": orphans_after,
        "ai_hubs_total": total,
        "deorphaned": deorphaned,
        "orphan_reduction_passed": orphan_reduction_passed,
        "conditions": conditions,
        # The conjunction, never the orphan rate alone. Gate 1 counts hubs and
        # is gameable by volume; the conditions are what make the count mean
        # something, so a verdict that ignores them is the defect restated.
        "passed": orphan_reduction_passed
        and all(c["passed"] for c in conditions.values()),
        "_total_after_sanity": total_after,
    }


def _log(report: dict[str, Any]) -> None:
    logger.info("=" * 66)
    logger.info("PHASE 2C GATE 1  (%s)", report["bridge_path"])
    logger.info(
        "  links: %d imported, %d counting toward the gate (confidence >= %d)",
        report["n_links_total"], report["n_links_counting"],
        PHASE2C_Q3_CONFIDENCE_FLOOR,
    )
    logger.info(
        "  orphans: %d -> %d of %d  (de-orphaned %d, need >= %d)  %s",
        report["orphans_before"], report["orphans_after"], report["ai_hubs_total"],
        report["deorphaned"], PHASE2C_GATE1_MIN_DEORPHANED,
        "PASS" if report["orphan_reduction_passed"] else "FAIL",
    )
    for name, condition in report["conditions"].items():
        logger.info(
            "  %-24s %8.4g (submitted %.4g, threshold %.4g)  %s",
            name, condition["value"], condition["submitted"],
            condition["threshold"],
            "PASS" if condition["passed"] else "FAIL",
        )
    agreement = report["conditions"]["Q4_double_annotated"]["agreement"]
    logger.info(
        "  human-human agreement    : %s",
        "not measured (fewer than two annotators on any control)"
        if agreement is None else f"{agreement:.4f}",
    )
    logger.info("  GATE 1: %s", "PASS" if report["passed"] else "FAIL")
    logger.info("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bridge", type=Path, help="Tier-2 bridge link JSONL.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Write the report as JSON here."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = gate1_report(args.bridge)
    _log(report)
    if args.out is not None:
        atomic_write_json(report, args.out)
        logger.info("Wrote %s", args.out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
