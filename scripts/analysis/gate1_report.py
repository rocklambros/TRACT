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
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Final, TypedDict

from scripts.analysis.orphan_rate import (
    bridge_link_pairs,
    load_framework_hub_links,
    strict_firewall_orphans,
)
from tract.bridge.links import BridgeLink, load_bridge_links
from tract.config import (
    BRIDGE_CORPUS_DIR,
    PHASE2C_GATE1_MAX_ORPHANS,
    PHASE2C_GATE1_MIN_DEORPHANED,
    PHASE2C_Q1_MIN_DISTINCT_CONTROLS,
    PHASE2C_Q2_MAX_HUBS_PER_CONTROL,
    PHASE2C_Q3_CONFIDENCE_FLOOR,
    PHASE2C_Q4_MIN_DOUBLE_ANNOTATED,
)
from tract.io import atomic_write_json

logger = logging.getLogger(__name__)

# The curated gold link files that live in data/training/. Named EXACTLY rather
# than matched by a "hub_links" prefix: a prefix also catches a legitimately
# named bridge corpus, and a guard that refuses valid input gets removed by
# whoever hits it.
GOLD_LINK_FILENAMES: Final[frozenset[str]] = frozenset({
    "hub_links.jsonl",
    "hub_links_curated.jsonl",
    "hub_links_training.jsonl",
})


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
    # Q4 only. All None when fewer than two annotators worked the same control:
    # one person cannot agree with themselves, and a fabricated 1.0 in the
    # project's first human-human measurement would be worse than an absence.
    #
    # `agreement` is kept as the both-linked figure for readers of earlier
    # artifacts. `cohen_kappa` is the headline, because a raw rate on a task
    # where most answers are NONE reads high at chance-level skill.
    agreement: float | None
    cohen_kappa: float | None
    agreement_both_linked: float | None
    agreement_either_linked: float | None
    n_both_linked: int
    n_same_hub: int
    n_one_linked_only: int


def _q4(
    counting: list[BridgeLink],
    submitted: list[BridgeLink],
    reviewed_by: dict[str, int] | None = None,
) -> Condition:
    """Double-annotation rate, and agreement under every defensible denominator.

    A single agreement number on this task is a CHOICE OF DENOMINATOR, and the
    choice has to be visible rather than made silently. Measured on the first
    real round:

        both linked, same hub          23   agree
        both linked, different hub      3   disagree
        one linked, other said NONE    28   disagree
        both said NONE                242   trivially agree

    Jaccard over "controls both linked" gives 23/26 = 0.885 and drops the 28 --
    the largest disagreement category. Counting NONE-NONE gives 265/300 = 0.883,
    which is the negative-class inflation the pre-registration named in advance:
    "the negative class dominates and raw agreement will read ~95% at
    chance-level skill". The same data chance-corrected gives kappa = 0.539.

    So `cohen_kappa` is the headline -- it is what the pre-registration asked
    for -- and the raw figures sit beside it labelled by their denominator.

    Everything is None, never 1.0 or 0.0, when fewer than two annotators worked
    the same control. One person cannot agree with themselves, and this is the
    project's first human-human measurement.
    """
    by_control: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for link in counting:
        by_control[link.section_id][link.annotator_id].add(link.cre_id)

    doubled = {c: a for c, a in by_control.items() if len(a) >= 2}
    rate = len(doubled) / len(by_control) if by_control else 0.0

    annotators = sorted({link.annotator_id for link in counting})
    kappa: float | None = None
    both_linked_agreement: float | None = None
    either_linked_agreement: float | None = None
    n_both = n_same = n_one_only = 0

    if len(annotators) >= 2:
        # The two most productive annotators. A pairwise statistic needs a pair;
        # with three or more this is the pair with the most overlap to report on.
        a_id, b_id = annotators[0], annotators[1]
        linked: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for link in counting:
            linked[link.annotator_id][link.section_id].add(link.cre_id)

        a_links, b_links = linked[a_id], linked[b_id]
        both = set(a_links) & set(b_links)
        n_both = len(both)
        n_same = sum(1 for c in both if a_links[c] == b_links[c])
        n_one_only = len(set(a_links) ^ set(b_links))

        if n_both:
            both_linked_agreement = n_same / n_both
        if n_both + n_one_only:
            either_linked_agreement = n_same / (n_both + n_one_only)

        # Cohen's kappa on the LINK / NO-LINK decision, over every control the
        # annotators REVIEWED -- which is not the same as the controls that
        # produced links.
        #
        # The corpus holds links only; a NONE judgement is not a link, so it is
        # recorded in the .reviewed.json sidecar instead. Deriving the
        # population from the links alone gave kappa = -0.1033 on the first real
        # round, because the ~242 controls both annotators judged NONE were
        # invisible and the population collapsed to the 54 where at least one
        # linked. With the true denominator the same data gives 0.54. A kappa
        # computed on the wrong population is as wrong as the flattering raw
        # rate, just in the other direction.
        n = min(reviewed_by.values()) if reviewed_by else 0
        if n:
            a_yes, b_yes = len(a_links), len(b_links)
            both_yes = len(both)
            observed = (both_yes + (n - a_yes - b_yes + both_yes)) / n
            expected = (
                (a_yes / n) * (b_yes / n)
                + ((n - a_yes) / n) * ((n - b_yes) / n)
            )
            if expected < 1.0:
                kappa = (observed - expected) / (1.0 - expected)

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
        agreement=both_linked_agreement,
        cohen_kappa=kappa,
        agreement_both_linked=both_linked_agreement,
        agreement_either_linked=either_linked_agreement,
        n_both_linked=n_both,
        n_same_hub=n_same,
        n_one_linked_only=n_one_only,
    )


def _resolve(bridge_path: Path) -> list[Path]:
    """One file, or every corpus in a directory.

    The importer refuses to overwrite and writes one file per annotator. This
    read one path with no merge, so Q4 -- which counts annotators WITHIN a
    corpus -- was structurally 0.0 under the exact workflow the importer
    prescribes.
    """
    if bridge_path.is_dir():
        found = sorted(bridge_path.glob("*.jsonl"))
        if not found:
            raise ValueError(
                f"{bridge_path} is a directory with no .jsonl corpus in it. "
                f"Per-annotator corpora belong in {BRIDGE_CORPUS_DIR}."
            )
        # Refuse the parent. data/training/ holds the curated GOLD link files,
        # so globbing it hands gold to a bridge loader -- which fails on a
        # missing field, leaving the operator reading a schema error instead of
        # being told they pointed at the wrong directory.
        gold = [p for p in found if p.name in GOLD_LINK_FILENAMES]
        if gold:
            raise ValueError(
                f"{bridge_path} contains the curated gold link files "
                f"({', '.join(p.name for p in gold)}), so it is not a bridge "
                f"corpus directory. Point Gate 1 at {BRIDGE_CORPUS_DIR}, which "
                "holds one <annotator_id>.jsonl per annotator."
            )
        return found
    return [bridge_path]


def _reviewed_counts(sources: list[Path]) -> dict[str, int]:
    """How many controls each annotator actually reviewed.

    From the .reviewed.json sidecars the importer writes. This is the
    denominator the corpus itself cannot supply: a NONE judgement is not a link,
    so it appears in no .jsonl, and without these counts an agreement statistic
    sees only the controls where somebody linked.
    """
    counts: dict[str, int] = {}
    for source in sources:
        sidecar = source.with_suffix(".reviewed.json")
        if not sidecar.is_file():
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        counts[payload["annotator_id"]] = int(payload["n_reviewed"])
    return counts


def gate1_report(bridge_path: Path) -> dict[str, Any]:
    """Compute Gate 1 in full. Returns the report; raises rather than guessing.

    `bridge_path` may be a single corpus or a DIRECTORY of per-annotator
    corpora, which is the layout the importer prescribes.
    """
    sources = _resolve(bridge_path)
    links: list[BridgeLink] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        for link in load_bridge_links(source):
            # Re-importing the same sheet under two names would otherwise
            # double the apparent overlap and inflate Q4.
            key = (link.annotator_id, link.section_id, link.cre_id)
            if key in seen:
                raise ValueError(
                    f"{source}: {link.annotator_id!r} already recorded "
                    f"{link.section_id!r} -> {link.cre_id!r} in another "
                    "corpus. Two files hold the same annotator's sheet."
                )
            seen.add(key)
            links.append(link)
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
        """(distinct controls, max hubs one ANNOTATOR gave one control).

        Q2 is keyed on (control, annotator), not on control alone. Keyed on
        control it measures the UNION across annotators -- so two people giving
        four hubs each to the same control, neither exceeding the limit of six,
        measure 8 and FAIL. Since Q4 mandates a double-annotated overlap, the
        two conditions were computed on incompatible groupings and satisfying
        one broke the other.
        """
        controls: set[str] = set()
        by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        for link in subset:
            controls.add(link.section_id)
            by_pair[(link.section_id, link.annotator_id)].add(link.cre_id)
        return len(controls), max((len(v) for v in by_pair.values()), default=0)

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
            cohen_kappa=None,
            agreement_both_linked=None,
            agreement_either_linked=None,
            n_both_linked=0,
            n_same_hub=0,
            n_one_linked_only=0,
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
            cohen_kappa=None,
            agreement_both_linked=None,
            agreement_either_linked=None,
            n_both_linked=0,
            n_same_hub=0,
            n_one_linked_only=0,
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
            cohen_kappa=None,
            agreement_both_linked=None,
            agreement_either_linked=None,
            n_both_linked=0,
            n_same_hub=0,
            n_one_linked_only=0,
        ),
        "Q4_double_annotated": _q4(counting, links, _reviewed_counts(sources)),
    }

    orphan_reduction_passed = (
        orphans_after <= PHASE2C_GATE1_MAX_ORPHANS
        and deorphaned >= PHASE2C_GATE1_MIN_DEORPHANED
    )

    return {
        "bridge_path": str(bridge_path),
        "sources": [str(p) for p in sources],
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
    q4 = report["conditions"]["Q4_double_annotated"]
    if q4["cohen_kappa"] is None:
        logger.info(
            "  human-human agreement    : not measured (fewer than two "
            "annotators on any control)"
        )
    else:
        # The kappa first, because a raw rate on a task where most answers are
        # NONE reads high at chance-level skill. The denominators follow, so
        # nobody has to guess which one a quoted figure came from.
        logger.info("  human-human agreement (Cohen's kappa, link decision): "
                    "%.4f", q4["cohen_kappa"])
        logger.info("    same hub, of %d both linked          : %.4f",
                    q4["n_both_linked"], q4["agreement_both_linked"])
        logger.info("    same hub, of %d either linked        : %.4f",
                    q4["n_both_linked"] + q4["n_one_linked_only"],
                    q4["agreement_either_linked"])
        logger.info("    %d controls one linked and the other did not",
                    q4["n_one_linked_only"])
    logger.info("  GATE 1: %s", "PASS" if report["passed"] else "FAIL")
    logger.info("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bridge", type=Path, nargs="?", default=BRIDGE_CORPUS_DIR,
        help=(
            "A Tier-2 bridge corpus, or a DIRECTORY of per-annotator corpora. "
            "Defaults to the bridge corpus directory."
        ),
    )
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
