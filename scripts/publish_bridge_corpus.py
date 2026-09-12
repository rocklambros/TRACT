"""Publish the Phase 2C bridge corpus. Run BEFORE Gate 2, not after.

The ordering is the whole point. `docs/phase2c-gate2-plan.md` says the corpus
ships regardless of the verdict, and that was the only load-bearing commitment
in the entire document set with no implementation, no destination, and no
binding home -- while `docs/phase2c-preregistration.md` conditioned Stage 2, and
therefore publication, on both gates passing. A FAIL would have quietly ended
with two volunteers' 600 control-judgements each sitting in a gitignored
directory on one laptop.

So it is committed first. A commitment that runs after the result it is supposed
to be independent of is not independent of it.

WHAT IS PUBLISHED, AND WHY IT CARRIES NAMES

`docs/phase2c-annotator-handbook.md` promised, in the part handed to
contributors verbatim: "Your answers become a public dataset of Tier-2 links,
with your name or chosen pseudonym recorded on each one... Accepted links are
proposed upstream to OpenCRE, with credit... Your rationales are published
alongside the links."

So the pseudonym and the rationale STAY. Stripping them would be a different
kind of failure -- it would break the credit that was promised, and this round's
contributors chose anonymity, not anonymity plus erasure. What they consented to
is exactly this: a pseudonym they chose, on work they can point to.

`data/training/hub_links_bridge.*.jsonl` stays gitignored because an IMPORT
command should not publish a volunteer's words as a side effect. Publishing is a
deliberate act, and this is it.

Read-only over the corpus. Loads no model.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from scripts.build_gate2_corpus import _git_sha, _sha256
from tract.bridge.links import load_bridge_links
from tract.config import PROJECT_ROOT, TRAINING_DIR
from tract.io import atomic_write_json, repo_relative

logger = logging.getLogger(__name__)

DEFAULT_OUT: Final[Path] = (
    PROJECT_ROOT / "results" / "phase2c" / "phase2c_bridge_corpus.json"
)
GATE1_REPORTS: Final[tuple[str, ...]] = (
    "gate1_report.json", "gate1_report_r2.json",
)

PROVENANCE: Final[str] = (
    "Tier 2 under results/phase1b/CAMPAIGN3.md section 2: independently "
    "human-authored, with no model output shown to the annotator at any point. "
    "Two annotators independently judged 300 NIST SP 800-53 controls against a "
    "sheet of 78 AI-region CRE hubs carrying only hub id, name, hierarchy path "
    "and branch. Weaker than Tier 1, which asserts OpenCRE curated it "
    "independently of TRACT; far stronger than Tier 3, which is produced by or "
    "ratified in the presence of a model."
)

LIMITS: Final[tuple[str, ...]] = (
    "The candidate hub sheet was derived from BRIDGE_AI_FRAMEWORK_IDS, which "
    "includes the frameworks supplying Gate 2's evaluation gold. The support "
    "set therefore could not have missed the gold, and any downstream result "
    "licenses a claim about this hub region rather than about bridging in "
    "general. Checkpoint-2 item C6, disclosed and not fixed.",
    "The round-1 handbook stated the answer distribution before annotators had "
    "read anything. Round 2 removed that sentence and re-ran the same 300 "
    "controls with the same two people: 106 NONE->link, 0 reversals, 0 hub "
    "changes. Neither round's link rate is established as a property of the "
    "task, and docs/phase2c-results.md section 4 governs what may be reported.",
    "Inter-annotator agreement depends on the denominator. On the link/no-link "
    "decision Cohen's kappa is 0.597 (round 1) and 0.508 (round 2); on WHICH "
    "hub, among controls both annotators linked, it is 0.885 and 0.891. Quote "
    "the denominator with the figure.",
    "Contributors retain a right to withdraw. "
    "`scripts/build_gate2_corpus.py --exclude-annotator` re-cuts the corpus; a "
    "model already trained on it cannot be un-trained.",
)


def build_publication(corpus_path: Path, round_label: str) -> dict[str, Any]:
    """The corpus plus everything a reader needs to weigh it."""
    links = load_bridge_links(corpus_path)
    annotators = sorted({link.annotator_id for link in links})
    return {
        "name": "TRACT Phase 2C bridge corpus",
        "description": (
            "Human-curated traditional-control -> AI-hub links, closing the "
            "structural gap that the AI and traditional CRE hub regions are "
            "disjoint: 78 AI hubs and 380 traditional hubs with an "
            "intersection of exactly 0, so nothing in the curated gold "
            "positions a traditional control against an AI hub."
        ),
        "license": "CC0-1.0",
        "source_framework": "NIST SP 800-53 Rev. 5 (US Government work)",
        "round_label": round_label,
        "provenance_tier": "T2",
        "provenance": PROVENANCE,
        "contributors": annotators,
        "credit": (
            "Contributed by volunteer annotators, recorded under the "
            "pseudonyms they chose. Proposed upstream to OpenCRE with credit."
        ),
        "n_links": len(links),
        "n_distinct_controls": len({link.section_id for link in links}),
        "n_distinct_hubs": len({link.cre_id for link in links}),
        "corpus_sha256": _sha256(corpus_path),
        "git_sha": _git_sha(),
        "limits": list(LIMITS),
        "gate1": _gate1_verdicts(),
        "links": [asdict(link) for link in links],
    }


def _gate1_verdicts() -> dict[str, Any]:
    """Both rounds' Gate 1 reports, so the verdict travels with the corpus."""
    out: dict[str, Any] = {}
    for name in GATE1_REPORTS:
        path = PROJECT_ROOT / "results" / "phase2c" / name
        if not path.is_file():
            logger.warning("%s absent; publishing without it", name)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        # Key names read from the real report rather than guessed. The first
        # draft asked for "verdict" and "n_deorphaned", which gate1_report does
        # not write -- so every verdict published as null, and the artifact
        # would have shipped saying nothing about whether the gate passed.
        out[name] = {
            key: report[key]
            for key in ("passed", "orphan_reduction_passed", "orphans_before",
                        "orphans_after", "deorphaned", "n_links_total",
                        "n_links_counting", "n_annotators", "bridge_path")
            if key in report
        }
        out[name]["conditions"] = {
            condition: {
                k: v for k, v in body.items()
                if k in ("passed", "value", "threshold", "submitted",
                         "cohen_kappa", "agreement_both_linked")
                and v is not None
            }
            for condition, body in (report.get("conditions") or {}).items()
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=TRAINING_DIR / "hub_links_bridge.r2.jsonl",
    )
    parser.add_argument("--round-label", default="r2")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = build_publication(args.corpus, args.round_label)
    atomic_write_json(payload, args.out)

    logger.info("=" * 68)
    logger.info("Phase 2C bridge corpus -- PUBLISHED")
    logger.info("  links        : %d over %d controls and %d hubs",
                payload["n_links"], payload["n_distinct_controls"],
                payload["n_distinct_hubs"])
    logger.info("  contributors : %s", ", ".join(payload["contributors"]))
    logger.info("  tier         : T2")
    logger.info("  limits       : %d recorded", len(payload["limits"]))
    logger.info("  -> %s", repo_relative(args.out))
    logger.info("")
    logger.info(
        "  COMMIT THIS BEFORE GATE 2 RUNS. The corpus ships regardless of the "
        "verdict, and a commitment that executes after the result it is "
        "supposed to be independent of is not independent of it."
    )
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
