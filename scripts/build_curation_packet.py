"""Build the blind annotation packet for a Tier-2 curation round.

WHAT THIS IS FOR. `results/phase1b/CAMPAIGN3.md` needs more evaluation items and
can only use them if they are Tier 2: independently human-authored, BLIND to
model output. A single model-derived artifact anywhere in the annotator's hands
makes the whole round Tier 3 and worthless for the gate, so the packet is
generated rather than assembled by hand, and every column traces to a source
this module names.

THE TRAP THIS EXISTS TO AVOID. `results/ceiling_study/hub_reference.md` is the obvious
navigation aid to hand an annotator: it already exists, it covers all 522 hubs,
and it reads well. 400 of its hub descriptions were written by an LLM
conditioned on the gold links. Handing it over would quietly convert the round
into Tier 3. This module builds a replacement whose hub descriptions are the
hierarchy path OpenCRE published plus the titles of real controls OpenCRE
already linked -- no generated prose at any point.

WHY THE ILLUSTRATION FRAMEWORKS ARE FILTERED. A hub is easier to understand
alongside examples of what maps to it, but the examples must not be the answer.
Two exclusions:

  - Every AI-security framework, because those are the labels the round exists
    to extend and their hubs are the ones under test.
  - Cloud Controls Matrix, because 203 of csa_aicm's 243 control ids are
    inherited from CCM and 91 carry byte-identical statements. A CCM example
    beside a CCM-derived control is the answer with a different framework's name
    on it.

That leaves 4,054 links illustrating 374 of the 522 hubs. The remaining 148 get
path and name only, which is the honest presentation: OpenCRE has not linked
anything to them either.

Read-only with respect to the repository; writes only into the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

from tract.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Frameworks whose existing links must never illustrate a hub. See the module
# docstring: the AI set is what the round extends, and CCM is csa_aicm's
# parent corpus.
EXCLUDED_ILLUSTRATION_FRAMEWORKS: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS", "NIST AI 100-2", "OWASP AI Exchange",
    "OWASP Top10 for LLM", "OWASP Top10 for ML",
    "ENISA", "ETSI", "BIML",
    "Cloud Controls Matrix",
})

# Enough to convey what a hub covers without turning the sheet into a corpus.
MAX_EXAMPLES_PER_HUB: Final[int] = 4

# Frameworks CAMPAIGN3.md section 6 permits curating.
CURATION_TARGETS: Final[tuple[str, ...]] = (
    "csa_aicm", "cosai", "aiuc_1", "nist_ai_rmf",
)

# csa_aicm full_text runs to a median of 17,115 characters, most of it a shared
# "shared:" responsibility appendix. An annotator reading 243 of those would
# spend the whole engagement on boilerplate, so the packet carries the
# description and the control's own statement is offered separately.
CONTROL_TEXT_PREVIEW_CHARS: Final[int] = 1200


def _load_hierarchy() -> dict[str, Any]:
    path = PROCESSED_DIR / "cre_hierarchy.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; the hub sheet is its output")
    hierarchy: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return hierarchy


def build_hub_sheet(output_dir: Path) -> Path:
    """One row per CRE hub: id, name, path, and safe example controls.

    The annotator's only reference. Everything in it comes from
    `cre_hierarchy.json` and from `hub_links_curated.jsonl` filtered by
    EXCLUDED_ILLUSTRATION_FRAMEWORKS.
    """
    from scripts.phase0.common import load_curated_links

    hierarchy = _load_hierarchy()
    hubs = hierarchy["hubs"]

    examples: dict[str, list[str]] = defaultdict(list)
    for link in load_curated_links():
        if link.standard_name in EXCLUDED_ILLUSTRATION_FRAMEWORKS:
            continue
        bucket = examples[link.cre_id]
        label = f"{link.standard_name}: {link.section_name}"
        if label not in bucket and len(bucket) < MAX_EXAMPLES_PER_HUB:
            bucket.append(label)

    target = output_dir / "hub_reference_sheet.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hub_id", "hub_name", "branch", "hierarchy_path",
            "example_controls_already_mapped_here",
        ])
        for hub_id in sorted(hubs):
            node = hubs[hub_id]
            path = node.get("hierarchy_path", "") or ""
            writer.writerow([
                hub_id,
                node.get("name", ""),
                path.split(" > ")[0] if path else "",
                path,
                " | ".join(examples.get(hub_id, [])),
            ])

    illustrated = sum(1 for h in hubs if examples.get(h))
    logger.info(
        "Hub sheet: %d hubs, %d illustrated by non-AI non-CCM links, %d with "
        "path and name only.", len(hubs), illustrated, len(hubs) - illustrated,
    )
    return target


def build_control_sheet(output_dir: Path, framework_id: str) -> Path:
    """The controls to annotate, one row each, with a blank answer column.

    Deliberately carries NO model prediction, NO confidence, NO shortlist and
    NO ranking. If a future version adds any of those the round stops being
    Tier 2.
    """
    # merged_corpus_path, not the licensed path directly: it prefers the
    # overlay where one is staged and falls back to the tracked corpus
    # otherwise. Hardcoding the overlay made this unrunnable on any checkout
    # without it, CI included, and none of the four curation targets is a
    # RESTRICTED framework -- all four are in the tracked corpus too.
    from tract.text_selection import merged_corpus_path  # noqa: PLC0415

    corpus = json.loads(merged_corpus_path().read_text(encoding="utf-8"))
    frameworks = {f["framework_id"]: f for f in corpus["frameworks"]}
    if framework_id not in frameworks:
        raise ValueError(
            f"{framework_id!r} is not in all_controls.json. Available: "
            f"{sorted(frameworks)}"
        )
    controls = frameworks[framework_id]["controls"]

    target = output_dir / f"annotate_{framework_id}.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "row", "control_id", "control_title", "control_statement",
            "ANSWER_hub_id", "ANSWER_second_hub_id_optional",
            "ANSWER_confidence_1_to_3", "ANSWER_notes",
        ])
        for index, control in enumerate(controls, start=1):
            statement = (control.get("description") or "").strip()
            if not statement:
                statement = (control.get("full_text") or "")[:CONTROL_TEXT_PREVIEW_CHARS]
            writer.writerow([
                index,
                control.get("control_id", ""),
                (control.get("title") or "").strip(),
                statement.replace("\n", " ").strip(),
                "", "", "", "",
            ])

    logger.info("%s: %d controls to annotate -> %s",
                framework_id, len(controls), target.name)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("curation_packet"),
        help="Where to write the packet. Should NOT be inside the repository "
             "working tree that the annotator does not get.",
    )
    parser.add_argument(
        "--frameworks", nargs="*", default=list(CURATION_TARGETS),
        help=f"Frameworks to build sheets for. Default: {', '.join(CURATION_TARGETS)}",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hub_sheet = build_hub_sheet(args.output_dir)
    sheets = [build_control_sheet(args.output_dir, f) for f in args.frameworks]

    logger.info("")
    logger.info("Packet written to %s", args.output_dir.resolve())
    logger.info("  %s", hub_sheet.name)
    for sheet in sheets:
        logger.info("  %s", sheet.name)
    logger.info("")
    logger.info("NOTHING in this packet is model-derived. Do not add "
                "results/ceiling_study/hub_reference.md: 400 of its hub descriptions "
                "were LLM-written from the gold links and would make the round "
                "Tier 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
