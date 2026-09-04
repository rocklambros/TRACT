"""Emit the Phase 2C annotator packet, from non-model sources only.

Two CSVs and nothing else:

    ai_hubs.csv   hub_id, hub_name, hierarchy_path, branch   -- all 78 AI hubs
    controls.csv  control_id, control_title, control_text     -- one framework

An annotator reads a control and names the hub it belongs to. Nothing in the
packet suggests an answer, ranks a candidate, or reports a similarity, because
a label produced in the presence of model output is Tier 3 under CAMPAIGN3.md
Section 2 and cannot sit in a gate denominator at any ratio.

ALL 78 HUBS, UNRANKED. An earlier design scoped this to the top 20 by "eval
weight". That was wrong twice. Gate 1 needs 23 hubs de-orphaned and a link
carries one cre_id, so a flawless annotator working a 20-hub sheet reaches 20
and fails -- the design terminated its own funding path. And "eval weight"
counts how often a hub appears as gold in the held-out split, which is a
selection rule derived from the test set: the leakage shape that withdrew two
prior campaigns, entering through the sampling frame instead of the corpus.

WHAT IS DELIBERATELY OMITTED. `cre_hierarchy.json` carries `related_hub_ids` on
51 hubs. That field is 100% Phase 2B's model-proposed bridge set (Tier 3,
results/bridge/PROVENANCE.md) with no OpenCRE-native content and no per-edge
provenance marker. It is never read here. `tests/test_bridge_packet.py` checks
that by scanning every cell for hub-id VALUES rather than by checking column
names, because a column called `see_also` carrying bare ids passes a header
check.

Read-only over the corpus. Writes only the packet directory it is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Final

from tract.config import (
    BRIDGE_AI_FRAMEWORK_IDS,
    PROCESSED_DIR,
    TRAINING_DIR,
)
from tract.licensing import refuse_external_redistribution

logger = logging.getLogger(__name__)

HUB_SHEET_NAME: Final[str] = "ai_hubs.csv"
CONTROL_SHEET_NAME: Final[str] = "controls.csv"

HUB_FIELDS: Final[tuple[str, ...]] = (
    "hub_id", "hub_name", "hierarchy_path", "branch",
)
CONTROL_FIELDS: Final[tuple[str, ...]] = (
    "control_id", "control_title", "control_text",
)

CURATED_BY_FRAMEWORK_PATH: Final[Path] = (
    TRAINING_DIR / "hub_links_by_framework_curated.json"
)


def ai_hub_ids() -> list[str]:
    """The AI-only hubs, from the curated link set. Sorted, unranked.

    AI-only rather than AI: a hub some traditional framework already links to is
    not orphaned and does not need a bridge. Today those two sets are identical
    -- the intersection is 0 -- and the distinction is kept because the whole
    point of the round is to change that.
    """
    payload: dict[str, list[dict[str, str]]] = json.loads(
        CURATED_BY_FRAMEWORK_PATH.read_text(encoding="utf-8")
    )
    ai: set[str] = set()
    traditional: set[str] = set()
    for framework_id, links in payload.items():
        target = ai if framework_id in BRIDGE_AI_FRAMEWORK_IDS else traditional
        for link in links:
            target.add(link["cre_id"])
    return sorted(ai - traditional)


def build_hub_sheet(path: Path) -> int:
    """Write the AI hub sheet. Returns the row count."""
    hierarchy = json.loads(
        (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
    )
    hubs = hierarchy["hubs"]
    branch_names = {
        hub_id: hubs.get(node.get("branch_root_id") or "", {}).get("name", "")
        for hub_id, node in hubs.items()
    }

    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HUB_FIELDS))
        writer.writeheader()
        for hub_id in ai_hub_ids():
            node = hubs.get(hub_id)
            if node is None:
                raise ValueError(
                    f"Hub {hub_id} is linked in the curated set but absent from "
                    "cre_hierarchy.json. The packet would name a hub the "
                    "annotator cannot look up."
                )
            # Field-by-field, never `**node`: the node also carries
            # related_hub_ids, which is Tier 3.
            writer.writerow({
                "hub_id": hub_id,
                "hub_name": node["name"],
                "hierarchy_path": node["hierarchy_path"],
                "branch": branch_names.get(hub_id, ""),
            })
            rows += 1
    return rows


def build_control_sheet(path: Path, framework_id: str) -> int:
    """Write the control sheet for one framework. Returns the row count.

    The caller has already refused restricted frameworks; this reads prose.
    """
    payload = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    frameworks = {f["framework_id"]: f for f in payload["frameworks"]}
    framework = frameworks[framework_id]

    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONTROL_FIELDS))
        writer.writeheader()
        for control in framework["controls"]:
            writer.writerow({
                "control_id": control.get("control_id", ""),
                "control_title": control.get("title", ""),
                # Full prose, per CLAUDE.md: the title is a last resort, not a
                # default. An annotator mapping titles is doing a different and
                # easier task than the one the model is scored on.
                "control_text": control.get("description", ""),
            })
            rows += 1
    return rows


def build_bridge_packet(
    out_dir: Path, framework_id: str, *, allow_undetermined: bool = False
) -> None:
    """Emit the packet for one framework into `out_dir`.

    Raises before reading any prose when the framework is restricted, so
    licensed text never enters memory on a refused call.
    """
    # Before any prose is read. This used to test OVERLAY_FRAMEWORK_IDS, on
    # the reasoning that a packet is external redistribution and so needs a
    # wider set than RESTRICTED. The reasoning was right and the constant was
    # wrong: OVERLAY is the git-TRACKING tier, and it omits csa_aicm and
    # csa_ccm, which REDISTRIBUTION_RESERVED_FRAMEWORK_IDS names precisely
    # because they may not be sent to a third party.
    refuse_external_redistribution(
        framework_id, allow_undetermined=allow_undetermined
    )

    payload = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    known = {f["framework_id"] for f in payload["frameworks"]}
    if framework_id not in known:
        raise ValueError(
            f"{framework_id!r} is not a parsed framework. Known ids: "
            f"{sorted(known)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_hubs = build_hub_sheet(out_dir / HUB_SHEET_NAME)
    n_controls = build_control_sheet(out_dir / CONTROL_SHEET_NAME, framework_id)
    logger.info(
        "Packet written to %s: %d AI hubs, %d %s controls.",
        out_dir, n_hubs, n_controls, framework_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Directory to write the packet.")
    parser.add_argument(
        "--allow-undetermined",
        action="store_true",
        help=(
            "Redistribute a framework whose licence this repository records as "
            "UNDETERMINED. Required for the D2 default, nist_800_53: its terms "
            "were never adjudicated here, though nist_800_63 and nist_ssdf are "
            "recorded as US Government works not subject to copyright. Cannot "
            "unlock a framework with a recorded prohibition."
        ),
    )
    parser.add_argument(
        "--framework-id",
        default="nist_800_53",
        help="Framework whose controls the annotator maps (D2: NIST 800-53 first).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_bridge_packet(
        args.out_dir,
        framework_id=args.framework_id,
        allow_undetermined=args.allow_undetermined,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
