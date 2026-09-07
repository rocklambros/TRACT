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
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from tract.config import (
    BRIDGE_AI_FRAMEWORK_IDS,
    PROJECT_ROOT,
    PROCESSED_DIR,
    TRAINING_DIR,
)
from tract.io import atomic_write_json
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

# The sheet the annotator actually works in: the control's text beside empty
# answer columns, in one file. Before this existed the packet emitted only the
# three reference fields above while the importer required control_id, cre_id,
# confidence and rationale -- an overlap of exactly one column -- so a volunteer
# had to invent the response format and the import then rejected it.
ANNOTATION_SHEET_NAME: Final[str] = "annotate.csv"

# Provenance for the round. A returning sheet could not previously be tied to
# the packet it came from: if the hub roster shifts between builds -- which is
# what this round is for -- nothing recorded which 78 hubs a given annotator
# actually saw. The manifest also names the framework, which is what lets a test
# refuse a COMMITTED packet whose prose may not be redistributed.
MANIFEST_NAME: Final[str] = "manifest.json"

# Empty on emission, always. Anything in an answer column is a suggestion, and a
# suggestion makes the round Tier 3.
ANSWER_FIELDS: Final[tuple[str, ...]] = ("cre_id", "confidence", "rationale")

ANNOTATION_FIELDS: Final[tuple[str, ...]] = CONTROL_FIELDS + ANSWER_FIELDS

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
                "control_text": control_prose(control),
            })
            rows += 1
    return rows


def control_prose(control: dict[str, object]) -> str:
    """The fullest text the corpus holds for one control.

    `description` is capped at 2,000 characters by the parser's sanitiser, which
    cuts 58 of NIST 800-53's 300 controls mid-word -- one ends "Procedures can
    be documente". An annotator reading that is judging a control on partial
    text, and judging what the control is FOR is the entire task.

    Every one of those 58 carries a longer `full_text` (median +535 characters,
    up to +3,508), and across the framework `full_text` is never shorter than
    `description`, so taking the longer of the two recovers them with no case
    where it loses anything.

    CLAUDE.md: "Consider all available prose, always. Prefer a control's full
    text over its title everywhere text is selected."
    """
    description = str(control.get("description") or "").strip()
    full_text = str(control.get("full_text") or "").strip()
    return max(description, full_text, key=len)


def build_annotation_sheet(path: Path, framework_id: str) -> int:
    """Write the fillable sheet: control text beside empty answer columns.

    One file, so the annotator does not transcribe between a reference sheet
    and an answer sheet. `controls.csv` stays as the read-only reference.

    The answer columns are named exactly as `import_bridge_links` requires, so
    a filled sheet imports without an intermediate step. A round-trip test
    carries a real packet through a real fill to a real import, because both
    sides were previously tested in isolation and the join by nobody.
    """
    payload = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    framework = {f["framework_id"]: f for f in payload["frameworks"]}[framework_id]

    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_FIELDS))
        writer.writeheader()
        for control in framework["controls"]:
            writer.writerow({
                "control_id": control.get("control_id", ""),
                "control_title": (control.get("title") or "").strip(),
                "control_text": control_prose(control),
                # Empty, every one. See ANSWER_FIELDS.
                **dict.fromkeys(ANSWER_FIELDS, ""),
            })
            rows += 1
    return rows


def write_manifest(out_dir: Path, framework_id: str, n_hubs: int, n_controls: int) -> Path:
    """Record what this packet is, and pin its bytes.

    Metadata only -- no control prose. The digests let a filled sheet be tied to
    the exact packet an annotator was sent, and `framework_id` is what
    `tests/test_packet_manifest.py` checks against the licence table before
    allowing a packet to stay committed.
    """
    files = {}
    for name in (HUB_SHEET_NAME, CONTROL_SHEET_NAME, ANNOTATION_SHEET_NAME):
        files[name] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()

    manifest = {
        "framework_id": framework_id,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "n_hubs": n_hubs,
        "n_controls": n_controls,
        "files": files,
    }
    path = out_dir / MANIFEST_NAME
    atomic_write_json(manifest, path)
    return path


def _git_sha() -> str:
    """Short SHA of the tree that built this packet, or "unknown"."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


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
    n_rows = build_annotation_sheet(out_dir / ANNOTATION_SHEET_NAME, framework_id)
    write_manifest(out_dir, framework_id, n_hubs, n_controls)
    logger.info(
        "Packet written to %s: %d AI hubs, %d %s controls, %d annotation rows.",
        out_dir, n_hubs, n_controls, framework_id, n_rows,
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
