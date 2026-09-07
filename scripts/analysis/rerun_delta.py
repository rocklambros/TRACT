"""Compare round 2 against round 1: did removing the biased sentence move answers?

Round 1's handbook told annotators, before they had read anything, that "most
NIST 800-53 controls are about traditional IT security and have no AI-specific
hub". That is a claim about the answer distribution, made by the experimenter.
Round 2 removes it and re-runs the same 300 controls with the same annotators.

WHAT THIS CAN AND CANNOT ESTABLISH. The same people re-reading controls they
have already judged cannot produce an independent second measurement. Anchoring
is unavoidable.

But anchoring runs toward the NULL -- it makes people repeat themselves -- and
that asymmetry is what makes the round usable:

  * Answers barely move  -> WEAK evidence the sentence was not driving the NONE
    rate, because anchoring predicts that outcome too. Report it as
    consistent-with, never as "the instructions were fine".
  * Answers move materially -> STRONG evidence, because the movement happened
    DESPITE anchoring pushing the other way. It is a lower bound on the effect.

Because both rounds cover the same controls, the comparison is PAIRED, and the
right test on a paired binary decision is McNemar's -- which looks only at the
controls where an annotator changed their mind, and ignores the (large) majority
where they did not. A raw before/after rate difference on this data would be
dominated by the ~250 controls nobody was ever going to link.

Read-only. Loads no model.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Final, TypedDict

from tract.io import atomic_write_json

logger = logging.getLogger(__name__)

NO_HUB: Final[str] = "NONE"
DEFAULT_ROUND1: Final[Path] = Path.home() / "tract-inbox" / "phase2c"
DEFAULT_ROUND2: Final[Path] = Path.home() / "tract-inbox" / "phase2c-r2"


class AnnotatorDelta(TypedDict):
    """One annotator's movement between rounds."""

    annotator: str
    n_common: int
    r1_linked: int
    r2_linked: int
    none_to_link: int
    link_to_none: int
    unchanged: int
    hub_changed: int
    mcnemar_p: float | None
    direction: str


def _answers(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return {
            r["control_id"]: (r["cre_id"] or "").strip().upper()
            for r in csv.DictReader(handle)
            if (r["control_id"] or "").strip()
        }


def _mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value on the discordant pairs.

    Exact rather than chi-squared: the discordant counts here are small, and the
    chi-squared approximation is unreliable below about 25.
    """
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def compare(round1: Path, round2: Path, annotator: str) -> AnnotatorDelta:
    """Paired comparison for one annotator."""
    a = _answers(round1 / f"{annotator}.csv")
    b = _answers(round2 / f"{annotator}.csv")
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(
            f"{annotator}: the two rounds share no control ids, so nothing is "
            "paired and no comparison is defined."
        )

    none_to_link = sum(1 for c in common if a[c] == NO_HUB and b[c] != NO_HUB)
    link_to_none = sum(1 for c in common if a[c] != NO_HUB and b[c] == NO_HUB)
    hub_changed = sum(
        1 for c in common
        if a[c] != NO_HUB and b[c] != NO_HUB and a[c] != b[c]
    )
    unchanged = sum(1 for c in common if a[c] == b[c])

    p = _mcnemar_exact(none_to_link, link_to_none)
    if none_to_link == link_to_none:
        direction = "no net movement"
    elif none_to_link > link_to_none:
        direction = "toward linking (the direction removing the sentence predicts)"
    else:
        direction = "toward NONE (against the hypothesis)"

    return AnnotatorDelta(
        annotator=annotator,
        n_common=len(common),
        r1_linked=sum(1 for c in common if a[c] != NO_HUB),
        r2_linked=sum(1 for c in common if b[c] != NO_HUB),
        none_to_link=none_to_link,
        link_to_none=link_to_none,
        unchanged=unchanged,
        hub_changed=hub_changed,
        mcnemar_p=p,
        direction=direction,
    )


def _log(delta: AnnotatorDelta) -> None:
    logger.info("  %s", delta["annotator"])
    logger.info(
        "    linked: %d -> %d  of %d controls",
        delta["r1_linked"], delta["r2_linked"], delta["n_common"],
    )
    logger.info(
        "    NONE -> link : %3d      link -> NONE : %3d",
        delta["none_to_link"], delta["link_to_none"],
    )
    logger.info(
        "    same answer  : %3d      different hub: %3d",
        delta["unchanged"], delta["hub_changed"],
    )
    p = delta["mcnemar_p"]
    logger.info(
        "    McNemar (exact, two-sided): %s   %s",
        "no discordant pairs" if p is None else f"p = {p:.4f}",
        delta["direction"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", action="append", required=True)
    parser.add_argument("--round1", type=Path, default=DEFAULT_ROUND1)
    parser.add_argument("--round2", type=Path, default=DEFAULT_ROUND2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    deltas = [compare(args.round1, args.round2, a) for a in args.annotator]

    logger.info("=" * 66)
    logger.info("ROUND 2 vs ROUND 1 -- did removing the biased sentence move answers?")
    for delta in deltas:
        _log(delta)

    total_to_link = sum(d["none_to_link"] for d in deltas)
    total_to_none = sum(d["link_to_none"] for d in deltas)
    pooled = _mcnemar_exact(total_to_link, total_to_none)
    logger.info("")
    logger.info(
        "  pooled: %d NONE->link, %d link->NONE, McNemar %s",
        total_to_link, total_to_none,
        "n/a" if pooled is None else f"p = {pooled:.4f}",
    )
    logger.info("")
    logger.info(
        "  Anchoring biases this comparison TOWARD no movement. Movement is a "
        "lower bound on the effect; stillness is weak evidence against it and "
        "must not be reported as 'the instructions were fine'."
    )
    logger.info("=" * 66)

    if args.out is not None:
        atomic_write_json(
            {"annotators": deltas,
             "pooled_none_to_link": total_to_link,
             "pooled_link_to_none": total_to_none,
             "pooled_mcnemar_p": pooled},
            args.out,
        )
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
