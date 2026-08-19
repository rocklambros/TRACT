"""Print or persist the corpus join report.

    PYTHONPATH=. "$PY" scripts/corpus_report.py
    PYTHONPATH=. "$PY" scripts/corpus_report.py --tag before
    PYTHONPATH=. "$PY" scripts/corpus_report.py --out results/corpus/scratch.json

The same entry point produces the BEFORE artifact, every per-parser acceptance
check, and the final corpus report. One instrument, one code path: a parser
accepted by a measurement its consumer does not use is a parser accepted by
nothing.

--tag writes the pair a later reader needs together: results/corpus/<tag>.json
for the summary and results/corpus/link_resolution_<tag>.jsonl for the per-link
record. Both paths are anchored to PROJECT_ROOT, so the working directory does
not decide where evidence lands.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tract.corpus_report import (
    CORPUS_EVIDENCE_DIR,
    build_corpus_report,
    format_table,
    require_full_corpus,
    require_portable_paths,
    write_link_resolution,
    wrong_anchor_applicable,
)
from tract.io import atomic_write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tag", type=str, default=None,
        help="write results/corpus/<tag>.json and link_resolution_<tag>.jsonl",
    )
    args = parser.parse_args()

    report = build_corpus_report(args.links, args.corpus)
    print(format_table(report))
    print()
    print(f"corpus  {report.corpus_path}  sha256 {report.corpus_sha256[:16]}")
    print(f"links   {report.links_path}  sha256 {report.links_sha256[:16]}")
    print(f"frameworks in corpus  {report.corpus_framework_count}")
    print()
    print("wrong-anchor checks applicable, per framework:")
    for framework_id, applicable in sorted(wrong_anchor_applicable(report).items()):
        risk = report.by_id(framework_id).wrong_anchor_risk
        note = "" if applicable else "   (blind: no detector applies)"
        print(f"  {framework_id:26s} {risk:4d} of {applicable:5d}{note}")

    if args.out is not None:
        atomic_write_json(report.to_json(), args.out)
        print(f"wrote {args.out}")

    if args.tag is not None:
        # A tagged artifact is committed evidence that a later run is compared
        # against, so both guards run before anything is written. --out stays
        # unguarded on purpose: it is the scratch path, and nothing is gated
        # on what it produces.
        require_full_corpus(report)
        require_portable_paths(report)
        summary = CORPUS_EVIDENCE_DIR / f"{args.tag}.json"
        detail = CORPUS_EVIDENCE_DIR / f"link_resolution_{args.tag}.jsonl"
        atomic_write_json(report.to_json(), summary)
        write_link_resolution(report, detail)
        print(f"wrote {summary}")
        print(f"wrote {detail}")


if __name__ == "__main__":
    main()
