"""Generate the corpus-derived stop word list.

    python -m scripts.build_stopwords
    python -m scripts.build_stopwords --min-doc-freq 0.20 --dry-run

Runs over every control text in data/processed/all_controls.json plus the hub
representations, so the list reflects both sides of the assignment. Output is
committed: the list is an input to every downstream metric, so it has to be
versioned rather than recomputed per run.
"""
from __future__ import annotations

import argparse
import logging
import sys

from tract.config import PROCESSED_DIR
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.stopwords import (
    DEFAULT_MAX_WORDS,
    DEFAULT_MIN_DOC_FREQ,
    generate_stopwords,
    save_stopwords,
    tokenize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_documents() -> tuple[list[str], set[str]]:
    """Every control text and hub name, plus the hub-name vocabulary to protect."""
    documents: list[str] = []
    hub_vocabulary: set[str] = set()

    data = load_json(PROCESSED_DIR / "all_controls.json")
    records = data if isinstance(data, list) else next(
        (v for v in data.values() if isinstance(v, list)), []
    )
    for record in records:
        for control in record.get("controls") or []:
            text = " ".join(str(control.get(field) or "") for field in
                            ("title", "description", "full_text"))
            if text.strip():
                documents.append(text)

    hierarchy = CREHierarchy.model_validate(
        load_json(PROCESSED_DIR / "cre_hierarchy.json")
    )
    for node in hierarchy.hubs.values():
        documents.append(f"{node.hierarchy_path} {node.name}")
        # The hierarchy PATH is protected too, not just the name. A hub's text
        # representation is "{hierarchy_path} | {name}", so every word in the
        # path is something an assignment has to match on -- and protecting
        # only the name left the entire path vocabulary eligible for removal
        # from the control text being matched against it.
        hub_vocabulary.update(tokenize(node.name))
        hub_vocabulary.update(tokenize(node.hierarchy_path))

    logger.info("Collected %d documents, %d protected hub words",
                len(documents), len(hub_vocabulary))
    return documents, hub_vocabulary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the stop word list")
    parser.add_argument("--min-doc-freq", type=float, default=DEFAULT_MIN_DOC_FREQ,
                        help="Fraction of documents a word must appear in")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the list without writing it")
    args = parser.parse_args()

    documents, hub_vocabulary = collect_documents()
    words = generate_stopwords(
        documents, min_doc_freq=args.min_doc_freq, max_words=args.max_words,
        protect=hub_vocabulary,
    )

    print(f"\n{len(words)} stop words at min_doc_freq={args.min_doc_freq}:\n")
    for i in range(0, len(words), 8):
        print("   " + "  ".join(f"{w:<14}" for w in words[i:i + 8]))

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0

    path = save_stopwords(
        words,
        min_doc_freq=args.min_doc_freq,
        max_words=args.max_words,
        n_documents=len(documents),
        n_protected_hub_words=len(hub_vocabulary),
    )
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
