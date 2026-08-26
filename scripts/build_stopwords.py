"""Generate the corpus-derived stop word list and the framework-identity set.

    python -m scripts.build_stopwords
    python -m scripts.build_stopwords --min-doc-freq 0.20 --dry-run

Runs over every control text in data/processed/all_controls.json plus the hub
representations, so the list reflects both sides of the assignment. Output is
committed: both artifacts are inputs to every downstream metric, so they have
to be versioned rather than recomputed per run.

The two are built together because they are coupled. Framework-identity tokens
are protected from the frequency list, so document frequency can never nominate
one framework's acronym and leave the rest, which is precisely what happened to
"owasp": it spans ten frameworks and so cleared a document-frequency threshold
that "cwe" misses despite a larger raw count. Which acronyms get stripped is
then a single arm-level decision covering all of them, not a side effect of how
many frameworks happen to share a publisher.
"""
from __future__ import annotations

import argparse
import logging
import sys

from tract.config import PROCESSED_DIR
from tract.framework_identity import (
    IdentityDerivation,
    assert_identity_symmetry,
    derive_framework_identity_tokens,
    load_framework_corpora,
    load_hub_vocabulary,
    save_framework_identity_tokens,
)
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
    """Every control text and hub name, plus the vocabulary to protect.

    The protected vocabulary is the hub names and hierarchy paths, plus every
    framework-identity token. The second addition is what stops the frequency
    list from stripping one framework's acronym and not another's: those
    tokens are removed as a set by their own arm or not at all, and document
    frequency has no vote.
    """
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

    hub_vocabulary.update(build_identity_derivation().tokens)

    logger.info("Collected %d documents, %d protected words",
                len(documents), len(hub_vocabulary))
    return documents, hub_vocabulary


def build_identity_derivation() -> IdentityDerivation:
    """Run the three identity gates over the corpus this checkout holds."""
    return derive_framework_identity_tokens(
        load_framework_corpora(), load_hub_vocabulary(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the stop word list and the framework-identity set")
    parser.add_argument("--min-doc-freq", type=float, default=DEFAULT_MIN_DOC_FREQ,
                        help="Fraction of documents a word must appear in")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print both artifacts without writing them")
    args = parser.parse_args()

    corpora = load_framework_corpora()
    identity = derive_framework_identity_tokens(corpora, load_hub_vocabulary())

    documents, hub_vocabulary = collect_documents()
    words = generate_stopwords(
        documents, min_doc_freq=args.min_doc_freq, max_words=args.max_words,
        protect=hub_vocabulary,
    )

    # Every arm a run can select, checked before either artifact is written.
    # An asymmetric pair cannot be caught downstream: each fold applies one
    # set to every anchor and reports a single number, so the framework that
    # lost its name looks exactly like the ones that kept theirs.
    hub_words = load_hub_vocabulary()
    for use_stopwords in (False, True):
        for use_identity in (False, True):
            active: set[str] = set()
            if use_stopwords:
                active |= set(words)
            if use_identity:
                active |= set(identity.tokens)
            assert_identity_symmetry(frozenset(active), corpora, hub_words)

    print(f"\n{len(words)} stop words at min_doc_freq={args.min_doc_freq}:\n")
    for i in range(0, len(words), 8):
        print("   " + "  ".join(f"{w:<14}" for w in words[i:i + 8]))

    print(f"\n{len(identity.tokens)} framework-identity tokens:\n")
    for i in range(0, len(identity.tokens), 8):
        print("   " + "  ".join(f"{w:<14}" for w in identity.tokens[i:i + 8]))
    print(f"\n   rejected as hub vocabulary: {list(identity.rejected_hub_vocabulary)}")
    print(f"   rejected as not capitalised: {list(identity.rejected_not_capitalised)}")
    print(f"   rejected as absent: {list(identity.rejected_absent)}")

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
    identity_path = save_framework_identity_tokens(
        identity,
        n_frameworks=len(corpora),
        n_hub_vocabulary_words=len(hub_words),
    )
    print(f"Wrote {identity_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
