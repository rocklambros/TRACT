"""Corpus-derived stop word list and filtering.

Standing rule: filter common, low-information words so the model sees the
distinctive terms. The list is generated from this corpus rather than borrowed
from a general English list, because the words that carry no signal here are
security-document boilerplate -- "shall", "ensure", "appropriate", "system" --
and a generic list contains none of them while removing "not" and "no", which
in a control statement invert meaning.

Two things worth knowing before turning this on:

Removing function words moves text off the distribution a contextual encoder
was pretrained on, so the technique that reliably helps TF-IDF and BM25 can
cost recall for a dense bi-encoder like BGE. That is why filtering is a flag
rather than a default, and why the training pipeline carries it as an ablation
arm. Measure it, do not assume it.

Negations and modals are never removed regardless of frequency. "The system
shall not permit" and "the system shall permit" differ by one high-frequency
word, and a crosswalk that confuses them is worse than useless.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Final, Iterable

from tract.config import PROCESSED_DIR
from tract.io import atomic_write_json, load_json

logger = logging.getLogger(__name__)

# Collapses the gaps left where stop words were removed.
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

STOPWORDS_PATH: Final[Path] = PROCESSED_DIR / "stopwords.json"

# A word must appear in at least this fraction of documents to count as
# boilerplate. Set low because hub-name protection removes most candidates: at
# 0.10 it rejected 54 of 58, since the words frequency wants to strip are
# overwhelmingly the words CRE hubs are named after. 0.05 is where a list of
# genuine boilerplate survives that filter.
DEFAULT_MIN_DOC_FREQ: Final[float] = 0.05
# Hard cap so a small corpus cannot nominate half its vocabulary.
DEFAULT_MAX_WORDS: Final[int] = 300
# Below this, a token is a list marker, an initial, or a possessive fragment
# rather than a word. "b", "c", "s" and "ap" all reached the list without it.
MIN_WORD_LENGTH: Final[int] = 3

# Letters, with an internal hyphen or apostrophe only when a letter follows, so
# "third-party" and "model's" survive intact while "AC-2" yields "ac" rather
# than a dangling "ac-".
_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")

# Never removed, whatever the frequency. Each one changes the meaning of a
# control statement rather than decorating it.
PROTECTED_WORDS: Final[frozenset[str]] = frozenset({
    # negation and exclusion
    "no", "not", "nor", "never", "none", "without", "except", "unless",
    "neither", "cannot", "exclude", "excluding", "prohibited", "deny", "denied",
    # modality: "shall" and "may" are not interchangeable in a control
    "shall", "must", "should", "may", "required", "optional", "recommended",
    # quantifiers that change scope
    "all", "any", "only", "each", "every", "least", "most",
})


def tokenize(text: str) -> list[str]:
    """Lowercased alphabetic tokens. Digits and IDs are left out deliberately."""
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]


# Suffixes stripped when deriving a protection key, longest first. This is a
# deliberately crude stemmer, not linguistics: both sides of the comparison
# get the same treatment, so over-stemming can only ever protect a word that
# did not need protecting. That is the safe direction -- keeping a word costs
# a little filtering, deleting a discriminative one costs accuracy.
_PROTECTION_SUFFIXES: Final[tuple[str, ...]] = (
    "ations", "ation", "ities", "ity", "ings", "ing", "ness", "ments",
    "ment", "ives", "ive", "ally", "als", "al", "ies", "ers", "er",
    "ed", "es", "ly", "s",
)
# Below this a stem is more likely to collide than to help ("data" -> "dat").
_MIN_STEM_LENGTH: Final[int] = 4

# The CRE hubs are written in British English ("AI model behaviour integrity
# threats") while the control corpora are largely American, so "behavior" and
# "behaviour" never met and "behavior" was filtered out of text being matched
# against those hubs. Normalising one way is enough because both sides are
# keyed through the same function.
_SPELLING_NORMALISATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("iour", "ior"), ("our", "or"), ("yse", "yze"), ("ise", "ize"),
    ("isation", "ization"), ("ogue", "og"), ("aeon", "eon"), ("ae", "e"),
)


def protection_keys(word: str) -> set[str]:
    """Forms under which *word* counts as CRE vocabulary.

    Exact token matching was too literal and let real hub vocabulary into the
    stop word list. The hub "Privacy-preserving personal data logic" yields
    the single token "privacy-preserving", which protected that compound but
    not "privacy"; "Organizational AI security controls" protected
    "organizational" but not "organization". Both words were then filtered out
    of control text that had to be matched against those very hubs.

    So a word is keyed by itself, by each side of any hyphen, and by a light
    stem of each -- and a candidate is protected if ANY of its keys matches
    ANY key of the hub vocabulary.
    """
    keys: set[str] = set()
    lowered = word.lower().strip()
    if not lowered:
        return keys

    parts = {lowered}
    parts.update(p for p in lowered.replace("'", "-").split("-") if p)

    variants: set[str] = set()
    for part in parts:
        variants.add(part)
        for british, american in _SPELLING_NORMALISATIONS:
            if british in part:
                variants.add(part.replace(british, american))

    for part in variants:
        keys.add(part)
        # Derivation regularly turns a trailing y into i ("adversary" ->
        # "adversarial"), so key both. Without this the two never met and
        # "adversary" was filtered out of an AI-security corpus.
        if part.endswith("y") and len(part) > _MIN_STEM_LENGTH:
            keys.add(part[:-1] + "i")
        # EVERY applicable suffix, not just the first that matches. Stopping
        # at the first meant "examples" keyed only to "exampl" (strip "es")
        # and never to "example" (strip "s"), so the singular stayed
        # unprotected while the plural was hub vocabulary.
        for suffix in _PROTECTION_SUFFIXES:
            if part.endswith(suffix) and len(part) - len(suffix) >= _MIN_STEM_LENGTH:
                keys.add(part[: -len(suffix)])
    return keys


def generate_stopwords(
    documents: Iterable[str],
    min_doc_freq: float = DEFAULT_MIN_DOC_FREQ,
    max_words: int = DEFAULT_MAX_WORDS,
    protect: Iterable[str] | None = None,
) -> list[str]:
    """Derive stop words by document frequency over this corpus.

    Document frequency rather than raw count, so a single long document cannot
    push its own vocabulary into the list. Output is sorted for determinism:
    the same corpus must always produce the same list, or the list becomes an
    unrecorded input to every downstream metric.

    Args:
        protect: Words that must survive whatever their frequency. Callers pass
            the vocabulary of the CRE hub names here. This corpus is small and
            entirely about one subject, so frequency alone starts nominating
            "access", "control" and "data" well before it exhausts the function
            words, and those are precisely the tokens an assignment has to match
            on. A word that names a hub is never boilerplate.
    """
    doc_count = 0
    seen_in_docs: Counter[str] = Counter()
    for document in documents:
        tokens = set(tokenize(document))
        if not tokens:
            continue
        doc_count += 1
        seen_in_docs.update(tokens)

    if doc_count == 0:
        raise ValueError("Cannot generate stop words from an empty corpus.")

    protected: set[str] = set(PROTECTED_WORDS)
    for word in (protect or ()):
        protected |= protection_keys(word)

    candidates = [
        (word, count / doc_count)
        for word, count in seen_in_docs.items()
        if count / doc_count >= min_doc_freq
        and not (protection_keys(word) & protected)
        and len(word) >= MIN_WORD_LENGTH
    ]
    # Most frequent first for the cap, then alphabetical for a stable artifact.
    candidates.sort(key=lambda item: (-item[1], item[0]))
    words = sorted(word for word, _ in candidates[:max_words])

    logger.info(
        "Generated %d stop words from %d documents (min_doc_freq=%.2f)",
        len(words), doc_count, min_doc_freq,
    )
    return words


def filter_stopwords(text: str, stopwords: frozenset[str]) -> str:
    """Drop stop words, preserving the order and casing of what remains.

    Returns the original text unchanged if filtering would empty it. A control
    reduced to nothing is not a cheaper control, it is an unusable one.
    """
    if not text:
        return text

    # Delete the stop words in place rather than rebuilding the string from
    # matched tokens. _TOKEN matches alphabetic runs only, so re-joining its
    # matches silently discarded every digit and every punctuation mark in the
    # text: "AC-1" became "AC", "CWE-79" became "CWE", "V1.1.1" became "V",
    # "ML01:2023" became "ML", and "99.9%" vanished. In a corpus of security
    # controls the identifiers and thresholds are among the most distinctive
    # tokens there are, so the filter was destroying exactly what stop word
    # removal is supposed to leave behind. It also erased the " | " separator
    # that firewall.py recovers hub names from.
    filtered = _TOKEN.sub(
        lambda m: "" if m.group(0).lower() in stopwords else m.group(0), text,
    )
    filtered = _WHITESPACE.sub(" ", filtered).strip()
    if not filtered or not _TOKEN.search(filtered):
        # A control reduced to nothing, or to punctuation alone, is not a
        # cheaper control; it is an unusable one.
        return text
    return filtered


def load_stopwords(path: Path | None = None) -> frozenset[str]:
    """Load the committed list. Raises if it has not been generated."""
    target = path or STOPWORDS_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"No stop word list at {target}. Generate it with "
            "`python -m scripts.build_stopwords` so the list is a committed, "
            "versioned input rather than something recomputed per run."
        )
    data = load_json(target)
    return frozenset(data["stopwords"])


def save_stopwords(words: list[str], path: Path | None = None, **metadata: object) -> Path:
    """Write the list as a versioned artifact."""
    target = path or STOPWORDS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {"stopwords": sorted(words), "count": len(words), **metadata}, target,
    )
    logger.info("Wrote %d stop words to %s", len(words), target)
    return target
