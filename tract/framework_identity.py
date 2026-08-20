"""Framework-identity tokens: the acronyms that name a framework, not a control.

A bi-encoder trained on "OWASP ASVS V2.1.1 requires ..." can learn "the anchor
says OWASP, so answer an OWASP hub" instead of learning the mapping. That is a
shortcut, not security content, and the project's core constraint is a semantic
assignment g(control_text) -> CRE_position. Task 13 removed 656 characters of
ETSI document identifier from anchors on the same reasoning.

The defect this module exists to close was worse than the shortcut. The
corpus-derived stop word list nominated "owasp" on document frequency, because
OWASP spans ten frameworks and so clears a DOCUMENT-frequency threshold that
"cwe" misses despite a larger raw count concentrated inside one framework's own
documents. So one framework lost its identity token and thirteen kept theirs.
Under leave-one-framework-out that is an inconsistency across folds: the fold
holding out OWASP saw anchors scrubbed of their giveaway while every other fold
did not.

The set is DERIVED against measurable properties rather than hand-listed,
because a hand-list is an allowlist that drifts as the corpus changes, which is
exactly how "owasp" arrived. Three gates, each rejecting a class the others
admit:

1. The token is a component of a framework's machine id (``csa_ccm``,
   ``nist_ssdf``), not merely a word in its human title. A first cut keyed on
   the long title admitted "matrix" from "Cloud Controls Matrix", plus
   "regulation", "profile" and "landscape". Stripping "regulation" from every
   control because one framework is named a regulation removes real signal.

2. The token is written in capitals in most of its occurrences, measured over
   markup-stripped control text. This is what separates an acronym from an
   ordinary word that happens to sit in a machine id: "act", "agentic",
   "cheat", "cop", "exchange", "proactive", "sheets" and "top" are all id
   components and none is ever capitalised in prose. Measured on this corpus
   the band between the two classes is empty from 0.014 ("top") to 0.690
   ("mitre"), so the majority rule sits in open space rather than on a
   tuned edge. Markup stripping matters: URLs such as "cwe.mitre.org" and
   "owaspai.org" contribute lowercase spellings that the encoder never sees,
   and they alone dragged "mitre" to 0.152 before the URLs were removed.

3. The token appears nowhere in hub data. Gate 1 and gate 2 together still
   admit "nist", "ai", "llm" and "ml", and every one of those names or
   describes a CRE hub. A token an assignment has to match on is never
   stripped, whatever else it also happens to be.

The gates do unequal amounts of work, and it is worth being exact about which.
Gate 1 bounds the candidate universe: without it, gates 2 and 3 admit 1,137
capitalised non-hub tokens, "JWT" and "SIEM" and "CVE" and "FIPS" among them,
and a control that has lost the token "JWT" has lost its meaning. Gate 2 is
what separates an acronym from a word, and on this corpus it is the only thing
that separates them stably: gate 3 happens to catch "act", "cheat" and "top"
today because 400 generated hub descriptions use those words, and descriptions
get regenerated. Gate 3 is the protection, and it is the only gate that keeps
"nist", "ai", "llm" and "ml" out.

On this corpus, dropping gate 1 to the human titles alone changes the output
not at all, because every title-only word fails gate 2 anyway. That is a fact
about this corpus and not a reason to widen the gate, so the candidate universe
is pinned by a test rather than left to the accident that makes it invisible.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from tract.config import PROCESSED_DIR
from tract.io import atomic_write_json, load_json
from tract.stopwords import load_stopwords, tokenize
from tract.text_selection import strip_markup

logger = logging.getLogger(__name__)

FRAMEWORK_IDENTITY_PATH: Final[Path] = (
    PROCESSED_DIR / "framework_identity_tokens.json"
)

# Share the tokenizer's shape so the two sides agree on what a token is, but
# keep the surface form: the casing gate has nothing to measure once the text
# is lowercased.
_SURFACE_TOKEN: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z]+(?:['-][A-Za-z]+)*"
)

# A token counts as an acronym when most of its occurrences are capitalised.
# Stated as a majority rather than tuned: see gate 2 above for the measured
# band this sits inside.
MIN_UPPERCASE_FRACTION: Final[float] = 0.5

# Below this a machine-id component is an initial rather than a name. No
# framework id in this corpus carries a one-letter component, so the floor
# costs nothing today and stops a future id like "x_ai" from nominating "x".
MIN_TOKEN_LENGTH: Final[int] = 2


@dataclass(frozen=True)
class FrameworkCorpus:
    """One framework's identity and the control text it contributed."""

    framework_id: str
    framework_name: str
    documents: tuple[str, ...]


@dataclass(frozen=True)
class IdentityDerivation:
    """What the three gates admitted, what they rejected, and on what evidence.

    The rejections are part of the artifact on purpose. A derived set that
    reports only its members cannot be reviewed, because the interesting
    question is always which near-miss was excluded and by how much.
    """

    tokens: tuple[str, ...]
    uppercase_fraction: Mapping[str, float]
    occurrences: Mapping[str, int]
    rejected_absent: tuple[str, ...]
    rejected_hub_vocabulary: tuple[str, ...]
    rejected_not_capitalised: tuple[str, ...]

    def as_frozenset(self) -> frozenset[str]:
        return frozenset(self.tokens)


def identity_candidates(framework_id: str) -> list[str]:
    """The machine-id components of one framework, lowercased.

    Digits drop out with the tokenizer, so "nist_800_53" yields "nist" and
    "owasp_top10_2021" yields "owasp" and "top". The second is a candidate, not
    a member: gate 2 rejects it.
    """
    return [
        token for token in tokenize(framework_id.replace("_", " "))
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def self_acronym(framework_id: str, framework_name: str) -> str:
    """The acronym a framework calls itself by, derived from its title.

    Deliberately a DIFFERENT derivation from the three gates above, so the
    symmetry check below is an independent measurement rather than the set
    restated against itself.

    The rule is the first token of the human name when that token is written
    in capitals, and the first component of the machine id otherwise. Taking
    the first capitalised token anywhere in the name was wrong: "CoSAI
    Landscape of AI Security Risk Map" would have named itself "ai".

    Raises:
        ValueError: If neither the name nor the id yields a token.
    """
    first = _SURFACE_TOKEN.search(framework_name or "")
    if first is not None:
        surface = first.group(0)
        if surface.isupper() and len(surface) >= MIN_TOKEN_LENGTH:
            return surface.lower()

    parts = identity_candidates(framework_id)
    if not parts:
        raise ValueError(
            f"Framework {framework_id!r} named {framework_name!r} yields no "
            f"identity token from either its name or its id, so it cannot be "
            f"checked for symmetry."
        )
    return parts[0]


def _surface_counts(
    corpora: Sequence[FrameworkCorpus],
) -> tuple[Counter[str], Counter[str]]:
    """(total occurrences, capitalised occurrences) per lowercased token.

    Counted over markup-stripped text, which is what select_control_text hands
    the encoder. Counting the raw field would measure URLs the model never
    reads.
    """
    total: Counter[str] = Counter()
    capitalised: Counter[str] = Counter()
    for corpus in corpora:
        for document in corpus.documents:
            for match in _SURFACE_TOKEN.finditer(strip_markup(document)):
                surface = match.group(0)
                total[surface.lower()] += 1
                if surface.isupper():
                    capitalised[surface.lower()] += 1
    return total, capitalised


def derive_framework_identity_tokens(
    corpora: Sequence[FrameworkCorpus],
    hub_vocabulary: Iterable[str],
    min_uppercase_fraction: float = MIN_UPPERCASE_FRACTION,
) -> IdentityDerivation:
    """Run the three gates over every framework's machine id.

    Args:
        corpora: Every framework, its human name, and its control text.
        hub_vocabulary: Lowercased tokens drawn from hub names, hierarchy
            paths and hub descriptions. Anything here survives gate 3.
        min_uppercase_fraction: The majority rule of gate 2.

    Raises:
        ValueError: If the corpora are empty, the threshold is outside (0, 1],
            or the gates admit nothing. An empty result means the inputs are
            wrong rather than that the corpus contains no framework acronyms,
            and returning it would silently disable the filter.
    """
    if not corpora:
        raise ValueError(
            "Cannot derive framework-identity tokens from no frameworks."
        )
    if not 0.0 < min_uppercase_fraction <= 1.0:
        raise ValueError(
            f"min_uppercase_fraction must be in (0, 1], got "
            f"{min_uppercase_fraction!r}."
        )

    protected = {word.lower() for word in hub_vocabulary}
    total, capitalised = _surface_counts(corpora)

    candidates: set[str] = set()
    for corpus in corpora:
        candidates.update(identity_candidates(corpus.framework_id))

    tokens: list[str] = []
    absent: list[str] = []
    hub_words: list[str] = []
    not_capitalised: list[str] = []
    fractions: dict[str, float] = {}
    counts: dict[str, int] = {}

    for token in sorted(candidates):
        occurrences = total[token]
        counts[token] = occurrences
        fraction = 0.0 if occurrences == 0 else capitalised[token] / occurrences
        fractions[token] = fraction

        # Hub vocabulary is checked before the casing gate so that the
        # protected words land in the bucket that says WHY they survived.
        # "nist" is capitalised in every one of its 450 occurrences, so the
        # casing bucket would report it as an acronym and say nothing about
        # the reason it is not stripped.
        if token in protected:
            hub_words.append(token)
        elif occurrences == 0:
            # No occurrence means no evidence of acronym shape and nothing to
            # strip. Admitting it on faith would let a future id component
            # that IS an ordinary word in, unmeasured.
            absent.append(token)
        elif fraction < min_uppercase_fraction:
            not_capitalised.append(token)
        else:
            tokens.append(token)

    if not tokens:
        raise ValueError(
            f"No framework-identity token survived the gates over "
            f"{len(corpora)} frameworks. Rejected {len(hub_words)} as hub "
            f"vocabulary, {len(absent)} as absent from the corpus and "
            f"{len(not_capitalised)} as not capitalised. An empty set would "
            f"disable the filter without saying so."
        )

    derivation = IdentityDerivation(
        tokens=tuple(tokens),
        uppercase_fraction=fractions,
        occurrences=counts,
        rejected_absent=tuple(absent),
        rejected_hub_vocabulary=tuple(hub_words),
        rejected_not_capitalised=tuple(not_capitalised),
    )
    logger.info(
        "Derived %d framework-identity tokens from %d frameworks "
        "(rejected %d hub words, %d absent, %d not capitalised)",
        len(tokens), len(corpora), len(hub_words), len(absent),
        len(not_capitalised),
    )
    return derivation


def assert_identity_symmetry(
    active: frozenset[str],
    corpora: Sequence[FrameworkCorpus],
    hub_vocabulary: Iterable[str],
) -> None:
    """Refuse a filter set that strips one framework's name and not another's.

    This is the defect, stated directly. Whether framework acronyms are
    stripped at all is a measurement question the ablation arm answers; whether
    OWASP loses its name while CWE and CAPEC keep theirs is not, because that
    difference shows up as a per-fold inconsistency under leave-one-framework-out
    and no metric reports it.

    Both directions are checked. A framework may keep its acronym only when
    that acronym is hub vocabulary, and a hub-vocabulary acronym must never be
    stripped however the set was built.

    Raises:
        ValueError: If some but not all eligible acronyms are stripped, or if
            a hub-vocabulary acronym is stripped.
    """
    protected = {word.lower() for word in hub_vocabulary}
    total, _ = _surface_counts(corpora)

    eligible: dict[str, str] = {}
    guarded: dict[str, str] = {}
    for corpus in corpora:
        acronym = self_acronym(corpus.framework_id, corpus.framework_name)
        if acronym in protected:
            guarded[acronym] = corpus.framework_id
        elif total[acronym] > 0:
            eligible[acronym] = corpus.framework_id

    leaked = sorted(set(guarded) & active)
    if leaked:
        raise ValueError(
            f"The filter set strips {leaked}, which name or describe a CRE "
            f"hub. A token an assignment has to match on is never removed: "
            f"filtering it from the control side makes the hub unreachable "
            f"and filtering it from both sides deletes the match entirely."
        )

    stripped = sorted(set(eligible) & active)
    kept = sorted(set(eligible) - active)
    if stripped and kept:
        raise ValueError(
            f"The filter set is asymmetric across frameworks. It strips "
            f"{stripped} (from {sorted(eligible[t] for t in stripped)}) while "
            f"keeping {kept} (from {sorted(eligible[t] for t in kept)}). "
            f"Every framework whose acronym is not hub vocabulary must lose "
            f"it or keep it together, or a leave-one-framework-out fold is "
            f"measuring which framework got scrubbed. If the kept frameworks "
            f"are the licensed overlay's, this checkout holds frameworks the "
            f"committed set never saw: rebuild it here with "
            f"`python -m scripts.build_stopwords` pointed at the overlay, and "
            f"do not commit the result."
        )


def filter_set(
    *, use_stopwords: bool, use_framework_identity: bool,
) -> frozenset[str] | None:
    """The words a run removes from control AND hub text, or None for neither.

    Returns None rather than an empty frozenset when both arms are off, because
    every caller downstream treats None as "do not filter" and records
    ``stopwords is not None`` in the fold provenance. An empty set would filter
    nothing while claiming the arm ran.
    """
    words: set[str] = set()
    if use_stopwords:
        words |= load_stopwords()
    if use_framework_identity:
        words |= load_framework_identity_tokens()
    return frozenset(words) if words else None


def load_framework_corpora(path: Path | None = None) -> list[FrameworkCorpus]:
    """Every framework's identity and control text. Sorted, for determinism.

    Defaults to the TRACKED corpus rather than merged_corpus_path(), matching
    what scripts/build_stopwords.py has always read. Two reasons, and both are
    about the artifact rather than the derivation. The committed set has to
    reproduce from a fresh clone, which is what the training host does, and the
    licensed overlay exists on no such clone. And a set derived from the
    overlay would carry "etsi" and per-token counts measured over restricted
    prose into a CC0 repository, which is the leak channel this project has
    already closed three times.

    A checkout that HOLDS the overlay therefore trains on frameworks the
    committed set does not cover. That is not left to trust:
    assert_identity_symmetry runs per fold against the corpus the fold read,
    so an overlay checkout running the identity arm fails loudly and names the
    framework whose acronym the set is missing.

    Raises:
        ValueError: If the corpus holds no framework records.
    """
    source = path or PROCESSED_DIR / "all_controls.json"
    data = load_json(source)
    records = data if isinstance(data, list) else next(
        (value for value in data.values() if isinstance(value, list)), []
    )
    if not records:
        raise ValueError(
            f"No framework records in {source}. The identity set would be "
            f"derived from nothing."
        )

    corpora: list[FrameworkCorpus] = []
    for record in records:
        documents: list[str] = []
        for control in record.get("controls") or []:
            text = " ".join(
                str(control.get(field) or "")
                for field in ("title", "description", "full_text")
            )
            if text.strip():
                documents.append(text)
        corpora.append(FrameworkCorpus(
            framework_id=str(record.get("framework_id") or ""),
            framework_name=str(record.get("framework_name") or ""),
            documents=tuple(documents),
        ))
    return sorted(corpora, key=lambda corpus: corpus.framework_id)


def load_hub_vocabulary(
    hierarchy_path: Path | None = None,
    description_paths: Sequence[Path] | None = None,
) -> set[str]:
    """Every token that names or describes a CRE hub.

    Wider than the vocabulary scripts/build_stopwords.py protects, and
    deliberately so. That one covers hub names and hierarchy paths, which is
    the "{hierarchy_path} | {name}" text a hub is actually represented by.
    This one adds the hub descriptions, because gate 3 is a protection rather
    than a matching surface: "nist" appears in no hub name and in no hierarchy
    path, only in two descriptions, and it is the live example of a framework
    acronym that must never be stripped. Over-inclusive is the safe direction
    here, since a token wrongly kept costs a little filtering and a token
    wrongly deleted costs a hub the assignment can no longer reach.

    Reads the description files only when they exist, so a checkout without
    them still derives a set rather than failing.
    """
    hierarchy = load_json(hierarchy_path or PROCESSED_DIR / "cre_hierarchy.json")
    vocabulary: set[str] = set()
    for node in hierarchy["hubs"].values():
        vocabulary.update(tokenize(node["name"]))
        vocabulary.update(tokenize(node["hierarchy_path"]))

    paths = description_paths if description_paths is not None else (
        PROCESSED_DIR / "hub_descriptions.json",
        PROCESSED_DIR / "hub_descriptions_reviewed.json",
    )
    for path in paths:
        if not path.exists():
            continue
        data = load_json(path)
        for entry in data.get("descriptions", {}).values():
            vocabulary.update(tokenize(str(entry.get("description") or "")))
    return vocabulary


def load_framework_identity_tokens(path: Path | None = None) -> frozenset[str]:
    """Load the committed set. Raises if it has not been generated."""
    target = path or FRAMEWORK_IDENTITY_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"No framework-identity token set at {target}. Generate it with "
            "`python -m scripts.build_stopwords` so the set is a committed, "
            "versioned input rather than something recomputed per run."
        )
    data = load_json(target)
    return frozenset(data["tokens"])


def save_framework_identity_tokens(
    derivation: IdentityDerivation,
    path: Path | None = None,
    **metadata: object,
) -> Path:
    """Write the derivation as a versioned artifact.

    Sorted throughout, because the set is an input to every anchor the
    identity arm produces and an artifact that reorders between runs is an
    unrecorded input to every downstream metric.
    """
    target = path or FRAMEWORK_IDENTITY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "tokens": sorted(derivation.tokens),
        "count": len(derivation.tokens),
        "min_uppercase_fraction": MIN_UPPERCASE_FRACTION,
        "rejected_absent": sorted(derivation.rejected_absent),
        "rejected_hub_vocabulary": sorted(derivation.rejected_hub_vocabulary),
        "rejected_not_capitalised": sorted(derivation.rejected_not_capitalised),
        "uppercase_fraction": {
            token: round(derivation.uppercase_fraction[token], 4)
            for token in sorted(derivation.uppercase_fraction)
        },
        "occurrences": {
            token: derivation.occurrences[token]
            for token in sorted(derivation.occurrences)
        },
        **metadata,
    }
    atomic_write_json(payload, target)
    logger.info(
        "Wrote %d framework-identity tokens to %s",
        len(derivation.tokens), target,
    )
    return target
