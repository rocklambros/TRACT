"""Detect verbatim licensed source text without storing any of it.

This repository is CC0 (see LICENSE and NOTICE), which is not a disclaimer. It
is an affirmative grant asserting the publisher holds the rights and waives
them. A restricted standard's control statement inside any tracked artifact is
a rights claim the project cannot make, for every downstream fork and mirror.

The gate that enforced this used to read the licensed source itself out of
`data/raw/`, which is gitignored. On a fresh clone and in CI that file is
absent, the gate skipped, and the skip reported green. It only ever ran on the
machine that happened to hold the source.

The fix is to keep the detector's knowledge in a form that can be committed:
salted SHA-256 hashes of normalised word n-grams drawn from each restricted
document. Hashes are one-way, so the tracked fingerprint file carries no
licensed text, and the gate works with `data/raw/` absent.

The salt is public and recorded inside the fingerprint file. It is not a
secret and does not need to be. Its job is to stop a generic precomputed
table of English n-grams from inverting the file, not to defend against
someone who already holds the standard.

Owner: TRACT. Regenerate with `python -m scripts.build_licensed_fingerprints`
whenever a restricted source is re-pinned.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Final

from tract.config import (
    FRAMEWORK_LICENSES,
    OVERLAY_FRAMEWORK_IDS,
    PROJECT_ROOT,
    REDISTRIBUTION_RESERVED_FRAMEWORK_IDS,
    UNDETERMINED_LICENSE,
)

logger = logging.getLogger(__name__)

# ── Shipped licence texts ─────────────────────────────────────────────────
#
# GPL-3.0 section 4 requires that a recipient of a covered work receive "a copy
# of this License along with the Program", and CC BY-SA 4.0 section 3(a)(1)(A)
# requires retaining a URI or hyperlink to the licence. Recording an SPDX
# identifier in NOTICE names the terms; it does not deliver them. This
# directory delivers them.
#
# The layout is the one Scancode, FOSSA, ClearlyDefined and `reuse lint` parse:
# a top-level LICENSES/ directory, one file per SPDX identifier, filename equal
# to the identifier plus .txt, body equal to the publisher's own text.
LICENSE_TEXTS_DIR: Final[Path] = PROJECT_ROOT / "LICENSES"
LICENSE_TEXT_SUFFIX: Final[str] = ".txt"

# TRACT's own contributions. Not in FRAMEWORK_LICENSES, which records only
# third-party framework terms, so it is named separately rather than derived.
PROJECT_LICENSE_ID: Final[str] = "CC0-1.0"

# SPDX expression operators, per the SPDX specification's licence expression
# grammar. Tokens that are operators are not identifiers.
_SPDX_OPERATORS: Final[frozenset[str]] = frozenset({"AND", "OR", "WITH"})

# The SPDX short-identifier grammar: letters, digits, dot, plus and hyphen.
_SPDX_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")


# ── Licence metadata for published artifacts ──────────────────────────────
#
# One source for the model card, the dataset card and the Zenodo record. They
# used to declare `mit`, `cc-by-sa-4.0` and `CC-BY-SA-4.0` respectively while
# pyproject.toml declared nothing, so a consumer reading any two of them got
# different terms for the same work.
#
# The value is `other` because no single identifier is true. The dataset draws
# on 31 frameworks including GPL-3.0 DSOMM and sources whose notices reserve
# redistribution. A CC BY-SA 4.0 grant over that is a conflicting affirmative
# grant, which is a worse error than the CC0 over-claim it was meant to fix:
# CC0 at least claims only what TRACT holds, while a share-alike grant purports
# to license other publishers' terms onto a downstream recipient.
#
# HuggingFace card metadata supports exactly this shape: `license: other` with
# a `license_name` and a `license_link`. The link is the NOTICE file shipped
# inside the artifact, so it resolves for a consumer who holds only the
# download and never visits the source repository.
PUBLISHED_LICENSE_ID: Final[str] = "other"
PUBLISHED_LICENSE_NAME: Final[str] = "tract-mixed-sources"
NOTICE_FILENAME: Final[str] = "NOTICE"
PUBLISHED_LICENSE_LINK: Final[str] = NOTICE_FILENAME

# The canonical source repository, quoted in both cards so a consumer who wants
# the full record can reach it. README.md and the model card citation agree on
# this URL.
SOURCE_REPOSITORY_URL: Final[str] = "https://github.com/rocklambros/TRACT"


def published_license_frontmatter() -> str:
    """The three YAML lines every published card declares its licence with.

    Returned as text rather than as a dict because both cards are rendered from
    f-string templates and a structured value would be re-serialised twice, in
    two places, which is how they diverged the first time.
    """
    return (
        f"license: {PUBLISHED_LICENSE_ID}\n"
        f"license_name: {PUBLISHED_LICENSE_NAME}\n"
        f"license_link: {PUBLISHED_LICENSE_LINK}"
    )


def withheld_control_text(framework_id: str) -> str:
    """What stands in for a control statement TRACT may not redistribute.

    Used by the canonical export, whose stated destination is an OpenCRE RFC.
    That is a third-party channel outside git, so a .gitignore line does
    nothing for it and the filter has to live in the exporter.

    An explanatory sentence rather than an empty string. A recipient holding
    only snapshot.json needs to tell "this publisher's terms do not permit us
    to hand you the text" apart from "this control has no description", and an
    empty field says the second thing. The sentence is TRACT's own wording, so
    nobody can mistake it for the standard's.

    Identifiers and titles are not withheld. OpenCRE already publishes the
    section identifiers and names of every framework in this corpus, and the
    mapping is the deliverable an RFC is asking for. Omitting these frameworks
    outright would drop their mappings from the proposal entirely, which
    withholds TRACT's own CC0 contribution to protect somebody else's text.
    """
    identifiers = spdx_identifiers(FRAMEWORK_LICENSES.get(framework_id, ""))
    terms = " AND ".join(identifiers) if identifiers else "its publisher's own notice"
    return (
        f"[Control text withheld. {framework_id} is published under {terms}, "
        f"which TRACT cannot sublicense. Section identifier, title and CRE "
        f"mapping are unaffected. See {NOTICE_FILENAME} at "
        f"{SOURCE_REPOSITORY_URL}.]"
    )


def exportable_description(framework_id: str, description: str) -> str:
    """The control text this framework's licence permits TRACT to hand over.

    Returns the description unchanged for a framework TRACT may redistribute,
    and `withheld_control_text` for one it may not.

    Lives here rather than in either exporter because there are two of them and
    they were not the same. `tract export-canonical` gained a licence filter and
    `tract export --opencre` did not, so the CSV path kept writing a populated
    `<Standard>|description` column for every framework it was given, into a
    default output directory at the repository root that was not gitignored. One
    function, two call sites, so the next exporter added inherits the filter
    rather than repeating the omission.

    Keyed on the framework rather than on the text, deliberately. A check that
    tried to judge "is this string too much of the standard" would need a
    threshold, and a threshold is the kind of parameter that gets raised until
    the gate stops firing.

    Raises:
        ValueError: framework_id is empty. A row with no framework cannot be
            checked against any tier, and defaulting it to "publishable" is
            how an unfiltered source would reach a third-party channel.
    """
    if not framework_id:
        raise ValueError(
            "cannot decide export licensing for a control with no "
            "framework_id. Every exported control must name its framework."
        )
    if framework_id not in OVERLAY_FRAMEWORK_IDS:
        return description
    return withheld_control_text(framework_id)


def copy_licensing_files(staging_dir: Path) -> None:
    """Copy LICENSE, NOTICE and LICENSES/ into a published artifact.

    Shared by the dataset bundle and the model publish path. Neither shipped
    NOTICE before, so the whole licensing record lived only in the source
    repository while the artifacts most consumers download carried a single
    wrong grant and no per-framework terms at all. Both cards' `license_link`
    resolves to the NOTICE this function delivers.

    Raises:
        FileNotFoundError: LICENSE, NOTICE or LICENSES/ is missing. Publishing
            an artifact whose licence record is incomplete is the outcome this
            function exists to prevent, so it is fatal rather than a warning.
    """
    import shutil

    # Every precondition before any write. A partial licence record in a
    # staging directory is worse than none: the publish path continues past it
    # and uploads an artifact that looks complete.
    sources = (PROJECT_ROOT / "LICENSE", PROJECT_ROOT / NOTICE_FILENAME)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(
                f"{source} is missing, so the published artifact would carry "
                f"no licence record. Restore it before publishing."
            )
    if not LICENSE_TEXTS_DIR.is_dir():
        raise FileNotFoundError(
            f"{LICENSE_TEXTS_DIR} is missing. GPL-3.0 section 4 requires the "
            f"licence text to travel with the work, and {NOTICE_FILENAME} "
            f"points recipients at that directory."
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, staging_dir / source.name)
    shutil.copytree(
        LICENSE_TEXTS_DIR,
        staging_dir / LICENSE_TEXTS_DIR.name,
        dirs_exist_ok=True,
    )
    logger.info(
        "Copied LICENSE, %s and %s/ into %s",
        NOTICE_FILENAME, LICENSE_TEXTS_DIR.name, staging_dir,
    )


def spdx_identifiers(licence: str) -> tuple[str, ...]:
    """The SPDX identifiers inside one recorded licence, in registry order.

    A recorded licence is treated as an SPDX expression when every
    whitespace-separated token is either an operator or matches the SPDX
    identifier grammar. Prose reservations fail that test because English
    carries commas and parentheses, which the grammar excludes: "(c) ENISA
    2021. Reproduction authorised..." stops at "(c)". The UNDETERMINED sentinel
    is excluded by name, since it matches the grammar and names no licence.

    A prose reservation returns an empty tuple. There is no licence text to
    ship for a source that grants nothing, and inventing one would be the
    guess FRAMEWORK_LICENSES exists to make visible.

    The residual risk is a future single-word prose licence such as
    "Proprietary", which this reads as an identifier and which then demands a
    file that cannot exist. That surfaces as a red build naming the framework,
    which is the intended direction to fail in.
    """
    if licence.strip() == UNDETERMINED_LICENSE:
        return ()
    tokens = licence.split()
    if not tokens:
        return ()
    identifiers: list[str] = []
    for token in tokens:
        if token in _SPDX_OPERATORS:
            continue
        if not _SPDX_IDENTIFIER.match(token):
            return ()
        identifiers.append(token)
    return tuple(identifiers)


def required_license_text_ids() -> frozenset[str]:
    """Every SPDX identifier this repository must ship a licence text for.

    Derived from FRAMEWORK_LICENSES rather than kept as a second hand-written
    list, so a newly ingested framework under a licence nobody has shipped the
    text of turns the gate red instead of joining the tree silently.
    """
    required = {PROJECT_LICENSE_ID}
    for licence in FRAMEWORK_LICENSES.values():
        required.update(spdx_identifiers(licence))
    return frozenset(required)


def shipped_license_text_ids() -> frozenset[str]:
    """Every SPDX identifier with a text file under LICENSES/ right now.

    Raises:
        FileNotFoundError: LICENSES/ is absent. Fatal rather than empty: an
            empty result would read as "nothing is required" to a caller that
            compares the two sets, which is the silence this directory ends.
    """
    if not LICENSE_TEXTS_DIR.is_dir():
        raise FileNotFoundError(
            f"{LICENSE_TEXTS_DIR} is missing. It carries the licence texts "
            f"GPL-3.0 section 4 and CC BY-SA section 3(a)(1)(A) require this "
            f"repository to deliver alongside the framework content it "
            f"redistributes."
        )
    return frozenset(
        path.stem
        for path in LICENSE_TEXTS_DIR.glob(f"*{LICENSE_TEXT_SUFFIX}")
    )


# ── Fingerprint parameters ────────────────────────────────────────────────
#
# N-GRAM LENGTH: 12 words. Chosen against measurement, not taste.
#
# Too short and the gate fires on shared security boilerplate that two
# independent documents arrive at on their own. Measured over this repository's
# 440 tracked text files against ISO/IEC 27001:2022 Annex A:
#
#   n=8   6 hits, every one a false positive (CSA AICM's own control text, an
#         OpenCRE export CSV, and the ISO parser's own docstring)
#   n=10  5 hits, 4 false positives, led by CSA AICM HRS-10, which shares
#         "agreements reflecting the organization s needs for the" with ISO
#         A.6.6 because both descend from the same NDA boilerplate
#   n=12  1 hit, a real partial quotation, no false positives
#   n=14  0 hits
#
# So 12 is the shortest window at which no independently authored document in
# this corpus collides. Anything shorter turns the gate into noise people learn
# to ignore, which is the failure mode this control exists to avoid.
#
# Too long and a partial quotation walks through. ISO Annex A statements run a
# 17-word median, so a 12-word window trips on a quotation of roughly 70% of a
# median statement, and 74 of the 93 statements contribute at least one window.
# The 19 statements shorter than 12 words cannot be fingerprinted at any useful
# length; they are one-line requirements whose wording is close to their own
# title, and the title column is deliberately excluded here because OpenCRE
# publishes those titles openly and they are already tracked.
#
# STANDING RULE, adopted 2026-08-19: RAISING THIS NUMBER TO CLEAR A HIT IS
# FORBIDDEN. It is the "gate that cannot fire" defect and this project has
# rejected that shape repeatedly. If the tree collides at 12, fix the tree.
#
# The rule has teeth because the numbers are known. During the R18 redaction the
# longest run of licensed text in a tracked document was 30 words and the
# shortest quoted control statement was 13, so n=14 would have cleared one
# offender and n=31 would have cleared every one of them, silently, with the
# text still in git. tests/test_licensed_text_not_tracked.py pins this constant
# so a change has to be argued rather than typed.
#
# The window is a FALSE-POSITIVE control, not a licence boundary. Text shorter
# than 12 words is not thereby licensed to anyone. The same redaction swept at
# n=7 and found four real DSOMM fragments of 7 to 9 words that no gate would
# ever catch; they were removed because they were the publisher's wording, not
# because a test demanded it.
NGRAM_WORDS: Final[int] = 12

# 128 bits of each digest. Over ~7 million candidate windows against ~17,000
# stored fingerprints the chance of a collision is about 3e-28, and the file is
# half the size of full-width hex.
FINGERPRINT_HEX_CHARS: Final[int] = 32

# Public, versioned, and recorded in the generated file. Bump the suffix only
# alongside a regeneration, because the gate refuses a file whose salt does not
# match this constant.
FINGERPRINT_SALT: Final[str] = "tract-licensed-fingerprint-v1"

FINGERPRINT_PATH: Final[Path] = (
    PROJECT_ROOT / "tests" / "fixtures" / "licensed_text_fingerprints.json"
)

# ── Which frameworks the fingerprint corpus covers ────────────────────────
#
# The corpus used to cover RESTRICTED_FRAMEWORK_IDS, so it held etsi and
# iso_27001 and nothing else. dsomm and csa_ccm route to the same gitignored
# overlay for the same reason -- their publishers' terms are ones a CC0 grant
# cannot carry -- and neither contributed a single n-gram. A task brief then
# asked for a tracked test fixture quoting real CCM specification text, the
# implementer invented the strings instead of copying them, and had they not,
# nothing here would have fired. A control whose enforcement is "someone reads
# carefully" is not a control.
#
# Derived from OVERLAY_FRAMEWORK_IDS rather than listed, so a framework moved
# into the overlay tier later is either fingerprinted or names itself in the
# KeyError that scripts/build_licensed_fingerprints.py raises for a framework
# with no extractor. Neither outcome is silence.
#
# CLOSED 2026-08-26. Both deferrals are gone and the set is empty, so
# fingerprinted_framework_ids() now returns OVERLAY_FRAMEWORK_IDS exactly and
# every framework whose prose is withheld from git is covered by the gate that
# exists to keep it withheld.
#
# Owner decision D1(b) removed csa_ccm from CONDITIONAL_FRAMEWORK_IDS on an
# explicit reading of the CSA membership terms. That dissolved both entries at
# once, and it is worth being precise about why, because "the exclusions went
# away" could describe a gate that stopped looking:
#
#   csa_ccm   No longer in the overlay, so there is nothing to defer. Its
#             prose is tracked under the owner's ruling. The deferral existed
#             because fingerprinting it would have red-flagged tracked
#             csa_aicm text -- 138 of AICM's 243 descriptions are
#             byte-identical to a CCM specification carrying the same control
#             id -- and that conflict is gone because CCM is no longer a
#             withheld framework.
#
#   csa_aicm  Never was in the overlay, so subtracting it never subtracted
#             anything. The entry existed to stop a future ruling from
#             switching the gate on as a side effect. That ruling has now
#             happened and went the other way, so the entry is dead.
#
# What was NOT done, then or now: trimming a fingerprint corpus to skip the
# shared 138. That produces a gate which passes because it stopped looking.
# The three withheld frameworks are fingerprinted at full coverage.
#
# If any framework is ever added to the overlay, it is fingerprinted. That is
# now the whole rule, with no exceptions to read.
FINGERPRINT_EXCLUDED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset()


RedistributionStatus = Literal["permitted", "reserved", "undetermined"]


def redistribution_status(framework_id: str) -> RedistributionStatus:
    """Three states, because "forbidden" and "nobody checked" are different facts.

    Redistribution is a DIFFERENT question from git-tracking, and this
    repository has a constant for each. RESTRICTED/OVERLAY answers "may this
    text exist in the tree"; REDISTRIBUTION_RESERVED answers this one. They are
    not nested:

        REDISTRIBUTION_RESERVED : csa_aicm, csa_ccm, etsi, iso_27001
        OVERLAY                 : dsomm, etsi, iso_27001

    So a guard built on OVERLAY -- which is what one packet builder used, and
    the other lacked entirely -- permits both CSA frameworks, and csa_aicm's
    recorded licence is "Proprietary ... no redistribution".

    "undetermined" is not "permitted". tract/config.py states the reason
    against itself: guessing a permissive licence for a source that never
    granted one is the mistake the licence table exists to make visible. It is
    separated from "reserved" because it is resolvable -- an owner can
    adjudicate it -- whereas a recorded prohibition is not.
    """
    if (
        framework_id in REDISTRIBUTION_RESERVED_FRAMEWORK_IDS
        or framework_id in OVERLAY_FRAMEWORK_IDS
    ):
        return "reserved"
    licence = FRAMEWORK_LICENSES.get(framework_id)
    if licence is None or licence == UNDETERMINED_LICENSE:
        return "undetermined"
    return "permitted"


def externally_redistributable(framework_id: str) -> bool:
    """True only for a framework with a recorded permissive licence."""
    return redistribution_status(framework_id) == "permitted"


def refuse_external_redistribution(
    framework_id: str, *, allow_undetermined: bool = False
) -> None:
    """Raise unless this framework's prose may leave the project.

    Call BEFORE reading any prose, so refused text never enters memory.

    `allow_undetermined` is the owner's acknowledgement that they have checked
    a source whose terms this repository never recorded. It cannot unlock a
    `reserved` framework, because that prohibition is recorded rather than
    unknown -- an override there would be overriding a fact, not a gap.
    """
    status = redistribution_status(framework_id)
    if status == "permitted":
        return
    if status == "undetermined" and allow_undetermined:
        logger.warning(
            "Redistributing %s under an UNDETERMINED licence on explicit "
            "acknowledgement. Record the adjudication in tract/config.py so "
            "the next caller does not have to make this decision again.",
            framework_id,
        )
        return

    licence = FRAMEWORK_LICENSES.get(framework_id, "unknown framework id")
    if status == "reserved":
        raise ValueError(
            f"{framework_id!r} may not be redistributed outside the project. "
            f"Recorded licence: {licence!r}. An annotator packet is "
            f"redistribution -- the prose goes to a person outside the "
            f"project. This is a recorded prohibition and has no override; if "
            f"a written grant covering named annotators exists, record it in "
            f"tract/config.py."
        )
    raise ValueError(
        f"{framework_id!r} has an UNDETERMINED licence, so it may not be "
        f"redistributed by default. Nobody has adjudicated its terms. Either "
        f"record them in tract/config.py, or pass allow_undetermined=True to "
        f"state that you have checked them yourself."
    )


def fingerprinted_framework_ids() -> frozenset[str]:
    """Every framework the tracked fingerprint corpus must cover.

    Raises:
        ValueError: an excluded id names no framework this project knows. A
            typo would silently exclude nothing and read as though it excluded
            something, which is the shape of a gate that quietly widened.
    """
    unknown = FINGERPRINT_EXCLUDED_FRAMEWORK_IDS - set(FRAMEWORK_LICENSES)
    if unknown:
        raise ValueError(
            f"FINGERPRINT_EXCLUDED_FRAMEWORK_IDS names {sorted(unknown)}, "
            f"which is not in FRAMEWORK_LICENSES. An exclusion that matches no "
            f"framework excludes nothing while reading as a live decision."
        )
    return frozenset(OVERLAY_FRAMEWORK_IDS) - FINGERPRINT_EXCLUDED_FRAMEWORK_IDS

# The generated file's schema. The gate asserts these are the ONLY keys
# present, which is what makes "this file contains no licensed text" a checked
# property rather than a claim: there is no free-text field to hide prose in.
FINGERPRINT_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset({
    "salt", "algorithm", "ngram_words", "hash_hex_chars", "generator",
    "documents", "fingerprints", "deferred_framework_ids",
})

# `deferred_framework_ids` records WHICH overlay frameworks this corpus does not
# cover, so a reader of the fixture alone can see the hole rather than infer it
# from a count. It holds framework IDS AND NOTHING ELSE, deliberately: the
# reason a framework is deferred is prose, and this file has no free-text field
# on purpose. The reasons live in FINGERPRINT_EXCLUDED_FRAMEWORK_IDS above and
# in the gate's docstring. The gate asserts this list equals that constant, so
# the fixture and the code cannot drift.
FINGERPRINT_DOCUMENT_KEYS: Final[frozenset[str]] = frozenset({
    "framework_id", "filename", "source_sha256", "ngram_count",
})

FINGERPRINT_ALGORITHM: Final[str] = "sha256"
FINGERPRINT_GENERATOR: Final[str] = "scripts/build_licensed_fingerprints.py"

# JSON string escapes reach a tracked .json file as two literal characters. Left
# alone, "\n" between two sentences normalises to a stray "n" token that splits
# an n-gram in half and lets a quotation inside a JSON string walk through.
_JSON_ESCAPE: Final[re.Pattern[str]] = re.compile(r"\\u[0-9a-f]{4}|\\[nrtbf\"'\\/]")

# PDF-to-markdown extraction breaks words across a line as "secu - rity". Both
# sides of the comparison have to heal that the same way or the healed side
# never matches the unhealed one.
_HYPHEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"([a-z])\s+-\s+([a-z])")

_NON_ALPHANUMERIC: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def normalise_for_fingerprint(text: str) -> str:
    """Reduce *text* to lowercase alphanumeric words separated by one space.

    Punctuation, case, typographic quotes and line breaks all vary between a
    source PDF, a parsed JSON string and a hand-typed quotation of the same
    sentence. Everything that varies is removed so that what remains is the
    wording, which is the thing the licence covers.
    """
    lowered = text.lower()
    lowered = _JSON_ESCAPE.sub(" ", lowered)
    lowered = _HYPHEN_SPLIT.sub(r"\1\2", lowered)
    return _NON_ALPHANUMERIC.sub(" ", lowered).strip()


def fingerprint_ngrams(
    text: str,
    ngram_words: int = NGRAM_WORDS,
    salt: str = FINGERPRINT_SALT,
    hex_chars: int = FINGERPRINT_HEX_CHARS,
) -> list[str]:
    """Salted, truncated SHA-256 of every *ngram_words*-word window in *text*.

    Stride is one word on both sides. A quotation that starts at any offset
    produces at least one window identical to a window of the source, provided
    it is at least *ngram_words* long.

    Raises:
        ValueError: ngram_words is not positive, or hex_chars is outside the
            width a sha256 hex digest can supply.
    """
    if ngram_words < 1:
        raise ValueError(f"ngram_words must be >= 1, got {ngram_words}")
    if not 1 <= hex_chars <= 64:
        raise ValueError(f"hex_chars must be in 1..64, got {hex_chars}")

    words = normalise_for_fingerprint(text).split()
    prefix = f"{salt}:".encode("utf-8")
    return [
        hashlib.sha256(prefix + " ".join(words[i:i + ngram_words]).encode("utf-8"))
        .hexdigest()[:hex_chars]
        for i in range(len(words) - ngram_words + 1)
    ]


@dataclass(frozen=True)
class LicensedDocument:
    """One restricted source document that contributed fingerprints."""

    framework_id: str
    filename: str
    source_sha256: str
    ngram_count: int


@dataclass(frozen=True)
class LicensedFingerprints:
    """The tracked fingerprint set, loaded and ready to test text against."""

    salt: str
    ngram_words: int
    hash_hex_chars: int
    documents: tuple[LicensedDocument, ...]
    fingerprints: frozenset[str]
    deferred_framework_ids: frozenset[str]

    @classmethod
    def load(cls, path: Path | None = None) -> LicensedFingerprints:
        """Read the tracked fingerprint file.

        Raises:
            FileNotFoundError: the file is absent. This is deliberately fatal.
                The predecessor gate skipped when its input was missing and
                reported green for months. A gate whose evidence is gone has
                failed, not passed.
            ValueError: the file's schema, salt or parameters disagree with
                this module, which means the two would silently compare
                against different normalisations.
        """
        target = path or FINGERPRINT_PATH
        if not target.exists():
            raise FileNotFoundError(
                f"{target} is missing. It is a tracked file and the licensed "
                f"text gate cannot run without it. Regenerate it with "
                f"`python -m scripts.build_licensed_fingerprints` on a "
                f"checkout that holds the restricted sources under data/raw/."
            )

        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{target}: expected a JSON object")

        unexpected = set(data) - FINGERPRINT_TOP_LEVEL_KEYS
        missing = FINGERPRINT_TOP_LEVEL_KEYS - set(data)
        if unexpected or missing:
            raise ValueError(
                f"{target}: key mismatch, unexpected={sorted(unexpected)} "
                f"missing={sorted(missing)}"
            )
        if data["salt"] != FINGERPRINT_SALT:
            raise ValueError(
                f"{target}: salt {data['salt']!r} does not match "
                f"FINGERPRINT_SALT {FINGERPRINT_SALT!r}. Every fingerprint in "
                f"the file was computed under the other salt and none of them "
                f"would ever match. Regenerate the file."
            )
        if data["algorithm"] != FINGERPRINT_ALGORITHM:
            raise ValueError(
                f"{target}: algorithm {data['algorithm']!r} != "
                f"{FINGERPRINT_ALGORITHM!r}"
            )

        documents = tuple(
            LicensedDocument(
                framework_id=str(entry["framework_id"]),
                filename=str(entry["filename"]),
                source_sha256=str(entry["source_sha256"]),
                ngram_count=int(entry["ngram_count"]),
            )
            for entry in data["documents"]
        )
        deferred = frozenset(str(v) for v in data["deferred_framework_ids"])
        if deferred != FINGERPRINT_EXCLUDED_FRAMEWORK_IDS:
            raise ValueError(
                f"{target}: deferred_framework_ids is {sorted(deferred)} but "
                f"FINGERPRINT_EXCLUDED_FRAMEWORK_IDS is "
                f"{sorted(FINGERPRINT_EXCLUDED_FRAMEWORK_IDS)}. The fixture and "
                f"the code disagree about which frameworks this corpus does not "
                f"cover, so one of them is telling a reader the wrong thing "
                f"about the size of the hole. Regenerate the file."
            )
        return cls(
            salt=str(data["salt"]),
            ngram_words=int(data["ngram_words"]),
            hash_hex_chars=int(data["hash_hex_chars"]),
            documents=documents,
            fingerprints=frozenset(str(value) for value in data["fingerprints"]),
            deferred_framework_ids=deferred,
        )

    def first_hit(self, text: str) -> str | None:
        """The first normalised window of *text* that is a stored fingerprint.

        Returns the offending window's own normalised words, not the licensed
        original, so a failure message can point at the quotation in the file
        under test without this repository ever printing the source.
        """
        words = normalise_for_fingerprint(text).split()
        prefix = f"{self.salt}:".encode("utf-8")
        width = self.hash_hex_chars
        n = self.ngram_words
        for i in range(len(words) - n + 1):
            window = " ".join(words[i:i + n])
            digest = hashlib.sha256(prefix + window.encode("utf-8")).hexdigest()
            if digest[:width] in self.fingerprints:
                return window
        return None
