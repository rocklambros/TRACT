"""Licensed framework text must never enter git.

This repository is CC0 (see LICENSE and NOTICE), which is not a disclaimer. It
is an affirmative grant asserting the publisher holds the rights and waives
them. Committing a copyrighted standard's control statements under it asserts
rights the project does not hold, for every downstream fork and mirror.

The control has to live here rather than in the publish path. `git push` is a
publication event and it fires before any `tract publish-*` command runs, so a
filter on the publish path guards the narrow channel and leaves the wide one
open. That was the first defect: the design specified an ISO filter on the
publish path while `data/processed/` was tracked and the repo was public.

The second defect was this file. The tree-wide scan read the licensed source
out of `data/raw/`, which is gitignored, and called `pytest.skip` when it was
absent. CI has no fetch step, so the gate never ran there, and on any fresh
clone it reported "1 skipped" and moved on. It only ever executed on the one
laptop holding the source. A gate that cannot fire is not a gate, and this one
was written as the fix for a prior escape.

It now runs off `tests/fixtures/licensed_text_fingerprints.json`, a tracked
file of salted hashes carrying no text. A missing fingerprint file is a
failure, never a skip. See tract/licensing.py for the parameters and
scripts/build_licensed_fingerprints.py for the rebuild.

The third defect was scope. The corpus covered RESTRICTED_FRAMEWORK_IDS, so it
held etsi and iso_27001 and nothing else, while dsomm and csa_ccm routed to the
same gitignored overlay for the same reason and contributed no fingerprints at
all. A task brief then asked an implementer to quote real CSA CCM specification
text into a tracked CC0 fixture. The implementer invented the strings instead,
and nothing here would have fired if they had not. Scanning the tree with the
widened corpus found the text already in git: six documents transcribing raw
sources under a "Verbatim sample" heading, a channel nobody had checked because
the gate that cleared them only knew two frameworks.

Scope is now `tract.licensing.fingerprinted_framework_ids()`: OVERLAY_FRAMEWORK_IDS
less FINGERPRINT_EXCLUDED_FRAMEWORK_IDS.

As of 2026-08-26 that subtraction removes nothing. FINGERPRINT_EXCLUDED_FRAMEWORK_IDS
is EMPTY and the scope is OVERLAY_FRAMEWORK_IDS exactly, so every framework whose
prose is withheld from git is covered here with no deferrals to read.

It held two until then, csa_aicm and csa_ccm, and the way they left is worth
keeping. csa_ccm was in the overlay and outside this corpus, which is the only
real hole this gate has ever had. It could not simply be added: 138 of csa_aicm's
243 TRACKED descriptions are byte-identical, after normalise_for_fingerprint, to
a CCM control specification under the SAME control id, so gating csa_ccm would
have failed the branch on tracked AICM text rather than on a defect. Owner
decision D1(b) ruled CSA material redistributable for this project, which made
csa_ccm's overlay membership a misclassification; it left the tier, its prose is
tracked, and the conflict went with it.

Trimming the CCM corpus to skip the shared 138 was considered and rejected, then
and now: it produces a gate that passes because it stopped looking. Nothing here
was trimmed. The three withheld frameworks are fingerprinted at full coverage.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tract.config import (
    FRAMEWORK_LICENSES,
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_DIR,
    RAW_FRAMEWORKS_DIR,
    RESTRICTED_FRAMEWORK_IDS,
)
from tract.licensing import (
    FINGERPRINT_DOCUMENT_KEYS,
    FINGERPRINT_EXCLUDED_FRAMEWORK_IDS,
    FINGERPRINT_PATH,
    FINGERPRINT_TOP_LEVEL_KEYS,
    NGRAM_WORDS,
    LicensedFingerprints,
    fingerprint_ngrams,
    fingerprinted_framework_ids,
)

# The generator's own extractors, imported rather than reimplemented. The
# positive control below used to hand-roll the ISO Annex A row parse, so the
# test's copy and the generator's could drift apart with nothing to notice.
from scripts.build_licensed_fingerprints import _EXTRACTORS

# RESTRICTED_FRAMEWORK_IDS lives in tract/config.py, not here. The merge step
# reads the same constant to decide what stays out of the tracked corpus, and
# a second copy in this file would let the writer and the gate disagree about
# which frameworks are licensed.

__all__ = ["RESTRICTED_FRAMEWORK_IDS"]

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Below this, a description is a section title rather than a control statement.
# ISO's tracked stub carried 93 titles at a 28-character median; its real Annex
# A statements run to a 138-character median.
_TITLE_LENGTH_CEILING: int = 60

# Text formats only. A .png or a .xlsx cannot be read as UTF-8, and a binary
# artifact that smuggled licensed prose would be a different problem with a
# different control.
# A 32-hex-character truncated sha256, the only shape a fingerprint may take.
_FINGERPRINT_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")


def _tracked_files(scope: str = "data/processed") -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", scope],
        capture_output=True, text=True, check=True,
        cwd=REPO_ROOT,
    )
    return {line for line in result.stdout.splitlines() if line}


@pytest.fixture(scope="module")
def fingerprints() -> LicensedFingerprints:
    """The tracked fingerprint set.

    No skip branch on purpose. LicensedFingerprints.load raises when the file
    is missing, which turns a fresh clone with no fixture into a red build
    rather than a green one.
    """
    return LicensedFingerprints.load()


class TestTheFingerprintFileCarriesNoText:
    """The file that lets the gate run offline must not itself be a copy.

    Structural, not heuristic. The schema has no free-text field, so there is
    nowhere for prose to hide, and these assertions are what keep it that way.
    """

    def test_every_value_is_a_hash_or_a_declared_metadata_field(self) -> None:
        data = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))

        assert set(data) == set(FINGERPRINT_TOP_LEVEL_KEYS), (
            "the fingerprint file grew or lost a top-level key. Any field not "
            "in FINGERPRINT_TOP_LEVEL_KEYS is a place licensed text could sit "
            "unnoticed."
        )
        for value in data["fingerprints"]:
            assert isinstance(value, str) and _FINGERPRINT_RE.match(value), (
                f"fingerprint {value!r} is not a 32-character hex digest"
            )
        for document in data["documents"]:
            assert set(document) == set(FINGERPRINT_DOCUMENT_KEYS)
            assert document["framework_id"] in fingerprinted_framework_ids()
            assert _SHA256_RE.match(document["source_sha256"])
            assert isinstance(document["ngram_count"], int)
            # A filename, not a document body. Bounded so a future change
            # cannot start writing an excerpt into this field.
            assert len(document["filename"]) <= 128

        # Ids and nothing else. The reason a framework is deferred is prose, and
        # this file has no free-text field on purpose, so a "reason" string here
        # would be the one place licensed text could sit unnoticed.
        for framework_id in data["deferred_framework_ids"]:
            assert isinstance(framework_id, str)
            assert framework_id in FRAMEWORK_LICENSES, (
                f"deferred_framework_ids names {framework_id!r}, which is not a "
                f"framework. This field carries ids, never prose."
            )

    def test_the_recorded_deferrals_match_the_code(self) -> None:
        """The fixture and the constant must agree on the size of the hole.

        A reader holding only the fixture should be able to see which overlay
        frameworks it does not cover. If that list drifts from the constant the
        generator reads, the fixture tells them the wrong thing.

        Reads the raw JSON rather than taking the `fingerprints` fixture, and
        that is the whole point. LicensedFingerprints.load already raises on a
        mismatch, so an assertion against the LOADED object could never fail:
        the loader would have raised first and the test would have been a
        tautology dressed as a check. Mutation testing caught exactly that --
        blanking the field in the file killed the suite through the loader while
        this assertion never ran.
        """
        data = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
        assert (
            frozenset(data["deferred_framework_ids"])
            == FINGERPRINT_EXCLUDED_FRAMEWORK_IDS
        ), (
            f"the fixture records {sorted(data['deferred_framework_ids'])} as "
            f"deferred but the code defers "
            f"{sorted(FINGERPRINT_EXCLUDED_FRAMEWORK_IDS)}."
        )

    def test_the_loader_rejects_a_fixture_that_misstates_its_deferrals(
        self, tmp_path: Path
    ) -> None:
        """The guard behind the tautology above, exercised where it lives.

        Reachable in both directions: the unmodified file loads, and the same
        file with a deferral the code does not declare does not.

        The mismatch is manufactured by ADDING an id, not by removing one.
        Removing the first element was the original construction and it stopped
        working on 2026-08-26, when owner decision D1(b) emptied
        FINGERPRINT_EXCLUDED_FRAMEWORK_IDS: `sorted(frozenset())[1:]` is `[]`,
        which equals the constant, so the loader was right to accept it and the
        test failed for being unable to build a bad case. Adding always
        disagrees, empty set or not, and empty is now the expected steady
        state.
        """
        data = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
        good = tmp_path / "good.json"
        good.write_text(json.dumps(data), encoding="utf-8")
        assert LicensedFingerprints.load(good).deferred_framework_ids == (
            FINGERPRINT_EXCLUDED_FRAMEWORK_IDS
        )

        data["deferred_framework_ids"] = sorted(
            set(FINGERPRINT_EXCLUDED_FRAMEWORK_IDS) | {"csa_ccm"}
        )
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="deferred_framework_ids"):
            LicensedFingerprints.load(bad)

    def test_the_gate_still_covers_something(self) -> None:
        """A deferral that reached every framework would empty the gate.

        Separated from the deferral-equality test because it fails for a
        different reason and would otherwise hide behind it.
        """
        assert fingerprinted_framework_ids(), (
            "every overlay framework is deferred, so this gate checks nothing"
        )

    def test_every_framework_in_scope_contributed_fingerprints(
        self, fingerprints: LicensedFingerprints
    ) -> None:
        """A source in scope with no fingerprints is an unguarded source."""
        covered = {document.framework_id for document in fingerprints.documents}
        assert covered == fingerprinted_framework_ids(), (
            f"fingerprints cover {sorted(covered)} but the corpus is scoped to "
            f"{sorted(fingerprinted_framework_ids())}. Run "
            f"`python -m scripts.build_licensed_fingerprints`."
        )
        for document in fingerprints.documents:
            assert document.ngram_count > 0, (
                f"{document.framework_id} contributed no n-grams, so nothing "
                f"about that source is actually being checked"
            )

    def test_every_overlay_framework_has_an_extractor(self) -> None:
        """Including the deferred ones, so a deferral stays reversible.

        A deferred framework whose extractor was never written or was later
        deleted turns a licence decision into a code project. csa_ccm is
        deferred and its extractor is registered for exactly this reason.
        """
        missing = sorted(set(OVERLAY_FRAMEWORK_IDS) - set(_EXTRACTORS))
        assert not missing, (
            f"overlay framework(s) {missing} have no fingerprint extractor. A "
            f"framework added to the tier must be gated or explicitly deferred, "
            f"and either way the extractor has to exist."
        )

    def test_the_ngram_window_has_not_been_widened(self) -> None:
        """Standing rule: raising NGRAM_WORDS to clear a hit is forbidden.

        Pinned so the change has to be argued rather than typed. During the R18
        redaction the longest run of licensed text in a tracked document was 30
        words and the shortest quoted control statement was 13, so n=14 would
        have cleared one offender and n=31 would have cleared all of them with
        the text still in git.

        12 is also the measured floor: at n=10 CSA AICM's own HRS-10 collides
        with ISO A.6.6 on shared NDA boilerplate. So this is not a free
        parameter in either direction.
        """
        assert NGRAM_WORDS == 12


def test_the_gate_fires_on_a_planted_quotation(
    fingerprints: LicensedFingerprints, tmp_path: Path
) -> None:
    """A positive control that needs neither data/raw/ nor licensed text.

    Proving the scanner fires would ordinarily mean planting a real quotation,
    which is the one thing this file must not contain. Instead the real
    fingerprint set is extended by one entry, the hash of an invented sentence,
    and the scanner is asked to find that sentence. Same normalisation, same
    hashing, same lookup as the tree scan below, so a break in any of them
    turns this red.
    """
    planted = (
        "Every regional facility badge shall be issued reviewed and revoked "
        "under a documented process approved by the security lead"
    )
    grams = fingerprint_ngrams(planted)
    assert grams, "the planted sentence is shorter than one n-gram window"

    augmented = LicensedFingerprints(
        salt=fingerprints.salt,
        ngram_words=fingerprints.ngram_words,
        hash_hex_chars=fingerprints.hash_hex_chars,
        documents=fingerprints.documents,
        fingerprints=fingerprints.fingerprints | set(grams),
        deferred_framework_ids=fingerprints.deferred_framework_ids,
    )

    victim = tmp_path / "innocent_looking.json"
    victim.write_text(
        json.dumps({"control_id": "X.1", "description": planted}), encoding="utf-8",
    )
    hit = augmented.first_hit(victim.read_text(encoding="utf-8"))
    assert hit is not None, "the scanner missed a quotation it holds the hash of"

    # And the unmodified set does not flag it, so the hit above came from the
    # planted entry rather than from a coincidence in the real fingerprints.
    assert fingerprints.first_hit(planted) is None


@pytest.mark.parametrize("framework_id", sorted(fingerprinted_framework_ids()))
def test_real_text_planted_in_a_tracked_looking_file_is_flagged(
    fingerprints: LicensedFingerprints, tmp_path: Path, framework_id: str,
) -> None:
    """End-to-end positive control, per framework, against the real source.

    Redundant with the planted-quotation test above, which is why this one may
    skip: it needs the gitignored source, and the gate itself must not. Kept
    because it is the only check that the committed hashes were computed from
    the documents they claim, one framework at a time. A rebuild that silently
    dropped a framework's n-grams would pass every structural test in this file
    and fail here.

    The statement is written into a file and the file's BODY is scanned, rather
    than passing the string straight to first_hit. That is the path
    test_no_verbatim_licensed_statement_anywhere_in_the_tree takes, JSON
    escaping and all, so a normalisation change that breaks the tree scan
    breaks this too.

    The extraction comes from the generator's own `_EXTRACTORS`. It used to be a
    hand-rolled copy of the ISO Annex A row parse living in this file, which
    could drift from the generator with nothing to notice. Hand-rolling this
    logic has already produced one false negative on a file the real gate caught
    seconds later.
    """
    filename, extract = _EXTRACTORS[framework_id]
    source = RAW_FRAMEWORKS_DIR / framework_id / filename
    if not source.exists():
        pytest.skip(
            f"raw {framework_id} source absent; the planted-quotation test "
            f"covers the mechanism"
        )

    units = extract(source)
    assert units, f"{framework_id}: the extractor returned no units"

    # The recorded count must match what the extractor produces right now. Only
    # checking that SOME window matches would let an extractor quietly narrow:
    # dropping `risk` and `measure` from a DSOMM statement still leaves
    # `description`, whose windows are a subset of the stored ones, so every
    # other assertion in this file would stay green while the corpus lost 90% of
    # its coverage.
    recorded = next(
        d.ngram_count for d in fingerprints.documents
        if d.framework_id == framework_id
    )
    rebuilt = sum(len(fingerprint_ngrams(unit)) for unit in units)
    assert rebuilt == recorded, (
        f"{framework_id} records {recorded} n-grams but its extractor now "
        f"yields {rebuilt}. The extraction changed without the fixture being "
        f"regenerated, so the corpus no longer describes what could leak."
    )

    longest = max(units, key=len)

    victim = tmp_path / "innocent_looking.json"
    victim.write_text(
        json.dumps({"control_id": "X.1", "description": longest}),
        encoding="utf-8",
    )
    hit = fingerprints.first_hit(victim.read_text(encoding="utf-8"))
    assert hit is not None, (
        f"the committed fingerprints do not flag {framework_id}'s own longest "
        f"statement. Either they were built from a different document than the "
        f"one they name, or that framework's n-grams were dropped. Re-run "
        f"`python -m scripts.build_licensed_fingerprints`."
    )


def test_no_verbatim_licensed_statement_anywhere_in_the_tree(
    fingerprints: LicensedFingerprints,
) -> None:
    """No tracked file may quote a restricted standard's text.

    The first version of this gate scanned only data/processed, so it could not
    see licensed text that reached git through a different door. It did not:
    a later change pinned ISO Annex A statements into tracked test fixtures and
    assertions, and this gate reported clean. A control that guards one channel
    while another stays open is the failure it was written to prevent.
    """
    offenders: list[tuple[str, str]] = []
    scanned = 0
    for relative in sorted(_tracked_files(".")):
        path = REPO_ROOT / relative
        if path == FINGERPRINT_PATH:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binary, or unreadable. Not skipped by NAME: the suffix allowlist
            # this replaced skipped 16 tracked files, and Path(".gitignore")
            # .suffix is "" -- which is how a 12-word run of Annex A ended up
            # in a .gitignore comment, in the very commit that fixed the
            # original leak, and this gate reported clean. Decoding is the
            # right test because it fails CLOSED: a new text file is scanned
            # whatever it is called.
            continue
        scanned += 1
        hit = fingerprints.first_hit(body)
        if hit is not None:
            offenders.append((relative, hit))

    assert scanned > 100, (
        f"only {scanned} tracked files were scanned. The gate is meant to "
        f"cover the whole tree, and a collapsed file list is how it goes quiet."
    )
    assert not offenders, (
        f"{len(offenders)} tracked file(s) reproduce {fingerprints.ngram_words} "
        f"or more consecutive words of a restricted source: {offenders[:5]}. "
        f"This repository is CC0, which asserts the publisher holds the rights "
        f"and waives them. Move the text to a gitignored path, or replace it "
        f"with synthetic wording that reproduces the same structure."
    )


def test_overlay_framework_files_are_not_tracked() -> None:
    """The per-framework JSON for an overlay source must be gitignored.

    Widened from RESTRICTED_FRAMEWORK_IDS to OVERLAY_FRAMEWORK_IDS. The narrow
    form checked two of the frameworks that route to the gitignored overlay, so
    dsomm's GPL-3.0 text could have been committed with nothing watching. All
    three pass today; the point is that the fourth cannot arrive unchecked.
    """
    tracked = _tracked_files()
    for framework_id in sorted(OVERLAY_FRAMEWORK_IDS):
        path = f"data/processed/frameworks/{framework_id}.json"
        assert path not in tracked, (
            f"{path} is tracked by git. This repository is CC0, so committing "
            f"{framework_id} control statements dedicates licensed text to the "
            f"public domain. Run: git rm --cached {path}"
        )


def test_every_overlay_framework_has_a_gitignore_line() -> None:
    """Untracking a file is not enough; the next parser run re-adds it.

    `git rm --cached` removes the file from the index. Without a .gitignore
    entry the next `git add -A` after a parser run puts it straight back, and
    the test above only catches that once it has already happened.

    Widened to OVERLAY_FRAMEWORK_IDS alongside the test above, for the same
    reason.
    """
    ignore_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    for framework_id in sorted(OVERLAY_FRAMEWORK_IDS):
        expected = f"data/processed/frameworks/{framework_id}.json"
        assert expected in ignore_lines, (
            f"{expected} is missing from .gitignore, so a parser run followed "
            f"by `git add -A` would commit {framework_id}'s licensed text."
        )


def test_every_overlay_framework_has_a_fold_predictions_gitignore_line() -> None:
    """A LOFO fold's predictions.json IS the held-out framework's prose.

    Distinct from the test above, which covers the parser's output path. This
    covers the LOFO orchestrator's: `collect` rsyncs a fleet's results into
    results/phase1b/<config>/fold_<framework>/predictions.json, and every row
    carries the eval anchor verbatim. When the held-out framework is licensed,
    that file is the licensed text.

    The .gitignore comment claimed a test enforced this and none did, which is
    how DSOMM -- roughly half the fingerprint corpus -- went uncovered while the
    two narrower RESTRICTED members were listed. Keyed on OVERLAY_FRAMEWORK_IDS
    for that reason.

    The directory name comes from the link's standard_name, not the framework
    id: run_fold.py builds it as `fold_{args.framework.replace(' ', '_')}` and
    args.framework is a standard_name. Deriving it here from the curated links
    rather than hardcoding it means a display-name change breaks this test
    instead of silently unprotecting a framework.
    """
    from scripts.phase0.common import CURATED_LINKS_PATH
    from tract.config import OVERLAY_FRAMEWORK_IDS

    # Read the JSONL rather than load_curated_links(): HubStandardLink drops
    # framework_id and keeps only standard_name, and the join between the two
    # is exactly what this test needs.
    names_by_id: dict[str, set[str]] = {}
    with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            framework_id = record.get("framework_id")
            standard_name = record.get("standard_name")
            if framework_id and standard_name:
                names_by_id.setdefault(framework_id, set()).add(standard_name)

    ignore_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    checked = 0
    for framework_id in sorted(OVERLAY_FRAMEWORK_IDS):
        for standard_name in sorted(names_by_id.get(framework_id, set())):
            fold_dir = f"fold_{standard_name.replace(' ', '_')}"
            expected = f"results/phase1b/**/{fold_dir}/predictions.json"
            checked += 1
            assert expected in ignore_lines, (
                f"{expected} is missing from .gitignore. A LOFO fold holding "
                f"out {standard_name!r} writes that framework's licensed prose "
                f"to predictions.json, and `collect` followed by `git add -A` "
                f"would commit it."
            )
    assert checked >= len(OVERLAY_FRAMEWORK_IDS), (
        f"Only {checked} fold paths checked for "
        f"{len(OVERLAY_FRAMEWORK_IDS)} overlay frameworks. An overlay framework "
        "with no curated link contributes no standard_name, so this test would "
        "pass while protecting nothing."
    )


def test_no_new_tracked_but_ignored_prediction_files_appear() -> None:
    """Pin the set of files that are BOTH tracked and ignored.

    Three campaign-1 fold predictions for ISO 27001 are in the index and also
    match a .gitignore rule added later. Git ignore rules do not apply to files
    already tracked, so for those three paths the rule is inert: a future run
    writing to them followed by `git add -A` would commit whatever they contain.

    They are NOT untracked here. All three hold title-only anchors and match
    zero licensed fingerprints (verified), several are cited by path from
    CAMPAIGN2.md, and PREINPUTS-ARCHIVE.md records the deliberate decision to
    keep superseded runs as evidence. `git rm --cached` would delete clean
    evidence to close a gap that the tree-wide scan in this module already
    covers -- `git ls-files` reads the INDEX, so tracked-but-ignored files are
    scanned like any other.

    What is not covered is a FOURTH such file appearing, which would arrive
    with no rule stopping it and no reason for anyone to look. This pins the
    set so that has to be a decision rather than an accident.
    """
    result = subprocess.run(
        ["git", "ls-files", "-i", "-c", "--exclude-standard"],
        capture_output=True, text=True, check=True, cwd=REPO_ROOT,
    )
    predictions = {
        line for line in result.stdout.splitlines()
        if line.endswith("predictions.json")
    }
    known = {
        "results/phase1b/c2_A1_prose_sw_bge/fold_ISO_27001/predictions.json",
        "results/phase1b/c2_A2_prose_sw_bge_bal3/fold_ISO_27001/predictions.json",
        "results/phase1b/c2_canary_qwen/fold_ISO_27001/predictions.json",
    }
    assert predictions == known, (
        "the set of tracked-AND-ignored prediction files changed.\n"
        f"  appeared: {sorted(predictions - known)}\n"
        f"  gone:     {sorted(known - predictions)}\n"
        "A new entry means a fold wrote a restricted framework's predictions to "
        "a path already in the index, where the ignore rule cannot stop it. "
        "Untrack it, or add it here with a reason."
    )


def test_merged_corpus_carries_no_unpublishable_prose() -> None:
    """A tracked all_controls.json must carry no overlay source's prose.

    The merged corpus is a build artifact that concatenates every per-framework
    file. Gitignoring the ISO file alone does not help if the merge output is
    tracked and contains the same text.

    Widened from RESTRICTED_FRAMEWORK_IDS to OVERLAY_FRAMEWORK_IDS. The narrow
    form was the same defect one tier down: the seven conditional frameworks
    carry GPL-3.0 and CC BY-SA text, their per-framework files are gitignored,
    and the merge inlined them into this tracked artifact with nothing
    watching. It passed only because none of them carries prose today.

    Reachable in both directions rather than vacuous. On current data 341
    tracked controls across the seven pass at zero offenders; planting one
    prose description in the tracked file turns it red, and
    tests/test_merge_licensed_overlay.py exercises that construction against
    the merge itself.
    """
    merged = PROCESSED_DIR / "all_controls.json"
    if merged.name not in {Path(p).name for p in _tracked_files()}:
        pytest.skip("all_controls.json is not tracked; nothing to enforce")
    if not merged.exists():
        pytest.skip("all_controls.json not present in this checkout")

    data = json.loads(merged.read_text(encoding="utf-8"))
    offenders: list[tuple[str, str, str]] = []
    checked = 0
    for framework in data.get("frameworks", []):
        if framework.get("framework_id") not in OVERLAY_FRAMEWORK_IDS:
            continue
        for control in framework.get("controls", []):
            checked += 1
            description = (control.get("description") or "").strip()
            title = (control.get("title") or "").strip()
            control_id = str(control.get("control_id", "?"))
            # Prose, not a restated title. Both tests must hold: the stub form
            # copies the title verbatim and is short.
            if description != title and len(description) > _TITLE_LENGTH_CEILING:
                offenders.append(
                    (str(framework["framework_id"]), control_id,
                     f"description {len(description)} chars")
                )
            # The second channel. sanitize_control moves anything over
            # DESCRIPTION_MAX_LENGTH into full_text, so a check on description
            # alone would miss the longest statements in the corpus.
            if control.get("full_text"):
                offenders.append(
                    (str(framework["framework_id"]), control_id, "full_text set")
                )

    assert checked > 0, (
        "no overlay framework's controls were inspected, so this gate proved "
        "nothing. Either the tracked corpus lost every conditional framework, "
        "which is a change that needs its own record, or the read is broken."
    )
    assert not offenders, (
        f"{len(offenders)} control statements from frameworks whose licence a "
        f"CC0 grant cannot carry are inside a tracked all_controls.json, e.g. "
        f"{offenders[:3]}. Re-run parsers/merge_all_controls.py, which reduces "
        f"these frameworks to identifiers and titles in the tracked artifact "
        f"and keeps the full text in the gitignored overlay."
    )
