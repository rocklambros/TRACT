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
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tract.config import (
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_DIR,
    RESTRICTED_FRAMEWORK_IDS,
)
from tract.licensing import (
    FINGERPRINT_DOCUMENT_KEYS,
    FINGERPRINT_PATH,
    FINGERPRINT_TOP_LEVEL_KEYS,
    LicensedFingerprints,
    fingerprint_ngrams,
)

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
_SCANNED_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".md", ".json", ".jsonl", ".txt", ".csv", ".yml", ".yaml", ".rst",
})

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
            assert document["framework_id"] in RESTRICTED_FRAMEWORK_IDS
            assert _SHA256_RE.match(document["source_sha256"])
            assert isinstance(document["ngram_count"], int)
            # A filename, not a document body. Bounded so a future change
            # cannot start writing an excerpt into this field.
            assert len(document["filename"]) <= 128

    def test_every_restricted_framework_contributed_fingerprints(
        self, fingerprints: LicensedFingerprints
    ) -> None:
        """A restricted source with no fingerprints is an unguarded source."""
        covered = {document.framework_id for document in fingerprints.documents}
        assert covered == set(RESTRICTED_FRAMEWORK_IDS), (
            f"fingerprints cover {sorted(covered)} but "
            f"RESTRICTED_FRAMEWORK_IDS is {sorted(RESTRICTED_FRAMEWORK_IDS)}. "
            f"Run `python -m scripts.build_licensed_fingerprints`."
        )
        for document in fingerprints.documents:
            assert document.ngram_count > 0, (
                f"{document.framework_id} contributed no n-grams, so nothing "
                f"about that source is actually being checked"
            )


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


def test_real_fingerprints_flag_a_real_statement(
    fingerprints: LicensedFingerprints,
) -> None:
    """End-to-end check against the actual restricted source.

    Redundant with the planted-quotation test above, which is why this one may
    skip: it needs the gitignored source, and the gate itself must not. Kept
    because it is the only check that the committed hashes were computed from
    the document they claim.
    """
    source = (
        REPO_ROOT / "data" / "raw" / "frameworks" / "iso_27001"
        / "ISO_IEC_27001_2022_en.md"
    )
    if not source.exists():
        pytest.skip("raw ISO source absent; the planted-quotation test covers this")

    statements = [
        line.strip().strip("|").split("|")[2].strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.startswith("|") and len(line.strip().strip("|").split("|")) == 3
        and re.match(r"^\d+\.\d+$", line.strip().strip("|").split("|")[0].strip())
    ]
    longest = max(statements, key=len)
    assert fingerprints.first_hit(longest) is not None, (
        "the committed fingerprints do not match the source they name. "
        "Re-run `python -m scripts.build_licensed_fingerprints`."
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
        if path.suffix not in _SCANNED_SUFFIXES or path == FINGERPRINT_PATH:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
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


def test_restricted_framework_files_are_not_tracked() -> None:
    """The per-framework JSON for a licensed source must be gitignored."""
    tracked = _tracked_files()
    for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
        path = f"data/processed/frameworks/{framework_id}.json"
        assert path not in tracked, (
            f"{path} is tracked by git. This repository is CC0, so committing "
            f"{framework_id} control statements dedicates licensed text to the "
            f"public domain. Run: git rm --cached {path}"
        )


def test_every_restricted_framework_has_a_gitignore_line() -> None:
    """Untracking a file is not enough; the next parser run re-adds it.

    `git rm --cached` removes the file from the index. Without a .gitignore
    entry the next `git add -A` after a parser run puts it straight back, and
    the test above only catches that once it has already happened.
    """
    ignore_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
        expected = f"data/processed/frameworks/{framework_id}.json"
        assert expected in ignore_lines, (
            f"{expected} is missing from .gitignore, so a parser run followed "
            f"by `git add -A` would commit {framework_id}'s licensed text."
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
