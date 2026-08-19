"""Every framework and every fetched source must carry a recorded licence.

This repository is CC0, an affirmative grant that the publisher holds the
rights and waives them. Framework content is not the publisher's to waive, and
until this landed the repository said nothing at all about the difference: 30
tracked framework artifacts, four of them under CC BY-SA and two under notices
reserving reproduction, all sitting under a file that grants everything away.

The defect was not any single framework. It was that nothing recorded the
question, so each new ingest inherited the silence. These tests make the
question mandatory: a framework file with no entry in FRAMEWORK_LICENSES fails,
a Source with no licence fails, and an entry that never reaches NOTICE fails.

UNDETERMINED is an acceptable answer here and a guess is not. A source whose
staged artifact states no terms is recorded as UNDETERMINED and listed in
NOTICE under open questions.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

from scripts.fetch_frameworks import MANIFEST_PATH, SOURCES
from tract.config import (
    CONDITIONAL_FRAMEWORK_IDS,
    FRAMEWORK_LICENSES,
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_FRAMEWORKS_DIR,
    RESTRICTED_FRAMEWORK_IDS,
    UNDETERMINED_LICENSE,
)
from tract.licensing import (
    LICENSE_TEXTS_DIR,
    shipped_license_text_ids,
    spdx_identifiers,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
NOTICE_PATH: Path = REPO_ROOT / "NOTICE"

# NOTICE's modified-work notice. GPL-3.0 section 5(a) and CC BY-SA 4.0 section
# 3(a)(1)(B) both require one, and this heading is where it lives.
MODIFICATION_STATEMENT_HEADING: str = "Modifications to framework text"

# "| framework_id | licence | upstream source |", data rows only.
_NOTICE_ROW: re.Pattern[str] = re.compile(
    r"^\|\s*(?P<framework_id>[a-z][a-z0-9_]*)\s*\|"
    r"\s*(?P<license>[^|]+?)\s*\|"
    r"\s*(?P<source>\S+)\s*\|$"
)


def _framework_ids_in_processed() -> set[str]:
    """Framework ids with an artifact, on disk or tracked by git.

    Both, because they differ on purpose. A restricted framework's JSON exists
    locally and is gitignored, so a git-only view would miss it. A checkout
    that has never run the parsers has the tracked files and not the untracked
    ones, so a disk-only view would miss those in CI.
    """
    on_disk = {
        path.stem for path in PROCESSED_FRAMEWORKS_DIR.glob("*.json")
    } if PROCESSED_FRAMEWORKS_DIR.exists() else set()
    tracked = subprocess.run(
        ["git", "ls-files", "data/processed/frameworks/*.json"],
        capture_output=True, text=True, check=True, cwd=REPO_ROOT,
    ).stdout.split()
    return on_disk | {Path(name).stem for name in tracked}


def _tracked_framework_files() -> set[str]:
    """Paths git tracks under data/processed/frameworks/, right now."""
    return set(
        subprocess.run(
            ["git", "ls-files", "data/processed/frameworks/"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
    )


def _copyleft() -> set[str]:
    """Frameworks whose recorded licence a CC0 grant cannot carry.

    GPL and CC BY-SA both require the copy to keep the licence. CC0 asserts
    the publisher holds the rights and waives them, which is the opposite
    claim, so a copyleft source cannot sit in a CC0 tree.
    """
    return {
        framework_id
        for framework_id, licence in FRAMEWORK_LICENSES.items()
        if "GPL" in licence or "CC-BY-SA" in licence
    }


def _notice_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in NOTICE_PATH.read_text(encoding="utf-8").splitlines():
        match = _NOTICE_ROW.match(line.strip())
        if match:
            rows[match["framework_id"]] = match["license"]
    return rows


def _modification_statement() -> str | None:
    """NOTICE's modified-work notice, or None when the section is gone.

    Scoped to the section rather than to the whole file on purpose. A read that
    fell back to the whole file would find every phrase it looked for in some
    other paragraph and report green on a NOTICE that had lost the notice.
    """
    body = NOTICE_PATH.read_text(encoding="utf-8")
    if MODIFICATION_STATEMENT_HEADING not in body:
        return None
    # NOTICE separates sections with two blank lines.
    return body.split(MODIFICATION_STATEMENT_HEADING, 1)[1].split("\n\n\n")[0]


def _framework_ids_without_a_notice_row(framework_ids: Iterable[str]) -> list[str]:
    """Which of *framework_ids* NOTICE's table does not name.

    Takes its input rather than reading `_copyleft()` so the same logic can be
    run against a constructed framework that does not exist in the registry.
    """
    rows = _notice_rows()
    return sorted(set(framework_ids) - set(rows))


def _framework_ids_without_a_shipped_licence_text(
    framework_ids: Iterable[str],
) -> list[str]:
    """Which of *framework_ids* declare an SPDX licence this tree does not ship.

    A framework whose recorded licence yields no SPDX identifier at all is
    reported too. Naming a copyleft licence in prose and shipping nothing is
    the same failure as naming an identifier and shipping nothing, and the
    substring derivation in `_copyleft` can reach a prose string.
    """
    shipped = shipped_license_text_ids()
    offenders: list[str] = []
    for framework_id in sorted(set(framework_ids)):
        identifiers = spdx_identifiers(FRAMEWORK_LICENSES.get(framework_id, ""))
        if not identifiers:
            offenders.append(
                f"{framework_id}: recorded licence yields no SPDX identifier, "
                f"so no text can be shipped for it"
            )
            continue
        absent = [name for name in identifiers if name not in shipped]
        if absent:
            offenders.append(f"{framework_id}: {absent} have no shipped text")
    return offenders


class TestEveryFrameworkHasALicence:
    def test_every_processed_framework_is_in_the_registry(self) -> None:
        missing = sorted(_framework_ids_in_processed() - set(FRAMEWORK_LICENSES))
        assert not missing, (
            f"{missing} have artifacts under data/processed/frameworks/ and no "
            f"entry in tract.config.FRAMEWORK_LICENSES. Read the licence off "
            f"that framework's staged source and record it, or record "
            f"'{UNDETERMINED_LICENSE}' if the artifact states no terms."
        )

    def test_no_registry_entry_is_empty(self) -> None:
        blank = sorted(k for k, v in FRAMEWORK_LICENSES.items() if not v.strip())
        assert not blank, (
            f"{blank} carry an empty licence. An empty string reads as 'not "
            f"applicable'; '{UNDETERMINED_LICENSE}' says what is actually true."
        )

    def test_the_registry_names_no_framework_that_does_not_exist(self) -> None:
        """A stale entry is a claim about a framework nobody ships.

        Overlay frameworks are exempt by name. Their artifacts are untracked
        by design and only exist after a local parser run, so on a fresh CI
        checkout they have no file on disk and no file in git. Without the
        exemption this test reports all nine as stale registry entries and CI
        is red on every run. The test still catches a registry name matching
        no framework at all, which is what it is for.
        """
        extra = sorted(
            set(FRAMEWORK_LICENSES)
            - _framework_ids_in_processed()
            - OVERLAY_FRAMEWORK_IDS
        )
        assert not extra, (
            f"{extra} are in FRAMEWORK_LICENSES with no artifact under "
            f"data/processed/frameworks/. Remove the entry or restore the file."
        )

    def test_every_restricted_framework_has_a_licence_on_record(self) -> None:
        for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
            assert FRAMEWORK_LICENSES.get(framework_id, "").strip(), (
                f"{framework_id} is restricted and has no recorded licence. "
                f"The reason it is restricted is the licence."
            )
            assert FRAMEWORK_LICENSES[framework_id] != UNDETERMINED_LICENSE, (
                f"{framework_id} is restricted on the strength of a licence "
                f"recorded as {UNDETERMINED_LICENSE}. Restricting a source "
                f"whose terms nobody read is a guess in the other direction."
            )


class TestLicenceTiering:
    """Two tiers, because a binary set modelled the wrong property.

    RESTRICTED means "must never appear in git in any form" and drives the
    fingerprint gate. CONDITIONAL means "reproduction is permitted on terms a
    CC0 grant cannot carry", which is a different fact with the same routing
    consequence. The old single set treated every framework whose licence
    permits reproduction on conditions as unconditionally publishable.

    CONDITIONAL held seven members and holds two. What the five that left have
    in common, and why dsomm and csa_ccm are different questions, is recorded
    against the constant in tract/config.py rather than restated here.
    """

    def test_the_two_tiers_do_not_overlap(self) -> None:
        """One framework cannot be both unconditionally barred and conditional.

        There is no assertion here that OVERLAY equals the union of the two.
        tract/config.py defines OVERLAY as that union, so such an assertion
        restates a definition and its attainable range is {True}.
        """
        assert not (RESTRICTED_FRAMEWORK_IDS & CONDITIONAL_FRAMEWORK_IDS)

    def test_no_framework_reaches_the_overlay_on_an_unread_licence(self) -> None:
        """Routing is a claim about terms, so the terms must have been read.

        UNDETERMINED means the staged artifact stated nothing. Withholding a
        framework's text on a licence nobody read is a guess, in the same way
        that publishing it would be.
        """
        unread = sorted(
            framework_id for framework_id in OVERLAY_FRAMEWORK_IDS
            if FRAMEWORK_LICENSES.get(framework_id, UNDETERMINED_LICENSE)
            in ("", UNDETERMINED_LICENSE)
        )
        assert not unread, (
            f"{unread} route to the overlay with a licence recorded as "
            f"{UNDETERMINED_LICENSE}. Read the terms off the staged source "
            f"and record them, or take the framework out of the tier."
        )

    def test_every_overlay_framework_has_a_gitignore_line(self) -> None:
        ignored = {
            line.strip()
            for line in (
                REPO_ROOT / ".gitignore"
            ).read_text(encoding="utf-8").splitlines()
        }
        for framework_id in sorted(OVERLAY_FRAMEWORK_IDS):
            expected = f"data/processed/frameworks/{framework_id}.json"
            assert expected in ignored, (
                f"{framework_id} routes to the overlay but {expected} is not in "
                f".gitignore, so its text would be tracked."
            )

    def test_no_overlay_framework_is_still_tracked(self) -> None:
        """A .gitignore line does nothing to a file git already tracks.

        All seven conditional files were tracked when the tier landed, so the
        seven new .gitignore lines were inert and the routing was decorative:
        `git check-ignore` reported them unignored, and the next parser run
        would have committed GPL-3.0 and CC BY-SA prose over the top of a stub.
        Asserting the .gitignore line and asserting the file is untracked are
        two different claims, and only the second one is the guarantee.
        """
        tracked = _tracked_framework_files()
        offenders = sorted(
            framework_id for framework_id in OVERLAY_FRAMEWORK_IDS
            if f"data/processed/frameworks/{framework_id}.json" in tracked
        )
        assert not offenders, (
            f"{offenders} route to the overlay and are still tracked. A "
            f".gitignore line is ignored for an already-tracked path. Run "
            f"`git rm --cached` on each before any parser writes prose into it."
        )


class TestCopyleftObligationsAreDischarged:
    """What a CC0 tree owes a copyleft source it carries, checked per source.

    This is the gate that used to read "every copyleft framework is
    CONDITIONAL". That demand was right while LICENSES/ did not exist and while
    the tier was the only thing standing between GPL-3.0 text and a CC0 grant.
    It stopped being right on the commit that tracked biml, samm, wstg,
    owasp_top10_2021 and owasp_proactive_controls: twelve of the thirteen
    copyleft frameworks in FRAMEWORK_LICENSES are now tracked deliberately, so
    the old form would need an exception list of twelve, and a gate that
    exempts twelve of thirteen names is not a gate.

    Deleting it was the other option and the worse one. It is the only
    automated check that a copyleft source ingested tomorrow does not join the
    tracked corpus in silence, and that silence is what this file's opening
    docstring was written to end. So the demand changes and the gate stays.

    CC BY-SA 4.0 section 3(a) and GPL-3.0 sections 4 and 5(a) ask a
    redistributor for the same three things, and all three are checkable here:

      1. the source attributed and its licence identified
         -> a row in NOTICE
      2. the licence delivered rather than merely named
         -> LICENSES/<id>.txt for every SPDX identifier the registry records
      3. prominent notice that the work was modified
         -> NOTICE's modification statement

    Condition 3 is corpus-wide where 1 and 2 are per framework, because the
    transforms are corpus-wide: every control statement that reaches
    data/processed/ goes through the same tract/sanitize.py path, so one notice
    covers thirteen frameworks. Losing it loses the notice for all of them at
    once, which is what its failure message says.
    """

    def test_the_gate_has_copyleft_frameworks_to_inspect(self) -> None:
        """Non-vacuity. Every assertion below loops over this set.

        A derivation that stopped matching anything would make the three
        conditions pass over an empty corpus, which is the shape of failure a
        conjunctive gate is most likely to take.
        """
        copyleft = _copyleft()
        assert copyleft, (
            "no framework in FRAMEWORK_LICENSES derives as copyleft, so every "
            "obligation checked below was checked against nothing. Either the "
            "corpus lost its GPL and CC BY-SA sources, which is a change with "
            "its own record, or _copyleft stopped matching."
        )

    def test_every_copyleft_framework_is_named_in_notice(self) -> None:
        """Condition 1. Attribution and the licence, where a reader looks."""
        missing = _framework_ids_without_a_notice_row(_copyleft())
        assert not missing, (
            f"{missing} carry a copyleft or share-alike licence and have no "
            f"row in NOTICE, so this repository redistributes their text with "
            f"no attribution and no statement of terms. CC BY-SA 4.0 section "
            f"3(a)(1) and GPL-3.0 section 4 both require the notice to travel "
            f"with the copy. Add the row, or take the framework out of the "
            f"corpus."
        )

    def test_every_copyleft_framework_has_its_licence_text_shipped(self) -> None:
        """Condition 2. Naming an identifier is not delivering the licence.

        GPL-3.0 section 4 wants the recipient to get "a copy of this License
        along with the Program", and CC BY-SA 4.0 section 3(a)(1)(A) wants the
        licence or a URI retained. A row in NOTICE names the terms; only
        LICENSES/ hands them over.
        """
        missing = _framework_ids_without_a_shipped_licence_text(_copyleft())
        assert not missing, (
            f"{missing}. Fetch the publisher's own plain-text licence and "
            f"commit it as {LICENSE_TEXTS_DIR.name}/<identifier>.txt. Do not "
            f"paraphrase one, and do not record an SPDX identifier for a "
            f"source whose notice grants nothing."
        )

    def test_the_modification_statement_covers_every_copyleft_framework(
        self,
    ) -> None:
        """Condition 3. Prominent notice that the text was modified.

        TRACT sanitises, normalises, truncates, strips stop words from and
        sometimes elides every statement it stores, so nothing it redistributes
        is the published wording. GPL-3.0 section 5(a) and CC BY-SA 4.0 section
        3(a)(1)(B) both require that to be stated.

        The two claims asserted here are the ones the section's scope rests on:
        the storage path it covers, and the module that implements the
        transforms. A section that named neither would be a notice a reader
        cannot check against the tree.
        """
        copyleft = sorted(_copyleft())
        statement = _modification_statement()
        assert statement is not None, (
            f"NOTICE has no {MODIFICATION_STATEMENT_HEADING!r} section, so all "
            f"{len(copyleft)} copyleft frameworks in this corpus ({copyleft}) "
            f"are redistributed as modified works carrying no notice that they "
            f"were modified."
        )
        unstated = [
            claim for claim in ("data/processed/", "tract/sanitize.py")
            if claim not in statement
        ]
        assert not unstated, (
            f"NOTICE's modification statement no longer states {unstated}, so "
            f"its scope no longer reaches the stored text of {copyleft}. The "
            f"statement has to say which text it covers and what alters it."
        )

    def test_a_copyleft_framework_with_no_notice_row_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Condition 1's failing case, constructed rather than argued.

        A share-alike source ingested tomorrow, recorded in the registry and
        nowhere else. The three tests above read module state, so this one
        injects the framework and re-runs the same helper against it.
        """
        new_id = "fictional_share_alike_source"
        assert new_id not in FRAMEWORK_LICENSES
        # The delta, not the absolute list. Asserting equality with [new_id]
        # would also fail when some unrelated framework lost its row, which
        # reports this constructed case as broken instead of reporting the real
        # regression where it belongs.
        before = set(_framework_ids_without_a_notice_row(_copyleft()))
        monkeypatch.setitem(FRAMEWORK_LICENSES, new_id, "CC-BY-SA-4.0")

        assert new_id in _copyleft(), (
            "the derivation stopped classifying CC-BY-SA-4.0 as copyleft, so "
            "no share-alike source would ever reach the conditions above"
        )
        after = set(_framework_ids_without_a_notice_row(_copyleft()))
        assert after - before == {new_id}, (
            f"injecting a copyleft framework with no NOTICE row did not make "
            f"condition 1 reject it: {sorted(after - before)}"
        )

    def test_a_copyleft_framework_with_no_shipped_text_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Condition 2's failing case, on a licence whose text is not in the tree.

        CC BY-SA 2.5 is share-alike, so the derivation catches it, and no
        LICENSES/CC-BY-SA-2.5.txt exists, so the delivery obligation is unmet.
        Checked separately from condition 1 because the two reject different
        omissions and a single constructed case would not tell them apart.
        """
        new_id = "fictional_unshipped_licence_source"
        assert new_id not in FRAMEWORK_LICENSES
        assert "CC-BY-SA-2.5" not in shipped_license_text_ids()
        before = set(_framework_ids_without_a_shipped_licence_text(_copyleft()))
        monkeypatch.setitem(FRAMEWORK_LICENSES, new_id, "CC-BY-SA-2.5")

        assert new_id in _copyleft()
        after = set(_framework_ids_without_a_shipped_licence_text(_copyleft()))
        added = sorted(after - before)
        assert len(added) == 1 and added[0].startswith(f"{new_id}:"), (
            f"injecting a copyleft framework under an unshipped licence did "
            f"not make condition 2 reject it: {added}"
        )


class TestEverySourceHasALicence:
    def test_no_source_licence_is_empty(self) -> None:
        blank = [
            f"{s.framework_id}/{s.filename}"
            for s in SOURCES if not s.license.strip()
        ]
        assert not blank, f"{blank} carry an empty Source.license"

    def test_a_single_document_framework_agrees_with_the_registry(self) -> None:
        """One document, one licence. Two copies of it must not drift.

        Frameworks with more than one source document are exempt: BIML's 2020
        report is CC BY-SA 3.0 and its 2024 report is 4.0, so no single
        framework-level string is right for both, and the registry records the
        union.
        """
        counts: dict[str, int] = {}
        for source in SOURCES:
            counts[source.framework_id] = counts.get(source.framework_id, 0) + 1
        for source in SOURCES:
            if counts[source.framework_id] != 1:
                continue
            assert source.license == FRAMEWORK_LICENSES[source.framework_id], (
                f"{source.framework_id}: Source.license "
                f"{source.license!r} disagrees with FRAMEWORK_LICENSES "
                f"{FRAMEWORK_LICENSES[source.framework_id]!r}"
            )

    def test_the_manifest_records_a_licence_for_every_entry(self) -> None:
        """The committed manifest is what a reader of data/processed/ sees."""
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sources"]
        blank = [
            f"{framework_id}/{filename}"
            for framework_id, files in manifest.items()
            for filename, record in files.items()
            if not str(record.get("license", "")).strip()
        ]
        assert not blank, (
            f"{blank} have no license in {MANIFEST_PATH.name}. Re-run "
            f"`python -m scripts.fetch_frameworks --all` to re-record them."
        )


class TestNoticeStaysInStepWithTheRegistry:
    def test_notice_lists_exactly_the_registry(self) -> None:
        rows = _notice_rows()
        assert set(rows) == set(FRAMEWORK_LICENSES), (
            f"NOTICE and FRAMEWORK_LICENSES disagree: "
            f"only in NOTICE {sorted(set(rows) - set(FRAMEWORK_LICENSES))}, "
            f"only in the registry {sorted(set(FRAMEWORK_LICENSES) - set(rows))}"
        )

    def test_notice_quotes_the_registry_verbatim(self) -> None:
        rows = _notice_rows()
        wrong = {
            framework_id: (licence, FRAMEWORK_LICENSES[framework_id])
            for framework_id, licence in rows.items()
            if licence != FRAMEWORK_LICENSES[framework_id]
        }
        assert not wrong, (
            f"NOTICE states a different licence from FRAMEWORK_LICENSES for "
            f"{sorted(wrong)}. NOTICE is what a downstream reader acts on, so "
            f"the two cannot diverge: {list(wrong.items())[:2]}"
        )

    def test_notice_names_every_restricted_source(self) -> None:
        body = NOTICE_PATH.read_text(encoding="utf-8")
        section = body.split("Frameworks\n----------")[0]
        for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
            assert framework_id in section, (
                f"NOTICE does not name {framework_id} as a restricted source, "
                f"so a reader cannot tell why its text is absent."
            )
