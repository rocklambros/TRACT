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
from pathlib import Path

from scripts.fetch_frameworks import MANIFEST_PATH, SOURCES
from tract.config import (
    CONDITIONAL_FRAMEWORK_IDS,
    FRAMEWORK_LICENSES,
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_FRAMEWORKS_DIR,
    RESTRICTED_FRAMEWORK_IDS,
    UNDETERMINED_LICENSE,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
NOTICE_PATH: Path = REPO_ROOT / "NOTICE"

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
    consequence. The old single set treated seven frameworks whose licences
    permit reproduction on conditions as unconditionally publishable.
    """

    # Copyleft frameworks whose processed files were already tracked when the
    # tiers landed. Moving them to the overlay would pull 691 curated links
    # out of the tracked corpus (asvs 277, owasp_cheat_sheets 391,
    # owasp_llm_top10 13, owasp_ml_top10 10) and shift every published metric,
    # so it is an owner decision with its own change record rather than a side
    # effect of adding the tiers.
    #
    # This is a ratchet, not an allowlist, and the difference is the whole
    # point. Three assertions hold it shut: the set is asserted by equality
    # below, so widening it means editing a line whose only purpose is to say
    # it must not widen. Every member must have a tracked artifact in git
    # today, which is what "pre-existing" actually means and which a new
    # framework cannot satisfy. No member may carry a non-copyleft
    # licence. A newly added copyleft framework fails
    # test_every_copyleft_framework_is_conditional and cannot be silenced by
    # appending a name here.
    PRE_EXISTING_EXPOSURE: frozenset[str] = frozenset({
        "asvs",
        "owasp_agentic_top10",
        "owasp_cheat_sheets",
        "owasp_dsgai",
        "owasp_llm_top10",
        "owasp_llm_top10_2026",
        "owasp_ml_top10",
    })

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

    def test_every_copyleft_framework_is_conditional(self) -> None:
        """The tier is derived from the recorded licence, not from a hand list.

        The binary set this replaces modelled licence STATUS and not licence
        CLASS, so seven frameworks whose licences permit reproduction on
        conditions were treated as unconditionally publishable. Deriving the
        assertion from FRAMEWORK_LICENSES means a newly added copyleft source
        fails this test rather than silently joining the tracked corpus.
        """
        missing = _copyleft() - OVERLAY_FRAMEWORK_IDS - self.PRE_EXISTING_EXPOSURE
        assert not missing, (
            f"{sorted(missing)} carry a copyleft or share-alike licence and are "
            f"not routed to the overlay. A CC0 repository cannot carry their "
            f"terms. Add them to CONDITIONAL_FRAMEWORK_IDS, or take the "
            f"exposure to the owner for a recorded ruling. Do not append them "
            f"to PRE_EXISTING_EXPOSURE: that set is closed by equality below "
            f"and by the tracked-artifact check, and a framework added after "
            f"the tiers landed is not pre-existing by definition."
        )

    def test_the_recorded_exposure_is_closed(self) -> None:
        """The ratchet. Widening the set means editing this line.

        Without an equality assertion the name PRE_EXISTING_EXPOSURE asserts a
        temporal property that nothing enforces, and the remedy printed by the
        test above becomes "append the id here", which retires the gate one
        name at a time.
        """
        assert self.PRE_EXISTING_EXPOSURE == frozenset({
            "asvs",
            "owasp_agentic_top10",
            "owasp_cheat_sheets",
            "owasp_dsgai",
            "owasp_llm_top10",
            "owasp_llm_top10_2026",
            "owasp_ml_top10",
        }), (
            "PRE_EXISTING_EXPOSURE changed. It records the copyleft frameworks "
            "already tracked when the licence tiers landed, which is a closed "
            "historical fact and not a list to extend. A framework that needs "
            "an exception now goes to the owner for a recorded ruling."
        )

    def test_every_recorded_exposure_is_tracked_in_git_today(self) -> None:
        """What "pre-existing" actually means, asserted rather than asserted by name.

        A framework added after the tiers landed has no tracked artifact,
        because CONDITIONAL_FRAMEWORK_IDS routes new copyleft sources to the
        gitignored overlay. So it cannot join this set without a second,
        visible change that untracks nothing and tracks licensed prose.
        """
        tracked = _tracked_framework_files()
        untracked = sorted(
            framework_id for framework_id in self.PRE_EXISTING_EXPOSURE
            if f"data/processed/frameworks/{framework_id}.json" not in tracked
        )
        assert not untracked, (
            f"{untracked} are recorded as pre-existing copyleft exposure and "
            f"have no tracked artifact under data/processed/frameworks/. The "
            f"set records what was already in git when the tiers landed. If "
            f"the exposure was resolved, remove the id from the set."
        )

    def test_the_recorded_exposure_names_only_copyleft_frameworks(self) -> None:
        """A stale name in the exposure list would hide a real routing gap."""
        stale = sorted(self.PRE_EXISTING_EXPOSURE - _copyleft())
        assert not stale, (
            f"{stale} are recorded as pre-existing copyleft exposure and carry "
            f"no copyleft licence. Remove them. An inflated exposure list can "
            f"absorb a genuine routing gap."
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
