"""NOTICE must state what this repository does to framework text, and to whom.

Two obligations and one open question.

GPL-3.0 section 5(a) and CC BY-SA 4.0 section 3(a)(1)(B) both require a
modified work to carry prominent notice that it was modified. TRACT sanitises,
normalises, truncates, strips stop words from and sometimes elides every
framework statement it stores, and until this landed nothing anywhere said so.
The tests below tie NOTICE's account of those transforms to the constants that
implement them, so changing a bound without changing the notice fails.

The open question is CSA. NOTICE records that the owner ruled CCM
redistributable and does not record on what basis, and AICM sits tracked in git
under the same publisher notice with no tier membership at all. Both facts are
stated in NOTICE and neither is resolved here. These tests hold them stated,
and go red when the underlying fact changes so that NOTICE has to be updated
rather than quietly becoming wrong.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tract.config import (
    CONTROL_ELISION_MARKER,
    DESCRIPTION_MAX_LENGTH,
    MAX_ANCHOR_CHARS,
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_DIR,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
NOTICE_PATH: Path = REPO_ROOT / "NOTICE"

_MODIFICATIONS_HEADING = "Modifications to framework text"
_OPEN_QUESTIONS_HEADING = "Open questions"


def _notice() -> str:
    return NOTICE_PATH.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """The body of one NOTICE section, from its heading to the next one.

    Raises:
        ValueError: the heading is absent. A section-scoped assertion that
            silently matched the whole file would pass on a NOTICE that had
            lost the section entirely, which is the failure it exists to catch.
    """
    body = _notice()
    if heading not in body:
        raise ValueError(
            f"NOTICE has no {heading!r} section. It is required: see the "
            f"module docstring of {Path(__file__).name} for which clause."
        )
    after = body.split(heading, 1)[1]
    # Sections are separated by a rule of hyphens under the next heading.
    parts = after.split("\n\n\n")
    return parts[0] if parts else after


class TestNoticeStatesTheModifications:
    """Each transform is named, and each bound is tied to its constant.

    A prose-only check would pass forever once written. Reading the numbers out
    of tract.config and asserting NOTICE quotes them means raising
    DESCRIPTION_MAX_LENGTH to 2500 fails this file until NOTICE says 2500.
    """

    def test_the_section_exists(self) -> None:
        assert _MODIFICATIONS_HEADING in _notice()

    def test_every_storage_transform_is_named(self) -> None:
        section = _section(_MODIFICATIONS_HEADING)
        # One entry per step of tract.sanitize.sanitize_text, in its own terms.
        required = [
            "Null bytes",
            "NFC",
            "Zero-width",
            "HTML",
            "ligatures",
            "Hyphenation",
            "whitespace",
            "Truncated",
        ]
        missing = [term for term in required if term not in section]
        assert not missing, (
            f"NOTICE's modification statement does not mention {missing}. "
            f"Every step of tract/sanitize.py::sanitize_text alters the "
            f"publisher's text and has to be stated."
        )

    def test_the_implementing_modules_are_named(self) -> None:
        section = _section(_MODIFICATIONS_HEADING)
        for module in (
            "tract/sanitize.py",
            "tract/text_selection.py",
            "tract/stopwords.py",
            "parsers/parse_iso_27001.py",
        ):
            assert module in section, (
                f"NOTICE names a transform without naming {module}, which "
                f"implements it. A reader cannot verify an unlocated claim."
            )

    def test_the_description_bound_matches_the_constant(self) -> None:
        section = _section(_MODIFICATIONS_HEADING)
        assert "DESCRIPTION_MAX_LENGTH" in section
        assert str(DESCRIPTION_MAX_LENGTH) in section, (
            f"NOTICE does not quote DESCRIPTION_MAX_LENGTH "
            f"({DESCRIPTION_MAX_LENGTH}). The stated truncation bound and the "
            f"one the code applies have diverged."
        )

    def test_the_anchor_bound_matches_the_constant(self) -> None:
        section = _section(_MODIFICATIONS_HEADING)
        assert "MAX_ANCHOR_CHARS" in section
        assert str(MAX_ANCHOR_CHARS) in section, (
            f"NOTICE does not quote MAX_ANCHOR_CHARS ({MAX_ANCHOR_CHARS}). "
            f"The training path truncates to a bound NOTICE does not state."
        )

    def test_the_elision_marker_matches_the_constant(self) -> None:
        section = _section(_MODIFICATIONS_HEADING)
        assert "CONTROL_ELISION_MARKER" in section
        assert CONTROL_ELISION_MARKER in section, (
            f"NOTICE does not show the elision marker "
            f"{CONTROL_ELISION_MARKER!r} it says is inserted."
        )

    def test_the_stopword_count_matches_the_committed_list(self) -> None:
        """The list is tracked, so its size is a checkable fact, not a claim."""
        stopwords_path = PROCESSED_DIR / "stopwords.json"
        data = json.loads(stopwords_path.read_text(encoding="utf-8"))
        count = len(data["stopwords"])
        section = _section(_MODIFICATIONS_HEADING)
        assert f"{count}-word" in section, (
            f"NOTICE does not state the stop word list is {count} words. "
            f"{stopwords_path} holds {count}; regenerating the list to a "
            f"different size makes the notice wrong."
        )


class TestNoticeStatesTheCsaExposure:
    """Two CSA facts, stated and unresolved. Held stated by these tests.

    The set is deliberately explicit rather than derived. Deriving "reserves
    redistribution" from the licence string means a substring heuristic over
    publisher prose, which fails silently in the permissive direction: the
    reader cannot tell "no match" from "no such source".

    CORRECTED 2026-08-26. This docstring used to justify that by saying a
    substring heuristic "is the exact structural defect that left csa_aicm in
    no tier", naming `_copyleft` in tests/test_framework_licenses.py. Measured,
    that was wrong, and the claim had already propagated into NOTICE before
    anyone checked it. `_copyleft` derives no tier -- tiers are hand-curated
    frozensets in tract/config.py, and `_copyleft` gates three share-alike
    obligations. `etsi` and `iso_27001` reserve rights outright, fail that same
    substring test, and are tiered anyway. csa_aicm was untiered because no
    owner ruling had been made about it.

    The preference for an explicit set stands on its own merits. The false
    supporting anecdote is gone.
    """

    # Tracked in git, publisher reserves redistribution, member of no tier.
    UNRESOLVED_EXPOSURE: frozenset[str] = frozenset({"csa_aicm"})

    def test_notice_names_every_unresolved_exposure(self) -> None:
        section = _section(_OPEN_QUESTIONS_HEADING)
        missing = sorted(
            framework_id for framework_id in self.UNRESOLVED_EXPOSURE
            if framework_id not in section
        )
        assert not missing, (
            f"{missing} are recorded as unresolved licensing exposure and are "
            f"not named under NOTICE's '{_OPEN_QUESTIONS_HEADING}'. A reader "
            f"of this repository cannot see the exposure."
        )

    def test_every_stated_exposure_is_still_untiered(self) -> None:
        """Fires the moment the owner resolves one, so NOTICE gets updated.

        This is the direction that makes the test worth having. When csa_aicm
        joins a licence tier, the paragraph calling it unresolved becomes
        false, and a stale open question is worse than none because it tells a
        downstream reader to worry about something already handled.
        """
        resolved = sorted(self.UNRESOLVED_EXPOSURE & OVERLAY_FRAMEWORK_IDS)
        assert not resolved, (
            f"{resolved} now route to the overlay, so NOTICE's open-questions "
            f"section is stale. Rewrite the paragraph to record the ruling, "
            f"then remove the id from UNRESOLVED_EXPOSURE."
        )

    def test_every_stated_exposure_is_still_tracked_in_git(self) -> None:
        """The exposure is that the prose is IN git. Assert the premise."""
        tracked = set(
            subprocess.run(
                ["git", "ls-files", "data/processed/frameworks/"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            ).stdout.split()
        )
        gone = sorted(
            framework_id for framework_id in self.UNRESOLVED_EXPOSURE
            if f"data/processed/frameworks/{framework_id}.json" not in tracked
        )
        assert not gone, (
            f"{gone} are recorded as tracked licensing exposure and have no "
            f"tracked artifact. If the text left git, say so in NOTICE and "
            f"remove the id from UNRESOLVED_EXPOSURE."
        )

class TestNoticePointsAtTheLicenceTexts:
    def test_notice_names_the_licenses_directory(self) -> None:
        """NOTICE is the licence_link both published cards point at.

        A reader who follows that link must find the texts from there. The
        published card's link resolves to this file and nothing else.
        """
        assert "LICENSES/" in _notice()
