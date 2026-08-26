"""The repository must deliver the licence texts it redistributes content under.

Recording an SPDX identifier in NOTICE names the terms. It does not deliver
them, and two of the licences in this tree require delivery: GPL-3.0 section 4
obliges anyone conveying a covered work to give recipients "a copy of this
License along with the Program", and CC BY-SA 4.0 section 3(a)(1)(A) obliges
retention of the licence or a URI to it. Before LICENSES/ landed the repository
carried DSOMM under GPL-3.0 and eleven sources under CC BY-SA and shipped
neither text, so the position was asserted rather than discharged.

The gate is derived from tract.config.FRAMEWORK_LICENSES rather than from a
second hand-written list. A framework ingested tomorrow under a licence whose
text is not in the tree turns this red on the same commit that adds it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from tract.config import FRAMEWORK_LICENSES
from tract.licensing import (
    LICENSE_TEXT_SUFFIX,
    LICENSE_TEXTS_DIR,
    PROJECT_LICENSE_ID,
    required_license_text_ids,
    shipped_license_text_ids,
    spdx_identifiers,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# One phrase per shipped licence, verified below to appear in that licence's
# file and in no other. A length floor alone would pass on a file holding the
# wrong licence's body, and a filename check alone would pass on an empty file.
#
# The GPL marker is the section 4 clause this directory exists to satisfy, so a
# truncated GPL text fails on the sentence that made the text mandatory.
_DISTINCTIVE_PHRASE: dict[str, str] = {
    "Apache-2.0": "Version 2.0, January 2004",
    "CC-BY-4.0": "Attribution 4.0 International",
    "CC-BY-SA-3.0": "Attribution-ShareAlike 3.0 Unported",
    "CC-BY-SA-4.0": "Attribution-ShareAlike 4.0 International",
    "CC0-1.0": "CC0 1.0 Universal",
    "GPL-3.0-only": "a copy of this License along with the Program",
}

# Where each text was fetched from, and the bytes that arrived.
#
# The phrase check above is diagnosis, not proof. Four of the six markers are
# the licence's own title line, so truncating a body to its first page leaves
# the marker intact and the check green. Measured: cutting CC-BY-4.0.txt to 20
# lines passes every phrase assertion. A licence is an immutable published
# document, so pinning the bytes is both available and the strongest form of
# the claim, and it is what makes truncation, re-wrapping and paraphrase fail.
#
# A legitimate re-fetch changes these digests and turns the pin red, which is
# the intended cost. Updating a licence text is a deliberate act, not a drift.
_PROVENANCE: dict[str, tuple[str, str]] = {
    "Apache-2.0": (
        "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    "CC-BY-4.0": (
        "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
        "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
    ),
    "CC-BY-SA-3.0": (
        "https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt",
        "3f941b3b89cf7b8370ceb83cc76d2120d471b58735d8ca60238a751a48d7f72f",
    ),
    "CC-BY-SA-4.0": (
        "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt",
        "28a9529c7d0bb4dc51f4bf5c116a3d16ef247a052f7591466768ddf563fd1cf5",
    ),
    "CC0-1.0": (
        "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt",
        "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499",
    ),
    "GPL-3.0-only": (
        "https://www.gnu.org/licenses/gpl-3.0.txt",
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    ),
}

# Every framework whose recorded licence yields no SPDX identifier, so no
# licence text exists to ship for it. Eight state no terms at all in the staged
# artifact (UNDETERMINED) and nine carry a prose statement, of which four
# reserve reproduction outright.
#
# Asserted by equality on purpose. The set is the repository's record of which
# sources granted nothing, and it must move only through a deliberate edit:
# resolving an UNDETERMINED to a real SPDX identifier shrinks it, and ingesting
# a new source under a publisher's own prose notice grows it. Both are facts
# NOTICE has to state, and both turn this red until it does.
_NO_SHIPPABLE_TEXT: frozenset[str] = frozenset({
    "aiuc_1",
    "capec",
    "csa_aicm",
    "csa_ccm",
    "cwe",
    "enisa",
    "etsi",
    "eu_ai_act",
    "eu_gpai_cop",
    "iso_27001",
    "nist_800_53",
    "nist_800_63",
    "nist_ai_100_2",
    "nist_ai_600_1",
    "nist_ai_rmf",
    "nist_ssdf",
    "owasp_ai_exchange",
})


def _bodies() -> dict[str, str]:
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in LICENSE_TEXTS_DIR.glob(f"*{LICENSE_TEXT_SUFFIX}")
    }


class TestEveryDeclaredLicenceIsShipped:
    def test_no_declared_spdx_licence_is_missing_its_text(self) -> None:
        missing = sorted(required_license_text_ids() - shipped_license_text_ids())
        assert not missing, (
            f"{missing} appear in FRAMEWORK_LICENSES and have no text under "
            f"{LICENSE_TEXTS_DIR.name}/. Fetch the publisher's own plain-text "
            f"licence and commit it as {LICENSE_TEXTS_DIR.name}/<id>"
            f"{LICENSE_TEXT_SUFFIX}. Do not paraphrase one."
        )

    def test_nothing_is_shipped_that_no_framework_declares(self) -> None:
        """A stray licence text is a claim about terms nothing in the tree uses.

        Scanners read this directory as the repository's licence inventory. An
        unused entry makes the inventory wrong in the permissive direction,
        which is the same class of error as the CC0 over-claim NOTICE exists to
        correct.
        """
        extra = sorted(shipped_license_text_ids() - required_license_text_ids())
        assert not extra, (
            f"{extra} have a text under {LICENSE_TEXTS_DIR.name}/ and are "
            f"declared by no framework in FRAMEWORK_LICENSES and are not "
            f"{PROJECT_LICENSE_ID}, this project's own licence. Remove the "
            f"file, or record the framework that needs it."
        )

    def test_the_project_licence_is_shipped(self) -> None:
        """REUSE and Scancode read the project's own licence from here too."""
        assert PROJECT_LICENSE_ID in shipped_license_text_ids()


class TestEveryShippedTextIsTheRealLicence:
    def test_every_shipped_file_has_a_phrase_registered_for_it(self) -> None:
        """The phrase table cannot go stale behind a newly shipped licence."""
        unchecked = sorted(shipped_license_text_ids() - set(_DISTINCTIVE_PHRASE))
        assert not unchecked, (
            f"{unchecked} are shipped with no distinctive phrase registered in "
            f"_DISTINCTIVE_PHRASE, so nothing checks their bodies are the "
            f"licence the filename claims."
        )

    def test_each_phrase_appears_in_its_own_file_and_no_other(self) -> None:
        """Catches a truncated body, an empty file, and two files swapped.

        Uniqueness is what makes this more than a length floor. Copying the CC
        BY 4.0 text into CC-BY-SA-4.0.txt leaves both markers in one file and
        one marker in none, and both halves of the assertion fire.
        """
        bodies = _bodies()
        wrong: list[str] = []
        for license_id, phrase in sorted(_DISTINCTIVE_PHRASE.items()):
            holders = sorted(k for k, body in bodies.items() if phrase in body)
            if holders != [license_id]:
                wrong.append(f"{license_id}: phrase found in {holders}")
        assert not wrong, (
            f"the shipped licence texts do not match their filenames: {wrong}. "
            f"Re-fetch the publisher's own plain-text licence."
        )

    def test_every_shipped_file_has_recorded_provenance(self) -> None:
        """An unpinned licence text is one nobody can trace to a publisher."""
        unpinned = sorted(shipped_license_text_ids() - set(_PROVENANCE))
        assert not unpinned, (
            f"{unpinned} are shipped with no entry in _PROVENANCE, so neither "
            f"the source URL nor the bytes are on record."
        )

    def test_every_shipped_file_matches_its_pinned_digest(self) -> None:
        """Catches truncation, re-wrapping, and any edit to a licence body."""
        wrong: list[str] = []
        for license_id, (url, expected) in sorted(_PROVENANCE.items()):
            path = LICENSE_TEXTS_DIR / f"{license_id}{LICENSE_TEXT_SUFFIX}"
            if not path.exists():
                wrong.append(f"{license_id}: absent, expected from {url}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                wrong.append(
                    f"{license_id}: sha256 {actual[:12]}... != pinned "
                    f"{expected[:12]}..., re-fetch from {url}"
                )
        assert not wrong, (
            f"shipped licence texts do not match their pinned bytes: {wrong}. "
            f"A licence is an immutable published document, so any difference "
            f"is an edit to text this repository has no right to edit."
        )


class TestSourcesThatGrantNothingShipNoText:
    def test_the_set_with_no_shippable_licence_text_is_unchanged(self) -> None:
        """A ratchet on which publishers granted nothing this repo can ship.

        Fails in both directions. Recording a real SPDX identifier for one of
        the eight UNDETERMINED sources shrinks the set, which is progress that
        has to reach NOTICE's open-questions section in the same commit.
        Ingesting a source under a publisher's own prose notice grows it, which
        is an exposure that has to reach NOTICE's framework table.
        """
        actual = frozenset(
            framework_id
            for framework_id, licence in FRAMEWORK_LICENSES.items()
            if not spdx_identifiers(licence)
        )
        assert actual == _NO_SHIPPABLE_TEXT, (
            f"the set of frameworks with no shippable licence text moved: "
            f"newly unshippable {sorted(actual - _NO_SHIPPABLE_TEXT)}, "
            f"newly shippable {sorted(_NO_SHIPPABLE_TEXT - actual)}. Update "
            f"NOTICE to say what changed, then update this set."
        )

    def test_no_framework_id_is_used_as_a_licence_filename(self) -> None:
        """LICENSES/ is keyed by SPDX identifier, never by framework.

        A file named csa_aicm.txt would read as a licence grant for a source
        whose notice reserves redistribution outright, which is the strongest
        possible misstatement this directory could make.
        """
        collisions = sorted(shipped_license_text_ids() & set(FRAMEWORK_LICENSES))
        assert not collisions, (
            f"{collisions} are framework ids with a file under "
            f"{LICENSE_TEXTS_DIR.name}/. That directory carries SPDX licence "
            f"texts, not per-framework terms."
        )


class TestThePublishedDeclarationIsNotASingleGrant:
    """`license: other` has to be earned by the corpus, not asserted.

    The four declarations this replaced were each a single identifier, and each
    was wrong in the same way: it granted terms over content drawn from
    publishers who grant different terms. Deriving the "no single identifier
    fits" claim from FRAMEWORK_LICENSES means the day it stops being true, this
    fails and the decision is retaken rather than inherited.
    """

    def test_the_corpus_really_carries_conflicting_terms(self) -> None:
        distinct = {
            identifier
            for licence in FRAMEWORK_LICENSES.values()
            for identifier in spdx_identifiers(licence)
        }
        reservations = {
            framework_id
            for framework_id, licence in FRAMEWORK_LICENSES.items()
            if not spdx_identifiers(licence)
        }
        assert len(distinct) > 1 and reservations, (
            f"the corpus now carries {sorted(distinct)} and "
            f"{len(reservations)} sources with no SPDX grant. If it has "
            f"narrowed to one set of terms, a single identifier may now be "
            f"correct for the published artifacts. Retake the decision in "
            f"tract/licensing.py rather than leaving `other` in place."
        )

    def test_the_published_declaration_is_other(self) -> None:
        """Guards the specific substitution the derivation above allows for.

        Setting PUBLISHED_LICENSE_ID to any single identifier while the corpus
        still carries conflicting terms is the original defect returning under
        a different value.
        """
        from tract.licensing import PUBLISHED_LICENSE_ID

        assert PUBLISHED_LICENSE_ID == "other", (
            f"the published artifacts declare {PUBLISHED_LICENSE_ID!r}, a "
            f"single grant over content from publishers whose terms conflict. "
            f"See the test above for the derivation."
        )

    def test_pyproject_declares_the_project_licence(self) -> None:
        """The wheel is TRACT's own code, so a single identifier IS right here.

        pyproject.toml carried no `license` key at all, so the package the CLI
        installs from stated nothing while three other artifacts stated three
        different things. `tool.setuptools.packages.find` includes `tract*` and
        nothing else, which is why CC0-1.0 is the whole truth for the wheel and
        why this assertion is different from the one above.
        """
        import tomllib

        data = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        project = data["project"]
        assert project.get("license") == PROJECT_LICENSE_ID, (
            f"pyproject.toml declares {project.get('license')!r}, not "
            f"{PROJECT_LICENSE_ID!r}. The wheel carries TRACT's own code only."
        )
        assert data["tool"]["setuptools"]["packages"]["find"]["include"] == [
            "tract*"
        ], (
            "the wheel now includes something other than tract*, so CC0-1.0 "
            "may no longer cover everything it ships. Re-derive the licence."
        )
        license_files = project.get("license-files", [])
        assert "LICENSE" in license_files
        assert "NOTICE" in license_files, (
            "the built distribution would carry no per-framework terms"
        )
        assert f"{LICENSE_TEXTS_DIR.name}/*{LICENSE_TEXT_SUFFIX}" in license_files


def test_the_shipped_cc0_text_is_the_one_the_repository_grants_under() -> None:
    """LICENSES/CC0-1.0.txt and the root LICENSE must be the same legal code.

    The root LICENSE carries the CC0 legal code plus a trailing scope note that
    is not part of it. Two copies of a licence in one tree drift, and a reader
    who trusts the LICENSES/ copy would then be reading terms the repository
    does not grant under.
    """
    shipped = (LICENSE_TEXTS_DIR / f"{PROJECT_LICENSE_ID}{LICENSE_TEXT_SUFFIX}").read_text(
        encoding="utf-8"
    )
    root = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert shipped.strip() in root, (
        f"{LICENSE_TEXTS_DIR.name}/{PROJECT_LICENSE_ID}{LICENSE_TEXT_SUFFIX} is "
        f"not the CC0 legal code carried in the root LICENSE. One of the two "
        f"has drifted."
    )
