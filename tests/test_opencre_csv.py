"""Tests for tract.export.opencre_csv — CSV generation (spec §3)."""
from __future__ import annotations

import csv
import subprocess
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tract.config import OVERLAY_FRAMEWORK_IDS, PHASE5_OPENCRE_EXPORT_DIR
from tract.export.filters import ExportableAssignment
from tract.export.opencre_csv import generate_opencre_csv, write_opencre_csv
from tract.export.opencre_names import (
    HYPERLINK_TEMPLATES,
    TRACT_TO_OPENCRE_NAME,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def _make_row(
    hub_id: str = "607-671",
    hub_name: str = "Protect against injection",
    framework_id: str = "mitre_atlas",
    section_id: str = "AML.M0015",
    title: str = "Adversarial Input Detection",
    description: str = "Detect adversarial inputs",
) -> ExportableAssignment:
    """One row as query_exportable_assignments returns it.

    Typed against the real TypedDict rather than a bare dict, so a column the
    exporter reads cannot be dropped from this fixture without mypy saying so.
    The four fields the CSV never writes are filled anyway, because the row
    shape is the contract and a partial one hides which fields the writer
    depends on.
    """
    return {
        "control_id": f"{framework_id}:{section_id}",
        "hub_id": hub_id,
        "hub_name": hub_name,
        "confidence": 0.8,
        "is_ood": 0,
        "provenance": "active_learning_round_2",
        "framework_id": framework_id,
        "section_id": section_id,
        "title": title,
        "description": description,
    }


class TestGenerateOpencreCsv:
    def test_header_uses_opencre_name(self) -> None:
        csv_text = generate_opencre_csv([_make_row()], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        header = next(reader)
        assert header[0] == "CRE 0"
        assert header[1] == "MITRE ATLAS|name"
        assert header[2] == "MITRE ATLAS|id"
        assert header[3] == "MITRE ATLAS|description"
        assert header[4] == "MITRE ATLAS|hyperlink"

    def test_cre0_pipe_delimited(self) -> None:
        csv_text = generate_opencre_csv([_make_row()], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        row = next(reader)
        assert row[0] == "607-671|Protect against injection"

    def test_standard_columns_populated(self) -> None:
        csv_text = generate_opencre_csv([_make_row()], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        row = next(reader)
        assert row[1] == "Adversarial Input Detection"
        assert row[2] == "AML.M0015"
        assert row[3] == "Detect adversarial inputs"
        # Parse the host rather than searching the whole URL for a substring.
        # `"atlas.mitre.org" in url` also accepts `https://evil.test/atlas.mitre.org`
        # and `https://atlas.mitre.org.evil.test/`, so it asserts something
        # weaker than it appears to. CodeQL flags the pattern as
        # py/incomplete-url-substring-sanitization, and it is right: the same
        # confusion between a host and a substring showed up in this project's
        # own data, where `cwe.mitre.org` in a control body made `mitre` look
        # like prose the encoder sees.
        assert urlparse(row[4]).hostname == "atlas.mitre.org"

    def test_sorted_by_hub_framework_section(self) -> None:
        rows = [
            _make_row(hub_id="999-999", section_id="B"),
            _make_row(hub_id="111-111", section_id="A"),
            _make_row(hub_id="111-111", section_id="C"),
        ]
        csv_text = generate_opencre_csv(rows, "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        data_rows = list(reader)
        assert data_rows[0][0].startswith("111-111")
        assert data_rows[1][0].startswith("111-111")
        assert data_rows[2][0].startswith("999-999")
        assert data_rows[0][2] == "A"
        assert data_rows[1][2] == "C"

    def test_empty_rows(self) -> None:
        csv_text = generate_opencre_csv([], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        header = next(reader)
        assert len(header) == 5
        remaining = list(reader)
        assert remaining == []

    def test_multiple_hubs_same_control(self) -> None:
        rows = [
            _make_row(hub_id="111-111", hub_name="Hub A"),
            _make_row(hub_id="222-222", hub_name="Hub B"),
        ]
        csv_text = generate_opencre_csv(rows, "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        data_rows = list(reader)
        assert len(data_rows) == 2

    def test_unknown_framework_raises(self) -> None:
        with pytest.raises(KeyError):
            generate_opencre_csv([_make_row()], "nonexistent_framework")

    def test_new_framework_csv(self) -> None:
        row = _make_row(framework_id="csa_aicm", section_id="AICM-01")
        csv_text = generate_opencre_csv([row], "csa_aicm")
        reader = csv.reader(StringIO(csv_text))
        header = next(reader)
        assert header[1] == "CSA AI Controls Matrix|name"


class TestWriteOpencreCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        rows = [_make_row()]
        result = write_opencre_csv(rows, "mitre_atlas", tmp_path)
        assert result.exists()
        assert result.suffix == ".csv"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "subdir" / "nested"
        result = write_opencre_csv([_make_row()], "mitre_atlas", out)
        assert result.exists()

    def test_file_content_matches_generate(self, tmp_path: Path) -> None:
        rows = [_make_row()]
        result = write_opencre_csv(rows, "mitre_atlas", tmp_path)
        expected = generate_opencre_csv(rows, "mitre_atlas")
        with open(result, encoding="utf-8", newline="") as f:
            actual = f.read()
        assert actual == expected


# ── Licence filtering on the CSV path ────────────────────────────────────
#
# tract/export/canonical.py grew a tier filter and this path did not, so the
# `<Standard>|description` column kept carrying the publisher's own control
# statement for whichever framework it was handed. The output directory sits at
# the repository root and holds tracked files, and the command's stated
# destination is OpenCRE's importer, which is outside git where no ignore rule
# reaches. Both halves are covered below.

# Invented prose, not a quotation of any licensed source. Long enough to be a
# control statement rather than a restated title, which is the shape the filter
# has to catch.
_LICENSED_PROSE = (
    "Regional facility credentials shall be issued, reviewed and revoked "
    "under a documented process approved by the accountable security lead."
)


class TestUnpublishableControlTextIsWithheld:
    """An overlay framework exports identifiers and no control statement.

    TRACT_TO_OPENCRE_NAME lists six frameworks today and none of them is in a
    licence tier, so nothing has leaked through this path yet. That is what
    makes it worth a gate rather than a note: the exposure is one name-map
    entry away, the entry is a one-line change nobody would think of as a
    licensing decision, and the output lands in a directory that already has
    tracked files in it.

    The fixture registers that entry, which is what makes every assertion here
    reachable in both directions. Without the filter these rows write the
    publisher's statement into the description column.
    """

    @pytest.fixture
    def overlay_id(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """A real overlay framework, made reachable by the CSV exporter.

        Taken from the live constant rather than named here, so the fixture
        cannot drift out of step with the tier it tests.
        """
        assert OVERLAY_FRAMEWORK_IDS, "no overlay tier to enforce"
        framework_id = sorted(OVERLAY_FRAMEWORK_IDS)[0]
        assert framework_id not in TRACT_TO_OPENCRE_NAME, (
            f"{framework_id} now has an OpenCRE name of its own, so this "
            f"fixture is masking the real mapping. Use the real one."
        )
        monkeypatch.setitem(
            TRACT_TO_OPENCRE_NAME, framework_id, framework_id.upper(),
        )
        monkeypatch.setitem(
            HYPERLINK_TEMPLATES, framework_id, "https://example.com/{section_id}",
        )
        return framework_id

    def test_an_overlay_framework_exports_no_control_text(
        self, overlay_id: str,
    ) -> None:
        row = _make_row(framework_id=overlay_id, description=_LICENSED_PROSE)
        csv_text = generate_opencre_csv([row], overlay_id)
        assert _LICENSED_PROSE not in csv_text, (
            f"{overlay_id}'s control statement reached an OpenCRE export CSV. "
            f"The export's destination is OpenCRE's importer, outside git, so "
            f"no .gitignore rule stops it."
        )

    def test_a_publishable_framework_keeps_its_control_text(self) -> None:
        """The other direction.

        A filter that withheld every framework's text would pass the test
        above and destroy the deliverable. Five of the six frameworks with an
        OpenCRE name have their description column populated in git today.
        """
        csv_text = generate_opencre_csv(
            [_make_row(description=_LICENSED_PROSE)], "mitre_atlas",
        )
        assert _LICENSED_PROSE in csv_text

    def test_identifier_title_and_hyperlink_survive(
        self, overlay_id: str,
    ) -> None:
        """Withheld text, not a withheld row.

        Dropping the row would drop TRACT's own CC0 mapping in order to
        protect somebody else's text, and OpenCRE already publishes these
        section identifiers and names.
        """
        row = _make_row(framework_id=overlay_id, description=_LICENSED_PROSE)
        reader = csv.reader(StringIO(generate_opencre_csv([row], overlay_id)))
        next(reader)
        data = next(reader)
        assert data[0] == "607-671|Protect against injection"
        assert data[1] == "Adversarial Input Detection"
        assert data[2] == "AML.M0015"
        assert data[4] == "https://example.com/AML.M0015"

    def test_the_placeholder_says_why_and_names_the_framework(
        self, overlay_id: str,
    ) -> None:
        """An empty cell reads as "no description", which is a lie.

        A recipient importing the CSV has to be able to tell a withheld
        statement from an absent one.
        """
        row = _make_row(framework_id=overlay_id, description=_LICENSED_PROSE)
        reader = csv.reader(StringIO(generate_opencre_csv([row], overlay_id)))
        next(reader)
        description = next(reader)[3]
        assert description, "the placeholder is empty, so it explains nothing"
        assert "withheld" in description.lower()
        assert overlay_id in description

    def test_every_overlay_framework_is_filtered_not_just_the_first(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Driven by the set. A per-framework hole would survive the tests above."""
        leaked: list[str] = []
        for framework_id in sorted(OVERLAY_FRAMEWORK_IDS):
            monkeypatch.setitem(
                TRACT_TO_OPENCRE_NAME, framework_id, framework_id.upper(),
            )
            monkeypatch.setitem(
                HYPERLINK_TEMPLATES, framework_id, "https://example.com/{section_id}",
            )
            row = _make_row(
                framework_id=framework_id, description=_LICENSED_PROSE,
            )
            if _LICENSED_PROSE in generate_opencre_csv([row], framework_id):
                leaked.append(framework_id)
        assert not leaked, f"{leaked} export their control text unfiltered"

    def test_a_row_keyed_on_another_framework_is_still_filtered(
        self, overlay_id: str,
    ) -> None:
        """The filter reads the ROW's framework, not the caller's argument.

        Every call site passes rows already filtered to one framework, so the
        two agree today. Keying on the argument would let a withheld
        framework's row ride out under a publishable one's name the first time
        a caller passes a mixed list, and the caller is a public function.
        """
        rows = [
            _make_row(description=_LICENSED_PROSE),
            _make_row(framework_id=overlay_id, section_id="OVL-1",
                      description=_LICENSED_PROSE),
        ]
        csv_text = generate_opencre_csv(rows, "mitre_atlas")
        assert csv_text.count(_LICENSED_PROSE) == 1, (
            "the overlay row's control statement survived a mixed-framework "
            "export because the filter read the argument instead of the row"
        )
        assert "OVL-1" in csv_text, "the overlay row's identifier was dropped"

    def test_a_row_with_no_framework_id_raises(self) -> None:
        """Fail loud. Defaulting an unattributed row to publishable is how an
        unfiltered source reaches OpenCRE's importer."""
        row = _make_row(framework_id="", description=_LICENSED_PROSE)
        with pytest.raises(ValueError, match="no framework_id"):
            generate_opencre_csv([row], "mitre_atlas")

    def test_the_written_csv_carries_no_control_text(
        self, overlay_id: str, tmp_path: Path,
    ) -> None:
        """End to end, on the bytes that actually leave the machine."""
        row = _make_row(framework_id=overlay_id, description=_LICENSED_PROSE)
        path = write_opencre_csv([row], overlay_id, tmp_path)
        body = path.read_text(encoding="utf-8")
        assert _LICENSED_PROSE not in body
        assert "AML.M0015" in body


def test_the_default_export_directory_is_gitignored() -> None:
    """The smaller half of the fix, and the one a stray `git add -A` needs.

    Asserted against git rather than against the .gitignore text, because a
    line that is present and shadowed by a later negation ignores nothing. The
    probe is a path this directory does not hold: `git check-ignore` skips
    paths already in the index, and seven files under here are tracked, so a
    probe naming one of them would report "not ignored" and say nothing about
    the rule.
    """
    probe = PHASE5_OPENCRE_EXPORT_DIR.relative_to(REPO_ROOT) / "New_Framework.csv"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(probe)],
        cwd=REPO_ROOT, capture_output=True,
    )
    assert result.returncode == 0, (
        f"{probe} is not ignored by git. `tract export --opencre && "
        f"git add -A` would stage every exported framework's control text."
    )


def test_no_tracked_export_csv_carries_an_overlay_framework_description() -> None:
    """The files already in git, checked rather than assumed.

    The seven tracked files under opencre_export/ predate the filter and are
    deliberately left alone: `git rm` on them un-publishes nothing, moves
    published metrics, and pre-empts the owner decision NOTICE records about
    csa_aicm. Left alone is not the same as unchecked. This reads each tracked
    CSV back, resolves its description column to the framework that wrote it,
    and fails if that framework is one whose text may not be redistributed.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "opencre_export/*.csv"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert tracked, (
        "no CSV under opencre_export/ is tracked, so this gate inspected "
        "nothing. If the files were removed, say so in the commit and delete "
        "this test."
    )

    name_to_framework = {
        name: framework_id
        for framework_id, name in TRACT_TO_OPENCRE_NAME.items()
    }
    offenders: list[str] = []
    populated = 0
    for relative in sorted(tracked):
        with open(REPO_ROOT / relative, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = [
            column for column in (rows[0] if rows else {})
            if column.endswith("|description")
        ]
        assert len(columns) == 1, (
            f"{relative} has {len(columns)} description columns, expected 1"
        )
        opencre_name = columns[0].removesuffix("|description")
        framework_id = name_to_framework.get(opencre_name)
        assert framework_id is not None, (
            f"{relative} carries the OpenCRE name {opencre_name!r}, which maps "
            f"to no framework in TRACT_TO_OPENCRE_NAME, so nothing can decide "
            f"whether its text may be redistributed."
        )
        filled = sum(1 for row in rows if (row[columns[0]] or "").strip())
        if filled:
            populated += 1
        if filled and framework_id in OVERLAY_FRAMEWORK_IDS:
            offenders.append(f"{relative}: {filled} {framework_id} descriptions")

    assert populated, (
        "every tracked export CSV has an empty description column, so this "
        "gate could not tell a filtered file from an unreadable one"
    )
    assert not offenders, (
        f"{offenders} are tracked in a CC0 repository and carry the control "
        f"text of a framework whose licence does not permit redistribution. "
        f"Re-run the export, which now withholds it."
    )
