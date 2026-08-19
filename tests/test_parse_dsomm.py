"""DSOMM's prose lives in risk and measure, not in description.

Measured on the pinned archive: description is non-empty on 51 of 194
activities, risk and measure on 194 of 194. A parser reading description alone
emits 143 empty statements and ProseIndex indexes none of them.

The join is the point. OpenCRE keys every DSOMM link on the activity uuid and
puts the SUB-DIMENSION in section_name, so a parser that titles its controls
with the sub-dimension collapses 214 links onto 18 anchors. These tests pin the
uuid as the control id and the activity name as the title, which is what takes
the same 214 links to 182 anchors.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from parsers.parse_dsomm import STATEMENT_FIELDS, DsommParser
from tract.parsers.base import BaseParser

MODEL: dict[str, Any] = {
    "Build and Deployment": {
        "Deployment": {
            "Inventory of production components": {
                "uuid": "2a44b708-734f-4463-b0cb-86dc46344b2f",
                "risk": "Without an inventory of deployed artifacts it is not "
                        "possible to know where a vulnerable image runs.",
                "measure": "A documented inventory of artifacts in production "
                           "is maintained and kept current.",
                "level": 1,
            },
            "Pinning of artifacts": {
                "uuid": "f3c4971e-9f4d-4e59-8ed0-f0bdb6262477",
                "description": "Pin base images and dependencies to an "
                               "immutable digest rather than a moving tag.",
                "risk": "Unauthorized manipulation of artifacts is hard to "
                        "spot when tags move under the build.",
                "measure": "Pinning ensures changes happen only when intended.",
                "level": 2,
            },
        },
    },
}

# Source order here disagrees with every sort a wrong implementation might
# reach for: by uuid, by activity name, by dimension name and by sub-dimension
# name all put the "a0000000" activity first. MODEL above cannot make that
# distinction, because its two activities happen to sort into source order on
# both keys, so an implementation that sorted would pass every assertion built
# on it.
ORDER_MODEL: dict[str, Any] = {
    "Test and Verification": {
        "Static depth for applications": {
            "Zeta activity": {
                "uuid": "f0000000-0000-4000-8000-000000000001",
                "risk": "Untested code reaches production without anyone "
                        "having read what it does.",
                "measure": "Every change is covered by an automated test "
                           "before it is allowed to merge.",
            },
        },
    },
    "Build and Deployment": {
        "Deployment": {
            "Alpha activity": {
                "uuid": "a0000000-0000-4000-8000-000000000002",
                "risk": "A deployment nobody recorded cannot be rolled back "
                        "to a known state.",
                "measure": "Each deployment writes an immutable record of "
                           "what was released and by whom.",
            },
        },
    },
}


def _archive(members: dict[str, str]) -> bytes:
    """A zip carrying exactly the members given, in the order given."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return payload.getvalue()


def _model_document(model: dict[str, Any]) -> str:
    """The two-document layout the generated model file uses.

    `sort_keys=False` is load-bearing. safe_dump sorts by default, so a fixture
    without it re-alphabetises the model on the way in and no test built on it
    can see a parser that sorts. The real generated file is insertion-ordered.
    """
    return (
        "---\nmeta:\n  version: test\n---\n"
        + yaml.safe_dump(model, sort_keys=False)
    )


def _parser_for(
    tmp_path: Path, members: dict[str, str], expected_count: int,
) -> DsommParser:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "dsomm_data.zip").write_bytes(_archive(members))

    instance = DsommParser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_count = expected_count  # type: ignore[misc]
    # A fixture archive is not the pinned one, so the real digest is stood
    # down here rather than the gate being widened to accept two archives.
    instance.expected_sha256 = None  # type: ignore[misc]
    return instance


@pytest.fixture()
def parser(tmp_path: Path) -> DsommParser:
    return _parser_for(
        tmp_path,
        {"repo-abc123/generated/model.yaml": _model_document(MODEL)},
        expected_count=2,
    )


@pytest.fixture()
def order_parser(tmp_path: Path) -> DsommParser:
    return _parser_for(
        tmp_path,
        {"repo-abc123/generated/model.yaml": _model_document(ORDER_MODEL)},
        expected_count=2,
    )


class TestParse:
    def test_control_id_is_the_uuid_opencre_links_against(
        self, parser: DsommParser,
    ) -> None:
        controls = parser.parse()
        assert [c.control_id for c in controls] == [
            "2a44b708-734f-4463-b0cb-86dc46344b2f",
            "f3c4971e-9f4d-4e59-8ed0-f0bdb6262477",
        ]

    def test_title_is_the_activity_name_not_the_sub_dimension(
        self, parser: DsommParser,
    ) -> None:
        titles = [c.title for c in parser.parse()]
        assert titles == ["Inventory of production components",
                          "Pinning of artifacts"]
        assert "Deployment" not in titles

    def test_statement_survives_an_absent_description(
        self, parser: DsommParser,
    ) -> None:
        first = parser.parse()[0]
        assert "inventory of deployed artifacts" in first.description
        assert "documented inventory" in first.description
        assert len(first.description) >= 60

    def test_description_leads_when_present(self, parser: DsommParser) -> None:
        second = parser.parse()[1]
        assert second.description.startswith("Pin base images")

    def test_statement_joins_all_three_fields_in_source_order(
        self, parser: DsommParser,
    ) -> None:
        """The publisher's own key order is description, risk, measure.

        `startswith` above pins the lead field only, so a parser joining
        description, measure, risk passes it. This pins the whole statement.

        The expected order is spelled out here rather than read from
        STATEMENT_FIELDS. Deriving it from the constant makes the assertion
        move with the code it is meant to hold still, and a swapped-order
        mutation survived this test until the constant came out of it.
        """
        second = parser.parse()[1]
        body = MODEL["Build and Deployment"]["Deployment"]["Pinning of artifacts"]
        assert STATEMENT_FIELDS == ("description", "risk", "measure")
        assert second.description == "\n\n".join(
            (body["description"], body["risk"], body["measure"])
        )

    def test_sub_dimension_is_recorded_as_the_parent(
        self, parser: DsommParser,
    ) -> None:
        first = parser.parse()[0]
        assert first.parent_id == "Deployment"
        assert first.parent_name == "Build and Deployment"
        assert first.metadata == {
            "sub_dimension": "Deployment",
            "dimension": "Build and Deployment",
        }

    def test_activities_keep_source_order_across_dimensions(
        self, order_parser: DsommParser,
    ) -> None:
        """Sorting is a silent reordering, so name the order the source has."""
        controls = order_parser.parse()
        assert [c.control_id for c in controls] == [
            "f0000000-0000-4000-8000-000000000001",
            "a0000000-0000-4000-8000-000000000002",
        ]
        assert [c.title for c in controls] == ["Zeta activity", "Alpha activity"]


class TestRefusals:
    """Every one of these is a silent wrong answer if it does not raise."""

    def test_an_activity_without_a_uuid_is_refused(self) -> None:
        model = {"Dim": {"Sub": {"Nameless": {
            "risk": "A risk statement long enough to be a real one.",
            "measure": "A measure statement long enough to be a real one.",
        }}}}
        with pytest.raises(ValueError, match="has no uuid"):
            DsommParser.activities_to_controls(model)

    def test_an_activity_with_a_blank_uuid_is_refused(self) -> None:
        model = {"Dim": {"Sub": {"Blank": {
            "uuid": "   ",
            "risk": "A risk statement long enough to be a real one.",
            "measure": "A measure statement long enough to be a real one.",
        }}}}
        with pytest.raises(ValueError, match="has no uuid"):
            DsommParser.activities_to_controls(model)

    def test_an_activity_with_no_statement_text_is_refused(self) -> None:
        model = {"Dim": {"Sub": {"Empty": {
            "uuid": "2a44b708-734f-4463-b0cb-86dc46344b2f",
            "risk": "",
            "measure": "   ",
            "level": 3,
        }}}}
        with pytest.raises(ValueError, match="has no text in any of"):
            DsommParser.activities_to_controls(model)

    def test_an_archive_without_the_generated_model_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """The per-sub-dimension YAMLs cannot answer the join level."""
        instance = _parser_for(
            tmp_path,
            {"repo-abc123/model.yaml": _model_document(MODEL)},
            expected_count=2,
        )
        with pytest.raises(ValueError, match="expected exactly one"):
            instance.parse()

    def test_two_generated_models_are_refused(self, tmp_path: Path) -> None:
        document = _model_document(MODEL)
        instance = _parser_for(
            tmp_path,
            {
                "repo-abc123/generated/model.yaml": document,
                "repo-def456/generated/model.yaml": document,
            },
            expected_count=2,
        )
        with pytest.raises(ValueError, match="expected exactly one"):
            instance.parse()

    def test_a_model_without_its_meta_document_is_refused(
        self, tmp_path: Path,
    ) -> None:
        instance = _parser_for(
            tmp_path,
            {"repo-abc123/generated/model.yaml": yaml.safe_dump(MODEL)},
            expected_count=2,
        )
        with pytest.raises(ValueError, match="not a meta document"):
            instance.parse()


class TestRun:
    def test_run_writes_and_clears_the_prose_floor(
        self, parser: DsommParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 2
        assert (tmp_path / "out" / "dsomm.json").exists()
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0
        assert [s.path for s in output.source_files] == ["dsomm_data.zip"]

    def test_a_placeholder_statement_trips_the_prose_floor(
        self, tmp_path: Path,
    ) -> None:
        """Nothing else here proves the floor is set to a non-zero value.

        Upstream really does ship placeholders: activity 7de0ae33 carries
        `risk: TODO.` and `measure: TODO`, which is the one link of 214 the
        source cannot answer. Half a corpus of those must not write.
        """
        model = {"Dim": {"Sub": {
            "Real activity": {
                "uuid": "2a44b708-734f-4463-b0cb-86dc46344b2f",
                "risk": "Without an inventory of deployed artifacts it is "
                        "not possible to know where a vulnerable image runs.",
                "measure": "A documented inventory is kept current.",
            },
            "Placeholder activity": {
                "uuid": "f3c4971e-9f4d-4e59-8ed0-f0bdb6262477",
                "risk": "TODO.",
                "measure": "TODO",
            },
        }}}
        instance = _parser_for(
            tmp_path,
            {"repo-abc123/generated/model.yaml": _model_document(model)},
            expected_count=2,
        )
        (tmp_path / "out").mkdir()
        with pytest.raises(ValueError, match="below the declared floor"):
            instance.run()

    def test_reparse_is_byte_identical(
        self, parser: DsommParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        first = (tmp_path / "out" / "dsomm.json").read_bytes()
        parser.run()
        assert (tmp_path / "out" / "dsomm.json").read_bytes() == first


class TestDigestGate:
    def test_a_different_archive_is_refused(self, parser: DsommParser) -> None:
        parser.expected_sha256 = "0" * 64  # type: ignore[misc]
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()

    def test_the_shipped_pin_is_the_one_the_fetcher_downloads(self) -> None:
        """One pin, two files, and the parser's error message says so.

        Without this the class could ship `expected_sha256 = None` and every
        other test here would still pass, because each of them stands the gate
        down on purpose.
        """
        from scripts.fetch_frameworks import SOURCES

        pins = {
            source.expected_sha256
            for source in SOURCES
            if source.framework_id == "dsomm"
        }
        assert pins == {DsommParser.expected_sha256}
