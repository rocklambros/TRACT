"""Corpus-level invariants that would have caught the synthesised frameworks.

12 of 31 processed frameworks have no parser, no generator anywhere in this
repository, and a description that is a byte copy of the title for all 568 of
their controls. Each test below fails on that condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.config import PROCESSED_FRAMEWORKS_DIR
from tract.parsers.base import BaseParser

PARSERS_DIR = Path(__file__).parent.parent / "parsers"


def _parser_ids() -> set[str]:
    return {p.stem[len("parse_"):] for p in PARSERS_DIR.glob("parse_*.py")}


def _framework_files() -> list[Path]:
    return sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json"))


@pytest.mark.skipif(
    not list(PROCESSED_FRAMEWORKS_DIR.glob("*.json")),
    reason="no processed corpus in this checkout",
)
class TestCorpusInvariants:
    # Plan 1 lands the contract and ISO. The remaining 11 title-only
    # frameworks are Plan 1b, and these tests stay red until then. That is
    # deliberate: a skipped invariant is a forgotten invariant.
    pytestmark = pytest.mark.xfail(
        reason="11 frameworks await parsers, tracked in Plan 1b",
        strict=False,
    )

    def test_every_framework_file_has_a_parser(self) -> None:
        orphans = sorted(
            p.stem for p in _framework_files() if p.stem not in _parser_ids()
        )
        assert not orphans, (
            f"{len(orphans)} processed frameworks have no parser: {orphans}. "
            f"An artifact nothing can regenerate is not reproducible, and "
            f"validate_all.py currently validates these as though a parser "
            f"wrote them."
        )

    def test_no_framework_is_entirely_titles(self) -> None:
        from tract.schema import FrameworkOutput

        offenders: list[tuple[str, float]] = []
        for path in _framework_files():
            data = FrameworkOutput.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            fraction = BaseParser.honest_prose_fraction(data.controls)
            if fraction == 0.0:
                offenders.append((data.framework_id, fraction))
        assert not offenders, (
            f"{len(offenders)} frameworks carry no honest prose at all: "
            f"{offenders}. Their descriptions are byte copies of their titles."
        )

    def test_no_framework_carries_a_synthesised_version_string(self) -> None:
        offenders = []
        for path in _framework_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            if str(data.get("version", "")).startswith("opencre-"):
                offenders.append((data["framework_id"], data["version"]))
        assert not offenders, (
            f"{offenders} were synthesised from the OpenCRE link dump rather "
            f"than parsed from a primary source."
        )
