"""Tests for parsers/parse_owasp_cheat_sheets.py."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from tract.config import RAW_FRAMEWORKS_DIR

RAW_DIR = RAW_FRAMEWORKS_DIR / "owasp_cheat_sheets"


def _parse() -> object:
    from parsers.parse_owasp_cheat_sheets import OwaspCheatSheetsParser

    if not (RAW_DIR / "cheatsheets.zip").exists():
        pytest.skip("Raw data not available")
    with tempfile.TemporaryDirectory() as td:
        return OwaspCheatSheetsParser(raw_dir=RAW_DIR, output_dir=Path(td)).run()


def test_parses_real_data() -> None:
    result = _parse()

    assert result.framework_id == "owasp_cheat_sheets"
    assert len(result.controls) == 120

    ids = {c.control_id for c in result.controls}
    assert "Docker Security Cheat Sheet" in ids
    assert all(c.description for c in result.controls)


def test_control_id_is_the_file_name_not_the_h1() -> None:
    """OpenCRE's section_id is the file name with underscores as spaces.

    28 sheets carry an H1 that disagrees with their file name, so keying on
    the H1 would lose those links. A divergent H1 becomes an alternate name.
    """
    result = _parse()

    for control in result.controls:
        stem = Path(control.metadata["source_file"]).stem
        assert control.control_id == stem.replace("_", " ")
        assert control.title == control.control_id


def test_every_control_is_pinned_to_one_commit() -> None:
    result = _parse()

    revisions = {c.metadata["revision"] for c in result.controls}
    assert len(revisions) == 1
    assert len(revisions.pop()) == 40


def test_a_commentless_archive_is_refused() -> None:
    """The pin is only a pin if a missing commit stops the parse.

    Skipping the comparison when the zip carries no comment and then stamping
    the pinned commit into every control's metadata is a provenance claim the
    archive never made.
    """
    from parsers.parse_owasp_cheat_sheets import OwaspCheatSheetsParser

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"
        raw.mkdir()
        with zipfile.ZipFile(raw / "cheatsheets.zip", "w") as bundle:
            bundle.writestr("cheatsheets/Example_Cheat_Sheet.md", "# Example\n\nBody.\n")
            # No bundle.comment assignment: this is the fail-open path.

        parser = OwaspCheatSheetsParser(raw_dir=raw, output_dir=Path(td) / "out")
        with pytest.raises(ValueError, match="no commit"):
            parser.parse()


def test_a_mismatched_commit_is_refused() -> None:
    from parsers.parse_owasp_cheat_sheets import OwaspCheatSheetsParser

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"
        raw.mkdir()
        with zipfile.ZipFile(raw / "cheatsheets.zip", "w") as bundle:
            bundle.writestr("cheatsheets/Example_Cheat_Sheet.md", "# Example\n\nBody.\n")
            bundle.comment = b"0" * 40

        parser = OwaspCheatSheetsParser(raw_dir=raw, output_dir=Path(td) / "out")
        with pytest.raises(ValueError, match="pinned to"):
            parser.parse()
