"""Every parser must read its inputs through the recording readers.

The source manifest replaced data/processed/framework_sources.json, which was
hand-maintained and had drifted to covering 7 of 19 parser-backed frameworks.
It then covered 1 of 20, because the mandate lived in a docstring while
nineteen parsers kept calling read_text and open() directly.

BaseParser.run() now refuses to write on an empty manifest. This test is the
static half: it fails on a parser that reads a file outside read_source even
if nothing ever runs that parser, which matters because two of them need a
package that is not installed everywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

PARSERS_DIR = Path(__file__).resolve().parent.parent / "parsers"

# Filesystem reads that bypass the manifest. Matched on the source text rather
# than by importing, so a parser whose third-party dependency is missing is
# still covered.
#
# `.open(` alone is not here: a zipfile member read is `archive.open(name)`
# and stays in memory. The text-mode keyword is what distinguishes a
# Path.open from it.
_DIRECT_READ = re.compile(
    r"\.read_text\(|\.read_bytes\(|\.open\(encoding=|json\.load\(|"
    r"load_json\(|(?<![.\w])open\(",
)

# Readers that take either a path or an already-open file object. Handed a
# path they read the file themselves, invisibly to the manifest, so the bytes
# have to arrive already recorded. Only the path-shaped argument is flagged:
# handing one a zipfile member handle keeps the read in memory and is fine.
_PATH_ARGUMENT = r"(?:[A-Za-z_]*(?:path|source|archive|file|dir)|self\.raw_dir[^)]*)"
_NEEDS_IN_MEMORY = re.compile(
    rf"(?:ZipFile|pdfplumber\.open|parse_xml)\(\s*{_PATH_ARGUMENT}\s*\)",
    re.IGNORECASE,
)


def _parser_files() -> list[Path]:
    return sorted(PARSERS_DIR.glob("parse_*.py"))


def test_at_least_the_known_parsers_are_present() -> None:
    """Guards the two tests below against silently scanning nothing."""
    assert len(_parser_files()) >= 20


def test_every_parser_records_its_sources_or_declares_an_exemption() -> None:
    offenders = []
    for path in _parser_files():
        source = path.read_text(encoding="utf-8")
        if "read_source" in source or "manifest_exempt_reason" in source:
            continue
        offenders.append(path.name)

    assert not offenders, (
        f"{len(offenders)} parsers never call read_source or "
        f"read_source_bytes: {offenders}. A file opened directly is invisible "
        f"to the manifest, so the artifact states nothing about which bytes "
        f"produced it."
    )


def test_no_parser_still_opens_a_raw_file_directly() -> None:
    """A migrated parser must not keep a second, unrecorded read path.

    One recorded read does not make a parser honest. It can read its main
    input through read_source and still pull a second file directly, and the
    run-time gate cannot tell, because it only checks that the manifest is
    non-empty.
    """
    offenders: list[tuple[str, int, str]] = []
    for path in _parser_files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _DIRECT_READ.search(stripped):
                offenders.append((path.name, number, stripped))
                continue
            if _NEEDS_IN_MEMORY.search(stripped):
                offenders.append((path.name, number, stripped))

    assert not offenders, (
        f"{len(offenders)} direct reads of a raw path remain: {offenders}. "
        f"Route them through read_source or read_source_bytes."
    )
