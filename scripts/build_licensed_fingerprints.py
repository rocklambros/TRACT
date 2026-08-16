"""Rebuild the tracked fingerprint set for restricted framework sources.

    python -m scripts.build_licensed_fingerprints
    python -m scripts.build_licensed_fingerprints --check

Writes `tests/fixtures/licensed_text_fingerprints.json`, the committed input to
`tests/test_licensed_text_not_tracked.py`. That gate used to read the licensed
source out of gitignored `data/raw/`, so on a fresh clone and in CI it skipped
and the skip reported green.

RE-PIN THE FINGERPRINTS WHENEVER A RESTRICTED SOURCE CHANGES. Every entry is
derived from one document's bytes, and the sha256 of that document is recorded
alongside the entry. Re-fetching ISO/IEC 27001 or ETSI GR SAI 005, correcting a
parse, or adding a framework to RESTRICTED_FRAMEWORK_IDS all require running
this script on a checkout that holds the sources under data/raw/, then
committing the regenerated file. `--check` reports drift without writing, so a
reviewer can tell a stale fingerprint file from a current one.

The output carries hashes only. There is no free-text field in the schema, and
tests/test_licensed_text_not_tracked.py asserts that, so this script cannot
leak the text it reads.

Owner: TRACT.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from tract.config import RAW_FRAMEWORKS_DIR, RESTRICTED_FRAMEWORK_IDS
from tract.io import atomic_write_json
from tract.licensing import (
    FINGERPRINT_ALGORITHM,
    FINGERPRINT_GENERATOR,
    FINGERPRINT_HEX_CHARS,
    FINGERPRINT_PATH,
    FINGERPRINT_SALT,
    NGRAM_WORDS,
    LicensedFingerprints,
    fingerprint_ngrams,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_BYTES: Final[int] = 1 << 16

# A row of ISO's Annex A table: "| 5.1 | <title> | Control <statement> |".
_ANNEX_A_CLAUSE: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+$")
_ANNEX_A_CELLS: Final[int] = 3


def _iso_27001_statements(path: Path) -> list[str]:
    """ISO/IEC 27001:2022 Annex A control statements, title column excluded.

    The titles are deliberately left out. Section names like "Privacy and
    protection of personal identifiable information (PII)" reach this project
    through OpenCRE's public link dump and are already tracked under
    data/training/hub_links*. They are not the normative text at issue. What
    must never enter git is the requirement statement, which the source table
    marks with a leading "Control".
    """
    statements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != _ANNEX_A_CELLS or not _ANNEX_A_CLAUSE.match(cells[0]):
            continue
        statements.append(cells[2])
    if not statements:
        raise ValueError(
            f"{path}: no Annex A table rows matched. The extraction changed "
            f"shape and the fingerprints would be silently empty."
        )
    return statements


# The page header ETSI repeats on every page: "13 ETSI GR SAI 005 V1.1.1 ...".
# It is furniture, and leaving it in interleaves five stray words into the word
# stream at every page break, which breaks windows that span one.
_ETSI_PAGE_HEADER: Final[re.Pattern[str]] = re.compile(r"^\d+\s+ETSI\s+GR\s+")
# Table-of-contents rows, recognised by their dot leaders.
_TOC_LEADER: Final[re.Pattern[str]] = re.compile(r"\.{5,}")
# The clause that opens the bibliography and the clause that closes it.
_ETSI_REFERENCES_START: Final[re.Pattern[str]] = re.compile(r"^2\s+References\s*$")
_ETSI_REFERENCES_END: Final[re.Pattern[str]] = re.compile(r"^3\s+Definition\b")


def _etsi_normative_text(path: Path) -> list[str]:
    """ETSI GR SAI 005's own prose, bibliography and contents page excluded.

    ETSI has no parser yet, so there is no statement column to isolate the way
    ISO's has. The whole deliverable is fingerprinted instead, front matter and
    copyright notification included, so this repository cannot quote that
    notice at length either.

    Two parts are deliberately left out because they are not ETSI's authored
    content, and including them produced false positives that were measured
    rather than imagined:

      - Clause 2, the informative references. It is a list of other people's
        paper titles and author names. NIST AI 100-2 and the OWASP AI Exchange
        cite several of the same papers, so their tracked JSON reproduced
        author lists such as the one behind [i.2] word for word. Flagging that
        as an ETSI quotation is wrong on the facts.
      - The contents page, whose dot leaders make heading text run together
        into windows that exist nowhere in the body.

    Raises:
        ValueError: neither clause boundary was found, which means the document
            structure changed and the bibliography would silently be back in.
    """
    import pdfplumber  # imported here so the gate never needs it, only the build

    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    raw = "\n".join(pages)
    if not raw.strip():
        raise ValueError(
            f"{path}: pdfplumber extracted no text. A scanned or protected PDF "
            f"would produce an empty fingerprint set that looks like success."
        )

    kept: list[str] = []
    in_references = False
    saw_start = saw_end = False
    for line in raw.splitlines():
        stripped = line.strip()
        if _ETSI_REFERENCES_START.match(stripped):
            in_references, saw_start = True, True
            continue
        if in_references and _ETSI_REFERENCES_END.match(stripped):
            in_references, saw_end = False, True
        if in_references:
            continue
        if _ETSI_PAGE_HEADER.match(stripped) or _TOC_LEADER.search(stripped):
            continue
        kept.append(stripped)

    if not (saw_start and saw_end):
        raise ValueError(
            f"{path}: could not bound the references clause "
            f"(start={saw_start}, end={saw_end}). The document structure "
            f"changed. Fix the boundaries rather than fingerprinting the "
            f"bibliography, which produces false positives on shared citations."
        )

    # One unit, not one per page. A sentence that runs across a page break is
    # still one sentence, and a quotation of it would carry the break too.
    return ["\n".join(kept)]


# One extractor per restricted framework. Keyed on framework_id so adding a
# framework to RESTRICTED_FRAMEWORK_IDS without deciding what its statement
# text is raises here instead of producing a fingerprint file with a hole.
_EXTRACTORS: Final[dict[str, tuple[str, Callable[[Path], list[str]]]]] = {
    "iso_27001": ("ISO_IEC_27001_2022_en.md", _iso_27001_statements),
    "etsi": ("etsi_gr_sai005_v010101p.pdf", _etsi_normative_text),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict[str, Any]:
    """Read every restricted source and return the fingerprint document.

    Raises:
        KeyError: a framework is restricted but has no extractor here.
        FileNotFoundError: a restricted source is not staged under data/raw/.
    """
    missing_extractors = sorted(set(RESTRICTED_FRAMEWORK_IDS) - set(_EXTRACTORS))
    if missing_extractors:
        raise KeyError(
            f"No fingerprint extractor for restricted framework(s) "
            f"{missing_extractors}. Add one to _EXTRACTORS naming the staged "
            f"document and the function that isolates its statement text."
        )

    documents: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
        filename, extract = _EXTRACTORS[framework_id]
        path = RAW_FRAMEWORKS_DIR / framework_id / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is not staged. This script has to run on a checkout "
                f"that holds every restricted source under data/raw/."
            )
        # Fingerprinted per unit, never across the join between two of them.
        # A window spanning the end of one ISO statement and the start of the
        # next is not text anyone could quote, and storing it only widens the
        # surface for an accidental match.
        units = extract(path)
        grams = [gram for unit in units for gram in fingerprint_ngrams(unit)]
        fingerprints.update(grams)
        documents.append({
            "framework_id": framework_id,
            "filename": filename,
            "source_sha256": _sha256(path),
            "ngram_count": len(grams),
        })
        logger.info(
            "%s/%s: %d unit(s), %d words, %d n-grams", framework_id, filename,
            len(units), sum(len(unit.split()) for unit in units), len(grams),
        )

    return {
        "salt": FINGERPRINT_SALT,
        "algorithm": FINGERPRINT_ALGORITHM,
        "ngram_words": NGRAM_WORDS,
        "hash_hex_chars": FINGERPRINT_HEX_CHARS,
        "generator": FINGERPRINT_GENERATOR,
        "documents": documents,
        # Sorted so a rebuild from unchanged sources is byte-identical and a
        # diff on this file means the sources actually moved.
        "fingerprints": sorted(fingerprints),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild tests/fixtures/licensed_text_fingerprints.json",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report drift against the committed file without writing it",
    )
    args = parser.parse_args()

    document = build()
    logger.info(
        "Built %d fingerprints from %d document(s) at n=%d words",
        len(document["fingerprints"]), len(document["documents"]), NGRAM_WORDS,
    )

    if args.check:
        try:
            committed = LicensedFingerprints.load(FINGERPRINT_PATH)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("committed fingerprint file unusable: %s", exc)
            return 1
        built = frozenset(document["fingerprints"])
        if committed.fingerprints == built:
            logger.info("%s is current", FINGERPRINT_PATH)
            return 0
        logger.error(
            "%s is STALE: %d fingerprints only in the committed file, %d only "
            "in the rebuild. Re-run without --check and commit the result.",
            FINGERPRINT_PATH,
            len(committed.fingerprints - built), len(built - committed.fingerprints),
        )
        return 1

    atomic_write_json(document, FINGERPRINT_PATH)
    logger.info("Wrote %s", FINGERPRINT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
