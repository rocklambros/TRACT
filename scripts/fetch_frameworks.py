"""Fetch primary-source framework documents into data/raw/frameworks/.

    python -m scripts.fetch_frameworks --list
    python -m scripts.fetch_frameworks capec cwe
    python -m scripts.fetch_frameworks --all --verify

Nineteen of the thirty-one frameworks in the corpus reach it through
parsers/fetch_opencre.py, which carries the link graph and each section's name
but never the standard's text. Those frameworks are anchored on three-word
titles, and CAPEC alone is 41% of the training set. This script pulls the real
documents so a parser has something to parse.

data/raw/ is gitignored and immutable once written: parsers read it, nothing
edits it. Every download records a sha256 in the manifest beside it, so a
changed upstream is visible rather than silent. Re-fetching an existing file
requires --force.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import requests

from tract.config import PROCESSED_DIR, RAW_FRAMEWORKS_DIR
from tract.io import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Committed, unlike data/raw/ itself. The point of recording a sha256 is to make
# an upstream change visible later and on another machine, which a gitignored
# manifest cannot do.
MANIFEST_PATH: Final[Path] = PROCESSED_DIR / "framework_sources.json"
TIMEOUT_S: Final[int] = 300
CHUNK_BYTES: Final[int] = 1 << 16


@dataclass(frozen=True)
class Source:
    """One downloadable primary source."""

    framework_id: str
    filename: str
    url: str
    note: str
    # Links in hub_links_curated.jsonl that gain prose from this source.
    training_links: int


SOURCES: Final[tuple[Source, ...]] = (
    Source(
        "capec", "capec_latest.xml",
        "https://capec.mitre.org/data/xml/capec_latest.xml",
        "MITRE CAPEC attack patterns, full descriptions", 1799,
    ),
    Source(
        "cwe", "cwec_latest.xml.zip",
        "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
        "MITRE CWE weaknesses, full descriptions", 613,
    ),
    Source(
        "owasp_cheat_sheets", "cheatsheets.zip",
        "https://github.com/OWASP/CheatSheetSeries/archive/refs/heads/master.zip",
        "OWASP Cheat Sheet Series markdown", 391,
    ),
    Source(
        "nist_800_53", "nist_800_53_catalog.json",
        "https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/"
        "SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json",
        "NIST SP 800-53 rev5 OSCAL catalog, full control text", 300,
    ),
    Source(
        "asvs", "asvs.zip",
        "https://github.com/OWASP/ASVS/archive/refs/heads/master.zip",
        "OWASP ASVS requirements", 277,
    ),
    Source(
        "nist_ai_100_2", "nist_ai_100_2_e2023.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf",
        "NIST AI 100-2e2023 adversarial ML taxonomy. LOFO eval fold.", 45,
    ),
    Source(
        "owasp_ml_top10", "owasp_ml_top10.zip",
        "https://github.com/OWASP/www-project-machine-learning-security-top-10/"
        "archive/refs/heads/master.zip",
        "OWASP ML Security Top 10. LOFO eval fold.", 10,
    ),
)

BY_ID: Final[dict[str, Source]] = {s.framework_id: s for s in SOURCES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    data = load_json(MANIFEST_PATH)
    return data.get("sources", {}) if isinstance(data, dict) else {}


def fetch(source: Source, force: bool = False) -> Path:
    """Download one source and record its hash. Returns the written path."""
    target_dir = RAW_FRAMEWORKS_DIR / source.framework_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.filename

    if target.exists() and not force:
        # Still record it. The manifest has to describe what is on disk, or a
        # file fetched before the manifest existed stays permanently unhashed
        # and its drift undetectable.
        logger.info("%s already present (%s); recording hash, use --force to "
                    "re-fetch", source.framework_id, target.name)
        _record(source, target)
        return target

    logger.info("Fetching %s from %s", source.framework_id, source.url)
    response = requests.get(source.url, timeout=TIMEOUT_S, stream=True)
    response.raise_for_status()

    # Write to a temp path then rename, so an interrupted download cannot leave
    # a truncated file that a parser would happily read.
    temp = target.with_suffix(target.suffix + ".part")
    try:
        with open(temp, "wb") as handle:
            for chunk in response.iter_content(CHUNK_BYTES):
                handle.write(chunk)
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

    _record(source, target)
    logger.info("Wrote %s (%d bytes)", target, target.stat().st_size)
    return target


def _record(source: Source, target: Path) -> None:
    """Hash the file on disk and write it into the committed manifest."""
    digest = _sha256(target)
    manifest = _load_manifest()
    previous = manifest.get(source.framework_id, {}).get("sha256")
    if previous and previous != digest:
        logger.warning(
            "%s changed upstream: %s -> %s. Re-run its parser and expect the "
            "processed output to differ.",
            source.framework_id, previous[:16], digest[:16],
        )
    manifest[source.framework_id] = {
        "url": source.url,
        "filename": source.filename,
        "sha256": digest,
        "bytes": str(target.stat().st_size),
        "fetched_date": date.today().isoformat(),
        "note": source.note,
    }
    atomic_write_json({"sources": manifest}, MANIFEST_PATH)


def verify() -> int:
    """Re-hash everything in the manifest. Returns the number of mismatches."""
    manifest = _load_manifest()
    if not manifest:
        logger.warning("No manifest at %s; nothing to verify", MANIFEST_PATH)
        return 0
    bad = 0
    for framework_id, entry in sorted(manifest.items()):
        path = RAW_FRAMEWORKS_DIR / framework_id / entry["filename"]
        if not path.exists():
            logger.error("%s MISSING: %s", framework_id, path)
            bad += 1
            continue
        actual = _sha256(path)
        status = "ok" if actual == entry["sha256"] else "MISMATCH"
        if actual != entry["sha256"]:
            bad += 1
        logger.info("%-20s %s  %s", framework_id, status, actual[:16])
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch primary framework sources")
    parser.add_argument("frameworks", nargs="*", help="framework ids to fetch")
    parser.add_argument("--all", action="store_true", help="fetch every source")
    parser.add_argument("--list", action="store_true", help="list known sources")
    parser.add_argument("--force", action="store_true", help="re-fetch if present")
    parser.add_argument("--verify", action="store_true",
                        help="re-hash what is on disk against the manifest")
    args = parser.parse_args()

    if args.list:
        print(f"{'framework_id':<22}{'links':>7}  source")
        for source in SOURCES:
            print(f"{source.framework_id:<22}{source.training_links:>7}  {source.note}")
        return 0

    if args.verify and not (args.all or args.frameworks):
        return 1 if verify() else 0

    selected = SOURCES if args.all else tuple(
        BY_ID[name] for name in args.frameworks if name in BY_ID
    )
    unknown = [n for n in args.frameworks if n not in BY_ID]
    if unknown:
        raise SystemExit(f"Unknown framework ids: {unknown}. Try --list.")
    if not selected:
        raise SystemExit("Nothing selected. Pass framework ids, --all, or --list.")

    failures: list[str] = []
    for source in selected:
        try:
            fetch(source, force=args.force)
        except Exception as exc:  # noqa: BLE001 - report all, stop for none
            logger.error("%s FAILED: %s", source.framework_id, exc)
            failures.append(source.framework_id)

    if args.verify:
        verify()
    if failures:
        logger.error("Failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
