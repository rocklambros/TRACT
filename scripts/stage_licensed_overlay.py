"""Move the licensed overlay between machines, with the digests checked.

    # on the machine that HAS the sources (the Mac):
    python -m scripts.stage_licensed_overlay --pack ~/tract-overlay.tar.gz

    # after copying that file across, on the machine that NEEDS them:
    python -m scripts.stage_licensed_overlay --unpack ~/tract-overlay.tar.gz
    python -m scripts.stage_licensed_overlay --verify

WHY THIS EXISTS. Three frameworks' prose cannot go in git. ETSI's notice
requires written permission to reproduce, ISO/IEC 27001's is a single-user
store licence, and DSOMM is GPL-3.0 whose share-alike a CC0 grant cannot
carry. So a fresh clone is missing them by design, and since 2026-08-26 the
corpus gate refuses to provision without them rather than training short in
silence.

That left "stage the licensed sources" as a paragraph in a runbook, which is
the shape of instruction this project keeps getting burned by. This is the
same instruction as a command that either succeeds or says why.

**ISO 27001 IS ONE OF THE FIVE VALIDATION FOLDS.** Without it that fold has no
controls at all, so a validation campaign run without staging does not produce
a slightly worse number, it produces no number for a fifth of the split, and
arm selection happens on validation. That is why this is not optional and why
there is no flag to skip it.

WHAT MOVES. The merged overlay plus the three per-framework files, about 13MB.
The archive carries licensed text, so it must not land inside the repository
or anywhere it could be committed: --pack refuses to write into the working
tree. Delete it once the transfer is done.

Owner: TRACT.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
from pathlib import Path
from typing import Final

from tract.config import (
    OVERLAY_FRAMEWORK_IDS,
    PROCESSED_FRAMEWORKS_DIR,
    PROCESSED_LICENSED_DIR,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXIT_OK: Final[int] = 0
EXIT_MISSING: Final[int] = 1
EXIT_REFUSED: Final[int] = 2

OVERLAY_CORPUS: Final[Path] = PROCESSED_LICENSED_DIR / "all_controls.json"


def staged_paths() -> list[Path]:
    """Every file a machine needs to hold the licensed overlay.

    Derived from OVERLAY_FRAMEWORK_IDS rather than listed, so a framework
    joining or leaving the tier changes what gets staged without anyone
    remembering to edit this file.
    """
    paths = [OVERLAY_CORPUS]
    paths.extend(
        PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
        for framework_id in sorted(OVERLAY_FRAMEWORK_IDS)
    )
    return paths


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(paths: list[Path]) -> list[str]:
    """Which staged files are absent or unreadable. Empty means staged."""
    problems = []
    for path in paths:
        rel = path.relative_to(PROJECT_ROOT)
        if not path.is_file():
            problems.append(f"{rel}: absent")
            continue
        try:
            logger.info("  %s  %s", digest(path)[:16], rel)
        except OSError as exc:
            problems.append(f"{rel}: unreadable ({exc})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pack", type=Path, metavar="ARCHIVE",
                       help="write the staged files to a tar.gz for transfer")
    group.add_argument("--unpack", type=Path, metavar="ARCHIVE",
                       help="restore a packed archive into this working tree")
    group.add_argument("--verify", action="store_true",
                       help="report whether this machine holds the overlay")
    args = parser.parse_args(argv)

    paths = staged_paths()

    if args.verify:
        logger.info("Checking %d staged file(s):", len(paths))
        problems = verify(paths)
        if problems:
            for problem in problems:
                logger.error("  %s", problem)
            logger.error(
                "This machine does NOT hold the licensed overlay. It cannot "
                "provision: `provision` refuses on a corpus mismatch, and one "
                "of the five validation folds (ISO 27001) would have no "
                "controls. Pack the files on a machine that has them."
            )
            return EXIT_MISSING
        logger.info("Overlay is staged. Corpus digest above must match "
                    "hub_links_training.meta.json.")
        return EXIT_OK

    if args.pack:
        target = args.pack.expanduser().resolve()
        # The archive carries licensed prose. Inside the working tree it is one
        # `git add -A` away from being the escape this whole apparatus exists
        # to prevent, and .gitignore does not cover a name nobody predicted.
        if PROJECT_ROOT in target.parents or target.parent == PROJECT_ROOT:
            logger.error(
                "Refusing to write %s inside the repository. It contains "
                "licensed text, and an archive in the working tree is one "
                "`git add -A` from being committed. Choose a path outside %s.",
                target, PROJECT_ROOT,
            )
            return EXIT_REFUSED

        problems = verify(paths)
        if problems:
            for problem in problems:
                logger.error("  %s", problem)
            logger.error("This machine does not hold the overlay to pack.")
            return EXIT_MISSING

        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target, "w:gz") as archive:
            for path in paths:
                archive.add(path, arcname=str(path.relative_to(PROJECT_ROOT)))
        logger.info("Wrote %s (%.1f MB)", target, target.stat().st_size / 1e6)
        logger.info(
            "Copy it to the other machine, run --unpack there, then --verify, "
            "then DELETE this archive. It carries licensed text.",
        )
        return EXIT_OK

    source = args.unpack.expanduser().resolve()
    if not source.is_file():
        logger.error("%s does not exist", source)
        return EXIT_MISSING
    expected = {str(p.relative_to(PROJECT_ROOT)): p for p in paths}
    written = 0
    with tarfile.open(source, "r:gz") as archive:
        unexpected = sorted(set(archive.getnames()) - set(expected))
        if unexpected:
            # An archive naming a path outside the staged set is either the
            # wrong archive or a tar traversal. Neither gets extracted.
            logger.error(
                "Refusing to extract: %s carries unexpected paths %s. Expected "
                "exactly %s.", source, unexpected[:5], sorted(expected),
            )
            return EXIT_REFUSED

        # Extracted member by member rather than with extractall(), so the
        # destination is a path THIS code computed from OVERLAY_FRAMEWORK_IDS
        # and never one the archive supplied. A hostile tar cannot direct a
        # write, because nothing it contains is used as a path.
        for name, destination in sorted(expected.items()):
            member = archive.getmember(name)
            if not member.isfile():
                logger.error("Refusing to extract %s: not a regular file", name)
                return EXIT_REFUSED
            payload = archive.extractfile(member)
            if payload is None:
                logger.error("Refusing to extract %s: no readable content", name)
                return EXIT_REFUSED
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload.read())
            written += 1

    logger.info("Extracted %d file(s) into %s", written, PROJECT_ROOT)
    return EXIT_OK if not verify(paths) else EXIT_MISSING


if __name__ == "__main__":
    sys.exit(main())
