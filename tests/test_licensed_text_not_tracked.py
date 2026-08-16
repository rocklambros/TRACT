"""Licensed framework text must never enter git.

This repository is CC0 (see LICENSE), which is not a disclaimer. It is an
affirmative grant asserting the publisher holds the rights and waives them.
Committing a copyrighted standard's control statements under it asserts rights
the project does not hold, for every downstream fork and mirror.

The control has to live here rather than in the publish path. `git push` is a
publication event and it fires before any `tract publish-*` command runs, so a
filter on the publish path guards the narrow channel and leaves the wide one
open. That was the actual defect: the design specified an ISO filter on the
publish path while `data/processed/` was tracked and the repo was public.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tract.config import PROCESSED_DIR, RESTRICTED_FRAMEWORK_IDS

# RESTRICTED_FRAMEWORK_IDS lives in tract/config.py, not here. The merge step
# reads the same constant to decide what stays out of the tracked corpus, and
# a second copy in this file would let the writer and the gate disagree about
# which frameworks are licensed.

__all__ = ["RESTRICTED_FRAMEWORK_IDS"]

# Below this, a description is a section title rather than a control statement.
# ISO's tracked stub carried 93 titles at a 28-character median; its real Annex
# A statements run to a 138-character median.
_TITLE_LENGTH_CEILING: int = 60


def _tracked_files(scope: str = "data/processed") -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", scope],
        capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    return {line for line in result.stdout.splitlines() if line}


def _normalised_source() -> str:
    """The ISO Annex A *control statements*, normalised for comparison.

    Deliberately excludes the title column. Section titles like "Privacy and
    protection of personal identifiable information (PII)" reach this project
    through OpenCRE's public link dump and are already tracked in
    data/training/hub_links*. They are not the normative text at issue. What
    this repository must not dedicate to the public domain is the requirement
    statement, which the source table marks with a leading "Control".
    """
    import re

    path = (
        Path(__file__).resolve().parent.parent
        / "data" / "raw" / "frameworks" / "iso_27001"
        / "ISO_IEC_27001_2022_en.md"
    )
    if not path.exists():
        return ""
    statements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not re.match(r"^\d+\.\d+$", cells[0]):
            continue
        statements.append(cells[2])
    text = re.sub(r"\s+", " ", "   ".join(statements))
    return re.sub(r"([a-z])\s+-\s+([a-z])", r"\1\2", text).lower()


def test_no_verbatim_licensed_statement_anywhere_in_the_tree() -> None:
    """No tracked file may quote a licensed standard's control statement.

    The first version of this gate scanned only data/processed, so it could not
    see licensed text that reached git through a different door. It did not:
    a later change pinned ISO Annex A statements into tracked test fixtures and
    assertions, and this gate reported clean. A control that guards one channel
    while another stays open is the failure it was written to prevent.
    """
    import re

    source = _normalised_source()
    if not source:
        pytest.skip("raw ISO source absent in this checkout")

    root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, str]] = []
    for rel in sorted(_tracked_files(".")):
        path = root / rel
        if path.suffix not in {".py", ".md", ".json", ".jsonl", ".txt", ".csv"}:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # Any run of 60+ characters that appears verbatim in the standard is a
        # quotation, not a coincidence.
        for candidate in re.findall(r"[A-Za-z][^\"'`|\n]{59,}", body):
            probe = re.sub(r"\s+", " ", candidate.replace("[...]", "")).strip().lower()
            if len(probe) >= 60 and probe[:60] in source:
                offenders.append((rel, probe[:70]))
                break

    assert not offenders, (
        f"{len(offenders)} tracked file(s) quote ISO 27001 control statements "
        f"verbatim: {offenders[:5]}. This repository is CC0, which asserts the "
        f"publisher holds the rights and waives them. Move the text to a "
        f"gitignored path and skip when absent, or replace it with synthetic "
        f"rows that reproduce the same structure."
    )


def test_restricted_framework_files_are_not_tracked() -> None:
    """The per-framework JSON for a licensed source must be gitignored."""
    tracked = _tracked_files()
    for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
        path = f"data/processed/frameworks/{framework_id}.json"
        assert path not in tracked, (
            f"{path} is tracked by git. This repository is CC0, so committing "
            f"{framework_id} control statements dedicates licensed text to the "
            f"public domain. Run: git rm --cached {path}"
        )


def test_merged_corpus_carries_no_restricted_prose() -> None:
    """A tracked all_controls.json must not carry a restricted source's prose.

    The merged corpus is a build artifact that concatenates every per-framework
    file. Gitignoring the ISO file alone does not help if the merge output is
    tracked and contains the same text.
    """
    merged = PROCESSED_DIR / "all_controls.json"
    if merged.name not in {Path(p).name for p in _tracked_files()}:
        pytest.skip("all_controls.json is not tracked; nothing to enforce")
    if not merged.exists():
        pytest.skip("all_controls.json not present in this checkout")

    data = json.loads(merged.read_text(encoding="utf-8"))
    offenders: list[tuple[str, str, int]] = []
    for framework in data.get("frameworks", []):
        if framework.get("framework_id") not in RESTRICTED_FRAMEWORK_IDS:
            continue
        for control in framework.get("controls", []):
            description = (control.get("description") or "").strip()
            title = (control.get("title") or "").strip()
            # Prose, not a restated title. Both tests must hold: the stub form
            # copies the title verbatim and is short.
            if description != title and len(description) > _TITLE_LENGTH_CEILING:
                offenders.append(
                    (framework["framework_id"], control.get("control_id", "?"),
                     len(description))
                )

    assert not offenders, (
        f"{len(offenders)} restricted-license control statements are inside a "
        f"tracked all_controls.json, e.g. {offenders[:3]}. Build the merged "
        f"corpus into the gitignored licensed overlay instead, or exclude "
        f"restricted frameworks from the tracked artifact."
    )
