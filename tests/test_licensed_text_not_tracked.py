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

from tract.config import PROCESSED_DIR

# Frameworks whose source text is licensed such that redistribution under CC0
# would assert rights the project does not hold. Add here when a licensed
# source is ingested; the parser still writes the file, git just never sees it.
RESTRICTED_FRAMEWORK_IDS: frozenset[str] = frozenset({"iso_27001"})

# Below this, a description is a section title rather than a control statement.
# ISO's tracked stub carried 93 titles at a 28-character median; its real Annex
# A statements run to a 138-character median.
_TITLE_LENGTH_CEILING: int = 60


def _tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "data/processed"],
        capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    return {line for line in result.stdout.splitlines() if line}


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
