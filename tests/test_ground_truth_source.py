"""import-ground-truth and training must read the same link file.

cli.py fed the uncurated hub_links_by_framework.json (4,406 links) while
training read hub_links_curated.jsonl (4,405, filtered to 4,127), a 279-edge
divergence between crosswalk.db and the model.

The check is on edge counts, not on the text of the source file. Grepping the
handler for a filename passes the moment the string appears anywhere in it,
including in a comment, and says nothing about whether the two readers end up
with the same edges.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tract.config import TRAINING_DIR

GROUPED = TRAINING_DIR / "hub_links_by_framework_curated.json"
FLAT = TRAINING_DIR / "hub_links_curated.jsonl"
UNCURATED = TRAINING_DIR / "hub_links_by_framework.json"


def _grouped_edges(path: Path) -> set[tuple[str, str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (link["cre_id"], link["framework_id"], link["section_id"])
        for links in data.values()
        for link in links
    }


def _flat_edges(path: Path) -> set[tuple[str, str, str]]:
    return {
        (link["cre_id"], link["framework_id"], link["section_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for link in [json.loads(line)]
    }


def test_the_import_handler_names_the_curated_grouped_file() -> None:
    """The cheap half: the handler must not point at the uncurated file."""
    from tract import cli

    source = inspect.getsource(cli._cmd_import_ground_truth)
    assert "hub_links_by_framework_curated.json" in source
    assert '"hub_links_by_framework.json"' not in source


@pytest.mark.skipif(
    not (GROUPED.exists() and FLAT.exists()),
    reason="curated link files are not present in this checkout",
)
def test_both_readers_see_the_same_edge_set() -> None:
    """The real half: the two files must agree edge for edge.

    crosswalk.db is built from the grouped file and the model is trained from
    the flat one. A divergence between them is a crosswalk that claims links
    the model never saw, which no count of either file alone reveals.
    """
    grouped = _grouped_edges(GROUPED)
    flat = _flat_edges(FLAT)

    assert len(grouped) == len(flat)
    assert grouped == flat


@pytest.mark.skipif(
    not (GROUPED.exists() and UNCURATED.exists()),
    reason="link files are not present in this checkout",
)
def test_the_curated_file_is_not_the_uncurated_one() -> None:
    """Guards the test above against passing on two copies of one file.

    If curation ever became a no-op, every assertion here would still pass
    while the divergence this test exists for went unnoticed.
    """
    assert _grouped_edges(GROUPED) != _grouped_edges(UNCURATED)
