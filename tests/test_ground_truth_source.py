"""import-ground-truth and training must read the same link file.

cli.py fed the uncurated hub_links_by_framework.json (4,406 links) while
training read hub_links_curated.jsonl (4,405, filtered to 4,127), a 279-edge
divergence between crosswalk.db and the model.
"""

from __future__ import annotations

import inspect


def test_import_ground_truth_reads_the_curated_link_file() -> None:
    from tract import cli

    source = inspect.getsource(cli._cmd_import_ground_truth)
    assert "hub_links_by_framework_curated.json" in source, (
        "import-ground-truth must read the curated grouped file, which sits "
        "unused in the same directory as the uncurated one it currently reads"
    )
