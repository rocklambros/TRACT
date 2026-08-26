"""merge_all_controls must not stamp a clock into all_controls.json.

The merged artifact's sha256 is recorded per fold and compared across folds
by orchestrate.load_fold_results. A date stamp makes two folds run either
side of midnight UTC disagree about identical content.
"""

from __future__ import annotations

import inspect


def test_merge_does_not_read_the_clock() -> None:
    import parsers.merge_all_controls as merge

    source = inspect.getsource(merge)
    assert "date.today" not in source and "datetime.now" not in source, (
        "merge_all_controls must derive generated_date from its inputs"
    )
