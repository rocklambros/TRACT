"""The regeneration script must not talk an operator into a short corpus.

`assert_corpus_matches_training_links` refuses to train when the corpus on disk
is not the one the training links were built from. That refusal gates
provision, run_folds and run_fold.py as of 2026-08-26.

There are two ways to satisfy a refusal. Stage the licensed overlay, which is
the corpus the sidecar records and the one the campaign is measured on. Or
regenerate the sidecar against whatever this machine happens to hold, which
makes the check pass and trains on 4,048 of 4,389 links while reporting the
same shape of output.

The script offered the second. Its `--check` mode ended with "Re-run without
--check", which is correct when a parser moved under the corpus and precisely
wrong on a machine that is simply missing the overlay -- and a machine missing
the overlay is the normal state of a fresh clone, which is what the Jetson is.

So the guard is on the two signals that separate those cases: the corpus PATH
the sidecar records, and whether the training set would shrink.
"""
from __future__ import annotations

from scripts.build_training_links import refuse_reason

OVERLAY = "data/processed/licensed/all_controls.json"
TRACKED = "data/processed/all_controls.json"


def _meta(path: str, n_links: int, digest: str = "a" * 64) -> dict[str, object]:
    return {
        "corpus_path": path,
        "corpus_sha256": digest,
        "curated_links_sha256": "b" * 64,
        "n_links": n_links,
        "output_sha256": "c" * 64,
    }


class TestRefusesToRegenerateAgainstADifferentCorpus:

    def test_a_missing_overlay_is_refused(self) -> None:
        """The Jetson's normal failure, and the one that must not be written."""
        reason = refuse_reason(
            before=_meta(OVERLAY, 4389), after=_meta(TRACKED, 4048, "d" * 64),
        )
        assert reason is not None
        assert OVERLAY in reason
        # The message has to name the fix, because the operator is standing in
        # front of a refusal looking for the shortest way past it.
        assert "stage" in reason.lower()

    def test_a_shrinking_training_set_is_refused(self) -> None:
        """Same corpus path, fewer links: something upstream lost anchors."""
        reason = refuse_reason(
            before=_meta(OVERLAY, 4389), after=_meta(OVERLAY, 4048, "d" * 64),
        )
        assert reason is not None
        assert "4048" in reason and "4389" in reason

    def test_a_moved_corpus_at_the_same_path_is_allowed(self) -> None:
        """The legitimate case: a parser moved, the corpus is still the overlay.

        This is what happened on 2026-08-26, when two phantom aiuc_1 controls
        left the corpus and the training set did not change. Refusing this
        would make the script useless for the job it exists to do.
        """
        assert refuse_reason(
            before=_meta(OVERLAY, 4389), after=_meta(OVERLAY, 4389, "d" * 64),
        ) is None

    def test_a_growing_training_set_is_allowed(self) -> None:
        """More anchors is the direction the corpus work is trying to go."""
        assert refuse_reason(
            before=_meta(OVERLAY, 4389), after=_meta(OVERLAY, 4400, "d" * 64),
        ) is None

    def test_a_first_run_with_no_sidecar_is_allowed(self) -> None:
        """Nothing to contradict, so nothing to refuse."""
        assert refuse_reason(before={}, after=_meta(OVERLAY, 4389)) is None
