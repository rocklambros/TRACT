"""The frozen echo partition keys on something that is not unique per item.

`frozen_echo_keys` returned a SET of `(framework_name, section_id)`. Membership
was tested by key, so every item sharing a key received the same verdict: if any
one of them was echo, all of them were. It is now `frozen_echo_indices`.

MEASURED on the live 147-item corpus: 5 keys collide, covering 13 items (8.8%).
One of those groups -- ('NIST AI 100-2', 'Sec. 2.2.4'), three items with echo
status [False, True, False] -- has mixed status, so **2 non-echo items are
forced into the echo stratum**. Premortem finding B4 put the collision at 8.8%
today and 24.4% under the proposed roster, which is right; its impact line
("a partition mislabelling a quarter of the corpus") conflates the collision
rate with the mislabel rate, which today is 1.4%.

Small, and still worth fixing: the binding side condition is evaluated on the
non-echo stratum at n=91, and two items is not nothing at that denominator.
Under the proposed roster the collision rate triples and the expected mislabel
count grows with it.

WHY THE KEY WAS CHOSEN. `apply_prose_to_corpus` replaces `control_text`, so a
key containing the text cannot be matched across the title and prose forms that
the frozen partition unions. `(framework, section_id)` survives that
substitution -- it just is not unique.

THE FIX. That same function guarantees "same items, same order, same ground
truth", and the guarantee is load-bearing enough to be documented in its
docstring. So the corpus INDEX is an item-exact identity that survives prose
substitution, which is what the partition needs.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from tract.config import PROCESSED_DIR
from tract.hierarchy import CREHierarchy
from tract.training.echo import frozen_echo_indices, is_echo


@pytest.fixture(scope="module")
def pieces():  # type: ignore[no-untyped-def]
    from scripts.phase0.common import (
        AI_FRAMEWORK_NAMES,
        build_evaluation_corpus,
        load_curated_links,
    )

    corpus = build_evaluation_corpus(load_curated_links(), AI_FRAMEWORK_NAMES, {})
    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")
    return corpus, hierarchy


class TestTheCollisionIsRealAndMeasured:
    """Pin the measurement, so the fix is not defended by a number nobody checked."""

    def test_the_legacy_key_collides_on_the_live_corpus(self, pieces) -> None:  # type: ignore[no-untyped-def]
        corpus, _ = pieces
        groups: dict[tuple[str, str], int] = defaultdict(int)
        for item in corpus:
            groups[(item.framework_name, item.section_id)] += 1
        collided = sum(n for n in groups.values() if n > 1)
        assert collided == 13, (
            f"{collided} items share a (framework, section_id) key, not the 13 "
            "this suite was written against. Re-measure the mislabel count "
            "before trusting the partition."
        )

    def test_at_least_one_colliding_group_has_mixed_echo_status(
        self, pieces
    ) -> None:  # type: ignore[no-untyped-def]
        """Without a mixed group the collision would be harmless.

        This is the test that makes the fix load-bearing rather than tidy.
        """
        corpus, hierarchy = pieces
        names = {h: n.name for h, n in hierarchy.hubs.items()}
        groups: dict[tuple[str, str], set[bool]] = defaultdict(set)
        for item in corpus:
            groups[(item.framework_name, item.section_id)].add(
                is_echo(item.control_text, names.get(item.ground_truth_hub_id, ""))
            )
        mixed = [k for k, v in groups.items() if len(v) > 1]
        assert mixed, (
            "No colliding key has mixed echo status, so the collision "
            "currently mislabels nothing and this suite is asserting a "
            "property that has stopped existing."
        )


class TestTheFrozenPartitionIsItemExact:
    def test_indices_distinguish_items_that_share_a_key(self, pieces) -> None:  # type: ignore[no-untyped-def]
        """The load-bearing assertion.

        Two NIST AI 100-2 'Sec. 2.2.4' items are not echo and one is. An
        item-exact partition contains the one and not the other two.
        """
        corpus, hierarchy = pieces
        names = {h: n.name for h, n in hierarchy.hubs.items()}
        frozen = frozen_echo_indices(corpus, hierarchy, None, None)

        by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, item in enumerate(corpus):
            by_key[(item.framework_name, item.section_id)].append(idx)

        checked = 0
        for indices in by_key.values():
            statuses = {
                idx: is_echo(
                    corpus[idx].control_text,
                    names.get(corpus[idx].ground_truth_hub_id, ""),
                )
                for idx in indices
            }
            if len(set(statuses.values())) < 2:
                continue
            checked += 1
            for idx, echo in statuses.items():
                assert (idx in frozen) == echo, (
                    f"item {idx} is echo={echo} but the frozen partition "
                    f"says {idx in frozen}; the key collapsed it onto a "
                    "sibling sharing its (framework, section_id)"
                )
        assert checked, "no mixed group was exercised"

    def test_it_returns_indices_not_keys(self, pieces) -> None:  # type: ignore[no-untyped-def]
        corpus, hierarchy = pieces
        frozen = frozen_echo_indices(corpus, hierarchy, None, None)
        assert all(isinstance(i, int) for i in frozen)
        assert all(0 <= i < len(corpus) for i in frozen)

    def test_the_partition_is_not_degenerate(self, pieces) -> None:  # type: ignore[no-untyped-def]
        """Guards the guard: an empty or total partition would pass the above."""
        corpus, hierarchy = pieces
        frozen = frozen_echo_indices(corpus, hierarchy, None, None)
        assert 0 < len(frozen) < len(corpus)


class TestTheRunArtifactDoesNotCarryTheBindingPartition:
    """`lexical_overlap` in aggregate_metrics.json is not the side condition.

    CAMPAIGN3.md Section 3 makes the non-echo stratum a BINDING side condition,
    and Section 1.3 records the frozen partition as echo 56 / non-echo 91,
    retiring the earlier n=109 and n=98 figures as "computed against anchors
    that no longer exist".

    But `compute_lexical_overlap` in tract/training/orchestrate.py computes its
    split per arm from each arm's own truncated anchors -- its docstring says so
    plainly, and says a cross-arm comparison "belongs to analysis, not to the
    per-run record". It is right about itself. The problem is that the number it
    writes is called `hit_at_1_non_echo`, sits in the run artifact, and is
    n=109: the retired figure, under the name of the binding one.

    Nothing in production computes the frozen partition. `frozen_echo_indices`
    is imported by tests alone. This is the same shape as the Section 3 gate
    probability that no library code computed.

    These tests do not fix that. They stop the retired number being read as the
    binding one.
    """

    RUN = "c2r_TEST_A3_prose_sw_qwen06b"
    FROZEN_NON_ECHO = 91

    def test_the_committed_artifact_carries_the_retired_denominator(self) -> None:
        import json

        from tract.config import PHASE1B_RESULTS_DIR

        path = PHASE1B_RESULTS_DIR / self.RUN / "aggregate_metrics.json"
        if not path.is_file():
            pytest.skip(f"{path} absent")
        overlap = json.loads(path.read_text(encoding="utf-8"))["lexical_overlap"]
        assert overlap["n_non_echo"] != self.FROZEN_NON_ECHO, (
            "The run artifact's n_non_echo now equals the frozen partition. "
            "Either the frozen partition was wired into the aggregate -- in "
            "which case delete this test and say so in CAMPAIGN3.md -- or two "
            "different quantities have silently converged."
        )

    def test_no_production_module_computes_the_frozen_partition(self) -> None:
        """Goes red when someone wires it up, which is the desired outcome.

        At that point the side condition becomes computable from an artifact
        rather than from an analysis script, and CAMPAIGN3.md Section 1.3
        should be updated to say which field carries it.
        """

        from tract.config import PROJECT_ROOT

        callers: list[str] = []
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if rel.startswith(("tests/", ".venv/", "wandb/", "build/")):
                continue
            if rel == "tract/training/echo.py":
                continue
            if "frozen_echo_indices" in path.read_text(
                encoding="utf-8", errors="replace"
            ):
                callers.append(rel)
        assert not callers, (
            "The frozen partition now has a production caller: "
            f"{callers}. Update CAMPAIGN3.md Section 1.3 to name the field "
            "that carries the binding non-echo figure, and remove this test."
        )
