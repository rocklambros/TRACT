"""The exposure partition, and the numbers Amendment 1 commits to.

The partition must be a property of the corpus and the gold links ONLY. If
anything in it could move with the arm being measured, it is a post-hoc stratum
of the kind this project has withdrawn headlines over.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis.gate2_strata import build_strata
from tract.config import PHASE2C_GATE2_EVAL_FRAMEWORKS

CORPUS = Path("data/training/hub_links_bridge.r2.jsonl")
STRATA = Path("results/phase2c/gate2_strata.json")


@pytest.fixture(scope="module")
def strata():  # type: ignore[no-untyped-def]
    if not CORPUS.is_file():
        pytest.skip(f"{CORPUS} is gitignored and absent on this machine")
    return build_strata(CORPUS)


class TestTheRealNumbers:
    """Pinned, because the design was rebuilt around them."""

    def test_seventy_four_items_over_sixty_two_scored_hubs(self, strata) -> None:  # type: ignore[no-untyped-def]
        assert strata.n_items == 74
        assert strata.n_scored_hubs == 62, (
            "hit@1 is credited against valid_hub_ids, not ground_truth_hub_id. "
            "The draft plan cited 32, which is the ground-truth count and not "
            "the scored set."
        )

    def test_the_partition_is_27_and_47(self, strata) -> None:  # type: ignore[no-untyped-def]
        assert len(strata.exposed_item_keys) == 27
        assert len(strata.unexposed_item_keys) == 47
        assert len(strata.exposed_item_keys) + len(
            strata.unexposed_item_keys
        ) == strata.n_items

    def test_the_exposed_stratum_spans_two_frameworks(self, strata) -> None:  # type: ignore[no-untyped-def]
        """ETSI was added for exactly this.

        Without it every exposed item is ENISA, and a single-framework
        treatment stratum makes the result a claim about ENISA.
        """
        assert strata.n_exposed_by_framework == {"ENISA": 18, "ETSI": 9}

    def test_biml_is_the_negative_control(self, strata) -> None:  # type: ignore[no-untyped-def]
        """17 items, 23% of the eval, that no bridge link can reach.

        The pre-registration originally scored ENISA+BIML alone, where this was
        34% of the denominator contributing nothing but noise.
        """
        assert strata.negative_control_frameworks == ["BIML"]
        assert strata.n_items_by_framework["BIML"] == 17

    def test_a_third_of_the_corpus_is_invisible_to_this_eval(
        self, strata
    ) -> None:  # type: ignore[no-untyped-def]
        """Stated rather than discovered afterwards.

        The corpus de-orphans 31 hubs; 8 of them are not in the scored set, so
        that supervision cannot show up in this measurement at any effect size.
        """
        assert strata.n_bridge_hubs == 31
        assert strata.n_bridge_hubs_invisible_to_eval == 8


class TestThePartitionCannotMoveWithAnArm:
    def test_it_depends_on_no_trained_artifact(self, strata) -> None:  # type: ignore[no-untyped-def]
        """Recomputing is idempotent, and needs no results directory."""
        again = build_strata(CORPUS)
        assert again.exposed_item_keys == strata.exposed_item_keys
        assert again.unexposed_item_keys == strata.unexposed_item_keys

    def test_keys_are_digests_not_prose(self, strata) -> None:  # type: ignore[no-untyped-def]
        """ETSI is restricted; the partition is committed. It carries no text."""
        for key in strata.exposed_item_keys[:5]:
            assert len(key) == 64
            assert all(c in "0123456789abcdef" for c in key)

    def test_the_two_strata_are_disjoint(self, strata) -> None:  # type: ignore[no-untyped-def]
        assert not set(strata.exposed_item_keys) & set(
            strata.unexposed_item_keys
        )


class TestTheCommittedFile:
    def test_it_exists_and_matches_a_recomputation(self, strata) -> None:  # type: ignore[no-untyped-def]
        if not STRATA.is_file():
            pytest.skip("strata file not generated on this machine")
        payload = json.loads(STRATA.read_text(encoding="utf-8"))
        assert payload["exposed_item_keys"] == strata.exposed_item_keys, (
            "the committed partition disagrees with a fresh computation, so "
            "one of the corpus or the file has moved since it was written"
        )

    def test_it_records_the_corpus_it_was_built_from(self) -> None:
        if not STRATA.is_file() or not CORPUS.is_file():
            pytest.skip("artifacts absent on this machine")
        payload = json.loads(STRATA.read_text(encoding="utf-8"))
        assert payload["corpus_sha256"] == hashlib.sha256(
            CORPUS.read_bytes()
        ).hexdigest()

    def test_it_names_the_eval_and_the_firewall(self) -> None:
        if not STRATA.is_file():
            pytest.skip("strata file not generated on this machine")
        payload = json.loads(STRATA.read_text(encoding="utf-8"))
        assert set(payload["eval_frameworks"]) == set(
            PHASE2C_GATE2_EVAL_FRAMEWORKS
        )
        assert len(payload["held_out_frameworks"]) == 8
