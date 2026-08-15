"""Tests for CRE-branch stratified sampling in tract/training/data.py.

72.1% of training links point at "Technical application security controls"
and 3.3% at the "Cross-cutting concerns" threat branch. None of CAPEC's 702
adversary-as-subject anchors point at a threat hub, so the model learns
"attack narrative -> the control that stops it" -- right for CAPEC, wrong for
MITRE ATLAS techniques, measured at -29.4 hit@1 points on that stratum.
"""
from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("torch")
pytest.importorskip("datasets")

import numpy as np  # noqa: E402
from datasets import Dataset  # noqa: E402

from tract.training.data import HubAwareTemperatureSampler  # noqa: E402


def _dataset(branches: list[str]) -> Dataset:
    return Dataset.from_dict({
        "anchor": [f"anchor {i}" for i in range(len(branches))],
        "positive": [f"hub {i}" for i in range(len(branches))],
        "hub_id": [f"h{i}" for i in range(len(branches))],
        "is_ai": [False] * len(branches),
        "anchor_key": [f"k{i}" for i in range(len(branches))],
        "branch": branches,
    })


# 90/10 skew, the shape of the real distribution.
SKEWED = ["controls"] * 90 + ["threat"] * 10


class TestStratifiedOrdering:

    def _order(self, temp: float) -> list[int]:
        s = HubAwareTemperatureSampler(
            _dataset(SKEWED), batch_size=8, strata_temperature=temp, seed=7,
        )
        return s._order_by_strata(np.random.default_rng(7)) if temp > 0 else []

    def test_every_example_appears_exactly_once(self) -> None:
        """Ordering, not resampling: no example duplicated or dropped."""
        order = self._order(3.0)
        assert sorted(order) == list(range(100))

    def test_flattening_pulls_the_rare_branch_forward(self) -> None:
        """The point of the knob: rare-branch examples reach early batches."""
        flat = self._order(4.0)
        first20 = [SKEWED[i] for i in flat[:20]]
        # Natural rate would put ~2 threat items in the first 20.
        assert Counter(first20)["threat"] > 2

    def test_higher_temperature_flattens_more(self) -> None:
        mild = [SKEWED[i] for i in self._order(1.2)[:20]].count("threat")
        strong = [SKEWED[i] for i in self._order(6.0)[:20]].count("threat")
        assert strong >= mild

    def test_ordering_is_deterministic_for_a_seed(self) -> None:
        """A run has to be reproducible from its recorded seed."""
        a = HubAwareTemperatureSampler(
            _dataset(SKEWED), batch_size=8, strata_temperature=3.0, seed=7,
        )._order_by_strata(np.random.default_rng(11))
        b = HubAwareTemperatureSampler(
            _dataset(SKEWED), batch_size=8, strata_temperature=3.0, seed=7,
        )._order_by_strata(np.random.default_rng(11))
        assert a == b

    def test_a_single_stratum_is_handled(self) -> None:
        s = HubAwareTemperatureSampler(
            _dataset(["only"] * 10), batch_size=4, strata_temperature=3.0,
        )
        assert sorted(s._order_by_strata(np.random.default_rng(1))) == list(range(10))


class TestDisabledByDefault:

    def test_zero_temperature_keeps_the_original_behaviour(self) -> None:
        """An existing run must reproduce exactly."""
        s = HubAwareTemperatureSampler(_dataset(SKEWED), batch_size=8)
        assert s.strata_temperature == 0.0
        batches = list(iter(s))
        assert sorted(i for b in batches for i in b) == list(range(100))

    def test_batches_never_repeat_a_hub(self) -> None:
        """Balancing must not break the MNRL false-negative guard."""
        s = HubAwareTemperatureSampler(
            _dataset(SKEWED), batch_size=8, strata_temperature=4.0,
        )
        for batch in iter(s):
            hubs = [s.hub_ids[i] for i in batch]
            assert len(hubs) == len(set(hubs))


class TestConfigReachesTheSampler:
    """The sampler is handed to the trainer as a CLASS, so config values can
    only arrive through the class overrides. sampling_temperature was recorded
    in every run record and never reached it."""

    def teardown_method(self) -> None:
        HubAwareTemperatureSampler.clear_metadata()

    def test_overrides_win_over_constructor_defaults(self) -> None:
        HubAwareTemperatureSampler.set_metadata(
            hub_ids=[f"h{i}" for i in range(100)],
            is_ai=[False] * 100,
            anchor_keys=[f"k{i}" for i in range(100)],
            strata=SKEWED,
            temperature=0.5,
            strata_temperature=3.5,
        )
        s = HubAwareTemperatureSampler(_dataset(SKEWED), batch_size=8)
        assert s.temperature == 0.5
        assert s.strata_temperature == 3.5

    def test_clear_metadata_resets_the_overrides(self) -> None:
        HubAwareTemperatureSampler.set_metadata(
            hub_ids=["h"], is_ai=[False], strata=["x"],
            temperature=0.5, strata_temperature=3.5,
        )
        HubAwareTemperatureSampler.clear_metadata()
        s = HubAwareTemperatureSampler(_dataset(SKEWED), batch_size=8)
        assert s.temperature == 2.0
        assert s.strata_temperature == 0.0
