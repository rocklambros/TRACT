"""Tests for fold scoping in scripts/phase1b/runpod_parallel.py.

A canary exists to validate the machinery for the price of one pod. Before
select_pod_configs, provision() always created all five, so the only way to
test the pipeline end to end was to pay for the whole fleet.
"""
from __future__ import annotations

import pytest

from scripts.phase1b.runpod_parallel import (
    FOLD_FRAMEWORKS,
    POD_CONFIGS,
    select_pod_configs,
)


def test_no_filter_is_the_full_fleet() -> None:
    assert select_pod_configs() == list(POD_CONFIGS)
    assert select_pod_configs([]) == list(POD_CONFIGS)
    assert len(select_pod_configs(None)) == len(FOLD_FRAMEWORKS)


def test_a_canary_is_one_pod() -> None:
    configs = select_pod_configs(["OWASP Top10 for LLM"])
    assert len(configs) == 1
    assert configs[0]["role"] == "OWASP Top10 for LLM"


def test_pod_names_are_stable_under_filtering() -> None:
    """A pod name is tied to the framework, not to its filtered position.

    If names were assigned by position in the filtered list, a canary pod and
    the same fold in a later full run would carry different names, and reap's
    orphan detection matches on name.
    """
    full = {c["role"]: c["name"] for c in select_pod_configs()}
    for framework in FOLD_FRAMEWORKS:
        scoped = select_pod_configs([framework])
        assert scoped[0]["name"] == full[framework]


def test_a_subset_keeps_canonical_order() -> None:
    configs = select_pod_configs(["OWASP Top10 for ML", "MITRE ATLAS"])
    assert [c["role"] for c in configs] == ["MITRE ATLAS", "OWASP Top10 for ML"]


def test_an_unknown_fold_is_refused() -> None:
    """A typo would otherwise provision nothing and report success."""
    with pytest.raises(ValueError, match="Unknown fold"):
        select_pod_configs(["MITRE ATLAS", "Mitre Atlas"])


def test_duplicates_do_not_double_provision() -> None:
    configs = select_pod_configs(["MITRE ATLAS", "MITRE ATLAS"])
    assert len(configs) == 1
