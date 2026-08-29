"""Tests for the single-pod smoke-test driver.

Only the provisioning decision is exercised. Bootstrap, rsync and teardown are
thin wrappers over `runpod_parallel` helpers that have their own tests, and
nothing here creates a pod: `create_pod` is replaced in every test.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.phase1b import smoke_on_pod


class _Capacity(RuntimeError):
    """Stands in for a RunPod capacity error."""


@pytest.fixture(autouse=True)
def _never_really_provision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that reaches the real RunPod API is a test that spends money."""
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("create_pod must be patched in every test")
    monkeypatch.setattr(smoke_on_pod, "create_pod", _boom)
    monkeypatch.setattr(
        smoke_on_pod, "is_capacity_error", lambda exc: isinstance(exc, _Capacity),
    )


def test_provision_falls_through_capacity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first candidate having no stock must not end the run."""
    monkeypatch.setattr(
        smoke_on_pod, "rank_available_gpus",
        lambda **_k: [("NVIDIA A100", 1.9), ("NVIDIA L40S", 0.9)],
    )
    tried: list[str] = []

    def _create(gpu_type_id: str, name: str) -> dict[str, Any]:
        tried.append(gpu_type_id)
        if gpu_type_id == "NVIDIA A100":
            raise _Capacity("no capacity")
        return {"pod_id": "p1", "ip": "1.2.3.4", "port": 22,
                "gpu_type": gpu_type_id, "cloud_type": "SECURE"}

    monkeypatch.setattr(smoke_on_pod, "create_pod", _create)
    pod = smoke_on_pod._provision()

    assert tried == ["NVIDIA A100", "NVIDIA L40S"]
    assert pod["pod_id"] == "p1"
    # The driver's teardown and reporting read both of these off the pod dict.
    assert pod["role"] == "agentic-smoke"
    assert pod["usd_per_hour"] == 0.9


def test_provision_reraises_non_capacity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An auth or validation failure must stop, not walk the whole catalogue."""
    monkeypatch.setattr(
        smoke_on_pod, "rank_available_gpus",
        lambda **_k: [("NVIDIA A100", 1.9), ("NVIDIA L40S", 0.9)],
    )
    tried: list[str] = []

    def _create(gpu_type_id: str, name: str) -> dict[str, Any]:
        tried.append(gpu_type_id)
        raise PermissionError("bad api key")

    monkeypatch.setattr(smoke_on_pod, "create_pod", _create)
    with pytest.raises(PermissionError):
        smoke_on_pod._provision()
    assert tried == ["NVIDIA A100"], "should not have tried the second candidate"


def test_provision_refuses_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_on_pod, "rank_available_gpus", lambda **_k: [])
    with pytest.raises(RuntimeError, match="nothing is billing"):
        smoke_on_pod._provision()


def test_provision_refuses_when_every_candidate_is_out_of_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_on_pod, "rank_available_gpus",
        lambda **_k: [("NVIDIA A100", 1.9), ("NVIDIA L40S", 0.9)],
    )

    def _create(gpu_type_id: str, name: str) -> dict[str, Any]:
        raise _Capacity("no capacity")

    monkeypatch.setattr(smoke_on_pod, "create_pod", _create)
    with pytest.raises(RuntimeError, match="capacity error"):
        smoke_on_pod._provision()


def test_price_ceiling_is_below_the_fleet_budget() -> None:
    """One pod doing inference should never cost fleet money.

    The driver bypasses runpod_parallel.provision, which is where the campaign
    budget gate lives, so this ceiling and the wall cap are the only limits on
    what a single run can spend.
    """
    assert smoke_on_pod.MAX_USD_PER_HOUR <= 3.0
    assert smoke_on_pod.WALL_CAP_S <= 3600
    # Named outside the tract-p1b-* family the reaper sweeps, deliberately.
    assert not smoke_on_pod.POD_NAME.startswith("tract-p1b")
