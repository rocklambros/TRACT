"""The SECURE-tier restriction guarded one of five pod-creation call sites.

`_require_secure_cloud()` lives in `scripts/phase1b/runpod_parallel.py` and is
consulted at exactly one place -- the fold-fleet provision. Four other call
sites take `create_pod`'s default `CLOUD_TYPE_PREFERENCE = (SECURE, COMMUNITY)`:

    scripts/phase1c/runpod_retrain.py     the Phase 1C retrain
    scripts/phase1b/smoke_on_pod.py       the pre-registered agentic smoke test
    scripts/phase1b/probe_on_pod.py       the domain-shortcut probe
    scripts/phase0/runpod_orchestrate.py  the Phase 0 fleet

Three of those rsync `PROJECT_ROOT` wholesale, and their exclude lists do not
name `data/processed/licensed`. `tract/staleness.py` records that the overlay
"is staged on every real run, because provision refuses on a corpus mismatch" --
so whenever the restriction matters, it is those four that are unguarded.

Two of them are campaign instruments, not adjacent tooling: the domain-shortcut
probe is the measurement CAMPAIGN3 §0 rests its funding decision on, and the
agentic smoke test is pre-registered in Campaign 2.

A control applied at one caller is not a control. It belongs at the boundary
where the pod is created, which is what these pin.
"""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestTheGuardLivesAtTheBoundary:

    def test_create_pod_consults_the_restriction_itself(self) -> None:
        from scripts.phase0 import runpod_provision
        assert hasattr(runpod_provision, "require_secure_cloud"), (
            "the SECURE restriction must be reachable from the module that "
            "creates pods, or every caller has to remember it"
        )

    def test_a_caller_cannot_widen_the_tier_while_the_overlay_is_staged(
        self, monkeypatch,
    ) -> None:
        from scripts.phase0 import runpod_provision
        monkeypatch.setattr(runpod_provision, "require_secure_cloud",
                            lambda: True)
        allowed = runpod_provision._effective_cloud_types(
            runpod_provision.CLOUD_TYPE_PREFERENCE)
        assert allowed == (runpod_provision.CLOUD_TYPE_SECURE,), (
            "a caller passing the permissive default must still be narrowed "
            "to SECURE when the licensed overlay is on disk"
        )

    def test_an_explicit_community_request_is_also_narrowed(
        self, monkeypatch,
    ) -> None:
        from scripts.phase0 import runpod_provision
        monkeypatch.setattr(runpod_provision, "require_secure_cloud",
                            lambda: True)
        allowed = runpod_provision._effective_cloud_types(
            (runpod_provision.CLOUD_TYPE_COMMUNITY,))
        assert allowed == (runpod_provision.CLOUD_TYPE_SECURE,)

    def test_a_public_corpus_run_is_not_obstructed(self, monkeypatch) -> None:
        from scripts.phase0 import runpod_provision
        monkeypatch.setattr(runpod_provision, "require_secure_cloud",
                            lambda: False)
        allowed = runpod_provision._effective_cloud_types(
            runpod_provision.CLOUD_TYPE_PREFERENCE)
        assert allowed == runpod_provision.CLOUD_TYPE_PREFERENCE


class TestEveryCallSiteIsCovered:
    """Enumerated rather than sampled: the defect was a caller nobody listed."""

    CALL_SITES = (
        "scripts/phase1b/probe_on_pod.py",
        "scripts/phase1b/smoke_on_pod.py",
        "scripts/phase1b/runpod_parallel.py",
        "scripts/phase1c/runpod_retrain.py",
        "scripts/phase0/runpod_orchestrate.py",
    )

    def test_the_enumeration_is_complete(self) -> None:
        """If a new caller appears, this test names it."""
        import subprocess
        out = subprocess.run(
            ["grep", "-rln", r"create_pod(\|create_pods_parallel(",
             "--include=*.py", "scripts/"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        ).stdout.split()
        callers = {p for p in out if not p.endswith("runpod_provision.py")}
        assert callers == set(self.CALL_SITES), (
            f"pod-creation call sites changed: {callers ^ set(self.CALL_SITES)}. "
            "Every one of them must be covered by the boundary guard."
        )
