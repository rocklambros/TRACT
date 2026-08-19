"""Tests for the sentence-transformers import compatibility layer.

The claim under test is a claim about real distributions, so nothing here mocks
the import system to prove a version-compatibility point. The layout matrix in
EXPECTED_SOURCE_BY_VERSION is asserted against whichever sentence-transformers
is genuinely installed: 3.2.0 on a machine carrying the serving pin, 5.7.0 in
the CI training-stack job. The synthetic candidate tables below exercise the
resolver's control flow using real importable modules, never a mock import hook.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tract.training import st_compat
from tract.training.st_compat import (
    SYMBOL_PATHS,
    TESTED_VERSIONS,
    SentenceTransformersLayoutError,
    _covers,
    is_tested_version,
    pinned_st_version,
    require_tested_version,
    resolve_symbol,
    resolve_symbol_source,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Read out of the published wheels for each version, independently of the table
# the shim ships, so this file disagrees with the shim when the shim is wrong.
EXPECTED_SOURCE_BY_VERSION: dict[str, dict[str, str]] = {
    "3.2.0": {
        "DefaultBatchSampler": "sentence_transformers.sampler",
        "MultipleNegativesRankingLoss": "sentence_transformers.losses",
        "BatchSamplers": "sentence_transformers.training_args",
    },
    "5.3.0": {
        "DefaultBatchSampler": "sentence_transformers.sampler",
        "MultipleNegativesRankingLoss": "sentence_transformers.losses",
        "BatchSamplers": "sentence_transformers.training_args",
    },
    "5.7.0": {
        "DefaultBatchSampler": "sentence_transformers.base.sampler",
        "MultipleNegativesRankingLoss": (
            "sentence_transformers.sentence_transformer.losses"
        ),
        "BatchSamplers": "sentence_transformers.sentence_transformer.training_args",
    },
}


class TestPinParsing:
    """pinned_st_version reads the version the PODS will install."""

    def test_reads_the_real_training_pin(self) -> None:
        # The training pin is what the GPU fleet installs. If this stops being
        # an exact version the preflight has nothing to gate on.
        assert pinned_st_version(REPO_ROOT / "requirements-train.txt") == "5.7.0"

    def test_reads_the_real_serving_pin(self) -> None:
        assert pinned_st_version(REPO_ROOT / "requirements-ml.txt") == "3.2.0"

    def test_the_two_pins_differ(self) -> None:
        # The whole reason the shim exists. If someone unifies them, the shim
        # is no longer load-bearing and this test should be revisited on
        # purpose rather than quietly passing.
        train = pinned_st_version(REPO_ROOT / "requirements-train.txt")
        serve = pinned_st_version(REPO_ROOT / "requirements-ml.txt")
        assert train != serve

    def test_missing_file_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No requirements file"):
            pinned_st_version(tmp_path / "absent.txt")

    def test_no_pin_at_all_fails_loud(self, tmp_path: Path) -> None:
        target = tmp_path / "r.txt"
        target.write_text("torch==2.13.0\nnumpy==2.0.2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no exact `sentence-transformers==` pin"):
            pinned_st_version(target)

    @pytest.mark.parametrize("spec", [
        "sentence-transformers>=5.7.0",
        "sentence-transformers~=5.7.0",
        "sentence-transformers",
        "sentence-transformers<6",
    ])
    def test_a_range_is_not_a_pin(self, tmp_path: Path, spec: str) -> None:
        target = tmp_path / "r.txt"
        target.write_text(f"{spec}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no exact `sentence-transformers==` pin"):
            pinned_st_version(target)

    def test_a_commented_pin_is_not_a_pin(self, tmp_path: Path) -> None:
        # requirements-ml.txt genuinely carries commented version prose, so a
        # regex that ignored the comment marker would read the wrong version.
        target = tmp_path / "r.txt"
        target.write_text(
            "# sentence-transformers==9.9.9 was considered and rejected\n"
            "sentence-transformers==5.7.0\n",
            encoding="utf-8",
        )
        assert pinned_st_version(target) == "5.7.0"

    def test_underscore_spelling_is_accepted(self, tmp_path: Path) -> None:
        target = tmp_path / "r.txt"
        target.write_text("sentence_transformers==5.3.0\n", encoding="utf-8")
        assert pinned_st_version(target) == "5.3.0"

    def test_inline_comment_is_stripped(self, tmp_path: Path) -> None:
        target = tmp_path / "r.txt"
        target.write_text(
            "sentence-transformers==5.7.0  # reorganised layout\n", encoding="utf-8",
        )
        assert pinned_st_version(target) == "5.7.0"

    def test_conflicting_duplicate_pins_fail_loud(self, tmp_path: Path) -> None:
        target = tmp_path / "r.txt"
        target.write_text(
            "sentence-transformers==5.7.0\nsentence-transformers==3.2.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="more than once"):
            pinned_st_version(target)

    def test_repeated_identical_pin_is_accepted(self, tmp_path: Path) -> None:
        target = tmp_path / "r.txt"
        target.write_text(
            "sentence-transformers==5.7.0\nsentence-transformers==5.7.0\n",
            encoding="utf-8",
        )
        assert pinned_st_version(target) == "5.7.0"


class TestVersionGate:
    """An untested version must be named, not guessed at."""

    @pytest.mark.parametrize("version", ["3.2.0", "5.3.0", "5.7.0"])
    def test_tested_versions_pass(self, version: str) -> None:
        assert is_tested_version(version) is True
        assert require_tested_version(version, "ctx") is None

    @pytest.mark.parametrize("version", ["6.0.0", "5.8.0", "5.7.1", "2.7.0", ""])
    def test_untested_versions_are_refused(self, version: str) -> None:
        assert is_tested_version(version) is False
        with pytest.raises(SentenceTransformersLayoutError) as excinfo:
            require_tested_version(version, "Refusing to provision")
        message = str(excinfo.value)
        # Naming the offending value and the caller's context is the whole
        # point: a bare "unsupported version" would not tell an operator which
        # pin to change.
        assert version in message or version == ""
        assert "Refusing to provision" in message
        for tested in TESTED_VERSIONS:
            assert tested in message

    def test_the_training_pin_is_a_tested_version(self) -> None:
        """The check that catches a future pin bump before it reaches a pod."""
        pinned = pinned_st_version(REPO_ROOT / "requirements-train.txt")
        assert is_tested_version(pinned), (
            f"requirements-train.txt pins sentence-transformers=={pinned}, whose "
            f"layout was never read from its wheel."
        )

    def test_the_serving_pin_is_a_tested_version(self) -> None:
        pinned = pinned_st_version(REPO_ROOT / "requirements-ml.txt")
        assert is_tested_version(pinned)


class TestCovers:
    """Distinguishing a layout mismatch from a broken environment."""

    @pytest.mark.parametrize("missing,candidate", [
        ("sentence_transformers.base", "sentence_transformers.base.sampler"),
        ("sentence_transformers.base.sampler", "sentence_transformers.base.sampler"),
        ("sentence_transformers", "sentence_transformers.losses"),
    ])
    def test_candidate_or_ancestor_is_covered(self, missing: str, candidate: str) -> None:
        assert _covers(missing, candidate) is True

    @pytest.mark.parametrize("missing,candidate", [
        ("torch", "sentence_transformers.base.sampler"),
        ("datasets", "sentence_transformers.sampler"),
        ("xxhash", "sentence_transformers.base.sampler"),
        # A string prefix that is not a package boundary. `sentence_transformers.bas`
        # is not a parent of `...base.sampler`, and a bare startswith would say
        # it is, swallowing an unrelated failure.
        ("sentence_transformers.bas", "sentence_transformers.base.sampler"),
        ("sentence_transformers_extra", "sentence_transformers.losses"),
    ])
    def test_unrelated_module_is_not_covered(self, missing: str, candidate: str) -> None:
        assert _covers(missing, candidate) is False


class TestResolverControlFlow:
    """Exercised with real importable modules, not a patched import hook."""

    def test_unknown_symbol_fails_loud(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            resolve_symbol("NotASentenceTransformersSymbol")
        message = str(excinfo.value)
        assert "NotASentenceTransformersSymbol" in message
        for known in SYMBOL_PATHS:
            assert known in message

    def test_falls_through_an_absent_candidate_to_the_next(self) -> None:
        # First candidate names a module that does not exist anywhere, second
        # names one that does. Both go through the real import machinery.
        table = {"Probe": (
            ("tract_st_compat_absent_package_xyz", "dumps"),
            ("json", "dumps"),
        )}
        with patch.object(st_compat, "SYMBOL_PATHS", table):
            assert resolve_symbol_source("Probe") == "json"
            import json
            assert resolve_symbol("Probe") is json.dumps

    def test_falls_through_a_present_module_missing_the_attribute(self) -> None:
        table = {"Probe": (
            ("json", "no_such_attribute_on_json"),
            ("os", "getcwd"),
        )}
        with patch.object(st_compat, "SYMBOL_PATHS", table):
            assert resolve_symbol_source("Probe") == "os"

    def test_a_broken_dependency_is_not_reported_as_a_layout_change(
        self, tmp_path: Path,
    ) -> None:
        """The discrimination that stops a missing torch looking like a move.

        Uses a real package on sys.path whose __init__ imports something that
        is genuinely not installed, so the real import machinery raises the
        real ModuleNotFoundError.
        """
        package = tmp_path / "tract_st_compat_probe_pkg"
        package.mkdir()
        (package / "__init__.py").write_text(
            "import tract_st_compat_definitely_absent_dependency\n", encoding="utf-8",
        )
        table = {"Probe": (
            ("tract_st_compat_probe_pkg", "anything"),
            ("json", "dumps"),
        )}
        sys.path.insert(0, str(tmp_path))
        try:
            with patch.object(st_compat, "SYMBOL_PATHS", table):
                with pytest.raises(ModuleNotFoundError) as excinfo:
                    resolve_symbol("Probe")
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("tract_st_compat_probe_pkg", None)
        # The missing dependency, not the candidate, and not a layout error.
        assert excinfo.value.name == "tract_st_compat_definitely_absent_dependency"
        assert not isinstance(excinfo.value, SentenceTransformersLayoutError)


class TestInstalledDistribution:
    """Asserted against whatever sentence-transformers is genuinely installed."""

    @pytest.fixture(autouse=True)
    def _needs_sentence_transformers(self) -> None:
        pytest.importorskip(
            "sentence_transformers", reason="needs the phase0 or training extra",
        )

    def test_every_symbol_resolves(self) -> None:
        for symbol in SYMBOL_PATHS:
            assert resolve_symbol(symbol) is not None

    def test_resolution_matches_the_layout_matrix_for_this_version(self) -> None:
        version = st_compat.installed_version()
        expected = EXPECTED_SOURCE_BY_VERSION.get(version)
        if expected is None:
            pytest.skip(
                f"sentence-transformers=={version} has no row in this file's "
                f"matrix, so there is nothing to compare against."
            )
        actual = {s: resolve_symbol_source(s) for s in SYMBOL_PATHS}
        assert actual == expected

    def test_the_installed_version_is_one_the_shim_vouches_for(self) -> None:
        assert is_tested_version(st_compat.installed_version())

    def test_resolved_symbols_are_the_kinds_the_training_code_uses(self) -> None:
        # Classes, because loop.py instantiates the loss and data.py subclasses
        # the sampler. No model is loaded here: nothing is instantiated.
        assert isinstance(resolve_symbol("MultipleNegativesRankingLoss"), type)
        assert isinstance(resolve_symbol("DefaultBatchSampler"), type)
        # loop.py reads this member when the custom sampler is off. An enum
        # that dropped it would resolve fine and fail at training time.
        assert hasattr(resolve_symbol("BatchSamplers"), "BATCH_SAMPLER")

    def test_exhausted_ladder_names_the_version_and_every_path(self) -> None:
        table = {"Probe": (
            ("json", "no_such_attribute"),
            ("tract_st_compat_absent_package_xyz", "no_such_attribute"),
        )}
        with patch.object(st_compat, "SYMBOL_PATHS", table):
            with pytest.raises(SentenceTransformersLayoutError) as excinfo:
                resolve_symbol("Probe")
        message = str(excinfo.value)
        assert st_compat.installed_version() in message
        assert "json.no_such_attribute" in message
        assert "tract_st_compat_absent_package_xyz.no_such_attribute" in message
        for tested in TESTED_VERSIONS:
            assert tested in message

    def test_the_custom_sampler_really_subclasses_the_resolved_base(self) -> None:
        """Proves the shim feeds the class the training code inherits from."""
        pytest.importorskip("datasets", reason="tract.training.data imports datasets")
        from tract.training.data import HubAwareTemperatureSampler

        assert issubclass(
            HubAwareTemperatureSampler, resolve_symbol("DefaultBatchSampler"),
        )


class TestProvisioningIsBlocked:
    """The preflight must run before anything starts billing."""

    @staticmethod
    def _requirements_root(tmp_path: Path, pin: str) -> Path:
        (tmp_path / "requirements-train.txt").write_text(
            f"sentence-transformers=={pin}\n", encoding="utf-8",
        )
        return tmp_path

    def test_untested_pin_stops_provision_before_any_pod_is_created(
        self, tmp_path: Path,
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        def _must_not_run(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "provision() reached a billable step despite an untested pin"
            )

        root = self._requirements_root(tmp_path, "6.0.0")
        with patch.object(rp, "PROJECT_ROOT", root), \
                patch.object(rp, "_preflight_tracking", _must_not_run), \
                patch.object(rp, "select_pod_configs", _must_not_run), \
                patch.object(rp, "rank_available_gpus", _must_not_run), \
                patch.object(rp, "create_pods_parallel", _must_not_run), \
                patch.object(rp, "_save_pod_state", _must_not_run):
            with pytest.raises(SentenceTransformersLayoutError, match="6.0.0"):
                rp.provision()

    def test_a_tested_pin_lets_the_preflight_pass(self, tmp_path: Path) -> None:
        pytest.importorskip(
            "sentence_transformers", reason="the local resolve needs the stack",
        )
        from scripts.phase1b import runpod_parallel as rp

        root = self._requirements_root(tmp_path, "5.7.0")
        with patch.object(rp, "PROJECT_ROOT", root):
            assert rp._preflight_training_stack() is None

    def test_the_preflight_runs_ahead_of_the_tracking_preflight(
        self, tmp_path: Path,
    ) -> None:
        """Ordering matters: the cheap deterministic check should not sit behind
        a network call to WandB."""
        from scripts.phase1b import runpod_parallel as rp

        order: list[str] = []

        def _training(*args: Any, **kwargs: Any) -> None:
            order.append("training")

        def _tracking(*args: Any, **kwargs: Any) -> None:
            order.append("tracking")
            raise RuntimeError("stop here")

        with patch.object(rp, "_preflight_training_stack", _training), \
                patch.object(rp, "_preflight_tracking", _tracking):
            with pytest.raises(RuntimeError, match="stop here"):
                rp.provision()
        assert order == ["training", "tracking"]
