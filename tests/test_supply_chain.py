"""Tests for the pinned-dependency supply-chain gates.

Two defects are under test here, both of which reached a pushed branch.

The first: `scipy==1.18.0` declares `Requires-Python: >=3.12`, so the pinned
training stack could not install on any 3.11 interpreter. The check that was
supposed to catch that before a fleet starts billing asked an import question,
and nothing imports when nothing installed.

The second: the CI job whose claim is "the pinned training stack installs where
the folds run it" was running 3.11 while the folds run 3.12, so the claim was
about an interpreter no pod has.

The metadata assertions come in two flavours. The offline ones drive the
resolver with an explicit table, so they state the control flow without a
network. The ones marked `integration` read the real PyPI metadata, because the
version choice rests on a fact about published distributions and a table cannot
be wrong in the same direction the code is.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from tract import supply_chain
from tract.supply_chain import (
    AUDIT_SUPPRESSIONS,
    MetadataUnavailableError,
    PinViolation,
    RequirementsFormatError,
    expired_suppressions,
    fetch_requires_python,
    find_python_incompatible_pins,
    parse_exact_pins,
    python_version_admitted,
    suppressed_ids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_REQUIREMENTS = REPO_ROOT / "requirements-train.txt"
SERVE_REQUIREMENTS = REPO_ROOT / "requirements-ml.txt"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# An advisory identifier from any of the three schemes pip-audit emits.
_VULN_ID = re.compile(r"^(PYSEC|CVE|GHSA)-")


def _table_fetcher(table: dict[str, str]) -> Any:
    """Return a fetcher backed by {package: requires_python}."""
    def _fetch(name: str, version: str) -> str:
        if name not in table:
            raise AssertionError(f"unexpected metadata lookup for {name}=={version}")
        return table[name]
    return _fetch


class TestParseExactPins:
    """The parser decides what the pods will install, so it reports rather
    than skips anything that is not an exact pin."""

    def test_reads_the_real_training_file(self) -> None:
        pins = parse_exact_pins(TRAIN_REQUIREMENTS)
        # scipy is the regression: 1.18.0 excludes 3.11, 1.17.1 is the last
        # release that admits it.
        assert pins["scipy"] == "1.17.1"
        assert pins["torch"] == "2.13.0"
        assert pins["transformers"] == "4.57.6"
        assert pins["sentence-transformers"] == "5.7.0"

    def test_reads_the_real_serving_file(self) -> None:
        pins = parse_exact_pins(SERVE_REQUIREMENTS)
        assert pins["scipy"] == "1.17.1"
        assert pins["sentence-transformers"] == "3.2.0"

    def test_the_two_files_agree_on_scipy(self) -> None:
        """pyproject declares `requires-python = ">=3.11"`. Both files have to
        install there, so both take the last scipy whose floor admits it."""
        train = parse_exact_pins(TRAIN_REQUIREMENTS)["scipy"]
        serve = parse_exact_pins(SERVE_REQUIREMENTS)["scipy"]
        assert train == serve

    def test_ignores_comments_options_and_blank_lines(self, tmp_path: Path) -> None:
        target = tmp_path / "requirements-train.txt"
        target.write_text(
            "# a comment\n"
            "-c requirements.txt\n"
            "\n"
            "   \n"
            "-e .\n"
            "torch==2.13.0  # a trailing comment\n",
            encoding="utf-8",
        )
        assert parse_exact_pins(target) == {"torch": "2.13.0"}

    def test_normalises_the_distribution_name(self, tmp_path: Path) -> None:
        target = tmp_path / "requirements-train.txt"
        target.write_text("sentence_transformers==5.7.0\n", encoding="utf-8")
        assert parse_exact_pins(target) == {"sentence-transformers": "5.7.0"}

    def test_rejects_a_range(self, tmp_path: Path) -> None:
        target = tmp_path / "requirements-train.txt"
        target.write_text("torch==2.13.0\nscipy>=1.17.1\n", encoding="utf-8")
        with pytest.raises(RequirementsFormatError) as excinfo:
            parse_exact_pins(target)
        message = str(excinfo.value)
        assert ":2" in message, "the failing line number should be named"
        assert "scipy>=1.17.1" in message

    def test_rejects_an_environment_marker(self, tmp_path: Path) -> None:
        """A marker makes the pin conditional, and the check compares against
        one interpreter."""
        target = tmp_path / "requirements-train.txt"
        target.write_text(
            'scipy==1.18.0; python_version >= "3.12"\n', encoding="utf-8",
        )
        with pytest.raises(RequirementsFormatError, match="python_version"):
            parse_exact_pins(target)

    def test_rejects_conflicting_duplicates(self, tmp_path: Path) -> None:
        target = tmp_path / "requirements-train.txt"
        target.write_text("scipy==1.17.1\nscipy==1.18.0\n", encoding="utf-8")
        with pytest.raises(RequirementsFormatError, match="conflicting"):
            parse_exact_pins(target)

    def test_accepts_a_repeated_identical_pin(self, tmp_path: Path) -> None:
        target = tmp_path / "requirements-train.txt"
        target.write_text("scipy==1.17.1\nscipy==1.17.1\n", encoding="utf-8")
        assert parse_exact_pins(target) == {"scipy": "1.17.1"}

    def test_rejects_a_file_with_no_pins(self, tmp_path: Path) -> None:
        target = tmp_path / "requirements-train.txt"
        target.write_text("# nothing here\n-c requirements.txt\n", encoding="utf-8")
        with pytest.raises(RequirementsFormatError, match="no exact pins"):
            parse_exact_pins(target)

    def test_raises_on_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_exact_pins(tmp_path / "absent.txt")


class TestPythonVersionAdmitted:
    """Both directions of every boundary that matters."""

    @pytest.mark.parametrize(("requires_python", "python_version", "expected"), [
        # The regression, at its exact boundary.
        (">=3.12", "3.12", True),
        (">=3.12", "3.11", False),
        (">=3.11", "3.11", True),
        (">=3.11", "3.10", False),
        # A release with no declared floor admits everything.
        ("", "3.11", True),
        ("   ", "3.9", True),
        # Three-component floors are common on PyPI.
        (">=3.9.0", "3.12", True),
        (">=3.13.0", "3.12", False),
        # Upper bounds are rarer but real.
        (">=3.10,<3.13", "3.12", True),
        (">=3.10,<3.13", "3.13", False),
        (">=3.8,!=3.9.*", "3.9", False),
        (">=3.8,!=3.9.*", "3.10", True),
    ])
    def test_boundaries(
        self, requires_python: str, python_version: str, expected: bool,
    ) -> None:
        assert python_version_admitted(requires_python, python_version) is expected

    def test_raises_on_an_unparseable_specifier(self) -> None:
        with pytest.raises(MetadataUnavailableError, match="Requires-Python"):
            python_version_admitted("this is not a specifier", "3.12")

    def test_raises_on_a_non_version_python(self) -> None:
        with pytest.raises(ValueError, match="not a version"):
            python_version_admitted(">=3.11", "three-point-twelve")


class TestFindPythonIncompatiblePins:

    def test_the_scipy_regression_in_both_directions(self) -> None:
        """The defect that shipped, stated as the pods and CI each saw it."""
        pins = {"scipy": "1.18.0", "torch": "2.13.0"}
        fetch = _table_fetcher({"scipy": ">=3.12", "torch": ">=3.10"})

        on_311 = find_python_incompatible_pins(pins, "3.11", fetch)
        assert [v.package for v in on_311] == ["scipy"]
        assert on_311[0].version == "1.18.0"

        on_312 = find_python_incompatible_pins(pins, "3.12", fetch)
        assert on_312 == []

    def test_reports_every_violation_not_only_the_first(self) -> None:
        """Three bad pins should cost one preflight, not three."""
        pins = {"scipy": "1.18.0", "torch": "2.13.0", "aaa": "1.0", "zzz": "1.0"}
        fetch = _table_fetcher({
            "scipy": ">=3.12", "torch": ">=3.10", "aaa": ">=3.13", "zzz": ">=3.14",
        })
        violations = find_python_incompatible_pins(pins, "3.11", fetch)
        assert [v.package for v in violations] == ["aaa", "scipy", "zzz"]

    def test_empty_when_every_pin_admits(self) -> None:
        pins = {"scipy": "1.17.1", "torch": "2.13.0"}
        fetch = _table_fetcher({"scipy": ">=3.11", "torch": ">=3.10"})
        assert find_python_incompatible_pins(pins, "3.12", fetch) == []

    def test_a_missing_floor_is_not_a_violation(self) -> None:
        pins = {"einops": "0.8.2"}
        assert find_python_incompatible_pins(
            pins, "3.11", _table_fetcher({"einops": ""}),
        ) == []

    def test_the_message_names_the_version_and_the_floor(self) -> None:
        violation = PinViolation(
            package="scipy", version="1.18.0",
            requires_python=">=3.12", python_version="3.11",
        )
        message = violation.message()
        assert "scipy==1.18.0" in message
        assert ">=3.12" in message
        assert "3.11" in message


class TestPodAndCiAgree:
    """The guard for the second defect: CI proving a claim about the wrong
    interpreter."""

    @staticmethod
    def _training_stack_python() -> str:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["training-stack"]["steps"]
        versions = [
            str(step["with"]["python-version"])
            for step in steps
            if isinstance(step.get("with"), dict) and "python-version" in step["with"]
        ]
        assert len(versions) == 1, (
            f"expected exactly one interpreter in the training-stack job, "
            f"found {versions}"
        )
        return versions[0]

    def test_ci_training_job_runs_the_pod_interpreter(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        assert self._training_stack_python() == rp.POD_PYTHON_VERSION, (
            "the training-stack job exists to prove the pinned stack installs "
            "where the folds run it, so it has to run the pod's Python"
        )

    def test_the_pod_interpreter_sits_with_the_image_it_describes(self) -> None:
        """The Python is a property of the digest, so the two move together."""
        from scripts.phase1b import runpod_parallel as rp

        assert rp.DOCKER_IMAGE.startswith("runpod/pytorch@sha256:")
        assert rp.POD_PYTHON_VERSION == "3.12"

    def test_every_training_pin_admits_the_pod_interpreter_offline(self) -> None:
        """Guards the shape of the call the preflight makes, with a table."""
        from scripts.phase1b import runpod_parallel as rp

        pins = parse_exact_pins(TRAIN_REQUIREMENTS)
        fetch = _table_fetcher({name: ">=3.11" for name in pins})
        assert find_python_incompatible_pins(
            pins, rp.POD_PYTHON_VERSION, fetch,
        ) == []


class TestAuditSuppressions:
    """A suppression is an argument with a date on it, not an exception."""

    def test_none_has_expired(self) -> None:
        expired = expired_suppressions(date.today())
        assert expired == (), (
            "these suppressions are past their date. Re-make the argument or "
            "clear the finding: "
            + ", ".join(f"{e.vuln_id} (expired {e.expires})" for e in expired)
        )

    def test_the_expiry_bites_on_the_day_and_not_before(self) -> None:
        entry = min(AUDIT_SUPPRESSIONS, key=lambda e: e.expires)
        assert entry in expired_suppressions(entry.expires)
        assert entry not in expired_suppressions(entry.expires - timedelta(days=1))
        assert entry in expired_suppressions(entry.expires + timedelta(days=1))

    def test_ids_are_unique(self) -> None:
        ids = [entry.vuln_id for entry in AUDIT_SUPPRESSIONS]
        assert len(ids) == len(set(ids))
        assert suppressed_ids() == frozenset(ids)

    def test_every_entry_names_a_scheme_a_package_and_evidence(self) -> None:
        for entry in AUDIT_SUPPRESSIONS:
            assert _VULN_ID.match(entry.vuln_id), entry.vuln_id
            assert entry.package, entry.vuln_id
            # A reason has to point at something checkable: a path in this
            # repo, or the search that found nothing.
            assert "tract/" in entry.reason or "grep" in entry.reason, entry.vuln_id
            assert len(entry.reason) >= 100, entry.vuln_id

    def test_the_registry_holds_only_the_transformers_ceiling(self) -> None:
        """Everything clearable by a bump was cleared. Only the advisories
        blocked by the transformers 5.x / huggingface_hub 1.x boundary stay."""
        assert {entry.package for entry in AUDIT_SUPPRESSIONS} == {"transformers"}


class TestWorkflowMatchesTheRegistry:

    @staticmethod
    def _audit_run_block() -> str:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["audit"]["steps"]
        blocks = [
            str(step["run"]) for step in steps
            if "pip-audit" in str(step.get("run", ""))
        ]
        assert len(blocks) == 1, f"expected one audit step, found {len(blocks)}"
        return blocks[0]

    def test_ignore_flags_match_the_registry_exactly(self) -> None:
        flagged = set(re.findall(r"--ignore-vuln\s+(\S+)", self._audit_run_block()))
        assert flagged == suppressed_ids(), (
            "the workflow and tract/supply_chain.py disagree about what is "
            "suppressed, so one of them is not the record"
        )

    def test_every_flag_names_an_advisory(self) -> None:
        """A blanket ignore would pass the set comparison above only by also
        emptying the registry, so check the shape too."""
        for flag in re.findall(r"--ignore-vuln\s+(\S+)", self._audit_run_block()):
            assert _VULN_ID.match(flag), flag

    def test_the_audit_still_audits_the_serving_file(self) -> None:
        block = self._audit_run_block()
        assert "-r requirements-ml.txt" in block
        assert "|| true" not in block, "the audit must be able to fail"

    def test_the_audit_job_is_not_allowed_to_fail(self) -> None:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        assert workflow["jobs"]["audit"].get("continue-on-error") is not True


@pytest.mark.integration
class TestRealMetadata:
    """The version choice rests on published metadata, so read it."""

    def test_pypi_reports_the_scipy_floors_the_pins_were_chosen_from(self) -> None:
        assert fetch_requires_python("scipy", "1.18.0") == ">=3.12"
        assert fetch_requires_python("scipy", "1.17.1") == ">=3.11"

    def test_every_training_pin_admits_the_pod_interpreter(self) -> None:
        from scripts.phase1b import runpod_parallel as rp

        violations = find_python_incompatible_pins(
            parse_exact_pins(TRAIN_REQUIREMENTS),
            rp.POD_PYTHON_VERSION,
            fetch_requires_python,
        )
        assert violations == [], [v.message() for v in violations]

    def test_every_pin_in_both_files_admits_the_declared_floor(self) -> None:
        """pyproject declares `requires-python = ">=3.11"`."""
        for path in (TRAIN_REQUIREMENTS, SERVE_REQUIREMENTS):
            violations = find_python_incompatible_pins(
                parse_exact_pins(path), "3.11", fetch_requires_python,
            )
            assert violations == [], [v.message() for v in violations]

    def test_an_unknown_release_is_a_refusal_not_a_pass(self) -> None:
        with pytest.raises(MetadataUnavailableError):
            fetch_requires_python("scipy", "0.0.0-not-a-release")


class TestProvisioningRefusesAnUninstallablePin:
    """The money gate. A fleet that dies at dependency install has already
    been paid for."""

    @staticmethod
    def _requirements_root(tmp_path: Path, scipy_pin: str) -> Path:
        (tmp_path / "requirements-train.txt").write_text(
            f"sentence-transformers==5.7.0\nscipy=={scipy_pin}\n", encoding="utf-8",
        )
        return tmp_path

    @staticmethod
    def _fetcher() -> Any:
        return _table_fetcher({
            "sentence-transformers": ">=3.9", "scipy": ">=3.12",
        })

    def test_a_pin_excluding_the_pod_python_stops_provision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts.phase1b import runpod_parallel as rp

        def _must_not_run(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "provision() reached a billable step despite an uninstallable pin"
            )

        monkeypatch.setattr(
            supply_chain, "fetch_requires_python", self._fetcher(),
        )
        monkeypatch.setattr(rp, "PROJECT_ROOT", self._requirements_root(tmp_path, "1.18.0"))
        monkeypatch.setattr(rp, "POD_PYTHON_VERSION", "3.11")
        for name in (
            "_preflight_tracking", "select_pod_configs", "rank_available_gpus",
            "create_pods_parallel", "_save_pod_state",
        ):
            monkeypatch.setattr(rp, name, _must_not_run)

        with pytest.raises(RuntimeError, match=r"scipy==1\.18\.0"):
            rp.provision()

    def test_the_same_pin_passes_on_the_interpreter_that_admits_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The other direction: 1.18.0 is fine on 3.12, which is why the pods
        never saw this failure and CI did."""
        pytest.importorskip(
            "sentence_transformers", reason="the local resolve needs the stack",
        )
        from scripts.phase1b import runpod_parallel as rp

        monkeypatch.setattr(
            supply_chain, "fetch_requires_python", self._fetcher(),
        )
        monkeypatch.setattr(rp, "PROJECT_ROOT", self._requirements_root(tmp_path, "1.18.0"))
        monkeypatch.setattr(rp, "POD_PYTHON_VERSION", "3.12")
        assert rp._preflight_training_stack() is None

    def test_the_install_check_runs_before_the_layout_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ordering: an import question about a stack that never installed is
        the wrong answer to give an operator."""
        from scripts.phase1b import runpod_parallel as rp

        (tmp_path / "requirements-train.txt").write_text(
            "sentence-transformers==6.0.0\nscipy==1.18.0\n", encoding="utf-8",
        )
        monkeypatch.setattr(
            supply_chain, "fetch_requires_python", self._fetcher(),
        )
        monkeypatch.setattr(rp, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(rp, "POD_PYTHON_VERSION", "3.11")

        # Both checks would fire. The install one has to be the one that speaks.
        with pytest.raises(RuntimeError, match="dependency resolution"):
            rp._preflight_training_stack()
