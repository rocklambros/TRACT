"""Supply-chain gates over the pinned dependency sets.

Two questions live here, both of which used to be answered by noticing a red
job after the fact.

WILL EVERY PIN INSTALL ON THE PYTHON THAT RUNS IT?
The training fleet installs requirements-train.txt inside `_bootstrap_pod`,
which is after the pods exist and after billing starts. A pin whose
`Requires-Python` excludes the pod's interpreter turns a successful
provisioning run into a paid-for dependency-resolution failure, and the
existing preflight cannot see it: that check imports sentence-transformers
symbols, and nothing imports when nothing installed. `scipy==1.18.0` is the
worked example. It declares `Requires-Python: >=3.12`, so it resolves on the
pod image and fails on any 3.11 interpreter, which is what the CI job that was
supposed to prove the pod install happened to be running.

WHICH pip-audit FINDINGS ARE SUPPRESSED, WHY, AND UNTIL WHEN?
A suppression with no expiry is a decision nobody revisits. Every entry in
AUDIT_SUPPRESSIONS names one advisory, the package it lands on, the reason the
code cannot reach it, and the date the argument has to be made again.
`expired_suppressions` turns that date into a test failure rather than a
comment nobody reads.

Owner: TRACT maintainers.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# PyPI's per-release JSON endpoint. Queried per pin rather than resolved with
# pip, because resolving the training set downloads gigabytes of wheels and the
# only fact needed is one metadata field.
_PYPI_RELEASE_URL: Final[str] = "https://pypi.org/pypi/{name}/{version}/json"
_PYPI_TIMEOUT_S: Final[float] = 30.0

# An exact pin, with an optional trailing comment. Deliberately narrow: the
# requirements files this reads are exact-pin files by contract, so anything
# else is a defect to report rather than a line to skip.
_EXACT_PIN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;#]+)\s*(?:#.*)?$",
)

# Lines that carry no pin: pip options (-c, -r, --hash and friends) and the
# `-e .` editable form.
_OPTION_LINE: Final[re.Pattern[str]] = re.compile(r"^-")


class SupplyChainError(Exception):
    """Base for every refusal raised by this module."""


class RequirementsFormatError(SupplyChainError, ValueError):
    """A requirements file does not hold to the exact-pin contract."""


class MetadataUnavailableError(SupplyChainError):
    """A pin's release metadata could not be read, so it cannot be checked."""


def _normalise(name: str) -> str:
    """Return the PEP 503 normalised form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_exact_pins(requirements_path: Path) -> dict[str, str]:
    """Return {normalised name: version} for every exact pin in the file.

    Raises rather than skipping on anything that is not an exact pin, a
    comment, a blank line or a pip option. A range that slipped into a pin file
    would otherwise be checked against nothing and reported as clean, which is
    the failure mode this module exists to remove.
    """
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"No requirements file at {requirements_path}, so the pins that "
            f"will be installed are unknown."
        )
    pins: dict[str, str] = {}
    for number, raw in enumerate(
        requirements_path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#") or _OPTION_LINE.match(line):
            continue
        match = _EXACT_PIN.match(line)
        if match is None:
            raise RequirementsFormatError(
                f"{requirements_path}:{number} is not an exact pin: {raw!r}. "
                f"This file is read to decide what the pods will install, and a "
                f"range or an environment marker cannot be checked against a "
                f"single interpreter. Write `name==version`."
            )
        name = _normalise(match.group("name"))
        version = match.group("version")
        previous = pins.get(name)
        if previous is not None and previous != version:
            raise RequirementsFormatError(
                f"{requirements_path} pins {name} twice with conflicting "
                f"versions: {previous} and {version}."
            )
        pins[name] = version
    if not pins:
        raise RequirementsFormatError(
            f"{requirements_path} carries no exact pins at all, so there is "
            f"nothing to check."
        )
    return pins


def python_version_admitted(requires_python: str, python_version: str) -> bool:
    """Return True when `python_version` satisfies a `Requires-Python` string.

    An empty specifier means the distribution declares no floor, which admits
    every interpreter. That is PyPI's representation for a release with no
    `requires_python` field, not a missing lookup.
    """
    specifier = requires_python.strip()
    if not specifier:
        return True
    try:
        parsed = SpecifierSet(specifier)
    except InvalidSpecifier as exc:
        raise MetadataUnavailableError(
            f"Cannot read the Requires-Python specifier {requires_python!r}, so "
            f"whether it admits Python {python_version} is unknown."
        ) from exc
    try:
        target = Version(python_version)
    except InvalidVersion as exc:
        raise ValueError(
            f"{python_version!r} is not a version, so it cannot be checked "
            f"against {requires_python!r}."
        ) from exc
    return parsed.contains(target)


def fetch_requires_python(name: str, version: str) -> str:
    """Return the `Requires-Python` string PyPI records for one exact release.

    Returns the empty string when the release declares no floor. Raises when the
    metadata cannot be read at all, because a gate in front of a billable
    resource has to refuse rather than assume the pin is fine.
    """
    url = _PYPI_RELEASE_URL.format(name=name, version=version)
    try:
        with urllib.request.urlopen(url, timeout=_PYPI_TIMEOUT_S) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise MetadataUnavailableError(
            f"Could not read release metadata for {name}=={version} from PyPI "
            f"({exc}). The pins cannot be checked against the pod interpreter, "
            f"so provisioning is refused rather than risking a fleet that dies "
            f"at dependency install."
        ) from exc
    info = payload.get("info")
    if not isinstance(info, dict):
        raise MetadataUnavailableError(
            f"PyPI returned no `info` block for {name}=={version}, so its "
            f"Requires-Python is unknown."
        )
    requires_python = info.get("requires_python")
    return "" if requires_python is None else str(requires_python)


@dataclass(frozen=True)
class PinViolation:
    """One pinned release that cannot install on the target interpreter."""

    package: str
    version: str
    requires_python: str
    python_version: str

    def message(self) -> str:
        return (
            f"{self.package}=={self.version} requires Python "
            f"{self.requires_python!r}, which excludes {self.python_version}"
        )


def find_python_incompatible_pins(
    pins: Mapping[str, str],
    python_version: str,
    fetch: Callable[[str, str], str],
) -> list[PinViolation]:
    """Return every pin whose `Requires-Python` excludes `python_version`.

    Every pin is checked before returning, sorted by package name, so a file
    with three bad pins reports three rather than costing three round trips
    through the fix-and-rerun loop.

    `fetch` is a required argument rather than a default so that callers state
    where the metadata comes from, and so tests can supply a table instead of
    reaching PyPI.
    """
    violations: list[PinViolation] = []
    for package in sorted(pins):
        version = pins[package]
        requires_python = fetch(package, version)
        if not python_version_admitted(requires_python, python_version):
            violations.append(PinViolation(
                package=package, version=version,
                requires_python=requires_python, python_version=python_version,
            ))
    return violations


@dataclass(frozen=True)
class AuditSuppression:
    """One pip-audit finding held open on a stated argument, with an end date."""

    vuln_id: str
    package: str
    expires: date
    reason: str


# WHY THESE FOUR CANNOT BE CLEARED BY A BUMP
# All four are transformers findings whose fix landed in the 5.x line.
# transformers 5.0.0 requires huggingface-hub >= 1.3.0 and 5.5.0 requires
# >= 1.5.0, while pyproject.toml declares `huggingface_hub>=0.24,<1` as a core
# dependency of tract itself, and sentence-transformers 3.2.0 independently
# caps transformers below 5.0.0. Clearing them therefore needs a
# sentence-transformers major bump AND a huggingface_hub 0.x -> 1.x migration
# across six call sites including tract/model_resolver.py, the sha256-verified
# download path behind `tract assign`. That is the upgrade path, and it is a
# change to make on its own, not inside a dependency-pin repair.
#
# Everything that COULD be cleared by a bump was: torch 2.4.1 -> 2.13.0 removed
# 15 findings including PYSEC-2026-2286, an arbitrary-code-execution flaw in
# the `weights_only` unpickler that maps directly onto what TRACT does, and
# datasets 3.6.0 -> 5.0.1 removed the PYSEC-2026-3716 path traversal.
AUDIT_SUPPRESSIONS: Final[tuple[AuditSuppression, ...]] = (
    AuditSuppression(
        vuln_id="PYSEC-2025-217",
        package="transformers",
        expires=date(2026, 11, 17),
        reason=(
            "Deserialization RCE in the X-CLIP checkpoint conversion script "
            "shipped with transformers. TRACT converts no checkpoints and "
            "references no X-CLIP model: a grep of tract/ and scripts/ for "
            "x_clip returns nothing, and every encoder in tract/encoders.py is "
            "bert, modernbert or xlm-roberta. No transformers release carries a "
            "fix, so this one cannot be closed by any bump."
        ),
    ),
    AuditSuppression(
        vuln_id="PYSEC-2026-2288",
        package="transformers",
        expires=date(2026, 11, 17),
        reason=(
            "Trainer._load_rng_state calls torch.load without weights_only=True. "
            "The method runs only when a Trainer resumes from a checkpoint "
            "directory, and a grep of tract/ and scripts/ for "
            "resume_from_checkpoint returns nothing, so it is never reached. "
            "The advisory's own precondition is torch < 2.6, where "
            "safe_globals() is inert; both pinned stacks now carry torch "
            "2.13.0. Fixed in transformers 5.0.0."
        ),
    ),
    AuditSuppression(
        vuln_id="PYSEC-2026-2289",
        package="transformers",
        expires=date(2026, 11, 17),
        reason=(
            "MITIGATED, NOT UNREACHABLE. A config.json carrying "
            "_attn_implementation_internal makes from_pretrained fetch and run "
            "code from an attacker-named Hub repo. TRACT reaches from_pretrained "
            "through sentence-transformers, so the path exists. What blocks it "
            "is tract/model_resolver.py, which verifies a recorded sha256 per "
            "file against tract.config.TRACT_MODEL_PINNED_FILE_HASHES before the "
            "model is used, so config.json cannot change under the default repo "
            "and revision. The mitigation lapses when TRACT_MODEL_REPO or "
            "TRACT_MODEL_REVISION overrides the default, which model_resolver "
            "logs as revision-trust only. Fixed in transformers 5.3.0."
        ),
    ),
    AuditSuppression(
        vuln_id="PYSEC-2026-2290",
        package="transformers",
        expires=date(2026, 11, 17),
        reason=(
            "Nested AutoConfig in LightGlueConfig re-reads trust_remote_code "
            "from an untrusted config. The path is instantiated only for "
            "model_type lightglue. TRACT loads bert, modernbert and "
            "xlm-roberta encoders and publishes a bert model, and the sha256 "
            "pin in tract/model_resolver.py fixes the config bytes on the "
            "default path. Fixed in transformers 5.5.0."
        ),
    ),
)


def suppressed_ids() -> frozenset[str]:
    """Return every advisory id the audit is allowed to ignore."""
    return frozenset(entry.vuln_id for entry in AUDIT_SUPPRESSIONS)


def expired_suppressions(today: date) -> tuple[AuditSuppression, ...]:
    """Return the suppressions whose expiry has passed, oldest first.

    A suppression expires on its date rather than after it: the argument was
    made to hold until that day, so the day itself is when it has to be
    remade.
    """
    return tuple(sorted(
        (entry for entry in AUDIT_SUPPRESSIONS if today >= entry.expires),
        key=lambda entry: entry.expires,
    ))
