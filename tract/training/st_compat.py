"""Version-tolerant imports for the sentence-transformers training symbols.

WHY THIS MODULE EXISTS
Three versions of sentence-transformers are in play at once. requirements-ml.txt
pins 3.2.0 for serving, requirements-train.txt pins 5.7.0 for training, and the
published model records 5.3.0 as its build stack. 5.4 reorganised the package
into per-encoder subpackages, so the import paths that work on 3.2.0 raise
ModuleNotFoundError on 5.7.0 and the reverse holds too. Hard-coding either
layout breaks the other environment.

The failure this prevents is expensive rather than merely annoying. Training
runs on a RunPod fleet that installs requirements-train.txt, so a bad import
path surfaces after the pods are already billing.

WHERE THE PATHS COME FROM
Every entry in SYMBOL_PATHS was read out of the real distribution rather than
recalled. The wheels for 3.2.0, 5.3.0 and 5.7.0 were downloaded and their
archives inspected for the module that defines or re-exports each symbol. The
resulting matrix:

    symbol                        3.2.0                  5.3.0                  5.7.0
    DefaultBatchSampler           .sampler               .sampler               .base.sampler
    MultipleNegativesRankingLoss  .losses                .losses                .sentence_transformer.losses
    BatchSamplers                 .training_args         .training_args         .sentence_transformer.training_args

SentenceTransformer, SentenceTransformerTrainer and SentenceTransformerTrainingArguments
are re-exported from the top-level package in all three versions, so the modules
that import those directly need no shim.

Owner: TRACT maintainers.
"""
from __future__ import annotations

import logging
import re
from importlib import import_module
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# The versions whose package layout was read from the published distribution.
# Adding a version here without inspecting its wheel defeats the point of the
# module, because the candidate ladder below would then be a guess.
TESTED_VERSIONS: Final[tuple[str, ...]] = ("3.2.0", "5.3.0", "5.7.0")

# Candidate (module, attribute) pairs per symbol, ordered newest layout first so
# the current training pin resolves on its first try and the older serving pin
# costs one failed import.
#
# The candidates are exact module paths rather than the top-level package on
# purpose. The top-level __init__ re-exports DefaultBatchSampler on 5.3.0 and
# 5.7.0 but not on 3.2.0, and it re-exports neither MultipleNegativesRankingLoss
# nor BatchSamplers on any of the three, so the package root is not a path that
# covers the matrix.
#
# MultipleNegativesRankingLoss exists twice on 5.7.0, once under cross_encoder
# and once under sentence_transformer. TRACT trains a bi-encoder, so the
# sentence_transformer copy is the correct one and the cross_encoder copy is
# deliberately absent from the ladder.
SYMBOL_PATHS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "DefaultBatchSampler": (
        ("sentence_transformers.base.sampler", "DefaultBatchSampler"),
        ("sentence_transformers.sampler", "DefaultBatchSampler"),
    ),
    "MultipleNegativesRankingLoss": (
        (
            "sentence_transformers.sentence_transformer.losses",
            "MultipleNegativesRankingLoss",
        ),
        ("sentence_transformers.losses", "MultipleNegativesRankingLoss"),
    ),
    "BatchSamplers": (
        ("sentence_transformers.sentence_transformer.training_args", "BatchSamplers"),
        ("sentence_transformers.training_args", "BatchSamplers"),
    ),
}

_ROOT_PACKAGE: Final[str] = "sentence_transformers"

# Matches an exact pin for the project on a requirements line. The distribution
# name normalises hyphen and underscore, so both spellings are accepted.
_PIN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*sentence[-_]transformers\s*==\s*([0-9][^\s;#]*)", re.IGNORECASE,
)


class SentenceTransformersLayoutError(ImportError):
    """Raised when no known import path yields a required training symbol."""


def _covers(missing: str, candidate_module: str) -> bool:
    """Return True when `missing` is the candidate module or a parent package.

    Import failures have two very different causes and only one of them means
    "this layout is not the installed one". A missing `sentence_transformers.base`
    while probing `sentence_transformers.base.sampler` says the installed version
    predates the reorganisation, so the next candidate should be tried. A missing
    `torch` underneath a module that does exist says the environment is broken,
    and treating that as a layout mismatch would report the wrong defect.
    """
    return missing == candidate_module or candidate_module.startswith(missing + ".")


def installed_version() -> str:
    """Return the installed sentence-transformers version string."""
    try:
        root = import_module(_ROOT_PACKAGE)
    except ModuleNotFoundError as exc:
        raise SentenceTransformersLayoutError(
            "sentence-transformers is not installed, so no training symbol can "
            "be resolved. Install the training stack with "
            "`pip install -r requirements.txt -r requirements-train.txt`."
        ) from exc
    version = getattr(root, "__version__", "")
    if not version:
        raise SentenceTransformersLayoutError(
            f"sentence-transformers imported from {getattr(root, '__file__', '?')} "
            f"but exposes no __version__, so the package layout it uses cannot "
            f"be identified."
        )
    return str(version)


def is_tested_version(version: str) -> bool:
    """Return True when this exact version's layout was read from its wheel."""
    return version in TESTED_VERSIONS


def require_tested_version(version: str, context: str) -> None:
    """Raise unless `version` is one whose layout was verified against its wheel.

    Callers that are about to spend money use this rather than the warning that
    resolve_symbol emits. A version outside the matrix might happen to import,
    and it might equally move a symbol again, so the gate in front of a paid
    resource refuses rather than guesses.
    """
    if is_tested_version(version):
        return
    raise SentenceTransformersLayoutError(
        f"{context}: sentence-transformers=={version} is outside the tested set "
        f"{', '.join(TESTED_VERSIONS)}. Its package layout has not been read "
        f"from its wheel, so the import paths in "
        f"tract/training/st_compat.py:SYMBOL_PATHS may not cover it. Verify the "
        f"layout and add the version to TESTED_VERSIONS, or change the pin."
    )


def pinned_st_version(requirements_path: Path) -> str:
    """Return the exact sentence-transformers version pinned in a requirements file.

    The pods install this file, so this is the version the training code will
    run under, whatever the machine driving the campaign happens to have.
    """
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"No requirements file at {requirements_path}, so the "
            f"sentence-transformers version the pods will install is unknown."
        )
    text = requirements_path.read_text(encoding="utf-8")
    matches = [
        m.group(1)
        for line in text.splitlines()
        if (m := _PIN_PATTERN.match(line)) is not None
    ]
    if not matches:
        raise ValueError(
            f"{requirements_path} carries no exact `sentence-transformers==` pin. "
            f"A range is not a pin, and the training code depends on the package "
            f"layout, which changes between minor versions."
        )
    if len(set(matches)) > 1:
        raise ValueError(
            f"{requirements_path} pins sentence-transformers more than once with "
            f"conflicting versions: {sorted(set(matches))}."
        )
    return matches[0]


def resolve_symbol(symbol: str) -> Any:
    """Return `symbol` from whichever sentence-transformers layout is installed."""
    return _resolve(symbol)[0]


def resolve_symbol_source(symbol: str) -> str:
    """Return the module path `symbol` resolves from in this environment.

    Exposed so a test can assert which row of the layout matrix an installed
    distribution actually takes, rather than only that the import succeeded. A
    ladder that fell through to the wrong candidate would still return a usable
    object, so success alone does not prove the matrix is right.
    """
    return _resolve(symbol)[1]


def _resolve(symbol: str) -> tuple[Any, str]:
    """Return (object, source module) for `symbol` under the installed layout.

    Raises SentenceTransformersLayoutError naming the installed version and every
    path tried when no candidate yields the symbol. Returning a sentinel here
    would push a NameError or an AttributeError into the training loop, which is
    where the cost of a late failure is highest.
    """
    candidates = SYMBOL_PATHS.get(symbol)
    if candidates is None:
        raise KeyError(
            f"No sentence-transformers import candidates registered for {symbol!r}. "
            f"Known symbols: {', '.join(sorted(SYMBOL_PATHS))}."
        )

    attempts: list[str] = []
    for module_name, attribute in candidates:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == _ROOT_PACKAGE:
                raise SentenceTransformersLayoutError(
                    f"sentence-transformers is not installed, so {symbol} cannot "
                    f"be resolved. Install the training stack with "
                    f"`pip install -r requirements.txt -r requirements-train.txt`."
                ) from exc
            if not _covers(missing, module_name):
                # A dependency underneath the candidate is missing. That is an
                # environment defect, not a layout difference, and swallowing it
                # would report the wrong cause.
                raise
            attempts.append(f"{module_name}.{attribute} (no module {missing})")
            continue
        try:
            resolved = getattr(module, attribute)
        except AttributeError:
            # The module exists in this layout but does not carry the symbol,
            # which is what a partial reorganisation looks like. A sentinel
            # default would conflate that with an attribute whose value is None.
            attempts.append(
                f"{module_name}.{attribute} (module imported, attribute absent)"
            )
            continue
        # Scoped to sentence-transformers modules because that is what the
        # warning is about. It also keeps the resolver usable against a
        # synthetic candidate table in a test environment with no ML stack.
        if module_name.split(".")[0] == _ROOT_PACKAGE:
            version = installed_version()
            if not is_tested_version(version):
                logger.warning(
                    "sentence-transformers==%s is outside the tested set (%s). "
                    "%s resolved from %s, but this layout was never read from a "
                    "wheel, so another symbol may have moved.",
                    version, ", ".join(TESTED_VERSIONS), symbol, module_name,
                )
            logger.debug(
                "Resolved %s from %s under sentence-transformers==%s",
                symbol, module_name, version,
            )
        return resolved, module_name

    raise SentenceTransformersLayoutError(
        f"Cannot resolve {symbol} under sentence-transformers=="
        f"{installed_version()}. Tried: {'; '.join(attempts)}. Tested versions "
        f"are {', '.join(TESTED_VERSIONS)}. If this is a new release, read its "
        f"wheel and add the path to tract/training/st_compat.py:SYMBOL_PATHS."
    )
