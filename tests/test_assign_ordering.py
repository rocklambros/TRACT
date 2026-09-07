"""`tract assign` did expensive work before cheap validation, twice over.

Two defects in four lines of `_cmd_assign`, both found by a product-surface
audit -- and the second one caught the auditor: it ran
`tract assign --file /nonexistent`, reasonably expecting a file-not-found check
to be cheap, and loaded the deployment model on a machine where this project
forbids that.

    1. `from tract.inference import TRACTPredictor` ran BEFORE
       `_require_inference_runtime()`. `tract/inference.py` imports numpy at
       module level and numpy is only in the `[phase0]` extra, so a base install
       got an uncaught `ModuleNotFoundError` and exit 1 -- not the documented
       exit 5 with its actionable "pip install 'tract[phase0]'" message. The
       guard also never checked numpy, only torch and sentence_transformers.

    2. `TRACTPredictor(...)` was constructed BEFORE `--file` was checked for
       existence, so a typo cost a ~1.3 GB model load before "File not found".

These tests do not load a model. They assert the ORDER, which is the property
that was wrong.
"""

from __future__ import annotations

import inspect
import re
from typing import Final

import pytest

from tract import cli

SOURCE: Final[str] = inspect.getsource(cli._cmd_assign)


def _line_of(pattern: str) -> int:
    """Line index within _cmd_assign of the first line matching `pattern`."""
    for index, line in enumerate(SOURCE.splitlines()):
        if re.search(pattern, line):
            return index
    raise AssertionError(f"no line in _cmd_assign matches {pattern!r}")


class TestCheapValidationComesFirst:
    def test_the_file_check_precedes_the_predictor_construction(self) -> None:
        """A typo must not cost a model load. This is what caught the auditor."""
        assert _line_of(r"file_path\.exists\(\)") < _line_of(r"TRACTPredictor\("), (
            "_cmd_assign constructs the predictor before checking that --file "
            "exists, so a mistyped path loads ~1.3 GB first."
        )

    def test_the_size_check_precedes_the_predictor_construction(self) -> None:
        assert _line_of(r"st_size") < _line_of(r"TRACTPredictor\(")

    def test_the_file_check_precedes_the_model_resolution(self) -> None:
        """Resolution can download. Downloading for a typo is worse than loading."""
        assert _line_of(r"file_path\.exists\(\)") < _line_of(
            r"_resolve_model_or_exit\("
        )


class TestTheRuntimeGuardPrecedesTheImportItGuards:
    def test_the_guard_runs_before_the_inference_import(self) -> None:
        """Otherwise the import raises first and the guard never speaks."""
        assert _line_of(r"_require_inference_runtime\(\)") < _line_of(
            r"from tract\.inference import"
        ), (
            "_cmd_assign imports tract.inference before calling the guard, so "
            "a base install gets ModuleNotFoundError and exit 1 rather than "
            "the documented exit 5 and its install instruction."
        )


class TestTheGuardChecksEverythingTheImportNeeds:
    def test_it_checks_numpy(self) -> None:
        """tract/inference.py imports numpy at module level.

        numpy lives in the phase0 extra, so it is exactly as absent as torch on
        a base install -- and the guard named only torch and
        sentence_transformers.
        """
        guard = inspect.getsource(cli._require_inference_runtime)
        assert "numpy" in guard, (
            "the guard does not check numpy, which tract.inference imports at "
            "module level, so the documented exit 5 is unreachable for it."
        )

    @pytest.mark.parametrize("module", ["torch", "sentence_transformers", "numpy"])
    def test_every_module_the_import_chain_needs_is_named(
        self, module: str
    ) -> None:
        assert module in inspect.getsource(cli._require_inference_runtime)

    def test_the_guard_exits_with_the_documented_code(self) -> None:
        guard = inspect.getsource(cli._require_inference_runtime)
        assert "EXIT_MISSING_RUNTIME" in guard
        assert cli.EXIT_MISSING_RUNTIME == 5


class TestTheGuardActuallyFires:
    """Exercise it, rather than only reading it."""

    def test_a_missing_module_exits_five(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.util

        real = importlib.util.find_spec

        def fake(name: str, *args: object, **kwargs: object) -> object | None:
            return None if name == "numpy" else real(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(importlib.util, "find_spec", fake)
        with pytest.raises(SystemExit) as excinfo:
            cli._require_inference_runtime()
        assert excinfo.value.code == cli.EXIT_MISSING_RUNTIME

    def test_it_returns_quietly_when_everything_is_present(self) -> None:
        """Guards the guard: a function that always exits would pass above."""
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        pytest.importorskip("numpy")
        cli._require_inference_runtime()
