"""Every documented command must at least parse as documented.

A product-surface audit found four documented invocations that fail on the
command line as written:

    tract publish-hf --repo-id X --dry-run     -> exit 2, --zero-shot-results
                                                  is required (made so here,
                                                  and the docs were not updated)
    tract review-validate <path>               -> exit 2, needs --input
    tract review-import <path>                 -> exit 2, needs --input and
                                                  --reviewer
    README quick start step 2                  -> exit 1, `prepare` names its
                                                  output from the input stem,
                                                  so the file the README then
                                                  validates does not exist

Three of the four are documentation that describes an older interface. The
fourth is a filename the docs guessed. All four fail for a new user on their
first attempt, and the publish-hf one was introduced by a change made in this
repository that did not update its own docs.

Parsing is a low bar and it is the bar that was being missed. These tests
extract every `tract ...` line from the documentation and run it through the
real parser with a stub that stops before any handler executes -- so nothing
downloads, loads a model, or publishes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

from tract import cli
from tract.config import PROJECT_ROOT

DOCS: Final[tuple[Path, ...]] = (
    PROJECT_ROOT / "CLAUDE.md",
    PROJECT_ROOT / "README.md",
)

# Lines that are illustrative rather than runnable: they carry a placeholder the
# reader is expected to replace. A placeholder is fine; a MISSING REQUIRED FLAG
# is not, and that is the difference these tests are drawing.
PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


def _documented_invocations() -> list[tuple[str, str]]:
    """(source file, command line) for every `tract ...` line in the docs."""
    found: list[tuple[str, str]] = []
    for doc in DOCS:
        if not doc.is_file():
            continue
        for line in doc.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("tract "):
                continue
            # Drop a trailing comment; keep the command.
            command = stripped.split("#", 1)[0].strip()
            if command and command != "tract":
                found.append((doc.name, command))
    return found


INVOCATIONS: Final[list[tuple[str, str]]] = _documented_invocations()


# What a <placeholder> becomes. "1" parses as str, int, float and Path, so one
# substitute satisfies every argparse type the CLI uses. A word like
# "PLACEHOLDER" fails on --gpu-hours, which takes a float -- and that failure is
# the test's own brittleness, not a defect in the documentation.
PLACEHOLDER_VALUE: Final[str] = "1"


def _argv(command: str) -> list[str]:
    """Split a documented line into argv, substituting placeholders."""
    import shlex

    parts = shlex.split(command)[1:]  # drop the leading "tract"
    return [
        (PLACEHOLDER_VALUE if PLACEHOLDER.fullmatch(part) else part)
        for part in parts
    ]


class TestTheDocsWereRead:
    """Guards the guard: an empty scrape makes every test below vacuous."""

    def test_documented_commands_were_found(self) -> None:
        assert len(INVOCATIONS) >= 15, (
            f"only {len(INVOCATIONS)} documented commands found; the scrape "
            "is probably broken rather than the docs being empty"
        )

    def test_both_documents_contribute(self) -> None:
        sources = {source for source, _ in INVOCATIONS}
        assert sources == {"CLAUDE.md", "README.md"}, sources


class TestEveryDocumentedInvocationParses:
    @pytest.mark.parametrize(
        "source,command",
        INVOCATIONS,
        ids=[f"{s}:{c[:48]}" for s, c in INVOCATIONS],
    )
    def test_it_parses(
        self, source: str, command: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parse only. The handler is stubbed, so nothing runs.

        A SystemExit of 2 is argparse rejecting the documented line -- a missing
        required argument, an unknown flag, or a renamed subcommand. That is the
        defect this file exists for.
        """
        called: list[str] = []

        def _stub(args: object) -> None:
            called.append(getattr(args, "command", "?"))

        # Replace every handler so main() dispatches into a no-op.
        for name in dir(cli):
            if name.startswith("_cmd_") and callable(getattr(cli, name)):
                monkeypatch.setattr(cli, name, _stub)

        try:
            cli.main(_argv(command))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            assert code != 2, (
                f"{source} documents `{command}`, which argparse rejects with "
                f"exit 2. The documentation describes an interface the CLI "
                f"does not have."
            )
        except Exception as exc:  # noqa: BLE001 - a stubbed handler should not raise
            pytest.fail(
                f"{source} documents `{command}`, which raised "
                f"{type(exc).__name__} before reaching a handler: {exc}"
            )


class TestTheReadmeQuickStartIsSelfConsistent:
    """The first thing a new user runs, and it referred to a file that is
    never created.

    `prepare` names its output from the INPUT stem, not from --framework-id, so
    `--file examples/sample_framework.csv --framework-id demo` writes
    `sample_framework_prepared.json` and the README's next line validated
    `demo_prepared.json`.
    """

    def test_the_validated_filename_matches_what_prepare_writes(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        prepare = re.search(r"tract prepare --file (\S+)", readme)
        validate = re.search(r"tract validate --file (\S+)", readme)
        assert prepare and validate, "the quick start no longer has both steps"

        source = Path(prepare.group(1))
        expected = source.with_name(f"{source.stem}_prepared.json")
        assert validate.group(1) == str(expected), (
            f"README prepares {source} -- which writes {expected} -- and then "
            f"validates {validate.group(1)}, a file nothing creates."
        )

    def test_the_input_file_exists(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"tract prepare --file (\S+)", readme)
        assert match
        assert (PROJECT_ROOT / match.group(1)).is_file(), (
            f"the quick start's input file {match.group(1)} does not exist"
        )
