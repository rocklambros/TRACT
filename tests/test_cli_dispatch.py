"""`main()`'s dispatch dict was covered by nothing, and it maps names to handlers.

A product-surface audit found 14 of 20 subcommands had no test reaching their
real handler, and the dispatch dict at the bottom of `main()` had none at all --
the suite's only `main()` call is `cli.main(["--version"])`, which exits inside
argparse before dispatch. A wrong name-to-function pairing there would pass the
entire suite while sending every user of that command somewhere else.

That gap is why two commands shipped unable to run:

  publish-hf      called _load_fold_results with two of three required args
  publish-dataset queried two columns the published database does not have

Both had parser tests. Neither had a test that reached the code.

These tests do not execute handlers -- most load models, hit the network, or
publish. They assert the WIRING: every subcommand the parser accepts has an
entry, every entry names a real function, and every entry points at the handler
whose name matches. That is cheap, it is what was missing, and it is what would
have caught a mis-paired row.
"""

from __future__ import annotations

import inspect
import re
from typing import Final

import pytest

from tract import cli
from tract.config import PROJECT_ROOT

# Every subcommand the parser accepts, taken from the parser itself rather than
# from a list here -- a hand-maintained copy would drift and hide the gap.
def _parser_subcommands() -> frozenset[str]:
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        source = (PROJECT_ROOT / "tract" / "cli.py").read_text(encoding="utf-8")
        return frozenset(re.findall(r'add_parser\(\s*"([a-z0-9-]+)"', source))
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        if hasattr(action, "choices"):
            return frozenset(action.choices)
    raise AssertionError("no subparser action found")


def _dispatch_table() -> dict[str, str]:
    """The name -> handler-name mapping, read out of main()'s source."""
    source = inspect.getsource(cli.main)
    block = source[source.index("handlers = {"):]
    block = block[: block.index("}")]
    return dict(re.findall(r'"([a-z0-9-]+)":\s*(_cmd_[a-z_]+)', block))


SUBCOMMANDS: Final[frozenset[str]] = _parser_subcommands()
DISPATCH: Final[dict[str, str]] = _dispatch_table()


class TestTheTableWasReadCorrectly:
    """Guards the guard: an empty parse would make every test below vacuous."""

    def test_the_parser_exposes_subcommands(self) -> None:
        assert len(SUBCOMMANDS) >= 15, f"only found {sorted(SUBCOMMANDS)}"

    def test_the_dispatch_table_was_parsed(self) -> None:
        assert len(DISPATCH) >= 15, f"only found {sorted(DISPATCH)}"


class TestEverySubcommandDispatches:
    def test_no_subcommand_is_missing_a_handler(self) -> None:
        """A subcommand the parser accepts and main() ignores exits silently."""
        missing = sorted(SUBCOMMANDS - set(DISPATCH))
        assert not missing, (
            f"These subcommands parse but have no handler entry, so they "
            f"accept their arguments and then do nothing: {missing}"
        )

    def test_no_handler_entry_is_for_an_unknown_subcommand(self) -> None:
        """A dead entry means a command was renamed and the table was not."""
        stray = sorted(set(DISPATCH) - SUBCOMMANDS)
        assert not stray, (
            f"These handler entries name subcommands the parser does not "
            f"accept, so they are unreachable: {stray}"
        )

    @pytest.mark.parametrize("command", sorted(DISPATCH))
    def test_the_handler_exists_and_is_callable(self, command: str) -> None:
        handler = getattr(cli, DISPATCH[command], None)
        assert handler is not None, (
            f"{command!r} dispatches to {DISPATCH[command]}, which does not "
            "exist in tract.cli."
        )
        assert callable(handler)

    @pytest.mark.parametrize("command", sorted(DISPATCH))
    def test_the_handler_name_matches_the_subcommand(self, command: str) -> None:
        """The mis-pairing this file exists for.

        `"export": _cmd_export_canonical` would pass every other test here and
        send every `tract export` user to the wrong code.
        """
        expected = "_cmd_" + command.replace("-", "_")
        assert DISPATCH[command] == expected, (
            f"{command!r} dispatches to {DISPATCH[command]}, not {expected}. "
            "If that is deliberate, the pairing needs a comment saying why."
        )

    @pytest.mark.parametrize("command", sorted(DISPATCH))
    def test_the_handler_takes_the_parsed_arguments(self, command: str) -> None:
        handler = getattr(cli, DISPATCH[command])
        params = inspect.signature(handler).parameters
        assert len(params) == 1, (
            f"{DISPATCH[command]} takes {len(params)} parameters; main() calls "
            "it with exactly one (the parsed argparse namespace)."
        )


class TestHandlerCallsResolve:
    """The publish-hf failure mode, checked across every handler.

    `_cmd_publish_hf` called `_load_fold_results` with two arguments where the
    callee requires three and raises otherwise, so the command could not run.
    Nothing caught it because no test reached the handler.
    """

    @pytest.mark.parametrize("command", sorted(DISPATCH))
    def test_the_handler_body_has_no_obvious_arity_mismatch(
        self, command: str
    ) -> None:
        handler = getattr(cli, DISPATCH[command])
        source = inspect.getsource(handler)
        module = inspect.getmodule(handler)
        assert module is not None

        problems: list[str] = []
        for name in re.findall(r"\b(_[a-z_]+)\(", source):
            callee = getattr(module, name, None)
            if callee is None or not callable(callee) or name == handler.__name__:
                continue
            try:
                sig = inspect.signature(callee)
            except (TypeError, ValueError):
                continue
            required = [
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            # Count the arguments at each call site of `name` in this handler.
            for call in re.finditer(rf"\b{re.escape(name)}\(([^()]*)\)", source):
                args = [a for a in call.group(1).split(",") if a.strip()]
                if args and len(args) < len(required):
                    problems.append(
                        f"{name}(...) called with {len(args)} of "
                        f"{len(required)} required"
                    )
        assert not problems, (
            f"{DISPATCH[command]} has call sites that cannot satisfy their "
            f"callee: {problems}"
        )
