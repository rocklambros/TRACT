"""Every parser must declare its own prose floor.

`BaseParser.min_prose_fraction` defaults to 0.0, and `run()` refuses a parse
whose honest prose fraction falls below it. At 0.0 nothing can fall below it,
so a parser that never overrides the default has the gate switched off and the
source can degrade to bare titles without anything failing.

That was the state of the fleet: thirteen parsers declared a floor and nineteen
did not, which made the coverage a property of who happened to remember rather
than of anything enforced. This file is the enforcement. A parser added
tomorrow without a floor fails here instead of shipping with the gate off.

Read with `ast` rather than by importing, for the reason
tests/test_prose_reachability.py gives: five parsers need pdfplumber, two need
defusedxml, one needs openpyxl, and a test that cannot run without the full
parse toolchain is a test that does not run. The declaration is a literal in a
class body, so the source says everything needed and no `data/raw/` has to
exist. That matters because CI has no `data/raw/` at all.

The value must be a literal. A floor hidden behind a module constant is not
readable at the point a reviewer looks, and the whole purpose of the number is
to be read next to the parser it governs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from tract.config import PARSERS_DIR
from tract.parsers.base import BaseParser

# The off position. A parser sitting here has no prose gate, because no
# fraction can be less than zero.
GATE_OFF: Final[float] = 0.0

# honest_prose_fraction() returns honest / measurable, so 1.0 is the maximum
# attainable value. A floor above it can never be met and would make the parser
# permanently unrunnable.
MAX_ATTAINABLE: Final[float] = 1.0

# Guards the scan against going quiet. If the glob, the directory or the naming
# convention moves, this fails loudly rather than the loops below simply having
# nothing to iterate. 32 parsers exist today.
MIN_PARSER_FILES: Final[int] = 30

_BASE_CLASS_NAME: Final[str] = "BaseParser"
_ATTRIBUTE: Final[str] = "min_prose_fraction"


def _parser_files() -> list[Path]:
    """Every parser module, derived from the tree rather than listed."""
    return sorted(PARSERS_DIR.glob("parse_*.py"))


def _base_names(node: ast.ClassDef) -> set[str]:
    """The bare names this class inherits from.

    Attribute bases such as `base.BaseParser` resolve to their final segment,
    so an import-style change does not make a parser invisible to this scan.
    """
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _parser_classes(tree: ast.Module) -> list[ast.ClassDef]:
    """Classes in one module that reach BaseParser, ancestors included.

    Transitive on purpose. A module is free to define a shared intermediate
    base and two concrete parsers under it, and every one of those is a class
    whose floor governs a real parse.
    """
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    parser_names = {_BASE_CLASS_NAME}
    # Repeat to a fixed point rather than assuming a base is defined before its
    # subclass. Source order is not a guarantee.
    changed = True
    while changed:
        changed = False
        for node in classes:
            if node.name in parser_names:
                continue
            if _base_names(node) & parser_names:
                parser_names.add(node.name)
                changed = True
    return [n for n in classes if n.name in parser_names]


def _own_floor(node: ast.ClassDef) -> float | None:
    """This class's own literal floor, or None when it declares none.

    Raises:
        ValueError: If the class assigns the attribute something other than a
            numeric literal, which hides the floor from a reader of the class.
    """
    for statement in node.body:
        targets: list[ast.expr] = []
        if isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        elif isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        else:
            continue

        if not any(
            isinstance(t, ast.Name) and t.id == _ATTRIBUTE for t in targets
        ):
            continue

        value = statement.value
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            return float(value.value)
        raise ValueError(
            f"{node.name} assigns {_ATTRIBUTE} something that is not a numeric "
            f"literal. Inline the measured value so it is readable and "
            f"diffable in the class body next to the parser it governs."
        )
    return None


def _resolved_floors() -> dict[tuple[str, str], float | None]:
    """(module, class) -> the floor in force, following local inheritance.

    A class with no declaration of its own inherits from an ancestor defined in
    the same module. None means nothing in that chain declared one, so the
    class falls through to the BaseParser default.
    """
    floors: dict[tuple[str, str], float | None] = {}
    for path in _parser_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = _parser_classes(tree)
        own = {node.name: _own_floor(node) for node in classes}
        for node in classes:
            resolved = own[node.name]
            seen = {node.name}
            current = node
            while resolved is None:
                ancestors = [
                    c for c in classes
                    if c.name in _base_names(current) and c.name not in seen
                ]
                if not ancestors:
                    break
                current = ancestors[0]
                seen.add(current.name)
                resolved = own[current.name]
            floors[(path.name, node.name)] = resolved
    return floors


def test_the_scan_covers_the_parser_fleet() -> None:
    """Guards every assertion below against silently scanning nothing."""
    files = _parser_files()
    assert len(files) >= MIN_PARSER_FILES, (
        f"only {len(files)} parser module(s) found under {PARSERS_DIR}. The "
        f"glob or the naming convention moved and this file stopped covering "
        f"what it claims to."
    )


def test_every_parser_module_defines_a_parser_class() -> None:
    """A module this scan cannot see is a module it cannot hold to a floor."""
    empty = [
        path.name for path in _parser_files()
        if not _parser_classes(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not empty, (
        f"{len(empty)} parser module(s) define no class reaching "
        f"{_BASE_CLASS_NAME}: {empty}. Either the module is misnamed or the "
        f"base class was renamed, and in both cases the prose gate is no "
        f"longer checked for it."
    )


def test_the_inherited_default_is_still_the_off_position() -> None:
    """Pins what the assertion below is measuring against.

    If the base default ever moves off 0.0, "declares a floor" stops meaning
    "the gate is on" and this file has to be rewritten rather than quietly
    keep passing.
    """
    assert BaseParser.min_prose_fraction == GATE_OFF, (
        f"{_BASE_CLASS_NAME}.{_ATTRIBUTE} is "
        f"{BaseParser.min_prose_fraction}, not {GATE_OFF}. The default is the "
        f"off position by design, so a parser that declares nothing is "
        f"visibly ungated rather than partially gated."
    )


def test_every_parser_declares_a_floor_above_the_default() -> None:
    """The gap this file exists to keep closed."""
    floors = _resolved_floors()
    ungated = sorted(
        f"{module}:{cls}" for (module, cls), floor in floors.items()
        if floor is None or floor <= GATE_OFF
    )
    assert not ungated, (
        f"{len(ungated)} parser class(es) leave {_ATTRIBUTE} at the inherited "
        f"default of {GATE_OFF}, which switches the prose gate off entirely "
        f"and lets the source degrade to bare titles without failing: "
        f"{ungated}. Measure the parser's honest prose fraction, declare it "
        f"rounded down to two decimal places, and comment what the floor "
        f"protects against."
    )


def test_no_parser_declares_a_floor_it_can_never_meet() -> None:
    """A floor above 1.0 is a typo that makes the parser unrunnable.

    honest_prose_fraction() divides honest controls by measurable controls, so
    1.0 is the ceiling and run() would raise on every correct parse.
    """
    floors = _resolved_floors()
    unreachable = sorted(
        f"{module}:{cls}={floor}" for (module, cls), floor in floors.items()
        if floor is not None and floor > MAX_ATTAINABLE
    )
    assert not unreachable, (
        f"{len(unreachable)} parser class(es) declare a {_ATTRIBUTE} above "
        f"{MAX_ATTAINABLE}, which no parse can reach: {unreachable}."
    )
