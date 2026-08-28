"""A process probe must ask for an identity, never for a string. Twice it did not.

TRACT's RunPod reaper decides whether to terminate a billing GPU fleet by
asking the process table two questions. Both questions came back about the
wrong process, and the second was written five hours after the first was
fixed, in the same file, by the same author, who had just spent a morning on
the first.

INCIDENT 1 -- written 2026-08-26 in 47ddcd2, fixed 2026-08-27 in cca5517.
`reaper_guard.orchestrator_pids`. The guard stands down while the orchestrator
is alive, on the theory that a live orchestrator cleans up after itself. It
decided "alive" by testing `any("runpod_parallel" in arg for arg in argv)`
over every process in /proc -- and a FILENAME is an argv element.
`tail -f results/phase1b/runpod_parallel.log`, which is exactly how an operator
watches a run, classified as a live fleet driver. The reaper stood down for as
long as that terminal stayed open, and it was the only automatic bound on
spend. Reproduced before fixing: `assert 362512 not in [362512]`.

INCIDENT 2 -- written 2026-08-27 in a60cb05, fixed 2026-08-28 in d4eb5db.
`reaper_guard.pod_training_state` asked each pod whether it was still training
by running `pgrep -f run_fold` over SSH. The shell running that pgrep carries
"run_fold" in its own command line, so pgrep found itself. Verified on a live
pod: the only match was `bash -c ... pgrep -af run_fold`, while the only real
python process was supervisord. Every pod reported BUSY unconditionally --
including one that had never received the repository and could not possibly
train. A probe stuck on BUSY means the guard never reaps, so the protection
against killing live work would have protected a dead fleet forever at four
GPUs an hour. Four fleets were provisioned that night, zero folds trained, and
about $40 burned.

WHAT THE TWO HAVE IN COMMON, AND WHY A FAKE PROCESS LIST NEVER FINDS IT. Both
probes asked "does this string appear somewhere in a command line" when the
question was "is this the process I mean". A unit test cannot catch that,
because it is tested against a hand-built process table, and a hand-built
process table only disagrees with the broken code if whoever wrote it had
already thought of the case -- a log reader holding the filename, a probing
shell holding its own pattern. Nobody invents the case they just failed to
imagine. So this file does not model the process table at all. It reads the
source of scripts/ and tract/ and objects to the SHAPE.

CONTROL A -- SELF-MATCH. Every construct that interrogates the process table
(pgrep, pkill, killall, ps piped to grep, a /proc enumeration, psutil) must
exclude its own pid explicitly -- $$ in shell, os.getpid() in Python -- or
carry a written exemption. This is the control the premortem asked for and it
catches incident 2 exactly.

It would NOT have caught incident 1, and that is stated here because the next
person will assume otherwise: the broken orchestrator_pids already opened with
`me = os.getpid()` and skipped that pid. Its own identity was never the
problem; a third party holding the pattern was. Verified against the real
pre-fix source at cca5517^ -- Control A reports it clean.

CONTROL B -- SUBSTRING IDENTITY. Inside any scope that enumerates /proc, a
membership test whose left operand is a string literal, or whose right operand
is a `.join(...)` of a command line, is a violation. That is incident 1's exact
shape, and it is checked through the AST rather than a regex so that the shell
`case "${exe##*/}" in python*)` living inside a Python string in the fixed pod
probe is not mistaken for one. A pid exclusion does not clear a Control B site:
excluding yourself is no answer to a probe that cannot tell two other processes
apart. Only a written exemption does.

BOTH CONTROLS WERE RUN AGAINST THE COMMITS THAT WROTE THE DEFECTS, which is
the only evidence that matters for a control invented after the fact:

    47ddcd2  incident 1 as written   Control B: 2 violations (lines 96, 98)
    a60cb05  incident 2 as written   Control A: 1 violation  (line 418)
    HEAD     162 files scanned       0 violations

Both would have failed CI on the day they were written. The equivalent of each
run is a test below, done by mutating the current reaper_guard.py in memory
rather than by naming a commit, so that a rebase cannot quietly turn the proof
into a skip.

TWO DESIGN CHOICES KEEP THIS FROM BECOMING NOISE.

First, prose is not code. Comments and docstrings are blanked before the scan,
because the most careful thing in reaper_guard.py is the comment block quoting
`pgrep -f run_fold` and explaining why it is gone. A check that punished that
comment would teach people to delete the incident record, which is the exact
opposite of the point. This docstring quotes the offending commands too, for
the same reason, and is likewise not scanned -- tests/ is outside the scanned
roots regardless.

Second, the escape hatch demands a sentence, not a pragma. A bare
`# process-probe-exempt:` does not clear a site; the marker has to be followed
by a real reason, so the hatch costs about as much thought as the fix.

Owner: TRACT
"""
from __future__ import annotations

import ast
import bisect
import io
import re
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tract import config

REPO_ROOT: Final[Path] = config.PROJECT_ROOT

# Where process probes actually live. scripts/ holds the fleet orchestrator and
# the reaper; tract/ is the library both import. Adding a root here is the only
# change needed to widen the control.
SCANNED_ROOTS: Final[tuple[str, ...]] = ("scripts", "tract")

# Python and shell, because both incidents crossed that line: the defect was
# written in Python and executed as a shell command on the far side of an SSH
# connection.
SCANNED_SUFFIXES: Final[tuple[str, ...]] = (".py", ".sh", ".bash")

SKIPPED_DIRECTORY_NAMES: Final[frozenset[str]] = frozenset(
    {"__pycache__", ".ipynb_checkpoints", ".git", ".mypy_cache", ".pytest_cache"}
)

# How far from a probe a self-exclusion may sit and still count. For Python the
# enclosing function is the scope, which is the honest unit -- a getpid() three
# functions away protects nothing. This window is the fallback for shell files
# and for module-level code, where no function body delimits the probe.
PROBE_SCOPE_LINES: Final[int] = 20

# An exemption is usually written immediately above the def it applies to,
# which is outside the function body the scope is built from.
EXEMPTION_LOOKBACK_LINES: Final[int] = 5

EXEMPTION_MARKER: Final[str] = "process-probe-exempt:"

# Long enough that "n/a", "safe", or "ok" will not do. The hatch is meant to
# cost a sentence of thought.
MIN_EXEMPTION_REASON_CHARS: Final[int] = 20

# How much of the offending source line to echo back in the failure message.
SNIPPET_CHARS: Final[int] = 140

# A floor on discovery, not a census. 162 files match today; the number is far
# enough below that to survive ordinary churn, and its only job is to catch a
# scan that has quietly stopped finding anything -- the failure mode where a
# hygiene test passes forever because it reads zero files.
MIN_EXPECTED_SCANNED_FILES: Final[int] = 100

# Characters that separate a command from its arguments differently in Python
# source than on a command line. Blanking them lets one pattern match both
# `subprocess.run(["pgrep", "-f", pat])` and the string `"pgrep -f pat"`,
# whether or not the list is spread across several lines.
_NORMALISED_TO_SPACE: Final[str] = "\"'`,\n\r\t"

# Every way this repo could ask the operating system "which processes match
# this string". The rule names appear verbatim in failure messages.
_PROBE_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("pgrep/pkill", re.compile(r"\b(?:pgrep|pkill)\b")),
    ("killall", re.compile(r"\bkillall\b")),
    ("ps-piped-to-grep", re.compile(r"\bps\b[^|]{0,80}\|\s*e?grep\b")),
    (
        "proc-table-scan",
        re.compile(
            r"\b(?:listdir|scandir|iterdir|iglob|glob|walk)\b.{0,40}?/proc\b"
            r"|/proc\b.{0,40}?\b(?:listdir|scandir|iterdir|iglob|glob|walk)\b"
            r"|/proc/\[[0-9]"
            r"|/proc/\*"
        ),
    ),
    ("psutil-process-iter", re.compile(r"\bprocess_iter\b|\bpsutil\s*\.\s*pids\b")),
)

# The rule name Control B reports under. It is deliberately not in
# _PROBE_RULES: Control B is an AST walk, not a text match, because the fixed
# pod probe contains the shell fragment `case "${exe##*/}" in python*)` inside
# a Python string, and any regex for `"literal" in name` reads that as a
# violation. The AST does not, because a Str constant has no Compare node in it.
ARGV_SUBSTRING_RULE: Final[str] = "argv-substring-identity"

# A /proc scan may only enumerate the process table; deciding WHICH process it
# found by substring is incident 1. Anything scanning /proc is in scope for B.
PROC_SCAN_RULE: Final[str] = "proc-table-scan"

# Which rules a pid exclusion actually answers. Control B is absent on purpose:
# `me = os.getpid()` was already present, and correct, in the code that caused
# incident 1. Excluding yourself says nothing about whether you can tell a log
# reader from an orchestrator.
SELF_EXCLUDABLE_RULES: Final[frozenset[str]] = frozenset(
    {"pgrep/pkill", "killall", "ps-piped-to-grep", PROC_SCAN_RULE, "psutil-process-iter"}
)

# Ways of saying "not me". Each is an explicit identity check, which is what
# both incidents lacked: the broken probes matched on a string and never asked
# whose process they had found.
_SELF_EXCLUSION_MARKERS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("os.getpid()", re.compile(r"\bgetpid\s*\(\s*\)")),
    ("os.getppid()", re.compile(r"\bgetppid\s*\(\s*\)")),
    ("$$", re.compile(r"\$\$")),
    ("$BASHPID", re.compile(r"\$BASHPID\b")),
    ("$PPID", re.compile(r"\$PPID\b")),
    ("psutil.Process()/current_process()", re.compile(r"\bcurrent_process\s*\(\s*\)")),
)

# A shell comment is a `#` at the start of a line or after whitespace. Anything
# else -- `${p##*/}`, `$#` -- is parameter expansion and must survive, because
# the fixed pod probe is written in exactly that dialect.
_SHELL_COMMENT: Final[re.Pattern[str]] = re.compile(r"(?:^|\s)#")


@dataclass(frozen=True)
class ProbeSite:
    """One place in the source that interrogates the process table."""

    path: Path
    line: int
    rule: str
    snippet: str
    scope: tuple[int, int]
    self_exclusion: str | None
    exemption: str | None
    saw_unusable_exemption: bool

    @property
    def is_violation(self) -> bool:
        """True when this probe can answer about a process it did not mean."""
        if self.exemption is not None:
            return False
        return not (self.rule in SELF_EXCLUDABLE_RULES and self.self_exclusion is not None)

    def describe(self) -> str:
        """One paragraph a reader can act on without opening the file."""
        relative = (
            self.path.relative_to(REPO_ROOT)
            if self.path.is_relative_to(REPO_ROOT)
            else self.path
        )
        hint = (
            "    an exemption marker is present but its reason is shorter than "
            f"{MIN_EXEMPTION_REASON_CHARS} characters\n"
            if self.saw_unusable_exemption
            else ""
        )
        if self.rule in SELF_EXCLUDABLE_RULES:
            remedy = (
                f"    no self-exclusion in lines {self.scope[0]}-{self.scope[1]} "
                f"(looked for: {', '.join(name for name, _ in _SELF_EXCLUSION_MARKERS)})\n"
            )
        else:
            remedy = (
                "    substring identity test inside a /proc scan; compare argv "
                "elements exactly instead, the way _is_orchestrator_argv does. "
                "A pid exclusion does not clear this.\n"
            )
        return f"  {relative}:{self.line}  [{self.rule}]\n    {self.snippet}\n{hint}{remedy}"


def _line_starts(text: str) -> list[int]:
    """Character offset of the first character of each line, 0-indexed list."""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _line_of(starts: list[int], offset: int) -> int:
    """The 1-based line number containing *offset*."""
    return bisect.bisect_right(starts, offset)


def _blank(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Replace each span with spaces, keeping newlines so line numbers hold."""
    chars = list(text)
    for start, end in spans:
        for index in range(max(start, 0), min(end, len(chars))):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _parse(path: Path, text: str) -> ast.Module:
    """The AST of *path*, or a loud failure.

    A file in a scanned root that cannot be parsed must not be scanned as if
    it were clean: that would be a blind spot the size of a whole file, and
    the guard lives in exactly the kind of file people edit at 2am.
    """
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"{path} is in a scanned root but does not parse as Python: {exc}") from exc


def _python_prose(
    path: Path, text: str, starts: list[int], tree: ast.Module
) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """Spans of comments and docstrings, plus every comment keyed by line.

    Docstrings count as prose for the same reason comments do: the clearest
    account of incident 1 lives in `orchestrator_pids`' docstring, which says
    the word pgrep three times while the function deliberately does not shell
    to it. A check that could not tell an explanation from an instruction would
    make the explanation unwritable.

    Comments come from tokenize rather than a regex, and that is what keeps the
    fixed pod probe readable: its shell contains `${p##*/}` and `$#`, and any
    hand-rolled `#`-to-end-of-line rule that survived those would be one
    special case away from eating the `$$` that makes the probe safe.
    """
    spans: list[tuple[int, int]] = []
    comments: dict[int, str] = {}

    docstring_owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, docstring_owners):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
            continue
        if not isinstance(first.value.value, str):
            continue
        literal = first.value
        end_lineno = literal.end_lineno if literal.end_lineno is not None else literal.lineno
        end_col = literal.end_col_offset if literal.end_col_offset is not None else 0
        spans.append(
            (starts[literal.lineno - 1] + literal.col_offset, starts[end_lineno - 1] + end_col)
        )

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError(f"{path} is in a scanned root but could not be tokenised: {exc}") from exc

    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row, col = token.start
        end_row, end_col = token.end
        spans.append((starts[row - 1] + col, starts[end_row - 1] + end_col))
        comments[row] = comments.get(row, "") + token.string

    return spans, comments


def _shell_prose(text: str, starts: list[int]) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """Comment spans for a shell script, plus every comment keyed by line.

    There is no tokenizer here, so `# inside a quoted string` is read as a
    comment and blanked. The consequence is a missed probe, never a false
    alarm, and it is bounded: every line of shell TRACT actually sends to a pod
    is a string literal inside a .py file, which goes through tokenize above
    and gets this right. This path covers the two standalone .sh files, which
    talk to runpodctl rather than to the process table.
    """
    spans: list[tuple[int, int]] = []
    comments: dict[int, str] = {}
    for index, line in enumerate(text.split("\n")):
        found = _SHELL_COMMENT.search(line)
        if found is None:
            continue
        column = found.end() - 1
        spans.append((starts[index] + column, starts[index] + len(line)))
        comments[index + 1] = line[column:]
    return spans, comments


def _function_spans(tree: ast.Module) -> list[tuple[int, int]]:
    """Line ranges of every Python function, innermost resolved by the caller."""
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno if node.end_lineno is not None else node.lineno))
    return spans


def _argv_substring_lines(tree: ast.Module) -> list[int]:
    """Lines holding a membership test that decides identity by substring.

    Two shapes, both taken from the code that caused incident 1:

        any("runpod_parallel" in arg for arg in argv)   -- literal on the left
        "reaper_guard" in " ".join(argv)                -- argv flattened first

    The second is the more insidious: joining argv destroys the element
    boundaries that make an exact comparison possible, so any test downstream
    of a join is a substring test whatever it looks like.

    `arg in ALL_ACTIONS` is untouched, and deliberately -- exact membership in
    a known set is the answer here, not the problem. That line survived the
    incident 1 fix unchanged.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.In):
            continue
        left_is_literal = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
        right = node.comparators[0]
        right_is_join = (
            isinstance(right, ast.Call)
            and isinstance(right.func, ast.Attribute)
            and right.func.attr == "join"
        )
        if left_is_literal or right_is_join:
            lines.add(node.lineno)
    return sorted(lines)


def _scope_for(line: int, spans: list[tuple[int, int]], last_line: int) -> tuple[int, int]:
    """The innermost function containing *line*, or a window around it."""
    containing = [span for span in spans if span[0] <= line <= span[1]]
    if containing:
        return max(containing, key=lambda span: span[0])
    return (max(1, line - PROBE_SCOPE_LINES), min(last_line, line + PROBE_SCOPE_LINES))


def _slice_lines(text: str, starts: list[int], scope: tuple[int, int]) -> str:
    """The substring of *text* covering the inclusive line range *scope*."""
    begin = starts[scope[0] - 1]
    end = starts[scope[1]] if scope[1] < len(starts) else len(text)
    return text[begin:end]


def _find_self_exclusion(normalised: str, starts: list[int], scope: tuple[int, int]) -> str | None:
    """The name of the first self-exclusion in *scope*, or None."""
    region = _slice_lines(normalised, starts, scope)
    for name, pattern in _SELF_EXCLUSION_MARKERS:
        if pattern.search(region):
            return name
    return None


def _find_exemption(comments: dict[int, str], scope: tuple[int, int]) -> tuple[str | None, bool]:
    """The written exemption covering *scope*, and whether an unusable one exists."""
    saw_marker = False
    low = scope[0] - EXEMPTION_LOOKBACK_LINES
    for line, comment in comments.items():
        if not low <= line <= scope[1]:
            continue
        position = comment.find(EXEMPTION_MARKER)
        if position < 0:
            continue
        saw_marker = True
        reason = comment[position + len(EXEMPTION_MARKER):].strip()
        if len(reason) >= MIN_EXEMPTION_REASON_CHARS:
            return reason, saw_marker
    return None, saw_marker


def _scan_text(path: Path, text: str) -> list[ProbeSite]:
    """Every process-table probe in *text*, each judged safe or not."""
    starts = _line_starts(text)
    tree: ast.Module | None = None
    if path.suffix == ".py":
        tree = _parse(path, text)
        spans, comments = _python_prose(path, text, starts, tree)
        scopes = _function_spans(tree)
    else:
        spans, comments = _shell_prose(text, starts)
        scopes = []

    code = _blank(text, spans)
    normalised = code.translate({ord(char): " " for char in _NORMALISED_TO_SPACE})
    source_lines = text.split("\n")
    last_line = len(source_lines)

    sites: dict[tuple[int, str], ProbeSite] = {}

    def record(line: int, rule: str) -> None:
        if (line, rule) in sites:
            return
        scope = _scope_for(line, scopes, last_line)
        exemption, saw_unusable = _find_exemption(comments, scope)
        sites[(line, rule)] = ProbeSite(
            path=path,
            line=line,
            rule=rule,
            snippet=source_lines[line - 1].strip()[:SNIPPET_CHARS],
            scope=scope,
            self_exclusion=_find_self_exclusion(normalised, starts, scope),
            exemption=exemption,
            saw_unusable_exemption=saw_unusable and exemption is None,
        )

    for rule, pattern in _PROBE_RULES:
        for match in pattern.finditer(normalised):
            record(_line_of(starts, match.start()), rule)

    # Control B, and it runs second because it needs Control A's answer first:
    # a substring identity test is only interesting where the code is deciding
    # which process it is looking at. Everywhere else in this repo `"x" in y`
    # is ordinary string handling, and flagging it would drown the file.
    if tree is not None:
        proc_scopes = [site.scope for site in sites.values() if site.rule == PROC_SCAN_RULE]
        for line in _argv_substring_lines(tree):
            if any(start <= line <= end for start, end in proc_scopes):
                record(line, ARGV_SUBSTRING_RULE)

    return sorted(sites.values(), key=lambda site: (str(site.path), site.line, site.rule))


def _scan_source(path: Path) -> list[ProbeSite]:
    """Every process-table probe in the file at *path*."""
    return _scan_text(path, path.read_text(encoding="utf-8"))


def _scanned_files() -> list[Path]:
    """Every file the control covers, sorted so failures are reproducible."""
    found: list[Path] = []
    for root in SCANNED_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            raise ValueError(f"scanned root {base} does not exist; SCANNED_ROOTS is stale")
        for path in base.rglob("*"):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            if SKIPPED_DIRECTORY_NAMES & set(path.parts):
                continue
            found.append(path)
    return sorted(found)


def _scan_tree() -> list[ProbeSite]:
    """Every process-table probe in the scanned roots."""
    return [site for path in _scanned_files() for site in _scan_source(path)]


def _write(tmp_path: Path, name: str, source: str) -> list[ProbeSite]:
    """Scan a constructed source file. Real sources are never edited to test."""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return _scan_source(path)


def _violations(sites: list[ProbeSite]) -> list[ProbeSite]:
    return [site for site in sites if site.is_violation]


# --------------------------------------------------------------------------
# The control itself.
# --------------------------------------------------------------------------


def test_repo_has_no_unsafe_process_probes() -> None:
    """No probe in scripts/ or tract/ can answer about a process it did not mean.

    A failure here names a third instance of the defect that was written twice
    on 2026-08-27. Read the site's own line for which control it broke.
    """
    violations = _violations(_scan_tree())
    assert not violations, (
        f"{len(violations)} process probe(s) can answer about the wrong "
        "process.\n\n"
        + "".join(site.describe() for site in violations)
        + "\nIf the probe genuinely cannot, say why in a comment containing "
        f"'{EXEMPTION_MARKER} <reason>' -- at least "
        f"{MIN_EXEMPTION_REASON_CHARS} characters of reason."
    )


def test_the_incident_sites_are_still_seen_and_still_safe() -> None:
    """The two probes that failed sit under the control, not beside it.

    Without this, the check above could pass by finding nothing at all -- one
    regex typo and a clean tree and a blind scanner are the same result. These
    assertions pin the control to the exact code that failed: reaper_guard
    interrogates the process table twice, once in Python and once in shell
    over SSH, and both must still be visible to the scanner and still carry
    their exclusion.
    """
    guard = REPO_ROOT / "scripts" / "phase1b" / "reaper_guard.py"
    sites = _scan_source(guard)

    assert sites, (
        f"{guard} contains two process-table scans and the scanner found "
        "neither; the detector is broken, not the guard"
    )
    assert not _violations(sites), "".join(site.describe() for site in _violations(sites))

    exclusions = {site.line: site.self_exclusion for site in sites}
    assert "os.getpid()" in exclusions.values(), (
        "orchestrator_pids no longer excludes its own pid. Note this is not "
        "incident 1 returning -- the code that caused incident 1 called "
        "os.getpid() and was wrong anyway; Control B is what guards that. This "
        f"is a new self-match hole in the same function. Sites: {exclusions}"
    )
    assert "$$" in exclusions.values(), (
        "the pod probe no longer excludes the probing shell's own pid -- this "
        f"is incident 2 (a60cb05, 2026-08-27) returning. Sites: {exclusions}"
    )


def test_removing_the_real_guards_exclusions_reintroduces_both_violations() -> None:
    """Delete both self-exclusions from the real file and Control A must object.

    The constructed sources further down prove the detector fires on the SHAPE
    of the defect. This proves it fires on the real thing, which is a different
    claim: it holds the scanner's scope rules, prose stripping and regexes
    against reaper_guard.py exactly as written, not against a miniature that
    flatters them.

    The mutation happens in memory. reaper_guard.py is never edited -- a test
    run killed halfway through must not be able to leave the only bound on GPU
    spend disarmed on disk.

    Note which `$$` this deletes. There are two in the file: one in the comment
    explaining the fix, one in the shell that does the work. Only the shell one
    goes, so a green result here would also mean the scanner had accepted a
    comment as a working exclusion.
    """
    guard = REPO_ROOT / "scripts" / "phase1b" / "reaper_guard.py"
    original = guard.read_text(encoding="utf-8")

    anchors = {"    me = os.getpid()\n": "    me = -1\n", '= \\"$$\\"': '= \\"0\\"'}
    mutated = original
    for anchor, replacement in anchors.items():
        if anchor not in mutated:
            raise ValueError(
                f"{guard} no longer contains {anchor!r}; this test's mutation is "
                "stale and would have passed without proving anything"
            )
        mutated = mutated.replace(anchor, replacement)

    violations = _violations(_scan_text(guard, mutated))
    assert [(site.line, site.rule) for site in violations] == [
        (318, "proc-table-scan"),
        (433, "proc-table-scan"),
    ], (
        "the real orchestrator_pids and pod_training_state probes were stripped "
        "of their self-exclusions and the control did not notice:\n"
        + "".join(site.describe() for site in violations)
    )
    assert guard.read_text(encoding="utf-8") == original


def test_restoring_the_real_substring_match_reintroduces_incident_one() -> None:
    """Put incident 1's identity test back into the real function; B must object.

    orchestrator_pids now delegates to _is_orchestrator_argv, which compares
    argv elements exactly. The line below replaces that call with the test the
    function actually shipped on 2026-08-26 -- `any("runpod_parallel" in arg
    for arg in argv)` -- which is what `tail -f runpod_parallel.log` walked
    into. Control A cannot see this: the function's `me = os.getpid()` is
    untouched and was never the problem.

    In memory only. reaper_guard.py is never edited.
    """
    guard = REPO_ROOT / "scripts" / "phase1b" / "reaper_guard.py"
    original = guard.read_text(encoding="utf-8")

    anchor = "        if _is_orchestrator_argv(_read_argv(pid)):\n"
    if anchor not in original:
        raise ValueError(
            f"{guard} no longer contains {anchor!r}; this test's mutation is "
            "stale and would have passed without proving anything"
        )
    mutated = original.replace(
        anchor, '        if any("runpod_parallel" in a for a in _read_argv(pid)):\n'
    )

    sites = _scan_text(guard, mutated)
    substring = [site for site in sites if site.rule == ARGV_SUBSTRING_RULE]
    assert len(substring) == 1, f"Control B missed the restored incident 1: {sites}"
    assert substring[0].is_violation
    assert substring[0].self_exclusion == "os.getpid()", (
        "the mutated function still excludes its own pid, which is the whole "
        "point: Control A passes it and Control B must not"
    )
    assert guard.read_text(encoding="utf-8") == original


def test_scanned_tree_is_populated() -> None:
    """The scan reaches both roots and skips build artefacts.

    A control that silently scans zero files passes forever.
    """
    files = _scanned_files()
    roots = {path.relative_to(REPO_ROOT).parts[0] for path in files}
    assert roots == set(SCANNED_ROOTS), f"scan covered {roots}, expected {set(SCANNED_ROOTS)}"
    assert len(files) >= MIN_EXPECTED_SCANNED_FILES, (
        f"only {len(files)} files scanned; discovery is broken"
    )
    assert not [p for p in files if "__pycache__" in p.parts]


# --------------------------------------------------------------------------
# The detector must fail on a reintroduction. Every offending source below is
# constructed here and written to tmp_path; no real source file is touched.
# --------------------------------------------------------------------------


def test_bare_pgrep_over_ssh_is_flagged(tmp_path: Path) -> None:
    """Incident 2, reconstructed exactly as it was written."""
    sites = _write(
        tmp_path,
        "probe.py",
        "import subprocess\n"
        "\n"
        "def pod_training_state(host: str) -> str:\n"
        '    result = subprocess.run(["ssh", host, "pgrep -f run_fold"],\n'
        "                            capture_output=True, check=False)\n"
        '    return "BUSY" if result.stdout else "IDLE"\n',
    )
    violations = _violations(sites)
    assert [site.rule for site in violations] == ["pgrep/pkill"]
    assert violations[0].line == 4


def test_pgrep_in_a_subprocess_list_is_flagged(tmp_path: Path) -> None:
    """The same defect spread over several lines, which a line-wise regex misses."""
    sites = _write(
        tmp_path,
        "listform.py",
        "import subprocess\n"
        "\n"
        "def reap() -> None:\n"
        "    subprocess.run([\n"
        '        "pkill",\n'
        '        "-f",\n'
        '        "run_fold",\n'
        "    ], check=False)\n",
    )
    violations = _violations(sites)
    assert [(site.rule, site.line) for site in violations] == [("pgrep/pkill", 5)]


def test_ps_piped_to_grep_in_a_shell_script_is_flagged(tmp_path: Path) -> None:
    """The shell spelling of the same mistake, in the dialect the pods run."""
    sites = _write(
        tmp_path,
        "bootstrap.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "\n"
        "if ps aux | grep -q run_fold; then\n"
        '  echo "BUSY"\n'
        "fi\n",
    )
    violations = _violations(sites)
    assert [(site.rule, site.line) for site in violations] == [("ps-piped-to-grep", 4)]


def test_proc_scan_without_an_exclusion_is_flagged(tmp_path: Path) -> None:
    """Reading /proc directly is not a fix on its own; incident 1 read /proc too."""
    sites = _write(
        tmp_path,
        "procscan.py",
        "import os\n"
        "\n"
        "def orchestrator_pids() -> list[int]:\n"
        "    found = []\n"
        '    for entry in os.listdir("/proc"):\n'
        "        if entry.isdigit():\n"
        "            found.append(int(entry))\n"
        "    return found\n",
    )
    violations = _violations(sites)
    assert [(site.rule, site.line) for site in violations] == [("proc-table-scan", 5)]


def test_substring_identity_inside_a_proc_scan_is_flagged(tmp_path: Path) -> None:
    """Incident 1, reconstructed: both of its membership tests, both flagged.

    Note what this source already does right. It excludes its own pid, exactly
    as the shipped code did. Control A therefore has nothing to say about it,
    and the file would be clean if Control B did not exist -- which is the
    reason Control B exists.
    """
    sites = _write(
        tmp_path,
        "incident_one.py",
        "import os\n"
        "\n"
        "def orchestrator_pids() -> list[int]:\n"
        "    me = os.getpid()\n"
        "    found = []\n"
        '    for entry in os.listdir("/proc"):\n'
        "        if not entry.isdigit() or int(entry) == me:\n"
        "            continue\n"
        "        argv = _read_argv(int(entry))\n"
        '        if not any("runpod_parallel" in arg for arg in argv):\n'
        "            continue\n"
        '        if "reaper_guard" in " ".join(argv):\n'
        "            continue\n"
        "        found.append(int(entry))\n"
        "    return found\n",
    )
    violations = _violations(sites)
    assert [(site.rule, site.line) for site in violations] == [
        (ARGV_SUBSTRING_RULE, 10),
        (ARGV_SUBSTRING_RULE, 12),
    ], "".join(site.describe() for site in sites)


def test_a_pid_exclusion_does_not_clear_a_substring_identity_test(tmp_path: Path) -> None:
    """The escape hatches are not interchangeable, and that is load-bearing.

    If os.getpid() cleared Control B, the control would report the real
    pre-fix reaper_guard.py as clean, because that file called it. The two
    controls answer different questions and only a written exemption answers
    both.
    """
    sites = _write(
        tmp_path,
        "confused.py",
        "import os\n"
        "\n"
        "def probe() -> bool:\n"
        "    me = os.getpid()\n"
        '    for entry in os.listdir("/proc"):\n'
        "        if int(entry) != me and \"runpod_parallel\" in _read_argv(int(entry)):\n"
        "            return True\n"
        "    return False\n",
    )
    substring = [site for site in sites if site.rule == ARGV_SUBSTRING_RULE]
    assert substring and substring[0].self_exclusion == "os.getpid()"
    assert substring[0].is_violation
    assert not [
        site for site in sites if site.rule == PROC_SCAN_RULE and site.is_violation
    ]


def test_exact_argv_comparison_inside_a_proc_scan_is_clean(tmp_path: Path) -> None:
    """The shape incident 1 was fixed into: identity, not containment."""
    sites = _write(
        tmp_path,
        "exact.py",
        "import os\n"
        "\n"
        "ORCHESTRATOR_MODULE = \"scripts.phase1b.runpod_parallel\"\n"
        "\n"
        "def orchestrator_pids() -> list[int]:\n"
        "    me = os.getpid()\n"
        "    found = []\n"
        '    for entry in os.listdir("/proc"):\n'
        "        argv = _read_argv(int(entry))\n"
        "        if int(entry) != me and ORCHESTRATOR_MODULE in argv:\n"
        "            found.append(int(entry))\n"
        "    return found\n",
    )
    assert not _violations(sites), "".join(site.describe() for site in sites)
    assert [site.rule for site in sites] == [PROC_SCAN_RULE]


def test_substring_test_outside_a_proc_scan_is_not_flagged(tmp_path: Path) -> None:
    """Ordinary string handling stays ordinary. Control B is not a style rule."""
    sites = _write(
        tmp_path,
        "ordinary.py",
        "def looks_like_a_fold(name: str) -> bool:\n"
        '    return "fold" in name or "run_fold" in " ".join([name])\n',
    )
    assert sites == [], "".join(site.describe() for site in sites)


def test_exclusion_in_another_function_does_not_cover_the_probe(tmp_path: Path) -> None:
    """Scope discipline: a getpid() elsewhere in the file protects nothing.

    Incident 1's file already called os.getpid() in a neighbouring function
    when incident 2 was written, so a file-wide search for the marker would
    have blessed the broken probe.
    """
    sites = _write(
        tmp_path,
        "neighbour.py",
        "import os\n"
        "import subprocess\n"
        "\n"
        "def who_am_i() -> int:\n"
        "    return os.getpid()\n"
        "\n"
        "def probe() -> bytes:\n"
        '    return subprocess.run(["pgrep", "-af", "run_fold"],\n'
        "                          capture_output=True, check=False).stdout\n",
    )
    violations = _violations(sites)
    assert [(site.rule, site.line) for site in violations] == [("pgrep/pkill", 8)]


# --------------------------------------------------------------------------
# ... and it must not fire on the fixed shapes, or on the incident record.
# --------------------------------------------------------------------------


def test_shell_probe_excluding_its_own_pid_is_clean(tmp_path: Path) -> None:
    """The shape incident 2 was actually fixed into, in miniature."""
    sites = _write(
        tmp_path,
        "fixed_probe.py",
        "import subprocess\n"
        "\n"
        "def pod_training_state(host: str) -> str:\n"
        "    script = (\n"
        '        "for p in /proc/[0-9]*; do "\n'
        '        "  pid=${p##*/}; [ \\"$pid\\" = \\"$$\\" ] && continue; "\n'
        '        "  grep -q run_fold $p/cmdline && { echo BUSY; exit 0; }; "\n'
        '        "done; echo IDLE"\n'
        "    )\n"
        '    return subprocess.run(["ssh", host, script], check=False).stdout\n',
    )
    assert not _violations(sites), "".join(site.describe() for site in _violations(sites))
    assert [site.rule for site in sites] == ["proc-table-scan"]
    assert sites[0].self_exclusion == "$$"


def test_python_probe_excluding_its_own_pid_is_clean(tmp_path: Path) -> None:
    """The shape incident 1 was actually fixed into, in miniature."""
    sites = _write(
        tmp_path,
        "fixed_pids.py",
        "import os\n"
        "\n"
        "def orchestrator_pids() -> list[int]:\n"
        "    me = os.getpid()\n"
        "    found = []\n"
        '    for entry in os.listdir("/proc"):\n'
        "        if entry.isdigit() and int(entry) != me:\n"
        "            found.append(int(entry))\n"
        "    return found\n",
    )
    assert not _violations(sites), "".join(site.describe() for site in _violations(sites))
    assert sites[0].self_exclusion == "os.getpid()"


def test_prose_about_the_incident_is_not_a_probe(tmp_path: Path) -> None:
    """Explaining the defect must stay free. This is the precision half.

    reaper_guard.py carries a nine-line comment quoting `pgrep -f run_fold`
    and a docstring saying pgrep three times. Both are the most valuable text
    in the file. A check that flagged them would be answered by deleting them.
    """
    sites = _write(
        tmp_path,
        "prose.py",
        '"""Reads /proc rather than shelling to pgrep -f run_fold.\n'
        "\n"
        "The shell running `pgrep -f run_fold` matches the pattern itself, so\n"
        "every pod reported BUSY. See the 2026-08-27 incident.\n"
        '"""\n'
        "\n"
        "def documented() -> None:\n"
        '    """Never runs `ps aux | grep run_fold`; see above."""\n'
        "    # `pkill -f run_fold` would find this very shell.\n"
        "    return None\n",
    )
    assert sites == [], "".join(site.describe() for site in sites)


def test_runpodctl_piped_to_grep_is_not_a_probe(tmp_path: Path) -> None:
    """A pipe into grep is not a process probe unless `ps` feeds it.

    scripts/phase0/runpod_setup.sh pipes runpodctl into grep twice. Flagging
    that would make the control noise on the day it landed.
    """
    sites = _write(
        tmp_path,
        "pods.sh",
        "#!/usr/bin/env bash\n"
        'status=$(runpodctl get pod "$pod_id" 2>/dev/null | grep -oP \'status: \\K\\w+\')\n'
        'pod_id=$(runpodctl create pod --ports "22/tcp" 2>&1 | grep -oP \'pod "\\K[^"]+\')\n',
    )
    assert sites == [], "".join(site.describe() for site in sites)


# --------------------------------------------------------------------------
# The escape hatch, and the price of using it.
# --------------------------------------------------------------------------


def test_documented_exemption_clears_the_probe(tmp_path: Path) -> None:
    """A probe that provably cannot match itself may say so and pass."""
    sites = _write(
        tmp_path,
        "exempt.py",
        "import subprocess\n"
        "\n"
        "# process-probe-exempt: this runs inside a container whose only other\n"
        "# process is supervisord, and the pattern is a path that no shell\n"
        "# command line in that container can contain.\n"
        "def probe() -> bytes:\n"
        '    return subprocess.run(["pgrep", "-f", "/opt/tract/bin/train"],\n'
        "                          capture_output=True, check=False).stdout\n",
    )
    assert not _violations(sites), "".join(site.describe() for site in sites)
    assert sites[0].exemption is not None
    assert sites[0].exemption.startswith("this runs inside a container")


def test_a_written_exemption_clears_a_substring_identity_test(tmp_path: Path) -> None:
    """Control B has one hatch and it is the same one. Nothing is unfixable.

    A probe with no way out invites someone to delete the test instead.
    """
    sites = _write(
        tmp_path,
        "exempt_substring.py",
        "import os\n"
        "\n"
        "# process-probe-exempt: the pattern is a 32-character uuid this run\n"
        "# generated itself, so no other command line on the box can contain it.\n"
        "def probe(token: str) -> bool:\n"
        '    for entry in os.listdir("/proc"):\n'
        '        if "tract-" in _read_argv(int(entry))[0]:\n'
        "            return True\n"
        "    return False\n",
    )
    assert not _violations(sites), "".join(site.describe() for site in sites)
    assert {site.rule for site in sites} == {PROC_SCAN_RULE, ARGV_SUBSTRING_RULE}


def test_exemption_without_a_reason_does_not_clear_the_probe(tmp_path: Path) -> None:
    """The hatch costs a sentence. A bare marker is a pragma, and pragmas rot."""
    sites = _write(
        tmp_path,
        "bare_exempt.py",
        "import subprocess\n"
        "\n"
        "def probe() -> bytes:\n"
        "    # process-probe-exempt: safe\n"
        '    return subprocess.run(["pgrep", "-f", "run_fold"],\n'
        "                          capture_output=True, check=False).stdout\n",
    )
    violations = _violations(sites)
    assert [site.rule for site in violations] == ["pgrep/pkill"]
    assert violations[0].saw_unusable_exemption
    assert str(MIN_EXEMPTION_REASON_CHARS) in violations[0].describe()


def test_exemption_below_the_probe_scope_does_not_leak_upward(tmp_path: Path) -> None:
    """An exemption written for one function does not silence the next one."""
    sites = _write(
        tmp_path,
        "leak.py",
        "import subprocess\n"
        "\n"
        "def unexempted() -> bytes:\n"
        '    return subprocess.run(["pgrep", "-f", "run_fold"],\n'
        "                          capture_output=True, check=False).stdout\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "# process-probe-exempt: this one really is unable to match itself,\n"
        "# because the pattern is an absolute path to a compiled binary.\n"
        "def exempted() -> bytes:\n"
        '    return subprocess.run(["pgrep", "-f", "/opt/tract/bin/train"],\n'
        "                          capture_output=True, check=False).stdout\n",
    )
    violations = _violations(sites)
    assert [(site.rule, site.line) for site in violations] == [("pgrep/pkill", 4)]


def test_malformed_python_in_a_scanned_root_fails_loudly(tmp_path: Path) -> None:
    """Silence on an unparseable file would be a hole the size of a file."""
    path = tmp_path / "broken.py"
    path.write_text("def probe(:\n    pass\n", encoding="utf-8")
    try:
        _scan_source(path)
    except ValueError as exc:
        assert "does not parse as Python" in str(exc)
    else:
        raise AssertionError("an unparseable source file was scanned silently")
