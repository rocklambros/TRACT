# v3 splice: Global Constraints, Task 16, Self-review

Author scope: Global Constraints (plan lines 17-70), Task 16 (6596-6878), Self-review (6880-6987).
Every claim below was re-verified against source on 2026-08-19 before it was written. Departures
from `V3-CONTRACT.md` are marked **CONTRACT DEPARTURE** with the measurement that forced them.

---

<<<GLOBAL CONSTRAINTS>>>

## Global Constraints

Copied from the spec (Part 1), the run ledger (`.superpowers/autonomous-run/RUN-LEDGER.md`) and
`premortem-v2/V3-CONTRACT.md`. These bind every task below.

- **All inference and training runs on RunPod, never locally.** Nothing in this plan loads a model,
  so all of it runs locally. Unit tests, lint and typecheck are local by CLAUDE.md.
- **`data/raw/` is immutable.** Parsers read it, never write it.
- **Three licence tiers, not two.** `RESTRICTED_FRAMEWORK_IDS` modelled licence *status* and nothing
  modelled licence *class*, so seven frameworks under a copyleft or no-redistribution notice were
  treated as unconditionally publishable. A grep of the 6,987-line v2 plan returns **0** hits for
  `CC-BY-SA` and **0** for `GPL-3.0` **[measured, this run]**, while `tract/config.py`
  `FRAMEWORK_LICENSES` already records `dsomm -> GPL-3.0-only`, `biml -> CC-BY-SA-3.0 AND
  CC-BY-SA-4.0`, `samm`/`wstg`/`owasp_top10_2021`/`owasp_proactive_controls` -> `CC-BY-SA-4.0`, and
  `csa_ccm -> Proprietary, all rights reserved, no redistribution` **[measured, this run]**. Task 1
  adds to `tract/config.py`:

  ```python
  RESTRICTED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({"etsi", "iso_27001"})
  # Reproduction permitted, but on terms a CC0 grant cannot carry. Text routes to the gitignored
  # overlay exactly as RESTRICTED does. ASSIGNMENTS stay tracked and published, because a mapping
  # is a fact about two documents rather than a reproduction of either. Training reads the
  # overlay, so this costs zero anchors.
  CONDITIONAL_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
      "dsomm",                    # GPL-3.0-only
      "biml",                     # CC-BY-SA-3.0 AND CC-BY-SA-4.0
      "samm", "wstg",             # CC-BY-SA-4.0
      "owasp_top10_2021", "owasp_proactive_controls",   # CC-BY-SA-4.0
      "csa_ccm",                  # all rights reserved, no redistribution
  })
  OVERLAY_FRAMEWORK_IDS: Final[frozenset[str]] = (
      RESTRICTED_FRAMEWORK_IDS | CONDITIONAL_FRAMEWORK_IDS
  )
  ```

  Everything that branches on `RESTRICTED_FRAMEWORK_IDS` for **text routing** branches on
  `OVERLAY_FRAMEWORK_IDS` instead. `RESTRICTED_FRAMEWORK_IDS` keeps its current meaning everywhere
  else, including `tests/test_licensed_text_not_tracked.py`, whose fingerprint coverage assertion
  and `.gitignore`-line assertion both key on it **[measured: that file imports
  `RESTRICTED_FRAMEWORK_IDS` and no other tier]**. `dsomm.json` today holds 183 title stubs with
  description lengths of 5 / 21 / 32 at min / median / max and zero over 200 characters
  **[measured, this run]**. Task 3 replaces them with 182 full GPL-3.0 activity statements, so the
  question changes from theoretical to live inside this plan.
- **Licensed text never enters git.** Both `RESTRICTED` processed files are already in `.gitignore`
  (lines 37-38). Task 1 adds a line per `CONDITIONAL` member and routes those parsers' output to
  the gitignored `data/processed/licensed/` overlay. **Never `git add -f`.** Forcing an ignored path
  into git is how licensed text escaped four times.
- **Evidence artifacts are tracked by design, never by force.** `.gitignore:3` is `results/`
  **[measured]**, and `git add real.py results/corpus/before.json` prints `The following paths are
  ignored`, returns **exit 1**, and stages `real.py` only **[measured, reproduced this run: git
  2.50.1]**. Git stages the legal paths and refuses the ignored one, so a task that adds an
  artifact alongside code commits the code and silently loses the artifact. Task 1 changes line 3
  and adds one negation:

  ```
  results/*
  !results/corpus/
  ```

  **The form matters and the obvious form does not work.** `results/` with a trailing slash
  excludes the directory, git never descends into it, and the negation is never evaluated:
  `results/` plus `!results/corpus/` plus `!results/corpus/**` staged **nothing but `.gitignore`**
  in a scratch repository, while `results/*` plus `!results/corpus/` staged
  `results/corpus/after.json` and left `results/other/y.json` ignored **[measured, both forms run
  this run]**. A negation that looks correct and re-ignores everything is the same defect class this
  plan exists to close, so Task 1 Step 1 verifies the negation with `git check-ignore` before
  writing the BEFORE artifact.
  `results/corpus/` carries counts, sha256 digests and the per-link resolution record only, never
  anchor text, so it is safe to track for overlay frameworks. Once tracked it falls under
  `tests/test_licensed_text_not_tracked.py::test_no_verbatim_licensed_statement_anywhere_in_the_tree`,
  which scans every tracked `.json` **[measured: `_SCANNED_SUFFIXES` includes `.json`]**.
- **Every task carries an `**Invalidates:**` line** naming every artifact it makes stale. A grep of
  the v2 plan for `invalidates` returns **0** **[measured, this run]**, which is ledger lesson 6
  recurring a fourth time. `RUN-LEDGER.md:93` states the column is mandatory and that every future
  amendment must fill it in. A task with nothing to invalidate writes `**Invalidates:** nothing` and
  says why. At minimum:

  | task | invalidates |
  |---|---|
  | 14 (training links) | `hub_links_training.jsonl` consumers, and the ceiling study's pool mirror in `tract/ceiling_study.py` |
  | 15 (corpus rebuild) | `data/processed/stopwords.json` (13 consumers), `all_controls.json`, every `data_hash` recorded before it, and `results/corpus/before_8cf44b3.json` as a description of the current corpus |
  | 16 (AFTER report) | `results/corpus/before_8cf44b3.json` as the current state, and any RUN-LEDGER row quoting a pre-Task-16 corpus figure |

- **Never republish to HuggingFace.** No task here touches a publish path.
- **No AI attribution** in commit messages, comments, or docs. The git author stays the human.
- **Type everything.** All signatures fully typed. `mypy tract/ parsers/ scripts/ --strict` must pass.
- **Fail loud.** `raise ValueError` with a specific message. No bare `except`. No `return None` to
  signal failure.
- **Atomic writes only**, via `tract.io.atomic_write_json`.
- **Deterministic output.** Sorted keys, no clock reads in any written artifact. `fetched_date` is a
  `ClassVar` per parser, never `date.today()`.
- **Every number carries `[measured]`, `[derived]` or `[unmeasured]`.** No threshold anywhere may
  depend on an `[unmeasured]` value (ledger lesson 8).
- **Any transform that moves or synthesises text emits an audit record and fails closed** (ledger
  lesson 7). `BaseParser.write_repair_audit` exists, so use it. It is written unconditionally, empty
  list included, so a missing file means the parser never ran.
- **Compute the attainable range of every threshold in BOTH directions and state it in a comment
  next to the assertion** (ledger lesson 3, extended by ledger lesson 9). A floor above a source's
  arithmetic maximum is a guaranteed failure. A floor at or below the arithmetic minimum is a
  guaranteed pass, which is worse, because it reports green. Four assertions in the v2 acceptance
  suite could only ever return one value:
  - `assert floor <= 1.0` is tautological against the literals three lines above it.
  - `assert report.by_id(f).wrong_anchor_risk == 0` increments only inside the `channel == "title"`
    branch **[measured: plan line 531 sits under `if channel == "title":`]**, and the resolution
    table below engineers nine of eleven frameworks to resolve entirely through the id channel, so
    the maximum attainable value is 0 for those nine.
  - `assert BaseParser.honest_prose_fraction(controls) > 0.0` compares a **ratio** against zero, so
    one prose control in `csa_ccm`'s 224 gives 0.0045 and passes **[measured: the function returns
    `honest / len(measurable)`]**. Compare against the parser's declared `min_prose_fraction`.
  - Task 1's channel-parity test builds `ProseIndex(data if isinstance(data, list) else [])`, and
    both corpus files are dicts with keys `[framework_count, frameworks, generated_date,
    total_controls]` **[measured]**, so the index is built from `[]` and all 4,405 assertions reduce
    to `True == True`. Use the dict-aware loader.
- **Pre-registered thresholds live in tracked code, committed before the run they gate.**
  `git check-ignore -v` on this plan file returns `.gitignore:25:docs/superpowers/` and `git log` on
  it is empty **[measured, this run]**, so a floor edited down mid-execution leaves no diff.
  `JOIN_FLOORS` and `JOIN_WRONG_ANCHOR_BUDGET` are therefore committed in **Task 1**, in
  `tract/corpus_report.py`, before any parser exists. Task 16 consumes both and defines neither.
  This is the recorded defect `gate-preregistration-is-retrospective` in a stricter form: criterion
  and PASS could otherwise land in zero commits.
- **CI cannot see the overlay, and that must fail loudly rather than skip silently.** The tracked
  corpus holds **29** frameworks and the gitignored overlay holds **31**, the difference being
  `etsi` and `iso_27001` **[measured, this run]**. `merged_corpus_path()` returns the tracked file
  whenever the overlay is absent and the tracked file always exists **[measured:
  `text_selection.py:76-77`]**, so `if not merged_corpus_path().exists(): pytest.skip(...)` never
  skips and the assertions hard-fail on data that cannot legally be present. Gate on
  `report.corpus_framework_count`, admit exactly two counts, and fail on any third:

  ```python
  full = len(expected_framework_ids())
  if report.corpus_framework_count == full:
      overlay_present = True
  elif report.corpus_framework_count == full - len(OVERLAY_FRAMEWORK_IDS):
      overlay_present = False          # a fresh clone or CI, which is legal
  else:
      raise AssertionError(          # short by frameworks no licence explains
          f"corpus reports {report.corpus_framework_count} frameworks, "
          f"expected {full} or {full - len(OVERLAY_FRAMEWORK_IDS)}"
      )
  ```

  Overlay rows then skip **as a named group with the reason stated**, and every other row still
  asserts. Never delete or relax a floor to make CI green. That retires the only gate on a parser
  nobody can inspect.
- **`pytest -x` collects alphabetically, so an early abort disarms a later gate.**
  `.github/workflows/ci.yml:65` runs `pytest tests/ -x -q --timeout=60 -m "not integration"` with no
  fetch step **[measured]**, and `tests/test_corpus_acceptance.py` sorts before
  `tests/test_licensed_text_not_tracked.py`. A corpus-acceptance failure under `-x` stops the
  licensed-text gate from running at all. Task 16 Step 6 runs the licensed-text gate first, on its
  own, before the full `-x` suite.

### The interpreter

`python3` on this machine is Homebrew 3.13.7 and has none of this project's dependencies. The
`tract` console script's interpreter (`/Users/klambros/.local/share/uv/tools/tract/bin/python3`) has
pydantic, PyYAML, bs4, lxml and numpy but **not** pdfplumber, openpyxl, defusedxml, pytest or mypy.
**[measured]** The interpreter that has them is the one `pytest` already resolves to:

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

Every command in this plan uses that variable. Verify it once, at the top of Task 1, and never
substitute `python3`.

`defusedxml==0.7.1` and `openpyxl` are **both now installed** in that environment
**[measured, this run: `defusedxml 0.7.1`, `import openpyxl` succeeds]**, which supersedes the v2
claim that defusedxml was absent. With them installed, `parse_capec.py` and `parse_cwe.py` were run
into a scratch directory and reproduced **558 of 558** and **1,331 of 1,331** baseline control
hashes with **0 mismatch** **[measured, orchestrator]**, taking pre-measured rebuild coverage to
**89.7%**, not the 45% the v2 plan assumed. `openpyxl` still appears in neither `requirements.txt`
nor `pyproject.toml` **[measured]**, so Task 8 must still add it pinned.

**One pin is drifting and no task closes it.** `requirements.txt` pins `pdfplumber==0.11.10` and the
mandated interpreter has **0.11.4** **[measured, this run]**. Every `[measured]` PDF number in
Tasks 12 and 13 is a function of the extractor version, so those numbers were produced by a build CI
will not reproduce. Either align the local install to the pin before the PDF tasks run, or record
each PDF figure as measured under 0.11.4 and re-derive it under 0.11.10. Do not leave the two
readings undistinguished.

### Three contract facts that decide parser design

Read out of `tract/text_selection.py` and `tract/parsers/base.py` at `8cf44b3`, not inferred.

1. **`ProseIndex` prefers `full_text` over `description`, unconditionally.** `ProseIndex.__init__`
   takes `full_text` when it is non-empty and never looks at `description`. Whatever a parser puts
   in `full_text` **is the anchor the model sees**. `full_text` is not free storage.
2. **`BaseParser._sanitize_control` sets `full_text` behind the parser's back.** When `description`
   exceeds `DESCRIPTION_MAX_LENGTH` (2000), `sanitize_text(..., return_full=True)` returns the full
   text and it is written to `full_text`, **discarding whatever the parser put there**
   (`base.py:377-383`). A parser that emits a 3,000-character description has chosen a
   3,000-character anchor whether it meant to or not.
3. **A control whose `description` does not exceed its `title` by `PROSE_MIN_EXTRA_CHARS` (20) and
   has no `full_text` is not indexed at all.** `ProseIndex.__init__` hits `continue`. Its links
   resolve to nothing and fall back to the section title. This is invisible to
   `honest_prose_fraction`, which uses a different rule (60 characters, and merely different from
   the title). Task 1's instrument counts these. Nothing did before.

### Resolution order, and why it is not changed

`ProseIndex.lookup` tries **title first, then id**. That order was written to fix a real defect:
NIST AI 100-2 links carry the containing subsection's id for three distinct mitigations, and
id-first gave all three the same paragraph. The order stays. Each parser states how it avoids the
other failure mode, a link's `section_name` matching the wrong control's title:

| framework | risk | how this plan avoids it |
|---|---|---|
| biml | `Data Confidentiality` names two different risks in two documents, `Hosting` names three link rows across two documents. 7 of 21 rows participate in a label collision. **[measured]** | titles are document-scoped (`Hosting (BIML-24(LLM))`), so no link name can match a title. Every row resolves through the id channel, with `alt_ids` for the 7 unprefixed ids |
| etsi | 24 technique names over 16 section ids, and three of those names span two clauses each **[measured]**, so registering them all as `alt_titles` would let the title channel answer with a clause the link did not name | only the two rows whose `section_id` is itself a name get an `alt_title`. The other 34 resolve through the id channel against a `control_id` that is the clause number |
| nist_ssdf | `section_name` is the task statement verbatim for 36 of 46 rows **[measured]**. If the parser also used it as `title`, `_is_prose` would drop every control | `title` is the task id, `description` is the task statement |
| wstg | `section_name == section_id` for all 118 rows **[measured]**, for example `WSTG-INFO-01` | `title` is the file's H1, which no link name spells, so every row resolves through the id channel |
| owasp_proactive_controls, nist_800_63 | same shape, `section_name == section_id`, 2-7 characters | same: the title is the human title, the id channel carries the join |
| csa_ccm | **the exception, and the only framework where the title channel carries the join.** 15 of 29 links target a bare domain code whose `section_name` is a descriptive domain title **[measured, orchestrator]**, so `by_title` is about **26 of 29**, not the 7 the v2 plan predicted. Title-first then resolves `IPY` to control IPY-01's title rather than to the IPY domain **[measured, ML Engineer]** | Task 6 states the measured `by_title` and the measured wrong-anchor count. Task 1 records the latter in `JOIN_WRONG_ANCHOR_BUDGET`, so Task 16 gates on a pre-registered number rather than on an unfailable `== 0` |

### The headline numbers, corrected

The v2 plan's `+452 distinct anchors` is wrong and **must never enter the RUN-LEDGER**. It counts
the eleven frameworks' new anchors against a baseline of zero, and the baseline is not zero: their
734 curated links already land on **299 distinct fallback anchors** today **[measured,
orchestrator, per framework: dsomm 18, wstg 59, nist_ssdf 44, enisa 33, samm 30, csa_ccm 29,
nist_800_63 25, biml 17, etsi 24, owasp_proactive_controls 10, owasp_top10_2021 10]**.

| v2 plan says | truth | source |
|---|---|---|
| `+452 distinct anchors` | **+153** (1,749 -> 1,902). Seven of eleven parsers move the anchor count by exactly zero, and ETSI **loses 10** (24 -> 14) | orchestrator **[measured]** |
| training links `4,127 -> 4,402` | **4,401**. A fourth link falls under the 10-character floor: `nist_800_63` `section_name == 'are g'` (5 characters) | Data Scientist **[measured]**, enumerated over 16 candidates |
| `dropped_by_prose_rule` total 522 | **558**. The v2 total sums only frameworks carrying curated links, so NIST AI RMF 25, AIUC-1 10 and CoSAI 1 are invisible | Data Scientist **[measured]** |
| biml `distinct_anchors == 20` | **19**. `inference:9` appears prefixed and unprefixed and `UNPREFIXED_IDS` routes both to the same control. 21 links over 19 anchors is 1.105 | ML Engineer **[measured]** |

**Anchor count is not this plan's only benefit and for seven parsers it is not the benefit at all.**
`distinct_anchors` is the v2 plan's declared load-bearing column, and gating on it alone gates on
the wrong thing. Every report and every ledger row carries a **text-quality delta beside the anchor
delta**: `anchor_source_full_text + anchor_source_description`, which is **0 today** for all eleven
frameworks because every one of them resolves 0 of its links **[measured]**. Their stored
`honest_prose_fraction` is **0.0000 for all eleven** against declared `min_prose_fraction` floors of
0.90 to 1.00 **[measured, this run]**. That is the change this plan buys.

<<<TASK 16>>>

### Task 16: The AFTER report, and the acceptance tests that keep it true

The v2 acceptance suite had nine terminal assertions and **six passed by construction**. Of the
three that bit, one asserted a number that is wrong. It also could not run at all: four assertions
hard-failed in CI on frameworks whose text cannot legally be in a fresh clone, and the one test
protecting the twenty untouched frameworks skipped on an artifact that `git add` refuses to stage.
This task rebuilds the suite so that every assertion has a reachable failure in both directions, and
so that the failures CI can produce are about parsers rather than about licences.

**Files:**
- Create: `results/corpus/after_parsers.json`
- Create: `tests/test_corpus_acceptance.py`
- Modify: `.superpowers/autonomous-run/RUN-LEDGER.md`

`tract/corpus_report.py` is **not** modified here. `JOIN_FLOORS` and `JOIN_WRONG_ANCHOR_BUDGET` were
committed in Task 1, before any parser existed, because this plan file is gitignored
(`.gitignore:25`) and untracked **[measured]**, so a threshold edited mid-run leaves no diff at all.
A criterion that can move in the same commit as the result it gates reproduces
`gate-preregistration-is-retrospective`. A criterion that can move in **zero** commits is worse.

**Interfaces:**
- Consumes: `tract.corpus_report.JOIN_FLOORS`, `tract.corpus_report.JOIN_WRONG_ANCHOR_BUDGET`,
  `build_corpus_report`, `check_join_floors`, `CorpusReport.corpus_framework_count`,
  `FrameworkJoin.fallback_anchors`, `FrameworkJoin.anchor_source_*`;
  `tract.config.OVERLAY_FRAMEWORK_IDS`, `RESTRICTED_FRAMEWORK_IDS`, `PROCESSED_FRAMEWORKS_DIR`,
  `PARSERS_DIR`, `PROJECT_ROOT`; `results/corpus/before_8cf44b3.json`.
- Produces: `results/corpus/after_parsers.json`, `tests/test_corpus_acceptance.py`.

**Invalidates:** `results/corpus/before_8cf44b3.json` as a description of the current corpus. It
stays tracked and stays the pre-registered baseline, and no later task may read it as current state.
Any RUN-LEDGER row quoting a pre-Task-16 corpus figure. Nothing downstream consumes this task's
output inside this plan, because Task 16 is terminal.

- [ ] **Step 1: Verify the pre-registered thresholds are already committed**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
git log --oneline -1 -- tract/corpus_report.py
git log -S 'JOIN_FLOORS' --oneline -- tract/corpus_report.py
PYTHONPATH=. "$PY" -c "
from tract.corpus_report import JOIN_FLOORS, JOIN_WRONG_ANCHOR_BUDGET
print(len(JOIN_FLOORS), 'floors'); print(len(JOIN_WRONG_ANCHOR_BUDGET), 'wrong-anchor budgets')"
git check-ignore -v results/corpus/after_parsers.json; echo "check-ignore exit=$?"
```

Expected: `JOIN_FLOORS` first appears in Task 1's commit, **11 floors**, and `check-ignore` exits
**1** with no output, meaning `results/corpus/` is trackable. If `check-ignore` exits 0, Task 1's
`.gitignore` negation is in the wrong form. `results/` with a trailing slash stops git descending
and the negation is never evaluated: `results/` plus `!results/corpus/**` staged nothing, while
`results/*` plus `!results/corpus/` staged the artifact **[measured, both forms reproduced in a
scratch repository, git 2.50.1]**. Fix the form before continuing. Do not use `git add -f`.

- [ ] **Step 2: Write the acceptance tests**

```python
# tests/test_corpus_acceptance.py - create

"""What the eleven parsers had to be true for, expressed as a gate.

Every threshold here was derived from the curated link file and the pinned
source before its parser was written, and committed in Task 1, or it is a
property the corpus already had that must not regress. The instrument is
tract.corpus_report, the same one the per-parser steps used, so a parser cannot
be accepted by a measurement its consumer does not perform.

Two rules govern every assertion in this file.

First, each one states its attainable range in both directions. The suite this
replaces had nine terminal assertions and six could only ever return one value:
`floor <= 1.0` against literals three lines above, `wrong_anchor_risk == 0` on
nine frameworks engineered to resolve entirely through the id channel where the
counter only increments in the title branch, and `honest_prose_fraction > 0.0`
against a ratio, where one prose control in csa_ccm's 224 gives 0.0045 and
passes. A gate that cannot fail reports green having measured nothing.

Second, no assertion may be silenced by a licence. The tracked corpus holds 29
frameworks and the gitignored overlay holds 31 [measured], merged_corpus_path()
falls back to the tracked file, and the tracked file always exists, so
`if not merged_corpus_path().exists(): pytest.skip(...)` never skips and four
assertions hard-failed in CI on text that cannot legally be in a fresh clone.
The predictable repair is deleting the ETSI floor, which retires the only gate
on a restricted parser nobody can inspect. Here the corpus-dependent tests
admit exactly two framework counts, skip the overlay rows as a named group with
the reason stated, keep every other row asserting, and fail on any third count.
The full eleven-framework result is asserted separately against the committed
AFTER artifact, which is tracked and needs no corpus at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tract.config import (
    OVERLAY_FRAMEWORK_IDS,
    PARSERS_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    PROJECT_ROOT,
    RESTRICTED_FRAMEWORK_IDS,
)
from tract.corpus_report import (
    JOIN_FLOORS,
    JOIN_WRONG_ANCHOR_BUDGET,
    build_corpus_report,
    check_join_floors,
)

# tract.config names the repository root PROJECT_ROOT. Every path below is
# anchored to it rather than to the working directory, because pytest can be
# invoked from anywhere and a relative path that misses turns an assertion into
# a skip.
REPO_ROOT: Path = PROJECT_ROOT

BEFORE_PATH: Path = REPO_ROOT / "results" / "corpus" / "before_8cf44b3.json"
AFTER_PATH: Path = REPO_ROOT / "results" / "corpus" / "after_parsers.json"

# The eleven frameworks this plan gives a parser to. Read off the pre-registered
# floors rather than restated, so the two cannot drift apart.
PENDING: tuple[str, ...] = tuple(sorted(JOIN_FLOORS))

# Rows that a fresh clone cannot assert, because their processed text routes to
# the gitignored overlay under CONDITIONAL_FRAMEWORK_IDS or
# RESTRICTED_FRAMEWORK_IDS. Named, not silent.
OVERLAY: frozenset[str] = OVERLAY_FRAMEWORK_IDS

_MIN_PROSE = re.compile(
    r"^\s*min_prose_fraction:\s*ClassVar\[float\]\s*=\s*([0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)


def _expected_framework_ids() -> frozenset[str]:
    """Framework ids the corpus must cover, from what this checkout can see.

    The union of what is on disk and the overlay set, because an overlay
    framework's per-framework JSON is absent from a fresh clone entirely: 32
    JSON files exist here and 30 are tracked [measured]. Deriving the count
    rather than hard-coding it keeps this from rotting the next time a
    framework lands.
    """
    on_disk = (
        {path.stem for path in PROCESSED_FRAMEWORKS_DIR.glob("*.json")}
        if PROCESSED_FRAMEWORKS_DIR.exists()
        else set()
    )
    assert on_disk, (
        f"no framework JSON under {PROCESSED_FRAMEWORKS_DIR}. The suite would "
        f"otherwise derive an expected count of zero and pass on an empty "
        f"corpus."
    )
    return frozenset(on_disk | OVERLAY)


def _load(path: Path) -> dict[str, Any]:
    """A committed corpus report. A missing one is a failure, never a skip.

    The v2 suite guarded this read with `pytest.skip("no BEFORE artifact in
    this checkout")`. Because `.gitignore:3` was `results/`, `git add` refused
    the artifact and exited 1 while staging the code beside it [measured,
    reproduced], so the skip would have fired on every machine forever and the
    only test protecting the twenty untouched frameworks would have reported
    green having run nothing.
    """
    if not path.exists():
        raise AssertionError(
            f"{path.relative_to(REPO_ROOT)} is missing. It is committed "
            f"evidence, not an optional local file. Regenerate it with "
            f"scripts/corpus_report.py and confirm `git check-ignore` exits 1 "
            f"for it. Never `git add -f`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(report_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["framework_id"]: row for row in report_json["per_framework"]}


@pytest.fixture(scope="module")
def live():  # type: ignore[no-untyped-def]
    """The report built from whatever corpus this checkout holds."""
    return build_corpus_report()


@pytest.fixture(scope="module")
def overlay_present(live) -> bool:  # type: ignore[no-untyped-def]
    """Whether the licensed overlay is readable here.

    Three outcomes, not two. The full count and the full count minus the
    overlay are both legal. Anything else means the corpus is short by
    frameworks no licence explains, which is a red build rather than a skip.
    """
    full = len(_expected_framework_ids())
    without_overlay = full - len(OVERLAY)
    count = live.corpus_framework_count
    if count == full:
        return True
    if count == without_overlay:
        return False
    raise AssertionError(
        f"the corpus reports {count} frameworks. Only {full} (with the "
        f"licensed overlay) and {without_overlay} (a fresh clone or CI, "
        f"without it) are explainable. Any other count means frameworks are "
        f"missing for a reason that is not a licence, and skipping here would "
        f"hide it. Overlay set: {sorted(OVERLAY)}."
    )


def _assertable(overlay_present: bool) -> tuple[str, ...]:
    """The pending frameworks whose live rows can be asserted here."""
    if overlay_present:
        return PENDING
    return tuple(f for f in PENDING if f not in OVERLAY)


class TestTheSuiteCanActuallyRun:
    """Positive controls. Without these the file below can go quiet."""

    def test_the_overlay_decision_is_one_of_two_named_states(
        self, overlay_present: bool
    ) -> None:
        # Attainable: True on a checkout holding the licensed overlay, False in
        # CI and any fresh clone. Any third corpus size raises inside the
        # fixture rather than reaching here.
        assert isinstance(overlay_present, bool)

    def test_at_least_three_pending_frameworks_assert_everywhere(
        self, overlay_present: bool
    ) -> None:
        """CI must still gate something real.

        Eight of the eleven route to the overlay under the three-tier licence
        model, leaving nist_800_63, enisa and nist_ssdf assertable in a fresh
        clone. If that set ever empties, this file measures nothing in CI and
        the committed-artifact class below becomes the only gate.
        """
        # Attainable: 3 in CI, 11 locally. Fails at 2 or fewer, which is what a
        # future licence reclassification would produce.
        assert len(_assertable(overlay_present)) >= 3

    def test_the_skipped_group_is_exactly_the_overlay(
        self, overlay_present: bool
    ) -> None:
        """A non-licensed framework may never join the silent group."""
        skipped = set(PENDING) - set(_assertable(overlay_present))
        # Attainable: empty locally, exactly the eight overlay members of
        # PENDING in CI. A ninth id appearing here is the failure being gated.
        assert skipped <= set(OVERLAY), sorted(skipped - set(OVERLAY))
        if not overlay_present:
            assert skipped == set(PENDING) & set(OVERLAY)


class TestJoinFloors:
    def test_every_assertable_framework_clears_its_derived_floor(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        assertable = _assertable(overlay_present)
        floors = {f: JOIN_FLOORS[f] for f in assertable}
        failures = check_join_floors(live, floors)
        # Attainable: [] when every parser hits its ceiling, up to 11 messages
        # when none does. Today, before any parser exists, all eleven resolve
        # 0 of 734 links [measured], so this returns 11 messages.
        assert failures == [], failures
        if not overlay_present:
            pytest.skip(
                f"the licensed overlay is absent from this checkout, so "
                f"{sorted(set(PENDING) - set(assertable))} were not asserted "
                f"here. TestCommittedAfterReport gates all eleven off the "
                f"tracked artifact."
            )

    def test_no_floor_sits_outside_the_range_its_column_can_produce(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        """A floor must be reachable from above and missable from below.

        The v2 suite wrote `assert floor <= 1.0` next to a dict of literals,
        which is tautological, and read it as discharging ledger lesson 3. It
        guarded against a floor that is unreachably high and did nothing about
        a floor that cannot be missed. Both halves are here.
        """
        for framework_id in _assertable(overlay_present):
            floor = JOIN_FLOORS[framework_id]
            row = live.by_id(framework_id)
            # resolution_rate is (by_title + by_id) / links, so its attainable
            # range is [0.0, 1.0] and every floor must sit strictly inside it.
            assert 0.0 < floor <= 1.0, f"{framework_id}: floor {floor}"
            assert row.links > 0, framework_id
            # The lower bound is what makes the floor a gate. A floor of 0.0
            # cannot be missed by any parser, however broken.
            assert floor > 0.0, framework_id
            assert row.resolution_rate >= floor, (
                f"{framework_id}: {row.resolution_rate:.4f} < {floor:.4f}"
            )


class TestAnchorSeparation:
    def test_dsomm_stopped_collapsing_onto_its_sub_dimensions(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        if "dsomm" not in _assertable(overlay_present):
            pytest.skip("dsomm is GPL-3.0-only and routes to the overlay")
        row = live.by_id("dsomm")
        # Attainable: distinct_anchors in [0, 213] against 213 resolvable
        # links, links_per_anchor in [1.0, 214.0]. Today dsomm's 214 links land
        # on 18 fallback anchors [measured], which is 11.9 links per anchor, so
        # both assertions fail before the parser and both can fail after it.
        assert row.distinct_anchors >= 182, row.distinct_anchors
        assert row.links_per_anchor <= 1.20, row.links_per_anchor

    def test_biml_did_not_collapse_on_shared_labels(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        """Seven of 21 rows share a section_name across two documents."""
        if "biml" not in _assertable(overlay_present):
            pytest.skip("biml is CC-BY-SA and routes to the overlay")
        row = live.by_id("biml")
        # 19, not the 20 the v2 plan asserted: `inference:9` appears prefixed
        # and unprefixed and UNPREFIXED_IDS routes both to the same control
        # [measured, ML Engineer]. 21 links over 19 anchors is 1.105. 19 is the
        # arithmetic maximum, so this fails downward only, and downward is the
        # collapse being gated. Today biml lands on 17 fallback anchors.
        assert row.distinct_anchors == 19, row.distinct_anchors
        assert row.by_title == 1, row.by_title

    def test_etsi_registered_only_the_two_names_that_cannot_collide(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        """Three ETSI technique names span two clauses each.

        Registering all 24 as alternate titles keeps the resolution rate at
        1.0000 while two rows resolve to a clause they did not name, so the
        rate cannot see this and by_title can.
        """
        if "etsi" not in _assertable(overlay_present):
            pytest.skip(
                "etsi is restricted, reproduction only by written permission, "
                "and its processed text is absent from this checkout"
            )
        # Attainable: [0, 36] across 36 links. Fails at 0 or 1 (an alternate
        # was dropped) and at 3 or more (an alternate was over-registered).
        assert live.by_id("etsi").by_title == 2, live.by_id("etsi").by_title

    def test_no_pending_framework_nests_an_anchor_inside_another(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        for framework_id in _assertable(overlay_present):
            row = live.by_id(framework_id)
            # nested_anchors is containment, not strict prefix, so ETSI 5.2
            # inside 5.2.2 is visible. Attainable [0, distinct_anchors].
            assert row.nested_anchors == 0, (
                f"{framework_id}: {row.nested_anchors} nested anchors"
            )

    def test_wrong_anchor_risk_is_gated_only_where_it_can_fire(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        """The counter increments only inside the title branch.

        Nine of eleven frameworks are engineered to resolve entirely through
        the id channel, so their maximum attainable wrong_anchor_risk is 0 and
        the v2 suite's `== 0` was unfailable on all nine. On csa_ccm it is the
        opposite problem: 15 of 29 links target a bare domain code whose
        section_name is a descriptive domain title [measured, orchestrator],
        `IPY` resolves to control IPY-01 rather than to the IPY domain
        [measured, ML Engineer], so `== 0` halts a healthy run.

        JOIN_WRONG_ANCHOR_BUDGET, committed in Task 1, holds one entry per
        framework whose task predicts by_title > 0, each derived from that
        task's pre-parser premise check. Frameworks outside it get no
        wrong-anchor assertion, because there is nothing there to assert.
        """
        for framework_id in _assertable(overlay_present):
            row = live.by_id(framework_id)
            if framework_id in JOIN_WRONG_ANCHOR_BUDGET:
                budget = JOIN_WRONG_ANCHOR_BUDGET[framework_id]
                # Non-vacuity: the title channel must be live, or the budget
                # entry is guarding a branch that never executes.
                assert row.by_title > 0, (
                    f"{framework_id} has a wrong-anchor budget but resolves "
                    f"nothing by title, so the budget guards nothing"
                )
                # Attainable [0, row.by_title]. Fails upward on a new wrong
                # anchor, which is the exposure being gated.
                assert row.wrong_anchor_risk <= budget, (
                    f"{framework_id}: {row.wrong_anchor_risk} wrong anchors "
                    f"against a pre-registered budget of {budget}"
                )
            else:
                # The gate that bites for the other seven: a framework
                # that starts resolving by title has acquired wrong-anchor
                # exposure its task never analysed. Attainable [0, row.links].
                assert row.by_title == 0, (
                    f"{framework_id} resolved {row.by_title} links through the "
                    f"title channel, which its task predicted would be zero. "
                    f"Measure the wrong-anchor count and add a budget entry in "
                    f"Task 1 before accepting this."
                )


class TestTextQuality:
    """The column the v2 plan never had.

    distinct_anchors was named the load-bearing column and it is the wrong one
    for seven of eleven parsers, whose anchor count does not move at all. What
    moves for all eleven is where the anchor text comes from.
    """

    def test_the_pending_frameworks_stopped_anchoring_on_fallback_titles(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        before = _rows(_load(BEFORE_PATH))
        for framework_id in _assertable(overlay_present):
            row = live.by_id(framework_id)
            baseline = before[framework_id]["fallback_anchors"]
            # A fallback anchor is a distinct section_name the trainer gets for
            # a link the prose index missed. Attainable [0, baseline]. The
            # eleven sum to 299 today [measured], so every one of these fails
            # before the parsers and each can fail after one.
            assert baseline > 0, (
                f"{framework_id} had no fallback anchors in the BEFORE state, "
                f"so this comparison would be vacuous. Check that the BEFORE "
                f"artifact was captured with the fallback_anchors column."
            )
            assert row.fallback_anchors < baseline, (
                f"{framework_id}: {row.fallback_anchors} fallback anchors "
                f"against {baseline} before the parser"
            )

    def test_the_pending_frameworks_now_anchor_on_real_statements(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        for framework_id in _assertable(overlay_present):
            row = live.by_id(framework_id)
            real = row.anchor_source_full_text + row.anchor_source_description
            # Attainable [0, row.links]. Every one of the eleven is at 0 today
            # because every one resolves 0 links [measured], so this is the
            # assertion that fails hardest before the work and can fail after
            # it if a parser stores titles where statements belong.
            assert real > 0, (
                f"{framework_id} resolved {row.by_title + row.by_id} links and "
                f"not one anchor came from a control statement"
            )
            # Rule 1's own invariant. Catches an instrument bug rather than a
            # parser bug, which is why it is cheap and worth keeping.
            assert (
                row.anchor_source_full_text
                + row.anchor_source_description
                + row.anchor_source_title
                + row.anchor_source_synthetic
            ) == row.by_title + row.by_id, framework_id


class TestNoRegression:
    def test_iso_still_resolves(self, live, overlay_present: bool) -> None:  # type: ignore[no-untyped-def]
        """ISO reached 92 of 94 before this plan. Nothing here may cost it."""
        if not overlay_present:
            pytest.skip(
                "iso_27001 is restricted, single-user store licence, no "
                "reproduction without prior written permission, and its "
                "processed text is absent from this checkout. The v2 suite "
                "asserted >= 92 here and got 0 in CI."
            )
        row = live.by_id("iso_27001")
        # Attainable [0, 94] and [0, 94]. ISO is the corpus's only high-prose
        # fold, so a regression here is the most expensive one available.
        assert row.by_title + row.by_id >= 92, row.by_title + row.by_id
        assert row.distinct_anchors >= 91, row.distinct_anchors

    def test_the_frameworks_this_plan_did_not_touch_are_unchanged(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        before = _rows(_load(BEFORE_PATH))
        checked = 0
        for framework_id, previous in sorted(before.items()):
            if framework_id in JOIN_FLOORS:
                continue
            if not overlay_present and framework_id in OVERLAY:
                continue
            current = live.by_id(framework_id)
            assert current.distinct_anchors == previous["distinct_anchors"], (
                framework_id
            )
            assert current.by_title + current.by_id == (
                previous["by_title"] + previous["by_id"]
            ), framework_id
            checked += 1
        # Twenty untouched frameworks, of which iso_27001 is the only overlay
        # member, so 19 in CI and 20 locally. Without this the loop could
        # iterate zero times and report green, which is exactly what the
        # deleted skip did.
        assert checked >= 19, f"only {checked} untouched frameworks compared"


class TestCommittedAfterReport:
    """All eleven, gated off tracked evidence, on every machine.

    The live-corpus tests above cannot assert eight of the eleven in CI,
    because those frameworks' text routes to the gitignored overlay. This class
    closes that hole without weakening a floor: the AFTER artifact is tracked,
    carries counts and digests and no anchor text, and is produced by a
    separate command from the floors it is checked against.
    """

    def test_the_after_artifact_was_captured_with_the_full_corpus(self) -> None:
        after = _load(AFTER_PATH)
        expected = len(_expected_framework_ids())
        # Attainable: any non-negative integer. This is the assertion that
        # stops the AFTER state being captured on the tracked corpus and then
        # certified, which is ledger lesson 5.
        assert after["corpus_framework_count"] == expected, (
            f"the AFTER report covers {after['corpus_framework_count']} "
            f"frameworks, not {expected}. It must be captured on a checkout "
            f"holding the licensed overlay, with the same command and the same "
            f"interpreter as the BEFORE state."
        )
        assert len(after["corpus_sha256"]) == 64
        assert after["corpus_sha256"] != _load(BEFORE_PATH)["corpus_sha256"], (
            "the AFTER report names the same corpus as the BEFORE report, so "
            "nothing was rebuilt between them"
        )

    def test_the_after_artifact_clears_every_floor_including_the_licensed_ones(
        self,
    ) -> None:
        rows = _rows(_load(AFTER_PATH))
        misses: list[str] = []
        for framework_id, floor in sorted(JOIN_FLOORS.items()):
            row = rows[framework_id]
            if row["resolution_rate"] + 1e-9 < floor:
                misses.append(
                    f"{framework_id} {row['resolution_rate']:.4f} < {floor:.4f}"
                )
        # Attainable [0, 11] misses. This is the only place etsi's floor of
        # 1.00 is asserted on a machine without the ETSI source, and deleting
        # it is the repair the v2 suite invited.
        assert misses == [], misses

    def test_the_after_artifact_carries_no_free_text(self) -> None:
        """It is tracked, so it must be safe to track.

        Structural rather than heuristic. The report is counts, ratios, ids and
        digests. A string field longer than a framework id is somewhere prose
        could sit unnoticed, and once tracked this file is also scanned by
        tests/test_licensed_text_not_tracked.py.
        """
        after = _load(AFTER_PATH)
        long_strings: list[tuple[str, int]] = []

        def walk(node: object, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")
            elif isinstance(node, str) and len(node) > 128:
                long_strings.append((path, len(node)))

        walk(after, "$")
        # Attainable [0, n]. The longest legitimate string is the corpus path.
        assert long_strings == [], long_strings

    def test_the_after_artifact_matches_the_live_report_where_both_can_see(
        self, live, overlay_present: bool
    ) -> None:  # type: ignore[no-untyped-def]
        """The committed artifact must not be hand-edited.

        Without this, the class above degrades into trusting a JSON file that a
        worker under a red build could open in an editor.
        """
        rows = _rows(_load(AFTER_PATH))
        compared = 0
        for framework_id, row in sorted(rows.items()):
            if not overlay_present and framework_id in OVERLAY:
                continue
            current = live.by_id(framework_id)
            assert current.by_title == row["by_title"], framework_id
            assert current.by_id == row["by_id"], framework_id
            assert current.distinct_anchors == row["distinct_anchors"], framework_id
            compared += 1
        # 22 or more in CI, every framework locally. A collapsed comparison is
        # how this goes quiet.
        assert compared >= 22, f"only {compared} rows cross-checked"


class TestSpecAcceptance:
    """Spec Part 1.9, checked against stored text rather than the join flag.

    Both surviving checks in the v2 suite globbed PROCESSED_FRAMEWORKS_DIR, and
    `.gitignore:37` and `:38` keep etsi.json and iso_27001.json off disk in a
    fresh clone, so the glob returned 30 of 32 [measured] and the two
    frameworks under the strictest licence got the least checking. Under the
    three-tier model the glob would return 23 of 32. These loops are driven
    from the union of the glob and OVERLAY_FRAMEWORK_IDS, and an absent file is
    recorded by name rather than skipped past.
    """

    def _resolve(self, framework_id: str) -> Path | None:
        path = PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
        return path if path.exists() else None

    def _partition(self) -> tuple[list[Path], list[str]]:
        present: list[Path] = []
        absent: list[str] = []
        for framework_id in sorted(_expected_framework_ids()):
            path = self._resolve(framework_id)
            if path is None:
                absent.append(framework_id)
            else:
                present.append(path)
        # An absent file is legal only under a licence. Anything else is a
        # missing parser output and must be loud.
        unexplained = [f for f in absent if f not in OVERLAY]
        assert not unexplained, (
            f"{unexplained} have no processed JSON and no licence that "
            f"explains the absence. A framework cannot leave these checks by "
            f"going missing."
        )
        return present, absent

    def test_every_processed_framework_has_a_parser(self) -> None:
        parsed = {p.stem[len("parse_"):] for p in PARSERS_DIR.glob("parse_*.py")}
        # Attainable: the empty set once the eleven land, up to eleven ids
        # before then.
        assert _expected_framework_ids() - parsed == set()

    def test_no_version_field_says_opencre(self) -> None:
        present, absent = self._partition()
        offenders = [
            path.name
            for path in present
            if "opencre-" in str(json.loads(path.read_text(encoding="utf-8"))["version"])
        ]
        # Attainable [0, 32]. Today it is exactly 11, and they are exactly the
        # eleven this plan gives a parser to, every one reading
        # "opencre-2026-04-28" [measured, this run]. The check is therefore red
        # today and turns green only when all eleven land, which is what a gate
        # looks like.
        assert offenders == [], offenders
        assert set(absent) <= OVERLAY, sorted(set(absent) - OVERLAY)

    def test_every_framework_meets_its_parsers_declared_prose_floor(self) -> None:
        """The gate with teeth, in place of a comparison against zero.

        honest_prose_fraction returns a ratio, so `> 0.0` passed on one prose
        control in csa_ccm's 224 [measured]. The declared floor is read out of
        the parser source with a regex rather than by importing, matching the
        convention in tests/test_parser_manifest_coverage.py, so a parser whose
        extraction dependency is missing is still covered.

        Only 2 of the 21 parsers that existed before this plan declare a floor,
        iso_27001 at 0.96 and owasp_llm_top10_2026 at 1.0 [measured, this run].
        The other 19 inherit BaseParser.min_prose_fraction = 0.0, which no
        parser can miss. Raising those 19 is separate work, recorded in "What
        this plan does not close". This test holds every declared floor and
        ratchets the unfloored count so it cannot grow.
        """
        from tract.parsers.base import BaseParser
        from tract.schema import Control

        present, absent = self._partition()
        offenders: list[str] = []
        unfloored: list[str] = []
        floors_held = 0
        for path in present:
            source_file = PARSERS_DIR / f"parse_{path.stem}.py"
            assert source_file.exists(), (
                f"{path.stem} has processed output and no parser module"
            )
            match = _MIN_PROSE.search(source_file.read_text(encoding="utf-8"))
            if match is None:
                unfloored.append(path.stem)
                continue
            floor = float(match.group(1))
            # A declared 0.0 is the inherited default wearing a costume.
            assert floor > 0.0, f"parse_{path.stem}.py declares a floor of 0.0"
            data = json.loads(path.read_text(encoding="utf-8"))
            controls = [Control(**c) for c in data["controls"]]
            fraction = BaseParser.honest_prose_fraction(controls)
            floors_held += 1
            if fraction + 1e-9 < floor:
                offenders.append(f"{path.stem}: {fraction:.4f} < {floor:.4f}")

        # Attainable [0, floors_held]. Every one of the eleven declares 0.90 to
        # 1.00 in its task and every one reports honest_prose_fraction exactly
        # 0.0000 today [measured, this run], so this is red before the work and
        # can go red again on any parser that stores titles where statements
        # belong.
        assert offenders == [], offenders

        missing = sorted(set(PENDING) & set(unfloored))
        assert missing == [], (
            f"{missing} have a parser and no declared min_prose_fraction. "
            f"Every one of the eleven declares one in its task, so a missing "
            f"floor means the declaration was dropped in implementation."
        )

        # Ratchet, not a refactor. Attainable [0, len(present)]. 19 of the 21
        # pre-existing parsers inherit the 0.0 default [measured], and this
        # fails upward the moment a new parser ships without a floor.
        assert len(unfloored) <= 19, sorted(unfloored)

        # Positive control against a collapsed file list: 4 floors are readable
        # in CI (nist_800_63, enisa, nist_ssdf, owasp_llm_top10_2026) and 13
        # locally, once the eleven land.
        assert floors_held >= 4, f"only {floors_held} declared floors were read"
        assert set(absent) <= OVERLAY, sorted(set(absent) - OVERLAY)

    def test_restricted_frameworks_are_named_when_they_cannot_be_checked(
        self,
    ) -> None:
        """The exemption is stated, not inferred.

        RESTRICTED_FRAMEWORK_IDS is imported here so that a framework leaving
        the restricted tier stops being exempt in the same commit.
        """
        _, absent = self._partition()
        unchecked = sorted(set(absent))
        for framework_id in unchecked:
            assert framework_id in OVERLAY, framework_id
        # ETSI declares min_prose_fraction = 1.0, which is the strictest floor
        # in the plan, and in a fresh clone nothing checks it. Recording the
        # names is the minimum honest reporting of that hole.
        if unchecked:
            print(
                f"unchecked under licence: {unchecked} "
                f"(restricted: {sorted(RESTRICTED_FRAMEWORK_IDS)})"
            )
```

- [ ] **Step 3: Run the acceptance suite on the full corpus**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_corpus_acceptance.py -q -rs
```

Run this on the machine holding the licensed overlay, so `overlay_present` is
`True` and all eleven assert. Expected: PASS with zero skips. Any failure names
the framework and the column, so fix the parser, not the threshold. A skip in
this run means the overlay was not found, and that invalidates the AFTER state
before it is captured.

If a floor genuinely cannot be met, the repair is a plan amendment with the
re-derivation written down, never an edit to `JOIN_FLOORS`. That constant is
committed in Task 1 and `git log -S JOIN_FLOORS` shows every touch.

- [ ] **Step 4: Capture the AFTER state**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" scripts/corpus_report.py --out results/corpus/after_parsers.json
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
from pathlib import Path

before = json.loads(Path("results/corpus/before_8cf44b3.json").read_text(encoding="utf-8"))
after = json.loads(Path("results/corpus/after_parsers.json").read_text(encoding="utf-8"))
rows = {r["framework_id"]: r for r in before["per_framework"]}

def real(row: dict) -> int:
    return row.get("anchor_source_full_text", 0) + row.get("anchor_source_description", 0)

header = f"{'framework':26s} {'resolved':>14s} {'anchors':>14s} {'fallback':>14s} {'statement anchors':>20s}"
print(header)
for row in after["per_framework"]:
    old = rows.get(row["framework_id"], {})
    print(
        f"{row['framework_id']:26s} "
        f"{old.get('by_title', 0) + old.get('by_id', 0):6d} -> {row['by_title'] + row['by_id']:5d} "
        f"{old.get('distinct_anchors', 0):6d} -> {row['distinct_anchors']:5d} "
        f"{old.get('fallback_anchors', 0):6d} -> {row['fallback_anchors']:5d} "
        f"{real(old):9d} -> {real(row):8d}"
    )

t_old, t_new = before["totals"], after["totals"]
print(f"\ncorpus frameworks {before['corpus_framework_count']} -> {after['corpus_framework_count']}")
print(f"resolved {t_old['by_title'] + t_old['by_id']} -> {t_new['by_title'] + t_new['by_id']} of {t_new['links']}")
print(f"distinct anchors {t_old['distinct_anchors']} -> {t_new['distinct_anchors']}")
print(f"fallback anchors {t_old['fallback_anchors']} -> {t_new['fallback_anchors']}")
print(f"statement-sourced anchors {real(t_old)} -> {real(t_new)}")
print(f"truncated {t_old['truncated']} -> {t_new['truncated']}")
print(f"not indexed {t_old['dropped_by_prose_rule']} -> {t_new['dropped_by_prose_rule']}")
PYEOF
```

Expected, by summing each task's stated ceiling onto the BEFORE totals
**[derived]**:

| total | before | after |
|---|---|---|
| links resolved | 3,666 | 4,389 of 4,405 |
| distinct anchors | **1,749** | 1,902, a delta of **+153** |
| fallback anchors, the eleven | **299** | about 16 |
| statement-sourced anchors, the eleven | **0** | about 718 |
| controls not in the prose index | **558** | about 83 |

The 723 newly resolved links come from dsomm 213, wstg 109, nist_800_63 78,
owasp_proactive_controls 76, enisa 68, nist_ssdf 46, etsi 36, samm 30,
csa_ccm 29, biml 21, owasp_top10_2021 17.

**The anchor delta is +153, not the +452 the v2 plan predicted, and the
correction matters more than its size.** The eleven frameworks' 734 links
already land on **299** distinct fallback anchors today **[measured,
orchestrator: dsomm 18, wstg 59, nist_ssdf 44, enisa 33, samm 30, csa_ccm 29,
nist_800_63 25, biml 17, etsi 24, owasp_proactive_controls 10,
owasp_top10_2021 10]**. Measured against that baseline, **seven of eleven
parsers move the anchor count by exactly zero and ETSI loses 10** (24 fallback
anchors to 14 clause anchors). Anchor count is therefore the wrong column to
headline. The right one is the last two rows of the table: 0 statement-sourced
anchors today, because all eleven resolve 0 of their links, rising to about 718.
Report both. The `+452` figure must never enter the RUN-LEDGER.

`dropped_by_prose_rule` is **558**, not the 522 the v2 plan states. The v2 total
sums only frameworks carrying curated links, so NIST AI RMF 25, AIUC-1 10 and
CoSAI 1 are invisible to it **[measured, Data Scientist]**.

`truncated` rises, mostly from wstg and etsi. Record the number and assert
nothing about it, because no task derived a ceiling for it.

- [ ] **Step 5: Write the result into the run ledger**

Append a `## Phase A-parsers COMPLETE` block to
`.superpowers/autonomous-run/RUN-LEDGER.md` carrying, at minimum:

- the before and after totals from Step 4, including `corpus_framework_count` on
  both sides, so a reader can tell which corpus each was measured on.
- the per-framework `distinct_anchors` change **and** the per-framework
  statement-anchor change, side by side.
- the corrected headline: **+153 distinct anchors [measured]**, with one line
  saying the plan predicted +452 against a baseline of zero and the baseline was
  299.
- the three frameworks whose anchor count fell or stayed flat, and why.
- the training-link count from Task 14 as **4,127 -> 4,401 [measured, Data
  Scientist]**, not 4,402. A fourth link falls under the 10-character floor:
  `nist_800_63` `section_name == 'are g'`, 5 characters.
- `dropped_by_prose_rule` as **558 [measured]**, not 522.
- the rebuild diff from Task 15, with the `renamed` bucket reported separately
  from `removed`.
- the sha256 of both corpus report artifacts and of the corpus each was built
  from.
- the frameworks whose acceptance rows were skipped under licence in the CI run,
  by name.

Every number tagged `[measured]` or `[derived]`. No number in this block may be
sourced from the plan file, which is gitignored and untracked **[measured]**.
Source each from `results/corpus/after_parsers.json` or from a named agent's
measurement.

- [ ] **Step 6: Licence gate first, then the full suite, typecheck, commit**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

# The licensed-text gate runs alone and first. `pytest tests/ -x` collects
# alphabetically and test_corpus_acceptance.py sorts before
# test_licensed_text_not_tracked.py, so an abort in the former would stop the
# latter from ever running. CI uses -x [measured: .github/workflows/ci.yml:65].
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py -q

PYTHONPATH=. "$PY" -m pytest tests/ -q -m "not integration"
"$PY" -m mypy tract/ parsers/ scripts/ --strict

git add results/corpus/after_parsers.json tests/test_corpus_acceptance.py \
        .superpowers/autonomous-run/RUN-LEDGER.md
git status --short results/corpus/after_parsers.json
git commit -m "test: gate the corpus on anchor provenance, not only on links resolved"
```

`git add` must exit **0** and `git status --short` must show
`A  results/corpus/after_parsers.json`. If it exits 1 with `The following paths
are ignored`, Task 1's `.gitignore` negation is in the non-working form. Fix the
negation. **Do not use `git add -f`.** Forcing an ignored path into git is how
licensed text escaped four times, and `results/corpus/` is trackable by design
precisely so that no evidence artifact ever needs it.

<<<SELF-REVIEW>>>

## Self-review

### Spec coverage

Spec Part 1 items, and where each is discharged.

| spec item | task |
|---|---|
| 1.1 the twelve frameworks with no parser | Tasks 3-13 cover eleven. ISO 27001 landed earlier and Task 16 asserts it did not regress, on any checkout holding the licensed overlay |
| 1.2 source manifest per parser | every parser reads through `read_source`/`read_source_bytes`. `tests/test_parser_manifest_coverage.py` is run in Task 15 Step 7 |
| 1.2 `expected_count` raises | every parser declares one, and `csa_ccm`'s corrected 224 is the reason that task exists |
| 1.2 `min_prose_fraction` on stored text | every parser declares one, each derived from a measured statement-length distribution. Task 16 now compares the stored fraction against that declared floor rather than against zero |
| 1.2 no clock | every parser declares `fetched_date` as a `ClassVar` |
| 1.3 repair layer with an audit record | Tasks 7, 8, 11, 12 and 13 all move or synthesise text and all write `write_repair_audit` |
| 1.4 the thirteen parsers | eleven here. ISO 27001 and OWASP LLM Top 10 2026 already landed |
| 1.5 retire both gates | Task 14, with the corrected 4,127 -> **4,401** |
| 1.6 OWASP LLM Top 10 2026 | out of scope by instruction. It has a tracked processed file and appears in neither merged corpus **[measured, this run]**, so Task 15's re-merge is what brings it in, and its 10 controls report as `added` |
| 1.7 ground-truth divergence | not in scope. It is a CLI path change, not a parser |
| 1.8 review status vs rebuild | not in scope. Task 15 supplies the per-control changed list the schema column would key on, and that column still does not exist |
| 1.9 acceptance tests | Task 16 `TestSpecAcceptance`, plus the parse-twice determinism test in Task 3 |

### Ledger lessons, and where each binds

- **Lesson 1, another open channel.** Task 13 Steps 1 and 8 check the ETSI routing on both the
  artifact and the merge. Task 1 adds a `.gitignore` line per `CONDITIONAL_FRAMEWORK_IDS` member,
  because the licence tiering opens seven channels that were previously unmodelled.
- **Lesson 2, read the file first.** Every snippet is written against `8cf44b3`, and Task 9 Step 1
  re-verifies a premise the previous plan asserted. That premise check is itself corrected: the v2
  snippet scans all columns and yields 54 cells over columns [3, 4] with 14 mid-sentence, not the
  stated 47 over [3] with 0 **[measured, ML Engineer, ran the snippet verbatim]**.
- **Lesson 3, a gate that cannot fire.** Every floor is derived from the link data's arithmetic
  maximum. The v2 self-review claimed this was discharged by
  `test_no_floor_exceeds_its_arithmetic_ceiling`, which is **half a control**: it guards a floor
  that is unreachably high and says nothing about a floor that cannot be missed. Task 16 asserts
  `0.0 < floor <= 1.0` and states the attainable range beside every assertion.
- **Lesson 4, a decorative control.** Task 14 deletes both retired constants rather than leaving
  them unread. Task 15's prose stop rule is an `raise SystemExit`, not an instruction to a reader.
- **Lesson 5, baseline captured differently.** Task 1 records `corpus_framework_count` in the BEFORE
  artifact, and Task 16's `test_the_after_artifact_was_captured_with_the_full_corpus` fails if the
  AFTER state was captured on the tracked corpus.
- **Lesson 6, a step preceding what rewrites its inputs.** Every task now carries an
  `**Invalidates:**` line. A grep of the v2 plan returned **0** hits for `invalidates`
  **[measured]**, the fourth recurrence.
- **Lesson 7, a fabricating transform.** The text-moving tasks write audit records and Task 13
  refuses the technique-level segmentation it cannot verify.
- **Lesson 8, a number without an artifact.** Every number carries a tag, and Step 5 forbids
  sourcing a ledger number from this plan file, which is untracked.
- **Lesson 9, new: a gate that cannot fail.** Six of the v2 suite's nine terminal assertions passed
  by construction. Named in the Global Constraints, fixed in Task 16.

### Premortem findings, honestly scored

**Closed by this revision:** the vacuous channel-parity test (dict-aware loader), the wrong
load-bearing column (`fallback_anchors` plus four `anchor_source_*` columns, and a text-quality
delta reported beside the anchor delta), the diff blind to `full_text`, `nested_anchors` counting
strict prefixes only, the three run-halting predictions (`csa_ccm` `by_title`, `owasp_top10_2021`
`by_title`, `biml` `distinct_anchors`), the CI hard-fail on `etsi` and `iso_27001`, the missing
pre-rebuild snapshot, the absent `invalidates` column, the gitignored floors, the unstageable
evidence artifacts, and the six unfailable assertions.

**Closed by measurement during adjudication, not by this plan:** CAPEC and CWE were never
test-rebuilt. With `defusedxml==0.7.1` installed, both parsers reproduce **558 of 558** and **1,331
of 1,331** baseline hashes with **0 mismatch** **[measured, orchestrator]**, taking pre-measured
rebuild coverage to **89.7%**. `openpyxl` now reports `DEFUSEDXML: True` from the same install. The
250-item ceiling study is safe: zero of its items fall in the eleven frameworks, capec and cwe
account for 111 of 250, and both reproduce byte-identically **[measured]**. Any claim of 45%
coverage is stale and must not be repeated.

**Three claims in the v2 self-review were false and are withdrawn:**

1. It said *"CAPEC and CWE are untouched"* while **Task 14 restores all 44 contested CAPEC links**.
   Given human alpha-1 of 0.181 on 83 CAPEC items against 0.572 pooled, recovering terse CAPEC links
   such as `'UDP Ping'` and `'Fuzzing'` is not self-evidently progress. Task 14 splits its commit by
   framework so the Part 5 weighting decision has a lever.
2. It said *"Tasks 7 and 12 are the only text-moving transforms"*. **Task 8 synthesises 17 CCM
   domain aggregates from member titles**, which `honest_prose_fraction` counts as prose and no
   column distinguishes from a normative statement, and **Task 13's ETSI parser captures the running
   page header** as the heading for clauses 5, 6 and 7, all three reading
   `'ETSI GR SAI 005 V1.1.1 (2021-03)'` **[measured]**, with clause 7 carrying about 22.6 KB of
   front matter as one control's statement. Task 11's ENISA Annex-C fallbacks are a third. All of
   them write audit records.
3. It said premortem **C9 "landed"**. The ledger records the cheap half captured and the durable fix
   needing a schema column that does not exist. A repository-wide grep for
   `control_hash|text_hash|description_hash` returns **zero** hits **[measured, this run]**, and the
   `assignments` DDL at `tract/crosswalk/schema.py:40-56` has 15 columns, none a content hash
   **[measured]**. C9 is open and is listed below.

### Placeholder scan

No step says "similar to Task N". No function body raises `NotImplementedError` or contains `...`.
No `expected_count` is `0`. Every count is stated: dsomm 194, samm 30, owasp_top10_2021 10,
owasp_proactive_controls 10, wstg 115, csa_ccm 224, nist_ssdf 42, nist_800_63 100 (floor), enisa 50,
biml 146 (floor), etsi 25.

No step pauses for an owner decision. Four judgement calls are ruled with the evidence stated: the
CCM domain statement (member titles, not member specifications), the SAMM statement composition
(`shortDescription`, not `longDescription`), BIML's `output:2` (resolved by name, audited), and
ETSI's grain (clause, with the anchor-count regression declared in advance).

Two commands install software: `pip install "openpyxl==3.1.5"` and `"defusedxml==0.7.1"`. Both are
pinned and recorded in `requirements.txt`. Both are already present in the mandated interpreter
**[measured, this run]**, so both are idempotent verifications rather than changes.

### Type consistency

`FrameworkJoin` and `CorpusReport` are dataclasses with fully annotated fields. `FrameworkJoin`
gains eight fields in Task 1 and loses none: `fallback_anchors`,
`distinct_anchors_pre_truncation`, `contained_anchors`, `anchor_source_full_text`,
`anchor_source_description`, `anchor_source_title`, `anchor_source_synthetic`, `distinct_hubs`, plus
`links_per_hub`. `CorpusReport` gains `corpus_framework_count: int`. `build_corpus_report` returns
`CorpusReport`. `check_join_floors` returns `list[str]`. `JOIN_FLOORS` is
`Final[dict[str, float]]` and `JOIN_WRONG_ANCHOR_BUDGET` is `Final[dict[str, int]]`, both defined in
`tract/corpus_report.py` in Task 1.

Every parser's `parse()` returns `list[Control]`. The class-method entry points tests call directly
return what their task's Interfaces block declares:
`DsommParser.activities_to_controls -> list[Control]`, `SammParser.build_controls -> list[Control]`,
`OwaspTop102021Parser.control_from_markdown -> Control`,
`WstgParser.build_controls -> tuple[list[Control], list[dict[str, object]]]`,
`CsaCcmParser.rows_to_controls -> list[Control]`,
`NistSsdfParser.rows_to_controls -> list[Control]`,
`Nist80063Parser.sections_from_html -> list[Control]`,
`EnisaParser.rows_to_units -> list[tuple[str, str]]`,
`BimlParser.build_controls -> tuple[list[Control], list[dict[str, object]]]`,
`EtsiParser.clauses_from_text -> dict[str, tuple[str, str]]` and
`EtsiParser.build_controls(clauses, alternates_by_name: dict[str, str]) -> list[Control]`.

`Control.metadata` is `dict[str, str | list[str]] | None`, so every metadata literal uses only `str`
and `list[str]` values. The `alt_ids` and `alt_titles` channels both read `list[str] | str`.
`expected_sha256` is `ClassVar[str | None]` on ten parsers and `ClassVar[dict[str, str] | None]` on
BIML, which reads two files. `mypy tract/ parsers/ scripts/ --strict` runs in Task 16 Step 6 over
everything at once.

**One typecheck failure is predicted and must be closed in Task 8**: `openpyxl` ships no `py.typed`,
`types-openpyxl` is not added, no `[[tool.mypy.overrides]]` entry exists, and the pyproject line the
v2 plan specifies lands in `optional-dependencies.llm` rather than `dependencies`. CI installs
`-e .`, so it resolves the floor rather than the pin.

### What this plan does not close

- **CAPEC and CWE remain 57.3% of the training graph and Task 14 grows CAPEC.** The ceiling study
  measured human alpha-1 at **0.181 on 83 CAPEC items** against 0.572 pooled. This is a label-quality
  problem, not a parser problem, and it belongs to the training-mix weighting the spec's Part 5 will
  decide. Task 14's per-framework commit split is the lever, not the answer.
- **Source content integrity is unaddressed and one source gets promoted anyway.** Six upstream
  sources accept community pull requests. A SHA pin proves the bytes did not change in transit and
  says nothing about who wrote them. `nist_800_63` is deliberately unpinned, because Cloudflare
  injects a per-response bot token, and Task 14 promotes it from **0 to 79 training links**. `etsi`
  is fetched with a spoofed browser user-agent. A grep of the v2 plan returns **0** for `malicious`,
  `tamper`, `supply chain`, `untrusted`, `quarantine` and `content review` **[measured]**.
  `--accept-new-hash` is an alert with no adjudication rule, and an alert nobody knows how to answer
  gets approved. **The rule, until a better one is written:** a changed hash on a
  community-editable source is not accepted in the same session it is observed. Diff the extracted
  control text against the previous processed artifact, list every changed control id, and accept
  only when every change maps to a dated upstream commit or release note that a human has opened.
  A changed hash with no upstream record is quarantined, meaning the pin stays at the old value and
  the framework is excluded from the training links until adjudicated. For `nist_800_63`, which
  cannot be pinned at all, the substitute is the same text diff against the previous processed
  artifact on every fetch, and the diff is committed alongside the artifact.
- **Repair audits are unreadable and unreachable.** They store `statement_lengths`, a list of
  integers **[measured, plan line 2853]**, rather than the before and after text pair that
  `BaseParser.write_repair_audit`'s own docstring says a reviewer needs: *"A count says a repair
  fired. It does not say what moved, or where to, and a fragment attributed to the wrong control is
  a wrong compliance assertion carrying a plausible-looking provenance record. This is the file a
  reviewer reads to check one."* **[measured, `base.py:250-263`]**. `data/processed/repair_audit/`
  is gitignored at `.gitignore:43` **[measured]**, so no reviewer on another machine and no CI job
  can read one. The gitignore line is correct for restricted frameworks and wrong as a blanket rule.
- **The false human-reviewed claim lives in the generator, and the correction is partial.** The
  erratum **has** landed in two places: `tract/dataset/card.py:98` now says *"The dataset as a whole
  is not human-reviewed"*, and `tract/dataset/bundle.py:220` carries the same sentence in the Zenodo
  description **[measured, this run]**. It has **not** landed at `tract/dataset/card.py:137`, which
  still regenerates *"each individually reviewed by a cybersecurity domain expert"*, nor at
  `card.py:352`, `card.py:383`, `card.py:447` or `tract/publish/model_card.py:518`, all of which
  regenerate "expert-reviewed" on every publish **[measured]**. The next publish overwrites the
  correction and destroys the `#erratum-2026-08-15` anchor that `README.md:48` links to. Fix the
  generators before any publish, not after. This is held today only by the standing republication
  ban.
- **Nineteen of the 21 pre-existing parsers have no prose floor at all.** Only `iso_27001` (0.96)
  and `owasp_llm_top10_2026` (1.0) declare `min_prose_fraction` **[measured, this run]**. The other
  19 inherit `BaseParser.min_prose_fraction = 0.0`, which no parser output can miss, so the
  strongest stored-text gate in the codebase is switched off for 19 of 32 frameworks. This plan adds
  eleven declared floors and ratchets the unfloored count at 19 so it cannot grow. Raising the
  existing 19 needs a measured statement-length distribution per framework and is separate work.
- **Five of eleven parsers never reach `parse()` outside a skip in CI**: `csa_ccm`, `nist_ssdf`,
  `enisa`, `biml` and `etsi`, which are the two PDFs, the XLSX and the multi-document pair, meaning
  the most fragile extraction paths. This is Ruling R3's defect class verbatim. Task 16's
  `TestCommittedAfterReport` gates their **output** on every machine and does not exercise their
  **code**.
- **The three-tier licence model covers the frameworks this plan touches and not the corpus.**
  `csa_aicm` is tracked today with real prose under the identical CSA "all rights reserved, no
  redistribution" notice as `csa_ccm` **[measured, `tract/config.py` `FRAMEWORK_LICENSES`]** and is
  covered by no ruling. Seven more tracked frameworks carry CC-BY-SA-4.0 and sit outside
  `CONDITIONAL_FRAMEWORK_IDS`: `asvs`, `owasp_cheat_sheets`, `owasp_dsgai`, `owasp_llm_top10`,
  `owasp_llm_top10_2026`, `owasp_ml_top10`, `owasp_agentic_top10` **[measured]**. Nothing in this
  plan changes them, and a reader must not take the three tiers as the complete licence model.
- **`results/ceiling_study/` stays gitignored, so the 250 owner judgments stay unbacked by git.**
  `results/corpus/` is un-ignored here and `results/ceiling_study/` is not, because the latter holds
  `ceiling_items.json` with a `control_text` field per item **[measured, this run]**. None of the
  250 items belongs to a restricted or conditional framework, and 6 belong to `owasp_llm_top10`
  under CC-BY-SA-4.0 and 54 to `owasp_ai_exchange` under an undetermined licence **[measured]**.
  Tracking that file is a licence decision that belongs to the owner and not to this plan.
- **`.github/workflows/ci.yml:65` still runs `pytest tests/ -x`**, so the first failure anywhere
  stops every later test file, including the licensed-text gate. Task 16 Step 6 works around it
  locally by running that gate first and alone. The workflow itself is untouched here.
- **`owasp_cheat_sheets` still carries 391 links on 49 anchors with 384 truncated** **[measured]**.
  It has a parser, so it is out of scope, and after this plan it is the worst concentration in the
  corpus by a wide margin. The AFTER report says so.
- **ETSI's anchor count falls from 24 to 14.** Declared, not discovered. The alternative was
  prose-heuristic segmentation of technique names that appear mid-sentence in 9 of 24 cases.
- **Nine WSTG links, one NIST 800-63 link and one DSOMM link remain unresolvable.** Each has a named
  upstream cause and none is repairable from the source. Nine of those WSTG links reach the trainer
  with a literal id as the anchor, such as `"WSTG-BUSL-$$"` and `"WSTG-INPV-00"`, because Task 14's
  gate falls back to `section_name` and those names clear the 10-character floor.
- **`pdfplumber` is pinned at 0.11.10 and installed at 0.11.4** on the mandated interpreter
  **[measured, this run]**, so every `[measured]` PDF figure in Tasks 12 and 13 came from a build CI
  will not reproduce.
- **Premortem C9 is open.** The durable fix needs a content-hash column on `assignments` that does
  not exist. A repository-wide grep for `control_hash|text_hash|description_hash` returns zero
  **[measured]** and the DDL has 15 columns, none of them a content hash **[measured]**.
