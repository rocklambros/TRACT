# Eleven Remaining Parsers, Measured Against One Corpus Instrument

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the eleven frameworks that still have no parser a real one, and prove the corpus got better rather than merely louder, by building the measurement first and gating every parser on it.

**Architecture:** Task 1 builds `tract/corpus_report.py`, the single instrument. It resolves every curated link through `ProseIndex.lookup` and reports, per framework: links by channel, distinct resolved anchors, **the fallback anchors the trainer already gets today**, where each anchor's text came from, links per anchor, anchors truncated at `MAX_ANCHOR_CHARS`, anchors nested inside another anchor of the same framework, controls dropped by `_is_prose`, wrong-anchor risk, and the hub side. It is run against the current corpus before a line of parser code exists, and that BEFORE state is committed. The same instrument, the same code path, is the per-parser acceptance gate and the final corpus report. Each parser task states its framework's arithmetic join ceiling, derived from the link file and the source before the parser is written, and the floor the parser is measured against is that ceiling rounded down. Two frameworks need machinery that does not exist: an `alt_ids` channel on `ProseIndex` mirroring the existing `alt_titles`, and the retirement of two link gates that key on a section title.

**What v3 changes, and why the honest headline is smaller than v2's.** v2 claimed the eleven parsers add **+452 distinct anchors** against a BEFORE state of zero. That BEFORE state was wrong: `select_control_text` falls back to `section_name`, so the trainer already gets **299** distinct anchors for these 734 links. The honest delta is **+152 or more**, and seven of the eleven parsers move the anchor count by exactly nothing while ETSI loses ten. Their value is text quality, a paragraph in place of a phrase, and v2 had no column for it. v3 adds one. A plan whose load-bearing number is three times its truth is not a plan you can gate on.

**Tech Stack:** Python 3.12, pydantic v2, pdfplumber, openpyxl, PyYAML, beautifulsoup4, defusedxml, pytest, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-08-15-semantic-rebuild-design.md` (v2), Part 1.

**Supersedes:** `docs/superpowers/plans/2026-08-18-remaining-parsers-v2.md`, rejected 2026-08-19 after a second four-agent premortem across six lenses returned ~38 findings. v2 in turn superseded `docs/superpowers/plans/2026-08-16-remaining-parsers.md`, rejected 2026-08-16 on ~30 findings.

v2 was rejected for four reasons, all reproduced against source rather than argued: its channel-parity test built `ProseIndex([])` because the corpus JSON is a dict, so all 4,405 assertions read `True == True`; its headline metric was three times its truth; three acceptance gates halted a healthy run and the parser authors found six more; and its ETSI parser captured a running page header as a clause heading, shipping 22.6 KB of table of contents as one control's statement through every gate it declared.

**The parser bodies survived.** The premortem independently reproduced ENISA 68/68 end to end, csa_ccm 207+17=224, DSOMM 194/183, WSTG 109/118, and ETSI's clause resolution, and found them correct. v3 keeps Tasks 3, 4, 6, 7 and 10 almost verbatim. What changed is the instrument, the gates, the merge, the licence routing, and every number that was wrong.

**Adjudication:** `.superpowers/autonomous-run/premortem-v2/ADJUDICATION.md`. Rulings R4 through R7 and the author contradictions are in `V3-CONTRACT.md` and `V3-RESOLUTIONS.md` beside it.

---

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
### Task 1: The corpus quality instrument, and the BEFORE state

This is the only instrument in this plan. Everything downstream is measured with it, so it is built
and run before any parser exists. Version 2 of this task shipped an instrument with four defects,
each of which the premortem reproduced against source. All four are fixed here.

**Defect 1: the load-bearing column was the wrong column.** Version 2 reported the eleven pending
frameworks at `distinct_anchors == 0` and read the gain as `+452`. The trainer does not get zero
anchors for those links. `select_control_text` falls back to `(section_name or section_id)` when
`ProseIndex.lookup` returns nothing, so the eleven's 734 curated links already land on **299 distinct
fallback anchors** today: dsomm 18, wstg 59, nist_ssdf 44, enisa 33, samm 30, csa_ccm 29,
nist_800_63 25, biml 17, etsi 24, owasp_proactive_controls 10, owasp_top10_2021 10. **[measured,
orchestrator during adjudication, reproduced digit for digit by this task against
`data/processed/licensed/all_controls.json` sha256 in the artifact]** So `FrameworkJoin` gains a
`fallback_anchors` column and the report shows both sides.

**Defect 2: `nested_anchors` counted strict prefixes only.** An ETSI clause 5.2 rolled up over 5.2.2
nests undetected whenever the child is not sorted first. The column changes to containment, and the
old strict-prefix count survives as `contained_anchors` for continuity. Separately, two anchors that
share a 2,150-character prefix truncate to the identical string and merge into one set member, which
suppresses `nested_anchors` and quietly lowers `distinct_anchors` with no diagnostic. That is why
`distinct_anchors_pre_truncation` exists.

**Defect 3: `wrong_anchor_risk` could only fire in the title branch.** Nine of the eleven are
engineered to resolve through the id channel, so `== 0` was unfailable for them, while the one
framework where it does fire (csa_ccm, whose `IPY` link reaches control IPY-01's title rather than
the IPY domain) would halt a healthy run. The counter is redesigned into three detectors, two of
which reach the id branch, and this task states the attainable range per framework so nobody asserts
zero where zero is guaranteed.

**Defect 4: the channel-parity test was vacuous.** It built `ProseIndex(data if isinstance(data,
list) else [])`. Both `data/processed/all_controls.json` and `data/processed/licensed/all_controls.json`
are **dicts** with keys `[framework_count, frameworks, generated_date, total_controls]` **[measured,
this task]**, so the index was built from `[]` and all 4,405 assertions reduced to `True == True`.
The named guard for "the exact defect that got the previous plan rejected" asserted nothing. It now
uses the dict-aware loader.

#### What the honest gain is

Counting links resolved is not enough, and that is why the previous plan was rejected. 615 links
unstacked onto 615 distinct anchors and 615 links collapsed onto 40 coarse anchors produce the same
rising line. The report's load-bearing pair is `distinct_anchors` **beside** `fallback_anchors`.

| | the eleven's distinct anchors | source |
|---|---|---|
| BEFORE, as version 2 reported it | 0 | wrong, `distinct_anchors` alone |
| BEFORE, what the trainer gets | **299** | **[measured]** `fallback_anchors` |
| AFTER, summed from the eleven parser tasks' own predictions | **451 or more** | **[derived]** dsomm ≥182, wstg ≥55, nist_ssdf ≥44, enisa 33, samm 30, csa_ccm 29, nist_800_63 25, biml 19, etsi 14, proactive 10, top10 10 |
| honest delta | **+152 or more** | **[derived]** |

`+452` is the AFTER figure compared against a BEFORE of zero. It must never enter the run ledger.
The figure the ledger records is `299 -> 451+`, delta `+152 or more`, corpus-wide `1,749 -> 1,901+`.

Seven of the eleven parsers move the anchor count by **exactly zero**: csa_ccm 29 to 29, enisa 33 to
33, nist_800_63 25 to 25, nist_ssdf 44 to 44, samm 30 to 30, owasp_proactive_controls 10 to 10,
owasp_top10_2021 10 to 10. **[derived from the measured 299 against each parser task's own stated
prediction]** ETSI goes 24 to 14, a **loss** of ten anchors, because 36 links collapse onto 14 clause
anchors instead of 24 section names. The entire count gain comes from three frameworks: dsomm 18 to
182 or more, wstg 59 to 55 or more, biml 17 to 19.

If the count is flat or negative for eight of eleven, the parsers have to be worth something else,
and they are: **text quality**. A fallback anchor is a phrase. A parsed anchor is a paragraph. That
is a real change to what the encoder reads and version 2 had no column for it. This task adds the
four `anchor_source_*` columns and records `anchor_chars` per link, and the BEFORE median fallback
anchor length is the baseline the AFTER is read against:

| framework | median fallback anchor chars, BEFORE | mean |
|---|---|---|
| owasp_proactive_controls | 2.0 | 2.1 |
| nist_800_63 | 7.0 | 6.1 |
| wstg | 12.0 | 12.0 |
| biml | 14.0 | 14.5 |
| samm | 19.5 | 19.9 |
| etsi | 20.0 | 19.7 |
| dsomm | 21.5 | 18.8 |
| owasp_top10_2021 | 31.0 | 28.8 |
| csa_ccm | 37.0 | 37.1 |
| enisa | 44.0 | 45.1 |
| nist_ssdf | 156.5 | 155.9 |

**[measured, all, this task, over all 734 links including resolved-to-nothing rows]** Two of those
say the quiet part out loud. `owasp_proactive_controls` trains on a two-character anchor. `wstg`
trains on twelve characters of identifier. `nist_ssdf` at 156 characters is the one framework whose
fallback is already a sentence, which is exactly why its parser must not also use `section_name` as
the title.

#### The BEFORE state, measured

| framework | links | resolved | distinct anchors | fallback anchors | links/anchor | truncated |
|---|---|---|---|---|---|---|
| owasp_cheat_sheets | 391 | 391 | 49 | 0 | 7.98 | 384 |
| capec | 1799 | 1799 | 349 | 0 | 5.15 | 24 |
| cwe | 613 | 612 | 245 | 1 | 2.50 | 13 |
| nist_ai_100_2 | 45 | 45 | 22 | 0 | 2.05 | 21 |
| iso_27001 | 94 | 92 | 91 | 2 | 1.01 | 0 |
| the eleven pending | 734 | 0 | 0 | **299** | n/a | 0 |

**[measured, all]** The report reads `hub_links_curated.jsonl` (4,405 rows) rather than
`hub_links_by_framework.json` (4,406), because the curated file is what the trainer reads and the
extra row is an `owasp_ai_exchange` duplicate. **[measured]**

#### Two more numbers version 2 got wrong

`dropped_by_prose_rule` summed only frameworks carrying curated links, so it read 522 where the
corpus holds **558** un-indexed controls. The 36 invisible ones are NIST AI Risk Management
Framework 25, AIUC-1 Standard 10, CoSAI Landscape of AI Security Risk Map 1. **[measured, Data
Scientist, reproduced by this task]** The totals row now sums the whole corpus census, not the
link-bearing subset.

`wrong_anchor_risk` read 9 corpus-wide under the title-only detector. Under the three detectors it
reads **40**: asvs 1, cwe 2, iso_27001 5, nist_ai_100_2 20, owasp_ai_exchange 11, owasp_ml_top10 1.
**[measured, this task]** The 31 new flags are not noise. Four of them are ISO control titles the
parser wrote without spaces (`'Addressinginformationsecurity within supplier agreements'` for A.5.20,
`'Redundancyofinformation processing facilities'` for A.8.14), two are CWE-598 renamed upstream from
`'Use of GET Request Method With Sensitive Query Strings'` to `'Use of HTTP Request With Sensitive
Query String'`, and twelve are NIST AI 100-2 links where one coarse section id such as `Sec. 3.3.2`
answers for two differently named items. **[measured, this task, all five quoted strings read out of
the corpus]** Nothing in this repository reported any of them before now.

**Files:**
- Create: `tract/corpus_report.py`
- Create: `scripts/corpus_report.py`
- Create: `tests/test_corpus_report.py`
- Create: `results/corpus/before.json`
- Create: `results/corpus/link_resolution_before.jsonl`
- Modify: `.gitignore`
- Modify: `tract/text_selection.py`

**Interfaces:**
- Consumes: `tract.text_selection.ProseIndex`, `TextSelection`, `canonical_framework`,
  `normalize_section_id`, `prepare_anchor`, `strip_markup`, `merged_corpus_path`, `_is_prose`;
  `tract.config.MAX_ANCHOR_CHARS`, `PROJECT_ROOT`, `TRAINING_DIR`, `RESTRICTED_FRAMEWORK_IDS`;
  `tract.io.atomic_write_json`, `atomic_write_text`; `data/training/hub_links_curated.jsonl`.
- Produces:
  - `FrameworkJoin` with the exact field set in the v3 contract Rule 1, all twenty-two fields.
  - `LinkResolution`, one per curated link, digest and length only, no anchor text.
  - `CorpusReport` with `per_framework`, `totals`, `corpus_path`, `corpus_sha256`,
    `corpus_framework_count`, `links_path`, `links_sha256`, `max_anchor_chars`, `resolution_rows`,
    `to_json()` and `by_id()`.
  - `build_corpus_report(links_path: Path | None = None, corpus_path: Path | None = None) -> CorpusReport`
  - `check_join_floors(report: CorpusReport, floors: Mapping[str, float]) -> list[str]`
  - `floors_for_report(report, floors, restricted=RESTRICTED_FRAMEWORK_IDS) -> tuple[dict[str, float], list[str]]`
  - `write_link_resolution(report: CorpusReport, path: Path) -> None`
  - `wrong_anchor_applicable(report: CorpusReport) -> dict[str, int]`
  - `format_table(report: CorpusReport) -> str`
  - `JOIN_CEILINGS`, `JOIN_FLOORS`, `CORPUS_EVIDENCE_DIR`, `FULL_CORPUS_FRAMEWORK_COUNT`,
    `TEXT_ORIGIN_METADATA_KEY`, `SYNTHETIC_TEXT_ORIGIN`.
  - `ProseIndex.by_title`, `ProseIndex.by_id`.

**Invalidates:**
- The plan's own headline `+452 distinct anchors`, in the Architecture paragraph at line 7 and
  wherever Task 16 copies it into `.superpowers/autonomous-run/RUN-LEDGER.md`. The ledger figure is
  `299 -> 451+`.
- Task 16 Step 1. `JOIN_FLOORS` is committed here, before any parser exists, per contract Rule 6.
  Task 16 imports it and must not redefine it.
- Task 16's `BEFORE = Path("results/corpus/before_8cf44b3.json")`. The artifact is
  `tract.corpus_report.CORPUS_EVIDENCE_DIR / "before.json"`, anchored to `PROJECT_ROOT`, and the
  `pytest.skip("no BEFORE artifact in this checkout")` beneath it is deleted per Ruling R3.
- Task 16's `test_no_new_framework_nests_an_anchor_inside_another`. Under containment, csa_ccm's
  domain aggregates are expected to nest (the plan's own note says 17 of 17), so `nested_anchors == 0`
  for every pending framework is no longer the right assertion.
- Task 16's `test_no_new_framework_carries_wrong_anchor_risk`. See Step 11 for the attainable range
  per framework and what to assert instead.
- Nothing at runtime. No training, evaluation or publication artifact reads this module yet.

- [ ] **Step 1: Confirm the interpreter and the shape of the corpus**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -c "import pdfplumber, yaml, bs4, pydantic, defusedxml, openpyxl, sys; print(sys.version)"
"$PY" -c "
import json
for p in ['data/processed/all_controls.json', 'data/processed/licensed/all_controls.json']:
    d = json.load(open(p, encoding='utf-8'))
    print(p, type(d).__name__, sorted(d) if isinstance(d, dict) else len(d))
"
```

Expected:

```
3.12.2 ...
data/processed/all_controls.json dict ['framework_count', 'frameworks', 'generated_date', 'total_controls']
data/processed/licensed/all_controls.json dict ['framework_count', 'frameworks', 'generated_date', 'total_controls']
```

**[measured]** If the first command fails, stop, every later step depends on it. If either file
prints `list`, stop as well: the loader below prefers the `frameworks` key and a shape change means
the corpus writer changed under this plan.

- [ ] **Step 2: Make the evidence directory tracked by design**

`.gitignore:3` is `results/`, so `git add results/corpus/before.json` exits 1 and stages nothing for
that path. Global Constraints forbid `git add -f`, and forcing ignored paths into git is how licensed
text escaped four times. The fix is a negation, and the form matters: `results/` excludes the
directory itself, git never descends into it, and a `!results/corpus/**` negation underneath is
inert. `results/**` excludes the contents and leaves the negation reachable. Both forms were run
against this repository with `git check-ignore -v`. **[measured]**

```bash
# .gitignore: replace line 3, which currently reads "results/"
```

```
# Evidence artifacts are the record a later reader audits, so results/corpus/
# is tracked by design. "results/" alone excludes the directory, git never
# descends into it, and any negation underneath is inert; "results/**"
# excludes the contents and leaves the negations reachable. Verified with
# git check-ignore -v.
results/**
!results/corpus/
!results/corpus/**
```

Contract Rule 2 also lists `!results/ceiling_study/` and `!results/ceiling_study/**`. That negation
is **not** added here. 73 files under `results/` are already tracked, 69 of them in
`results/ceiling_study/`, and un-ignoring the tree newly exposes exactly four untracked files:
`answers_llm_proxy.json`, `ceiling_answers_LLM_PROXY.json`, `LLM_PROXY_report.md`,
`LLM_PROXY_score_report.txt`. **[measured, this task]** No task in this plan reviews those four, and
this plan's whole-tree licence gate is red on purpose. Whichever task first commits a ceiling-study
artifact owns that negation and owns reviewing those four files first.

Verify:

```bash
git check-ignore -v results/corpus/before.json || echo "not ignored, which is the goal"
git check-ignore -v results/phase1b/anything.json
git status --porcelain results/ | head
```

Expected: the first prints `not ignored, which is the goal`. The second still reports
`.gitignore:N:results/**`. The third lists nothing new beyond the files already modified in this
checkout. A tracked file is never affected by `.gitignore`, so the 73 already-tracked paths do not
move.

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_corpus_report.py: create

"""The corpus report is the only instrument in the parser plan.

Counting links resolved cannot tell 615 links unstacked onto 615 anchors from
615 links collapsed onto 40. Both make the same number rise. The tests below
pin the columns that can tell them apart: distinct anchors against the
fallback anchors the trainer already gets, links per anchor, truncation,
nesting by containment, controls the prose rule excludes from the index, and
the three wrong-anchor detectors.

Every path here is anchored to PROJECT_ROOT. A test that resolves a relative
path passes or fails on the directory pytest started in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tract.config import PROJECT_ROOT
from tract.corpus_report import (
    JOIN_CEILINGS,
    JOIN_FLOORS,
    build_corpus_report,
    check_join_floors,
)

TRACKED_CORPUS = PROJECT_ROOT / "data" / "processed" / "all_controls.json"

LONG = "A control statement long enough to clear every prose bar. " * 4


def _corpus(
    directory: Path,
    controls: list[dict[str, object]],
    name: str = "corpus",
) -> Path:
    """A corpus in the shape the real files use: a dict, not a list.

    Version 2's parity test assumed a list and silently indexed nothing. The
    fixtures use the real shape so the loader is exercised the way production
    exercises it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    payload = {
        "framework_count": 1,
        "frameworks": [
            {
                "framework_id": "demo",
                "framework_name": "Demo",
                "controls": controls,
            }
        ],
        "generated_date": "2026-01-01",
        "total_controls": len(controls),
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _corpus_as_list(
    directory: Path, controls: list[dict[str, object]], name: str = "legacy",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(
            [{
                "framework_id": "demo",
                "framework_name": "Demo",
                "controls": controls,
            }],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _links(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "links.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _row(section_id: str, section_name: str, cre_id: str = "1-1") -> dict[str, str]:
    return {
        "framework_id": "demo",
        "standard_name": "Demo",
        "section_id": section_id,
        "section_name": section_name,
        "cre_id": cre_id,
        "link_type": "LinkedTo",
    }


class TestCorpusShape:
    """The fact that made version 2's parity test assert nothing."""

    def test_the_tracked_corpus_is_a_dict(self) -> None:
        data = json.loads(TRACKED_CORPUS.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert sorted(data) == [
            "framework_count", "frameworks", "generated_date", "total_controls",
        ]

    def test_the_loader_reads_both_shapes(self, tmp_path: Path) -> None:
        controls = [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ]
        links = _links(tmp_path, [_row("C-1", "One")])
        as_dict = build_corpus_report(links, _corpus(tmp_path / "d", controls))
        as_list = build_corpus_report(
            links, _corpus_as_list(tmp_path / "l", controls),
        )
        assert as_dict.per_framework[0].by_title == 1
        assert as_list.per_framework[0].by_title == 1
        assert as_dict.corpus_framework_count == 1

    def test_a_corpus_with_no_records_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"total_controls": 0}), encoding="utf-8")
        with pytest.raises(ValueError, match="no list of framework records"):
            build_corpus_report(_links(tmp_path, [_row("C-1", "One")]), path)


class TestAnchorCollapse:
    def test_distinct_anchors_separates_collapse_from_coverage(
        self, tmp_path: Path,
    ) -> None:
        """Two corpora resolve every link. Only one of them is good."""
        rows = [_row(f"C-{n}", f"Control {n}") for n in range(1, 5)]
        spread = _corpus(tmp_path / "a", [
            {"control_id": f"C-{n}", "title": f"Control {n}",
             "description": f"{LONG} Variant {n}."}
            for n in range(1, 5)
        ])
        collapsed = _corpus(tmp_path / "b", [
            {"control_id": f"C-{n}", "title": f"Control {n}",
             "description": LONG}
            for n in range(1, 5)
        ])

        links = _links(tmp_path, rows)
        good = build_corpus_report(links, spread).per_framework[0]
        bad = build_corpus_report(links, collapsed).per_framework[0]

        assert good.links == bad.links == 4
        assert good.unresolved == bad.unresolved == 0
        assert good.distinct_anchors == 4
        assert bad.distinct_anchors == 1
        assert bad.links_per_anchor == pytest.approx(4.0)

    def test_nesting_is_containment_not_a_prefix(self, tmp_path: Path) -> None:
        """ETSI clause 5.2 rolled up over 5.2.2 does not start the parent's text.

        Version 2 counted strict prefixes, so this case read 0. The child here
        sits in the middle of the parent, which is what a clause rollup looks
        like when the parent opens with its own lead paragraph.
        """
        child = LONG + " The child clause statement."
        parent = "Lead-in paragraph for the parent clause. " + child + " Tail."
        corpus = _corpus(tmp_path, [
            {"control_id": "5.2.2", "title": "Child", "description": child},
            {"control_id": "5.2", "title": "Parent", "description": parent},
        ])
        links = _links(tmp_path, [_row("5.2.2", "Child"), _row("5.2", "Parent")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_anchors == 2
        assert row.nested_anchors == 1
        assert row.contained_anchors == 0

    def test_contained_keeps_the_strict_prefix_count(self, tmp_path: Path) -> None:
        """A domain aggregate that opens with its own first member."""
        member = LONG + " Member statement."
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Member", "description": member},
            {"control_id": "D-1", "title": "Domain",
             "description": member + " And the rest of the domain."},
        ])
        links = _links(tmp_path, [_row("C-1", "Member"), _row("D-1", "Domain")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_anchors == 2
        assert row.contained_anchors == 1
        assert row.nested_anchors == 1

    def test_truncation_can_merge_two_anchors_into_one(
        self, tmp_path: Path,
    ) -> None:
        """Two anchors sharing a long prefix collapse after MAX_ANCHOR_CHARS.

        distinct_anchors falls with no other column moving, which is how the
        collapse hid. distinct_anchors_pre_truncation is the witness.
        """
        from tract.config import MAX_ANCHOR_CHARS

        shared = "Shared opening text. " * (MAX_ANCHOR_CHARS // 21 + 2)
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": shared + " Tail A."},
            {"control_id": "C-2", "title": "Two", "description": shared + " Tail B."},
        ])
        links = _links(tmp_path, [_row("C-1", "One"), _row("C-2", "Two")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 2
        assert row.truncated == 2
        assert row.distinct_anchors == 1
        assert row.distinct_anchors_pre_truncation == 2


class TestFallbackAnchors:
    def test_unresolved_links_still_give_the_trainer_an_anchor(
        self, tmp_path: Path,
    ) -> None:
        """The column that turns +452 into +152.

        select_control_text falls back to section_name, so a framework with
        zero resolved links does not train on zero anchors.
        """
        corpus = _corpus(tmp_path, [])
        links = _links(tmp_path, [
            _row("C-1", "Access control policy"),
            _row("C-2", "Access control policy"),
            _row("C-3", "Cryptographic key management"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.unresolved == 3
        assert row.distinct_anchors == 0
        assert row.fallback_anchors == 2

    def test_a_link_with_no_name_falls_back_to_its_id(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [])
        links = _links(tmp_path, [_row("C-1", "")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.fallback_anchors == 1


class TestProseRuleExclusion:
    def test_control_whose_description_restates_its_title_is_counted(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Access control",
             "description": "Access control."},
        ])
        links = _links(tmp_path, [_row("C-1", "Access control")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.dropped_by_prose_rule == 1
        assert row.unresolved == 1
        assert row.distinct_anchors == 0
        assert row.fallback_anchors == 1

    def test_the_total_counts_frameworks_with_no_curated_links(
        self, tmp_path: Path,
    ) -> None:
        """522 was the link-bearing subset. The corpus holds 558."""
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps({
            "framework_count": 2,
            "frameworks": [
                {"framework_id": "demo", "framework_name": "Demo",
                 "controls": [{"control_id": "C-1", "title": "A",
                               "description": "A."}]},
                {"framework_id": "silent", "framework_name": "Silent",
                 "controls": [{"control_id": "S-1", "title": "B",
                               "description": "B."}]},
            ],
            "generated_date": "2026-01-01",
            "total_controls": 2,
        }, sort_keys=True), encoding="utf-8")
        report = build_corpus_report(_links(tmp_path, [_row("C-1", "A")]), path)
        assert report.by_id("demo").dropped_by_prose_rule == 1
        assert report.totals.dropped_by_prose_rule == 2


class TestAnchorSource:
    def test_the_four_sources_partition_the_resolved_links(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "Full", "description": "short",
             "full_text": LONG + " From full text."},
            {"control_id": "C-2", "title": "Described", "description": LONG},
            {"control_id": "C-3", "title": "Title restated as full text",
             "description": "short",
             "full_text": "Title restated as full text"},
            {"control_id": "C-4", "title": "Built", "description": LONG + " Built.",
             "metadata": {"text_origin": "synthetic"}},
        ])
        links = _links(tmp_path, [
            _row("C-1", "Full"), _row("C-2", "Described"),
            _row("C-3", "Title restated as full text"), _row("C-4", "Built"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.anchor_source_full_text == 1
        assert row.anchor_source_description == 1
        assert row.anchor_source_title == 1
        assert row.anchor_source_synthetic == 1
        assert (
            row.anchor_source_full_text
            + row.anchor_source_description
            + row.anchor_source_title
            + row.anchor_source_synthetic
        ) == row.by_title + row.by_id


class TestHubSide:
    def test_hub_concentration_is_reported(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": f"C-{n}", "title": f"T{n}",
             "description": f"{LONG} {n}."} for n in range(1, 5)
        ])
        links = _links(tmp_path, [
            _row("C-1", "T1", "hub-a"), _row("C-2", "T2", "hub-a"),
            _row("C-3", "T3", "hub-a"), _row("C-4", "T4", "hub-b"),
        ])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.distinct_hubs == 2
        assert row.links_per_hub == pytest.approx(2.0)


class TestWrongAnchorRisk:
    """Three detectors. Two of them reach the id branch, which is the fix."""

    def test_detector_a_title_hit_that_disagrees_with_the_id(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
            {"control_id": "3.2.2", "title": "Generative poisoning",
             "description": LONG + " Generative.",
             "metadata": {"alt_titles": ["Poisoning attacks"]}},
        ])
        links = _links(tmp_path, [_row("3.2.2", "Poisoning attacks")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 1
        assert row.wrong_anchor_risk == 1

    def test_detector_a_does_not_fire_when_the_channels_agree(
        self, tmp_path: Path,
    ) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "2.3", "title": "Poisoning attacks",
             "description": LONG + " Predictive."},
        ])
        links = _links(tmp_path, [_row("2.3", "Poisoning attacks")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_title == 1
        assert row.wrong_anchor_risk == 0

    def test_detector_b_id_hit_whose_control_does_not_carry_the_name(
        self, tmp_path: Path,
    ) -> None:
        """The id branch, where version 2 was blind."""
        corpus = _corpus(tmp_path, [
            {"control_id": "IPY", "title": "Interoperability and portability",
             "description": LONG + " Domain."},
        ])
        links = _links(tmp_path, [_row("IPY", "Data centre power redundancy")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_id == 1
        assert row.wrong_anchor_risk == 1

    def test_detector_b_does_not_fire_on_an_identifier_shaped_name(
        self, tmp_path: Path,
    ) -> None:
        """wstg and owasp_proactive_controls have section_name == section_id.

        Those links make no independent claim, so nothing is checked and
        nothing is flagged. This is why their attainable range is zero, and it
        is asserted rather than assumed.
        """
        corpus = _corpus(tmp_path, [
            {"control_id": "WSTG-INFO-01", "title": "Information gathering",
             "description": LONG + " Gathering."},
        ])
        links = _links(tmp_path, [_row("WSTG-INFO-01", "WSTG-INFO-01")])
        report = build_corpus_report(links, corpus)
        assert report.per_framework[0].by_id == 1
        assert report.per_framework[0].wrong_anchor_risk == 0
        assert all(
            not entry.wrong_anchor_checked for entry in report.resolution_rows
        )

    def test_detector_c_a_parent_id_and_a_child_id_reach_one_paragraph(
        self, tmp_path: Path,
    ) -> None:
        """The NIST AI 100-2 failure that put title first in the lookup order.

        Both controls exist and both carry the same paragraph, which is what a
        parser produces when a subsection's text is copied to each of the three
        mitigations it contains. Section names equal section ids here, so
        detector B cannot apply and the flag is C's alone.
        """
        shared = LONG + " Mitigations subsection."
        corpus = _corpus(tmp_path, [
            {"control_id": "3.3.2", "title": "Mitigations",
             "description": shared},
            {"control_id": "3.3.2.1", "title": "Adversarial training",
             "description": shared},
        ])
        links = _links(tmp_path, [_row("3.3.2", "3.3.2"), _row("3.3.2.1", "3.3.2.1")])
        row = build_corpus_report(links, corpus).per_framework[0]
        assert row.by_id == 2
        assert row.distinct_anchors == 1
        assert row.wrong_anchor_risk == 2

    def test_the_applicable_denominator_is_reported(self, tmp_path: Path) -> None:
        """A zero over a zero denominator proves nothing, so it is countable."""
        from tract.corpus_report import wrong_anchor_applicable

        corpus = _corpus(tmp_path, [
            {"control_id": "WSTG-INFO-01", "title": "Information gathering",
             "description": LONG + " Gathering."},
        ])
        links = _links(tmp_path, [_row("WSTG-INFO-01", "WSTG-INFO-01")])
        assert wrong_anchor_applicable(build_corpus_report(links, corpus)) == {
            "demo": 0,
        }


class TestFloors:
    def test_a_framework_below_its_floor_is_reported(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One"), _row("C-2", "Two")])
        report = build_corpus_report(links, corpus)
        assert check_join_floors(report, {"demo": 0.50}) == []
        assert len(check_join_floors(report, {"demo": 0.90})) == 1

    def test_a_floor_for_an_absent_framework_raises(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])
        report = build_corpus_report(links, corpus)
        with pytest.raises(KeyError, match="no curated links"):
            check_join_floors(report, {"absent": 0.50})

    def test_the_restricted_group_is_dropped_by_name_not_by_deletion(
        self, tmp_path: Path,
    ) -> None:
        """Rule 7. CI has 29 frameworks, the overlay has 31."""
        from tract.corpus_report import floors_for_report

        corpus = _corpus(tmp_path, [
            {"control_id": "C-1", "title": "One", "description": LONG},
        ])
        links = _links(tmp_path, [_row("C-1", "One")])
        report = build_corpus_report(links, corpus)
        applicable, skipped = floors_for_report(
            report, {"demo": 0.50, "etsi": 1.00}, frozenset({"etsi"}),
        )
        assert applicable == {"demo": 0.50}
        assert skipped == ["etsi"]


class TestDerivedFloors:
    """The criterion is committed before the run it gates, and it is checkable."""

    def test_every_floor_has_a_ceiling(self) -> None:
        assert sorted(JOIN_FLOORS) == sorted(JOIN_CEILINGS)

    def test_no_floor_exceeds_its_arithmetic_ceiling(self) -> None:
        """Not `floor <= 1.0`, which is true of every literal in the dict.

        The previous plan carried three impossible floors: dsomm 1.00 against a
        maximum of 0.9953, wstg 0.96 against 0.9322, enisa 0.80 against 0.721.
        """
        for framework_id, floor in JOIN_FLOORS.items():
            ceiling = JOIN_CEILINGS[framework_id]
            assert 0.0 < floor <= ceiling, framework_id

    def test_each_floor_is_its_ceiling_rounded_down(self) -> None:
        for framework_id, ceiling in JOIN_CEILINGS.items():
            expected = math.floor(round(ceiling * 100, 6)) / 100
            assert JOIN_FLOORS[framework_id] == pytest.approx(expected), (
                framework_id
            )

    def test_the_floors_cover_the_eleven_pending_frameworks(self) -> None:
        assert {
            "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
            "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021",
            "samm", "wstg",
        } <= set(JOIN_FLOORS)
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_corpus_report.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'tract.corpus_report'`.

- [ ] **Step 5: Add the two accessors the report needs**

`ProseIndex` exposes only `lookup`, and the report has to say which channel answered. Reaching into
`index._by_title` from another module would couple the report to a private attribute.

```python
# tract/text_selection.py: add to ProseIndex, immediately above lookup()

    def by_title(self, framework: str, section_name: str) -> TextSelection | None:
        """The selection a title lookup would return, or None.

        Exposed for tract.corpus_report, which must report which channel
        answered a link and cannot get that from lookup's return value.
        """
        return self._by_title.get(
            (canonical_framework(framework), section_name.strip().lower())
        )

    def by_id(self, framework: str, section_id: str) -> TextSelection | None:
        """The selection an id lookup would return, or None."""
        return self._by_id.get(
            (canonical_framework(framework), normalize_section_id(section_id))
        )
```

- [ ] **Step 6: Write the instrument**

```python
# tract/corpus_report.py: create

"""The corpus join report: the one instrument the parser plan is gated on.

A count of links resolved cannot distinguish 615 links unstacked onto 615
distinct anchors from 615 links collapsed onto 40 coarse ones. Both make the
same number rise, and the second is a regression dressed as progress. So this
module reports the anchor side as well as the link side, and reports both
through the same lookup order the training and evaluation paths use, rather
than through a set intersection that would accept a join the consumer cannot
perform.

Two anchor columns exist because the first version of this instrument reported
a three-times-overstated gain. `distinct_anchors` counts the anchors a resolved
link reaches. `fallback_anchors` counts the distinct section names the trainer
already gets for links that resolve to nothing, because `select_control_text`
falls back to `section_name` rather than failing. A framework with 734
unresolved links is not a framework with zero anchors, and reporting it that
way turned a +152 gain into a +452 headline.

Columns, and the failure each one answers:

    by_title / by_id / unresolved   which channel carried the join
    fallback_anchors                what the trainer gets without a join
    distinct_anchors                the number every downstream metric rests on
    distinct_anchors_pre_truncation two anchors merging into one after the cut
    links_per_anchor                collapse, visible
    truncated                       anchors the encoder budget cuts
    nested_anchors                  an anchor contained in another anchor
    contained_anchors               the stricter prefix-only form, for continuity
    dropped_by_prose_rule           controls ProseIndex never indexed, corpus-wide
    wrong_anchor_risk               three detectors, two of them id-side
    anchor_source_*                 what kind of text the anchor is
    distinct_hubs / links_per_hub   hub-side concentration, for a later
                                    agreement study
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from tract.config import (
    MAX_ANCHOR_CHARS,
    PROJECT_ROOT,
    RESTRICTED_FRAMEWORK_IDS,
    TRAINING_DIR,
)
from tract.io import atomic_write_text
from tract.text_selection import (
    ProseIndex,
    TextSelection,
    _is_prose,
    canonical_framework,
    merged_corpus_path,
    normalize_section_id,
    prepare_anchor,
    strip_markup,
)

logger = logging.getLogger(__name__)

CURATED_LINKS_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"

# Evidence, not results. Tracked by design through the .gitignore negations
# this task adds, and anchored to PROJECT_ROOT so a reader never depends on the
# directory pytest happened to start in.
CORPUS_EVIDENCE_DIR: Final[Path] = PROJECT_ROOT / "results" / "corpus"

# The tracked corpus carries 29 frameworks, the licensed overlay 31. [measured]
# A report built from fewer than the full set cannot assert the restricted
# rows, and gating on file existence never skips because the tracked file
# always exists. Task 15 owns updating this if the rebuild changes the census.
FULL_CORPUS_FRAMEWORK_COUNT: Final[int] = 31

# A parser that assembles an anchor out of several source fragments marks it,
# so the report can separate parser-written text from publisher-written text.
# Absent means the publisher wrote it.
TEXT_ORIGIN_METADATA_KEY: Final[str] = "text_origin"
SYNTHETIC_TEXT_ORIGIN: Final[str] = "synthetic"

_WHITESPACE = re.compile(r"\s+")

# Separators that make one identifier a parent of another: "5.2" of "5.2.2",
# "WSTG-INFO" of "WSTG-INFO-01", "IPY" of "IPY-01".
_ID_SEPARATORS: Final[tuple[str, ...]] = (".", "-", "_", ":", " ")


def _fold(text: str | None) -> str:
    """Whitespace-collapsed, case-folded form, for comparing a name to a title."""
    return _WHITESPACE.sub(" ", (text or "").strip()).casefold()


@dataclass(frozen=True)
class ControlFacts:
    """What the report needs about a control that TextSelection does not carry."""

    title: str
    origin: str


@dataclass
class LinkResolution:
    """One curated link, resolved, carrying no anchor text.

    Digest and length only. The file this serialises to is tracked for every
    framework including the licence-restricted ones, so it must hold nothing a
    publisher reserves. `section_id` and `section_name` come from
    hub_links_curated.jsonl, which is already tracked.
    """

    framework_id: str
    section_id: str
    section_name: str
    cre_id: str
    link_type: str
    channel: str
    anchor_source: str
    anchor_sha256: str
    anchor_chars: int
    truncated: bool
    wrong_anchor: bool
    wrong_anchor_checked: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameworkJoin:
    """One framework's join, on both the link side and the anchor side."""

    framework_id: str
    standard_name: str
    links: int = 0
    by_title: int = 0
    by_id: int = 0
    unresolved: int = 0
    fallback_anchors: int = 0
    distinct_anchors: int = 0
    distinct_anchors_pre_truncation: int = 0
    links_per_anchor: float = 0.0
    truncated: int = 0
    nested_anchors: int = 0
    contained_anchors: int = 0
    dropped_by_prose_rule: int = 0
    wrong_anchor_risk: int = 0
    anchor_source_full_text: int = 0
    anchor_source_description: int = 0
    anchor_source_title: int = 0
    anchor_source_synthetic: int = 0
    distinct_hubs: int = 0
    links_per_hub: float = 0.0
    resolution_rate: float = 0.0

    def finalise(self) -> None:
        resolved = self.by_title + self.by_id
        self.resolution_rate = 0.0 if not self.links else resolved / self.links
        self.links_per_anchor = (
            0.0 if not self.distinct_anchors else resolved / self.distinct_anchors
        )
        # All links, not only the resolved ones: hub concentration is a
        # property of the curated link file and must not move when a parser
        # lands.
        self.links_per_hub = (
            0.0 if not self.distinct_hubs else self.links / self.distinct_hubs
        )


@dataclass
class CorpusReport:
    """Every framework's join plus the identity of what produced it."""

    per_framework: list[FrameworkJoin]
    totals: FrameworkJoin
    corpus_path: str
    corpus_sha256: str
    links_path: str
    links_sha256: str
    corpus_framework_count: int = 0
    max_anchor_chars: int = MAX_ANCHOR_CHARS
    # Serialised to the JSONL rather than into to_json(): 4,405 rows would
    # dominate the summary artifact a reader opens first.
    resolution_rows: list[LinkResolution] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "corpus_framework_count": self.corpus_framework_count,
            "links_path": self.links_path,
            "links_sha256": self.links_sha256,
            "max_anchor_chars": self.max_anchor_chars,
            "totals": asdict(self.totals),
            "per_framework": [asdict(row) for row in self.per_framework],
        }

    def by_id(self, framework_id: str) -> FrameworkJoin:
        """One framework's row.

        Raises:
            KeyError: If the framework contributed no curated links.
        """
        for row in self.per_framework:
            if row.framework_id == framework_id:
                return row
        raise KeyError(
            f"{framework_id!r} has no curated links, so it has no join to "
            f"report. Check the framework_id spelling against "
            f"data/training/hub_links_curated.jsonl."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_links(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped.setdefault(row["framework_id"], []).append(row)
    return grouped


def _load_records(path: Path) -> list[dict[str, Any]]:
    """The framework records, from either corpus shape.

    Both real corpus files are mappings keyed
    [framework_count, frameworks, generated_date, total_controls]. [measured]
    Preferring the named key rather than "the first list value" means a new
    top-level list cannot silently take over.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = data.get("frameworks")
        if isinstance(records, list):
            return records
        for value in data.values():
            if isinstance(value, list):
                return value
    raise ValueError(
        f"{path} holds no list of framework records. The merged corpus is "
        f"either a list or a mapping carrying one under 'frameworks'."
    )


def _control_facts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], ControlFacts], dict[str, int]]:
    """Title and text origin per indexed anchor, plus the prose-rule census.

    Keyed on the selection text rather than on the control id, because
    TextSelection carries no back-reference and duplicating ProseIndex's key
    logic here would be a second implementation to keep in step. Two controls
    with byte-identical text collapse to one entry, and the first in corpus
    order wins, and their anchors are indistinguishable to the encoder anyway.

    The census counts every framework in the corpus, including those with no
    curated links. Summing only the link-bearing subset read 522 where the
    corpus holds 558.
    """
    facts: dict[tuple[str, str], ControlFacts] = {}
    dropped: dict[str, int] = {}
    for record in records:
        framework = canonical_framework(str(record.get("framework_name") or ""))
        for control in record.get("controls") or []:
            title = str(control.get("title") or "")
            description = str(control.get("description") or "")
            full_text = str(control.get("full_text") or "")
            if full_text.strip():
                text = full_text.strip()
            elif _is_prose(description, title):
                text = description.strip()
            else:
                dropped[framework] = dropped.get(framework, 0) + 1
                continue
            metadata = control.get("metadata") or {}
            facts.setdefault(
                (framework, text),
                ControlFacts(
                    title=title,
                    origin=str(
                        metadata.get(TEXT_ORIGIN_METADATA_KEY) or "source"
                    ),
                ),
            )
    return facts, dropped


def _lookup_with_channel(
    index: ProseIndex,
    canonical: str,
    section_id: str | None,
    section_name: str | None,
) -> tuple[TextSelection | None, str]:
    """ProseIndex.lookup, plus which channel answered.

    Deliberately reimplements lookup's branch order rather than calling it: the
    report has to say *how* a link resolved, and lookup returns only the text.
    The order here must stay identical to lookup's, title then id, and
    tests/test_corpus_report.py::TestChannelParity asserts the two agree on
    every curated link in the real corpus.
    """
    if section_name:
        hit = index.by_title(canonical, str(section_name))
        if hit is not None:
            return hit, "title"
    normalized = normalize_section_id(section_id)
    if normalized:
        hit = index.by_id(canonical, normalized)
        if hit is not None:
            return hit, "id"
    return None, "unresolved"


def _fallback_anchor(section_id: str | None, section_name: str | None) -> str:
    """What select_control_text hands the trainer when the index misses.

    Same expression and same normalisation as the fallback branch of
    select_control_text, so the count here is the count the trainer sees.
    """
    fallback = (section_name or section_id or "").strip()
    if not fallback:
        return ""
    text, _ = prepare_anchor(strip_markup(fallback))
    return text


def _classify_anchor(
    selection: TextSelection, anchor: str, facts: ControlFacts | None,
) -> str:
    """Which of the four kinds of text this anchor is.

    Parser-assembled text wins over everything, because its provenance is the
    parser rather than the publisher. A stored anchor that only restates the
    control's own title is reported as `title` even though it arrived through
    full_text or description, since that is what the encoder reads and the
    prose rule cannot see it once a parser writes full_text.
    """
    if facts is not None and facts.origin == SYNTHETIC_TEXT_ORIGIN:
        return "synthetic"
    if facts is not None and facts.title and _fold(anchor) == _fold(facts.title):
        return "title"
    return selection.source


def _count_nested(anchors: set[str]) -> int:
    """Anchors contained anywhere inside a longer anchor of the same framework.

    Containment, not a strict prefix. An ETSI clause 5.2 that rolls up 5.2.2
    opens with its own lead paragraph, so the child sits in the middle of the
    parent and a prefix test reads 0. Measured on the current corpus this
    column reads 0 for every framework. [measured]
    """
    ordered = sorted(anchors, key=lambda item: (len(item), item))
    return sum(
        1
        for position, short in enumerate(ordered)
        if any(short in longer for longer in ordered[position + 1:])
    )


def _count_contained(anchors: set[str]) -> int:
    """The strict-prefix count the first version of this module reported.

    Kept so a reader comparing two runs across this change can separate the
    definition change from a corpus change.
    """
    ordered = sorted(anchors, key=lambda item: (len(item), item))
    return sum(
        1
        for position, short in enumerate(ordered)
        if any(longer.startswith(short) for longer in ordered[position + 1:])
    )


def _is_ancestor_id(parent: str, child: str) -> bool:
    """Whether one normalised section id is a parent of another."""
    if not parent or not child or len(child) <= len(parent):
        return False
    if not child.startswith(parent):
        return False
    return child[len(parent)] in _ID_SEPARATORS


@dataclass
class _Resolved:
    """One link that reached an anchor, with everything a detector needs."""

    link: Mapping[str, Any]
    normalized_id: str
    channel: str
    anchor: str
    raw_text: str
    truncated: bool
    anchor_source: str
    control_title: str


def _wrong_anchor(
    index: ProseIndex,
    canonical: str,
    entry: _Resolved,
    by_id_anchor: Mapping[str, str],
) -> tuple[bool, bool]:
    """Whether this link's anchor is suspect, and whether anything checked it.

    Three detectors, because the first version had one and it lived entirely
    inside the title branch. Nine of the eleven frameworks this plan adds are
    engineered to resolve through the id channel, so a title-only detector made
    `wrong_anchor_risk == 0` unfailable for them.

    A: the title channel answered and the id channel would have answered
       differently. The curator wrote both, and they disagree.
    B: the id channel answered, the link also carried a name that says
       something the id does not, and the control the id reached does not carry
       that name anywhere in its title.
    C: a coarser id and a finer id in the same framework reached the same
       paragraph. This is the NIST AI 100-2 failure that put title first in the
       lookup order, and neither A nor B can see it.

    The second return value is the denominator. A framework whose links carry
    `section_name == section_id` and no ancestor relations has zero applicable
    checks, so a zero in this column proves nothing about it, and
    wrong_anchor_applicable() makes that legible instead of leaving it implied.
    """
    name = str(entry.link.get("section_name") or "")
    checked = False
    flagged = False

    if entry.channel == "title":
        if entry.normalized_id:
            checked = True
            other = index.by_id(canonical, entry.normalized_id)
            if other is not None and prepare_anchor(other.text)[0] != entry.anchor:
                flagged = True
        return flagged, checked

    if name and _fold(name) != _fold(entry.normalized_id):
        checked = True
        title = _fold(entry.control_title)
        if title and _fold(name) not in title and title not in _fold(name):
            flagged = True

    for other_id, other_anchor in by_id_anchor.items():
        if other_id == entry.normalized_id:
            continue
        if _is_ancestor_id(other_id, entry.normalized_id) or _is_ancestor_id(
            entry.normalized_id, other_id
        ):
            checked = True
            if other_anchor == entry.anchor:
                flagged = True
    return flagged, checked


def build_corpus_report(
    links_path: Path | None = None, corpus_path: Path | None = None,
) -> CorpusReport:
    """Resolve every curated link through ProseIndex and report the join."""
    links_file = links_path or CURATED_LINKS_PATH
    corpus_file = corpus_path or merged_corpus_path()

    records = _load_records(corpus_file)
    index = ProseIndex(records)
    facts, dropped = _control_facts(records)
    grouped = _load_links(links_file)

    rows: list[FrameworkJoin] = []
    resolution_rows: list[LinkResolution] = []
    totals = FrameworkJoin(framework_id="TOTAL", standard_name="")
    all_anchors: set[str] = set()
    all_pre_truncation: set[str] = set()
    all_fallbacks: set[str] = set()
    all_hubs: set[str] = set()

    for framework_id in sorted(grouped):
        links = grouped[framework_id]
        standard = str(links[0].get("standard_name") or "")
        canonical = canonical_framework(standard)
        row = FrameworkJoin(
            framework_id=framework_id,
            standard_name=standard,
            links=len(links),
            dropped_by_prose_rule=dropped.get(canonical, 0),
        )

        anchors: set[str] = set()
        pre_truncation: set[str] = set()
        fallbacks: set[str] = set()
        hubs: set[str] = set()
        resolved: list[_Resolved] = []
        unresolved_rows: list[tuple[Mapping[str, Any], str]] = []

        for link in links:
            hubs.add(str(link.get("cre_id") or ""))
            selection, channel = _lookup_with_channel(
                index, canonical, link.get("section_id"), link.get("section_name"),
            )
            normalized = normalize_section_id(link.get("section_id"))
            if selection is None:
                row.unresolved += 1
                fallback = _fallback_anchor(
                    link.get("section_id"), link.get("section_name"),
                )
                if fallback:
                    fallbacks.add(fallback)
                unresolved_rows.append((link, fallback))
                continue

            anchor, was_cut = prepare_anchor(selection.text)
            control = facts.get((canonical, selection.text))
            resolved.append(
                _Resolved(
                    link=link,
                    normalized_id=normalized,
                    channel=channel,
                    anchor=anchor,
                    raw_text=selection.text,
                    truncated=was_cut,
                    anchor_source=_classify_anchor(selection, anchor, control),
                    control_title=control.title if control is not None else "",
                )
            )
            anchors.add(anchor)
            pre_truncation.add(selection.text)
            row.truncated += int(was_cut)
            if channel == "title":
                row.by_title += 1
            else:
                row.by_id += 1

        by_id_anchor: dict[str, str] = {}
        for entry in resolved:
            if entry.channel == "id" and entry.normalized_id:
                by_id_anchor.setdefault(entry.normalized_id, entry.anchor)

        for entry in resolved:
            flagged, checked = _wrong_anchor(index, canonical, entry, by_id_anchor)
            row.wrong_anchor_risk += int(flagged)
            if entry.anchor_source == "full_text":
                row.anchor_source_full_text += 1
            elif entry.anchor_source == "description":
                row.anchor_source_description += 1
            elif entry.anchor_source == SYNTHETIC_TEXT_ORIGIN:
                row.anchor_source_synthetic += 1
            else:
                row.anchor_source_title += 1
            resolution_rows.append(
                LinkResolution(
                    framework_id=framework_id,
                    section_id=str(entry.link.get("section_id") or ""),
                    section_name=str(entry.link.get("section_name") or ""),
                    cre_id=str(entry.link.get("cre_id") or ""),
                    link_type=str(entry.link.get("link_type") or ""),
                    channel=entry.channel,
                    anchor_source=entry.anchor_source,
                    anchor_sha256=hashlib.sha256(
                        entry.anchor.encode("utf-8")
                    ).hexdigest(),
                    anchor_chars=len(entry.anchor),
                    truncated=entry.truncated,
                    wrong_anchor=flagged,
                    wrong_anchor_checked=checked,
                )
            )

        # Unresolved links carry the fallback anchor the trainer receives, so
        # the BEFORE file holds the text-quality baseline the AFTER is read
        # against. Without these rows the JSONL would describe only the links
        # that already work.
        for missed, fallback in unresolved_rows:
            resolution_rows.append(
                LinkResolution(
                    framework_id=framework_id,
                    section_id=str(missed.get("section_id") or ""),
                    section_name=str(missed.get("section_name") or ""),
                    cre_id=str(missed.get("cre_id") or ""),
                    link_type=str(missed.get("link_type") or ""),
                    channel="unresolved",
                    anchor_source="title",
                    anchor_sha256=hashlib.sha256(
                        fallback.encode("utf-8")
                    ).hexdigest(),
                    anchor_chars=len(fallback),
                    truncated=False,
                    wrong_anchor=False,
                    wrong_anchor_checked=False,
                )
            )

        row.distinct_anchors = len(anchors)
        row.distinct_anchors_pre_truncation = len(pre_truncation)
        row.fallback_anchors = len(fallbacks)
        row.distinct_hubs = len(hubs)
        row.nested_anchors = _count_nested(anchors)
        row.contained_anchors = _count_contained(anchors)
        row.finalise()
        rows.append(row)

        all_anchors |= anchors
        all_pre_truncation |= pre_truncation
        all_fallbacks |= fallbacks
        all_hubs |= hubs
        totals.links += row.links
        totals.by_title += row.by_title
        totals.by_id += row.by_id
        totals.unresolved += row.unresolved
        totals.truncated += row.truncated
        totals.nested_anchors += row.nested_anchors
        totals.contained_anchors += row.contained_anchors
        totals.wrong_anchor_risk += row.wrong_anchor_risk
        totals.anchor_source_full_text += row.anchor_source_full_text
        totals.anchor_source_description += row.anchor_source_description
        totals.anchor_source_title += row.anchor_source_title
        totals.anchor_source_synthetic += row.anchor_source_synthetic

    # The census covers every framework in the corpus, including the ones with
    # no curated links and therefore no row above.
    totals.dropped_by_prose_rule = sum(dropped.values())
    totals.distinct_anchors = len(all_anchors)
    totals.distinct_anchors_pre_truncation = len(all_pre_truncation)
    totals.fallback_anchors = len(all_fallbacks)
    totals.distinct_hubs = len(all_hubs)
    totals.finalise()

    logger.info(
        "Corpus join: %d links, %d resolved, %d distinct anchors, %d fallback "
        "anchors, %d controls outside the prose index, over %d frameworks",
        totals.links, totals.by_title + totals.by_id, totals.distinct_anchors,
        totals.fallback_anchors, totals.dropped_by_prose_rule, len(records),
    )

    return CorpusReport(
        per_framework=rows,
        totals=totals,
        corpus_path=str(corpus_file),
        corpus_sha256=_sha256(corpus_file),
        corpus_framework_count=len(records),
        links_path=str(links_file),
        links_sha256=_sha256(links_file),
        resolution_rows=resolution_rows,
    )


def write_link_resolution(report: CorpusReport, path: Path) -> None:
    """One row per curated link, digests only, safe to track for any framework.

    This is what a later label-agreement study needs to sample the frameworks
    this plan re-weights: which channel carried each link, what kind of text it
    reached, how long that text was, and whether a detector questioned it. The
    premortem's answer to "does the plan create that artifact" was no.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        "".join(
            json.dumps(entry.to_json(), sort_keys=True) + "\n"
            for entry in report.resolution_rows
        ),
        path,
    )
    logger.info("wrote %d link resolutions to %s", len(report.resolution_rows), path)


def wrong_anchor_applicable(report: CorpusReport) -> dict[str, int]:
    """Per framework, how many links a wrong-anchor detector could fire on.

    `wrong_anchor_risk == 0` over a denominator of 0 is a fact about the link
    file, not about the parser. Reporting the denominator is what stops that
    zero from being read as a pass.
    """
    counts: dict[str, int] = {row.framework_id: 0 for row in report.per_framework}
    for entry in report.resolution_rows:
        if entry.wrong_anchor_checked:
            counts[entry.framework_id] = counts.get(entry.framework_id, 0) + 1
    return counts


def check_join_floors(
    report: CorpusReport, floors: Mapping[str, float],
) -> list[str]:
    """One message per framework whose resolution rate is under its floor.

    A floor is derived from the link file and the source before the parser is
    written, never pasted from the run being gated. See JOIN_CEILINGS for the
    arithmetic that produced each one.

    Raises:
        KeyError: If a floor names a framework with no curated links. Silently
            skipping it would retire the gate.
    """
    failures: list[str] = []
    for framework_id, floor in sorted(floors.items()):
        row = report.by_id(framework_id)
        if row.resolution_rate + 1e-9 < floor:
            failures.append(
                f"{framework_id}: resolved {row.by_title + row.by_id} of "
                f"{row.links} links ({row.resolution_rate:.4f}) against a "
                f"derived floor of {floor:.4f}. The floor is the arithmetic "
                f"ceiling of this framework's link data rounded down, so a "
                f"miss means the parser lost anchors the source supplies."
            )
    return failures


def floors_for_report(
    report: CorpusReport,
    floors: Mapping[str, float],
    restricted: frozenset[str] = RESTRICTED_FRAMEWORK_IDS,
) -> tuple[dict[str, float], list[str]]:
    """The floors this corpus can carry, and the named group it cannot.

    CI has no licensed overlay, so its corpus holds 29 frameworks against the
    overlay's 31, and every restricted row would read 0.0000 and hard-fail.
    Gating on file existence never skips, because the tracked corpus always
    exists. This gates on content instead, and returns the skipped group by
    name so the reason is stated rather than implied. Never delete a floor to
    make CI green: that retires the only gate on a parser nobody can inspect.

    The Rule 3 author widens the default to OVERLAY_FRAMEWORK_IDS when text
    routing moves the conditional frameworks into the overlay.
    """
    if report.corpus_framework_count >= FULL_CORPUS_FRAMEWORK_COUNT:
        return dict(floors), []
    applicable = {k: v for k, v in floors.items() if k not in restricted}
    skipped = sorted(k for k in floors if k in restricted)
    if skipped:
        logger.warning(
            "corpus has %d frameworks against %d in the full set, so the "
            "licensed overlay is absent from this checkout and these floors "
            "cannot be asserted: %s",
            report.corpus_framework_count, FULL_CORPUS_FRAMEWORK_COUNT,
            ", ".join(skipped),
        )
    return applicable, skipped


def format_table(report: CorpusReport) -> str:
    """The report as a fixed-width table, for logs and for the run ledger."""
    header = (
        f"{'framework':26s} {'links':>5s} {'ttl':>5s} {'id':>4s} {'unres':>5s} "
        f"{'fb':>4s} {'anch':>5s} {'pre':>5s} {'l/a':>5s} {'trunc':>5s} "
        f"{'nest':>4s} {'cont':>4s} {'noidx':>5s} {'wrong':>5s} "
        f"{'ftxt':>5s} {'desc':>5s} {'titl':>4s} {'synt':>4s} "
        f"{'hubs':>5s} {'l/h':>5s} {'rate':>6s}"
    )
    lines = [header, "-" * len(header)]
    for row in [*report.per_framework, report.totals]:
        lines.append(
            f"{row.framework_id:26s} {row.links:5d} {row.by_title:5d} "
            f"{row.by_id:4d} {row.unresolved:5d} {row.fallback_anchors:4d} "
            f"{row.distinct_anchors:5d} {row.distinct_anchors_pre_truncation:5d} "
            f"{row.links_per_anchor:5.2f} {row.truncated:5d} "
            f"{row.nested_anchors:4d} {row.contained_anchors:4d} "
            f"{row.dropped_by_prose_rule:5d} {row.wrong_anchor_risk:5d} "
            f"{row.anchor_source_full_text:5d} {row.anchor_source_description:5d} "
            f"{row.anchor_source_title:4d} {row.anchor_source_synthetic:4d} "
            f"{row.distinct_hubs:5d} {row.links_per_hub:5.2f} "
            f"{row.resolution_rate:6.4f}"
        )
    return "\n".join(lines)
```

- [ ] **Step 7: Commit the floors, before any parser exists**

The criterion cannot move in the same commit as the result it gates. Version 2 put `JOIN_FLOORS` in
Task 16 alongside the report it gates, and the plan file itself is gitignored (`.gitignore:25`), so a
floor edited mid-run would leave no diff anywhere. Both dicts land here, in tracked code, in the same
commit as the instrument and before a line of parser code exists.

```python
# tract/corpus_report.py: append

# Per-framework join ceilings, each derived from the curated link file and the
# pinned source in that framework's own plan task, BEFORE its parser existed.
# Written as the fraction so a transcription error is visible. The eleven
# pending frameworks resolve 0 of 734 links today, so none of these was read
# off the run it gates.
#
#   dsomm        213/214  one activity's statement is 11 characters
#   wstg         109/118  nine links name ids absent from the archive
#   nist_800_63   78/79   one section_id is the fragment "are g"
#   biml          21/21   with the two declared alternates
#   enisa         68/68   with Table 3, Annex C and name repair
#   etsi          36/36   every technique declared to its own clause
#   csa_ccm       29/29   seven renamed ids resolve by title
#   nist_ssdf     46/46   with the two declared alt_ids
#   samm          30/30
#   owasp_top10_2021          17/17
#   owasp_proactive_controls  76/76
#
# The eleven below the fold already resolve today, and their ceilings are the
# rates measured on the BEFORE corpus. They are here as a regression gate: the
# rebuild in Task 15 must not cost them. Each miss is known and named.
#   cwe          612/613  CWE-937 was withdrawn upstream
#   nist_800_53  298/300  SC-23(1) and SC-23(3) were withdrawn
#   iso_27001     92/94   A.7.8 and A.7.9 are shorter than their own titles
#                         plus PROSE_MIN_EXTRA_CHARS, so ProseIndex excludes
#                         them on purpose
JOIN_CEILINGS: Final[Mapping[str, float]] = {
    "asvs": 277 / 277,
    "biml": 21 / 21,
    "capec": 1799 / 1799,
    "csa_ccm": 29 / 29,
    "cwe": 612 / 613,
    "dsomm": 213 / 214,
    "enisa": 68 / 68,
    "etsi": 36 / 36,
    "iso_27001": 92 / 94,
    "mitre_atlas": 65 / 65,
    "nist_800_53": 298 / 300,
    "nist_800_63": 78 / 79,
    "nist_ai_100_2": 45 / 45,
    "nist_ssdf": 46 / 46,
    "owasp_ai_exchange": 64 / 64,
    "owasp_cheat_sheets": 391 / 391,
    "owasp_llm_top10": 13 / 13,
    "owasp_ml_top10": 10 / 10,
    "owasp_proactive_controls": 76 / 76,
    "owasp_top10_2021": 17 / 17,
    "samm": 30 / 30,
    "wstg": 109 / 118,
}

# Each ceiling rounded down to two decimals, which
# tests/test_corpus_report.py::TestDerivedFloors asserts rather than trusts.
# A floor of 1.00 where the ceiling is 1.00 is the right number, not an
# oversight: below it, a link the source can answer stopped resolving.
JOIN_FLOORS: Final[Mapping[str, float]] = {
    "asvs": 1.00,
    "biml": 1.00,
    "capec": 1.00,
    "csa_ccm": 1.00,
    "cwe": 0.99,
    "dsomm": 0.99,
    "enisa": 1.00,
    "etsi": 1.00,
    "iso_27001": 0.97,
    "mitre_atlas": 1.00,
    "nist_800_53": 0.99,
    "nist_800_63": 0.98,
    "nist_ai_100_2": 1.00,
    "nist_ssdf": 1.00,
    "owasp_ai_exchange": 1.00,
    "owasp_cheat_sheets": 1.00,
    "owasp_llm_top10": 1.00,
    "owasp_ml_top10": 1.00,
    "owasp_proactive_controls": 1.00,
    "owasp_top10_2021": 1.00,
    "samm": 1.00,
    "wstg": 0.92,
}


# Pre-registered wrong-anchor counts, one entry per framework where the title
# channel can answer at all. Task 16 gates on these instead of on `== 0`.
#
# `== 0` is unfailable for nine of the eleven, because their links resolve
# entirely through the id channel and `wrong_anchor_risk` increments only inside
# the title branch. A gate whose maximum attainable value is zero certifies
# nothing. These two are the frameworks where the title channel is live and the
# count is genuinely non-zero, so the assertion can fail in both directions.
#
#   csa_ccm  1  link `IPY` carries `section_name` "Interoperability and
#               portability policy and procedures", which is control IPY-01's
#               title, not the IPY domain's name ("Interoperability &
#               Portability"). Title-first therefore answers with IPY-01.
#               Task 8 rules that IPY-01 is the correct target. [measured,
#               ML Engineer; link text confirmed by the orchestrator]
#   etsi     1  link `6.3.1` carries the name "Mitigating model stealing",
#               which resolves to clause 6.3. [measured, ML Engineer]
#
# A framework absent from this mapping must report zero, and Task 16 asserts
# `by_title == 0` for it rather than asserting an unfailable risk count.
JOIN_WRONG_ANCHOR_BUDGET: Final[Mapping[str, int]] = {
    "csa_ccm": 1,
    "etsi": 1,
}
```

- [ ] **Step 7a: Land the licence tiers before any parser writes a processed file**

Tasks 3 through 13 each branch on `OVERLAY_FRAMEWORK_IDS` to decide whether their processed file is
tracked. Task 3 is DSOMM, which is GPL-3.0-only, so the routing has to exist before the first parser
runs, not when the corpus is merged. Add to `tract/config.py`:

```python
# tract/config.py: add below RESTRICTED_FRAMEWORK_IDS

# Reproduction is permitted, but on terms a CC0 grant cannot carry. CC0 is not a
# disclaimer; it is an affirmative assertion that the publisher holds the rights
# and waives them, which is false for GPL-3.0 and for share-alike text. These
# frameworks' processed files route to the gitignored overlay exactly as the
# restricted ones do, and their ASSIGNMENTS stay tracked and published, because
# a mapping is a fact about two documents rather than a reproduction of either.
# Training reads the overlay, so this costs zero anchors. See rulings R4 to R6.
CONDITIONAL_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "dsomm",                     # GPL-3.0-only
    "biml",                      # CC-BY-SA-3.0 AND CC-BY-SA-4.0
    "samm",                      # CC-BY-SA-4.0
    "wstg",                      # CC-BY-SA-4.0
    "owasp_top10_2021",          # CC-BY-SA-4.0
    "owasp_proactive_controls",  # CC-BY-SA-4.0
    "csa_ccm",                   # all rights reserved, no redistribution
})

# What routes to the overlay. RESTRICTED_FRAMEWORK_IDS keeps its narrower
# meaning everywhere else: the fingerprint gate and the "must never appear in
# git in any form" rule.
OVERLAY_FRAMEWORK_IDS: Final[frozenset[str]] = (
    RESTRICTED_FRAMEWORK_IDS | CONDITIONAL_FRAMEWORK_IDS
)
```

Then the `.gitignore` lines that make the routing real, appended beside the two restricted entries:

```
data/processed/frameworks/dsomm.json
data/processed/frameworks/biml.json
data/processed/frameworks/samm.json
data/processed/frameworks/wstg.json
data/processed/frameworks/owasp_top10_2021.json
data/processed/frameworks/owasp_proactive_controls.json
data/processed/frameworks/csa_ccm.json
```

- [ ] **Step 7b: Prove the tiering holds in both directions**

```python
# tests/test_framework_licenses.py: create

from __future__ import annotations

from pathlib import Path

from tract.config import (
    CONDITIONAL_FRAMEWORK_IDS,
    FRAMEWORK_LICENSES,
    OVERLAY_FRAMEWORK_IDS,
    RESTRICTED_FRAMEWORK_IDS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_two_tiers_do_not_overlap() -> None:
    assert not (RESTRICTED_FRAMEWORK_IDS & CONDITIONAL_FRAMEWORK_IDS)
    assert OVERLAY_FRAMEWORK_IDS == RESTRICTED_FRAMEWORK_IDS | CONDITIONAL_FRAMEWORK_IDS


def test_every_copyleft_framework_is_conditional() -> None:
    """The tier is derived from the recorded licence, not from a hand list.

    The binary set this replaces modelled licence STATUS and not licence CLASS,
    so seven frameworks whose licences permit reproduction on conditions were
    treated as unconditionally publishable. Deriving the assertion from
    FRAMEWORK_LICENSES means a newly added copyleft source fails this test
    rather than silently joining the tracked corpus.
    """
    copyleft = {
        framework_id
        for framework_id, licence in FRAMEWORK_LICENSES.items()
        if "GPL" in licence or "CC-BY-SA" in licence
    }
    missing = copyleft - OVERLAY_FRAMEWORK_IDS
    assert not missing, (
        f"{sorted(missing)} carry a copyleft or share-alike licence and are not "
        f"routed to the overlay. A CC0 repository cannot carry their terms."
    )


def test_every_overlay_framework_has_a_gitignore_line() -> None:
    ignored = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    for framework_id in sorted(OVERLAY_FRAMEWORK_IDS):
        expected = f"data/processed/frameworks/{framework_id}.json"
        assert expected in ignored, (
            f"{framework_id} routes to the overlay but {expected} is not in "
            f".gitignore, so its text would be tracked."
        )
```

Run: `PYTHONPATH=. "$PY" -m pytest tests/test_framework_licenses.py -q`
Expected: 3 passed. `test_every_copyleft_framework_is_conditional` fails before
`CONDITIONAL_FRAMEWORK_IDS` exists, which is the point.

- [ ] **Step 8: Add the channel-parity test, with the loader that reads a dict**

The report reimplements `lookup`'s branch order. If the two ever disagree, the report describes a
join the pipeline does not perform, which is the defect that got the previous plan rejected. Version
2's test built `ProseIndex([])` and asserted `True == True` 4,405 times. There is no skip: parity is
a property of the code, both input files are tracked or always present, and a missing baseline is a
failure rather than a pass (Ruling R3).

```python
# tests/test_corpus_report.py: append

class TestChannelParity:
    def test_report_and_lookup_agree_on_every_curated_link(self) -> None:
        """The report must describe the join the pipeline performs."""
        from tract.corpus_report import (
            CURATED_LINKS_PATH, _load_records, _lookup_with_channel,
        )
        from tract.text_selection import (
            ProseIndex, canonical_framework, merged_corpus_path,
        )

        corpus = merged_corpus_path()
        records = _load_records(corpus)
        # The bug this test exists to prevent: an empty index agrees with
        # itself on everything.
        assert records, f"{corpus} produced no framework records"
        index = ProseIndex(records)
        assert len(index) > 1000, (
            f"prose index holds {len(index)} controls; a near-empty index "
            f"makes every assertion below vacuous"
        )

        compared = 0
        with CURATED_LINKS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                canonical = canonical_framework(row.get("standard_name", ""))
                mine, _ = _lookup_with_channel(
                    index, canonical, row.get("section_id"),
                    row.get("section_name"),
                )
                theirs = index.lookup(
                    row.get("standard_name", ""), row.get("section_id"),
                    row.get("section_name"),
                )
                assert (mine is None) == (theirs is None), row
                if mine is not None and theirs is not None:
                    assert mine.text == theirs.text, row
                compared += 1
        assert compared == 4405, compared

    def test_the_report_resolves_a_useful_number_of_links(self) -> None:
        """A second guard against an index that silently held nothing."""
        report = build_corpus_report()
        resolved = report.totals.by_title + report.totals.by_id
        assert report.totals.links == 4405
        assert resolved >= 3600, resolved
        assert report.corpus_framework_count >= 29
```

- [ ] **Step 9: Write the CLI**

```python
# scripts/corpus_report.py: create

"""Print or persist the corpus join report.

    PYTHONPATH=. "$PY" scripts/corpus_report.py
    PYTHONPATH=. "$PY" scripts/corpus_report.py --tag before
    PYTHONPATH=. "$PY" scripts/corpus_report.py --out results/corpus/scratch.json

The same entry point produces the BEFORE artifact, every per-parser acceptance
check, and the final corpus report. One instrument, one code path: a parser
accepted by a measurement its consumer does not use is a parser accepted by
nothing.

--tag writes the pair a later reader needs together: results/corpus/<tag>.json
for the summary and results/corpus/link_resolution_<tag>.jsonl for the per-link
record. Both paths are anchored to PROJECT_ROOT, so the working directory does
not decide where evidence lands.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tract.corpus_report import (
    CORPUS_EVIDENCE_DIR,
    build_corpus_report,
    format_table,
    write_link_resolution,
    wrong_anchor_applicable,
)
from tract.io import atomic_write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tag", type=str, default=None,
        help="write results/corpus/<tag>.json and link_resolution_<tag>.jsonl",
    )
    args = parser.parse_args()

    report = build_corpus_report(args.links, args.corpus)
    print(format_table(report))
    print()
    print(f"corpus  {report.corpus_path}  sha256 {report.corpus_sha256[:16]}")
    print(f"links   {report.links_path}  sha256 {report.links_sha256[:16]}")
    print(f"frameworks in corpus  {report.corpus_framework_count}")
    print()
    print("wrong-anchor checks applicable, per framework:")
    for framework_id, applicable in sorted(wrong_anchor_applicable(report).items()):
        risk = report.by_id(framework_id).wrong_anchor_risk
        note = "" if applicable else "   (blind: no detector applies)"
        print(f"  {framework_id:26s} {risk:4d} of {applicable:5d}{note}")

    if args.out is not None:
        atomic_write_json(report.to_json(), args.out)
        print(f"wrote {args.out}")

    if args.tag is not None:
        summary = CORPUS_EVIDENCE_DIR / f"{args.tag}.json"
        detail = CORPUS_EVIDENCE_DIR / f"link_resolution_{args.tag}.jsonl"
        atomic_write_json(report.to_json(), summary)
        write_link_resolution(report, detail)
        print(f"wrote {summary}")
        print(f"wrote {detail}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run the tests and the typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_corpus_report.py -q
"$PY" -m mypy tract/corpus_report.py scripts/corpus_report.py \
    tract/text_selection.py --strict
```

Expected: all tests pass, no mypy errors. If `TestChannelParity` fails on `compared == 4405`, the
curated link file changed and every number in this task needs re-measuring before going further.

- [ ] **Step 11: Record the attainable range of `wrong_anchor_risk`, per framework**

A gate that cannot fail is a defect. Before any parser lands, measure the denominator so nobody
writes `assert wrong_anchor_risk == 0` where zero is guaranteed.

```bash
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
import re
from tract.config import TRAINING_DIR
from tract.text_selection import normalize_section_id

SEP = (".", "-", "_", ":", " ")
WS = re.compile(r"\s+")


def fold(text):
    return WS.sub(" ", (text or "").strip()).casefold()


def ancestor(parent, child):
    return (
        bool(parent) and bool(child) and len(child) > len(parent)
        and child.startswith(parent) and child[len(parent)] in SEP
    )


grouped = {}
with (TRAINING_DIR / "hub_links_curated.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["framework_id"], []).append(row)

eleven = [
    "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63", "nist_ssdf",
    "owasp_proactive_controls", "owasp_top10_2021", "samm", "wstg",
]
print(f"{'framework':26s}{'links':>6}{'B max':>7}{'C max':>7}")
for framework_id in eleven:
    links = grouped[framework_id]
    ids = [normalize_section_id(link.get("section_id")) for link in links]
    unique = sorted({i for i in ids if i})
    related = {
        i for i in unique
        if any(o != i and (ancestor(o, i) or ancestor(i, o)) for o in unique)
    }
    b = sum(
        1 for link in links
        if link.get("section_name")
        and fold(link["section_name"])
        != fold(normalize_section_id(link.get("section_id")))
    )
    c = sum(1 for i in ids if i in related)
    print(f"{framework_id:26s}{len(links):6d}{b:7d}{c:7d}")
PYEOF
```

Expected, and these are the numbers to write into the run ledger **[measured, this task]**:

```
framework                  links  B max  C max
biml                          21     21      0
csa_ccm                       29     29      0
dsomm                        214    214      3
enisa                         68     58      0
etsi                          36     34      8
nist_800_63                   79      0     14
nist_ssdf                     46     44      0
owasp_proactive_controls      76      0      0
owasp_top10_2021              17     17      0
samm                          30     30      0
wstg                         118      0      0
```

Read it this way. Detector A applies only to links that resolve through the title channel, and nine
of the eleven are designed to resolve through the id channel, so A contributes nothing for them.
Detectors B and C are the id-side ones. **`owasp_proactive_controls` and `wstg` have an attainable
maximum of zero on all three detectors**, because their `section_name` equals their `section_id` (so
B never applies) and their link ids carry no ancestor relations (so C never applies). For those two,
`wrong_anchor_risk == 0` is unfailable and asserting it measures nothing. Every other framework has a
real range and `csa_ccm` is expected to fire: its `IPY` link resolves to control IPY-01's title
rather than the IPY domain. **[measured, ML Engineer]**

What Task 16 asserts instead of `== 0`:

- For every pending framework, `wrong_anchor_risk` is either 0 or every flagged link appears in a
  committed adjudication list. The flags are in
  `results/corpus/link_resolution_after.jsonl` as `"wrong_anchor": true`, so an unadjudicated new
  flag is a diff, not a judgment call.
- For `owasp_proactive_controls` and `wstg`, record the blindness by name in the ledger rather than
  recording a pass.

- [ ] **Step 12: Capture the BEFORE state**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" scripts/corpus_report.py --tag before
```

Expected, and this is the free test of the instrument. These values were measured independently
while writing this task **[measured, all]**:

```
framework                  links   ttl   id unres   fb  anch   pre   l/a trunc nest cont noidx wrong  ftxt  desc titl synt  hubs   l/h   rate
asvs                         277   270    7     0    0   277   277  1.00     0    0    0     0     1     0   277    0    0   277  1.00 1.0000
biml                          21     0    0    21   17     0     0  0.00     0    0    0    20     0     0     0    0    0    11  1.91 0.0000
capec                       1799  1799    0     0    0   349   349  5.15    24    0    0     2     0    24  1775    0    0   194  9.27 1.0000
csa_ccm                       29     0    0    29   29     0     0  0.00     0    0    0    29     0     0     0    0    0    27  1.07 0.0000
cwe                          613   594   18     1    1   245   245  2.50    13    0    0    43     2    13   599    0    0   268  2.29 0.9984
dsomm                        214     0    0   214   18     0     0  0.00     0    0    0   183     0     0     0    0    0    24  8.92 0.0000
enisa                         68     0    0    68   33     0     0  0.00     0    0    0    38     0     0     0    0    0    56  1.21 0.0000
etsi                          36     0    0    36   24     0     0  0.00     0    0    0    27     0     0     0    0    0    29  1.24 0.0000
iso_27001                     94    88    4     2    2    91    91  1.01     0    0    0     2     5     0    92    0    0    47  2.00 0.9787
mitre_atlas                   65    56    9     0    0    43    43  1.51     0    0    0     0     0     0    65    0    0    41  1.59 1.0000
nist_800_53                  300   298    0     2    2   298   298  1.00    63    0    0     0     0    70   228    0    0    66  4.55 0.9933
nist_800_63                   79     0    0    79   25     0     0  0.00     0    0    0    25     0     0     0    0    0    70  1.13 0.0000
nist_ai_100_2                 45    29   16     0    0    22    22  2.05    21    0    0     0    20    24    21    0    0    27  1.67 1.0000
nist_ssdf                     46     0    0    46   44     0     0  0.00     0    0    0    44     0     0     0    0    0    28  1.64 0.0000
owasp_ai_exchange             64    38   26     0    0    63    63  1.02    31    0    0     0    11    35    29    0    0    64  1.00 1.0000
owasp_cheat_sheets           391   391    0     0    0    49    49  7.98   384    0    0     0     0   384     7    0    0   180  2.17 1.0000
owasp_llm_top10               13    12    1     0    0     6     6  2.17    13    0    0     0     0    13     0    0    0    10  1.30 1.0000
owasp_ml_top10                10     9    1     0    0     7     7  1.43    10    0    0     0     1    10     0    0    0     7  1.43 1.0000
owasp_proactive_controls      76     0    0    76   10     0     0  0.00     0    0    0    10     0     0     0    0    0    73  1.04 0.0000
owasp_top10_2021              17     0    0    17   10     0     0  0.00     0    0    0    10     0     0     0    0    0    16  1.06 0.0000
samm                          30     0    0    30   30     0     0  0.00     0    0    0    30     0     0     0    0    0    22  1.36 0.0000
wstg                         118     0    0   118   59     0     0  0.00     0    0    0    59     0     0     0    0    0   112  1.05 0.0000
TOTAL                       4405  3584   82   739  304  1450  1450  2.53   559    0    0   558    40   573  3093    0    0   458  9.62 0.8322

frameworks in corpus  31
```

Three properties must hold before going further.

1. **`frameworks in corpus` reads 31.** If it reads 29, the licensed overlay is missing from this
   checkout and `merged_corpus_path()` fell back to the tracked corpus. Fix that first: every later
   comparison would be against the wrong baseline (ledger lesson 5), and ISO is the corpus's only
   0.967-prose fold.
2. **ISO is not zero.** `iso_27001` resolves 92 of 94 at 0.9787, so the instrument can see a working
   join.
3. **The eleven are all zero on `anch` and non-zero on `fb`.** Zero resolved anchors, 299 fallback
   anchors between them, so the instrument can see a broken join and can also see what the trainer
   gets in spite of it.

Five totals are worth naming before any parser exists. **1,450 distinct anchors** carry 3,666
resolved links. **304 fallback anchors** carry the other 739, and **299 of the 304 belong to the
eleven**. **559 anchors are truncated** at `MAX_ANCHOR_CHARS`, 384 of them in `owasp_cheat_sheets`
alone. **558 controls are in the corpus and absent from the prose index**, because their description
does not exceed their title by `PROSE_MIN_EXTRA_CHARS`: 475 in the eleven, 43 CWE weaknesses, 2 CAPEC
patterns, ISO's A.7.8 and A.7.9, and 36 in three frameworks that carry no curated links at all (NIST
AI Risk Management Framework 25, AIUC-1 Standard 10, CoSAI 1). **`nest` and `cont` both read 0
corpus-wide**, so the containment change costs nothing today and exists for what the parsers will
build. **[measured, all]** No artifact in this repository reported any of the five before now.

- [ ] **Step 13: Confirm the evidence is stageable without `-f`**

```bash
git check-ignore -v results/corpus/before.json \
    || echo "before.json: not ignored"
git check-ignore -v results/corpus/link_resolution_before.jsonl \
    || echo "link_resolution_before.jsonl: not ignored"
"$PY" -c "
import json, pathlib
rows = [
    json.loads(line)
    for line in pathlib.Path(
        'results/corpus/link_resolution_before.jsonl'
    ).read_text(encoding='utf-8').splitlines()
]
print('rows', len(rows))
print('keys', sorted(rows[0]))
assert len(rows) == 4405, len(rows)
assert not any('anchor_text' in r for r in rows)
assert all(len(r['anchor_sha256']) == 64 for r in rows)
print('no anchor text in the file')
"
```

Expected: both paths report `not ignored`, `rows 4405`, twelve keys, and `no anchor text in the
file`. The JSONL carries digests and lengths only, which is what makes it safe to track for ETSI and
ISO alongside everything else.

- [ ] **Step 14: Commit**

```bash
git add tract/corpus_report.py scripts/corpus_report.py \
        tests/test_corpus_report.py tract/text_selection.py .gitignore \
        results/corpus/before.json \
        results/corpus/link_resolution_before.jsonl
git status --porcelain --short
git commit -m "feat: measure the corpus join by anchor and by fallback, not only by link"
```

`git status --porcelain --short` before the commit is not decoration. `git add` on an ignored path
exits 1 while still staging the other paths on the same command line, so a partially staged commit
looks successful. Confirm all seven paths are staged before committing.

---

### Task 2: An `alt_ids` channel on ProseIndex

`ProseIndex` reads `metadata["alt_titles"]` and nothing else. **[measured]** There is no id-side
equivalent, and three frameworks need one:

- **nist_ssdf**: two of 46 curated links carry a mid-sentence text fragment where a `PS.1.1`-style id
  belongs. Both are recoverable: the first fragment appears verbatim inside task `PS.1.1`'s
  statement, the second inside `PW.8.1`'s. **[measured]** Without `alt_ids` the ceiling is 44/46.
  With it, 46/46.
- **biml**: 8 of 21 curated links carry an unprefixed `category:number` id while the same id means
  something different in the other BIML document. Seven resolve to one document by exact tag-label
  match. **[measured]**
- **wstg / csa_ccm**: neither needs it. CSA CCM's seven retired `IVS-*` ids resolve through the title
  channel already **[measured]**, so the rename map the previous plan carried is dead weight and is
  not built here.

#### The two-pass rule, stated correctly

Version 2 said `alt_ids` "follows `alt_titles`' two-pass rule exactly". The **second** passes match.
The **first** passes do not, and the difference decides what the tests below have to pin.

| | real entries | alternate entries |
|---|---|---|
| `_by_title` | **first writer wins**: `if key and (framework, key) not in self._by_title` | first writer wins, and never displaces a real title |
| `_by_id` | **last writer wins**, unguarded: `self._by_id[(framework, control_id)] = selection` | this task: first writer wins, and never displaces a real id |

**[measured, read out of `tract/text_selection.py` at lines 270-300]** So the second pass carries the
whole guarantee on the id side, and the tests must cover two collisions version 2 never mentioned:
two controls declaring the **same real id** (last one silently wins today) and two controls declaring
the **same alternate id** (first one wins after this task).

Real-id behaviour does not change here. Switching `_by_id` to first-writer-wins would move the join
and invalidate the BEFORE artifact captured one commit ago, which is a different decision with a
different owner. The current corpus carries **zero** real `control_id` collisions among indexed
controls, so the asymmetry is latent rather than live. It carries **82** real title collisions, all
in AIUC-1 Standard 73, NIST AI 100-2 7, EU AI Act 1 and NIST AI RMF 1, where first-writer-wins
silently drops the later control's title. **[measured, this task]** Both facts are pinned below so a
later change to either rule turns a test red.

#### The regression check version 2 shipped was vacuous

Version 2's Step 6 diffed the BEFORE artifact and expected `identical: True`, calling that proof the
new channel is safe. Zero controls in the corpus carry `alt_ids` today **[measured]**, so that check
verifies the metadata-binding move and says nothing at all about the channel. It is kept, because the
binding move is a real code change that could break `alt_titles`, and a second check is added that
exercises the channel end to end on real data: seed the one CWE link that does not resolve today as
an `alt_id` and watch cwe go 612/613 to 613/613. That link is `section_id == "937"`. Control 937 is
in the corpus, an obsolete CWE **category** whose 85-character description does not exceed its
76-character title by `PROSE_MIN_EXTRA_CHARS`, so `ProseIndex` never indexes it and no channel can
reach it. **[measured, this task]**

#### ISO's 92 of 94 becomes an assertion

`92/94` appears nowhere in any test. It lives in a comment at `tests/test_prose_reachability.py:52`,
and `iso_27001` is deliberately absent from that file's `PARSER_BACKED_WITH_LINKS` set because its
prose is restricted. **[measured]** So the one framework whose join this task could break has no
guard. This task adds one, gated on corpus content rather than on file existence per Rule 7.

**Files:**
- Modify: `tract/text_selection.py`
- Modify: `tests/test_text_selection.py`
- Modify: `tests/test_corpus_report.py`

**Interfaces:**
- Consumes: `tract.corpus_report.build_corpus_report`, `CORPUS_EVIDENCE_DIR`,
  `FULL_CORPUS_FRAMEWORK_COUNT` from Task 1.
- Produces: `ProseIndex` honouring `metadata["alt_ids"]: list[str] | str`, normalised through
  `normalize_section_id`, never displacing a real `control_id`, and counting both collision kinds.

**Invalidates:**
- Nothing at runtime. No control in the corpus carries `alt_ids` yet **[measured]**, and Step 6
  asserts the report is byte-identical, so training data, evaluation corpora and every published
  artifact are untouched.
- `results/corpus/before.json` and `results/corpus/link_resolution_before.jsonl` are re-verified
  rather than regenerated. If they move, the second pass displaced a real id and the change is wrong.
- `ProseIndex.load`'s log line, which stops meaning "controls with a real id".
- The comment at `tests/test_prose_reachability.py:52`. It stays as documentation and stops being the
  only place the number exists.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_text_selection.py: append

class TestAlternateIds:
    """A retired or malformed id may add a key. It may never displace one."""

    LONG = "A control statement long enough to clear the prose bar easily. " * 3

    def _index(self) -> ProseIndex:
        return ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "PS.1.1", "title": "Protect code",
                 "description": self.LONG + " Real PS.1.1.",
                 "metadata": {"alt_ids": ["code, executable code"]}},
                {"control_id": "PW.8.1", "title": "Test executable",
                 "description": self.LONG + " Real PW.8.1."},
            ],
        }])

    def test_alternate_id_resolves(self) -> None:
        hit = self._index().lookup("Demo", "code, executable code", None)
        assert hit is not None
        assert hit.text.endswith("Real PS.1.1.")

    def test_alternate_never_displaces_a_real_id(self) -> None:
        """The alternate is declared first, in corpus order, on purpose."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "A-1", "title": "First",
                 "description": self.LONG + " First.",
                 "metadata": {"alt_ids": ["A-2"]}},
                {"control_id": "A-2", "title": "Second",
                 "description": self.LONG + " Second."},
            ],
        }])
        hit = index.lookup("Demo", "A-2", None)
        assert hit is not None
        assert hit.text.endswith("Second.")

    def test_alternate_never_displaces_a_real_id_declared_earlier(self) -> None:
        """Order must not matter. The second pass is what carries this."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "A-2", "title": "Second",
                 "description": self.LONG + " Second."},
                {"control_id": "A-1", "title": "First",
                 "description": self.LONG + " First.",
                 "metadata": {"alt_ids": ["A-2"]}},
            ],
        }])
        hit = index.lookup("Demo", "A-2", None)
        assert hit is not None
        assert hit.text.endswith("Second.")

    def test_two_alternates_claiming_one_key_keep_the_first(self) -> None:
        """Alternates are first-writer-wins among themselves, and counted."""
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "B-1", "title": "One",
                 "description": self.LONG + " One.",
                 "metadata": {"alt_ids": ["shared"]}},
                {"control_id": "B-2", "title": "Two",
                 "description": self.LONG + " Two.",
                 "metadata": {"alt_ids": ["shared"]}},
            ],
        }])
        hit = index.lookup("Demo", "shared", None)
        assert hit is not None
        assert hit.text.endswith("One.")
        assert index.alternate_id_collisions == 1

    def test_two_real_ids_claiming_one_key_keep_the_last(self) -> None:
        """Real ids are last-writer-wins and unguarded. Pinned, not fixed.

        Changing this would move the join measured in Task 1's BEFORE
        artifact. The current corpus carries zero such collisions among
        indexed controls. [measured]
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "C-1", "title": "One",
                 "description": self.LONG + " One."},
                {"control_id": "C-1", "title": "Two",
                 "description": self.LONG + " Two."},
            ],
        }])
        hit = index.lookup("Demo", "C-1", None)
        assert hit is not None
        assert hit.text.endswith("Two.")
        assert index.real_id_collisions == 1

    def test_two_real_titles_claiming_one_key_keep_the_first(self) -> None:
        """The other half of the asymmetry, pinned for the same reason.

        82 of these exist in the corpus today: AIUC-1 Standard 73,
        NIST AI 100-2 7, EU AI Act 1, NIST AI RMF 1. [measured]
        """
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "D-1", "title": "Same title",
                 "description": self.LONG + " One."},
                {"control_id": "D-2", "title": "Same title",
                 "description": self.LONG + " Two."},
            ],
        }])
        hit = index.lookup("Demo", None, "Same title")
        assert hit is not None
        assert hit.text.endswith("One.")

    def test_alternate_id_is_normalised_like_a_real_one(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "X-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": ["2.2"]}},
            ],
        }])
        assert index.lookup("Demo", "Sec. 2.2", None) is not None

    def test_a_bare_string_alternate_is_accepted(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "Y-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": "Y-legacy"}},
            ],
        }])
        assert index.lookup("Demo", "Y-legacy", None) is not None

    def test_an_empty_alternate_is_ignored(self) -> None:
        index = ProseIndex([{
            "framework_name": "Demo",
            "controls": [
                {"control_id": "Z-1", "title": "Only",
                 "description": self.LONG + " Only.",
                 "metadata": {"alt_ids": ["", "   "]}},
            ],
        }])
        assert len(index) == 1
        assert index.alternate_id_collisions == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_text_selection.py::TestAlternateIds -q
```

Expected: FAIL. `test_alternate_id_resolves` fails on `assert hit is not None`, and the collision
tests fail with `AttributeError: 'ProseIndex' object has no attribute
'alternate_id_collisions'`.

- [ ] **Step 3: Implement**

```python
# tract/text_selection.py: in ProseIndex.__init__, replace the pending list
        pending_alternates: list[tuple[tuple[str, str], TextSelection]] = []
        pending_alternate_ids: list[tuple[tuple[str, str], TextSelection]] = []
        # Two controls can claim one key on either side. Neither case raises,
        # because the corpus is a fact rather than an input this class
        # validates, but neither is silent either: an unreported collision is
        # a control that vanished from the join with no column to see it in.
        self.real_id_collisions = 0
        self.alternate_id_collisions = 0
```

```python
# tract/text_selection.py: in ProseIndex.__init__, replace the control_id block
                # metadata is bound here rather than below, because both kinds
                # of alternate read it and one binding cannot drift from the
                # other. The alt_titles read below is unchanged and uses it.
                metadata = control.get("metadata") or {}

                control_id = normalize_section_id(control.get("control_id"))
                if control_id:
                    # Last writer wins, deliberately unchanged: the join
                    # measured in results/corpus/before.json depends on it.
                    if (framework, control_id) in self._by_id:
                        self.real_id_collisions += 1
                        logger.warning(
                            "Two %s controls claim id %r; the later one wins "
                            "and the earlier is unreachable by id.",
                            framework, control_id,
                        )
                    self._by_id[(framework, control_id)] = selection

                # Retired and malformed ids, held back for the same reason
                # alt_titles are: an alternate must never take the key of a
                # control whose real id spells it. NIST SSDF has two curated
                # links whose section_id is a mid-sentence fragment of the
                # task text, and BIML has eight whose id is document-scoped
                # upstream but unprefixed in OpenCRE.
                alternate_ids = metadata.get("alt_ids") or []
                if isinstance(alternate_ids, str):
                    alternate_ids = [alternate_ids]
                for raw_id in alternate_ids:
                    alt_id = normalize_section_id(str(raw_id))
                    if alt_id:
                        pending_alternate_ids.append(
                            ((framework, alt_id), selection)
                        )
```

The existing `metadata = control.get("metadata") or {}` assignment a few lines below, where
`alt_titles` is read, is deleted by this edit. The `alternates = metadata.get("alt_titles") or []`
line beneath it is unchanged and now reads the binding above.

```python
# tract/text_selection.py: after the existing alternates second pass
        # Alternates are applied after every real id is in place, so a real id
        # always wins regardless of corpus order. Among themselves the first
        # writer wins, matching alt_titles, and a loser is counted rather than
        # dropped in silence.
        claimed_by_alternate: set[tuple[str, str]] = set()
        for key_pair, selection in pending_alternate_ids:
            if key_pair in self._by_id:
                if key_pair in claimed_by_alternate:
                    self.alternate_id_collisions += 1
                    logger.warning(
                        "Two %s controls declare alternate id %r; the first "
                        "one keeps the key.", key_pair[0], key_pair[1],
                    )
                continue
            self._by_id[key_pair] = selection
            claimed_by_alternate.add(key_pair)
```

- [ ] **Step 4: Update the load() log line**

```python
# tract/text_selection.py: in ProseIndex.load, replace the logger.info call
        logger.info(
            "Prose index from %s: %d controls by id (real and alternate), "
            "%d by title, %d real id collisions, %d alternate id collisions",
            source.name, len(index._by_id), len(index._by_title),
            index.real_id_collisions, index.alternate_id_collisions,
        )
```

- [ ] **Step 5: Prove the channel works end to end on the real corpus**

Version 2 had no test that exercised `alt_ids` against real data, because no control carries the
field. This one seeds it, on a framework that is in the **tracked** corpus so CI can run it. `cwe` is
the right subject: its single unresolved curated link is `section_id == "937"`, CWE-937 having been
withdrawn upstream. **[measured]**

```python
# tests/test_corpus_report.py: append

class TestAlternateIdsAgainstTheRealCorpus:
    """The new channel, exercised on the corpus rather than on a fixture.

    cwe resolves 612 of 613 curated links. The miss is section_id "937", an
    obsolete CWE category whose description is shorter than its title plus
    PROSE_MIN_EXTRA_CHARS, so ProseIndex never indexes it. [measured]
    Attaching "937" as an alt_id to any indexed CWE control closes the gap,
    which is a property of the channel rather than of that control.
    """

    def _cwe_only(self, tmp_path: Path) -> tuple[Path, Path]:
        from tract.corpus_report import CURATED_LINKS_PATH, _load_records
        from tract.text_selection import merged_corpus_path

        records = [
            record for record in _load_records(merged_corpus_path())
            if record.get("framework_name") == "CWE"
        ]
        assert len(records) == 1, "expected exactly one CWE record"
        corpus = tmp_path / "cwe.json"
        corpus.write_text(
            json.dumps(
                {"framework_count": 1, "frameworks": records,
                 "generated_date": "2026-01-01",
                 "total_controls": len(records[0]["controls"])},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rows = [
            line for line in
            CURATED_LINKS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["framework_id"] == "cwe"
        ]
        links = tmp_path / "cwe.jsonl"
        links.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return corpus, links

    def test_the_unresolved_cwe_link_stays_unresolved_without_an_alt_id(
        self, tmp_path: Path,
    ) -> None:
        corpus, links = self._cwe_only(tmp_path)
        row = build_corpus_report(links, corpus).by_id("cwe")
        assert row.links == 613
        assert row.unresolved == 1
        assert row.by_title + row.by_id == 612

    def test_an_alt_id_closes_it(self, tmp_path: Path) -> None:
        corpus, links = self._cwe_only(tmp_path)
        data = json.loads(corpus.read_text(encoding="utf-8"))
        controls = data["frameworks"][0]["controls"]
        target = next(
            c for c in controls
            if str(c.get("full_text") or "").strip()
            or len(str(c.get("description") or "").strip())
            > len(str(c.get("title") or "").strip()) + 20
        )
        metadata = dict(target.get("metadata") or {})
        metadata["alt_ids"] = ["937"]
        target["metadata"] = metadata
        corpus.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

        row = build_corpus_report(links, corpus).by_id("cwe")
        assert row.unresolved == 0
        assert row.by_id == 19
        assert row.resolution_rate == pytest.approx(1.0)

    def test_an_alt_id_cannot_take_a_live_cwe_id(self, tmp_path: Path) -> None:
        """The guarantee, on real data: 79 is a real CWE and must not move."""
        corpus, links = self._cwe_only(tmp_path)
        data = json.loads(corpus.read_text(encoding="utf-8"))
        controls = data["frameworks"][0]["controls"]
        real = next(c for c in controls if str(c["control_id"]) == "79")
        other = next(
            c for c in controls
            if str(c["control_id"]) != "79"
            and (str(c.get("full_text") or "").strip()
                 or len(str(c.get("description") or "").strip())
                 > len(str(c.get("title") or "").strip()) + 20)
        )
        metadata = dict(other.get("metadata") or {})
        metadata["alt_ids"] = ["79"]
        other["metadata"] = metadata
        corpus.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

        from tract.corpus_report import _load_records
        from tract.text_selection import ProseIndex

        index = ProseIndex(_load_records(corpus))
        hit = index.by_id("CWE", "79")
        assert hit is not None
        expected = str(real.get("full_text") or "").strip() or str(
            real["description"]
        ).strip()
        assert hit.text == expected


class TestIsoStillResolves:
    """92 of 94 lived only in a comment. Now it is a gate.

    Skipped as a named group when the licensed overlay is absent, per Rule 7,
    because gating on file existence never skips: the tracked corpus always
    exists and the restricted rows would hard-fail in CI on data that cannot
    legally be there.
    """

    def test_iso_resolves_92_of_94_with_91_distinct_anchors(self) -> None:
        from tract.corpus_report import FULL_CORPUS_FRAMEWORK_COUNT

        report = build_corpus_report()
        if report.corpus_framework_count < FULL_CORPUS_FRAMEWORK_COUNT:
            pytest.skip(
                f"corpus has {report.corpus_framework_count} frameworks "
                f"against {FULL_CORPUS_FRAMEWORK_COUNT} in the full set, so "
                f"the licensed overlay is absent from this checkout and the "
                f"restricted rows cannot be asserted"
            )
        row = report.by_id("iso_27001")
        assert row.links == 94
        assert row.by_title + row.by_id == 92
        assert row.distinct_anchors == 91
        assert row.dropped_by_prose_rule == 2
```

- [ ] **Step 6: Run the tests and the typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_text_selection.py tests/test_corpus_report.py \
    tests/test_prose_reachability.py -q
"$PY" -m mypy tract/text_selection.py tract/corpus_report.py --strict
```

Expected: PASS. `tests/test_prose_reachability.py` measures the same join and must not move.

- [ ] **Step 7: Confirm the BEFORE state is unchanged**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" scripts/corpus_report.py --out /tmp/after_alt_ids.json
"$PY" - <<'PYEOF'
import json
from pathlib import Path

before = json.loads(
    Path("results/corpus/before.json").read_text(encoding="utf-8")
)
after = json.loads(Path("/tmp/after_alt_ids.json").read_text(encoding="utf-8"))
print("per_framework identical:", before["per_framework"] == after["per_framework"])
print("totals identical:", before["totals"] == after["totals"])
print("corpus sha identical:", before["corpus_sha256"] == after["corpus_sha256"])
assert before["per_framework"] == after["per_framework"]
assert before["totals"] == after["totals"]
PYEOF
```

Expected: three `True` lines. No control in the corpus carries `alt_ids` yet **[measured]**, so
adding the channel must move nothing. A difference here means the second pass displaced a real id,
and the guarantee this task exists to provide is broken.

This check alone is not proof the channel works. It verifies the metadata-binding move and the second
pass. Step 5 is what proves the channel, by seeding a real `alt_id` on a real framework and watching
`cwe` go 612 to 613.

- [ ] **Step 8: Commit**

```bash
git add tract/text_selection.py tests/test_text_selection.py \
        tests/test_corpus_report.py
git status --porcelain --short
git commit -m "feat: resolve a link through a control's retired identifiers"
```

---
### Task 3: DSOMM, 214 links off 18 title anchors onto 182 control anchors

**Invalidates:** `data/processed/frameworks/dsomm.json` and the overlay corpus built from it. DSOMM moves from 18 fallback anchors to 182 control anchors, so every corpus report, every join figure and any training run reading DSOMM prose is stale until Task 15 reruns the merge. Its 214 links are all `AutomaticallyLinkedTo` onto 24 hubs, and its label agreement is unmeasured, so this also invalidates any assumption that training weight is uniform in quality.

The largest anchor win in the plan and the one that best demonstrates why the
instrument counts anchors. Today DSOMM's 214 curated links resolve to nothing
and fall back to their `section_name`, which is the **sub-dimension** name, not
the activity name: 214 links land on **18** distinct anchors, 11.89 links each.
**[measured]** The `section_id` is the activity's own uuid and matches the
source verbatim for 214 of 214 rows over 183 distinct uuids. **[measured]**

**The source-structures document is wrong about the prose field.** It says
`description` carries the activity's prose. Measured on the pinned archive:
`description` is present on 53 of 194 activities and non-empty on **51**.
`risk` and `measure` are present and non-empty on **194 of 194**. **[measured]**
A parser that took `description` alone would emit 143 empty descriptions,
`Control(description=...)` would raise on `min_length=1`, and the ones that
survived would drop out of `ProseIndex` on the prose rule.

**Ceiling, derived before the parser is written.** 214 curated links, all 214
uuids present in the model. With `description + risk + measure` as the
statement, 193 of 194 activities clear `_is_prose` against their own name; the
single failure is *Correlate known vulnerabilities in infrastructure with new
image versions*, an 11-character body under a 73-character title. **[measured]**
One curated link targets it. Ceiling **213/214 = 0.99533** **[derived]**, floor
**0.99**. The previous plan set 1.00 against this same maximum and would have
failed on landing.

Description lengths run 11 / 282 / 4,565 (min / median / max); 4 exceed
`DESCRIPTION_MAX_LENGTH`, so 4 controls acquire a `full_text` from
`_sanitize_control` and anchor on it. **[measured]**

**Files:**
- Create: `parsers/parse_dsomm.py`
- Create: `tests/test_parse_dsomm.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`; `tract.corpus_report.check_join_floors` from Task 1.
- Produces: `DsommParser` with `framework_id = "dsomm"`, `framework_name = "DSOMM"`; `DsommParser.activities_to_controls(model: dict[str, Any]) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_dsomm.py: create

"""DSOMM's prose lives in risk and measure, not in description.

Measured on the pinned archive: description is non-empty on 51 of 194
activities, risk and measure on 194 of 194. A parser reading description alone
emits 143 empty statements and ProseIndex indexes none of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from parsers.parse_dsomm import DsommParser
from tract.parsers.base import BaseParser

MODEL: dict[str, Any] = {
    "Build and Deployment": {
        "Deployment": {
            "Inventory of production components": {
                "uuid": "2a44b708-734f-4463-b0cb-86dc46344b2f",
                "risk": "Without an inventory of deployed artifacts it is not "
                        "possible to know where a vulnerable image runs.",
                "measure": "A documented inventory of artifacts in production "
                           "is maintained and kept current.",
                "level": 1,
            },
            "Pinning of artifacts": {
                "uuid": "f3c4971e-9f4d-4e59-8ed0-f0bdb6262477",
                "description": "Pin base images and dependencies to an "
                               "immutable digest rather than a moving tag.",
                "risk": "Unauthorized manipulation of artifacts is hard to "
                        "spot when tags move under the build.",
                "measure": "Pinning ensures changes happen only when intended.",
                "level": 2,
            },
        },
    },
}


@pytest.fixture()
def parser(tmp_path: Path) -> DsommParser:
    import io
    import zipfile

    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "repo-abc123/generated/model.yaml",
            "---\nmeta:\n  version: test\n---\n" + yaml.safe_dump(MODEL),
        )
    (raw / "dsomm_data.zip").write_bytes(payload.getvalue())

    instance = DsommParser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_count = 2
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_control_id_is_the_uuid_opencre_links_against(
        self, parser: DsommParser,
    ) -> None:
        controls = parser.parse()
        assert [c.control_id for c in controls] == [
            "2a44b708-734f-4463-b0cb-86dc46344b2f",
            "f3c4971e-9f4d-4e59-8ed0-f0bdb6262477",
        ]

    def test_title_is_the_activity_name_not_the_sub_dimension(
        self, parser: DsommParser,
    ) -> None:
        titles = [c.title for c in parser.parse()]
        assert titles == ["Inventory of production components",
                          "Pinning of artifacts"]
        assert "Deployment" not in titles

    def test_statement_survives_an_absent_description(
        self, parser: DsommParser,
    ) -> None:
        first = parser.parse()[0]
        assert "inventory of deployed artifacts" in first.description
        assert "documented inventory" in first.description
        assert len(first.description) >= 60

    def test_description_leads_when_present(self, parser: DsommParser) -> None:
        second = parser.parse()[1]
        assert second.description.startswith("Pin base images")

    def test_sub_dimension_is_recorded_as_the_parent(
        self, parser: DsommParser,
    ) -> None:
        first = parser.parse()[0]
        assert first.parent_id == "Deployment"
        assert first.parent_name == "Build and Deployment"


class TestRun:
    def test_run_writes_and_clears_the_prose_floor(
        self, parser: DsommParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 2
        assert (tmp_path / "out" / "dsomm.json").exists()
        assert BaseParser.honest_prose_fraction(output.controls) == 1.0
        assert [s.path for s in output.source_files] == ["dsomm_data.zip"]

    def test_reparse_is_byte_identical(
        self, parser: DsommParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        first = (tmp_path / "out" / "dsomm.json").read_bytes()
        parser.run()
        assert (tmp_path / "out" / "dsomm.json").read_bytes() == first


class TestDigestGate:
    def test_a_different_archive_is_refused(self, parser: DsommParser) -> None:
        parser.expected_sha256 = "0" * 64
        with pytest.raises(ValueError, match="not the pinned"):
            parser.parse()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_dsomm.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_dsomm'`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_dsomm.py: create

"""Parser for the OWASP DevSecOps Maturity Model.

OpenCRE keys every DSOMM link on the activity's `uuid`, and 214 of 214 curated
links carry a uuid this file defines. What OpenCRE puts in `section_name` is
the SUB-DIMENSION, one level above the activity, so 214 links share 18 names.
Falling back to that name is what the corpus does today and it is why DSOMM
reads 11.89 links per anchor. Joining on the uuid takes it to 182 anchors.

The statement is `description`, `risk` and `measure` concatenated in that
order, and the order is not cosmetic. `description` is present on 53 of the 194
activities and non-empty on 51; `risk` and `measure` are non-empty on all 194.
A parser reading `description` alone would emit 143 empty statements, which
`Control` rejects, and the survivors would fail the prose rule that decides
whether ProseIndex indexes a control at all.

`level`, `usefulness`, `isImplemented` and `evidence` are deliberately unused:
they are assessment state, not control text.
"""
from __future__ import annotations

import hashlib
import logging
import zipfile
from io import BytesIO
from typing import Any, ClassVar, Final

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "dsomm_data.zip"
# The archive root carries the pinned commit sha, so the member is located by
# suffix rather than by a path that changes on every re-pin.
MODEL_SUFFIX: Final[str] = "generated/model.yaml"

SOURCE_SHA256: Final[str] = (
    "a6d773129591d59e7c0757651142c39a341400333f40c1555fb2481ae89f2c66"
)

# Statement fields, in the order they are joined. See the module docstring for
# why `description` cannot be the only one.
STATEMENT_FIELDS: Final[tuple[str, ...]] = ("description", "risk", "measure")


class DsommParser(BaseParser):
    framework_id: ClassVar[str] = "dsomm"
    # Matches canonical_framework("DevSecOps Maturity Model (DSOMM)"), which
    # FRAMEWORK_NAME_ALIASES already maps to "DSOMM". No new alias is needed.
    framework_name: ClassVar[str] = "DSOMM"
    version: ClassVar[str] = "4.3.1"
    source_url: ClassVar[str] = (
        "https://github.com/devsecopsmaturitymodel/DevSecOps-MaturityModel-data"
    )
    mapping_unit_level: ClassVar[str] = "activity"
    # 194 leaf activities in the pinned archive. [measured]
    expected_count: ClassVar[int] = 194
    fetched_date: ClassVar[str] = "2026-08-15"
    # 192 of 194 clear the 60-character honest-prose bar and differ from their
    # title. The two that do not are single-sentence measures with long names.
    # [measured] The floor sits just under, close enough that a regression
    # toward name-only extraction still trips it.
    min_prose_fraction: ClassVar[float] = 0.98
    # Class-level so a fixture-backed test declares its own digest instead of
    # the real gate being widened to accept two archives.
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        model = self._read_model(payload)
        controls = self.activities_to_controls(model)
        logger.info(
            "%s: %d activities across %d dimensions",
            self.framework_id, len(controls),
            len({c.parent_name for c in controls}),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual == self.expected_sha256:
            return
        raise ValueError(
            f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, not "
            f"the pinned {self.expected_sha256}. expected_count, "
            f"min_prose_fraction and the join floor were all measured against "
            f"the pinned bytes. Re-measure before moving the pin, and move it "
            f"in scripts/fetch_frameworks.py at the same time."
        )

    def _read_model(self, payload: bytes) -> dict[str, Any]:
        """The second YAML document of the generated model file.

        Raises:
            ValueError: If the member is absent or the stream is not two
                documents with a mapping second.
        """
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.endswith(MODEL_SUFFIX)]
            if len(names) != 1:
                raise ValueError(
                    f"{self.framework_id}: expected exactly one "
                    f"{MODEL_SUFFIX} in {ARCHIVE_NAME}, found {names}. The "
                    f"generated file is what flattens 26 per-subdimension "
                    f"YAMLs into one document; without it the join level is "
                    f"guesswork."
                )
            raw = archive.read(names[0]).decode("utf-8")

        documents = list(yaml.safe_load_all(raw))
        if len(documents) != 2 or not isinstance(documents[1], dict):
            raise ValueError(
                f"{self.framework_id}: {MODEL_SUFFIX} is not a meta document "
                f"followed by a model mapping (got {len(documents)} "
                f"document(s)). The layout changed."
            )
        return documents[1]

    @classmethod
    def activities_to_controls(cls, model: dict[str, Any]) -> list[Control]:
        """One Control per leaf activity, in source order.

        Raises:
            ValueError: On an activity with no uuid, or with no statement text
                in any of STATEMENT_FIELDS.
        """
        controls: list[Control] = []
        for dimension, sub_dimensions in model.items():
            for sub_dimension, activities in sub_dimensions.items():
                for name, body in activities.items():
                    controls.append(
                        cls._to_control(dimension, sub_dimension, name, body)
                    )
        return controls

    @classmethod
    def _to_control(
        cls, dimension: str, sub_dimension: str, name: str, body: dict[str, Any],
    ) -> Control:
        uuid = str(body.get("uuid") or "").strip()
        if not uuid:
            raise ValueError(
                f"dsomm: activity {name!r} under {dimension}/{sub_dimension} "
                f"has no uuid. The uuid is what OpenCRE links against, so an "
                f"activity without one cannot be joined and must not be "
                f"emitted as though it could."
            )
        statement = cls._statement(body)
        if not statement:
            raise ValueError(
                f"dsomm: activity {name!r} (uuid {uuid}) has no text in any "
                f"of {STATEMENT_FIELDS}. All 194 activities in the pinned "
                f"archive have risk and measure, so an empty one means the "
                f"schema changed."
            )
        return Control(
            control_id=uuid,
            title=name.strip(),
            description=statement,
            hierarchy_level="activity",
            parent_id=sub_dimension,
            parent_name=dimension,
            metadata={"sub_dimension": sub_dimension, "dimension": dimension},
        )

    @staticmethod
    def _statement(body: dict[str, Any]) -> str:
        parts = [
            str(body.get(field) or "").strip() for field in STATEMENT_FIELDS
        ]
        return "\n\n".join(part for part in parts if part)


def main() -> None:
    DsommParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_dsomm.py -q
mypy parsers/parse_dsomm.py --strict
```

Expected: PASS, no mypy errors.

- [ ] **Step 5: Run the parser against the real source**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" parsers/parse_dsomm.py
```

Expected: `dsomm: 194 activities across 5 dimensions`, then
`dsomm: honest prose fraction 0.990 (floor 0.980)` and
`Wrote 194 controls to data/processed/frameworks/dsomm.json`. **[measured]**

- [ ] **Step 6: Rebuild the merged corpus and run the instrument**

```bash
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(dsomm|framework|TOTAL)"
```

Expected row, derived before the parser was written **[derived from measured]**:

```
dsomm                        214     0  213     1     182  1.17     1    0     1     0 0.9953
```

Accept only if all five hold:
- `resolution_rate >= 0.99` (the derived floor)
- `distinct_anchors >= 182`, up from 0 resolved and from 18 title anchors
- `links_per_anchor <= 1.20`, down from 11.89 on the fallback anchors
- `nested_anchors == 0`
- `wrong_anchor_risk == 0`

If `by_title` is not 0, the parser is emitting activity names that collide with
a sub-dimension name and the join is going through the wrong channel.

- [ ] **Step 7: Commit**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -c "
from tract.config import OVERLAY_FRAMEWORK_IDS
assert 'dsomm' in OVERLAY_FRAMEWORK_IDS, 'Contract Rule 3 has not landed; stop'
print('overlay routing: on')
"
git check-ignore -v data/processed/frameworks/dsomm.json \
  || { echo "NOT IGNORED. the Rule 3 .gitignore lines are missing; stop"; exit 1; }
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py \
                              tests/test_framework_licenses.py -q
git add parsers/parse_dsomm.py tests/test_parse_dsomm.py
git commit -m "feat: join DSOMM on the activity uuid instead of its sub-dimension"
```

---

### Task 4: SAMM, 30 streams, and the `full_text` trap

**Invalidates:** `data/processed/frameworks/samm.json` and the overlay corpus built from it. SAMM's 30 anchors keep their count and change their text, so the anchor column will not move and the text-quality columns will.

SAMM is the cleanest source in the batch and it is placed here because it makes
one contract fact concrete before a harder parser has to reason about it.

`model/streams/<P>-<S>.yml` is the join level: 30 files whose stem matches
`section_id` for 30 of 30 curated links, and whose `name` matches `section_name`
for 30 of 30. **[measured]** Ceiling **30/30 = 1.0000** **[derived]**, floor
**1.00**. At the ceiling a floor of 1.00 is the right number: anything less
means a link that resolves today stopped resolving.

**The composition decision is measured, not assumed.** Four candidate
statements, over the 30 streams **[measured]**:

| statement | min | median | max | over `MAX_ANCHOR_CHARS` (2150) | over `DESCRIPTION_MAX_LENGTH` |
|---|---|---|---|---|---|
| stream `description` only | 110 | 185 | 577 | 0 | 0 |
| description + 3 × `shortDescription` | 347 | 465 | 986 | **0** | **0** |
| description + level-1 `longDescription` | 858 | 1283 | 2529 | 2 | 3 |
| description + 3 × `longDescription` | 2548 | 3558 | 6678 | **30** | **30** |

The last row is the trap. It reads as the richest option and it is the worst:
every one of the 30 descriptions exceeds `DESCRIPTION_MAX_LENGTH`, so
`_sanitize_control` writes the overflow into `full_text`, `ProseIndex` prefers
`full_text`, and all 30 anchors are then cut at 2,150 characters, a 100%
truncation rate on a framework that has none today. The second row is chosen:
real prose from all three maturity levels, no truncation, no implicit
`full_text`.

**The source-structures document is wrong about `level`.** It is a GUID, not an
integer. **[measured]** Maturity level comes from the activity **filename**
(`D-SA-1-A.yml` → level 1); every one of the 30 streams has exactly 3
activities and the filename stems partition cleanly. **[measured]**

**Files:**
- Create: `parsers/parse_samm.py`
- Create: `tests/test_parse_samm.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `SammParser` with `framework_id = "samm"`, `framework_name = "SAMM"`; `SammParser.build_controls(streams: dict[str, dict[str, Any]], activities: dict[str, list[tuple[int, dict[str, Any]]]]) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_samm.py: create

"""SAMM joins at the stream, and the statement must not overflow the budget.

Measured: stream description plus the three activities' longDescription runs
2,548 to 6,678 characters, so all 30 descriptions exceed DESCRIPTION_MAX_LENGTH,
_sanitize_control moves the overflow into full_text, and ProseIndex then
anchors on a text prepare_anchor cuts at 2,150. shortDescription keeps every
statement inside both budgets.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import yaml

from parsers.parse_samm import SammParser
from tract.config import DESCRIPTION_MAX_LENGTH, MAX_ANCHOR_CHARS

STREAM = {
    "practice": "4753e55e943c4d418303bf90d599c6b1",
    "id": "253b012094cf4e0988e08fd22609227d",
    "name": "Architecture Design",
    "letter": "A",
    "description": "The design of a software architecture significantly "
                   "affects the security posture of the software.",
    "order": 1,
    "type": "Stream",
}


def _activity(level: int) -> dict[str, object]:
    return {
        "stream": "253b012094cf4e0988e08fd22609227d",
        "level": "a11b78917dec4cfdad983cf6d1d17b61",
        "title": f"Level {level} activity",
        "shortDescription": f"Teams apply level {level} design practices "
                            f"during architecture review sessions.",
        "longDescription": "x" * 3000,
        "type": "Activity",
    }


@pytest.fixture()
def parser(tmp_path: Path) -> SammParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("core-abc/model/streams/D-SA-A.yml",
                         yaml.safe_dump(STREAM))
        for level in (1, 2, 3):
            archive.writestr(f"core-abc/model/activities/D-SA-{level}-A.yml",
                             yaml.safe_dump(_activity(level)))
    (raw / "samm_core.zip").write_bytes(payload.getvalue())

    instance = SammParser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_count = 1
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_control_id_is_the_filename_stem(self, parser: SammParser) -> None:
        assert [c.control_id for c in parser.parse()] == ["D-SA-A"]

    def test_title_is_the_stream_name(self, parser: SammParser) -> None:
        assert parser.parse()[0].title == "Architecture Design"

    def test_statement_carries_all_three_maturity_levels(
        self, parser: SammParser,
    ) -> None:
        text = parser.parse()[0].description
        for level in (1, 2, 3):
            assert f"level {level} design practices" in text

    def test_statement_uses_short_not_long_descriptions(
        self, parser: SammParser,
    ) -> None:
        control = parser.parse()[0]
        assert "x" * 100 not in control.description
        assert len(control.description) <= DESCRIPTION_MAX_LENGTH
        assert len(control.description) <= MAX_ANCHOR_CHARS

    def test_full_text_is_left_unset_so_the_anchor_is_the_statement(
        self, parser: SammParser,
    ) -> None:
        assert parser.parse()[0].full_text is None


class TestMissingActivities:
    def test_a_stream_with_no_activities_is_refused(
        self, parser: SammParser, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "bare"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("core-abc/model/streams/D-SA-A.yml",
                             yaml.safe_dump(STREAM))
        (raw / "samm_core.zip").write_bytes(payload.getvalue())
        bare = SammParser(raw_dir=raw, output_dir=tmp_path / "out")
        bare.expected_sha256 = None
        with pytest.raises(ValueError, match="no activities"):
            bare.parse()


class TestRun:
    def test_run_writes(self, parser: SammParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 1
        assert [s.path for s in output.source_files] == ["samm_core.zip"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_samm.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'parsers.parse_samm'`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_samm.py: create

"""Parser for OWASP SAMM, at the stream level.

The repository has three granularities and only one of them is what OpenCRE
links against. `model/security_practices/` holds the 15 practices,
`model/streams/` holds the 30 practice-and-stream pairs, and
`model/activities/` holds the 90 (practice, stream, level) triples. Every one
of the 30 curated links carries a `section_id` equal to a stream filename stem
and a `section_name` equal to that stream's `name`. Activity filenames
(`D-SA-1-A`) match no section_id at all.

The statement is the stream's own `description` plus the three activities'
`shortDescription`, in maturity-level order. Measured over the 30 streams that
lands between 347 and 986 characters. Using `longDescription` instead runs
2,548 to 6,678, which puts every description over DESCRIPTION_MAX_LENGTH, makes
BaseParser._sanitize_control write the overflow into full_text, and hands
ProseIndex -- which prefers full_text -- an anchor that prepare_anchor then
truncates. Richer text that the encoder never reads is not richer text.

Maturity level comes from the activity filename. The `level` field is a GUID
into SAMM's own model, not an ordinal, and sorting on it produces an arbitrary
order that changes with the release.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from typing import Any, ClassVar, Final

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "samm_core.zip"
SOURCE_SHA256: Final[str] = (
    "16eb608b70bad3039b14ca4e3f300893d29bbc4205c737ac07fcbdfb4f7493a6"
)

STREAM_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/model/streams/([A-Z]-[A-Z]{2}-[AB])\.yml$"
)
# D-SA-1-A.yml: practice code, maturity level, stream letter. The level is read
# here because the `level` FIELD is a GUID.
ACTIVITY_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/model/activities/([A-Z]-[A-Z]{2})-(\d)-([AB])\.yml$"
)

EXPECTED_LEVELS: Final[tuple[int, ...]] = (1, 2, 3)


class SammParser(BaseParser):
    framework_id: ClassVar[str] = "samm"
    framework_name: ClassVar[str] = "SAMM"
    version: ClassVar[str] = "2.0"
    source_url: ClassVar[str] = "https://owaspsamm.org/model/"
    mapping_unit_level: ClassVar[str] = "stream"
    # 15 practices x 2 streams. [measured]
    expected_count: ClassVar[int] = 30
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every statement is at least 347 characters and none equals its name.
    # [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        streams, activities = self._read_members(payload)
        controls = self.build_controls(streams, activities)
        logger.info(
            "%s: %d streams, statement length %d..%d characters",
            self.framework_id, len(controls),
            min(len(c.description) for c in controls),
            max(len(c.description) for c in controls),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}. The statement-length "
                f"measurements that chose shortDescription over "
                f"longDescription were taken against the pinned bytes."
            )

    def _read_members(
        self, payload: bytes,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[tuple[int, dict[str, Any]]]]]:
        """Streams keyed by stem, activities keyed by stem and level."""
        streams: dict[str, dict[str, Any]] = {}
        activities: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                stream = STREAM_MEMBER.search(name)
                if stream:
                    streams[stream.group(1)] = yaml.safe_load(archive.read(name))
                    continue
                activity = ACTIVITY_MEMBER.search(name)
                if activity:
                    key = f"{activity.group(1)}-{activity.group(3)}"
                    activities[key].append(
                        (int(activity.group(2)), yaml.safe_load(archive.read(name)))
                    )
        if not streams:
            raise ValueError(
                f"{self.framework_id}: no model/streams/*.yml members in "
                f"{ARCHIVE_NAME}. The stream stem is the only identifier "
                f"OpenCRE links against; without it there is no join."
            )
        return streams, dict(activities)

    @classmethod
    def build_controls(
        cls,
        streams: dict[str, dict[str, Any]],
        activities: dict[str, list[tuple[int, dict[str, Any]]]],
    ) -> list[Control]:
        """One Control per stream, statement built from its three activities.

        Raises:
            ValueError: If a stream has no activities, or its levels are not
                exactly EXPECTED_LEVELS.
        """
        controls: list[Control] = []
        for stem in sorted(streams):
            stream = streams[stem]
            owned = sorted(activities.get(stem, []))
            if not owned:
                raise ValueError(
                    f"samm: stream {stem} has no activities. The statement is "
                    f"built from them, so an empty list would emit a control "
                    f"carrying only the two-sentence stream description."
                )
            levels = tuple(level for level, _ in owned)
            if levels != EXPECTED_LEVELS:
                raise ValueError(
                    f"samm: stream {stem} has maturity levels {levels}, "
                    f"expected {EXPECTED_LEVELS}. A missing level means the "
                    f"statement is short by a third and nothing else would "
                    f"say so."
                )
            controls.append(Control(
                control_id=stem,
                title=str(stream.get("name") or "").strip(),
                description=cls._statement(stream, owned),
                hierarchy_level="stream",
                parent_id=str(stream.get("practice") or "").strip() or None,
                metadata={"stream_letter": str(stream.get("letter") or "")},
            ))
        return controls

    @staticmethod
    def _statement(
        stream: dict[str, Any], owned: list[tuple[int, dict[str, Any]]],
    ) -> str:
        parts = [str(stream.get("description") or "").strip()]
        parts += [
            str(activity.get("shortDescription") or "").strip()
            for _, activity in owned
        ]
        return "\n\n".join(part for part in parts if part)


def main() -> None:
    SammParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_samm.py -q
mypy parsers/parse_samm.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_samm.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(samm|framework)"
```

Expected: `samm: 30 streams, statement length 347..986 characters` **[measured]**,
then:

```
samm                          30    30    0     0      30  1.00     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000`, `distinct_anchors == 30`,
`truncated == 0`, `nested_anchors == 0`. `by_title` of 30 is expected here and
is correct: every `section_name` is the stream's own name and no two streams
share one.

- [ ] **Step 6: Commit**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# Ruling R10: this framework is CC BY-SA and is TRACKED. Seven other CC BY-SA
# frameworks were already tracked and published, so treating this one
# differently was defensible on no reading. LICENSES/, the NOTICE modification
# statement and one licence declaration across the published artifacts discharge
# the attribution and notice obligations. Assert it is NOT in the overlay, so a
# future tier change that silently recaptures it fails here.
"$PY" -c "
from tract.config import OVERLAY_FRAMEWORK_IDS
assert 'samm' not in OVERLAY_FRAMEWORK_IDS, 'tier changed under this task; stop'
print('routing: tracked')
"
git check-ignore -q data/processed/frameworks/samm.json \
  && { echo "samm.json is ignored but R10 tracks it; stop"; exit 1; }
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py \
                              tests/test_framework_licenses.py -q
git add parsers/parse_samm.py tests/test_parse_samm.py \
        data/processed/frameworks/samm.json data/processed/all_controls.json
git commit -m "feat: parse SAMM at the stream, with a statement that fits the encoder"
```

---

### Task 5: OWASP Top 10 2021, 10 categories

17 curated links over 10 `section_id` values `A01`..`A10`, all present. **[measured]**
Ceiling **17/17 = 1.0000** **[derived]**, floor **1.00**.

**The source-structures document is wrong about the file count.** It says three `A00_2021-*.md` files are
meta. There is exactly one `A00` and one `A11`: twelve files match `A\d\d_2021-*.md` in `2021/docs/en/`, of
which `A01` through `A10` are the categories, `A00` is *How to start an AppSec Program* and `A11` is
*Next Steps*. **[measured]** `A00`'s H1 carries no `A0N:2021` prefix at all, so a parser keyed on the H1
pattern excludes it automatically. `A11`'s H1 does carry one, so an explicit id allowlist excludes it.

**`by_title` is 11 of 17, not 0. Plan v2 asserted 0 and would have halted a healthy run.** Seven of the ten
source H1 titles match a link's `section_name` exactly, and `ProseIndex.lookup` lowercases both sides and
tries the title before the id. The three that diverge are `A01` (`Broken Access Controls` plural against the
source's singular), `A09` (`Logging and Monitoring Failures` against `Security Logging and Monitoring
Failures`) and `A10` (`Server Side Request Forgery (SSRF)` against `Server-Side`). Their link counts are 2,
3 and 1, so 6 links fall to the id channel and 11 resolve by title. **[measured]** No alias is needed and
none is added: every one of the 11 title hits lands on the control its own `section_id` also names, so
`wrong_anchor_risk` stays 0. **[measured]**

**Every one of the 17 links resolves onto a truncated anchor.** `ProseIndex` prefers `full_text`
unconditionally, `full_text` here is the whole category entry, and the ten entries run 2,263 to 9,706
characters against `MAX_ANCHOR_CHARS` of 2,150. **[measured]** So the anchor the encoder reads is the first
2,150 characters of the entry, which opens with `## Overview` and reaches into `## Description`. The
docstring below states that plainly rather than claiming the Overview is excluded: it is excluded from
`description`, and `description` is not what the index selects.

The archive is 196 MB and 199 of its members are markdown for other years and languages. **[measured]** Only
the twelve `2021/docs/en/A*` members are read.

**Licensing.** `owasp_top10_2021` is CC-BY-SA-4.0 and sits in `CONDITIONAL_FRAMEWORK_IDS` under Contract
Rule 3, so `data/processed/frameworks/owasp_top10_2021.json` routes to the gitignored licensed overlay and
is never staged. The assignments stay tracked.

**Files:**
- Create: `parsers/parse_owasp_top10_2021.py`
- Create: `tests/test_parse_owasp_top10_2021.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `tract.config.REMEDIATION_HEADINGS`.
- Produces: `OwaspTop102021Parser` with `framework_id = "owasp_top10_2021"`, `framework_name = "OWASP Top 10 2021"`; `OwaspTop102021Parser.control_from_markdown(text: str) -> Control`.

**Invalidates:**
- `data/processed/licensed/all_controls.json` and its sha256, so every `corpus_sha256` recorded before this task.
- `data/processed/stopwords.json`, which Task 15 regenerates from the rebuilt corpus.
- `results/corpus/before_8cf44b3.json` as a *current* reading. It stays valid as the BEFORE baseline and must not be regenerated.
- Task 14's `hub_links_training.jsonl` for the 17 `owasp_top10_2021` rows, which move from a `section_name` fallback anchor onto a category entry.

- [ ] **Step 1: Write the failing test**

The fixture carries all ten categories. Plan v2's fixture carried one, and `parse()` raises when the found
id tuple is not `CATEGORY_IDS`, so four of that version's own tests failed against its own implementation.
**[measured: `ValueError: expected categories ['A01'..'A10'], found ['A01']`]** That completeness check is
the point of the parser, so the fixture satisfies it and a separate guard test removes one category to prove
the check fires.

```python
# tests/test_parse_owasp_top10_2021.py: create

"""Ten categories, and neither A00 nor A11 is one of them.

The fixture carries all ten because parse() refuses a short list, which is the
whole reason the completeness check exists. TestGuards removes one to prove the
refusal fires.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from parsers.parse_owasp_top10_2021 import OwaspTop102021Parser

TITLES: dict[str, str] = {
    "A01": "Broken Access Control",
    "A02": "Cryptographic Failures",
    "A03": "Injection",
    "A04": "Insecure Design",
    "A05": "Security Misconfiguration",
    "A06": "Vulnerable and Outdated Components",
    "A07": "Identification and Authentication Failures",
    "A08": "Software and Data Integrity Failures",
    "A09": "Security Logging and Monitoring Failures",
    "A10": "Server-Side Request Forgery (SSRF)",
}

BODY = """
## Factors

| CWEs Mapped | Max Incidence Rate |
|---|---|
| 34 | 55.97% |

## Overview

Moving up from the fifth position, 94% of applications were tested for some
form of this weakness.

## Description

{title} covers the case where a system does not enforce the policy it claims
to enforce, and failures typically lead to unauthorized information
disclosure, modification, or destruction of data.

## How to Prevent

Enforcement is only effective in trusted server-side code.

## References

- OWASP Proactive Controls
"""

META = """# How to start an AppSec Program with the OWASP Top 10

Previously, the OWASP Top 10 was never designed to be the basis of anything.
"""

NEXT_STEPS = """# A11:2021 – Next Steps

The Top 10 is not the end of the journey.

## Description

There is more to application security than ten risks.
"""


def _category(code: str, title: str) -> str:
    return f"# {code}:2021 – {title}\n" + BODY.format(title=title)


def _archive(codes: tuple[str, ...]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for code in codes:
            archive.writestr(
                f"Top10-abc/2021/docs/en/{code}_2021-Entry.md",
                _category(code, TITLES[code]),
            )
        archive.writestr("Top10-abc/2021/docs/en/A00_2021-How_to_start.md", META)
        archive.writestr("Top10-abc/2021/docs/en/A11_2021-Next_Steps.md", NEXT_STEPS)
        archive.writestr(
            "Top10-abc/2017/docs/en/A01_2017-Injection.md",
            _category("A01", TITLES["A01"]),
        )
        archive.writestr(
            "Top10-abc/2021/docs/fr/A01_2021-Broken.md",
            _category("A01", TITLES["A01"]),
        )
    return payload.getvalue()


@pytest.fixture()
def parser(tmp_path: Path) -> OwaspTop102021Parser:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "owasp_top10_2021.zip").write_bytes(_archive(tuple(TITLES)))
    instance = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_only_the_ten_english_2021_categories_are_read(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert [c.control_id for c in parser.parse()] == list(TITLES)

    def test_title_drops_the_code_and_the_en_dash(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        assert parser.parse()[0].title == "Broken Access Control"

    def test_description_is_the_description_section(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        text = parser.parse()[0].description
        assert text.startswith("Broken Access Control covers the case")
        assert "Moving up from the fifth position" not in text
        assert "trusted server-side code" not in text
        assert "CWEs Mapped" not in text

    def test_full_text_carries_the_whole_entry(
        self, parser: OwaspTop102021Parser,
    ) -> None:
        control = parser.parse()[0]
        assert control.full_text is not None
        assert "trusted server-side code" in control.full_text
        assert "Moving up from the fifth position" in control.full_text


class TestGuards:
    def test_a_short_list_is_refused(
        self, parser: OwaspTop102021Parser, tmp_path: Path,
    ) -> None:
        """The band would accept 9 of 10. The exact tuple does not.

        COUNT_TOLERANCE is 0.10 and abs(9 - 10) / 10 is 0.1, so
        _check_expected_count would pass a parser that lost a category. This
        assertion is the one that beats the band.
        """
        raw = tmp_path / "short"
        raw.mkdir()
        codes = tuple(c for c in TITLES if c != "A07")
        (raw / "owasp_top10_2021.zip").write_bytes(_archive(codes))
        short = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
        short.expected_sha256 = None
        with pytest.raises(ValueError, match="expected categories"):
            short.parse()

    def test_a_missing_description_section_is_refused(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "broken"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "Top10-abc/2021/docs/en/A01_2021-X.md",
                "# A01:2021 – X\n\n## Overview\n\nNo body.\n",
            )
        (raw / "owasp_top10_2021.zip").write_bytes(payload.getvalue())
        broken = OwaspTop102021Parser(raw_dir=raw, output_dir=tmp_path / "out")
        broken.expected_sha256 = None
        with pytest.raises(ValueError, match="no '## Description' section"):
            broken.parse()


class TestRun:
    def test_run_writes(
        self, parser: OwaspTop102021Parser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 10
        assert [s.path for s in output.source_files] == ["owasp_top10_2021.zip"]

    def test_the_anchor_is_full_text_and_it_exceeds_the_budget(
        self, parser: OwaspTop102021Parser, tmp_path: Path,
    ) -> None:
        """ProseIndex prefers full_text, so full_text is what the model reads.

        Measured on the real archive: all ten entries run 2,263 to 9,706
        characters against MAX_ANCHOR_CHARS of 2,150, so all 17 curated links
        land on a truncated anchor. This test states the contract on the
        fixture so a later edit that moves the whole entry out of full_text
        shows up here rather than as a silent change of anchor.
        """
        (tmp_path / "out").mkdir()
        output = parser.run()
        for control in output.controls:
            assert control.full_text is not None
            assert len(control.full_text) > len(control.description)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_owasp_top10_2021.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_owasp_top10_2021.py: create

"""Parser for the OWASP Top 10 2021.

The archive carries every Top 10 edition from 2003 to 2025 in every
translation, 199 markdown files and 196 MB. Only `2021/docs/en/A0N_2021-*.md`
and `2021/docs/en/A10_2021-*.md` are read, and the member list is filtered by
name so the other 187 files are never decompressed.

Twelve files match the A-prefix pattern. `A00` is *How to start an AppSec
Program* and `A11` is *Next Steps*; neither is a category and neither carries a
curated link. `A00` is excluded by its H1, which has no `A0N:2021` code; `A11`
is excluded by CATEGORY_IDS, because its H1 does carry one.

`description` is the `## Description` section, which drops `## Overview`
(release commentary about where the category moved in the rankings) and the two
remediation headings in tract.config.REMEDIATION_HEADINGS that this framework is
the original reason for.

`full_text` is the whole entry, and `full_text` is the anchor. ProseIndex
prefers it over description unconditionally, so the Overview and the
How-to-Prevent text DO reach the encoder, cut at MAX_ANCHOR_CHARS. Measured on
the pinned archive the ten entries run 2,263 to 9,706 characters against a
2,150-character budget, so all 17 curated links resolve onto a truncated anchor
and the corpus report records `truncated == 17`. That is the honest reading of
this framework: the anchor is the opening 2,150 characters of the category page.
The narrower `description` stays on the record for a reader and for any consumer
that asks for description-only text.

The found id tuple must equal CATEGORY_IDS exactly. COUNT_TOLERANCE is 10%, so
_check_expected_count accepts 9 categories of 10, and those ten carry 1.7
curated links each. The tuple check is what beats the band.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from io import BytesIO
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "owasp_top10_2021.zip"
SOURCE_SHA256: Final[str] = (
    "7f4747a7d7958d58ae3a4c7f7329740b9363c4788655bc3f28da8fdbedf48b5d"
)

MEMBER: Final[re.Pattern[str]] = re.compile(r"/2021/docs/en/A\d\d_2021-.*\.md$")
# The en dash is the source's own separator. A hyphen and an em dash are
# accepted too, so a typographic edit upstream does not silently drop a
# category.
HEADING: Final[re.Pattern[str]] = re.compile(
    r"^#\s+(A\d\d):2021\s*[–—-]\s*(.+?)\s*$", re.M
)
DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s|\Z)", re.M | re.S
)

CATEGORY_IDS: Final[tuple[str, ...]] = tuple(
    f"A{n:02d}" for n in range(1, 11)
)


class OwaspTop102021Parser(BaseParser):
    framework_id: ClassVar[str] = "owasp_top10_2021"
    framework_name: ClassVar[str] = "OWASP Top 10 2021"
    version: ClassVar[str] = "2021"
    source_url: ClassVar[str] = "https://owasp.org/Top10/"
    mapping_unit_level: ClassVar[str] = "category"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    # All ten descriptions run 581 to 1,998 characters and none equals its
    # title, so the attainable value is exactly 1.0 and a floor of 1.0 fires
    # the moment any category loses its Description section. [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        controls: list[Control] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                text = archive.read(name).decode("utf-8")
                heading = HEADING.search(text)
                if heading is None:
                    logger.info(
                        "%s: %s has no A0N:2021 heading, so it is front "
                        "matter rather than a category", self.framework_id,
                        name.rsplit("/", 1)[-1],
                    )
                    continue
                if heading.group(1) not in CATEGORY_IDS:
                    logger.info(
                        "%s: skipping %s, which is numbered outside the ten "
                        "categories", self.framework_id, heading.group(1),
                    )
                    continue
                controls.append(self.control_from_markdown(text))
        found = tuple(c.control_id for c in controls)
        if found != CATEGORY_IDS:
            raise ValueError(
                f"{self.framework_id}: expected categories "
                f"{list(CATEGORY_IDS)}, found {list(found)}. A short list "
                f"would ship a partial Top 10, and at 9 of 10 the count "
                f"deviation is 10.0% against a COUNT_TOLERANCE of 10%, so "
                f"_check_expected_count would accept it."
            )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}."
            )

    @classmethod
    def control_from_markdown(cls, text: str) -> Control:
        """One category from one markdown file.

        Raises:
            ValueError: If the file has no heading or no Description section.
        """
        heading = HEADING.search(text)
        if heading is None:
            raise ValueError(
                "owasp_top10_2021: markdown with no 'A0N:2021 - Title' "
                "heading reached control_from_markdown."
            )
        body = DESCRIPTION.search(text)
        if body is None:
            raise ValueError(
                f"owasp_top10_2021: {heading.group(1)} has no '## Description' "
                f"section. Unhandled, its statement would fall back to the "
                f"Overview, which is commentary about the survey rather than "
                f"about the risk."
            )
        return Control(
            control_id=heading.group(1),
            title=heading.group(2).split("![", 1)[0].strip(),
            description=body.group(1).strip(),
            full_text=text[heading.end():].strip(),
            hierarchy_level="category",
        )


def main() -> None:
    OwaspTop102021Parser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_owasp_top10_2021.py -q
"$PY" -m mypy parsers/parse_owasp_top10_2021.py --strict
```

Both must pass. `mypy --strict` on this file is clean as written. **[measured]**

- [ ] **Step 5: Run against the real source and check the join**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" parsers/parse_owasp_top10_2021.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py --framework owasp_top10_2021 --json
```

Accept only on this exact set. Every value is **[measured]** on the pinned archive and the curated link
file, before the parser is written.

| field | value | how it can fail in each direction |
|---|---|---|
| `links` | 17 | fewer means the link file changed |
| `by_title` | **11** | 7 of the 10 source H1 titles equal a link `section_name`, and those seven carry 11 links. Below 11 means a title stopped matching. Above 11 means `A01`, `A09` or `A10` gained an alias this task did not add |
| `by_id` | 6 | the `A01`, `A09` and `A10` links, 2 + 3 + 1 |
| `unresolved` | 0 | attainable maximum is 17 |
| `distinct_anchors` | 10 | one per category. 9 or fewer means two entries truncated to the same 2,150 characters |
| `distinct_anchors_pre_truncation` | 10 | equal to `distinct_anchors`, so truncation collapses nothing |
| `fallback_anchors` (BEFORE) | 10 | orchestrator measured. The anchor count does not move. The gain is text |
| `links_per_anchor` | 1.70 | OpenCRE links three categories twice |
| `truncated` | **17** | all ten entries exceed `MAX_ANCHOR_CHARS`. A 0 here means `full_text` stopped carrying the whole entry |
| `nested_anchors` | 0 | containment definition, Contract Rule 1 |
| `contained_anchors` | 0 | strict-prefix definition |
| `dropped_by_prose_rule` | 0 | all ten carry `full_text`, so none can be dropped |
| `wrong_anchor_risk` | 0 | every title hit lands on the control its own `section_id` names. This is the one row in my range where 0 is the measured value rather than an unfailable assertion |
| `anchor_source_full_text` | 17 | |
| `anchor_source_description` / `_title` / `_synthetic` | 0 / 0 / 0 | |
| `distinct_hubs` | 16 | |
| `links_per_hub` | 1.06 | |
| `resolution_rate` | 1.0000 | floor 1.00 |

- [ ] **Step 6: Confirm the overlay routing, then commit**

`owasp_top10_2021` is CC-BY-SA-4.0 and belongs to `CONDITIONAL_FRAMEWORK_IDS` under Contract Rule 3, so its
processed file routes to the gitignored overlay and is not staged.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# Ruling R10: this framework is CC BY-SA and is TRACKED. Seven other CC BY-SA
# frameworks were already tracked and published, so treating this one
# differently was defensible on no reading. LICENSES/, the NOTICE modification
# statement and one licence declaration across the published artifacts discharge
# the attribution and notice obligations. Assert it is NOT in the overlay, so a
# future tier change that silently recaptures it fails here.
"$PY" -c "
from tract.config import OVERLAY_FRAMEWORK_IDS
assert 'owasp_top10_2021' not in OVERLAY_FRAMEWORK_IDS, 'tier changed under this task; stop'
print('routing: tracked')
"
git check-ignore -q data/processed/frameworks/owasp_top10_2021.json \
  && { echo "owasp_top10_2021.json is ignored but R10 tracks it; stop"; exit 1; }
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py \
                              tests/test_framework_licenses.py -q
git add parsers/parse_owasp_top10_2021.py tests/test_parse_owasp_top10_2021.py \
        data/processed/frameworks/owasp_top10_2021.json data/processed/all_controls.json
git commit -m "feat: parse the ten OWASP Top 10 2021 categories from the English 2021 tree"
```

---

<<<TASK 8>>>

---

### Task 6: OWASP Proactive Controls, 76 links that buy nothing until Task 14

**Invalidates:** `data/processed/frameworks/owasp_proactive_controls.json` and the overlay corpus built from it. The 76 links stay unresolved until Task 14 retires the link gates, so nothing downstream changes at this task; it is Task 14 that makes these anchors reachable.

Ten controls `C1`..`C10`, all present, all with a `## Description` section.
Ceiling **76/76 = 1.0000** **[derived]**, floor **1.00**.

**State the post-gate contribution honestly.** All 76 of this framework's
curated links are dropped before training by `assign_quality_tier`, twice over:
`owasp_proactive_controls` is in `PHASE1B_DROPPED_FRAMEWORKS`, and all 76 of its
`section_name` values are `C1`..`C10`, two or three characters, under
`PHASE1B_MIN_SECTION_TEXT_LENGTH = 10`. **[measured]** This parser contributes
**0 training links** until Task 14 retires both gates. The same is true of
`nist_800_63`'s 79. The previous plan's task table credited 155 links that buy
nothing while leaving the gates in place.

**Anchor count does not improve.** 76 links land on 10 anchors today and on 10
anchors after. The gain here is entirely in text: from a two-character label to
a paragraph. The instrument will show `links_per_anchor` unchanged at 7.60,
and that is the correct outcome for a source with ten mapping units and 76
links, not a regression.

Two decoys share the archive and the same `c<N>-` filename pattern:
`docs/archive/2018/c*.md` is the superseded v3 text and `v3/*` is 24 MB of
binary exports. **[measured]** The member filter anchors on
`docs/the-top-10/`.

**Files:**
- Create: `parsers/parse_owasp_proactive_controls.py`
- Create: `tests/test_parse_owasp_proactive_controls.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`.
- Produces: `OwaspProactiveControlsParser` with `framework_id = "owasp_proactive_controls"`, `framework_name = "OWASP Proactive Controls"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_owasp_proactive_controls.py: create

"""Ten controls, from docs/the-top-10 only.

The archive carries the same c<N>- filenames twice more: docs/archive/2018 is
the superseded v3 prose and v3/ is 24 MB of binary exports of it. A glob on
the filename pattern rather than on the directory picks up both.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from parsers.parse_owasp_proactive_controls import OwaspProactiveControlsParser

CURRENT = """# C1: Implement Access Control

## Description

Access Control (or Authorization) is allowing or denying specific requests from
a user, program, or process. With each access control decision, a given subject
requests access to a given object.

## Threats

Broken access control is the most commonly encountered risk.

## Implementation

### 1) Design access control thoroughly up front
"""

ARCHIVED = """# C1: Access Control (2018)

## Description

The superseded 2018 wording that must never reach the corpus.
"""


@pytest.fixture()
def parser(tmp_path: Path) -> OwaspProactiveControlsParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("pc-abc/docs/the-top-10/c1-accesscontrol.md", CURRENT)
        archive.writestr("pc-abc/docs/archive/2018/c1-accesscontrol.md", ARCHIVED)
    (raw / "owasp_proactive_controls.zip").write_bytes(payload.getvalue())

    instance = OwaspProactiveControlsParser(
        raw_dir=raw, output_dir=tmp_path / "out",
    )
    instance.expected_count = 1
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_reads_the_current_edition_only(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        controls = parser.parse()
        assert [c.control_id for c in controls] == ["C1"]
        assert "superseded" not in controls[0].description

    def test_title_is_the_heading_after_the_code(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        assert parser.parse()[0].title == "Implement Access Control"

    def test_description_stops_before_implementation(
        self, parser: OwaspProactiveControlsParser,
    ) -> None:
        text = parser.parse()[0].description
        assert text.startswith("Access Control (or Authorization)")
        assert "Design access control thoroughly" not in text


class TestRun:
    def test_run_writes(
        self, parser: OwaspProactiveControlsParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 1
        assert [s.path for s in output.source_files] == [
            "owasp_proactive_controls.zip",
        ]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_owasp_proactive_controls.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_owasp_proactive_controls.py: create

"""Parser for the OWASP Proactive Controls, current mkdocs edition.

Ten controls, C1 through C10, one markdown file each under
`docs/the-top-10/`. Every one of the 76 curated links carries a section_id of
`C1`..`C10` and a section_name that is the same two or three characters, so the
join runs entirely through the id channel and the title channel never fires.

This framework contributes zero training links today: it is named in
PHASE1B_DROPPED_FRAMEWORKS, and every one of its section_names is shorter than
PHASE1B_MIN_SECTION_TEXT_LENGTH. Both gates test a title. Retiring them so they
test the resolved anchor is a separate task; this parser is the thing that
makes retiring them safe, because until the prose exists there is nothing for
the gate to test.

Two decoys in the same archive use the same c<N>- filenames: docs/archive/2018
holds the superseded v3 prose and v3/ holds binary exports of it. The member
filter anchors on the directory, not on the filename.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from io import BytesIO
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "owasp_proactive_controls.zip"
SOURCE_SHA256: Final[str] = (
    "6db1aafd6ecd758f05cf6b4133ec7085eb95016ec41afc5f462b4683c603b19d"
)

MEMBER: Final[re.Pattern[str]] = re.compile(r"/docs/the-top-10/c\d+-.*\.md$")
HEADING: Final[re.Pattern[str]] = re.compile(r"^#\s+(C\d+):\s*(.+?)\s*$", re.M)
DESCRIPTION: Final[re.Pattern[str]] = re.compile(
    r"^##\s+Description\s*$(.*?)(?=^##\s|\Z)", re.M | re.S
)

CONTROL_IDS: Final[frozenset[str]] = frozenset(f"C{n}" for n in range(1, 11))


class OwaspProactiveControlsParser(BaseParser):
    framework_id: ClassVar[str] = "owasp_proactive_controls"
    framework_name: ClassVar[str] = "OWASP Proactive Controls"
    version: ClassVar[str] = "2024"
    source_url: ClassVar[str] = "https://top10proactive.owasp.org/"
    mapping_unit_level: ClassVar[str] = "control"
    expected_count: ClassVar[int] = 10
    fetched_date: ClassVar[str] = "2026-08-15"
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        controls: list[Control] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                controls.append(
                    self._control(archive.read(name).decode("utf-8"), name)
                )
        # Both directions, not just the superset check. `expected_count = 10`
        # against COUNT_TOLERANCE = 0.10 means nine controls gives a deviation
        # of exactly 0.1, and `0.1 <= 0.10` is True, so a short catalogue would
        # pass silently. These ten carry 7.6 links each, so one lost control is
        # roughly eight lost links. [measured: orchestrator, from COUNT_TOLERANCE
        # in tract/config.py and the 76 curated links]
        found = {c.control_id for c in controls}
        unknown = found - CONTROL_IDS
        if unknown:
            raise ValueError(
                f"{self.framework_id}: read control id(s) {sorted(unknown)} "
                f"outside C1..C10. Either the edition renumbered or a decoy "
                f"directory reached the member filter."
            )
        missing = CONTROL_IDS - found
        if missing:
            raise ValueError(
                f"{self.framework_id}: did not read control id(s) "
                f"{sorted(missing)}. The band around expected_count would let "
                f"a nine-control catalogue through without a word, so the "
                f"completeness check is the exact set, not the count."
            )
        return sorted(controls, key=lambda c: int(c.control_id[1:]))

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}."
            )

    @staticmethod
    def _control(text: str, member: str) -> Control:
        """One control from one markdown file.

        Raises:
            ValueError: If the heading or the Description section is absent.
        """
        heading = HEADING.search(text)
        if heading is None:
            raise ValueError(
                f"owasp_proactive_controls: {member} has no '# Cn: Title' "
                f"heading. The code in that heading is the only identifier "
                f"OpenCRE links against."
            )
        body = DESCRIPTION.search(text)
        if body is None:
            raise ValueError(
                f"owasp_proactive_controls: {heading.group(1)} in {member} "
                f"has no '## Description' section, so its statement would be "
                f"the Threats section, which describes the attack rather "
                f"than the control."
            )
        return Control(
            control_id=heading.group(1),
            title=heading.group(2).strip(),
            description=body.group(1).strip(),
            full_text=text[heading.end():].strip(),
            hierarchy_level="control",
        )


def main() -> None:
    OwaspProactiveControlsParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_owasp_proactive_controls.py -q
mypy parsers/parse_owasp_proactive_controls.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_owasp_proactive_controls.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(owasp_proactive|framework)"
```

Expected:

```
owasp_proactive_controls      76     0   76     0      10  7.60     0    0     0     0 1.0000
```

Accept only if `resolution_rate == 1.0000` and `distinct_anchors == 10`. A
`links_per_anchor` of 7.60 is the source's own shape, not a parser defect: ten
controls carry 76 links. Record it as a known concentration; it is the second
worst in the corpus after `owasp_cheat_sheets` at 7.98.

- [ ] **Step 6: Commit**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# Ruling R10: this framework is CC BY-SA and is TRACKED. Seven other CC BY-SA
# frameworks were already tracked and published, so treating this one
# differently was defensible on no reading. LICENSES/, the NOTICE modification
# statement and one licence declaration across the published artifacts discharge
# the attribution and notice obligations. Assert it is NOT in the overlay, so a
# future tier change that silently recaptures it fails here.
"$PY" -c "
from tract.config import OVERLAY_FRAMEWORK_IDS
assert 'owasp_proactive_controls' not in OVERLAY_FRAMEWORK_IDS, 'tier changed under this task; stop'
print('routing: tracked')
"
git check-ignore -q data/processed/frameworks/owasp_proactive_controls.json \
  && { echo "owasp_proactive_controls.json is ignored but R10 tracks it; stop"; exit 1; }
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py \
                              tests/test_framework_licenses.py -q
git add parsers/parse_owasp_proactive_controls.py \
        tests/test_parse_owasp_proactive_controls.py \
        data/processed/frameworks/owasp_proactive_controls.json \
        data/processed/all_controls.json
git commit -m "feat: parse the ten Proactive Controls from the current mkdocs tree"
```

---

### Task 7: WSTG, 115 tests, and nine links that can never resolve

**Invalidates:** `data/processed/frameworks/wstg.json`, the overlay corpus, and `data/processed/repair_audit/wstg.jsonl`. The merged control's text is assembled by this parser, so it carries `text_origin: synthetic` and shows up in the instrument's synthetic column rather than silently counting as prose.

118 curated links over 59 distinct `section_id` values. The archive's
`document/4-Web_Application_Security_Testing/` tree has 130 test markdown files
excluding category READMEs, of which **115** carry the two-row ID table and 14
do not, the 14 are sub-tests (`05.1-Testing_for_Oracle.md` and similar) that
share their parent's id. **[measured]** The structures document says 144 files;
that count included the category README files.

**Four of the 59 section_ids exist nowhere in the archive**: `WSTG-APPE-D`,
`WSTG-BUSL-$$`, `WSTG-INFO-##`, `WSTG-INPV-00`, carrying 2, 3, 1 and 3 links.
**[measured]** Grepping all 199 markdown members for each string returns zero
hits. Those 9 links are unresolvable by any parser.

**Ceiling: 109/118 = 0.92373** **[derived]**, floor **0.92**. The previous plan
set 0.96 against a stated maximum of 0.9322; both numbers are wrong and 0.96 was
unreachable. Note that 0.9237 is exactly the bogus-id limit: under the statement
rule below, no linked id fails the prose test, so the id gap is the whole gap.

**One id maps to two files.** `WSTG-INPV-13` is the ID table value in both
`13-Testing_for_Buffer_Overflow.md` and `13-Testing_for_Format_String_Injection.md`.
**[measured]** One curated link targets it. Ruling: emit one control whose title
joins both H1s with ` / ` and whose statement concatenates both bodies in
filename order. A merge across source files is a text-moving transform, so it
writes a repair-audit record naming both members.

**Statement rule.** The body from the end of the ID table to the first `##`
heading in `REMEDIATION_HEADINGS` ∪ {`How to Test`, `Related Test Cases`,
`Tools`}, with heading markers stripped. Measured over the 115 ids: 2 fail the
prose test (`WSTG-CONF-08`, `WSTG-CLNT-08`, neither linked), median 1,631
characters, 38 over `MAX_ANCHOR_CHARS` and 45 over `DESCRIPTION_MAX_LENGTH`.
**[measured]** Taking `## Summary` alone instead leaves 8 ids with no section at
all.

**Files:**
- Create: `parsers/parse_wstg.py`
- Create: `tests/test_parse_wstg.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `BaseParser.write_repair_audit`, `tract.config.REMEDIATION_HEADINGS`.
- Produces: `WstgParser` with `framework_id = "wstg"`, `framework_name = "WSTG"`; `WstgParser.build_controls(entries: list[tuple[str, str, str]]) -> tuple[list[Control], list[dict[str, object]]]` returning controls and audit records.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_wstg.py: create

"""WSTG joins on the ID table, and one id owns two files.

Measured: 115 of 130 test files carry an ID table, WSTG-INPV-13 appears in two
of them, and four section_ids in the curated links appear in none of the 199
markdown members in the archive.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from parsers.parse_wstg import WstgParser

BASE = "document/4-Web_Application_Security_Testing"


def _test_file(test_id: str, title: str, summary: str) -> str:
    return (
        f"# {title}\n\n"
        f"|ID          |\n|------------|\n|{test_id}|\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## How to Test\n\nRun the tool and read the output carefully.\n\n"
        f"## References\n\n- Somewhere\n"
    )


@pytest.fixture()
def parser(tmp_path: Path) -> WstgParser:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    long_summary = (
        "Search engines crawl and index billions of pages, and the index can "
        "retain content the owner has since removed from the live site."
    )
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            f"wstg-abc/{BASE}/01-Information_Gathering/01-Conduct_Search.md",
            _test_file("WSTG-INFO-01", "Conduct Search Engine Discovery",
                       long_summary),
        )
        archive.writestr(
            f"wstg-abc/{BASE}/07-Input_Validation_Testing/13-Testing_for_Buffer_Overflow.md",
            _test_file("WSTG-INPV-13", "Testing for Buffer Overflow",
                       "A buffer overflow overwrites adjacent memory and can "
                       "hand an attacker control of execution flow."),
        )
        archive.writestr(
            f"wstg-abc/{BASE}/07-Input_Validation_Testing/13-Testing_for_Format_String_Injection.md",
            _test_file("WSTG-INPV-13", "Testing for Format String Injection",
                       "A format string bug lets an attacker read and write "
                       "process memory through the conversion specifiers."),
        )
        archive.writestr(
            f"wstg-abc/{BASE}/07-Input_Validation_Testing/05.1-Testing_for_Oracle.md",
            "# Testing for Oracle\n\nA sub-test with no ID table of its own.\n",
        )
        archive.writestr(f"wstg-abc/{BASE}/01-Information_Gathering/README.md",
                         "# Information Gathering\n\nCategory intro.\n")
    (raw / "wstg.zip").write_bytes(payload.getvalue())

    instance = WstgParser(
        raw_dir=raw, output_dir=tmp_path / "out", audit_dir=tmp_path / "audit",
    )
    instance.expected_count = 2
    instance.expected_sha256 = None
    return instance


class TestParse:
    def test_only_files_with_an_id_table_become_controls(
        self, parser: WstgParser,
    ) -> None:
        assert sorted(c.control_id for c in parser.parse()) == [
            "WSTG-INFO-01", "WSTG-INPV-13",
        ]

    def test_title_is_the_h1_not_the_id(self, parser: WstgParser) -> None:
        first = next(c for c in parser.parse() if c.control_id == "WSTG-INFO-01")
        assert first.title == "Conduct Search Engine Discovery"

    def test_statement_stops_before_how_to_test(
        self, parser: WstgParser,
    ) -> None:
        first = next(c for c in parser.parse() if c.control_id == "WSTG-INFO-01")
        assert "Search engines crawl" in first.description
        assert "Run the tool" not in first.description

    def test_a_shared_id_merges_both_files(self, parser: WstgParser) -> None:
        shared = next(c for c in parser.parse() if c.control_id == "WSTG-INPV-13")
        assert shared.title == (
            "Testing for Buffer Overflow / Testing for Format String Injection"
        )
        assert "adjacent memory" in shared.description
        assert "conversion specifiers" in shared.description


class TestAudit:
    def test_the_merge_is_recorded(
        self, parser: WstgParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        parser.run()
        lines = (tmp_path / "audit" / "wstg.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == 1
        assert records[0]["control_id"] == "WSTG-INPV-13"
        assert len(records[0]["members"]) == 2

    def test_the_audit_file_is_written_even_with_no_merges(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "solo"
        raw.mkdir()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                f"wstg-abc/{BASE}/01-Information_Gathering/01-One.md",
                _test_file("WSTG-INFO-01", "One",
                           "A statement long enough to clear every prose bar "
                           "that this project applies to a description."),
            )
        (raw / "wstg.zip").write_bytes(payload.getvalue())
        solo = WstgParser(raw_dir=raw, output_dir=tmp_path / "out2",
                          audit_dir=tmp_path / "audit2")
        solo.expected_count = 1
        solo.expected_sha256 = None
        (tmp_path / "out2").mkdir()
        solo.run()
        assert (tmp_path / "audit2" / "wstg.jsonl").read_text(
            encoding="utf-8",
        ) == ""


class TestRun:
    def test_run_writes(self, parser: WstgParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 2
        assert [s.path for s in output.source_files] == ["wstg.zip"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_wstg.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_wstg.py: create

"""Parser for the OWASP Web Security Testing Guide.

The join key is the value in the two-row ID table under each test's H1, and it
is read from that table rather than derived from the path: the directory prefix
is a zero-padded number (`01-Information_Gathering`) while the id prefix is a
four-letter mnemonic (`WSTG-INFO-01`).

OpenCRE sets section_name equal to section_id for all 118 curated links, so
the link side carries no human title at all. The parser's title is the file's
H1, which is richer and which no link name spells, so the whole join runs
through the id channel and the title channel cannot misfire.

Nine of the 118 links can never resolve. Their section_ids -- WSTG-APPE-D,
WSTG-BUSL-$$, WSTG-INFO-## and WSTG-INPV-00 -- appear in none of the 199
markdown members of the pinned archive. That is an upstream extraction
artifact, not something a parser can fix, and it sets this framework's ceiling
at 109 of 118.

WSTG-INPV-13 is the ID table value of two files, Buffer Overflow and Format
String Injection. They are merged into one control, because the id is what the
link targets and emitting two controls with the same control_id would let
whichever came last silently win. The merge moves text across a file boundary
so it writes an audit record naming both members.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from typing import ClassVar, Final

from tract.config import REMEDIATION_HEADINGS
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "wstg.zip"
SOURCE_SHA256: Final[str] = (
    "e093f1648fbf4195f2a8fccac4f80315fb6b6281af85aa557edb34d0f9c58b33"
)

MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/document/4-Web_Application_Security_Testing/[^/]+/(?!README\.md$).*\.md$"
)
ID_TABLE: Final[re.Pattern[str]] = re.compile(
    r"^\|\s*(WSTG-[A-Z]+-\d+)\s*\|\s*$", re.M
)
H1: Final[re.Pattern[str]] = re.compile(r"^#\s+(.+?)\s*$", re.M)

# Where the statement ends. REMEDIATION_HEADINGS supplies Remediation,
# References and the rest; these three are WSTG's own procedural sections and
# describe how to run the test rather than what the test is for.
_EXTRA_CUTS: Final[tuple[str, ...]] = (
    "How to Test", "Related Test Cases", "Tools",
)
CUT: Final[re.Pattern[str]] = re.compile(
    r"^##\s+(?:"
    + "|".join(re.escape(h) for h in (*REMEDIATION_HEADINGS, *_EXTRA_CUTS))
    + r")\s*$",
    re.M,
)
_HEADING_MARKER: Final[re.Pattern[str]] = re.compile(r"^#{2,6}\s*", re.M)


class WstgParser(BaseParser):
    framework_id: ClassVar[str] = "wstg"
    # canonical_framework maps OpenCRE's "OWASP Web Security Testing Guide
    # (WSTG)" onto this through the existing FRAMEWORK_NAME_ALIASES entry.
    framework_name: ClassVar[str] = "WSTG"
    version: ClassVar[str] = "4.2"
    source_url: ClassVar[str] = "https://owasp.org/www-project-web-security-testing-guide/"
    mapping_unit_level: ClassVar[str] = "test"
    # 115 distinct ids across 130 test files; 14 sub-tests carry no id table
    # of their own and one id owns two files. [measured]
    expected_count: ClassVar[int] = 115
    fetched_date: ClassVar[str] = "2026-08-15"
    # 113 of 115 statements clear the 60-character bar and differ from their
    # H1. The two that do not, WSTG-CONF-08 and WSTG-CLNT-08, carry no curated
    # link. [measured]
    min_prose_fraction: ClassVar[float] = 0.98
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        entries = self._read_entries(payload)
        controls, audit = self.build_controls(entries)
        self.write_repair_audit(audit)
        for record in audit:
            logger.warning(
                "%s: %s owns %d source files; their statements were merged: %s",
                self.framework_id, record["control_id"],
                len(record["members"]), record["members"],
            )
        logger.info(
            "%s: %d distinct ids from %d files with an ID table",
            self.framework_id, len(controls), len(entries),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}."
            )

    def _read_entries(self, payload: bytes) -> list[tuple[str, str, str]]:
        """(test_id, member name, file text) for every file with an ID table.

        Raises:
            ValueError: If no member carries an ID table at all.
        """
        entries: list[tuple[str, str, str]] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(n for n in archive.namelist() if MEMBER.search(n)):
                text = archive.read(name).decode("utf-8")
                table = ID_TABLE.search(text)
                if table is None:
                    logger.debug(
                        "%s: %s has no ID table; it is a sub-test that shares "
                        "its parent's id", self.framework_id, name,
                    )
                    continue
                entries.append((table.group(1), name, text))
        if not entries:
            raise ValueError(
                f"{self.framework_id}: no member of {ARCHIVE_NAME} carries a "
                f"WSTG-XXXX-NN ID table. The table is the only join key; the "
                f"path prefix does not spell the id."
            )
        return entries

    @classmethod
    def build_controls(
        cls, entries: list[tuple[str, str, str]],
    ) -> tuple[list[Control], list[dict[str, object]]]:
        """One Control per distinct id, merging any id that owns two files."""
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for test_id, member, text in entries:
            grouped[test_id].append((member, text))

        controls: list[Control] = []
        audit: list[dict[str, object]] = []
        for test_id in sorted(grouped):
            members = sorted(grouped[test_id])
            titles = [cls._title(text, member) for member, text in members]
            statements = [cls._statement(text) for _, text in members]
            bodies = [cls._body(text) for _, text in members]
            if len(members) > 1:
                audit.append({
                    "control_id": test_id,
                    "members": [member for member, _ in members],
                    "titles": titles,
                    # The text itself, not its length. write_repair_audit's own
                    # docstring says why: "A count says a repair fired. It does
                    # not say what moved, or where to, and a fragment attributed
                    # to the wrong control is a wrong compliance assertion
                    # carrying a plausible-looking provenance record. This is
                    # the file a reviewer reads to check one." A list of
                    # integers cannot be checked against anything.
                    "statements": statements,
                    "merged_description": "\n\n".join(s for s in statements if s),
                    "reason": (
                        "one WSTG id is the ID table value of more than one "
                        "file; emitting one control per file would make two "
                        "controls share a control_id and let the later one win"
                    ),
                })
            controls.append(Control(
                control_id=test_id,
                title=" / ".join(titles),
                description="\n\n".join(s for s in statements if s),
                full_text="\n\n".join(b for b in bodies if b) or None,
                hierarchy_level="test",
                parent_id=members[0][0].split("/")[-2],
                metadata={
                    "source_members": [member for member, _ in members],
                    # A merged control's text was assembled by this parser, not
                    # read from one file. The instrument reads text_origin to
                    # populate anchor_source_synthetic; without the key the
                    # column reads zero and the synthetic arm is invisible,
                    # which is the defect the column exists to expose.
                    **({"text_origin": "synthetic"} if len(members) > 1 else {}),
                },
            ))
        return controls, audit

    @staticmethod
    def _title(text: str, member: str) -> str:
        """The file's H1.

        Raises:
            ValueError: If the file has no H1.
        """
        heading = H1.search(text)
        if heading is None:
            raise ValueError(
                f"wstg: {member} carries an ID table but no H1. The H1 is the "
                f"control title, and OpenCRE supplies none of its own for "
                f"this framework."
            )
        return heading.group(1).strip()

    @classmethod
    def _statement(cls, text: str) -> str:
        body = cls._body(text)
        cut = CUT.search(body)
        head = body[: cut.start()] if cut else body
        return _HEADING_MARKER.sub("", head).strip()

    @staticmethod
    def _body(text: str) -> str:
        table = ID_TABLE.search(text)
        return text[table.end():].strip() if table else text.strip()


def main() -> None:
    WstgParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
pytest tests/test_parse_wstg.py -q
mypy parsers/parse_wstg.py --strict
```

- [ ] **Step 5: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_wstg.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(wstg|framework)"
```

Expected `wstg: 115 distinct ids from 116 files with an ID table` and one
`WSTG-INPV-13 owns 2 source files` warning **[measured]**, then:

```
wstg                         118     0  109     9      55  1.98    ~20    0     2     0 0.9237
```

Accept only if `resolution_rate >= 0.92`, `distinct_anchors >= 55`,
`unresolved == 9`, `by_title == 0`, `wrong_anchor_risk == 0`. Any value of
`unresolved` other than 9 means the bogus-id set changed; list it before
accepting. `truncated` is expected to be roughly 20 of 109 given that 38 of the
115 statements exceed `MAX_ANCHOR_CHARS`; record the actual number rather than
asserting it.

- [ ] **Step 6: Confirm the audit file is gitignored**

```bash
git check-ignore -v data/processed/repair_audit/wstg.jsonl \
  || echo "NOT IGNORED. stop and fix .gitignore before committing"
```

- [ ] **Step 7: Commit**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# Ruling R10: this framework is CC BY-SA and is TRACKED. Seven other CC BY-SA
# frameworks were already tracked and published, so treating this one
# differently was defensible on no reading. LICENSES/, the NOTICE modification
# statement and one licence declaration across the published artifacts discharge
# the attribution and notice obligations. Assert it is NOT in the overlay, so a
# future tier change that silently recaptures it fails here.
"$PY" -c "
from tract.config import OVERLAY_FRAMEWORK_IDS
assert 'wstg' not in OVERLAY_FRAMEWORK_IDS, 'tier changed under this task; stop'
print('routing: tracked')
"
git check-ignore -q data/processed/frameworks/wstg.json \
  && { echo "wstg.json is ignored but R10 tracks it; stop"; exit 1; }
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py \
                              tests/test_framework_licenses.py -q
git add parsers/parse_wstg.py tests/test_parse_wstg.py \
        data/processed/frameworks/wstg.json data/processed/all_controls.json
git commit -m "feat: parse WSTG on the ID table, merging the one id that owns two files"
```

---
### Task 8: CSA CCM, 207 controls, 17 domains, and what a domain aggregate should be

**Corrected count.** The `CCM` sheet has 229 rows: 208 with all four columns populated, of which **one is
the header row** `Control Domain | Control Title | Control ID | Control Specification`, leaving **207
control rows**; 19 with only column A, of which 17 are domain headers and 2 are the `End of Standard` and
copyright trailers; one title row and one blank. **[measured]** The correct declared count is **224**.

**No rename map is needed.** Seven curated links carry v4.0's `IVS-*` ids, which v4.1.0 renamed to `I&S-*`.
All seven `section_name` values match the corresponding `I&S-*` control's title exactly, so title-first
resolution answers all seven with no alias at all. **[measured]** Three more links carry v4.0's `AIS-*`
titles (`Secure Application Design and Development`, `Automated Application Security Testing`,
`Automated Secure Application Deployment`) against v4.1.0's shorter ones, so those three fall to the id
channel, where `AIS-04`, `AIS-05` and `AIS-06` still exist. **[measured]**

**`by_title` is 26 of 29, not 7. Plan v2 asserted 7 and would have halted a healthy run.** Its
resolution-order table reasoned only about the seven retired `IVS-*` rows. Fifteen more links target a bare
domain code, and fourteen of those fifteen carry the domain's own descriptive title as `section_name`
(`Audit & Assurance`, `Datacenter security` against the sheet's `Datacenter Security`, and twelve more).
`ProseIndex.lookup` lowercases both sides and tries the title first, so all fourteen resolve by title onto
the domain aggregate. 7 + 14 + 5 of the AIS rows = 26. **[measured]**

**`wrong_anchor_risk` is 1, not 0, and the assertion below says so.** The `IPY` link carries
`section_name = 'Interoperability and portability policy and procedures'`, which is control `IPY-01`'s title
and not the `IPY` domain's name (`Interoperability & Portability`). The title channel therefore answers with
`IPY-01` while the link's own `section_id` names the domain, and `wrong_anchor_risk` counts exactly that
shape. **[measured]** The seven `IVS-*` title hits do not count, because `IVS-01` and its siblings do not
exist in v4.1.0, so the id channel returns nothing to disagree with. **[measured]**

**The IPY link resolves to `IPY-01`, and that is the right answer.** Its CRE target is `847-247`, whose CRE
name is the same string as the `section_name`. **[measured]** So OpenCRE's `section_name` here was taken
from the CRE rather than from the CCM, and the id `IPY` is the coarser of the two candidate anchors. The two
candidates are `IPY-01`'s 462-character specification and the `IPY` domain's 185-character list of four
member titles. The specification is normative prose about exactly the subject the CRE names; the domain
aggregate is parser-synthesised text listing four subjects, one of which is that same control. Anchoring on
the specification gives the trainer a real sentence for a real CRE. Anchoring on the domain gives it a
semicolon list. The link resolves to `IPY-01`, the divergence is written to the repair audit, and
`wrong_anchor_risk` reads 1 on purpose: a 0 would mean the title channel stopped working for all fourteen
domain rows, and a 2 would mean a second `section_name` started naming something its id does not.

**`min_prose_fraction` is 0.99, not 1.0. Plan v2 declared 1.0 and `run()` would have raised.**
`honest_prose_fraction` needs 60 characters, and two specifications fall short: `IAM-07` at 58 characters
and `STA-06` at 43. 222 of 224 units clear the bar, giving **0.9911**. **[measured]** A floor of 0.99
passes at 0.9911 and fails at 221 of 224 (0.9866), so it can fail in both directions.

**The domain aggregate, decided by measurement.** Concatenating each domain's member specifications gives
lengths 1,022 to 4,292, **8 of 17 exceed `MAX_ANCHOR_CHARS`**, and because the concatenation opens with the
domain's own first member control, **all 17** aggregates are a strict prefix of a control that is itself an
anchor. **[measured, premortem]** Concatenating the member **titles** instead gives 163 to 596 characters,
**0 over budget and 0 nested** under both the containment and the strict-prefix definitions. **[measured]**
A domain in CCM is the set of subjects its controls cover, and the ordered list of those subjects is a fair
statement of it. The full specification text stays reachable through each member control.

**A domain aggregate is synthesised text and the report must say so.** Fourteen of the 29 links, 48% of this
framework's training signal, land on a semicolon-joined list of 6 to 21 member titles. **[measured]**
`honest_prose_fraction` counts that as prose and no column distinguishes it from a normative statement.
CLAUDE.md's standing rule is that title fallback is a last resort that gets logged and counted, so each of
the 17 aggregates carries `metadata["text_origin"] = "synthetic"` and gets a `write_repair_audit` record.
Plan v2's self-review listed only Tasks 7 and 12 as text-moving transforms. This is a third.

**Ceiling: 29/29 = 1.0000** **[measured]**. 3 by control id, 26 by title. Floor **1.00**.

**Licensing.** Ruling R5 records the CCM as all rights reserved with no redistribution, so `csa_ccm` sits in
`CONDITIONAL_FRAMEWORK_IDS` under Contract Rule 3 and its processed file routes to the gitignored licensed
overlay. Plan v2's Step 7 asserted the file stays tracked. That step is replaced by Step 8 below.

**Tooling.** `openpyxl 3.1.5` and `defusedxml 0.7.1` are already installed in the 3.12 environment and
`openpyxl.DEFUSEDXML` reads `True`, so this parser reads the workbook hardened. **[measured]** Plan v2's
claim that defusedxml is absent is stale and is removed. `openpyxl` ships no `py.typed`, so
`mypy --strict` fails on the import with `Library stubs not installed for "openpyxl"` until
`types-openpyxl` is pinned. **[measured]** With `types-openpyxl==3.1.5.20260807` installed, `mypy --strict`
on this parser reports `Success: no issues found`. **[measured]**

**Files:**
- Create: `parsers/parse_csa_ccm.py`
- Create: `tests/test_parse_csa_ccm.py`
- Modify: `requirements.txt`, `requirements-lint.txt`, `pyproject.toml`, `tract/config.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `BaseParser.write_repair_audit`.
- Produces: `CsaCcmParser` with `framework_id = "csa_ccm"`, `framework_name = "Cloud Controls Matrix"`; `CsaCcmParser.rows_to_controls(rows: list[tuple[str, str, str, str]]) -> list[Control]`; `CsaCcmParser.domain_audit_records(controls: list[Control]) -> list[dict[str, object]]`.
- Adds: `tract.config.CONTROL_TEXT_ORIGIN_METADATA_KEY`, `tract.config.CONTROL_TEXT_ORIGIN_SYNTHETIC`, read by Task 1's `anchor_source_synthetic` column and reused by Tasks 11 and 13.

**Invalidates:**
- `data/processed/licensed/all_controls.json` and its sha256, so every `corpus_sha256` recorded before this task.
- `data/processed/stopwords.json`, which Task 15 regenerates.
- Task 14's `hub_links_training.jsonl` for the 29 `csa_ccm` rows.
- `data/processed/repair_audit/csa_ccm.jsonl`, written for the first time here.
- Any reader of `metadata["member_ids"]` on a CCM domain. The key held titles, not ids, and is renamed to `member_titles`.

- [ ] **Step 1: Pin openpyxl and add the anchor-source constants**

```text
# requirements.txt: add under the pdfplumber block
# parse_csa_ccm.py reads the CCM workbook. The CCM sheet is a flat four-column
# table; reading it through the raw sheet XML would mean reimplementing shared
# strings and inline formatting for one parser.
openpyxl==3.1.5
```

```text
# requirements-lint.txt: add to the stubs block, which is the only block that
# carries its own pin. openpyxl itself is NOT added to the runtime block above
# it: that block exists for packages that ship py.typed and therefore buy real
# checking, and openpyxl ships none. The stubs give the same checking without
# installing the wheel into the lint job.
types-openpyxl==3.1.5.20260807
```

```toml
# pyproject.toml: add to [project].dependencies, NOT to
# [project.optional-dependencies].llm. The "pdfplumber>=0.10.0" line that plan
# v2 pointed at is inside the llm extra, so following that instruction put
# openpyxl where `pip install -e .` never sees it. [measured]
#
# Pinned with == rather than >=. ci.yml line 101 installs `-e .`, so a floor
# resolves to whatever is newest on the day, while requirements.txt pins
# 3.1.5. The three files stated three different things.
dependencies = [
    "pydantic>=2.0,<3.0",
    "pyyaml>=6.0",
    "beautifulsoup4>=4.14.3",
    "lxml>=5.0",
    "defusedxml>=0.7.1",
    "openpyxl==3.1.5",
    "requests>=2.31",
    "huggingface_hub>=0.24,<1",
]
```

```python
# tract/config.py: add beside CONTROL_DAMAGED_METADATA_KEY
# Set by a parser on a control whose statement it assembled rather than read.
# Task 1's corpus report counts a link landing on one of these against
# anchor_source_synthetic instead of against the TextSelection.source value,
# because "description" would say the text is a control statement and it is
# not. CLAUDE.md's standing rule is that a fallback gets logged and counted.
CONTROL_TEXT_ORIGIN_METADATA_KEY: Final[str] = "text_origin"
CONTROL_TEXT_ORIGIN_SYNTHETIC: Final[str] = "synthetic"
```

If Task 7 or Task 10 already added those two constants for the WSTG merges, reuse them and skip this hunk.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pip install "openpyxl==3.1.5" "types-openpyxl==3.1.5.20260807"
"$PY" -c "import openpyxl; from openpyxl import DEFUSEDXML; print(openpyxl.__version__, DEFUSEDXML)"
```

Expected: `3.1.5 True`. **[measured]**

- [ ] **Step 2: Write the failing test**

```python
# tests/test_parse_csa_ccm.py: create

"""207 controls and 17 domains, and a domain is its members' subjects.

Measured on the pinned workbook: concatenating member specifications makes 8 of
17 domain anchors exceed MAX_ANCHOR_CHARS and makes all 17 a strict prefix of
their own first member control. Concatenating member titles makes 0 of either.

TestSyntheticWorkbook drives parse() and run() against a workbook this file
builds, so the extraction path is covered in CI, where data/raw is absent.
Before this, every path through parse() sat behind a FileNotFoundError skip.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from parsers.parse_csa_ccm import SHEET_NAME, WORKBOOK_NAME, CsaCcmParser
from tract.config import (
    CONTROL_TEXT_ORIGIN_METADATA_KEY,
    CONTROL_TEXT_ORIGIN_SYNTHETIC,
    MAX_ANCHOR_CHARS,
)

ROWS: list[tuple[str, str, str, str]] = [
    ("{\"specification_name\":\"Cloud Controls Matrix\"}",
     "CLOUD CONTROLS MATRIX v4.1.0", "", ""),
    ("", "", "", ""),
    ("Control Domain", "Control Title", "Control ID", "Control Specification"),
    ("Audit & Assurance - A&A", "", "", ""),
    ("Audit & Assurance", "Audit and Assurance Policy and Procedures", "A&A-01",
     "Establish, document, approve, communicate, apply, evaluate and maintain "
     "audit and assurance policies and procedures and standards."),
    ("Audit & Assurance", "Independent Assessments", "A&A-02",
     "Conduct independent audit and assurance assessments according to "
     "relevant standards at least annually."),
    ("Infrastructure Security - I&S", "", "", ""),
    ("Infrastructure Security", "Capacity and Resource Planning", "I&S-02",
     "Plan and monitor the availability, quality, and adequate capacity of "
     "resources in order to deliver the required system performance."),
    ("Infrastructure Security", "Network Defense", "I&S-09",
     "Define, implement and evaluate processes, procedures and defense-in-depth "
     "techniques for protection against network-based attacks."),
    ("End of Standard", "", "", ""),
    ("© Copyright 2026 Cloud Security Alliance - All rights reserved.",
     "", "", ""),
]


def _write_workbook(directory: Path,
                    rows: list[tuple[str, str, str, str]]) -> Path:
    """A workbook shaped like the CCM, worded like nothing in particular."""
    book = openpyxl.Workbook()
    book.active.title = "Introduction"
    sheet = book.create_sheet(SHEET_NAME)
    for row in rows:
        sheet.append(list(row))
    path = directory / WORKBOOK_NAME
    book.save(path)
    return path


class TestRowsToControls:
    def test_the_header_row_is_not_a_control(self) -> None:
        controls = CsaCcmParser.rows_to_controls(ROWS)
        assert "Control ID" not in {c.control_id for c in controls}
        assert len([c for c in controls if c.hierarchy_level == "control"]) == 4

    def test_the_two_trailers_are_not_domains(self) -> None:
        domains = [
            c for c in CsaCcmParser.rows_to_controls(ROWS)
            if c.hierarchy_level == "domain"
        ]
        assert sorted(c.control_id for c in domains) == ["A&A", "I&S"]

    def test_a_domain_statement_lists_its_member_titles(self) -> None:
        domain = next(
            c for c in CsaCcmParser.rows_to_controls(ROWS)
            if c.control_id == "A&A"
        )
        assert domain.description == (
            "Audit and Assurance Policy and Procedures. Independent Assessments."
        )

    def test_a_domain_statement_does_not_open_its_first_member(self) -> None:
        controls = CsaCcmParser.rows_to_controls(ROWS)
        domain = next(c for c in controls if c.control_id == "A&A")
        first = next(c for c in controls if c.control_id == "A&A-01")
        assert not domain.description.startswith(first.description[:60])
        assert len(domain.description) <= MAX_ANCHOR_CHARS

    def test_a_domain_statement_is_marked_synthetic(self) -> None:
        """The trainer must be able to tell a list of subjects from a rule."""
        controls = CsaCcmParser.rows_to_controls(ROWS)
        domain = next(c for c in controls if c.control_id == "A&A")
        member = next(c for c in controls if c.control_id == "A&A-01")
        assert domain.metadata is not None
        assert (domain.metadata[CONTROL_TEXT_ORIGIN_METADATA_KEY]
                == CONTROL_TEXT_ORIGIN_SYNTHETIC)
        assert CONTROL_TEXT_ORIGIN_METADATA_KEY not in (member.metadata or {})

    def test_a_control_carries_its_domain_as_parent(self) -> None:
        control = next(
            c for c in CsaCcmParser.rows_to_controls(ROWS)
            if c.control_id == "I&S-02"
        )
        assert control.parent_id == "I&S"
        assert control.parent_name == "Infrastructure Security"

    def test_a_domain_with_no_members_is_refused(self) -> None:
        rows = [*ROWS[:4], *ROWS[6:]]
        with pytest.raises(ValueError, match="no controls under it"):
            CsaCcmParser.rows_to_controls(rows)


class TestAudit:
    def test_one_record_per_aggregate_plus_the_named_divergence(
        self, tmp_path: Path,
    ) -> None:
        controls = CsaCcmParser.rows_to_controls(ROWS)
        records = CsaCcmParser.domain_audit_records(controls)
        codes = [r["control_id"] for r in records if r["kind"] == "aggregate"]
        assert codes == ["A&A", "I&S"]
        assert all(r["member_count"] == 2 for r in records
                   if r["kind"] == "aggregate")
        divergences = [r for r in records if r["kind"] == "wrong_anchor_risk"]
        assert [r["opencre_section_id"] for r in divergences] == ["IPY"]


class TestSyntheticWorkbook:
    """parse() and run() against a workbook this test writes.

    data/raw is gitignored, so every earlier version of this file reached
    parse() only through a FileNotFoundError skip and CI never executed the
    openpyxl path at all. Ruling R3.
    """

    def test_parse_reads_the_sheet(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_workbook(raw, ROWS)
        parser = CsaCcmParser(raw_dir=raw, output_dir=tmp_path / "out")
        parser.expected_sha256 = None
        parser.expected_count = 6
        parser.expected_control_rows = 4
        parser.expected_domains = 2
        controls = parser.parse()
        assert sorted(c.control_id for c in controls) == [
            "A&A", "A&A-01", "A&A-02", "I&S", "I&S-02", "I&S-09",
        ]

    def test_a_missing_ccm_sheet_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw_nosheet"
        raw.mkdir()
        book = openpyxl.Workbook()
        book.active.title = "CAIQ"
        book.save(raw / WORKBOOK_NAME)
        parser = CsaCcmParser(raw_dir=raw, output_dir=tmp_path / "out")
        parser.expected_sha256 = None
        with pytest.raises(ValueError, match="has no 'CCM' sheet"):
            parser.parse()

    def test_a_short_sheet_is_refused(self, tmp_path: Path) -> None:
        """The band would accept 202 of 224. The structural check does not."""
        raw = tmp_path / "raw_short"
        raw.mkdir()
        _write_workbook(raw, [r for r in ROWS if r[2] != "A&A-02"])
        parser = CsaCcmParser(raw_dir=raw, output_dir=tmp_path / "out")
        parser.expected_sha256 = None
        parser.expected_control_rows = 4
        parser.expected_domains = 2
        with pytest.raises(ValueError, match="control rows"):
            parser.parse()

    def test_run_writes_and_writes_the_audit(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw_run"
        raw.mkdir()
        _write_workbook(raw, ROWS)
        parser = CsaCcmParser(
            raw_dir=raw,
            output_dir=tmp_path / "out",
            audit_dir=tmp_path / "audit",
        )
        (tmp_path / "out").mkdir()
        parser.expected_sha256 = None
        parser.expected_count = 6
        parser.expected_control_rows = 4
        parser.expected_domains = 2
        output = parser.run()
        assert len(output.controls) == 6
        assert [s.path for s in output.source_files] == [WORKBOOK_NAME]
        lines = (tmp_path / "audit" / "csa_ccm.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        assert len(lines) == 3


class TestRun:
    def test_run_writes_from_the_real_workbook(self, tmp_path: Path) -> None:
        parser = CsaCcmParser(output_dir=tmp_path, audit_dir=tmp_path / "audit")
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 224
        assert sum(1 for c in output.controls if c.hierarchy_level == "domain") == 17
        assert [s.path for s in output.source_files] == [WORKBOOK_NAME]
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_csa_ccm.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 4: Write the parser**

```python
# parsers/parse_csa_ccm.py: create

"""Parser for the CSA Cloud Controls Matrix v4.1.0.

Not the AI Controls Matrix. This workbook's title cell reads CLOUD CONTROLS
MATRIX v4.1.0 and it is a different framework from csa_aicm, which has 243
controls and no CRE links.

The CCM sheet is a flat four-column table with two row types interleaved. A
control row populates all four columns; a domain header populates only column A
in the form "<Full Name> - <CODE>". Three column-A-only rows are neither: the
workbook title, the End of Standard trailer, and the copyright paragraph. And
one all-four-columns row is the header itself, which is why a naive count gives
208 control rows where there are 207.

Both granularities are emitted, because OpenCRE links both: 14 of the 29
curated links target a control id and 15 target a bare domain code.

A domain's statement is the ordered list of its member control TITLES, not the
concatenation of their specifications. Measured on this workbook, concatenating
specifications makes 8 of 17 domain anchors exceed MAX_ANCHOR_CHARS and makes
all 17 a strict prefix of their own first member control, which puts 17
near-duplicate pairs into a 29-link framework. The title list runs 163 to 596
characters, exceeds nothing, and prefixes nothing.

That list is assembled text, not a rule anyone wrote, and 14 of the 29 curated
links land on one. So every aggregate carries anchor_source = synthetic and
every aggregate gets a repair-audit record. honest_prose_fraction cannot tell
the difference and would count a semicolon list of subjects as a control
statement.

The seven curated links that still use v4.0's IVS-* ids need no rename map:
their section_name matches the corresponding I&S-* control's title exactly, and
ProseIndex resolves title before id. Three AIS rows carry v4.0 titles against
v4.1.0's shorter ones and fall through to the id channel, where AIS-04, AIS-05
and AIS-06 still exist.

One divergence is real and is recorded rather than repaired. The IPY link's
section_name is control IPY-01's title, not the IPY domain's name, so the title
channel answers with IPY-01 while its section_id names the domain. Its CRE
target 847-247 carries that same string as its CRE name, so the name came from
the CRE rather than from the CCM. IPY-01's 462-character specification is the
better anchor for that CRE than a four-item list of subjects, the link is left
to resolve there, and the corpus report reads wrong_anchor_risk = 1 for this
framework on purpose.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import ClassVar, Final

import openpyxl

from tract.config import (
    CONTROL_TEXT_ORIGIN_METADATA_KEY,
    CONTROL_TEXT_ORIGIN_SYNTHETIC,
)
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WORKBOOK_NAME: Final[str] = "CCMv4.1.0-generated_at_2026_01_13.xlsx"
SOURCE_SHA256: Final[str] = (
    "5e721628c8ab297bdbd355afa4c01699971fcbb9cb16802ccb9d42c7176ab32b"
)
SHEET_NAME: Final[str] = "CCM"

DOMAIN_HEADER: Final[re.Pattern[str]] = re.compile(r"^(.+?)\s+-\s+([A-Z&]{2,5})$")
HEADER_CELL: Final[str] = "Control ID"
# Column-A-only rows that look like a domain header and are not.
TRAILERS: Final[tuple[str, ...]] = ("End of Standard", "©", "{")

# The one curated row whose section_name names a control while its section_id
# names that control's domain. Hand-verified against the pinned workbook and
# against the link's CRE target. Recorded, never repaired.
KNOWN_DIVERGENCES: Final[dict[str, tuple[str, str]]] = {
    "IPY": (
        "IPY-01",
        "OpenCRE's section_id is the IPY domain and its section_name is "
        "control IPY-01's title, which is also the name of its CRE target "
        "847-247. The title channel therefore answers with IPY-01. That is "
        "left standing: IPY-01's specification is a rule about the subject "
        "the CRE names, and the IPY domain's statement is a four-item list "
        "of subjects this parser assembled.",
    ),
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class CsaCcmParser(BaseParser):
    framework_id: ClassVar[str] = "csa_ccm"
    # Matches the curated links' standard_name exactly; no alias entry exists
    # or is needed. [measured]
    framework_name: ClassVar[str] = "Cloud Controls Matrix"
    version: ClassVar[str] = "4.1.0"
    source_url: ClassVar[str] = (
        "https://cloudsecurityalliance.org/artifacts/cloud-controls-matrix-v4/"
    )
    mapping_unit_level: ClassVar[str] = "control"
    # 207 control rows plus 17 domains. The 208th all-four-columns row is the
    # sheet header. [measured]
    expected_count: ClassVar[int] = 224
    # COUNT_TOLERANCE is 10%, so the band around 224 is 202 to 246 and a
    # parser that lost 22 controls would write in silence. These two are the
    # structural check that beats the band. Overridable so a synthetic
    # workbook can drive parse() in CI. [measured]
    expected_control_rows: ClassVar[int] = 207
    expected_domains: ClassVar[int] = 17
    fetched_date: ClassVar[str] = "2026-08-15"
    # 222 of 224 units clear HONEST_PROSE_MIN_CHARS. IAM-07's specification is
    # 58 characters and STA-06's is 43, giving 0.9911. A floor of 1.0 refuses
    # to write on correct output; 0.99 passes at 0.9911 and fails at 221 of
    # 224. [measured]
    min_prose_fraction: ClassVar[float] = 0.99
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(WORKBOOK_NAME)
        self._check_digest(payload)
        rows = self._read_sheet()
        controls = self.rows_to_controls(rows)
        self._check_shape(controls)
        self.write_repair_audit(self.domain_audit_records(controls))
        logger.info(
            "%s: %d controls and %d domains, %d synthesised domain statement(s)",
            self.framework_id,
            sum(1 for c in controls if c.hierarchy_level == "control"),
            sum(1 for c in controls if c.hierarchy_level == "domain"),
            sum(1 for c in controls if c.hierarchy_level == "domain"),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a workbook that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {WORKBOOK_NAME} has sha256 {actual}, "
                f"not the pinned {self.expected_sha256}. expected_count of "
                f"224 and the domain-statement length measurements were both "
                f"taken against the pinned bytes."
            )

    def _check_shape(self, controls: list[Control]) -> None:
        """Refuse a parse whose row shape moved, before the band can hide it.

        Raises:
            ValueError: If the control-row or domain count differs from the
                declared shape.
        """
        found_controls = sum(1 for c in controls if c.hierarchy_level == "control")
        found_domains = sum(1 for c in controls if c.hierarchy_level == "domain")
        if found_controls != self.expected_control_rows:
            raise ValueError(
                f"{self.framework_id}: {found_controls} control rows, "
                f"expected {self.expected_control_rows}. COUNT_TOLERANCE puts "
                f"the band around the 224 total at 202 to 246, so a loss of "
                f"up to 22 controls would write without a word."
            )
        if found_domains != self.expected_domains:
            raise ValueError(
                f"{self.framework_id}: {found_domains} domains, expected "
                f"{self.expected_domains}. Fifteen of the 29 curated links "
                f"target a bare domain code, so a lost domain is a lost link."
            )

    def _read_sheet(self) -> list[tuple[str, str, str, str]]:
        """The CCM sheet's first four columns, as stripped strings.

        openpyxl needs a path, and BaseParser has already read and hashed the
        bytes, so the file is opened a second time here. That is a deliberate
        exception to the in-memory rule: the manifest already records the
        digest of exactly these bytes, and openpyxl's read-only mode has no
        file-object-free entry point that avoids reimplementing shared strings.
        openpyxl.DEFUSEDXML reads True in this environment, so the workbook's
        XML is parsed hardened. [measured]

        Raises:
            ValueError: If the CCM sheet is absent.
        """
        path = self.raw_dir / WORKBOOK_NAME
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            if SHEET_NAME not in workbook.sheetnames:
                raise ValueError(
                    f"{self.framework_id}: {WORKBOOK_NAME} has no {SHEET_NAME!r} "
                    f"sheet, only {workbook.sheetnames}. The CAIQ sheet is the "
                    f"self-assessment questionnaire and is not the controls."
                )
            rows: list[tuple[str, str, str, str]] = []
            for row in workbook[SHEET_NAME].iter_rows(values_only=True):
                cells = [
                    _WHITESPACE.sub(" ", str(cell or "")).strip()
                    for cell in (list(row) + [""] * 4)[:4]
                ]
                rows.append((cells[0], cells[1], cells[2], cells[3]))
            return rows
        finally:
            workbook.close()

    @classmethod
    def rows_to_controls(
        cls, rows: list[tuple[str, str, str, str]],
    ) -> list[Control]:
        """Controls then domains, from the sheet's interleaved row types.

        Raises:
            ValueError: If a domain header has no control rows under it.
        """
        controls: list[Control] = []
        domains: list[tuple[str, str]] = []
        members: dict[str, list[str]] = defaultdict(list)
        current_code = ""
        current_name = ""

        for first, title, control_id, specification in rows:
            if control_id and specification:
                if control_id == HEADER_CELL:
                    continue
                controls.append(Control(
                    control_id=control_id,
                    title=title,
                    description=specification,
                    hierarchy_level="control",
                    parent_id=current_code or None,
                    parent_name=current_name or first or None,
                ))
                members[current_code].append(title)
                continue

            if not first or title or control_id or specification:
                continue
            if first.startswith(TRAILERS):
                continue
            header = DOMAIN_HEADER.match(first)
            if header is None:
                logger.info(
                    "csa_ccm: column-A-only row %r is not a domain header",
                    first[:60],
                )
                continue
            current_name, current_code = header.group(1), header.group(2)
            domains.append((current_code, current_name))

        controls += cls._domain_controls(domains, members)
        return controls

    @staticmethod
    def _domain_controls(
        domains: list[tuple[str, str]], members: dict[str, list[str]],
    ) -> list[Control]:
        """One mapping unit per domain, stated as its members' subjects.

        Raises:
            ValueError: If a domain owns no controls, which means the row
                ordering changed and every domain statement built from that
                ordering is attached to the wrong domain.
        """
        built: list[Control] = []
        for code, name in domains:
            titles = members.get(code, [])
            if not titles:
                raise ValueError(
                    f"csa_ccm: domain {code} has no controls under it. Domain "
                    f"membership comes from row order, so an empty domain "
                    f"means the sheet was reordered and the domains that do "
                    f"have members may have the wrong ones."
                )
            built.append(Control(
                control_id=code,
                title=name,
                description=". ".join(titles) + ".",
                hierarchy_level="domain",
                metadata={
                    # Titles, not ids. The v2 key was named member_ids and
                    # held titles, which is the kind of thing a reader trusts
                    # once and never rechecks.
                    "member_titles": list(titles),
                    CONTROL_TEXT_ORIGIN_METADATA_KEY:
                        CONTROL_TEXT_ORIGIN_SYNTHETIC,
                },
            ))
        return built

    @classmethod
    def domain_audit_records(
        cls, controls: list[Control],
    ) -> list[dict[str, object]]:
        """What this parser assembled, and the one divergence it left alone.

        A count says a synthesis happened. It does not say what text a link
        now trains on. One record per domain aggregate names the members that
        were joined and how long the result is, and one record per entry in
        KNOWN_DIVERGENCES names a curated link whose two sides disagree.
        """
        records: list[dict[str, object]] = []
        for control in controls:
            if control.hierarchy_level != "domain":
                continue
            titles = list((control.metadata or {}).get("member_titles") or [])
            records.append({
                "kind": "aggregate",
                "control_id": control.control_id,
                "domain_name": control.title,
                "member_count": len(titles),
                "member_titles": titles,
                "statement_chars": len(control.description),
                "reason": (
                    "The CCM gives a domain no text of its own. Its statement "
                    "here is the ordered list of its member control titles, "
                    "assembled by this parser. Concatenating the member "
                    "specifications instead put 8 of 17 domain anchors over "
                    "MAX_ANCHOR_CHARS and made all 17 a strict prefix of their "
                    "own first member."
                ),
            })
        known = {c.control_id for c in controls}
        for section_id, (target, reason) in sorted(KNOWN_DIVERGENCES.items()):
            if target not in known:
                raise ValueError(
                    f"csa_ccm: KNOWN_DIVERGENCES names {target}, which this "
                    f"parse did not produce. The IPY link resolves through "
                    f"that control, so a stale entry sends it somewhere else "
                    f"while the resolution rate stays at 1.0000."
                )
            records.append({
                "kind": "wrong_anchor_risk",
                "opencre_section_id": section_id,
                "resolved_to": target,
                "resolved_by": "section_name",
                "reason": reason,
            })
        return records


def main() -> None:
    CsaCcmParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_csa_ccm.py -q
"$PY" -m mypy parsers/parse_csa_ccm.py --strict
```

`mypy --strict` reports `Success: no issues found` once `types-openpyxl==3.1.5.20260807` is installed, and
reports `Library stubs not installed for "openpyxl"` without it. **[measured]** If the error appears, Step 1
did not run.

- [ ] **Step 6: Run against the real source and check the join**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" parsers/parse_csa_ccm.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py --framework csa_ccm --json
```

Expected log line: `csa_ccm: 207 controls and 17 domains, 17 synthesised domain statement(s)` **[measured]**.

Accept only on this exact set. Every value is **[measured]** on the pinned workbook and the curated link
file, before the parser is written.

| field | value | how it can fail in each direction |
|---|---|---|
| `links` | 29 | |
| `by_title` | **26** | 14 domain names + 7 renamed `IVS-*` titles + 5 `AIS-*` titles that did not change. Below 26 means a title stopped matching. Above 26 means an `AIS` v4.0 title came back |
| `by_id` | 3 | `AIS-04`, `AIS-05`, `AIS-06`, whose v4.0 titles no longer match |
| `unresolved` | 0 | attainable maximum is 29 |
| `distinct_anchors` | 29 | 29 links onto 29 different texts. A drop to 15 means the domain aggregates collapsed onto their members |
| `distinct_anchors_pre_truncation` | 29 | nothing truncates |
| `fallback_anchors` (BEFORE) | 29 | orchestrator measured. The anchor count does not move. The gain is text on 15 of the 29 and a synthesised list on 14 |
| `links_per_anchor` | 1.00 | |
| `truncated` | 0 | longest resolved anchor is 596 characters against a 2,150 budget |
| `nested_anchors` | **0** | the column that would have caught the specification-concatenation design. A 17 means the domain statement rule was not applied |
| `contained_anchors` | 0 | |
| `dropped_by_prose_rule` | 1 | `STA-06`'s 43-character specification does not clear its 25-character title by `PROSE_MIN_EXTRA_CHARS`. No curated link targets it |
| `wrong_anchor_risk` | **1** | the `IPY` row. A 0 means the title channel stopped answering for the 14 domain rows. A 2 or more means a second `section_name` started naming a control its own id does not |
| `anchor_source_synthetic` | **14** | 48% of this framework's links land on an assembled list of subjects |
| `anchor_source_description` | 15 | |
| `anchor_source_full_text` / `_title` | 0 / 0 | no specification exceeds `DESCRIPTION_MAX_LENGTH`, so none gets `full_text` |
| `distinct_hubs` | 27 | |
| `links_per_hub` | 1.07 | |
| `resolution_rate` | 1.0000 | floor 1.00 |

- [ ] **Step 7: Read the repair audit before trusting the 14 synthetic anchors**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -c "
import json, pathlib
rows = [json.loads(l) for l in
        pathlib.Path('data/processed/repair_audit/csa_ccm.jsonl').read_text(
            encoding='utf-8').splitlines()]
agg = [r for r in rows if r['kind'] == 'aggregate']
print(len(agg), 'aggregates,', len(rows) - len(agg), 'divergence record(s)')
print('member counts:', sorted(r['member_count'] for r in agg))
print('statement chars:', sorted(r['statement_chars'] for r in agg))
"
```

Expected: `17 aggregates, 1 divergence record(s)`, member counts 4 to 21, statement lengths 163 to 596.
**[measured]** If any statement exceeds `MAX_ANCHOR_CHARS`, the domain-statement rule regressed to
specification concatenation and `nested_anchors` will show it.

- [ ] **Step 8: Confirm the overlay routing, then commit**

Ruling R5 records the CCM as all rights reserved with no redistribution, so it belongs to
`CONDITIONAL_FRAMEWORK_IDS` and its processed file routes to the gitignored overlay. Plan v2 asserted the
file stays tracked on the strength of an owner ruling of 2026-08-16. The contract supersedes that. If the
owner wants the earlier ruling to stand, that is a decision to take before this commit, not after `git push`.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -c "
from tract.config import OVERLAY_FRAMEWORK_IDS
assert 'csa_ccm' in OVERLAY_FRAMEWORK_IDS, 'Contract Rule 3 has not landed; stop'
print('overlay routing: on')
"
git check-ignore -v data/processed/frameworks/csa_ccm.json \
  || { echo "NOT IGNORED. the Rule 3 .gitignore lines are missing; stop"; exit 1; }
git check-ignore -v data/processed/repair_audit/csa_ccm.jsonl \
  || { echo "audit NOT IGNORED. stop"; exit 1; }
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py \
                              tests/test_framework_licenses.py -q
git add parsers/parse_csa_ccm.py tests/test_parse_csa_ccm.py \
        requirements.txt requirements-lint.txt pyproject.toml tract/config.py
git commit -m "feat: parse the CCM at both granularities, stating a domain by its members' subjects"
```

---

<<<TASK 9>>>

---

### Task 9: NIST SSDF, the table is already ruled, and there is no merge step

**Verify the row shape before writing anything, and verify the extractor first.** Every `[measured]` PDF
number in this plan is a function of the extractor version. `requirements.txt` pins
`pdfplumber==0.11.10`; the interpreter this plan mandates has **0.11.4** installed. **[measured]** A premise
check that does not assert the version is measuring an unknown.

Measured against the pinned PDF with pdfplumber 0.11.4: `pdfplumber.extract_tables()` on pages 13 through
26 returns **47 task cells at column index 3**, each a whole cell with the task statement complete and
newlines inside it. **[measured]** The rows below a task row repeat wrapped fragments of the *practice* cell
in column 1 and carry nothing in column 3. There is nothing to absorb. No merge step is built here.

**Seven pages also carry a truncated second copy of a task at column index 4.** Pages 15, 18, 20, 21, 22, 24
and 26 each hold one, for `PO.2.3`, `PS.2.1`, `PW.2.1`, `PW.4.4`, `PW.6.1`, `PW.9.1` and `RV.2.2`, and every
one of them is cut mid-phrase. **[measured]** So a scan across all columns returns 54 cells over columns
`[3, 4]` with 7 duplicate task ids, and plan v2's premise check, which scanned all columns, could not
produce the `47 / [3] / 0` it told the executor to expect. The parser reads column 3 only and is unaffected.
Plan v2's prose sentence "every one of them at column index 3" was wrong about the document and right about
the parser.

**Seven column-3 cells do not end in a period or a colon, not zero.** Five are the `Moved to` redirects and
two are real tasks that close with a `[Formerly PW.3.1]` and a `[Formerly PW.4.3]` bracket. **[measured]**
Mid-sentence truncation is not what that test detects, so the premise check below counts duplicate ids and
column-4 copies instead, which is what actually distinguishes a whole cell from a fragment.

**Five stub rows, not two.** `PW.3.1`, `PW.3.2`, `PW.4.3`, `PW.4.5` and `PW.5.2` have bodies of the form
`Moved to <target>`. **[measured]** None of the five is targeted by a curated link **[measured]**, so they
are recorded as redirects in metadata and excluded from the emitted controls. That leaves **42 real tasks**,
which matches the document's own count, spread over **19 practices** with 0 tasks left without one after the
forward fill. **[measured]**

**Two malformed link rows, and plan v2 quoted one of them wrong.** Two of the 46 curated links carry a
mid-sentence fragment where a `PS.1.1`-style id belongs. The `PW.8.1` fragment ends
`...which types of testing should be used.` The v2 constant ended `...should be performed.`, which matches
nothing in the link file, so that link stayed unresolved at 45 of 46 and the declared ceiling of 46/46 was
unreachable. **[measured]** Both fragments appear verbatim inside their task's statement. **[measured]**
Both are registered as `alt_ids`, which is what Task 2 built.

**Title must be the task id.** OpenCRE sets `section_name` to a task statement for all 46 links, and 44 of
them carry a real task id in `section_id`. **[measured]** Task statements run 54 to 333 characters, median
163. **[measured]** If the parser used the statement as both `title` and `description`, `_is_prose` would
exclude every control from `ProseIndex`. Using the practice name as the title also costs links: 5 task
statements are shorter than their practice name plus 20 characters, and the ceiling falls from 46/46 to
39/46. **[measured, premortem]** With `title = task id`: ceiling **44/46 = 0.9565** without `alt_ids` and
**46/46 = 1.0000** with them. **[measured]** Floor **1.00**.

**The anchor count goes down, and this task says so before the run.** 46 links land on 44 distinct
`section_name` anchors today and on **42** clause-level task anchors after, because all 42 tasks are linked,
`PO.3.3` and `RV.1.1` each carry two links, and both malformed fragments alias onto tasks that are already
linked directly. **[measured]** 42 is the arithmetic maximum: the framework has 42 controls. Plan v2 told
the executor to accept only `distinct_anchors >= 44`, which no correct parse can reach.

**Files:**
- Create: `parsers/parse_nist_ssdf.py`
- Create: `tests/test_parse_nist_ssdf.py`
- Create: `tests/synthetic_pdf.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `alt_ids` from Task 2.
- Produces: `NistSsdfParser` with `framework_id = "nist_ssdf"`, `framework_name = "NIST SSDF"`; `NistSsdfParser.rows_to_controls(rows: list[list[str | None]], require_alternate_targets: bool = False) -> list[Control]`.
- Produces: `tests.synthetic_pdf.build_pdf(pages, rules=None, width=612, height=792) -> bytes`, used by Tasks 11, 12 and 13 as well.

**Invalidates:**
- `data/processed/all_controls.json` and its sha256, so every `corpus_sha256` recorded before this task.
- `data/processed/stopwords.json`, which Task 15 regenerates.
- Task 14's `hub_links_training.jsonl` for the 46 `nist_ssdf` rows.
- Any consumer of the 44-anchor BEFORE figure for this framework. The AFTER figure is 42, a named regression of 2.

- [ ] **Step 1: Re-verify the premise yourself, extractor included**

Do not skip this. Plan v2's headline tests failed against its own implementation because it assumed a shape
it never checked, and its premise check could not print the output it promised.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" - <<'PYEOF'
import collections
import re

import pdfplumber

# Every [measured] number below is a function of this version. requirements.txt
# pins 0.11.10 and this interpreter has 0.11.4, so the pin and the measurement
# disagree. Assert the one the numbers came from and re-measure if it moves.
EXPECTED_PDFPLUMBER = "0.11.4"
if pdfplumber.__version__ != EXPECTED_PDFPLUMBER:
    raise SystemExit(
        f"pdfplumber {pdfplumber.__version__}, not {EXPECTED_PDFPLUMBER}. "
        f"Every task-cell count below was measured on {EXPECTED_PDFPLUMBER}. "
        f"Re-measure before writing the parser, and reconcile "
        f"requirements.txt, which pins 0.11.10."
    )

TASK = re.compile(r'^(P[OSW]|RV)\.\d+\.\d+:')
TASK_COLUMN = 3
found = []
with pdfplumber.open('data/raw/frameworks/nist_ssdf/nist_sp_800_218.pdf') as pdf:
    for page in range(13, 28):
        for table in pdf.pages[page].extract_tables():
            if max(len(r) for r in table) < 4:
                continue
            for row in table:
                for index, cell in enumerate(row):
                    if cell and TASK.match(cell.strip()):
                        found.append((page, index, re.sub(r'\s+', ' ', cell.strip())))

tasks = [(p, t) for p, index, t in found if index == TASK_COLUMN]
others = [(p, index, t) for p, index, t in found if index != TASK_COLUMN]
ids = [t.split(':')[0] for _, t in tasks]
print('task cells at column 3:', len(tasks))
print('columns seen anywhere:', sorted({index for _, index, _ in found}))
print('duplicate task ids at column 3:',
      [k for k, v in collections.Counter(ids).items() if v > 1])
print('truncated second copies at column 4:', len(others),
      'on pages', sorted({p for p, _, _ in others}))
print('stub rows:', [t.split(':')[0] for _, t in tasks
                     if t.split(': ', 1)[1].lower().startswith('moved to')])
PYEOF
```

Expected, and every line is **[measured]**:

```
task cells at column 3: 47
columns seen anywhere: [3, 4]
duplicate task ids at column 3: []
truncated second copies at column 4: 7 on pages [15, 18, 20, 21, 22, 24, 26]
stub rows: ['PW.3.1', 'PW.3.2', 'PW.4.3', 'PW.4.5', 'PW.5.2']
```

Seven column-4 copies are expected and are not a defect. The parser reads column 3 only. If the column-3
count is not 47, or a duplicate id appears at column 3, stop and re-measure before writing the parser.

- [ ] **Step 2: Write the shared synthetic-PDF helper**

Five of the eleven parsers in this plan never reach `parse()` outside a `FileNotFoundError` skip, and they
are the two PDFs, the workbook and the multi-document pair, which is the most fragile extraction in the
batch. Ruling R3. `data/raw` is gitignored, so CI cannot use the real bytes. This helper writes an
uncompressed PDF that pdfplumber reads back, for text and for ruled tables. Verified against
`pdfplumber 0.11.4`. **[measured]**

```python
# tests/synthetic_pdf.py: create

"""A minimal PDF writer, so a parser's extraction path runs in CI.

data/raw is gitignored, so a test that opens the real source skips wherever the
tree is absent, which is every CI run. Five parsers in this plan reached
parse() only through such a skip. This module writes an uncompressed PDF with a
standard Type 1 font and, when asked, ruled lines, which is enough for both
pdfplumber.extract_text() and pdfplumber.extract_tables() to read the content
back. Verified against pdfplumber 0.11.4.

It adds no dependency. Building the bytes by hand is the point: a fixture
generator that needed reportlab would put a new package into the lint and test
environments to test a parser that reads PDFs it did not write.
"""

from __future__ import annotations

TextRun = tuple[float, float, str]
Rule = tuple[float, float, float, float]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(
    pages: list[list[TextRun]],
    rules: list[list[Rule]] | None = None,
    width: float = 612,
    height: float = 792,
) -> bytes:
    """Pages of (x, y_from_top, text) runs, with optional (x0, y0, x1, y1) rules.

    y is measured from the top of the page, because a fixture is easier to read
    that way than in PDF user space. Rules are what pdfplumber's default
    "lines" table strategy detects, so a table fixture must draw its own grid.
    """
    line_sets = rules if rules is not None else [[] for _ in pages]
    if len(line_sets) != len(pages):
        raise ValueError(
            f"build_pdf: {len(pages)} page(s) of text and {len(line_sets)} "
            f"of rules. They index together."
        )
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    contents: list[int] = []
    for runs, lines in zip(pages, line_sets):
        stream = ["0.5 w"]
        for x0, y0, x1, y1 in lines:
            stream.append(
                f"{x0:.2f} {height - y0:.2f} m {x1:.2f} {height - y1:.2f} l S"
            )
        stream.append("BT /F1 10 Tf")
        for x, y, text in runs:
            stream.append(
                f"1 0 0 1 {x:.2f} {height - y:.2f} Tm ({_escape(text)}) Tj"
            )
        stream.append("ET")
        payload = "\n".join(stream).encode("latin-1")
        contents.append(add(
            b"<< /Length " + str(len(payload)).encode() + b" >>\nstream\n"
            + payload + b"\nendstream"
        ))

    pages_id = len(objects) + len(contents) + 1
    page_ids: list[int] = []
    for content_id in contents:
        page_ids.append(add(
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        ))
    pages_obj = add(
        b"<< /Type /Pages /Count " + str(len(page_ids)).encode() + b" /Kids ["
        + b" ".join(f"{p} 0 R".encode() for p in page_ids) + b"] >>"
    )
    catalog = add(b"<< /Type /Catalog /Pages " + str(pages_obj).encode() + b" 0 R >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{start}\n%%EOF\n"
    ).encode()
    return bytes(out)
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_parse_nist_ssdf.py: create

"""The SSDF table is ruled: extract_tables returns whole task cells.

Measured against the pinned PDF with pdfplumber 0.11.4: 47 task cells at column
index 3, no duplicate ids there, and 7 truncated second copies at column 4 on
seven pages. The parser reads column 3 only. The continuation rows below a task
repeat wrapped fragments of the practice cell in column 1 and hold nothing in
column 3, so the practice column needs a forward fill and nothing needs a merge.

TestSyntheticPdf drives parse() through pdfplumber against a PDF this file
builds, so the extraction path runs in CI, where data/raw is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_nist_ssdf import SOURCE_FILE, NistSsdfParser
from tests.synthetic_pdf import build_pdf

PRACTICE = (
    "Define Security Requirements for Software\nDevelopment (PO.1): Ensure "
    "that security requirements for software development are known at all "
    "times."
)

ROWS: list[list[str | None]] = [
    [None, "Practices", None, "Tasks", None, None, "Notional Implementation "
     "Examples", None, None, None, "References", None],
    [PRACTICE, "Define Security Requirements for Software", None,
     "PO.1.1: Identify and document all security requirements for the "
     "organization's software development infrastructures and processes, and "
     "maintain the requirements over time.",
     None, None, "Example 1: Define policies for securing software "
     "development infrastructures.", None, None, None,
     "BSAFSS: SM.3, DE.1", None],
    [None, "Development (PO.1): Ensure that security", None, None, None, None,
     None, None, None, None, "BSIMM: CP1.1, CP1.3", None],
    [None, None, None,
     "PO.1.2: Identify and document all security requirements for "
     "organization-developed software to meet, and maintain the requirements "
     "over time.",
     None, None, "Example 1: Define a set of secure coding standards.",
     None, None, None, "EO14028: 4e(ix)", None],
    [None, None, None, "PW.3.2: Moved to PW.4.4", None, None, None, None,
     None, None, None, None],
]

# Column and row boundaries for the synthetic table. Seven cells per row, so
# the task lands at index 3 and the examples at index 6, exactly as the source
# does. Verified to round-trip through pdfplumber 0.11.4.
COLUMNS = [40.0, 200.0, 215.0, 230.0, 430.0, 445.0, 460.0, 580.0]
BANDS = [60.0, 130.0, 215.0, 305.0, 400.0, 490.0, 530.0]
CELLS: list[dict[int, list[str]]] = [
    {0: ["Practices"], 3: ["Tasks"], 6: ["Examples"]},
    {0: ["Define Security Reqs for", "Software Dev (PO.1):",
         "Ensure that security", "requirements are known."],
     3: ["PO.1.1: Identify and", "document all security",
         "requirements for the", "organization and keep",
         "them current over time."],
     6: ["Example 1: Define a", "policy."]},
    {3: ["PO.1.2: Identify and", "document every security",
         "requirement that developed", "software has to meet, and",
         "maintain them over time."],
     6: ["Example 1: Coding", "standards."]},
    {0: ["Protect the Software", "(PS.1): Protect all forms",
         "of code from tampering."],
     3: ["PS.1.1: Store all forms of", "code - including source",
         "code, executable code,", "and configuration-as-code",
         "- based on the principle", "of least privilege so that",
         "only authorized personnel,", "tools, services, etc. have",
         "access."]},
    {0: ["Review Code (PW.8):", "Test executable code."],
     3: ["PW.8.1: Determine whether", "executable code testing",
         "should be performed to find", "vulnerabilities not identified",
         "by previous reviews,", "analysis, or testing and, if",
         "so, which types of testing", "should be used."]},
    {3: ["PW.3.2: Moved to PW.4.4"]},
]


def _table_pdf() -> bytes:
    """One ruled table on page 13, so TABLE_PAGES needs no override."""
    rules = [(x, BANDS[0], x, BANDS[-1]) for x in COLUMNS]
    rules += [(COLUMNS[0], y, COLUMNS[-1], y) for y in BANDS]
    runs: list[tuple[float, float, str]] = []
    for index, cells in enumerate(CELLS):
        top = BANDS[index] + 11
        for column, lines in cells.items():
            for offset, line in enumerate(lines):
                runs.append((COLUMNS[column] + 3, top + offset * 10, line))
    blank_text: list[list[tuple[float, float, str]]] = [[] for _ in range(13)]
    blank_rules: list[list[tuple[float, float, float, float]]] = [
        [] for _ in range(13)
    ]
    return build_pdf(blank_text + [runs], blank_rules + [rules])


class TestRowsToControls:
    def test_only_task_cells_become_controls(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert [c.control_id for c in controls] == ["PO.1.1", "PO.1.2"]

    def test_the_whole_task_statement_survives(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.description.endswith("maintain the requirements over time.")
        assert "Identify and document all security requirements" in first.description

    def test_title_is_the_task_id_not_the_statement(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert first.title == "PO.1.1"
        assert first.description != first.title

    def test_the_practice_is_forward_filled_onto_later_tasks(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert controls[0].parent_id == "PO.1"
        assert controls[1].parent_id == "PO.1"
        assert controls[1].parent_name is not None
        assert controls[1].parent_name.startswith(
            "Define Security Requirements for Software Development"
        )

    def test_a_moved_to_stub_is_not_emitted(self) -> None:
        controls = NistSsdfParser.rows_to_controls(ROWS)
        assert "PW.3.2" not in {c.control_id for c in controls}

    def test_notional_examples_are_kept_out_of_the_statement(self) -> None:
        first = NistSsdfParser.rows_to_controls(ROWS)[0]
        assert "Example 1" not in first.description
        assert first.full_text is not None
        assert "Example 1" in first.full_text


class TestMalformedIdMap:
    def test_an_alternate_whose_target_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names task"):
            NistSsdfParser.rows_to_controls(ROWS, require_alternate_targets=True)

    def test_both_declared_fragments_end_the_way_the_link_file_does(
        self,
    ) -> None:
        """The v2 map ended the PW.8.1 fragment 'should be performed.'

        The curated link ends 'should be used.', so that entry matched
        nothing, the link stayed unresolved, and the declared 46/46 ceiling
        was unreachable while every other gate stayed green.
        """
        from parsers.parse_nist_ssdf import MALFORMED_SECTION_IDS

        fragments = sorted(MALFORMED_SECTION_IDS)
        assert len(fragments) == 2
        assert any(f.endswith("which types of testing should be used.")
                   for f in fragments)
        assert not any(f.endswith("should be performed.") for f in fragments)


class TestSyntheticPdf:
    """parse() through pdfplumber, with no dependency on data/raw."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> NistSsdfParser:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(_table_pdf())
        instance = NistSsdfParser(raw_dir=raw, output_dir=tmp_path / "out")
        instance.expected_sha256 = None
        instance.expected_count = 4
        instance.expected_task_cells = 5
        instance.expected_redirects = 1
        instance.expected_practices = 3
        return instance

    def test_parse_reads_the_ruled_table(self, parser: NistSsdfParser) -> None:
        controls = parser.parse()
        assert [c.control_id for c in controls] == [
            "PO.1.1", "PO.1.2", "PS.1.1", "PW.8.1",
        ]

    def test_the_practice_is_forward_filled_across_the_real_extraction(
        self, parser: NistSsdfParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["PO.1.2"].parent_id == "PO.1"
        assert controls["PW.8.1"].parent_id == "PW.8"

    def test_both_declared_fragments_reach_their_task(
        self, parser: NistSsdfParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        for task_id, needle in (
            ("PS.1.1", "configuration-as-code"),
            ("PW.8.1", "which types of testing should be used."),
        ):
            metadata = controls[task_id].metadata
            assert metadata is not None
            assert any(needle in alt for alt in metadata["alt_ids"])

    def test_a_short_table_is_refused(self, parser: NistSsdfParser) -> None:
        """The band would accept 38 of 42. The cell count does not."""
        parser.expected_task_cells = 6
        with pytest.raises(ValueError, match="task cell"):
            parser.parse()

    def test_run_writes(self, parser: NistSsdfParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == 4
        assert [s.path for s in output.source_files] == [SOURCE_FILE]


class TestRun:
    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = NistSsdfParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 42
        assert len({c.parent_id for c in output.controls}) == 19
        assert [s.path for s in output.source_files] == [SOURCE_FILE]
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_nist_ssdf.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 5: Write the parser**

```python
# parsers/parse_nist_ssdf.py: create

"""Parser for NIST SP 800-218, the Secure Software Development Framework.

The tasks live in one ruled table spanning pages 14 through 27 of the PDF
(0-indexed 13 through 26). Measured against the pinned bytes with pdfplumber
0.11.4, pdfplumber.extract_tables() returns 47 task cells at column index 3,
all whole: the task statement arrives complete with its own newlines. The rows
below a task repeat wrapped fragments of the PRACTICE cell in column 1 and hold
nothing in column 3. So the practice column needs a forward fill and nothing
needs a rowspan merge. extract_text() is the call that interleaves this table's
columns; extract_tables() does not.

Seven pages carry a truncated second copy of a task at column index 4. This
parser reads column 3 only, so those copies are invisible to it, and a premise
check that scans every column will see 54 cells rather than 47.

Five task cells are redirects of the form "PW.3.1: Moved to PO.1.3". None is
targeted by a curated link. They are recorded as `retired_tasks` metadata on
the framework's first control and excluded from the emitted set: a 15-character
statement is not a control, and emitting one would put a non-statement anchor
in the corpus and drag the prose floor for no join.

The title is the task ID and this is not cosmetic. OpenCRE sets section_name to
a task statement for all 46 curated links, so a parser that used the statement
as its title would make description equal title, which is exactly the case
ProseIndex refuses to index. Using the parent practice name instead costs five
links, because five statements are shorter than their practice name plus
PROSE_MIN_EXTRA_CHARS.

Two curated links carry a mid-sentence text fragment in section_id instead of a
task id. Both fragments appear verbatim inside a task statement, so they are
declared here as alternate ids and resolved through the alt_ids channel. They
are declared rather than derived: a substring search that ran at parse time
would silently re-attach a link to a different task after a source refresh.
Both strings are quoted from the link file, character for character, including
the en dashes and the closing "should be used." The earlier draft of this map
ended that second fragment "should be performed.", which matched no link, and
the framework sat at 45 of 46 with every gate green.
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "nist_sp_800_218.pdf"
SOURCE_SHA256: Final[str] = (
    "617746e553a9e2da49bfbd4eef0dfc3094758a39b869314e4173ac36605cde22"
)
# 0-indexed. The table starts on printed page 14 and ends on 28.
TABLE_PAGES: Final[range] = range(13, 28)

PRACTICE_COLUMN: Final[int] = 0
TASK_COLUMN: Final[int] = 3
EXAMPLES_COLUMN: Final[int] = 6

TASK_ID: Final[re.Pattern[str]] = re.compile(r"^((?:P[OSW]|RV)\.\d+\.\d+):\s*(.+)$")
PRACTICE_ID: Final[re.Pattern[str]] = re.compile(
    r"^(.+?)\s*\(((?:P[OSW]|RV)\.\d+)\)\s*:\s*(.*)$", re.S
)
REDIRECT: Final[re.Pattern[str]] = re.compile(r"^moved to\b", re.IGNORECASE)

# OpenCRE section_id values that are a sentence fragment rather than a task id,
# mapped to the task whose statement contains that fragment verbatim.
# Hand-verified against the pinned PDF and against the curated link file;
# never derived at parse time.
MALFORMED_SECTION_IDS: Final[dict[str, str]] = {
    "code, executable code, and configuration-as-code – based on the principle "
    "of least privilege so that only authorized personnel, tools, services, "
    "etc. have access.": "PS.1.1",
    "should be performed to find vulnerabilities not identified by previous "
    "reviews, analysis, or testing and, if so, which types of testing should "
    "be used.": "PW.8.1",
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class NistSsdfParser(BaseParser):
    framework_id: ClassVar[str] = "nist_ssdf"
    framework_name: ClassVar[str] = "NIST SSDF"
    version: ClassVar[str] = "1.1"
    source_url: ClassVar[str] = (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf"
    )
    mapping_unit_level: ClassVar[str] = "task"
    # 47 task cells minus 5 "Moved to" redirects. [measured]
    expected_count: ClassVar[int] = 42
    # COUNT_TOLERANCE is 10%, so the band around 42 is 38 to 46 and a parser
    # that lost four tasks would write without a word. These three are the
    # structural check that beats the band. Overridable so a synthetic PDF can
    # drive parse() in CI. [measured]
    expected_task_cells: ClassVar[int] = 47
    expected_redirects: ClassVar[int] = 5
    expected_practices: ClassVar[int] = 19
    fetched_date: ClassVar[str] = "2026-08-15"
    # 41 of 42 statements clear the 60-character bar; the shortest real task
    # statement is 54 characters, giving 0.9762. [measured]
    min_prose_fraction: ClassVar[float] = 0.97
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        rows = self._read_rows(payload)
        controls = self.rows_to_controls(rows, require_alternate_targets=True)
        self._check_shape(rows, controls)
        logger.info(
            "%s: %d tasks across %d practices",
            self.framework_id, len(controls),
            len({c.parent_id for c in controls}),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not "
                f"the pinned {self.expected_sha256}. Both entries in "
                f"MALFORMED_SECTION_IDS quote this document's text verbatim, "
                f"so a changed source can attach a link to the wrong task."
            )

    def _check_shape(
        self, rows: list[list[str | None]], controls: list[Control],
    ) -> None:
        """Refuse a parse whose table shape moved, before the band hides it.

        Raises:
            ValueError: If the task-cell, redirect or practice count differs
                from the declared shape.
        """
        cells = sum(
            1 for row in rows
            if len(row) > TASK_COLUMN
            and row[TASK_COLUMN]
            and TASK_ID.match(_WHITESPACE.sub(" ", str(row[TASK_COLUMN]).strip()))
        )
        if cells != self.expected_task_cells:
            raise ValueError(
                f"{self.framework_id}: {cells} task cell(s) at column "
                f"{TASK_COLUMN}, expected {self.expected_task_cells}. "
                f"COUNT_TOLERANCE puts the band around 42 emitted tasks at 38 "
                f"to 46, so a loss of four would write in silence."
            )
        redirects = cells - len(controls)
        if redirects != self.expected_redirects:
            raise ValueError(
                f"{self.framework_id}: {redirects} redirect stub(s), expected "
                f"{self.expected_redirects}. A stub that stopped being "
                f"recognised becomes a 15-character anchor in the corpus."
            )
        practices = len({c.parent_id for c in controls})
        if practices != self.expected_practices:
            raise ValueError(
                f"{self.framework_id}: {practices} practice(s), expected "
                f"{self.expected_practices}. The practice is forward-filled "
                f"from the first task row of each group, so a wrong count "
                f"means tasks are attached to the wrong practice."
            )
        empty = [c.control_id for c in controls if not c.parent_id]
        if empty:
            raise ValueError(
                f"{self.framework_id}: task(s) {empty} have no practice after "
                f"the forward fill. The fill is the only thing that gives a "
                f"task its parent, so an empty one means the practice cell "
                f"moved out of column {PRACTICE_COLUMN}."
            )

    def _read_rows(self, payload: bytes) -> list[list[str | None]]:
        """Every row of the widest table on each table page.

        Raises:
            ValueError: If no page yields a table at least four columns wide.
        """
        rows: list[list[str | None]] = []
        with pdfplumber.open(BytesIO(payload)) as pdf:
            for page_number in TABLE_PAGES:
                if page_number >= len(pdf.pages):
                    break
                for table in pdf.pages[page_number].extract_tables():
                    if max(len(row) for row in table) < 4:
                        continue
                    rows.extend(table)
        if not rows:
            raise ValueError(
                f"{self.framework_id}: no table of four or more columns on "
                f"pages {TABLE_PAGES.start}-{TABLE_PAGES.stop} of "
                f"{SOURCE_FILE}. extract_text() interleaves this table's "
                f"columns, so falling back to it would ship task statements "
                f"truncated by the adjacent Examples column."
            )
        return rows

    @classmethod
    def rows_to_controls(
        cls,
        rows: list[list[str | None]],
        require_alternate_targets: bool = False,
    ) -> list[Control]:
        """One Control per real task, practice forward-filled.

        Raises:
            ValueError: If require_alternate_targets is set and a declared
                malformed-id target is absent from the parsed tasks.
        """
        practice_id = ""
        practice_name = ""
        controls: list[Control] = []
        redirects: dict[str, str] = {}

        for row in rows:
            cells = [
                _WHITESPACE.sub(" ", str(cell).strip()) if cell else ""
                for cell in (list(row) + [None] * 12)[:12]
            ]
            practice = PRACTICE_ID.match(cells[PRACTICE_COLUMN])
            if practice is not None:
                practice_name = practice.group(1).strip()
                practice_id = practice.group(2)

            task = TASK_ID.match(cells[TASK_COLUMN])
            if task is None:
                continue
            task_id, statement = task.group(1), task.group(2).strip()
            if REDIRECT.match(statement):
                redirects[task_id] = statement
                continue

            alternates = [
                fragment for fragment, target in MALFORMED_SECTION_IDS.items()
                if target == task_id
            ]
            metadata: dict[str, str | list[str]] = {"practice": practice_id}
            if alternates:
                metadata["alt_ids"] = alternates
            controls.append(Control(
                control_id=task_id,
                title=task_id,
                description=statement,
                full_text=cls._full_text(statement, cells[EXAMPLES_COLUMN]),
                hierarchy_level="task",
                parent_id=practice_id or None,
                parent_name=practice_name or None,
                metadata=metadata,
            ))

        cls._check_alternate_targets(
            {c.control_id for c in controls}, require_alternate_targets,
        )
        if redirects and controls:
            cls._record_redirects(controls[0], redirects)
        logger.info("nist_ssdf: %d redirect stub(s) excluded: %s",
                    len(redirects), sorted(redirects))
        return controls

    @staticmethod
    def _full_text(statement: str, examples: str) -> str | None:
        """The statement with its notional implementation examples appended.

        Returned as full_text rather than folded into description on purpose,
        and only when it adds something: ProseIndex prefers full_text, so this
        IS the anchor for the 35 links that land on a task carrying examples.
        The examples say how an organisation might satisfy the task, which is
        remediation guidance; putting it in front of the encoder pulls the
        anchor toward tasks that share tooling rather than meaning. It is kept
        because a reviewer needs it and because no task's combined text
        exceeds MAX_ANCHOR_CHARS, so nothing is displaced by it. [measured]
        """
        if not examples:
            return None
        return f"{statement}\n\n{examples}"

    @staticmethod
    def _record_redirects(control: Control, redirects: dict[str, str]) -> None:
        """Attach the retired task numbering to the framework's first control.

        Recorded rather than dropped silently. A reader who finds PW.3.2 cited
        in an older mapping needs somewhere in this artifact that says where it
        went.
        """
        metadata = dict(control.metadata or {})
        metadata["retired_tasks"] = [
            f"{task}: {target}" for task, target in sorted(redirects.items())
        ]
        control.metadata = metadata

    @staticmethod
    def _check_alternate_targets(task_ids: set[str], required: bool) -> None:
        """Refuse a declared malformed-id map that no longer matches the source.

        Raises:
            ValueError: If a declared target task is absent.
        """
        if not required:
            return
        missing = sorted({
            target for target in MALFORMED_SECTION_IDS.values()
            if target not in task_ids
        })
        if missing:
            raise ValueError(
                f"nist_ssdf: MALFORMED_SECTION_IDS names task(s) {missing} "
                f"that this parse did not produce. Two curated links carry a "
                f"sentence fragment where a task id belongs and reach their "
                f"task only through this map; a stale entry leaves them "
                f"unresolved while every other gate stays green."
            )


def main() -> None:
    NistSsdfParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_nist_ssdf.py -q
"$PY" -m mypy parsers/parse_nist_ssdf.py --strict
```

`mypy --strict` on this file is clean as written. **[measured]**

- [ ] **Step 7: Run against the real source and check the join**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" parsers/parse_nist_ssdf.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py --framework nist_ssdf --json
```

Expected log lines: `nist_ssdf: 5 redirect stub(s) excluded: ['PW.3.1', 'PW.3.2', 'PW.4.3', 'PW.4.5',
'PW.5.2']` and `nist_ssdf: 42 tasks across 19 practices`. **[measured]**

Accept only on this exact set. Every value is **[measured]**.

| field | value | how it can fail in each direction |
|---|---|---|
| `links` | 46 | |
| `by_title` | 0 | the title is a task id and no `section_name` is one. Above 0 means the parser is emitting statements as titles, and `_is_prose` will start excluding controls |
| `by_id` | 46 | 44 by task id and 2 through `alt_ids`. A 44 means the Task 2 channel is not reaching this parser's metadata |
| `unresolved` | 0 | attainable maximum is 46 |
| `distinct_anchors` | **42** | the framework has 42 controls, so 42 is the arithmetic ceiling. Plan v2 asked for 44, which no correct parse can reach. Below 42 means a task lost its anchor |
| `distinct_anchors_pre_truncation` | 42 | nothing truncates |
| `fallback_anchors` (BEFORE) | 44 | orchestrator measured. This is a **named regression of 2**: `PO.3.3` and `RV.1.1` each carry two links, and both malformed fragments alias onto tasks that are already linked directly. Carry it into the AFTER report as a regression on the anchor column and a gain on the text column |
| `links_per_anchor` | 1.10 | |
| `truncated` | 0 | no statement plus its examples exceeds 2,150 characters |
| `nested_anchors` / `contained_anchors` | 0 / 0 | |
| `dropped_by_prose_rule` | 0 | 31 of 42 carry `full_text` and the rest clear `_is_prose` against a 6-character title |
| `wrong_anchor_risk` | 0 | the counter increments only in the title branch and `by_title` is 0, so this value is unfailable for this framework. It is recorded, not asserted |
| `anchor_source_full_text` | 35 | the tasks that carry notional examples |
| `anchor_source_description` | 11 | |
| `anchor_source_title` / `_synthetic` | 0 / 0 | |
| `distinct_hubs` | 28 | |
| `links_per_hub` | 1.64 | |
| `resolution_rate` | 1.0000 | floor 1.00. A 0.9565 means the `alt_ids` channel is missing |

- [ ] **Step 8: Commit**

```bash
git add parsers/parse_nist_ssdf.py tests/test_parse_nist_ssdf.py \
        tests/synthetic_pdf.py \
        data/processed/frameworks/nist_ssdf.json data/processed/all_controls.json
git commit -m "feat: read the SSDF task table as ruled cells, with the two malformed ids declared"
```

---

<<<TASK 11>>>

---

### Task 10: NIST SP 800-63B, the version blocker is already resolved

**Invalidates:** `data/processed/frameworks/nist_800_63.json` and the tracked corpus built from it. NIST 800-63 is a US Government work and stays tracked. Its 79 links contribute nothing to training until Task 14, which promotes it from 0 links to 78 after the anchor-length floor drops the one row whose section_name is the 5-character fragment `are g`.

**The structures document's CRITICAL FINDING is stale.** It reports the fetched
file as SP 800-63-4, whose restructured numbering matches none of OpenCRE's 79
links. The fetch was corrected before this plan was written. The file on disk
now has `<title>NIST Special Publication 800-63B</title>`, contains
`Memorized Secret` 17 times, and **all 25 distinct curated `section_id` values
appear in it**. **[measured]** Parsing the headings gives **118 numbered
sections, all distinct**, and 24 of the 25 link ids are one of them.
**[measured]**

**Ceiling: 78/79 = 0.98734** **[derived]**, floor **0.98**. The single miss is a
curated link whose `section_id` is the two-character fragment `are g`, an
upstream extraction artifact with no recoverable target. Unlike the SSDF
fragments it does not appear inside any section's text, so no `alt_ids` entry
can honestly be written for it and none is.

**No digest gate is possible here and that is deliberate.** `pages.nist.gov`
sits behind Cloudflare, which injects a fresh nonce into every response, so
`scripts/fetch_frameworks.py` leaves `expected_sha256` at `None` for this
source. **[measured]** The parser gates on structure instead: it refuses to
write unless it finds at least 100 numbered sections and unless every id in
`REQUIRED_SECTION_IDS` is present. That list is the 25 distinct curated link ids
minus `are g`, so the gate fires on exactly the failure a revision swap would
cause.

**All 79 links contribute zero training links today**, for the same two reasons
as Task 6: `nist_800_63` is in `PHASE1B_DROPPED_FRAMEWORKS`, and every one of
its 79 `section_name` values is the section number, 3 to 7 characters, under
`PHASE1B_MIN_SECTION_TEXT_LENGTH`. **[measured]** Task 14 is what changes that.

**Files:**
- Create: `parsers/parse_nist_800_63.py`
- Create: `tests/test_parse_nist_800_63.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `bs4.BeautifulSoup`.
- Produces: `Nist80063Parser` with `framework_id = "nist_800_63"`, `framework_name = "NIST 800-63"`; `Nist80063Parser.sections_from_html(html: str) -> list[Control]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_nist_800_63.py: create

"""SP 800-63B numbers its headings, and the section number is the join key.

Measured on the staged file: 118 numbered headings, all distinct, covering 24
of the 25 distinct curated section_ids. The 25th is the fragment 'are g', an
upstream artifact that appears in no section and is left unresolved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_nist_800_63 import Nist80063Parser

HTML = """<html><head><title>NIST Special Publication 800-63B</title></head>
<body>
<h2 id="sec5">5 Authenticator and Verifier Requirements</h2>
<p>This section is normative.</p>
<h3 id="reqauthtype">5.1 Requirements by Authenticator Type</h3>
<p>Authenticators are described by type.</p>
<h4 id="memsecret">5.1.1 Memorized Secrets</h4>
<p>A Memorized Secret authenticator, commonly referred to as a password or PIN,
is a secret value intended to be chosen and memorized by the user.</p>
<h5 id="memsecretver">5.1.1.2 Memorized Secret Verifiers</h5>
<p>Verifiers SHALL require subscriber-chosen memorized secrets to be at least 8
characters in length, and SHALL permit at least 64 characters.</p>
<h2 id="glossary">Appendix A Definitions and Abbreviations</h2>
<p>Not numbered in the dotted scheme.</p>
</body></html>
"""


class TestSectionsFromHtml:
    def test_the_section_number_is_the_control_id(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert [c.control_id for c in controls] == [
            "5", "5.1", "5.1.1", "5.1.1.2",
        ]

    def test_title_is_the_heading_text_without_the_number(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert controls[3].title == "Memorized Secret Verifiers"

    def test_body_stops_at_the_next_heading(self) -> None:
        controls = Nist80063Parser.sections_from_html(HTML)
        assert controls[2].description.startswith("A Memorized Secret")
        assert "at least 8" not in controls[2].description

    def test_an_unnumbered_appendix_heading_is_not_a_section(self) -> None:
        assert "Appendix A" not in {
            c.title for c in Nist80063Parser.sections_from_html(HTML)
        }


class TestStructureGate:
    def test_a_document_missing_a_required_section_is_refused(
        self, tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(HTML, encoding="utf-8")
        parser = Nist80063Parser(raw_dir=raw, output_dir=tmp_path / "out")
        parser.min_sections = 2
        parser.required_section_ids = ("5.1.1.2", "9.9.9")
        with pytest.raises(ValueError, match=r"9\.9\.9"):
            parser.parse()

    def test_a_thin_document_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw2"
        raw.mkdir()
        (raw / "sp800_63b.html").write_text(HTML, encoding="utf-8")
        parser = Nist80063Parser(raw_dir=raw, output_dir=tmp_path / "out2")
        parser.required_section_ids = ()
        with pytest.raises(ValueError, match="numbered section"):
            parser.parse()


class TestRun:
    def test_run_writes_from_the_real_document(self, tmp_path: Path) -> None:
        parser = Nist80063Parser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) >= 100
        assert [s.path for s in output.source_files] == ["sp800_63b.html"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_parse_nist_800_63.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_nist_800_63.py: create

"""Parser for NIST SP 800-63B, Authentication and Lifecycle Management.

The revision matters more than the parsing does. OpenCRE's 79 links carry
800-63-3B section numbers (5.1.1.2, 6.1.2.3, A.3). Revision 4 restructured the
document and uses slug ids with a single-integer chapter attribute, so none of
those numbers exists in it and a fetch of revision 4 would leave every link
unjoinable while looking like a successful fetch. The staged file is revision
3B: its title element says so, and all 25 distinct curated section_ids appear
in it.

There is no digest gate. pages.nist.gov sits behind Cloudflare, which injects a
per-response nonce into the body, so two fetches of the identical document have
different hashes and scripts/fetch_frameworks.py deliberately leaves
expected_sha256 unset. The gate here is structural instead: a minimum count of
numbered sections, and the presence of every id in REQUIRED_SECTION_IDS. That
list is the curated link ids, so a revision swap fails this parser rather than
producing a corpus whose join silently went to zero.

The mapping unit is the numbered section. Its id is the dotted number in the
heading text, its title is the rest of the heading, and its statement is
everything between that heading and the next one at any level.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar, Final

from bs4 import BeautifulSoup, Tag

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "sp800_63b.html"

HEADING_TAG: Final[re.Pattern[str]] = re.compile(r"^h[1-6]$")
# "5.1.1.2 Memorized Secret Verifiers" and "A.3 Appendix Section". Both forms
# appear in the curated section_ids.
NUMBERED: Final[re.Pattern[str]] = re.compile(
    r"^\s*((?:A\.)?\d+(?:\.\d+)*)\.?\s+(\S.*)$"
)

# Every distinct curated section_id except the fragment "are g", which is an
# upstream extraction artifact and appears in no section of this document.
# Hand-transcribed from data/training/hub_links_curated.jsonl and checked
# against the staged file. [measured]
REQUIRED_SECTION_IDS: Final[tuple[str, ...]] = (
    "5.1.1.1", "5.1.1.2", "5.1.2.2", "5.1.3.2", "5.1.4.2", "5.1.5.2",
    "5.1.7.2", "5.2.1", "5.2.2", "5.2.3", "5.2.5", "5.2.6", "5.2.8", "5.2.9",
    "5.2.10", "6.1.2.3", "6.1.3", "6.1.4", "7.1", "7.1.1", "7.1.2", "7.2",
    "7.2.1", "A.3",
)
# Well under the 118 measured, so ordinary editorial change does not trip it
# while a revision swap, which takes the dotted numbering to zero, does.
MIN_NUMBERED_SECTIONS: Final[int] = 100

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class Nist80063Parser(BaseParser):
    framework_id: ClassVar[str] = "nist_800_63"
    framework_name: ClassVar[str] = "NIST 800-63"
    version: ClassVar[str] = "800-63B rev 3"
    source_url: ClassVar[str] = "https://pages.nist.gov/800-63-3/sp800-63b.html"
    mapping_unit_level: ClassVar[str] = "section"
    # 118 numbered headings in the staged document. [measured] Declared as a
    # floor: this document is not digest-pinned, so an editorial revision may
    # add sections without changing what any link resolves to.
    expected_count: ClassVar[int] = 100
    expected_count_is_floor: ClassVar[bool] = True
    fetched_date: ClassVar[str] = "2026-08-15"
    # 113 of 118 numbered sections carry a body longer than their heading;
    # the five that do not are chapter headings whose text is entirely in
    # their subsections. [measured]
    min_prose_fraction: ClassVar[float] = 0.94
    # Instance-overridable so a fixture-backed test declares its own structural
    # gate rather than widening the real one.
    required_section_ids: ClassVar[tuple[str, ...]] = REQUIRED_SECTION_IDS
    min_sections: ClassVar[int] = MIN_NUMBERED_SECTIONS

    def parse(self) -> list[Control]:
        html = self.read_source(SOURCE_FILE)
        controls = self.sections_from_html(html)
        self._check_structure(controls)
        logger.info(
            "%s: %d numbered sections, %d of the %d required ids present",
            self.framework_id, len(controls),
            len({c.control_id for c in controls} & set(self.required_section_ids)),
            len(self.required_section_ids),
        )
        return controls

    def _check_structure(self, controls: list[Control]) -> None:
        """Refuse a document that is not the revision the links key to.

        Raises:
            ValueError: If the numbering is too sparse, or a required section
                id is absent.
        """
        if len(controls) < self.min_sections:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} yields {len(controls)} "
                f"numbered sections, under the floor of {self.min_sections}. "
                f"Revision 4 restructured this document to slug ids with no "
                f"dotted numbering, and fetching it would leave all 79 "
                f"curated links unjoinable while looking like a success. This "
                f"source is not digest-pinned because Cloudflare injects a "
                f"per-response nonce, so this count is the pin."
            )
        found = {c.control_id for c in controls}
        missing = sorted(set(self.required_section_ids) - found)
        if missing:
            raise ValueError(
                f"{self.framework_id}: section id(s) {missing} are absent from "
                f"{SOURCE_FILE}. Every one of them is targeted by a curated "
                f"OpenCRE link, so their absence means this is not the "
                f"revision the links were written against."
            )

    @classmethod
    def sections_from_html(cls, html: str) -> list[Control]:
        """One Control per numbered heading, in document order.

        Raises:
            ValueError: If two headings claim the same section number, which
                would let one section's text silently replace another's.
        """
        soup = BeautifulSoup(html, "lxml")
        controls: list[Control] = []
        seen: set[str] = set()
        for heading in soup.find_all(HEADING_TAG):
            match = NUMBERED.match(
                _WHITESPACE.sub(" ", heading.get_text()).strip()
            )
            if match is None:
                continue
            number, title = match.group(1), match.group(2).strip()
            if number in seen:
                raise ValueError(
                    f"nist_800_63: section number {number!r} appears on more "
                    f"than one heading. Emitting both would give one "
                    f"control_id two statements and let whichever is written "
                    f"last answer every link to it."
                )
            seen.add(number)
            controls.append(Control(
                control_id=number,
                title=title,
                description=cls._body(heading) or title,
                hierarchy_level="section",
                parent_id=number.rsplit(".", 1)[0] if "." in number else None,
            ))
        return controls

    @staticmethod
    def _body(heading: Tag) -> str:
        """Text between this heading and the next heading at any level."""
        parts: list[str] = []
        for sibling in heading.next_siblings:
            name = getattr(sibling, "name", None)
            if name and HEADING_TAG.match(name):
                break
            text = (
                sibling.get_text(" ") if hasattr(sibling, "get_text")
                else str(sibling)
            )
            cleaned = _WHITESPACE.sub(" ", text).strip()
            if cleaned:
                parts.append(cleaned)
        return " ".join(parts).strip()


def main() -> None:
    Nist80063Parser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Reconcile REQUIRED_SECTION_IDS against the curated link file**

The list above is transcribed. Confirm it before running anything, because a
wrong entry turns the structural gate into a guaranteed failure or a silent
pass.

```bash
"$PY" - <<'PYEOF'
import json
ids = set()
with open('data/training/hub_links_curated.jsonl', encoding='utf-8') as handle:
    for line in handle:
        row = json.loads(line)
        if row['framework_id'] == 'nist_800_63':
            ids.add(row['section_id'])
print(len(ids), sorted(ids))
PYEOF
```

Expected: 25 ids. Every one except `are g` must appear in
`REQUIRED_SECTION_IDS`; if the printed set differs from the literal above,
replace the literal with the printed set minus `are g` and say so in the commit
message.

- [ ] **Step 5: Run the tests and typecheck**

```bash
pytest tests/test_parse_nist_800_63.py -q
mypy parsers/parse_nist_800_63.py --strict
```

- [ ] **Step 6: Run against the real source and check the join**

```bash
PYTHONPATH=. "$PY" parsers/parse_nist_800_63.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py | grep -E "^(nist_800_63|framework)"
```

Expected:

```
nist_800_63                   79     0   78     1      25  3.12     0    5     0     0 0.9873
```

Accept only if `resolution_rate >= 0.98`, `distinct_anchors == 25`,
`unresolved == 1`. `links_per_anchor` of 3.12 is the source's shape: 79 links
over 25 sections. `dropped_by_prose_rule` of about 5 is the chapter headings
whose text lives entirely in their subsections; none of them is linked.

- [ ] **Step 7: Commit**

```bash
git add parsers/parse_nist_800_63.py tests/test_parse_nist_800_63.py \
        data/processed/frameworks/nist_800_63.json data/processed/all_controls.json
git commit -m "feat: parse SP 800-63B rev 3 sections, gating on the numbering the links use"
```

---
### Task 11: ENISA, three tables, one name space, and no stable id anywhere

The source defines no control identifier at all, so OpenCRE's extraction degraded: 40 of the 68 curated
links carry the literal placeholder `Table 5:` in `section_id` and 18 carry `Table 3:`. **[measured]** The
join must key on `section_name`, which holds 33 distinct values. **[measured]**

**Twenty of the 68 links point at Table 3, not Table 5.** `Poisoning`, `Evasion`, `Model disclosure`,
`Data disclosure`, `Oracle`, `Label modification`, `Compromise of ML application components`, `Model or
data disclosure`, `Denial of service due to inconsistent data or a sponge example`, and `Use of adversarial
examples crafted in white or grey box conditions (e.g. FGSM...)`. **[measured]** Table 3 is the threat
taxonomy, and a parser that emitted only the 37 security controls would leave 29% of this framework's links
unresolved. All 13 Table 3 entries are emitted as mapping units alongside the controls.

**Extraction, measured rather than assumed.** `pdfplumber.extract_tables()` returns several tables per page
and the one that matters is the widest; on Table 5's pages it is 34 or 35 columns. **[measured]** The
definition text lands in **column 2 on some rows and column 3 on others**, which is why a per-page "densest
column" heuristic loses rows: it picks one column for a page that is not column-uniform. The rule here is
per-row instead. The name is column 0 (columns 0 and 1 for Table 3, which has a threat column and a
sub-threat column), the definition is the join of columns 1 through 4 with any lone lifecycle `x` dropped,
and a row with an empty name is a continuation appended to the previous unit. Under that rule **0 of 35
Table 5 rows and 0 of 13 Table 3 rows extract with an empty definition** **[measured]**, against the 4 empty
Table 3 definitions the premortem found under the previous rule, including `Evasion` and `Poisoning`, the
two most-linked entries in the framework.

**The two tests in plan v2 failed against plan v2's own implementation, and both are fixed here.**
`rows_to_units` has the signature `(rows, name_columns, banners=())` and both tests called it without
`banners`, so no banner was ever filtered. Executed against the plan's own fixtures, `TABLE5` returned
`['Security controls', 'Apply modifications on inputs17', 'Ensure ML applications comply with third
parties' security requirements']`, which makes `assert "Security controls" not in names` false, and `TABLE3`
returned `['Threats sub- threats', 'Evasion', 'Data disclosure']`, which makes `assert names == ["Evasion",
"Data disclosure"]` false. **[measured, both]** Passing the banner tuples makes all five `TestRowsToUnits`
assertions pass. **[measured]** This is v1's Critical C7 recurring at a new task, so Step 2 below runs the
tests before the parser exists and again after.

**`mypy --strict` fails on plan v2's parser as written.** The annotation `pdf: pdfplumber.PDF` on `_collect`
raises `error: Name "pdfplumber.PDF" is not defined [name-defined]`, because `pyproject.toml` silences
`pdfplumber.*` with `ignore_missing_imports`, which makes the module `Any` and leaves nothing to resolve the
attribute against. **[measured]** Annotating `pdf: Any` clears it, and that is what the parser below does.
**[measured: `Success: no issues found in 1 source file`]**

**Three name-matching defects, each measured.**

| defect | rows lost | example |
|---|---|---|
| footnote digits fused onto the name | 6 | `Apply modifications on inputs17` |
| curly punctuation against OpenCRE's ASCII | 2 | `third parties’ security requirements` |
| ellipsis character against three periods | 1 | `(e.g. FGSM…)` vs `(e.g. FGSM...)` |

**[measured]** Naive exact matching over Table 5 and Table 3 resolves **51/68**. Adding NFKD normalisation
and footnote-digit removal takes it to **62/68**. **[measured]**

**The last 6 rows come from Annex C.** `Ensure reliable sources are used` (3 links) and `Use methods to clean
the training dataset from suspicious samples` (3 links) appear as row names in Annex C and nowhere in
Table 5. **[measured]** Emitting them from Annex C gives 35 + 2 = **37 controls, which is the count the
source states in its own text**, plus 13 threats, **50 mapping units**, and a ceiling of **68/68 = 1.0000**.
**[measured, end to end against the pinned PDF]** Floor **1.00**.

**Those two controls are the only synthesised text in this framework, and the report must say so.** Their
statement is Annex C's implementation-example column, which says how to satisfy the control rather than what
it is, because it is the only text the source gives them in a table this parser can read. Six of the 68
links land on one of the two, so `anchor_source_synthetic` reads **6**. **[measured]** Both carry
`metadata["text_origin"] = "synthetic"`, using the constants Task 8 adds.

Annex C also spells six Table 5 names differently (`least privilege` for `least privileged`, `minimise` for
`minimize`, and four more). Those six are registered as `alt_titles` on their Table 5 control rather than
emitted as separate controls: emitting them would put six near-duplicate anchors into a 33-anchor framework,
which is the collapse the instrument exists to catch.

**`min_prose_fraction` rises from 0.96 to 1.0.** Measured end to end on the pinned PDF, all 50 units clear
`HONEST_PROSE_MIN_CHARS`, the shortest definition is 80 characters, and no description equals its title, so
`honest_prose_fraction` is exactly **1.0000**. **[measured]** Plan v2's 0.96 permitted two units to degrade
to a bare title without the gate firing. The attainable range is 0.0 to 1.0 and the trigger sits at the
measured maximum, so any degradation fails.

**Anchor count does not improve and that is the honest outcome.** 68 links land on 33 title anchors today
and on 33 prose anchors after. **[measured, both]** The gain is that the 33 anchors become paragraphs
instead of phrases.

**Files:**
- Create: `parsers/parse_enisa.py`
- Create: `tests/test_parse_enisa.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `tests.synthetic_pdf.build_pdf` (Task 9), `tract.config.CONTROL_TEXT_ORIGIN_METADATA_KEY` (Task 8).
- Produces: `EnisaParser` with `framework_id = "enisa"`, `framework_name = "ENISA"`; `EnisaParser.rows_to_units(rows: list[list[str | None]], name_columns: int, banners: tuple[str, ...] = ()) -> list[tuple[str, str]]`; `EnisaParser.normalise_name(name: str) -> str`.

**Invalidates:**
- `data/processed/all_controls.json` and its sha256, so every `corpus_sha256` recorded before this task.
- `data/processed/stopwords.json`, which Task 15 regenerates.
- Task 14's `hub_links_training.jsonl` for the 68 `enisa` rows.
- The 39 baseline records that Contract Rule 8.5 expects to collide on key. ENISA control ids change from the `Table 3:` and `Table 5:` placeholders to slugs, so those rows are renames, not losses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_enisa.py: create

"""ENISA has no control id, so the join is the name, and the name is damaged.

Measured on the pinned PDF: 6 curated links lose to footnote digits fused onto
a control name, 2 to a curly apostrophe against OpenCRE's ASCII, and 1 to an
ellipsis character against three periods. The definition also lands in column 2
on some rows and column 3 on others, which is why the merge is per row.

Every call to rows_to_units passes banners. The previous version of this file
left the argument at its default, so no banner was filtered and two of its own
assertions were false against its own implementation.

TestSyntheticPdf drives parse() through pdfplumber against a PDF this file
builds, so the three-table extraction and all three declaration gates run in
CI, where data/raw is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_enisa import (
    ANNEX_C_BANNERS,
    ANNEX_C_ONLY,
    ANNEX_C_VARIANTS,
    FOOTNOTE_NAMES,
    SOURCE_FILE,
    TABLE3_BANNERS,
    TABLE3_PAGES,
    TABLE5_BANNERS,
    TABLE5_PAGES,
    ANNEX_C_PAGES,
    EnisaParser,
)
from tests.synthetic_pdf import build_pdf
from tract.config import (
    CONTROL_TEXT_ORIGIN_METADATA_KEY,
    CONTROL_TEXT_ORIGIN_SYNTHETIC,
)

# Column 0 name, definition in column 2 on the first unit and column 3 on the
# second, lifecycle x marks from column 5. Continuation rows carry no name.
TABLE5: list[list[str | None]] = [
    ["Security controls", "", "Definition", "", "", "Stages of the lifecycle"],
    ["", "", "", "", "", ""],
    ["", "ORGANISATIONAL", "", "", "", ""],
    ["Apply modifications on inputs17", "",
     "Modify the model inputs so that an adversarial perturbation loses its", "",
     "", "x"],
    ["", "", "effect before the input reaches the model.", "", "", ""],
    ["Ensure ML applications comply with third parties’ security requirements",
     "", "", "Third-party components used by an ML application must meet the", "",
     "x"],
    ["", "", "", "same security requirements as first-party components.", "", ""],
]

TABLE3: list[list[str | None]] = [
    ["Threats sub- threats", "", "Definition", "", "", "Stage"],
    ["Evasion", "",
     "A type of attack in which the attacker works on the ML algorithm inputs",
     "", "", "x"],
    ["", "", "to find small perturbations leading to large output errors.", "",
     "", ""],
    ["", "Data disclosure",
     "This threat refers to a leak of data manipulated by the ML application.",
     "", "", "x"],
]

# Six columns, so the widest table clears DEFINITION_END_COLUMN. Verified to
# round-trip through pdfplumber 0.11.4.
COLUMNS = [40.0, 250.0, 265.0, 280.0, 470.0, 500.0, 560.0]


def _wrap(text: str, width: int = 44) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _table(rows: list[tuple[str, str, int]]) -> tuple[
    list[tuple[float, float, str]], list[tuple[float, float, float, float]],
]:
    """(name, definition, definition_column) rows as a ruled table."""
    bands = [60.0]
    wrapped = [(_wrap(name, 40), _wrap(body), column)
               for name, body, column in rows]
    for names, bodies, _ in wrapped:
        bands.append(bands[-1] + 14 * max(len(names), len(bodies)) + 12)
    rules = [(x, bands[0], x, bands[-1]) for x in COLUMNS]
    rules += [(COLUMNS[0], y, COLUMNS[-1], y) for y in bands]
    runs: list[tuple[float, float, str]] = []
    for index, (names, bodies, column) in enumerate(wrapped):
        top = bands[index] + 11
        for offset, line in enumerate(names):
            runs.append((COLUMNS[0] + 3, top + offset * 12, line))
        for offset, line in enumerate(bodies):
            runs.append((COLUMNS[column] + 3, top + offset * 12, line))
    return runs, rules


def _three_table_pdf() -> bytes:
    """Table 3, Table 5 and Annex C on the pages the parser reads.

    Every declared name appears, because _check_declarations refuses a parse
    that does not produce all of FOOTNOTE_NAMES, every ANNEX_C_VARIANTS target
    and both ANNEX_C_ONLY names. A fixture that omitted one would fail the gate
    rather than test it.
    """
    table5_names = [*FOOTNOTE_NAMES, *ANNEX_C_VARIANTS.values()]
    table5 = [
        (name,
         f"A definition for this control that runs long enough to stand as a "
         f"statement, number {index}.",
         1 if index % 2 else 2)
        for index, name in enumerate(table5_names)
    ]
    table3 = [
        (name,
         f"A threat definition long enough to stand as a statement, "
         f"number {index}.",
         2)
        for index, name in enumerate(("Evasion", "Poisoning"))
    ]
    annex = [
        (name,
         f"An Annex C implementation example long enough to serve as a "
         f"statement, number {index}.",
         1)
        for index, name in enumerate([*ANNEX_C_VARIANTS, *ANNEX_C_ONLY])
    ]
    pages: list[list[tuple[float, float, str]]] = [[] for _ in range(44)]
    rules: list[list[tuple[float, float, float, float]]] = [[] for _ in range(44)]
    for page, rows in (
        (TABLE3_PAGES.start, table3),
        (TABLE5_PAGES.start, table5),
        (ANNEX_C_PAGES.start, annex),
    ):
        pages[page], rules[page] = _table(rows)
    return build_pdf(pages, rules)


class TestRowsToUnits:
    def test_a_definition_in_column_three_is_not_lost(self) -> None:
        units = dict(EnisaParser.rows_to_units(TABLE5, 1, TABLE5_BANNERS))
        key = "Ensure ML applications comply with third parties’ security requirements"
        assert "Third-party components" in units[key]
        assert "first-party components" in units[key]

    def test_continuation_rows_join_the_unit_above(self) -> None:
        units = dict(EnisaParser.rows_to_units(TABLE5, 1, TABLE5_BANNERS))
        assert units["Apply modifications on inputs17"].endswith(
            "before the input reaches the model."
        )

    def test_category_banners_are_not_units(self) -> None:
        names = [
            n for n, _ in EnisaParser.rows_to_units(TABLE5, 1, TABLE5_BANNERS)
        ]
        assert "ORGANISATIONAL" not in names
        assert "Security controls" not in names

    def test_a_threat_and_a_sub_threat_are_both_units(self) -> None:
        names = [
            n for n, _ in EnisaParser.rows_to_units(TABLE3, 2, TABLE3_BANNERS)
        ]
        assert names == ["Evasion", "Data disclosure"]

    def test_no_unit_extracts_with_an_empty_definition(self) -> None:
        for rows, columns, banners in (
            (TABLE5, 1, TABLE5_BANNERS), (TABLE3, 2, TABLE3_BANNERS),
        ):
            for name, body in EnisaParser.rows_to_units(rows, columns, banners):
                assert body, name

    def test_the_banner_argument_is_load_bearing(self) -> None:
        """Left at its default, the header row becomes a unit.

        The previous version of this file omitted the argument in every call
        and then asserted the header was absent. Stating the failure mode here
        keeps the default from quietly becoming the tested path again.
        """
        names = [n for n, _ in EnisaParser.rows_to_units(TABLE5, 1)]
        assert "Security controls" in names


class TestNameNormalisation:
    def test_a_fused_footnote_digit_is_removed(self) -> None:
        assert EnisaParser.normalise_name(
            EnisaParser._clean("Apply modifications on inputs17")
        ) == "apply modifications on inputs"

    def test_a_curly_apostrophe_matches_the_ascii_one(self) -> None:
        assert EnisaParser.normalise_name(
            "Ensure ML applications comply with third parties’ security requirements"
        ) == EnisaParser.normalise_name(
            "Ensure ML applications comply with third parties' security requirements"
        )

    def test_an_ellipsis_matches_three_periods(self) -> None:
        assert EnisaParser.normalise_name(
            "Use of adversarial examples crafted in white or grey box "
            "conditions (e.g. FGSM…)"
        ) == EnisaParser.normalise_name(
            "Use of adversarial examples crafted in white or grey box "
            "conditions (e.g. FGSM...)"
        )


class TestSyntheticPdf:
    """parse() through pdfplumber, with no dependency on data/raw."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> EnisaParser:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(_three_table_pdf())
        instance = EnisaParser(raw_dir=raw, output_dir=tmp_path / "out")
        instance.expected_sha256 = None
        instance.expected_table5_units = len(FOOTNOTE_NAMES) + len(ANNEX_C_VARIANTS)
        instance.expected_table3_units = 2
        instance.expected_count = (
            instance.expected_table5_units + len(ANNEX_C_ONLY) + 2
        )
        return instance

    def test_all_three_tables_are_read(self, parser: EnisaParser) -> None:
        controls = parser.parse()
        levels = [c.hierarchy_level for c in controls]
        assert levels.count("threat") == 2
        assert levels.count("control") == parser.expected_count - 2

    def test_the_annex_c_only_controls_are_marked_synthetic(
        self, parser: EnisaParser,
    ) -> None:
        controls = {c.title: c for c in parser.parse()}
        for name in ANNEX_C_ONLY:
            metadata = controls[name].metadata
            assert metadata is not None
            assert (metadata[CONTROL_TEXT_ORIGIN_METADATA_KEY]
                    == CONTROL_TEXT_ORIGIN_SYNTHETIC)

    def test_a_footnote_digit_is_gone_from_the_stored_title(
        self, parser: EnisaParser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert "Apply modifications on inputs" in titles
        assert "Apply modifications on inputs17" not in titles

    def test_an_annex_c_variant_lands_as_an_alternate_title(
        self, parser: EnisaParser,
    ) -> None:
        controls = {c.title: c for c in parser.parse()}
        target = controls["Include ML applications in asset management processes"]
        assert target.metadata is not None
        assert ("Include ML applications into asset management processes"
                in target.metadata["alt_titles"])

    def test_a_short_table_five_is_refused(self, parser: EnisaParser) -> None:
        """The band would accept 45 of 50. The unit counts do not."""
        parser.expected_table5_units += 1
        with pytest.raises(ValueError, match="Table 5"):
            parser.parse()

    def test_run_writes(self, parser: EnisaParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == parser.expected_count
        assert [s.path for s in output.source_files] == [SOURCE_FILE]


class TestRun:
    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = EnisaParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 50
        levels = {c.hierarchy_level for c in output.controls}
        assert levels == {"control", "threat"}
        assert [s.path for s in output.source_files] == [SOURCE_FILE]

    def test_the_two_annex_c_only_controls_are_present(
        self, tmp_path: Path,
    ) -> None:
        parser = EnisaParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        titles = {c.title for c in output.controls}
        assert "Ensure reliable sources are used" in titles
        assert (
            "Use methods to clean the training dataset from suspicious samples"
            in titles
        )
```

- [ ] **Step 2: Run the test to verify it fails, and prove the banner fix**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_enisa.py -q
```

Expected: FAIL, `ModuleNotFoundError`. After Step 3, all of `TestRowsToUnits` must pass, including
`test_the_banner_argument_is_load_bearing`. If that last one fails, `rows_to_units` grew a non-empty default
for `banners` and the four tests above it stopped testing anything.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_enisa.py: create

"""Parser for ENISA's Securing Machine Learning Algorithms.

There is no control identifier anywhere in this source, which is why OpenCRE's
own extraction degraded: 40 of the 68 curated links carry the literal string
"Table 5:" as their section_id and 18 carry "Table 3:". The join is therefore
the control NAME, and the name is what this parser has to get exactly right.

Three tables are read. Table 5 (pages 20-26) gives 35 security controls with
their definitions. Table 3 (pages 15-16) gives 13 threats and sub-threats, and
they are emitted too: 20 of the 68 curated links target them, including
Poisoning and Evasion, the two most-linked entries in the framework. Annex C
(pages 39-43) supplies the two controls that Table 5's extraction does not
reach, which brings the control count to the 37 the document states in its own
text.

Extraction is per row, not per page. pdfplumber puts a definition in column 2
on some rows and column 3 on others, so a per-page densest-column heuristic
loses whichever rows are in the other column. Here the name is the first
`name_columns` cells joined, the definition is columns `name_columns` through 4
joined with lone lifecycle `x` marks dropped, and a row with no name is a
continuation of the unit above it. Under that rule no unit extracts with an
empty definition.

`banners` has an empty default and every caller passes one. Left at the
default, the table's own header row becomes a unit named "Security controls".
The default is kept empty rather than filled so a caller that forgets shows up
as an extra unit in the count rather than as a silently different filter.

Names are normalised before they are stored, not at lookup time, because
ProseIndex matches a link's section_name against the stored title verbatim.
Three defects are corrected: NFKD folding plus an explicit punctuation map for
curly quotes, dashes and the ellipsis; and removal of a footnote digit fused
onto the end of a name. The footnote names are DECLARED rather than found by a
blanket regex, so a name that legitimately ends in a digit cannot be damaged
and a source refresh that moves the footnotes fails this parser instead of
quietly renaming a control.

Annex C's six variant spellings of Table 5 names are registered as alt_titles
rather than emitted as controls. Emitting them would add six anchors that are
near-duplicates of six existing ones, which is the collapse the corpus report
exists to make visible.

The two Annex-C-only controls take their statement from Annex C's
implementation-example column. That column says how to satisfy a control rather
than what it is, and it is the only text the source gives those two in a table
this parser can read. Six of the 68 links land there, so both controls are
marked anchor_source = synthetic and the report counts those six separately
from the 62 that land on a real definition.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from io import BytesIO
from typing import Any, ClassVar, Final

import pdfplumber

from tract.config import (
    CONTROL_TEXT_ORIGIN_METADATA_KEY,
    CONTROL_TEXT_ORIGIN_SYNTHETIC,
)
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "enisa_securing_ml_algorithms.pdf"
SOURCE_SHA256: Final[str] = (
    "4de967bbdf92a01339ae449b7d305b8ff266d7f16ed0a7d92a711ca20e20f087"
)

# 0-indexed page ranges, verified against the pinned PDF.
TABLE3_PAGES: Final[range] = range(14, 16)
TABLE5_PAGES: Final[range] = range(19, 26)
ANNEX_C_PAGES: Final[range] = range(38, 43)

# Where the lifecycle-stage columns begin. Everything from name_columns to
# here is definition text.
DEFINITION_END_COLUMN: Final[int] = 5

# Row names that are banners rather than units.
TABLE5_BANNERS: Final[tuple[str, ...]] = (
    "Security controls", "ORGANISATIONAL", "TECHNICAL", "SPECIFIC TO ML",
)
TABLE3_BANNERS: Final[tuple[str, ...]] = ("Threats",)
ANNEX_C_BANNERS: Final[tuple[str, ...]] = ("Security controls",)

# Table 5 names carrying a footnote reference fused onto the last word.
# Declared, hand-verified against the pinned PDF, and checked at parse time.
# A blanket trailing-digit strip would also damage a name that legitimately
# ends in a number.
FOOTNOTE_NAMES: Final[dict[str, str]] = {
    "Include ML applications into detection and response to security incident "
    "processes15":
        "Include ML applications into detection and response to security "
        "incident processes",
    "Add some adversarial examples to the training dataset16":
        "Add some adversarial examples to the training dataset",
    "Apply modifications on inputs17":
        "Apply modifications on inputs",
    "Reduce the information given by the model19":
        "Reduce the information given by the model",
    "Use less easily transferable models20":
        "Use less easily transferable models",
}

# Annex C spellings of a Table 5 control, keyed by the Annex C name.
# Registered as alt_titles on the Table 5 control. Hand-verified.
ANNEX_C_VARIANTS: Final[dict[str, str]] = {
    "Apply a RBAC model, respecting the least privilege principle":
        "Apply a RBAC model, respecting the least privileged principle",
    "Apply documentation requirements to Artificial Intelligence projects":
        "Apply documentation requirements to AI projects",
    "Ensure appropriate protection are deployed for test environments as well":
        "Ensure appropriate protection is deployed for test environments",
    "Ensure ML applications comply with identity management, authentication "
    "and access control policies":
        "Ensure ML applications comply with identity management, "
        "authentication, and access control policies",
    "Include ML applications into asset management processes":
        "Include ML applications in asset management processes",
    "Use federated learning to minimise risk of data breaches":
        "Use federated learning to minimize risk of data breaches",
}

# Controls Table 5's extraction does not reach; their statement comes from
# Annex C's implementation column, which is the only text the source gives them
# in a table this parser can read. Marked synthetic for that reason.
ANNEX_C_ONLY: Final[tuple[str, ...]] = (
    "Ensure reliable sources are used",
    "Use methods to clean the training dataset from suspicious samples",
)

_PUNCTUATION: Final[dict[str, str]] = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class EnisaParser(BaseParser):
    framework_id: ClassVar[str] = "enisa"
    framework_name: ClassVar[str] = "ENISA"
    version: ClassVar[str] = "2021-12"
    source_url: ClassVar[str] = (
        "https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms"
    )
    mapping_unit_level: ClassVar[str] = "control"
    # 35 Table 5 controls + 2 recovered from Annex C = the 37 the document
    # states, plus 13 Table 3 threats. [measured]
    expected_count: ClassVar[int] = 50
    # COUNT_TOLERANCE is 10%, so the band around 50 is 45 to 55 and a parser
    # that lost five units would write in silence. These two are the
    # structural check that beats the band. Overridable so a synthetic PDF can
    # drive parse() in CI. [measured]
    expected_table5_units: ClassVar[int] = 35
    expected_table3_units: ClassVar[int] = 13
    fetched_date: ClassVar[str] = "2026-08-15"
    # All 50 units clear HONEST_PROSE_MIN_CHARS, the shortest definition is 80
    # characters, and none equals its own name, so the measured value is
    # exactly 1.0. A floor of 0.96 would have let two units decay to a bare
    # title without firing. [measured end to end on the pinned PDF]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            table5 = self._collect(pdf, TABLE5_PAGES, 1, TABLE5_BANNERS)
            table3 = self._collect(pdf, TABLE3_PAGES, 2, TABLE3_BANNERS)
            annex_c = self._collect(pdf, ANNEX_C_PAGES, 1, ANNEX_C_BANNERS)

        self._check_shape(table5, table3)
        self._check_declarations(table5, annex_c)
        controls = self._build(table5, table3, annex_c)
        logger.info(
            "%s: %d controls and %d threats, %d Annex C alternate spelling(s), "
            "%d synthesised statement(s)",
            self.framework_id,
            sum(1 for c in controls if c.hierarchy_level == "control"),
            sum(1 for c in controls if c.hierarchy_level == "threat"),
            len(ANNEX_C_VARIANTS), len(ANNEX_C_ONLY),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not "
                f"the pinned {self.expected_sha256}. FOOTNOTE_NAMES, "
                f"ANNEX_C_VARIANTS, ANNEX_C_ONLY and the three page ranges all "
                f"quote this exact document."
            )

    def _check_shape(
        self, table5: list[tuple[str, str]], table3: list[tuple[str, str]],
    ) -> None:
        """Refuse a parse whose table sizes moved, before the band hides it.

        Raises:
            ValueError: If either table yields a different number of units
                than this parser declares.
        """
        if len(table5) != self.expected_table5_units:
            raise ValueError(
                f"{self.framework_id}: Table 5 yielded {len(table5)} unit(s), "
                f"expected {self.expected_table5_units}. COUNT_TOLERANCE puts "
                f"the band around 50 at 45 to 55, so a loss of five controls "
                f"would write without a word."
            )
        if len(table3) != self.expected_table3_units:
            raise ValueError(
                f"{self.framework_id}: Table 3 yielded {len(table3)} unit(s), "
                f"expected {self.expected_table3_units}. Twenty of the 68 "
                f"curated links target a Table 3 entry."
            )

    def _collect(
        self, pdf: Any, pages: range, name_columns: int,
        banners: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        """(name, definition) for one table, across its pages.

        `pdf` is typed Any rather than pdfplumber.PDF. pyproject silences
        pdfplumber with ignore_missing_imports, which makes the module Any, so
        the attribute annotation raises `Name "pdfplumber.PDF" is not defined`
        under mypy --strict. [measured]
        """
        rows: list[list[str | None]] = []
        for page_number in pages:
            if page_number >= len(pdf.pages):
                break
            tables = pdf.pages[page_number].extract_tables()
            if not tables:
                continue
            widest = max(tables, key=lambda t: max(len(r) for r in t))
            if max(len(row) for row in widest) < DEFINITION_END_COLUMN:
                continue
            rows.extend(widest)
        units = self.rows_to_units(rows, name_columns, banners)
        if not units:
            raise ValueError(
                f"{self.framework_id}: no table rows on pages "
                f"{pages.start}-{pages.stop} of {SOURCE_FILE}. "
                f"extract_text() interleaves these tables and returns rotated "
                f"headers as reversed character runs, so falling back to it "
                f"would produce garbage rather than a smaller result."
            )
        return units

    @classmethod
    def rows_to_units(
        cls,
        rows: list[list[str | None]],
        name_columns: int,
        banners: tuple[str, ...] = (),
    ) -> list[tuple[str, str]]:
        """(name, definition) per named row, continuations merged upward.

        Merging is per row rather than per page because pdfplumber places a
        definition in column 2 on some rows and column 3 on others.

        `banners` defaults to empty and every caller in this module passes one.
        A caller that forgets gets the table's header row back as a unit, which
        the unit-count check in _check_shape then refuses.
        """
        units: list[tuple[str, list[str]]] = []
        for row in rows:
            cells = [(cell or "").strip() for cell in row]
            padded = cells + [""] * DEFINITION_END_COLUMN
            name = " ".join(c for c in padded[:name_columns] if c).strip()
            body = " ".join(
                c for c in padded[name_columns:DEFINITION_END_COLUMN]
                if c and c.lower() != "x"
            ).strip()
            if name and any(name.startswith(b) for b in banners):
                continue
            if not name:
                if body and units:
                    units[-1][1].append(body)
                continue
            units.append((_WHITESPACE.sub(" ", name), [body] if body else []))
        return [(name, " ".join(parts).strip()) for name, parts in units]

    @staticmethod
    def normalise_name(name: str) -> str:
        """The comparison key for a control or threat name.

        NFKD folds compatibility forms, the punctuation map turns the source's
        typographic quotes, dashes and ellipsis into the ASCII OpenCRE stores,
        and a declared footnote reference is removed by _clean before this is
        called, not by a trailing-digit regex here.
        """
        folded = unicodedata.normalize("NFKD", name)
        for source, target in _PUNCTUATION.items():
            folded = folded.replace(source, target)
        return _WHITESPACE.sub(" ", folded).strip().lower()

    @classmethod
    def _clean(cls, name: str) -> str:
        """A stored title: footnote reference removed, punctuation as ASCII."""
        stripped = FOOTNOTE_NAMES.get(name, name)
        folded = unicodedata.normalize("NFKD", stripped)
        for source, target in _PUNCTUATION.items():
            folded = folded.replace(source, target)
        return _WHITESPACE.sub(" ", folded).strip()

    def _check_declarations(
        self, table5: list[tuple[str, str]], annex_c: list[tuple[str, str]],
    ) -> None:
        """Refuse declarations that no longer match the extracted tables.

        Raises:
            ValueError: If a declared footnote name, Annex C variant or
                Annex-C-only control is absent from this parse.
        """
        table5_names = {name for name, _ in table5}
        missing_footnotes = sorted(set(FOOTNOTE_NAMES) - table5_names)
        if missing_footnotes:
            raise ValueError(
                f"{self.framework_id}: FOOTNOTE_NAMES declares {missing_footnotes}, "
                f"which this parse did not produce. A stale entry means a "
                f"control name still carries a footnote digit, and every link "
                f"to it silently falls back to its own title."
            )

        table5_keys = {self.normalise_name(self._clean(n)) for n, _ in table5}
        missing_targets = sorted({
            target for target in ANNEX_C_VARIANTS.values()
            if self.normalise_name(target) not in table5_keys
        })
        if missing_targets:
            raise ValueError(
                f"{self.framework_id}: ANNEX_C_VARIANTS points at Table 5 "
                f"control(s) {missing_targets} that this parse did not "
                f"produce."
            )

        annex_names = {name for name, _ in annex_c}
        missing_only = sorted(set(ANNEX_C_ONLY) - annex_names)
        if missing_only:
            raise ValueError(
                f"{self.framework_id}: ANNEX_C_ONLY declares {missing_only}, "
                f"which Annex C did not yield. Those two controls appear "
                f"nowhere in Table 5's extraction, so without them six "
                f"curated links have no anchor at all."
            )

    @classmethod
    def _build(
        cls,
        table5: list[tuple[str, str]],
        table3: list[tuple[str, str]],
        annex_c: list[tuple[str, str]],
    ) -> list[Control]:
        """Controls from Table 5 and Annex C, threats from Table 3."""
        alternates: dict[str, list[str]] = {}
        for variant, target in ANNEX_C_VARIANTS.items():
            alternates.setdefault(cls.normalise_name(target), []).append(
                cls._clean(variant)
            )

        controls: list[Control] = []
        for name, definition in table5:
            title = cls._clean(name)
            metadata: dict[str, str | list[str]] = {"table": "Table 5"}
            names = alternates.get(cls.normalise_name(title))
            if names:
                metadata["alt_titles"] = names
            controls.append(Control(
                control_id=cls._slug(title),
                title=title,
                description=definition,
                hierarchy_level="control",
                metadata=metadata,
            ))

        annex_bodies = {name: body for name, body in annex_c}
        for name in ANNEX_C_ONLY:
            title = cls._clean(name)
            controls.append(Control(
                control_id=cls._slug(title),
                title=title,
                description=annex_bodies[name],
                hierarchy_level="control",
                metadata={
                    "table": "Annex C",
                    # The statement is Annex C's implementation example, which
                    # says how to satisfy the control rather than what it is.
                    # Six curated links land here and the report counts them
                    # apart from the 62 that land on a real definition.
                    CONTROL_TEXT_ORIGIN_METADATA_KEY:
                        CONTROL_TEXT_ORIGIN_SYNTHETIC,
                },
            ))

        for name, definition in table3:
            title = cls._clean(name)
            controls.append(Control(
                control_id=cls._slug(title),
                title=title,
                description=definition,
                hierarchy_level="threat",
                metadata={"table": "Table 3"},
            ))
        return controls

    @staticmethod
    def _slug(title: str) -> str:
        """A synthetic control id.

        The source has no identifier of any kind, so this is generated. It is
        derived from the cleaned title so it is stable across re-parses of the
        same bytes, and it is never used for the join: every curated link
        carries either "Table 5:" or "Table 3:" or the name itself.
        """
        return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def main() -> None:
    EnisaParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_enisa.py -q
"$PY" -m mypy parsers/parse_enisa.py --strict
```

Both must pass. `mypy --strict` reports `Success: no issues found` with `pdf: Any` and
`error: Name "pdfplumber.PDF" is not defined [name-defined]` without it. **[measured]**

- [ ] **Step 5: Run against the real source and check the join**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# Every count below is a function of the extractor. requirements.txt pins
# 0.11.10 and this interpreter has 0.11.4. [measured]
"$PY" -c "
import pdfplumber, sys
if pdfplumber.__version__ != '0.11.4':
    sys.exit(f'pdfplumber {pdfplumber.__version__}: re-measure the ENISA row')
"
PYTHONPATH=. "$PY" parsers/parse_enisa.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py --framework enisa --json
```

Expected log line: `enisa: 37 controls and 13 threats, 6 Annex C alternate spelling(s), 2 synthesised
statement(s)`. **[measured]**

Accept only on this exact set. Every value is **[measured]**, and the whole row was reproduced end to end by
running this parser against the pinned PDF and resolving all 68 links through `ProseIndex`.

| field | value | how it can fail in each direction |
|---|---|---|
| `links` | 68 | |
| `by_title` | 68 | every link keys on the name, because `section_id` is a table placeholder. Below 68 means `normalise_name` is not being applied to the stored title |
| `by_id` | 0 | `Table 5:` and `Table 3:` are not control ids and never will be |
| `unresolved` | 0 | a 6 means the two Annex-C-only controls are missing (rate 0.9118). A 17 means normalisation is not reaching the stored title (rate 0.7500) |
| `distinct_anchors` | 33 | |
| `distinct_anchors_pre_truncation` | 33 | nothing truncates |
| `fallback_anchors` (BEFORE) | 33 | orchestrator measured. The anchor count does not move, and this task said so in advance rather than discovering it in the report. The gain is that 33 phrases become 33 paragraphs |
| `links_per_anchor` | 2.06 | |
| `truncated` | 0 | longest definition is 1,183 characters against a 2,150 budget |
| `nested_anchors` / `contained_anchors` | 0 / 0 | a 6 means the Annex C variants were emitted as controls instead of registered as alternate titles |
| `dropped_by_prose_rule` | 0 | all 50 clear `_is_prose` |
| `wrong_anchor_risk` | 0 | the id channel resolves nothing for this framework, so there is never a second answer to disagree with. The maximum attainable value is 0 and this is recorded, not asserted |
| `anchor_source_description` | 62 | |
| `anchor_source_synthetic` | 6 | the two Annex-C-only controls, 3 links each |
| `anchor_source_full_text` / `_title` | 0 / 0 | no definition exceeds `DESCRIPTION_MAX_LENGTH` |
| `distinct_hubs` | 56 | |
| `links_per_hub` | 1.21 | |
| `resolution_rate` | 1.0000 | floor 1.00 |

- [ ] **Step 6: Commit**

```bash
git add parsers/parse_enisa.py tests/test_parse_enisa.py \
        data/processed/frameworks/enisa.json data/processed/all_controls.json
git commit -m "feat: join ENISA on repaired control names across Table 3, Table 5 and Annex C"
```

---

<<<TASK 12>>>

---

### Task 12: BIML, two documents, one id space, and the collision the title channel would cause

Both PDFs mark each named risk with an inline `[category:number:label]` tag, and that tag is a real
structural delimiter. Measured on the pinned files: `ara.pdf` (BIML-78, 2020) defines **78** distinct tags at
a line start and `BIML-LLM24.pdf` (BIML-24 LLM, 2024) defines **68**, for 146 in total. **[measured, end to
end]** Every tag this framework's links need carries a body of 153 to 1,319 characters. **[measured]**

**The two documents reuse the same id space for different risks**, and OpenCRE leaves 8 of 21 links
unprefixed. Measured by exact tag-label match:

| unprefixed id | link name | resolves to | evidence |
|---|---|---|---|
| `model:2` | Trojan | ara | exact label `trojan`; LLM24's `model:2` is `improper use` |
| `raw:3` | Storage | ara | exact `storage`; LLM24's is `data feudalism` |
| `input:2` | Controlled Input Stream | ara | exact; LLM24's is `prompt injection` |
| `inference:4` | Hosting | ara | exact `hosting`; LLM24's is `stochasticity` |
| `alg:11` | Parameters | ara | exact; absent from LLM24 |
| `inference:9` | Hosting | LLM24 | exact `hosting`; absent from ara |
| `output:4` | Output Data Confidentiality | LLM24 | LLM24's is `data confidentiality`; ara's is `inscrutability` |
| `output:2` | Direct Output | **ara `output:1`** | ara's `output:1` is `direct`; ara's own `output:2` is `provenance` and LLM24's is `wrongness` |

**[measured, all eight]** The first seven are registered as `alt_ids`. The eighth is an upstream
id-versus-name conflict: the name matches ara's `output:1` exactly and the id matches a control about
provenance. It resolves by name, not by id, and the conflict is written to the repair audit. Aliasing
`output:2` onto `output:1` would assert that OpenCRE's id is a typo, which the evidence does not support any
better than the name being right.

**`distinct_anchors` is 19, not 20. Plan v2 asserted 20 and would have halted a healthy run.** The curated
link file carries `inference:9` twice, once as `BIML-24(LLM): inference:9` and once bare, and
`UNPREFIXED_IDS` routes the bare form to that same control. So the 8 unprefixed rows add only 7 anchors on
top of the 12 distinct controls the 13 prefixed rows reach. 12 + 7 = **19**, and 21 links over 19 anchors is
**1.105**, not 1.05. **[measured]**

**`truncated` is 0, not about 5.** The longest risk body is 1,999 characters against `MAX_ANCHOR_CHARS` of
2,150, and no BIML control gets `full_text`. **[measured]**

**`dropped_by_prose_rule` is 1, not about 5.** Exactly one of the 146 risks fails `_is_prose`:
`BIML-78(2020): inference:3`, whose extracted body is the two characters `.)`. **[measured]** No curated
link targets it.

**Titles must be document-scoped or the title channel destroys the join.** `Data Confidentiality` is the
`section_name` of two links naming two different risks in two different documents; `Hosting` is the name of
three links across both. Seven of the 21 rows participate in a label collision. **[measured]** With a bare
label as the title, `ProseIndex.lookup`, which tries the title first, would hand all of them one anchor,
which is the NIST AI 100-2 collapse again. Titles are therefore `f"{Label} ({document})"`, which no link name
spells, so every row goes through the id channel where the document prefix disambiguates.

**`min_prose_fraction` rises from 0.90 to 0.99.** Measured on the pinned PDFs, 145 of 146 risks clear
`HONEST_PROSE_MIN_CHARS`, giving **0.9932**. **[measured]** A floor of 0.90 permitted fourteen risks to
decay to a bare label before the gate fired. A floor of 0.99 passes at 0.9932 and fails at 144 of 146
(0.9863).

**Ceiling: 21/21 = 1.0000** **[measured]**. 13 prefixed rows by id, 7 by `alt_ids`, 1 by `alt_title`. Floor
**1.00**. Without the two declared alternates for `output:2` and `output:4` it is 19/21 = 0.9048.

**Context, not an instruction.** BIML carries 21 of 4,127 training links. The ceiling study measured human
alpha-1 at 0.572 pooled and 0.181 for CAPEC, which is 42.8% of the training graph. Effort spent here buys
0.5% of the graph. The reason to do it well is that it is the case where the title-before-id order is most
obviously wrong, not its weight.

**Licensing.** `biml` is CC-BY-SA-3.0 and CC-BY-SA-4.0 and sits in `CONDITIONAL_FRAMEWORK_IDS` under
Contract Rule 3, so `data/processed/frameworks/biml.json` routes to the gitignored licensed overlay and is
not staged.

**Files:**
- Create: `parsers/parse_biml.py`
- Create: `tests/test_parse_biml.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `BaseParser.write_repair_audit`, `alt_ids` from Task 2, `tests.synthetic_pdf.build_pdf` (Task 9).
- Produces: `BimlParser` with `framework_id = "biml"`, `framework_name = "BIML"`; `BimlParser.risks_from_text(text: str, document: str) -> list[tuple[str, str, str]]`; `BimlParser.build_controls(texts: dict[str, str], require_targets: bool = False) -> tuple[list[Control], list[dict[str, object]]]`.

**Invalidates:**
- `data/processed/licensed/all_controls.json` and its sha256, so every `corpus_sha256` recorded before this task.
- `data/processed/stopwords.json`, which Task 15 regenerates.
- Task 14's `hub_links_training.jsonl` for the 21 `biml` rows.
- `data/processed/repair_audit/biml.jsonl`, written for the first time here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse_biml.py: create

"""BIML's two documents reuse one id space, and OpenCRE leaves 8 ids unprefixed.

Measured: 'Data Confidentiality' names two different risks across the two PDFs
and 'Hosting' names three link rows. With a bare label as the title, ProseIndex
-- which resolves title before id -- gives all of them one anchor.

TestSyntheticPdf drives parse() through pdfplumber against two PDFs this file
builds, so the multi-document read, the digest branch, the declaration gate and
the audit write all run in CI, where data/raw is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsers.parse_biml import ARA, LLM24, SOURCE_FILES, BimlParser
from tests.synthetic_pdf import build_pdf

ARA_TEXT = """[raw:3:storage]
As in other systems, data may be stored and managed in an insecure fashion.
Who has access to the data pool, and why? Think about [system:8:insider] when
working on storage of any training data.

[output:1:direct]
Direct output of a model is the answer it hands the requester, and the
requester may not be the party the model was built to serve.

[inference:4:hosting]
Where the model runs decides who can reach it, and a hosted model inherits the
trust boundary of whatever hosts it.
"""

LLM24_TEXT = """[raw:3:data feudalism]
A small number of parties control the data that everyone else trains on.

[inference:9:hosting]
An LLM served by a third party puts the prompt, the completion and the system
message inside somebody else's trust boundary.
"""

# Every declared target, so require_targets has something to check rather than
# something to trip over. UNPREFIXED_IDS names five ara tags and two LLM24
# tags, and NAME_CONFLICTS names ara's output:1.
ARA_TAGS: tuple[tuple[str, str], ...] = (
    ("raw:3", "storage"),
    ("model:2", "trojan"),
    ("input:2", "controlled input stream"),
    ("inference:4", "hosting"),
    ("alg:11", "parameters"),
    ("output:1", "direct"),
    ("output:2", "provenance"),
)
LLM24_TAGS: tuple[tuple[str, str], ...] = (
    ("inference:9", "hosting"),
    ("output:4", "data confidentiality"),
    ("raw:3", "data feudalism"),
)


def _document(tags: tuple[tuple[str, str], ...]) -> bytes:
    lines: list[str] = []
    for tag, label in tags:
        lines.append(f"[{tag}:{label}]")
        lines.append(
            f"A body for {label} that runs long enough to stand as a statement,"
        )
        lines.append(
            "and it names [system:8:insider] mid-sentence as a cross-reference."
        )
        lines.append("")
    return build_pdf([[(72.0, 60.0 + n * 14, text)
                       for n, text in enumerate(lines)]])


class TestRisksFromText:
    def test_only_line_start_tags_define_a_risk(self) -> None:
        risks = BimlParser.risks_from_text(ARA_TEXT, ARA)
        assert [tag for tag, _, _ in risks] == [
            "raw:3", "output:1", "inference:4",
        ]

    def test_a_body_runs_to_the_next_definition(self) -> None:
        risks = {t: b for t, _, b in BimlParser.risks_from_text(ARA_TEXT, ARA)}
        assert "insecure fashion" in risks["raw:3"]
        assert "Direct output" not in risks["raw:3"]


class TestScopedIdentity:
    def test_control_id_carries_the_document(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        assert f"{ARA}: raw:3" in {c.control_id for c in controls}
        assert f"{LLM24}: raw:3" in {c.control_id for c in controls}

    def test_titles_cannot_collide_across_documents(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        titles = [c.title for c in controls]
        assert len(titles) == len(set(titles))
        assert f"Hosting ({ARA})" in titles
        assert f"Hosting ({LLM24})" in titles

    def test_an_unprefixed_id_is_an_alternate_on_exactly_one_document(
        self,
    ) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        holders = [
            c.control_id for c in controls
            if c.metadata and "raw:3" in (c.metadata.get("alt_ids") or [])
        ]
        assert holders == [f"{ARA}: raw:3"]

    def test_the_named_conflict_resolves_by_name(self) -> None:
        controls = BimlParser.build_controls(
            {ARA: ARA_TEXT, LLM24: LLM24_TEXT},
        )[0]
        target = next(c for c in controls if c.control_id == f"{ARA}: output:1")
        assert target.metadata is not None
        assert "Direct Output" in target.metadata["alt_titles"]
        assert not any(
            "output:2" in (c.metadata or {}).get("alt_ids", [])
            for c in controls
        )


class TestAudit:
    def test_the_conflict_is_recorded(self, tmp_path: Path) -> None:
        parser = BimlParser(output_dir=tmp_path, audit_dir=tmp_path / "audit")
        _, audit = BimlParser.build_controls({ARA: ARA_TEXT, LLM24: LLM24_TEXT})
        parser.write_repair_audit(audit)
        records = [
            json.loads(line)
            for line in (tmp_path / "audit" / "biml.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        assert any(r["opencre_section_id"] == "output:2" for r in records)


class TestDeclarations:
    def test_an_alternate_whose_target_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="alt_ids"):
            BimlParser.build_controls({ARA: ARA_TEXT}, require_targets=True)


class TestSyntheticPdf:
    """parse() through pdfplumber, across both documents, with no data/raw."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> BimlParser:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILES[ARA]).write_bytes(_document(ARA_TAGS))
        (raw / SOURCE_FILES[LLM24]).write_bytes(_document(LLM24_TAGS))
        instance = BimlParser(
            raw_dir=raw,
            output_dir=tmp_path / "out",
            audit_dir=tmp_path / "audit",
        )
        instance.expected_sha256 = None
        instance.expected_count = len(ARA_TAGS) + len(LLM24_TAGS)
        instance.expected_tags = {ARA: len(ARA_TAGS), LLM24: len(LLM24_TAGS)}
        return instance

    def test_both_documents_are_read(self, parser: BimlParser) -> None:
        controls = parser.parse()
        documents = [c.metadata["document"] for c in controls
                     if c.metadata is not None]
        assert documents.count(ARA) == len(ARA_TAGS)
        assert documents.count(LLM24) == len(LLM24_TAGS)

    def test_the_two_hosting_risks_keep_separate_titles(
        self, parser: BimlParser,
    ) -> None:
        titles = {c.title for c in parser.parse()}
        assert f"Hosting ({ARA})" in titles
        assert f"Hosting ({LLM24})" in titles

    def test_every_declared_unprefixed_id_lands(
        self, parser: BimlParser,
    ) -> None:
        from parsers.parse_biml import UNPREFIXED_IDS

        holders: dict[str, str] = {}
        for control in parser.parse():
            for alternate in (control.metadata or {}).get("alt_ids", []) or []:
                holders[alternate] = control.control_id
        assert holders == {
            unprefixed: f"{document}: {tag}"
            for unprefixed, (document, tag) in UNPREFIXED_IDS.items()
        }

    def test_a_document_short_of_its_floor_is_refused(
        self, parser: BimlParser,
    ) -> None:
        """expected_count is a floor on the SUM, so 78 + 68 also passes as
        80 + 66. The per-document floors are what beat that."""
        parser.expected_tags = {ARA: len(ARA_TAGS) + 1, LLM24: len(LLM24_TAGS)}
        with pytest.raises(ValueError, match=ARA):
            parser.parse()

    def test_run_writes_and_writes_the_audit(
        self, parser: BimlParser, tmp_path: Path,
    ) -> None:
        (tmp_path / "out").mkdir()
        output = parser.run()
        assert len(output.controls) == len(ARA_TAGS) + len(LLM24_TAGS)
        assert sorted(s.path for s in output.source_files) == sorted(
            SOURCE_FILES.values()
        )
        lines = (tmp_path / "audit" / "biml.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        assert len(lines) == 1


class TestRun:
    def test_run_writes_from_the_real_pdfs(self, tmp_path: Path) -> None:
        parser = BimlParser(output_dir=tmp_path, audit_dir=tmp_path / "audit")
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 146
        documents = [c.metadata["document"] for c in output.controls
                     if c.metadata is not None]
        assert documents.count(ARA) == 78
        assert documents.count(LLM24) == 68
        assert sorted(s.path for s in output.source_files) == [
            "BIML-LLM24.pdf", "ara.pdf",
        ]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_biml.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the parser**

```python
# parsers/parse_biml.py: create

"""Parser for the Berryville Institute of Machine Learning risk analyses.

Two documents, both required. ara.pdf is the 2020 architectural risk analysis
of machine learning systems, "BIML-78"; BIML-LLM24.pdf is the 2024 analysis of
large language models, "BIML-24(LLM)". Both mark every named risk with an
inline [category:number:label] tag, and a tag that starts a line is the risk's
definition while the same tag inside a sentence is a cross-reference.

The two documents reuse the same category:number space for different risks:
ara's raw:3 is Storage and LLM24's is data feudalism. So a control_id must
carry its document, and it is spelled exactly as OpenCRE's prefixed ids are.

Titles carry the document too, and that is the load-bearing decision here.
ProseIndex.lookup resolves the section NAME before the section id. OpenCRE
gives two different risks the name "Data Confidentiality" and three link rows
the name "Hosting"; seven of the 21 curated rows participate in a label
collision. A bare label as the title would hand all of them one anchor, which
is the same collapse the title-first order was written to fix for NIST AI
100-2. A scoped title matches no link name at all, so every row falls through
to the id channel, where the document prefix disambiguates.

Eight curated links carry an unprefixed id. Seven are resolved by exact
tag-label match against one document and only one document, and are declared
in UNPREFIXED_IDS. The eighth, output:2 "Direct Output", matches ara's
[output:1:direct] by name while ara's own output:2 is provenance and LLM24's is
wrongness. It is resolved by NAME, as an alt_title on ara's output:1, and the
id conflict is written to the repair audit. Aliasing the id instead would
assert OpenCRE made a typo, which the evidence supports no better.

One of the 21 links duplicates another target. inference:9 appears both as
"BIML-24(LLM): inference:9" and bare, and UNPREFIXED_IDS routes the bare form
to that same control, so 21 links land on 19 anchors rather than 20.
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARA: Final[str] = "BIML-78(2020)"
LLM24: Final[str] = "BIML-24(LLM)"

SOURCE_FILES: Final[dict[str, str]] = {ARA: "ara.pdf", LLM24: "BIML-LLM24.pdf"}
SOURCE_SHA256: Final[dict[str, str]] = {
    ARA: "247d7f06d8c768cc734dc84ab7004c6e4d645e91911af61002fd1743807ef312",
    LLM24: "1a41ba1a9218e6aecdcab46d2cc6cf8a3b99f6cc1c98a3683bf3a6e4964e955f",
}

TAG: Final[re.Pattern[str]] = re.compile(r"\[([a-z]+):(\d+):([^\]]+)\]")

# Unprefixed OpenCRE ids, each resolved by exact tag-label match against one
# document and only one. Hand-verified against both PDFs; see the module
# docstring for the evidence per entry.
UNPREFIXED_IDS: Final[dict[str, tuple[str, str]]] = {
    "model:2": (ARA, "model:2"),
    "raw:3": (ARA, "raw:3"),
    "input:2": (ARA, "input:2"),
    "inference:4": (ARA, "inference:4"),
    "alg:11": (ARA, "alg:11"),
    "inference:9": (LLM24, "inference:9"),
    "output:4": (LLM24, "output:4"),
}

# The one row whose id and name disagree upstream. Resolved by name.
NAME_CONFLICTS: Final[dict[str, tuple[str, str, str]]] = {
    "Direct Output": (
        ARA, "output:1",
        "OpenCRE's section_id output:2 names ara's provenance risk and "
        "LLM24's wrongness risk; its section_name matches ara's "
        "[output:1:direct] exactly. Resolved by name, because the name is "
        "the only side of the row that matches anything in either document.",
    ),
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class BimlParser(BaseParser):
    framework_id: ClassVar[str] = "biml"
    framework_name: ClassVar[str] = "BIML"
    version: ClassVar[str] = "BIML-78 (2020) + BIML-24 LLM (2024)"
    source_url: ClassVar[str] = "https://berryvilleiml.com/results/"
    mapping_unit_level: ClassVar[str] = "risk"
    # 78 distinct definitional tags in ara.pdf and 68 in BIML-LLM24.pdf.
    # [measured] Declared as a floor: the tag vocabulary is what each document
    # names, and a re-pin that adds a risk should not fail the parser.
    expected_count: ClassVar[int] = 146
    expected_count_is_floor: ClassVar[bool] = True
    # A floor on the sum is satisfied by 80 + 66 as readily as by 78 + 68, so
    # one document could lose a dozen risks while the other gained them. These
    # per-document floors are the structural check. Overridable so a synthetic
    # pair of PDFs can drive parse() in CI. [measured]
    expected_tags: ClassVar[dict[str, int]] = {ARA: 78, LLM24: 68}
    fetched_date: ClassVar[str] = "2026-08-15"
    # 145 of 146 risk bodies clear HONEST_PROSE_MIN_CHARS, giving 0.9932. The
    # exception is BIML-78(2020) inference:3, whose extracted body is the two
    # characters ".)" and which no curated link targets. A floor of 0.90 would
    # have let fourteen risks decay before firing. [measured]
    min_prose_fraction: ClassVar[float] = 0.99
    expected_sha256: ClassVar[dict[str, str] | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        texts: dict[str, str] = {}
        for document, filename in SOURCE_FILES.items():
            payload = self.read_source_bytes(filename)
            self._check_digest(document, filename, payload)
            with pdfplumber.open(BytesIO(payload)) as pdf:
                texts[document] = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        controls, audit = self.build_controls(texts, require_targets=True)
        self._check_shape(controls)
        self.write_repair_audit(audit)
        for record in audit:
            logger.warning(
                "%s: %s resolved by name to %s, not by its own id: %s",
                self.framework_id, record["opencre_section_id"],
                record["resolved_to"], record["reason"],
            )
        logger.info(
            "%s: %d risks (%s)", self.framework_id, len(controls),
            ", ".join(
                f"{document} {sum(1 for c in controls if c.control_id.startswith(document))}"
                for document in SOURCE_FILES
            ),
        )
        return controls

    def _check_digest(self, document: str, filename: str, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match the pin for `document`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        expected = self.expected_sha256[document]
        if actual != expected:
            raise ValueError(
                f"{self.framework_id}: {filename} has sha256 {actual}, not "
                f"the pinned {expected}. UNPREFIXED_IDS and NAME_CONFLICTS "
                f"were resolved by comparing tag labels across these exact "
                f"two documents; a different revision can move a label onto a "
                f"different number and silently re-point a link."
            )

    def _check_shape(self, controls: list[Control]) -> None:
        """Refuse a parse where one document lost risks and the other gained.

        Raises:
            ValueError: If a document yields fewer tags than its floor.
        """
        for document, floor in sorted(self.expected_tags.items()):
            found = sum(
                1 for c in controls if c.control_id.startswith(f"{document}: ")
            )
            if found < floor:
                raise ValueError(
                    f"{self.framework_id}: {document} yielded {found} risk(s), "
                    f"below its floor of {floor}. expected_count is a floor on "
                    f"the sum, so a shortfall here can be hidden by a surplus "
                    f"in the other document."
                )

    @classmethod
    def risks_from_text(
        cls, text: str, document: str,
    ) -> list[tuple[str, str, str]]:
        """(tag, label, body) per definitional tag, in document order.

        A tag that starts a line defines a risk; the same tag inside a sentence
        is a cross-reference and is left alone. The body runs to the next
        definitional tag.
        """
        lines = text.split("\n")
        found: list[tuple[str, str, list[str]]] = []
        for line in lines:
            stripped = line.strip()
            match = TAG.match(stripped)
            if match is None:
                if found:
                    found[-1][2].append(stripped)
                continue
            found.append((
                f"{match.group(1)}:{match.group(2)}",
                _WHITESPACE.sub(" ", match.group(3)).strip(),
                [stripped[match.end():].strip()],
            ))
        seen: set[str] = set()
        risks: list[tuple[str, str, str]] = []
        for tag, label, body in found:
            if tag in seen:
                continue
            seen.add(tag)
            risks.append((tag, label, _WHITESPACE.sub(" ", " ".join(body)).strip()))
        return risks

    @classmethod
    def build_controls(
        cls, texts: dict[str, str], require_targets: bool = False,
    ) -> tuple[list[Control], list[dict[str, object]]]:
        """Document-scoped controls, plus the audit for the name conflict.

        Raises:
            ValueError: If require_targets is set and a declared alternate id
                or name conflict points at a tag this parse did not produce.
        """
        alt_ids: dict[str, list[str]] = {}
        for unprefixed, (document, tag) in UNPREFIXED_IDS.items():
            alt_ids.setdefault(f"{document}: {tag}", []).append(unprefixed)

        alt_titles: dict[str, list[str]] = {}
        audit: list[dict[str, object]] = []
        for name, (document, tag, reason) in NAME_CONFLICTS.items():
            alt_titles.setdefault(f"{document}: {tag}", []).append(name)
            audit.append({
                "opencre_section_id": "output:2",
                "opencre_section_name": name,
                "resolved_to": f"{document}: {tag}",
                "resolved_by": "section_name",
                "reason": reason,
            })

        controls: list[Control] = []
        for document in sorted(texts):
            for tag, label, body in cls.risks_from_text(texts[document], document):
                control_id = f"{document}: {tag}"
                metadata: dict[str, str | list[str]] = {
                    "document": document, "tag": tag, "label": label,
                }
                if control_id in alt_ids:
                    metadata["alt_ids"] = alt_ids[control_id]
                if control_id in alt_titles:
                    metadata["alt_titles"] = alt_titles[control_id]
                controls.append(Control(
                    control_id=control_id,
                    # Scoped so no OpenCRE section_name can match it. See the
                    # module docstring: seven of 21 rows share a label.
                    title=f"{label.title()} ({document})",
                    description=body or label,
                    hierarchy_level="risk",
                    parent_id=tag.split(":")[0],
                    metadata=metadata,
                ))

        cls._check_targets(
            {c.control_id for c in controls}, alt_ids, alt_titles, require_targets,
        )
        return controls, audit

    @staticmethod
    def _check_targets(
        control_ids: set[str],
        alt_ids: dict[str, list[str]],
        alt_titles: dict[str, list[str]],
        required: bool,
    ) -> None:
        """Refuse declarations pointing at tags this parse did not produce.

        Raises:
            ValueError: If any declared target is absent.
        """
        if not required:
            return
        missing_ids = sorted(set(alt_ids) - control_ids)
        if missing_ids:
            raise ValueError(
                f"biml: UNPREFIXED_IDS declares alt_ids on {missing_ids}, "
                f"which this parse did not produce. Seven curated links reach "
                f"their risk only through that channel; a stale entry leaves "
                f"them resolving to nothing."
            )
        missing_titles = sorted(set(alt_titles) - control_ids)
        if missing_titles:
            raise ValueError(
                f"biml: NAME_CONFLICTS declares alt_titles on "
                f"{missing_titles}, which this parse did not produce."
            )


def main() -> None:
    BimlParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_biml.py -q
"$PY" -m mypy parsers/parse_biml.py --strict
```

`mypy --strict` on this file is clean as written. **[measured]**

- [ ] **Step 5: Run against the real source and check the join**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# The 78 + 68 tag split and the 1,999-character maximum body were measured on
# 0.11.4, while requirements.txt pins 0.11.10. [measured]
"$PY" -c "
import pdfplumber, sys
if pdfplumber.__version__ != '0.11.4':
    sys.exit(f'pdfplumber {pdfplumber.__version__}: re-measure the BIML row')
"
PYTHONPATH=. "$PY" parsers/parse_biml.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py --framework biml --json
```

Expected: `biml: 146 risks (BIML-78(2020) 78, BIML-24(LLM) 68)` and one `output:2 resolved by name` warning.
**[measured]**

Accept only on this exact set. Every value is **[measured]**, and the whole row was reproduced end to end by
running this parser against both pinned PDFs and resolving all 21 links through `ProseIndex` with the
`alt_ids` channel in place.

| field | value | how it can fail in each direction |
|---|---|---|
| `links` | 21 | |
| `by_title` | **1** | the `Direct Output` conflict, and nothing else. A 7 or more means the titles are not document-scoped and the label collision is back. The rate would still read 1.0000 while `distinct_anchors` fell to 17, so this column is the only place it shows |
| `by_id` | 20 | 13 prefixed rows and 7 through `alt_ids` |
| `unresolved` | 0 | a 2 means the `alt_ids` channel from Task 2 is absent (rate 0.9048) |
| `distinct_anchors` | **19** | 12 controls from the 13 prefixed rows plus 7 more from the 8 unprefixed ones. `inference:9` appears prefixed and bare and both route to one control. Plan v2 asked for 20 |
| `distinct_anchors_pre_truncation` | 19 | nothing truncates |
| `fallback_anchors` (BEFORE) | 17 | orchestrator measured. A gain of 2, because `Hosting` and `Data Confidentiality` stop being one anchor each |
| `links_per_anchor` | 1.105 | not 1.05 |
| `truncated` | **0** | longest body is 1,999 characters against a 2,150 budget |
| `nested_anchors` / `contained_anchors` | 0 / 0 | |
| `dropped_by_prose_rule` | **1** | `BIML-78(2020): inference:3`, body `.)`. No curated link targets it |
| `wrong_anchor_risk` | 0 | the one title hit lands on `output:1`, whose id the link does not carry, so the id channel returns nothing to disagree with. The attainable value here is 0 or 1 and 0 is the measurement, not a tautology |
| `anchor_source_description` | 21 | no risk body exceeds `DESCRIPTION_MAX_LENGTH` |
| `anchor_source_full_text` / `_title` / `_synthetic` | 0 / 0 / 0 | |
| `distinct_hubs` | 11 | 1.91 links per hub, the densest in my range |
| `links_per_hub` | 1.91 | |
| `resolution_rate` | 1.0000 | floor 1.00 |

- [ ] **Step 6: Confirm the overlay routing and the audit file, then commit**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
# Ruling R10: this framework is CC BY-SA and is TRACKED. Seven other CC BY-SA
# frameworks were already tracked and published, so treating this one
# differently was defensible on no reading. LICENSES/, the NOTICE modification
# statement and one licence declaration across the published artifacts discharge
# the attribution and notice obligations. Assert it is NOT in the overlay, so a
# future tier change that silently recaptures it fails here.
"$PY" -c "
from tract.config import OVERLAY_FRAMEWORK_IDS
assert 'biml' not in OVERLAY_FRAMEWORK_IDS, 'tier changed under this task; stop'
print('routing: tracked')
"
git check-ignore -q data/processed/frameworks/biml.json \
  && { echo "biml.json is ignored but R10 tracks it; stop"; exit 1; }
git check-ignore -v data/processed/repair_audit/biml.jsonl \
  || { echo "audit NOT IGNORED. stop and fix .gitignore before committing"; exit 1; }
git add parsers/parse_biml.py tests/test_parse_biml.py \
        data/processed/frameworks/biml.json data/processed/all_controls.json
git commit -m "feat: scope BIML risks to their document so two id spaces stop colliding"
```

---

<<<TASK 13>>>

---

### Task 13: ETSI, restricted, coarse by construction, and honest about it

**ETSI is in `RESTRICTED_FRAMEWORK_IDS`.** Its processed JSON is already gitignored (`.gitignore:37`) and
`parsers/merge_all_controls.py` routes it to the gitignored `data/processed/licensed/` overlay. Its prose
must not appear in any tracked file, in any test fixture, in any commit message, or in this plan. Every
fixture below is synthetic and every assertion about the real document is a negative one, so no ETSI
sentence, heading or technique description enters git. `data/processed/repair_audit/` is gitignored for the
same reason.

**A real parser defect, and the only one the premortem found in a body.** The `CLAUSE` pattern
`^([5-7](?:\.\d+){0,3})\s+(\S.{2,80})$` matches running page footers. `extract_text()` renders every page
footer as `<page number> <document identifier>`, so pages 5, 6 and 7 are the three whose number falls in
`[5-7]`, and all three present as clauses `5`, `6` and `7` whose heading is the document identifier.
Reproduced against the pinned PDF (sha256 `46c2b6b8...`): clauses 5, 6 and 7 all captured the identifier as
their heading, with bodies of 2,765, 3,169 and **22,639** characters of table of contents, bibliography and
section-4 summary tables. **[measured]** The change-history table adds a fourth false match of the same
shape. **[measured]** The real headings at lines 535, 728 and 1142 were discarded by
`if match.group(1) in seen: continue`, because the footers come first in page order.

Every gate in plan v2 passed on that output. `expected_count = 25` was satisfied, because 25 distinct clause
numbers matched either way. `min_prose_fraction = 1.0` passed, because the garbage is long. The corpus
report was blind, because no curated link targets a bare `5`, `6` or `7`. And the processed file is
gitignored, so no reviewer would ever see it in a diff. This is ledger lesson 7's defect class inside a task
whose self-review claimed no text-moving transform.

**The fix, and the reason it is three parts and not one.** A furniture guard rejects a heading that is the
document identifier or a change-history date. The duplicate branch changes from a silent `continue` to a
`raise`, so the next furniture shape that slips through fails the parser rather than winning the slot. And a
test asserts on the synthetic fixture that a clause whose page carries a footer keeps its own heading.
Verified: with the guard, 25 clause numbers match, no number matches twice, and clauses 5, 6 and 7 carry
their real headings with rolled-up bodies of 17,657, 38,418 and 2,776 characters. **[measured]** ETSI was
the only text-mode PDF parser in this batch with no page-furniture guard. `parse_iso_27001.py` has
`PAGE_FURNITURE`, ENISA and NIST SSDF use `extract_tables()`, and BIML requires a tag at line start.

**The technique names are not structural.** 36 curated links carry 24 distinct `section_name` values over 16
distinct `section_id` values, 27 distinct pairs. **[measured]** All 24 names appear verbatim in the PDF
text, but only 2 of them are clause headings and only 1 is a bullet lead phrase before a colon; 9 appear
mid-sentence only. **[measured, premortem]** A technique-level parser would have to guess sentence
boundaries around a name that occurs 1 to 29 times across the document. That is prose heuristics, and ledger
lesson 7 says a transform that synthesises text has to fail closed rather than guess.

**Ruling: clause-level mapping units.** One control per numbered clause in sections 5 through 7, 25 of them
**[measured]**, with the clause's own text, rolled up from its descendants when the clause has none of its
own. Seven clauses have no body of their own and roll up: `5`, `5.2`, `5.3`, `6`, `6.2`, `6.3` and `6.4`.
**[measured]** Only leaf clauses contribute text to a roll-up, because an empty parent contributes an empty
string, so nothing is duplicated. Three of the seven are resolved anchors, so 4 of the 36 links land on
assembled text and carry `anchor_source_synthetic`. **[measured]**

**`by_title` is 5, not 2. Plan v2 asserted 2 and would have halted a healthy run.** Two rows carry a
technique name where a clause number belongs, and those two are the declared alternates. Two more rows carry
a `section_name` that is also a clause's own heading, so the title channel answers them first and lands on
the clause their own id names. A fifth row, `section_id` `6.3.1` with `section_name` naming clause 6.3's
heading, resolves by title to 6.3. **[measured, all five]**

**`wrong_anchor_risk` is 1, not 0, and the assertion below says so.** That fifth row is the one that counts:
the title channel answers with clause 6.3 while its own `section_id` names 6.3.1, and the two texts differ.
**[measured]** The divergence is benign in direction, because 6.3's rolled-up body opens with 6.3.1's entire
text and then continues, so the link gets a superset of what its id names. It is left standing and recorded.
A 0 on this column means the two clause-heading rows stopped matching, and a 2 or more means a name that
spans two clauses was registered as an alternate.

**And almost no `alt_titles`, which is the opposite of what it first looks like.** The `control_id` is the
clause number, and 34 of the 36 curated links carry a clause number in `section_id`, so those 34 resolve
through the id channel with nothing declared at all. Registering all 24 technique names as alternates would
be actively wrong: three names span two clauses each. **[measured]** Because `lookup` tries the title first,
a name registered on one clause would answer the link that named the other, which is a wrong anchor rather
than a fallback. So exactly **two** alternates are declared, for the two rows whose `section_id` is a name
rather than a clause number, each being the clause where that name occurs exactly once. **[measured]**

**This makes the anchor count worse and the plan says so before the run.** 36 links land on 24 title anchors
today and on **14** clause anchors after, at 2.57 links each. **[measured, both]** The trade is 24 short
phrases against 14 paragraphs. It is recorded in the AFTER report as a **named regression of 10** on the
anchor column and a gain on the text column, and it is the kind of trade plan v2's link-only instrument
could not have shown at all.

**Truncation is heavy and the number is 29, not about 9.** 18 of the 25 clause bodies exceed
`MAX_ANCHOR_CHARS` after roll-up, and 29 of the 36 links land on one of those. **[measured]** Plan v2 said
19 clause bodies and about 9 links.

**Ceiling: 36/36 = 1.0000** **[measured]**, floor **1.00**.

**Files:**
- Create: `parsers/parse_etsi.py`
- Create: `tests/test_parse_etsi.py`

**Interfaces:**
- Consumes: `BaseParser`, `Control`, `tests.synthetic_pdf.build_pdf` (Task 9), `tract.config.CONTROL_TEXT_ORIGIN_METADATA_KEY` (Task 8), `tract.config.HONEST_PROSE_MIN_CHARS`.
- Produces: `EtsiParser` with `framework_id = "etsi"`, `framework_name = "ETSI"`; `EtsiParser.clauses_from_text(text: str) -> dict[str, tuple[str, str]]`; `EtsiParser.build_controls(clauses: dict[str, tuple[str, str]], alternates_by_name: dict[str, str]) -> list[Control]`.

**Invalidates:**
- `data/processed/licensed/all_controls.json` and its sha256, so every `corpus_sha256` recorded before this task. The tracked `all_controls.json` does not change, because ETSI never enters it.
- `data/processed/stopwords.json`, which Task 15 regenerates.
- Task 14's `hub_links_training.jsonl` for the 36 `etsi` rows, which move from 24 short title anchors onto 14 clause paragraphs.
- The 24-anchor BEFORE figure for this framework. The AFTER figure is 14, a named regression of 10.

- [ ] **Step 1: Confirm the licensing routing before writing anything**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
git check-ignore -v data/processed/frameworks/etsi.json
"$PY" -c "
from tract.config import RESTRICTED_FRAMEWORK_IDS
print('etsi restricted:', 'etsi' in RESTRICTED_FRAMEWORK_IDS)
"
# The page-footer defect, the 25-clause count, the 18 over-budget bodies and
# the roll-up lengths were all measured on pdfplumber 0.11.4, while
# requirements.txt pins 0.11.10. The footer shape is the extractor's rendering
# of the page, so a different version can render it differently. [measured]
"$PY" -c "
import pdfplumber, sys
print('pdfplumber', pdfplumber.__version__)
if pdfplumber.__version__ != '0.11.4':
    sys.exit(
        'Every ETSI figure in this task, including the page-footer defect '
        'itself, was measured on 0.11.4. Re-measure clauses_from_text against '
        'this version before writing the parser.'
    )
"
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py -q
```

Expected: the file is ignored, `etsi restricted: True`, `pdfplumber 0.11.4`, and the gate passes.

- [ ] **Step 2: Confirm which rows actually need an alternate title**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" - <<'PYEOF'
import collections
import json

pairs: collections.Counter[tuple[str, str]] = collections.Counter()
with open('data/training/hub_links_curated.jsonl', encoding='utf-8') as handle:
    for line in handle:
        row = json.loads(line)
        if row['framework_id'] == 'etsi':
            pairs[(row['section_id'], row['section_name'])] += 1

by_name = collections.defaultdict(set)
for section_id, name in pairs:
    by_name[name].add(section_id)

print(sum(pairs.values()), 'rows,', len(pairs), 'pairs,',
      len({a for a, _ in pairs}), 'ids,', len({b for _, b in pairs}), 'names')
print('rows whose section_id is a name:',
      sorted(a for a, _ in pairs if not a[0].isdigit()))
print('names spanning more than one clause:')
for name, ids in sorted(by_name.items()):
    if len(ids) > 1:
        print(f'   {name!r} -> {sorted(ids)}')
PYEOF
```

Expected **[measured]**: `36 rows, 27 pairs, 16 ids, 24 names`; the two name-shaped ids `Data sanitisation`
and `Retraining`; and three names spanning two clauses. Those last two are exactly why the parser below
declares two alternates and not twenty-four.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_parse_etsi.py: create

"""ETSI is restricted: every fixture here is synthetic, none of it is source.

The technique names OpenCRE links are not structural anchors in the PDF -- 2 of
24 are clause headings and 9 appear mid-sentence only -- so the mapping unit is
the numbered clause and the names are registered as alternate titles on the
clause the link's own section_id names.

TestSyntheticPdf drives parse() through pdfplumber against a PDF this file
builds, and that fixture carries a page footer shaped like the real one. The
released version of this parser read those footers as clauses 5, 6 and 7 and
gave all three the document identifier as a heading, with 22,639 characters of
front matter as one control's statement, while every gate stayed green. The
document identifier in the fixture is invented, so nothing here quotes ETSI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.parse_etsi import (
    DOCUMENT_IDENTIFIER,
    NAME_SECTION_IDS,
    SOURCE_FILE,
    EtsiParser,
)
from tests.synthetic_pdf import build_pdf
from tract.config import (
    CONTROL_TEXT_ORIGIN_METADATA_KEY,
    CONTROL_TEXT_ORIGIN_SYNTHETIC,
)

# Synthetic. Shaped like the source, worded nothing like it.
TEXT = """4 General
Front matter that is not a mapping unit.

5 First area
An opening paragraph for the first area.

5.1 First topic
The first topic runs for a couple of sentences and says what it covers.

5.2 Second topic
5.2.1 First sub-topic
Sub-topic text that the parent clause has none of its own.

5.2.2 Second sub-topic
More sub-topic text, long enough to be a statement in its own right.

6 Second area
6.1 Another topic
Text for another topic that is long enough to count as a statement.
"""

# An invented identifier in the real one's shape. Nothing here is ETSI's text.
FOOTER = "ETSI GR SAI 999 V9.9.9 (2099-01)"

PAGES: list[list[str]] = [
    ["4 General",
     "Front matter that is not a mapping unit at all in this document."],
    ["5 First area", "5.1 First topic",
     "The first topic runs for a couple of sentences and says what it covers."],
    ["5.2 Second topic", "5.2.1 First sub-topic",
     "Sub-topic text carrying the statement the parent clause has none of.",
     "5.2.2 Second sub-topic",
     "More sub-topic text, long enough to be a statement in its own right."],
    ["5.3 Third topic", "5.3.1 Overview",
     "An overview paragraph long enough to count as a statement of its own.",
     "5.3.2 Second sub-topic",
     "Another sub-topic body long enough to clear the sixty character floor."],
    ["6 Second area", "6.1 Another topic",
     "Text for another topic that is long enough to count as a statement."],
    ["7 Conclusion",
     "A closing paragraph that is long enough to count as a statement too."],
]


def _footed_pdf() -> bytes:
    """Six pages, each closing with a footer in the real one's shape.

    The footer is what the released CLAUSE pattern read as a clause heading:
    pages 5, 6 and 7 are the only pages whose number falls in [5-7], so the
    three top-level clauses took the document identifier as their heading and
    the front matter as their body.
    """
    pages: list[list[tuple[float, float, str]]] = []
    for index, body in enumerate(PAGES):
        runs = [(72.0, 60.0 + offset * 16, line)
                for offset, line in enumerate(body)]
        runs.append((72.0, 740.0, f"{index + 4} {FOOTER}"))
        pages.append(runs)
    return build_pdf(pages)


class TestClausesFromText:
    def test_numbered_clauses_are_found(self) -> None:
        clauses = EtsiParser.clauses_from_text(TEXT)
        assert set(clauses) >= {"5", "5.1", "5.2", "5.2.1", "5.2.2", "6", "6.1"}

    def test_section_four_is_not_a_mapping_unit(self) -> None:
        assert "4" not in EtsiParser.clauses_from_text(TEXT)

    def test_a_parent_with_no_text_rolls_up_its_children(self) -> None:
        clauses = EtsiParser.clauses_from_text(TEXT)
        parent = clauses["5.2"][1]
        assert "Sub-topic text" in parent
        assert "More sub-topic text" in parent

    def test_a_clause_with_its_own_text_does_not_roll_up(self) -> None:
        clauses = EtsiParser.clauses_from_text(TEXT)
        assert "Sub-topic text" not in clauses["5.1"][1]

    def test_a_page_footer_is_not_a_clause_heading(self) -> None:
        """The defect, stated at the smallest scope that shows it."""
        text = f"5 {FOOTER}\nFront matter.\n\n5 First area\nReal body here.\n"
        clauses = EtsiParser.clauses_from_text(text)
        assert clauses["5"][0] == "First area"

    def test_a_repeated_clause_number_is_refused(self) -> None:
        """Silence here is how the footers won the slot for three clauses.

        The released version dropped a second match with `continue`, so
        whichever match came first in page order kept the number and the real
        heading was discarded without a word.
        """
        text = "5 First area\nA body.\n\n5 Another first area\nAnother body.\n"
        with pytest.raises(ValueError, match="matched twice"):
            EtsiParser.clauses_from_text(text)


class TestNameShapedIds:
    def test_a_declared_name_becomes_an_alternate_title_on_its_clause(
        self,
    ) -> None:
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(TEXT), {"Some technique": "5.1"},
        )
        first = next(c for c in controls if c.control_id == "5.1")
        assert first.metadata is not None
        assert "Some technique" in first.metadata["alt_titles"]

    def test_no_other_clause_gains_an_alternate(self) -> None:
        controls = EtsiParser.build_controls(
            EtsiParser.clauses_from_text(TEXT), {"Some technique": "5.1"},
        )
        holders = [
            c.control_id for c in controls
            if (c.metadata or {}).get("alt_titles")
        ]
        assert holders == ["5.1"]

    def test_a_declared_clause_that_is_absent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="clause"):
            EtsiParser.build_controls(
                EtsiParser.clauses_from_text(TEXT),
                {"Nowhere technique": "9.9"},
            )


class TestSyntheticPdf:
    """parse() through pdfplumber, against a PDF that carries page footers."""

    @pytest.fixture()
    def parser(self, tmp_path: Path) -> EtsiParser:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / SOURCE_FILE).write_bytes(_footed_pdf())
        instance = EtsiParser(raw_dir=raw, output_dir=tmp_path / "out")
        instance.expected_sha256 = None
        instance.expected_clauses = 11
        return instance

    def test_the_three_top_level_clauses_keep_their_own_headings(
        self, parser: EtsiParser,
    ) -> None:
        """Pages 5, 6 and 7 are the ones whose footer number is in [5-7].

        Without the furniture guard, clauses 5, 6 and 7 take the document
        identifier as their heading and the page's front matter as their body,
        and expected_count still reads 11 because 11 distinct numbers matched
        either way.
        """
        controls = {c.control_id: c for c in parser.parse()}
        assert controls["5"].title == "First area"
        assert controls["6"].title == "Second area"
        assert controls["7"].title == "Conclusion"

    def test_no_clause_heading_is_the_document_identifier(
        self, parser: EtsiParser,
    ) -> None:
        for control in parser.parse():
            assert DOCUMENT_IDENTIFIER.match(control.title) is None

    def test_a_rolled_up_clause_is_marked_synthetic(
        self, parser: EtsiParser,
    ) -> None:
        controls = {c.control_id: c for c in parser.parse()}
        assert (controls["5.2"].metadata or {}).get(
            CONTROL_TEXT_ORIGIN_METADATA_KEY
        ) == CONTROL_TEXT_ORIGIN_SYNTHETIC
        assert CONTROL_TEXT_ORIGIN_METADATA_KEY not in (
            controls["5.1"].metadata or {}
        )

    def test_a_short_clause_list_is_refused(self, parser: EtsiParser) -> None:
        """The band would accept 23 of 25. The clause count does not."""
        parser.expected_clauses = 12
        with pytest.raises(ValueError, match="clause"):
            parser.parse()

    def test_run_writes(self, parser: EtsiParser, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        parser.expected_count = 11
        output = parser.run()
        assert len(output.controls) == 11
        assert [s.path for s in output.source_files] == [SOURCE_FILE]


class TestRun:
    """Real-source assertions, all negative, so nothing licensed is quoted."""

    def test_run_writes_from_the_real_pdf(self, tmp_path: Path) -> None:
        parser = EtsiParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        assert len(output.controls) == 25
        assert [s.path for s in output.source_files] == [
            "etsi_gr_sai005_v010101p.pdf",
        ]

    def test_no_real_clause_takes_the_document_identifier_as_a_heading(
        self, tmp_path: Path,
    ) -> None:
        """The regression test for the released defect, stated as a negative.

        Asserting the correct headings would put ETSI's own text into a
        tracked file. Asserting that no heading is the document identifier
        catches the same defect and quotes nothing.
        """
        parser = EtsiParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        by_id = {c.control_id: c for c in output.controls}
        for control in output.controls:
            assert DOCUMENT_IDENTIFIER.match(control.title) is None
        # All three shared one heading under the defect, because all three
        # took the same page footer. Three distinct headings is the positive
        # half of the same check, and it names no ETSI text.
        assert len({by_id[n].title for n in ("5", "6", "7")}) == 3
        # Clause 7's statement was 22,639 characters of front matter,
        # bibliography and section-4 tables. Its real body is an order of
        # magnitude smaller, and description is capped at
        # DESCRIPTION_MAX_LENGTH, so the check is on full_text.
        assert len(by_id["7"].full_text or by_id["7"].description) < 5_000

    def test_exactly_two_alternates_are_registered(
        self, tmp_path: Path,
    ) -> None:
        """A name registered on one clause answers links naming the other.

        Three of the 24 technique names span two clauses, so registering all
        of them would put a wrong anchor in front of the encoder while the
        resolution rate still read 1.0000.
        """
        parser = EtsiParser(output_dir=tmp_path)
        try:
            output = parser.run()
        except FileNotFoundError:
            pytest.skip("data/raw is gitignored and absent in this checkout")
        named = [
            name
            for control in output.controls
            for name in (control.metadata or {}).get("alt_titles", [])
        ]
        assert sorted(named) == sorted(NAME_SECTION_IDS)
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_etsi.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 5: Write the parser**

```python
# parsers/parse_etsi.py: create

"""Parser for ETSI GR SAI 005, Securing AI Problem Statement.

RESTRICTED. ETSI's copyright notice reserves reproduction in any medium
without written permission, so this framework is in
tract.config.RESTRICTED_FRAMEWORK_IDS, data/processed/frameworks/etsi.json is
gitignored, and merge_all_controls routes its prose to the gitignored licensed
overlay. Nothing in this file, its tests, or its commit messages quotes the
source.

The mapping unit is the numbered clause, and that is a deliberate choice made
against a measured alternative. OpenCRE's 36 curated links carry 24 distinct
technique names over 16 section ids. All 24 names appear in the document, but
only 2 are clause headings and only 1 is a bullet lead phrase; 9 appear
mid-sentence only, and several occur more than twenty times as
cross-references. Segmenting a technique's text out of running prose would mean
guessing sentence boundaries around an ambiguous match, and a wrong guess
attributes a mitigation to the wrong attack class with a provenance record that
looks correct.

A clause heading is not just any line that opens with a number in [5-7].
extract_text() renders every page footer as "<page number> <document
identifier>", so the pages numbered 5, 6 and 7 look exactly like top-level
clauses, and the change-history table adds "<day> <Month> <year> <version>
<summary>" in the same shape. Without a guard the three top-level clauses took
the document identifier as their heading and the front matter, bibliography and
section-4 tables as their statement, 22,639 characters of it for clause 7.
Every gate passed: 25 distinct numbers still matched, the garbage was long
enough to clear the prose floor, and no curated link targets a bare 5, 6 or 7
so the corpus report could not see it. DOCUMENT_IDENTIFIER and CHANGE_HISTORY
reject those lines, and a clause number that matches twice now raises instead
of silently keeping whichever match came first in page order.

Almost nothing else is declared here, and that is deliberate. control_id IS the
clause number, so 34 of the 36 curated links resolve through the id channel
with no alias at all. Registering all 24 technique names as alternates would be
actively harmful: three of them span two clauses each, and because lookup tries
the title first, a name registered on one clause would answer the link that
named the other. So NAME_SECTION_IDS holds exactly the two rows whose
section_id is a name rather than a clause number, and everything else is
OpenCRE's own clause assertion honoured verbatim. Two further links carry a
section_name that is also a clause's own heading, and a fifth names clause 6.3
while carrying 6.3.1's id; that fifth is what wrong_anchor_risk counts, and it
is left standing because 6.3's rolled-up statement opens with the whole of
6.3.1 and continues.

The cost is stated rather than discovered: 36 links land on 24 short title
anchors today and on 14 clause anchors after, 2.57 links each. The corpus
report records that as a regression on the anchor column. The gain is that
those 14 anchors are paragraphs of the standard rather than three-word phrases.

Seven clauses are headings whose text is entirely in their subclauses, so a
clause with no body of its own takes the concatenation of its descendants.
Only leaf clauses contribute, because an empty parent contributes an empty
string. That concatenation is assembled text rather than a passage anyone
wrote, so a rolled-up clause carries anchor_source = synthetic and the report
counts the 4 links that land on one apart from the other 32.
"""
from __future__ import annotations

import hashlib
import logging
import re
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.config import (
    CONTROL_TEXT_ORIGIN_METADATA_KEY,
    CONTROL_TEXT_ORIGIN_SYNTHETIC,
    HONEST_PROSE_MIN_CHARS,
)
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_FILE: Final[str] = "etsi_gr_sai005_v010101p.pdf"
SOURCE_SHA256: Final[str] = (
    "46c2b6b880928ffe2e763fbd6e0d0660a0aa7de0ff0071f5e0694582d91d5622"
)

# Sections 1-4 are scope, references, definitions and an overview. The mapping
# units are the attack and mitigation clauses in 5 through 7.
CLAUSE: Final[re.Pattern[str]] = re.compile(r"^([5-7](?:\.\d+){0,3})\s+(\S.{2,80})$")

# Page furniture that CLAUSE would otherwise read as a heading. extract_text()
# renders the footer as "<page number> <document identifier>", so pages 5, 6
# and 7 present as clauses 5, 6 and 7. The change-history table repeats the
# shape with a date. Both are rejected on the heading, not on the number, so a
# real clause is never dropped for sharing a number with a page.
DOCUMENT_IDENTIFIER: Final[re.Pattern[str]] = re.compile(
    r"^ETSI\s+(?:GR|GS|TS|TR|EG|EN)\b", re.IGNORECASE
)
CHANGE_HISTORY: Final[re.Pattern[str]] = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4}\b"
)

# The only two curated rows whose section_id is a technique name rather than a
# clause number, mapped to the clause where that name occurs exactly once in
# the document. Hand-verified against the pinned PDF.
#
# This map is two entries and not twenty-four on purpose. See the module
# docstring: three technique names span two clauses each, and registering a
# name as an alternate title makes it answer every link that carries it,
# including the link that named the other clause.
NAME_SECTION_IDS: Final[dict[str, str]] = {
    "Data sanitisation": "5.2.2",
    "Retraining": "5.3.2",
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class EtsiParser(BaseParser):
    framework_id: ClassVar[str] = "etsi"
    framework_name: ClassVar[str] = "ETSI"
    version: ClassVar[str] = "1.1.1"
    source_url: ClassVar[str] = (
        "https://www.etsi.org/deliver/etsi_gr/SAI/001_099/005/"
        "01.01.01_60/gr_SAI005v010101p.pdf"
    )
    mapping_unit_level: ClassVar[str] = "clause"
    # Numbered clauses in sections 5 through 7. [measured]
    expected_count: ClassVar[int] = 25
    # COUNT_TOLERANCE is 10%, so the band around 25 is 23 to 27 and a parser
    # that lost two clauses would write in silence. This is the structural
    # check that beats the band. Overridable so a synthetic PDF can drive
    # parse() in CI. [measured]
    expected_clauses: ClassVar[int] = 25
    fetched_date: ClassVar[str] = "2026-08-15"
    # All 25 clauses clear HONEST_PROSE_MIN_CHARS after roll-up and none
    # equals its own heading, so the measured value is exactly 1.0. [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(SOURCE_FILE)
        self._check_digest(payload)
        with pdfplumber.open(BytesIO(payload)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        clauses = self.clauses_from_text(text)
        if len(clauses) != self.expected_clauses:
            raise ValueError(
                f"{self.framework_id}: {len(clauses)} clause(s) in sections 5 "
                f"through 7, expected {self.expected_clauses}. COUNT_TOLERANCE "
                f"puts the band around 25 at 23 to 27, so a loss of two would "
                f"write without a word."
            )
        controls = self.build_controls(clauses, NAME_SECTION_IDS)
        synthesised = sum(
            1 for c in controls
            if (c.metadata or {}).get(CONTROL_TEXT_ORIGIN_METADATA_KEY)
        )
        logger.info(
            "%s: %d clauses, %d rolled up from their children, %d name-shaped "
            "section id(s) registered as alternate titles: %s",
            self.framework_id, len(controls), synthesised,
            len(NAME_SECTION_IDS), sorted(NAME_SECTION_IDS),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.expected_sha256:
            raise ValueError(
                f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not "
                f"the pinned {self.expected_sha256}. Both entries in "
                f"NAME_SECTION_IDS were verified against this revision's "
                f"clause numbering, the 25-clause count was measured against "
                f"these bytes, and DOCUMENT_IDENTIFIER was written against "
                f"this revision's page footer."
            )

    @staticmethod
    def _is_page_furniture(heading: str) -> bool:
        """Whether a candidate heading is a footer or a change-history row."""
        return bool(
            DOCUMENT_IDENTIFIER.match(heading) or CHANGE_HISTORY.match(heading)
        )

    @classmethod
    def clauses_from_text(cls, text: str) -> dict[str, tuple[str, str]]:
        """clause number -> (heading, body), children rolled up where needed.

        Raises:
            ValueError: If a clause number matches on more than one line after
                the furniture guard.
        """
        lines = text.split("\n")
        starts: list[tuple[int, str, str]] = []
        seen: dict[str, str] = {}
        for index, line in enumerate(lines):
            match = CLAUSE.match(line.strip())
            if match is None:
                continue
            number, heading = match.group(1), match.group(2).strip()
            if cls._is_page_furniture(heading):
                logger.debug(
                    "etsi: line %d matches the clause pattern and is page "
                    "furniture, not a heading", index,
                )
                continue
            if number in seen:
                raise ValueError(
                    f"etsi: clause {number} matched twice, at two different "
                    f"headings. The released parser dropped the second with a "
                    f"silent continue, which is how the page footers on pages "
                    f"5, 6 and 7 took the three top-level clause numbers and "
                    f"the real headings were discarded. Add the new furniture "
                    f"shape to the guard rather than restoring the silence."
                )
            seen[number] = heading
            starts.append((index, number, heading))
        starts.sort()

        own: dict[str, tuple[str, str]] = {}
        for position, (start, number, heading) in enumerate(starts):
            end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
            body = _WHITESPACE.sub(
                " ", " ".join(line.strip() for line in lines[start + 1:end]),
            ).strip()
            own[number] = (heading, body)

        rolled: dict[str, tuple[str, str]] = {}
        for number, (heading, body) in own.items():
            if len(body) >= HONEST_PROSE_MIN_CHARS:
                rolled[number] = (heading, body)
                continue
            children = sorted(
                key for key in own if key.startswith(f"{number}.")
            )
            merged = " ".join(own[key][1] for key in children).strip()
            rolled[number] = (heading, merged or body)
        return rolled

    @classmethod
    def build_controls(
        cls,
        clauses: dict[str, tuple[str, str]],
        alternates_by_name: dict[str, str],
    ) -> list[Control]:
        """One Control per clause, name-shaped section ids as alternate titles.

        Raises:
            ValueError: If a declared name points at a clause this parse did
                not produce, or a clause has no text at all after roll-up.
        """
        alternates: dict[str, list[str]] = {}
        for name, clause in sorted(alternates_by_name.items()):
            alternates.setdefault(clause, []).append(name)

        missing = sorted(set(alternates) - set(clauses))
        if missing:
            raise ValueError(
                f"etsi: NAME_SECTION_IDS points at clause(s) {missing} that "
                f"this parse did not produce. Two curated links carry a "
                f"technique name where a clause number belongs and reach "
                f"their clause only through this map, so a stale entry leaves "
                f"them resolving to nothing while the parser still writes."
            )

        controls: list[Control] = []
        for number in sorted(clauses, key=lambda n: [int(p) for p in n.split(".")]):
            heading, body = clauses[number]
            if not body:
                raise ValueError(
                    f"etsi: clause {number} has no text of its own and no "
                    f"children to roll up. An empty statement would be "
                    f"excluded from the prose index without any gate saying so."
                )
            metadata: dict[str, str | list[str]] = {"clause": number}
            if any(key.startswith(f"{number}.") and clauses[key][1] in body
                   for key in clauses if key != number):
                # The statement is the concatenation of this clause's
                # descendants rather than a passage of the standard. Four of
                # the 36 curated links land on one of these.
                metadata[CONTROL_TEXT_ORIGIN_METADATA_KEY] = (
                    CONTROL_TEXT_ORIGIN_SYNTHETIC
                )
            if number in alternates:
                metadata["alt_titles"] = alternates[number]
            controls.append(Control(
                control_id=number,
                title=heading,
                description=body,
                hierarchy_level="clause",
                parent_id=number.rsplit(".", 1)[0] if "." in number else None,
                metadata=metadata,
            ))
        return controls


def main() -> None:
    EtsiParser().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and typecheck**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" -m pytest tests/test_parse_etsi.py -q
"$PY" -m mypy parsers/parse_etsi.py --strict
```

`mypy --strict` on this file is clean as written. **[measured]**
`TestSyntheticPdf::test_the_three_top_level_clauses_keep_their_own_headings` is the one that fails without
the furniture guard. Confirm it fails when the guard is removed before moving on: a regression test that
passes against the defect is decoration.

- [ ] **Step 7: Run against the real source and check the join**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" parsers/parse_etsi.py
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" scripts/corpus_report.py --framework etsi --json
```

Expected log line: `etsi: 25 clauses, 7 rolled up from their children, 2 name-shaped section id(s)
registered as alternate titles: ['Data sanitisation', 'Retraining']`. **[measured]**

Accept only on this exact set. Every value is **[measured]**, and the whole row was reproduced end to end by
applying the furniture guard to the pinned PDF and resolving all 36 links through `ProseIndex`.

| field | value | how it can fail in each direction |
|---|---|---|
| `links` | 36 | |
| `by_title` | **5** | two declared alternates, two rows whose `section_name` is a clause's own heading, and the `6.3.1` row. Below 5 means a clause heading stopped matching. Above 5 means more names were registered than the two whose `section_id` needs them, and the three names that span two clauses will start answering links they do not own |
| `by_id` | 31 | |
| `unresolved` | 0 | attainable maximum is 36 |
| `distinct_anchors` | 14 | `5.1`, `5.2`, `5.2.2`, `5.3`, `5.3.2`, `6.1`, `6.2.2`, `6.2.3`, `6.3`, `6.3.2`, `6.3.3`, `6.4.1`, `6.4.2`, `6.4.3` |
| `distinct_anchors_pre_truncation` | 14 | equal, so truncation collapses nothing. A 13 here would mean two clause bodies became one string at 2,150 characters |
| `fallback_anchors` (BEFORE) | 24 | orchestrator measured. This is a **named regression of 10**, declared in advance. Carry it into the AFTER report as a regression on the anchor column and a gain on the text column. Do not treat it as a surprise |
| `links_per_anchor` | 2.57 | |
| `truncated` | **29** | 18 of the 25 clause bodies exceed `MAX_ANCHOR_CHARS` after roll-up and 29 of the 36 links land on one. Plan v2 said about 9 |
| `nested_anchors` / `contained_anchors` | 0 / 0 | zero only because truncation cuts each rolled-up parent before it reaches its linked child's text. Record the value, and if the encoder budget rises, expect this column to move |
| `dropped_by_prose_rule` | 0 | 18 of 25 carry `full_text` and the rest clear `_is_prose` |
| `wrong_anchor_risk` | **1** | the `6.3.1` row, whose `section_name` names clause 6.3. A 0 means the two clause-heading rows stopped matching. A 2 or more means a name spanning two clauses got registered |
| `anchor_source_full_text` | 25 | |
| `anchor_source_description` | 7 | |
| `anchor_source_synthetic` | 4 | links landing on `5.2`, `5.3` or `6.3`, whose statement is the concatenation of their descendants |
| `anchor_source_title` | 0 | |
| `distinct_hubs` | 29 | |
| `links_per_hub` | 1.24 | |
| `resolution_rate` | 1.0000 | floor 1.00 |

- [ ] **Step 8: Confirm nothing licensed reached git, then commit**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
git status --porcelain data/processed/frameworks/etsi.json
git check-ignore -v data/processed/frameworks/etsi.json
PYTHONPATH=. "$PY" -m pytest tests/test_licensed_text_not_tracked.py \
                              tests/test_merge_licensed_overlay.py -q
```

Expected: `git status` prints nothing for that path, `git check-ignore` matches, both tests pass.

```bash
git add parsers/parse_etsi.py tests/test_parse_etsi.py
git commit -m "feat: parse ETSI at the clause, where OpenCRE's own section ids point"
```

Neither `data/processed/frameworks/etsi.json` nor `data/processed/all_controls.json` appears in that
`git add`. The first is the whole point of the restriction. The second does not change, because ETSI never
enters the tracked corpus.

---

---

### Task 14: Retire both link gates onto the resolved anchor

`assign_quality_tier` drops a link two ways and both test a section title.
`PHASE1B_DROPPED_FRAMEWORKS` names `nist_800_63` and `owasp_proactive_controls`
outright, and `_has_descriptive_text` drops any link whose `section_name` is
shorter than `PHASE1B_MIN_SECTION_TEXT_LENGTH = 10`. Reproduced exactly:
**278 of 4,405 curated links are dropped, 155 by the framework list and 123 by
the short title.** **[measured, orchestrator, `data/training/hub_links_curated.jsonl`]**

Per framework, the 123 short-title drops: capec 44, dsomm 38, cwe 17, enisa 9,
biml 7, iso_27001 2, nist_800_53 2, owasp_ai_exchange 2, etsi 1,
owasp_top10_2021 1. **[measured, orchestrator]**

**Sixty-four of those 123 already resolve to prose in today's corpus.**
**[measured, orchestrator, `ProseIndex.load()` over the 31-framework overlay]**
They are dropped for having a short title while the pipeline holds a paragraph
for them.

**The gate leaks in the other direction too, and no one counted it.** Of the
4,127 links that reach the trainer today, **525 resolve to nothing and train on
their section title**, spread over **251 distinct anchor strings**:
dsomm 176, wstg 118, enisa 59, nist_ssdf 46, etsi 35, samm 30, csa_ccm 29,
owasp_top10_2021 16, biml 14, iso_27001 2. **[measured, orchestrator]** That is
12.7% of the training file training on labels while production serves prose,
against CLAUDE.md's standing rule that title fallback is a last resort. The
count that matters for this task is not 4,127 rising, it is 525 falling.

**Retiring the framework list alone changes nothing for those two
frameworks.** Every one of `nist_800_63`'s 79 `section_name` values is a
section number of 3 to 7 characters and every one of
`owasp_proactive_controls`' 76 is `C1`..`C10`. **[measured, orchestrator]**
Remove the framework list and the short-title gate drops all 155 anyway. Both
must move together, which is why this is one task and why it lands after the
parsers that give those links a resolved anchor.

#### The gate requires a resolved anchor, and does not fall back to the title

The v2 gate was `text = (resolved_text or link.get("section_name", "")).strip()`.
That falls back to `section_name`, the exact field the task's own commit message
says it is moving away from, and twelve links clear the ten-character floor on a
string the model should never see:

| framework | links | anchor they would train on | length |
|---|---|---|---|
| wstg | 9 | `WSTG-BUSL-$$` (3), `WSTG-INPV-00` (3), `WSTG-APPE-D` (2), `WSTG-INFO-##` (1) | 11-12 chars |
| iso_27001 | 2 | `Security of assets off-premises`, `Equipment siting and protection` | 31 chars each |
| dsomm | 1 | the activity name for the one activity whose statement is 11 characters, which `_is_prose` refuses to index | long |

**[measured, orchestrator]** `section_name == section_id` for all 118 wstg rows,
and the four bogus ids appear in no WSTG archive file (Contract Rule 0: the
parser reaches 109 of 118). Those nine links would pair a literal control id
with a real CRE hub and call it training data.

The gate therefore requires `resolved_text is not None`. The cost is stated
rather than hidden: two real ISO 27001 links whose `section_name` is a genuine
descriptive title get dropped because the link's id and name match no parsed
control. Keeping them would mean shipping title anchors that nothing downstream
can distinguish from prose anchors, which is the defect this task exists to
close. Two links out of 4,405 is the price.

#### Derived outcome

| quantity | before | after | source |
|---|---|---|---|
| curated links | 4,405 | 4,405 | **[measured]** |
| training links | 4,127 | **4,389** | **[derived]**, see arithmetic |
| anchors that are section titles | 525 | **0** | **[derived]**, by construction of the gate |
| distinct title-fallback anchor strings | 251 | 0 | **[derived]** |

Arithmetic, every term measured or taken from a `JOIN_FLOORS` entry committed in
Task 16 before any parser existed:

```
4,127  training links today                                    [measured]
  +154  framework deny list retired: proactive 76 + nist_800_63 78 of 79
  +120  short-title drops that Tasks 3-13 give a resolved anchor
    -9  wstg links whose id is absent from the archive          [measured]
    -2  iso_27001 links that resolve to no parsed control       [measured]
    -1  dsomm link whose control statement is 11 characters     [JOIN_FLOORS]
=4,389                                                          [derived]
```

The sixteen links that stay dropped, named, so the acceptance test asserts
identity rather than a count:

| framework | links | reason | source |
|---|---|---|---|
| wstg | 9 | ids absent from the archive | `JOIN_FLOORS["wstg"] = 0.92` |
| nist_800_53 | 2 | `SC-23(1)`, `SC-23(3)` match no parsed control | **[measured]** |
| iso_27001 | 2 | `7.8`, `7.9` match no parsed control | **[measured]** |
| dsomm | 1 | statement is 11 characters, `_is_prose` skips it | `JOIN_FLOORS["dsomm"] = 0.99` |
| nist_800_63 | 1 | `section_id == section_name == "are g"`, a corrupt OpenCRE row | **[measured]** |
| cwe | 1 | `937` matches no parsed control | **[measured]** |

**The v2 plan's `4,402 of 4,405` was wrong twice.** Under its own
fallback-to-title gate the answer is **4,401**, because a fourth link falls under
the floor (`nist_800_63` with `section_name == "are g"`, 5 characters) and the
plan enumerated only three. Under the gate this task ships, the answer is
**4,389**, and the 12-link difference between 4,401 and 4,389 is precisely the
set that would have trained on a title. The wrong number was hard-coded into a
commit message and copied into the run ledger with no test behind it. Step 8 adds
the test.

#### The count depends on whether this checkout holds the licensed overlay

`ProseIndex.load()` calls `merged_corpus_path()`, which returns the gitignored
overlay when it exists and the tracked corpus otherwise. Measured against both
files today: the overlay resolves **3,666** of 4,405 curated links, the tracked
corpus resolves **3,574**. **[measured, orchestrator]** The whole 92-link gap is
`iso_27001`, whose text is licensed and is not in git.

That makes 4,389 reachable only where the overlay is present. Where it is not, the
expected count falls by every link belonging to a framework the corpus does not
carry. Under Contract Rule 3's `OVERLAY_FRAMEWORK_IDS` that is nine frameworks
holding 635 links, of which 623 would otherwise resolve, giving **3,766**.
**[derived]** The test computes the expectation from the corpus it read
rather than hard-coding either literal, so it asserts in both environments and
skips in neither.

Worse, the mechanism that was supposed to make the two runs distinguishable does
not work. `merged_corpus_path`'s docstring states "the fold metadata records the
corpus sha256". It does not: `tract/training/orchestrate.py:347` hashes
`PROCESSED_DIR / "all_controls.json"` while `ProseIndex.load()` at line 183 reads
`merged_corpus_path()`. **[measured, orchestrator]** Two runs 92 links apart
record the same digest. Step 6 fixes it.

#### The training file becomes a function of the corpus, and must say so

After this task, `filter_training_links` resolves every link through
`ProseIndex.load()`, so `hub_links_training.jsonl` depends on
`merged_corpus_path()`. Today `save_training_links(links, raw_hash)` records only
the curated-links hash, so two runs over different corpora produce the same
`raw_hash`. Task 15 then rewrites the corpus.

The v2 self-review claimed "Task 14 precedes Task 15" discharged ledger lesson 6.
It does the opposite: ordering a task before the thing that invalidates its output
is the lesson, not the remedy. Two fixes, neither of which depends on task
numbering:

1. `save_training_links` takes `corpus_sha256` as a required positional argument
   and writes `data/training/hub_links_training.meta.json` beside the JSONL.
2. `tests/test_data_quality.py` asserts the sidecar's `corpus_sha256` equals the
   digest of the corpus on disk. Any task that rewrites the corpus without
   regenerating the training file turns that test red. Task 15 Step 10
   regenerates it.

#### CAPEC and CWE: a lever, stated rather than assumed

This change restores every contested link in both frameworks. CAPEC training links
move 1,755 to 1,799 (all 44 recovered), CWE moves 596 to 612 (16 of 17 recovered,
`937` resolves to nothing). **[measured, orchestrator]** The recovered links are
the terse ones: `UDP Ping`, `Fuzzing`, `Pharming`, `HTTP DoS`, `XML Flood`.

The v2 self-review stated "CAPEC and CWE are untouched and remain 57.3% of the
training graph. Nothing here improves that." Both halves are wrong. They are
touched, and their combined share falls from **56.97%** (2,351 of 4,127) to
**54.94%** (2,411 of 4,389) because the eleven frameworks add more than CAPEC and
CWE do. **[derived from measured counts]**

The reason to make it a lever rather than a default: the human ceiling study
measured CAPEC's agreement with OpenCRE at **alpha-1 = 0.181 [0.113, 0.277] on
n=83**. **[measured, `results/ceiling_study/panel_agreement.md:8,77`]** A domain
expert and OpenCRE's curators pick the same best hub fewer than one time in five
on that framework. Recovering its shortest-labelled links is not self-evidently
progress. Ten new CAPEC items and six new CWE items also enter the validation
roster (1,244 to 1,264) **[measured, premortem Data Scientist and Governance]**,
drawn from the least-agreed stratum, after the ceiling was measured on a roster
without them.

`filter_training_links` therefore takes `recover_contested: bool = True`. The
default ships the recovery, the flag gives the later training-mix decision a lever
that is not entangled with the eleven frameworks' 274 legitimate recoveries, and
both values are asserted, so neither branch is dead code:

| `recover_contested` | training links | capec | cwe | capec+cwe share |
|---|---|---|---|---|
| `True` (default) | 4,389 | 1,799 | 612 | 54.94% |
| `False` | 4,329 | 1,755 | 596 | 54.26% |

**[derived from measured counts]**

#### The ceiling study stops mirroring training the moment this lands

`tract/ceiling_study.py:132` calls `assign_quality_tier(record)` with one argument,
and line 119 documents why: "Mirrors tract.training.data_quality.load_and_filter_curated_links
exactly (same assign_quality_tier call) so the study pool is the pool training
would use, not an approximation of it." Line 144 makes a second one-argument call.
**[measured, orchestrator]**

Give `resolved_text` a default and both calls keep compiling, the study pool keeps
the section-title gate, training moves to the anchor gate, nothing raises, and the
docstring becomes false. `resolved_text` therefore has **no default**, so both call
sites fail at import under `mypy --strict` and at runtime under pytest. Step 7
rewrites them to call the same function training calls.

**Files:**
- Modify: `tract/config.py`
- Modify: `tract/training/data_quality.py`
- Modify: `tract/text_selection.py`
- Modify: `tract/training/orchestrate.py`
- Modify: `tract/ceiling_study.py`
- Modify: `tests/test_data_quality.py`
- Modify: `tests/test_ceiling_study.py`
- Create: `data/training/hub_links_training.meta.json`

**Interfaces:**
- Consumes: `tract.text_selection.ProseIndex`, `merged_corpus_path`; the parsers from Tasks 3-13.
- Produces: `assign_quality_tier(link: dict[str, str], resolved_text: str | None) -> QualityTier` (no default on the second parameter); `FilterReport`; `filter_training_links(links, index, *, recover_contested: bool = True) -> FilterReport`; `curated_link_filter_report(path=None, index=None, *, recover_contested=True) -> tuple[FilterReport, str]`; `save_training_links(links, raw_hash, corpus_sha256, path=None) -> str`; `tract.text_selection.merged_corpus_sha256() -> str`; `PHASE1B_MIN_ANCHOR_TEXT_LENGTH` and `CONTESTED_RECOVERY_FRAMEWORK_IDS` replacing `PHASE1B_DROPPED_FRAMEWORKS` and `PHASE1B_MIN_SECTION_TEXT_LENGTH`.

**Invalidates:**
- `data/training/hub_links_training.jsonl` and every artifact derived from it: `results/phase1b/**/fold_result.json`, every `data_hash` and `curated_links_sha256` recorded before this commit, and the published `hit@1 = 0.531` whose training file this replaces.
- `tract/ceiling_study.py`'s anchor pool, and therefore the sampling frame of `results/ceiling_study/ceiling_items.json`. The 250 drawn items survive: zero of them fall in the eleven frameworks and CAPEC and CWE reproduce byte-identically **[measured, adjudication C-A and C-B]**. The frame they were drawn from gains 60 anchors, so any future sample is not comparable to the 250 without saying so.
- `tract/training/orchestrate.py`'s `inputs.all_controls_sha256` in every fold record written before Step 6, which named a file the run did not read.
- Itself, by Task 15. Task 15 Step 10 regenerates `hub_links_training.jsonl` and its sidecar, and the sidecar test in Step 8 fails until it does.

- [ ] **Step 1: Record the before state, with the column the v2 plan did not have**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
from tract.text_selection import ProseIndex, merged_corpus_path
from tract.training.data_quality import QualityTier, assign_quality_tier

index = ProseIndex.load()
links = [json.loads(l) for l in
         open("data/training/hub_links_curated.jsonl", encoding="utf-8") if l.strip()]
kept = [l for l in links if assign_quality_tier(l) is not QualityTier.DROPPED]


def resolves(link: dict[str, str]) -> bool:
    return index.lookup(link.get("standard_name", ""),
                        link.get("section_id"), link.get("section_name")) is not None


fallback = [l for l in kept if not resolves(l)]
print("corpus read:", merged_corpus_path())
print("curated links:", len(links))
print("training links before:", len(kept))
print("of which train on a section title:", len(fallback))
print("distinct title anchors:", len({
    (l["framework_id"], (l.get("section_name") or l.get("section_id") or "").strip().lower())
    for l in fallback
}))
PYEOF
```

Expected, with the overlay present:

```
curated links: 4405
training links before: 4127
of which train on a section title: 525
distinct title anchors: 251
```

**[measured, orchestrator]** Write all four numbers into the run ledger. Step 9
compares against them. If `training links before` is not 4,127 the curated file
changed since the premortem and nothing below is derived from the right base.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_data_quality.py: append

import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestGatesTestTheAnchorNotTheTitle:
    """Both drops used to test section_name, which the model never sees."""

    LONG = "A control statement long enough to be worth training on, twice. " * 3

    def _index(self) -> "ProseIndex":
        from tract.text_selection import ProseIndex

        return ProseIndex([{
            "framework_name": "OWASP Proactive Controls",
            "controls": [
                {"control_id": "C6", "title": "Use Secure Dependencies",
                 "description": self.LONG},
            ],
        }])

    def test_a_short_title_with_a_resolved_anchor_is_kept(self) -> None:
        from tract.training.data_quality import QualityTier, assign_quality_tier

        link = {
            "framework_id": "owasp_proactive_controls",
            "standard_name": "OWASP Proactive Controls",
            "section_id": "C6", "section_name": "C6",
            "link_type": "LinkedTo",
        }
        assert assign_quality_tier(link, self.LONG) is QualityTier.T1

    def test_an_unresolved_link_is_dropped_however_long_its_title(self) -> None:
        """The nine wstg links this closes carry 11 and 12 character ids.

        Falling back to section_name would train "WSTG-BUSL-$$" against a real
        CRE hub, because section_name == section_id for all 118 wstg rows and
        the four bogus ids clear the ten-character floor. [measured]
        """
        from tract.training.data_quality import QualityTier, assign_quality_tier

        for name in ("WSTG-BUSL-$$", "WSTG-INPV-00",
                     "Security of assets off-premises"):
            link = {
                "framework_id": "wstg", "standard_name": "OWASP WSTG",
                "section_id": name, "section_name": name,
                "link_type": "LinkedTo",
            }
            assert assign_quality_tier(link, None) is QualityTier.DROPPED, name

    def test_a_resolved_but_thin_anchor_is_dropped(self) -> None:
        from tract.training.data_quality import QualityTier, assign_quality_tier

        link = {
            "framework_id": "dsomm", "standard_name": "DSOMM",
            "section_id": "x", "section_name": "a long activity name here",
            "link_type": "AutomaticallyLinkedTo",
        }
        assert assign_quality_tier(link, "Do backups") is QualityTier.DROPPED

    def test_the_anchor_parameter_has_no_default(self) -> None:
        """A defaulted second parameter is how the ceiling study broke silently.

        tract/ceiling_study.py called assign_quality_tier(record) with one
        argument under a docstring promising it mirrored training. Give
        resolved_text a default and that call keeps compiling while the two
        pools diverge, and nothing raises.
        """
        from tract.training.data_quality import assign_quality_tier

        parameter = inspect.signature(assign_quality_tier).parameters["resolved_text"]
        assert parameter.default is inspect.Parameter.empty

    def test_the_framework_deny_list_is_gone(self) -> None:
        import tract.config as config

        assert not hasattr(config, "PHASE1B_DROPPED_FRAMEWORKS")
        assert not hasattr(config, "PHASE1B_MIN_SECTION_TEXT_LENGTH")

    def test_filter_reports_each_drop_reason_separately(self) -> None:
        from tract.training.data_quality import filter_training_links

        links = [
            {"framework_id": "owasp_proactive_controls",
             "standard_name": "OWASP Proactive Controls",
             "section_id": "C6", "section_name": "C6",
             "cre_id": "1", "link_type": "LinkedTo"},
            {"framework_id": "owasp_proactive_controls",
             "standard_name": "OWASP Proactive Controls",
             "section_id": "C9", "section_name": "C9",
             "cre_id": "2", "link_type": "LinkedTo"},
        ]
        report = filter_training_links(links, self._index())
        assert len(report.kept) == 1
        assert len(report.dropped_unresolved) == 1
        assert report.dropped_thin_anchor == []

    def test_contested_recovery_is_a_lever_with_both_values_live(self) -> None:
        """capec alpha-1 is 0.181, so restoring its terse links is a choice.

        [measured, results/ceiling_study/panel_agreement.md]
        """
        from tract.text_selection import ProseIndex
        from tract.training.data_quality import filter_training_links

        index = ProseIndex([{
            "framework_name": "CAPEC",
            "controls": [{"control_id": "125", "title": "Flooding",
                          "description": self.LONG}],
        }])
        link = {"framework_id": "capec", "standard_name": "CAPEC",
                "section_id": "125", "section_name": "Flooding",
                "cre_id": "1", "link_type": "LinkedTo"}
        assert len(filter_training_links([link], index).kept) == 1
        off = filter_training_links([link], index, recover_contested=False)
        assert off.kept == []
        assert len(off.dropped_contested) == 1
```

```python
# tests/test_data_quality.py: append, the staleness guard

class TestTrainingFileRecordsTheCorpusItRead:
    """hub_links_training.jsonl is a function of the corpus after this task.

    save_training_links previously recorded only the curated-links hash, so two
    runs over corpora 92 iso_27001 links apart produced the same raw_hash
    [measured]. Task 15 rewrites the corpus. This test is what makes that
    ordering enforceable rather than a claim in a self-review.
    """

    def test_the_sidecar_names_the_corpus_on_disk(self) -> None:
        import json

        from tract.text_selection import merged_corpus_path, merged_corpus_sha256
        from tract.training.data_quality import TRAINING_META_PATH

        meta = json.loads(TRAINING_META_PATH.read_text(encoding="utf-8"))
        assert meta["corpus_sha256"] == merged_corpus_sha256(), (
            "hub_links_training.jsonl was built against a different corpus than "
            f"{merged_corpus_path()}. Regenerate it before trusting any metric "
            "derived from it."
        )
        assert meta["n_links"] == sum(
            1 for line in
            (TRAINING_META_PATH.parent / "hub_links_training.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        )

    def test_save_requires_the_corpus_digest(self) -> None:
        from tract.training.data_quality import save_training_links

        parameter = inspect.signature(save_training_links).parameters["corpus_sha256"]
        assert parameter.default is inspect.Parameter.empty
```

The three existing classes in `tests/test_data_quality.py` call
`assign_quality_tier(link)` with one argument at lines 28, 40, 52, 64, 76, 88 and
107, and `filter_training_links(links)` at 134, 151 and 185. **[measured]** Update
all ten call sites in this step. Do not add a default to make them pass, which is
the defect the signature test exists to catch.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_data_quality.py -q
```

Expected: FAIL. `assign_quality_tier` takes one argument,
`PHASE1B_DROPPED_FRAMEWORKS` still exists, `filter_training_links` returns a list
rather than a `FilterReport`, and `TRAINING_META_PATH` does not exist.

- [ ] **Step 4: Change the config**

```python
# tract/config.py: replace the PHASE1B_DROPPED_FRAMEWORKS block

# A link is worth training on when the text the model sees is
# substantial. Both of the gates this replaces tested link["section_name"], a
# title the model never sees: a framework deny list naming nist_800_63 and
# owasp_proactive_controls, and a 10-character floor on the same field. Between
# them they dropped 278 of 4,405 curated links, 64 of which already had a
# resolved paragraph in the corpus, while letting 525 links through to train on
# a title. [measured]
#
# The threshold is unchanged at 10 characters. Only the field it is applied to
# moved, from the title to the anchor the encoder is handed.
PHASE1B_MIN_ANCHOR_TEXT_LENGTH: Final[int] = 10

# Frameworks whose recovered links are a decision rather than a repair. The
# anchor gate restores 44 capec and 16 cwe links that the title floor dropped,
# and those are the terse ones ("UDP Ping", "Fuzzing", "Pharming"). The human
# ceiling study measured capec agreement with OpenCRE at alpha-1 = 0.181
# [0.113, 0.277] on n=83 [measured, results/ceiling_study/panel_agreement.md],
# so recovering its least-agreed stratum is not self-evidently progress. The
# default recovers them; filter_training_links(recover_contested=False) is the
# lever the later training-mix decision needs, and it is not entangled with the
# eleven frameworks' 274 recoveries.
CONTESTED_RECOVERY_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "capec", "cwe",
})
```

Delete `PHASE1B_DROPPED_FRAMEWORKS` and `PHASE1B_MIN_SECTION_TEXT_LENGTH`
entirely. A constant left in place with no reader is the decorative-control
defect from ledger lesson 4.

- [ ] **Step 5: Add the corpus digest to `text_selection`**

```python
# tract/text_selection.py: append after merged_corpus_path

def merged_corpus_sha256(path: Path | None = None) -> str:
    """The digest of the corpus a run read.

    merged_corpus_path's docstring already claimed a run that used the overlay
    and a run that did not were distinguishable because "the fold metadata
    records the corpus sha256". They were not. orchestrate.py hashed
    PROCESSED_DIR / "all_controls.json" while ProseIndex.load() read
    merged_corpus_path(), so two runs 92 iso_27001 links apart recorded the
    same digest for two different corpora. [measured]
    """
    import hashlib

    source = path or merged_corpus_path()
    return hashlib.sha256(source.read_bytes()).hexdigest()
```

- [ ] **Step 6: Change the filter, and make the training file name its corpus**

```python
# tract/training/data_quality.py: replace the imports and the two functions

from tract.config import (
    CONTESTED_RECOVERY_FRAMEWORK_IDS,
    PHASE1B_MIN_ANCHOR_TEXT_LENGTH,
    TRAINING_DIR,
)
from tract.io import atomic_write_json
from tract.text_selection import ProseIndex, merged_corpus_path, merged_corpus_sha256

TRAINING_META_PATH: Final[Path] = TRAINING_DIR / "hub_links_training.meta.json"


def link_key(link: dict[str, str]) -> str:
    """Stable identity for one curated link, so a drop can be named.

    A count tells an operator that something moved. Only a name tells them
    whether the thing that moved is the thing they expected.
    """
    return "|".join((
        link.get("framework_id", ""),
        link.get("section_id", ""),
        link.get("section_name", ""),
        link.get("cre_id", ""),
    ))


@dataclass(frozen=True)
class FilterReport:
    """What the anchor gate kept, and why it dropped everything else.

    Three drop reasons, reported apart, because they call for three different
    responses: an unresolved link means a parser or a join is missing, a thin
    anchor means the source is that terse, and a contested drop is a
    deliberate exclusion this run chose.
    """

    kept: list[TieredLink]
    dropped_unresolved: list[str]
    dropped_thin_anchor: list[str]
    dropped_contested: list[str]
    corpus_path: str
    corpus_sha256: str

    @property
    def n_dropped(self) -> int:
        return (
            len(self.dropped_unresolved)
            + len(self.dropped_thin_anchor)
            + len(self.dropped_contested)
        )


def _is_contested_recovery(link: dict[str, str]) -> bool:
    """True for a link this change newly admits from capec or cwe.

    Exactly the links the retired title floor dropped: 44 capec and 17 cwe.
    [measured]
    """
    return (
        link.get("framework_id", "") in CONTESTED_RECOVERY_FRAMEWORK_IDS
        and len(link.get("section_name", "").strip()) < PHASE1B_MIN_ANCHOR_TEXT_LENGTH
    )


def assign_quality_tier(
    link: dict[str, str], resolved_text: str | None,
) -> QualityTier:
    """Assign a quality tier to a single hub link.

    `resolved_text` is the anchor the encoder will be handed for this link,
    from ProseIndex, or None when the link resolves to no parsed control. It
    has no default on purpose. tract/ceiling_study.py calls this function under
    a docstring promising it mirrors training, and a defaulted parameter would
    let that call keep compiling while the two pools silently diverged.

    A link with no resolved anchor is dropped rather than falling back to
    link["section_name"]. That fallback is the field this change exists to stop
    training on, and twelve links clear the ten-character floor on it: nine
    wstg ids absent from the archive, two iso_27001 titles, and one dsomm
    activity whose statement _is_prose refuses to index. [measured]
    """
    if resolved_text is None:
        return QualityTier.DROPPED

    if len(resolved_text.strip()) < PHASE1B_MIN_ANCHOR_TEXT_LENGTH:
        return QualityTier.DROPPED

    if link.get("standard_name", "") in AI_FRAMEWORK_NAMES:
        return QualityTier.T1_AI

    if link.get("link_type", "") == "AutomaticallyLinkedTo":
        return QualityTier.T3

    return QualityTier.T1


def filter_training_links(
    links: list[dict[str, str]],
    index: ProseIndex,
    *,
    recover_contested: bool = True,
) -> FilterReport:
    """Filter links by the resolved anchor and assign tier metadata."""
    kept: list[TieredLink] = []
    unresolved: list[str] = []
    thin: list[str] = []
    contested: list[str] = []
    tier_counts: dict[QualityTier, int] = {t: 0 for t in QualityTier}

    for link in links:
        if not recover_contested and _is_contested_recovery(link):
            contested.append(link_key(link))
            continue

        selection = index.lookup(
            link.get("standard_name", ""), link.get("section_id"),
            link.get("section_name"),
        )
        text = selection.text if selection else None
        tier = assign_quality_tier(link, text)
        tier_counts[tier] += 1
        if tier is not QualityTier.DROPPED:
            kept.append(TieredLink(link=link, tier=tier))
        elif text is None:
            unresolved.append(link_key(link))
        else:
            thin.append(link_key(link))

    for tier, count in tier_counts.items():
        logger.info("Quality tier %s: %d links", tier.value, count)
    logger.info(
        "Dropped %d unresolved, %d thin anchors, %d contested",
        len(unresolved), len(thin), len(contested),
    )

    return FilterReport(
        kept=kept,
        dropped_unresolved=sorted(unresolved),
        dropped_thin_anchor=sorted(thin),
        dropped_contested=sorted(contested),
        corpus_path=str(merged_corpus_path()),
        corpus_sha256=merged_corpus_sha256(),
    )


def curated_link_filter_report(
    path: Path | None = None,
    index: ProseIndex | None = None,
    *,
    recover_contested: bool = True,
) -> tuple[FilterReport, str]:
    """Load the curated links and run the anchor gate over them.

    The single implementation of the gate. tract/ceiling_study.py calls this
    rather than repeating the tier call beside its own loop, which is how the
    two pools stopped agreeing.

    Returns:
        (report, sha256 of the raw curated records).
    """
    p = path or CURATED_PATH
    raw_links: list[dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_links.append(json.loads(line))

    raw_hash = compute_data_hash(raw_links)
    logger.info("Loaded %d curated links (hash=%s)", len(raw_links), raw_hash[:16])

    report = filter_training_links(
        raw_links, index or ProseIndex.load(), recover_contested=recover_contested,
    )
    logger.info(
        "After the anchor gate: %d usable links (dropped %d) against %s",
        len(report.kept), report.n_dropped, report.corpus_path,
    )
    return report, raw_hash


def load_and_filter_curated_links(
    path: Path | None = None,
) -> tuple[list[TieredLink], str]:
    """Load curated links, filter by the resolved anchor, return with data hash.

    Kept at its original arity so the six existing callers in
    tract/training/orchestrate.py, scripts/phase1b/run_fold.py and
    scripts/phase1c/ do not change. Callers that need the drop reasons call
    curated_link_filter_report directly.
    """
    report, raw_hash = curated_link_filter_report(path)
    return report.kept, raw_hash
```

Delete `_has_descriptive_text`.

```python
# tract/training/data_quality.py: replace save_training_links' signature and tail

def save_training_links(
    links: list[TieredLink],
    raw_hash: str,
    corpus_sha256: str,
    path: Path | None = None,
) -> str:
    """Save filtered training links to JSONL, and record what produced them.

    corpus_sha256 has no default. After the anchor gate, this file is a
    function of the corpus as well as of the curated links, and recording only
    raw_hash made two runs over corpora 92 links apart indistinguishable.
    """
```

At the end of that function, after `os.replace(tmp, p)`:

```python
    atomic_write_json(
        {
            "corpus_path": str(merged_corpus_path()),
            "corpus_sha256": corpus_sha256,
            "curated_links_sha256": raw_hash,
            "n_links": len(output_records),
            "output_sha256": output_hash,
        },
        TRAINING_META_PATH,
    )
```

```python
# tract/training/orchestrate.py: line 347, hash the corpus the run read

        "all_controls_sha256": (
            merged_corpus_sha256() if prose_index is not None else None
        ),
```

Import `merged_corpus_sha256` from `tract.text_selection` in that module.

- [ ] **Step 7: Make the ceiling study call the same gate**

```python
# tract/ceiling_study.py: replace _load_eligible_links and _link_priority

def _load_eligible_links(allowed_framework_ids: frozenset[str]) -> list[dict[str, str]]:
    """Curated links, quality-filtered, restricted to the eligible frameworks.

    Calls tract.training.data_quality.curated_link_filter_report, the function
    training calls, rather than repeating the gate beside it. The previous
    version inlined an assign_quality_tier(record) call under a docstring
    claiming it mirrored training. When that function gained a resolved-anchor
    argument, the copy here would have kept compiling against the old contract
    and quietly stopped mirroring: the study pool would keep the section-title
    gate while training moved to the anchor gate, and nothing would raise.
    """
    report, _ = curated_link_filter_report()
    return [
        tiered.link for tiered in report.kept
        if tiered.link.get("framework_id") in allowed_framework_ids
    ]


def _link_priority(
    link: dict[str, str], prose_index: ProseIndex,
) -> tuple[int, str, str]:
    """Sort key preferring higher-quality tiers, then section id, then name.

    Takes the index because assign_quality_tier now needs the anchor. The one
    caller, build_anchor_pool, already holds it.
    """
    selection = prose_index.lookup(
        link.get("standard_name", ""), link.get("section_id"),
        link.get("section_name"),
    )
    tier = assign_quality_tier(
        link, selection.text if selection else None,
    ).value
    return (
        _TIER_PRIORITY.get(tier, 99),
        link.get("section_id", ""),
        link.get("section_name", ""),
    )
```

```python
# tract/ceiling_study.py: line 187, inside build_anchor_pool
        representative = min(members, key=lambda m: _link_priority(m, prose_index))
```

Change the import on line 43 to add `curated_link_filter_report`.

```python
# tests/test_ceiling_study.py: append

class TestTheStudyPoolIsTheTrainingPool:
    """The mirror the docstring promises, asserted rather than described."""

    def test_the_two_pools_hold_the_same_links(self) -> None:
        from tract.ceiling_study import _load_eligible_links, eligible_framework_ids
        from tract.training.data_quality import curated_link_filter_report, link_key

        eligible = eligible_framework_ids()
        report, _ = curated_link_filter_report()
        training = {
            link_key(t.link) for t in report.kept
            if t.link.get("framework_id") in eligible
        }
        study = {link_key(l) for l in _load_eligible_links(eligible)}
        assert study == training

    def test_no_anchor_in_the_pool_is_a_section_title(self) -> None:
        """The gate admits only resolved links, so the pool is prose throughout.

        build_anchor_pool calls select_control_text, which falls back to the
        title when the index misses. Before the anchor gate, 525 of the 4,127
        training links resolved to nothing [measured], and any of them landing
        in an eligible framework put a title into the pool that a reviewer
        would have scored as a control statement.
        """
        from tract.ceiling_study import (
            _load_eligible_links, build_anchor_pool, eligible_framework_ids,
        )
        from tract.text_selection import ProseIndex

        index = ProseIndex.load()
        pool = build_anchor_pool(_load_eligible_links(eligible_framework_ids()), index)
        titles = [
            record.anchor_key for records in pool.values() for record in records
            if record.text_source == "title"
        ]
        assert titles == []
```

- [ ] **Step 8: Add the acceptance test for the count itself**

The v2 plan compared 4,127 against 4,402 with a `print()` that an agent read.
No test asserted either number, and the wrong one reached a commit message and
the run ledger. This test computes its expectation from the corpus it read, so
it asserts with the overlay and without it, and skips in neither.

```python
# tests/test_data_quality.py: append

class TestTheAnchorGateReachesItsDerivedCount:
    """4,389 of 4,405, and the sixteen exceptions named rather than counted."""

    # Every link the gate is expected to drop after Tasks 3-13, keyed
    # (framework_id, section_id). Nine wstg and one dsomm come from the
    # JOIN_FLOORS entries committed in Task 16 before any parser existed. The
    # other six were measured against the corpus at 8cf44b3. [measured]
    EXPECTED_UNRESOLVED: frozenset[tuple[str, str]] = frozenset({
        ("wstg", "WSTG-BUSL-$$"), ("wstg", "WSTG-INPV-00"),
        ("wstg", "WSTG-APPE-D"), ("wstg", "WSTG-INFO-##"),
        ("nist_800_53", "SC-23(1)"), ("nist_800_53", "SC-23(3)"),
        ("iso_27001", "7.8"), ("iso_27001", "7.9"),
        ("nist_800_63", "are g"), ("cwe", "937"),
    })
    EXPECTED_KEPT_FULL_CORPUS = 4389   # [derived] 4,405 - 16 unresolved links
    CONTESTED_RECOVERED = 60           # capec 44 + cwe 16 [measured]

    def _report(self, **kwargs: object):  # type: ignore[no-untyped-def]
        from tract.training.data_quality import curated_link_filter_report

        report, _ = curated_link_filter_report(**kwargs)  # type: ignore[arg-type]
        return report

    def test_every_drop_is_one_this_plan_predicted(self) -> None:
        """Fails in both directions: an unexpected drop, or an unexpected keep."""
        report = self._report()
        surprises = sorted(
            key for key in report.dropped_unresolved
            if (key.split("|")[0], key.split("|")[1]) not in self.EXPECTED_UNRESOLVED
        )
        assert surprises == [], (
            "these links resolve to no parsed control and this plan did not "
            f"predict that: {surprises[:20]}"
        )
        assert report.dropped_thin_anchor == [], (
            "a control resolved to fewer than ten characters of text. No "
            "parser in Tasks 3-13 was expected to emit one, so this is a "
            f"parser defect, not a source limit: {report.dropped_thin_anchor}"
        )

    def test_the_count_matches_the_corpus_that_was_read(self) -> None:
        """4,389 needs all 31 frameworks. Derive, never hard-code one literal.

        merged_corpus_path returns the gitignored overlay when it exists and
        the tracked corpus otherwise, and the tracked corpus always exists, so
        an existence check never skips. Measured: the overlay resolves 3,666 of
        4,405 curated links and the tracked file resolves 3,574, a 92-link gap
        that is entirely iso_27001. [measured]
        """
        import json

        from tract.text_selection import merged_corpus_path

        report = self._report()
        data = json.loads(merged_corpus_path().read_text(encoding="utf-8"))
        present = {
            record.get("framework_id") for record in data["frameworks"]
        }
        absent_drops = [
            key for key in report.dropped_unresolved
            if key.split("|")[0] not in present
        ]
        expected = self.EXPECTED_KEPT_FULL_CORPUS - len(absent_drops)
        assert len(report.kept) == expected, (
            f"{len(report.kept)} kept against {expected} expected for a corpus "
            f"of {len(present)} frameworks reading {merged_corpus_path()}"
        )

    def test_no_kept_link_trains_on_a_section_title(self) -> None:
        """525 links did before this change, on 251 distinct strings. [measured]"""
        from tract.text_selection import ProseIndex

        index = ProseIndex.load()
        report = self._report()
        titles = [
            t.link for t in report.kept
            if index.lookup(t.link.get("standard_name", ""),
                            t.link.get("section_id"),
                            t.link.get("section_name")) is None
        ]
        assert titles == []

    def test_the_contested_lever_moves_exactly_the_contested_links(self) -> None:
        full = self._report()
        without = self._report(recover_contested=False)
        assert len(full.kept) - len(without.kept) == self.CONTESTED_RECOVERED
        assert {key.split("|")[0] for key in without.dropped_contested} == {
            "capec", "cwe",
        }
```

- [ ] **Step 9: Run everything and record the after state**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_data_quality.py tests/test_ceiling_study.py -q
"$PY" -m mypy tract/training/data_quality.py tract/config.py \
      tract/text_selection.py tract/ceiling_study.py \
      tract/training/orchestrate.py --strict
grep -rn "PHASE1B_DROPPED_FRAMEWORKS\|PHASE1B_MIN_SECTION_TEXT_LENGTH" \
     tract/ parsers/ scripts/ tests/ || echo "no readers left"
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.training.data_quality import curated_link_filter_report
report, _ = curated_link_filter_report()
print("corpus:", report.corpus_path)
print("training links after:", len(report.kept))
print("dropped unresolved:", len(report.dropped_unresolved),
      report.dropped_unresolved)
print("dropped thin anchor:", len(report.dropped_thin_anchor))
PYEOF
```

Expected with the overlay present: `training links after: 4389`, sixteen
unresolved drops matching the named set, zero thin anchors. **[derived]**

Reading 4,401 means the fallback to `section_name` survived somewhere and twelve
links are training on a title. Reading 4,127 means the index is not reaching the
filter. Reading 3,766 means this checkout has no overlay, which is a legitimate
state that the test in Step 8 accepts and this print does not explain on its own.

- [ ] **Step 10: Regenerate the training file and commit, split by decision**

Two commits. The second is revertable on its own, so the later training-mix
decision has a lever that does not disturb the eleven frameworks' recoveries.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.text_selection import merged_corpus_sha256
from tract.training.data_quality import curated_link_filter_report, save_training_links

report, raw_hash = curated_link_filter_report(recover_contested=False)
print("wrote", len(report.kept), "links, output hash",
      save_training_links(report.kept, raw_hash, merged_corpus_sha256())[:16])
PYEOF
git add tract/config.py tract/training/data_quality.py tract/text_selection.py \
        tract/training/orchestrate.py tract/ceiling_study.py \
        tests/test_data_quality.py tests/test_ceiling_study.py \
        data/training/hub_links_training.jsonl \
        data/training/hub_links_training.meta.json
git commit -m "fix: drop a link on the text the model sees, not on its section title

Training links move from 4,127 to 4,329 with the contested capec and cwe
recoveries held back for the next commit. The 154 dropped by the framework deny
list and 120 of the 123 dropped by the short-title floor now resolve to parsed
prose, and the 525 links that used to train on a section title fall to zero. The
sixteen that stay dropped resolve to no parsed control: nine wstg ids absent from
the archive, two nist_800_53, two iso_27001, one dsomm, one nist_800_63 and one
cwe. The training file now records the sha256 of the corpus it was built from."
```

```bash
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.text_selection import merged_corpus_sha256
from tract.training.data_quality import curated_link_filter_report, save_training_links

report, raw_hash = curated_link_filter_report()
print("wrote", len(report.kept), "links, output hash",
      save_training_links(report.kept, raw_hash, merged_corpus_sha256())[:16])
PYEOF
"$PY" -m pytest tests/test_data_quality.py -q
git add data/training/hub_links_training.jsonl \
        data/training/hub_links_training.meta.json
git commit -m "feat: restore the 60 contested capec and cwe links the title floor dropped

capec goes 1,755 to 1,799 and cwe 596 to 612, taking training links to 4,389 of
4,405. The recovered links are the terse ones. The human ceiling study measured
capec agreement with OpenCRE at alpha-1 = 0.181 [0.113, 0.277] on n=83, so this
commit is a choice rather than a repair, and it reverts on its own without
touching the eleven frameworks' recoveries. Ten capec and six cwe items enter
the validation roster, taking it from 1,244 to 1,264, after the ceiling was
measured on a roster without them."
```

Record in the run ledger: 4,127 to 4,389, the 525-to-0 title-anchor figure, the
sixteen named drops, the two commit SHAs, and which of the two corpora produced
the number.

---

### Task 15: Rebuild the corpus, and prove only the eleven changed

The previous plan re-ran all 31 parsers and committed `data/processed/`
wholesale, silently re-running CAPEC and CWE, and its only mutation was
`shutil.copy2` into `data/processed/frameworks/` with no snapshot and no way
back.

#### Coverage: 89.7% of the baseline is already proven to reproduce

Every non-eleven framework has been test-rebuilt. The 19 importable parsers
reproduce **1,897 of 1,897** baseline keys with 0 mismatch **[measured, plan v2
at `8cf44b3`]**, plus 10 controls for `owasp_llm_top10_2026`, which has a parser
and no corpus entry because it landed after the baseline was taken. `defusedxml==0.7.1`
is now installed and both XML parsers reproduce byte-identically: **capec 558 of
558, cwe 1,331 of 1,331, 0 mismatch** **[measured, adjudication C-A]**.

1,897 + 1,889 = **3,786**, which is exactly the number of baseline keys outside
the eleven. **[measured, orchestrator]** Pre-measured rebuild coverage is
3,786 of 4,222 = **89.7%**, not the 45% the v2 plan assumed. The v2 Step 1 that
installs `defusedxml` and `openpyxl` is deleted: both are present, and
`openpyxl DEFUSEDXML` flipped `False` to `True` on the same install
**[measured, adjudication C-C]**.

The remaining 436 baseline keys are the eleven frameworks, and **every one of
them must change**. Their current corpus entries are OpenCRE-derived stubs where
`description == title` and `full_text` is empty:

```
nist_800_63:5-1-1-1   title '5.1.1.1'    description '5.1.1.1'    full_text None
wstg:wstg-appe-d      title 'WSTG-APPE-D' description 'WSTG-APPE-D' full_text None
csa_ccm:AIS-01        title 'Application and Interface Security Policy and Procedures'
                      description identical to the title, full_text None
```

**[measured, orchestrator]** That is why 0 of their 734 links resolve today:
`_is_prose` refuses to index a description that does not exceed its title by
`PROSE_MIN_EXTRA_CHARS`. The rebuild turns those 436 stubs into prose, so
`unchanged` must land on exactly 3,786 and no stub may survive.

#### The baseline is lossy and must be regenerated before it can gate anything

`data/processed/pre_rebuild_control_hashes.json` declares `n_controls: 4222`
against **4,261 control records on disk**. The gap is key collision, not missing
controls: **9 keys absorb 39 extra records, every one with a distinct
description**, and the stored hash is the **first** writer's every time.
**[measured, orchestrator]**

| key | records | distinct descriptions |
|---|---|---|
| `enisa:enisa:Table 5:` | 22 | 22 |
| `enisa:enisa:Table 3:` | 8 | 8 |
| `etsi:etsi:6.2.2` | 4 | 4 |
| `etsi:etsi:6.1`, `etsi:etsi:6.2.3` | 3 each | 3 each |
| `etsi:etsi:5.2.2`, `6.4.1`, `6.4.2`, `6.4.3` | 2 each | 2 each |

**[measured, orchestrator]** So `unchanged` was never a per-control count, and
38 of the 39 shadowed records were invisible to any comparison. All nine
collisions sit inside the two frameworks this task rebuilds, which is the one
place a blind spot is least affordable.

The baseline also hashes `description` alone:

```python
digest = hashlib.sha256(str(control.get("description") or "").encode("utf-8")).hexdigest()
```

This plan's own Contract Fact 1 states that `ProseIndex` prefers `full_text`
over `description` unconditionally, so whatever a parser puts in `full_text` **is
the anchor the model sees**. `full_text` is set by `parse_wstg`,
`parse_owasp_top10_2021`, `parse_owasp_proactive_controls`, `parse_nist_ssdf`,
and by `_sanitize_control` for any description over 2,000 characters. `title` is
a join channel, and `metadata["alt_ids"]` and `metadata["alt_titles"]` decide
which control a link resolves to. A rebuild can re-point every BIML and SSDF link
and report `0 changed`.

Step 2 regenerates the baseline with a collision-safe value type and a five-field
content digest, and proves the regeneration reproduces the old description-only
hashes on the 4,213 non-colliding keys, so the change of instrument does not
smuggle in a change of answer.

#### Sixty-three published assignments point at ids this rebuild retires

Five parsers change the control_id shape. Baseline keys affected, measured
exactly:

| framework | change | keys |
|---|---|---|
| wstg | `wstg-appe-d` to `WSTG-APPE-D` | 59 |
| nist_800_63 | `5-1-1-1` to `5.1.1.1` | 25 |
| owasp_proactive_controls | `c1` to `C1` | 10 |
| enisa | `Table 3:` to a per-row slug | 10 |
| csa_ccm | `IVS-*` to `I&S-*` | 7 |
| **total** | | **111** |

**[measured, orchestrator]** A rename is not a loss, so Step 4 emits a `renamed`
bucket keyed on matching content digest. For these 111 the bucket will be empty
and that is the honest answer, not a failure: the old record is a stub and the
new record is prose, so nothing can content-match. `removed` therefore means
"the stub this parser replaces" for the eleven and "content gone" everywhere
else, and Step 6's assertion is on framework membership rather than on a count
the operator has to judge.

Downstream, `build/dataset/crosswalk_v1.0.jsonl` carries **63 published rows**
whose control identity this rebuild dissolves: **56** with `control_id`
`enisa:enisa:Table 5:` (38) or `Table 3:` (18), and **7** with retired
`csa_ccm:csa_ccm:IVS-0{1,2,4,5,6,8,9}`. **[measured, orchestrator]** All 63 carry
`review_status = "ground_truth"`. `tract/export/canonical.py:76` filters on
`WHERE a.review_status = 'accepted'`, so `diff_snapshots` never sees them and
`compute_content_hash` emits no `UPDATE_CONTROL` or `DELETE_CONTROL` for any of
them. **[measured, orchestrator]** Step 8 writes the artifact that says so.

**Depends on:** Contract Rule 3's licence tiering (`OVERLAY_FRAMEWORK_IDS` in
`tract/config.py`). Step 9 imports it to decide which per-framework files may be
staged. If it is absent the step fails with `ImportError`, which is the correct
answer: committing parser output before the licence tiers exist is how licensed
prose escaped three times.

**Files:**
- Create: `scripts/rebuild_corpus.py`
- Create: `tests/test_rebuild_corpus.py`
- Create: `results/corpus/rebuild_diff.json`
- Create: `results/corpus/retired_control_ids.json`
- Modify: `data/processed/pre_rebuild_control_hashes.json`
- Modify: `data/processed/stopwords.json`
- Modify: `data/processed/frameworks/*.json`, `data/processed/all_controls.json`
- Modify: `data/training/hub_links_training.jsonl`, `data/training/hub_links_training.meta.json`

**Interfaces:**
- Consumes: every `parsers/parse_*.py`; `data/processed/pre_rebuild_control_hashes.json`; `tract.io.atomic_write_json`, `atomic_write_text`; `tract.config.OVERLAY_FRAMEWORK_IDS`.
- Produces: `content_digest(control) -> str`; `build_baseline(corpus) -> dict[str, Any]`; `snapshot_processed(root) -> Path`; `restore_snapshot(path) -> int`; `run_all(output_dir, audit_dir) -> tuple[dict[str, list[dict]], dict[str, str]]`; `diff_against_baseline(parsed, baseline) -> RebuildReport` with `RebuildReport(changed, added, removed, renamed, unchanged, failed)`; `assert_expected_frameworks_only(report) -> None`.

**Invalidates:**
- `data/processed/stopwords.json`. It is derived from the corpus, committed, applied to every control and hub text by `tract/text_selection.py`, `tract/training/data.py` and `tract/training/firewall.py`, and hashed into every fold record at `tract/training/orchestrate.py:351`. Eight modules and five test files read it. **[measured]** Step 7 regenerates and commits it. Without that, every post-rebuild metric uses a stopword list built for a corpus that no longer exists.
- `data/processed/all_controls.json` and `data/processed/licensed/all_controls.json`, and therefore every `all_controls_sha256` recorded in `results/phase1b/**/fold_result.json`.
- `data/processed/pre_rebuild_control_hashes.json`, which this task replaces with a collision-safe, five-field version.
- `data/training/hub_links_training.jsonl` and its sidecar, which Task 14 produced against the previous corpus. Step 10 regenerates them, and the sidecar test from Task 14 Step 8 stays red until it does.
- `build/dataset/crosswalk_v1.0.jsonl`'s 63 rows, and the published HuggingFace dataset built from it. No republication happens here. Step 8 records the debt.
- `results/ceiling_study/ceiling_items.json`'s sampling frame. The 250 drawn items survive intact, because none falls in the eleven and capec and cwe reproduce byte-identically. **[measured, adjudication C-B]**

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rebuild_corpus.py: create

"""A corpus rebuild must be reversible and must diff the field the model reads.

Three things the previous version could not do. It hashed `description` while
ProseIndex prefers `full_text` unconditionally, so it could re-point every link
and report 0 changed. It stored one digest per key while nine keys hold 39 extra
records with distinct text. Its only mutation was shutil.copy2 over three
files that git cannot restore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rebuild_corpus import (
    RebuildReport,
    assert_expected_frameworks_only,
    build_baseline,
    content_digest,
    diff_against_baseline,
    restore_snapshot,
    snapshot_processed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _baseline(*pairs: tuple[str, dict[str, object]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, control in pairs:
        out.setdefault(key, []).append(content_digest(control))
    return out


class TestTheDiffSeesEveryAnchorField:
    def test_identical_content_reports_no_change(self) -> None:
        control = {"control_id": "C-1", "description": "statement one"}
        report = diff_against_baseline(
            {"demo": [control]}, _baseline(("demo:C-1", control)),
        )
        assert report.changed == []
        assert report.unchanged == 1

    def test_a_moved_full_text_is_a_change(self) -> None:
        """The defect this replaces. description is equal, full_text is not.

        ProseIndex.__init__ takes full_text when it is non-empty and never
        looks at description, so this control's anchor moved entirely while a
        description-only hash reports nothing.
        """
        old = {"control_id": "C-1", "description": "same", "full_text": "before"}
        new = {"control_id": "C-1", "description": "same", "full_text": "after"}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]
        assert report.unchanged == 0

    def test_a_moved_alt_id_is_a_change(self) -> None:
        """alt_ids decides which control a link resolves to."""
        old = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_ids": ["PO.1.1"]}}
        new = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_ids": ["PO.1.2"]}}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]

    def test_alt_lists_are_order_insensitive(self) -> None:
        old = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["b", "a"]}}
        new = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["a", "b"]}}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.unchanged == 1


class TestCollidingKeysAreCountedPerRecord:
    """Nine keys hold 39 extra records, all with distinct text. [measured]"""

    def test_the_baseline_counts_records_not_keys(self) -> None:
        """The committed baseline declared 4,222 against 4,261 records.

        Nine keys absorbed 48 records and stored 9 digests, every one the
        FIRST writer's, so 39 records with distinct text were compared against
        nothing. All nine sit in enisa and etsi, the two frameworks this
        rebuild replaces. [measured]
        """
        corpus = {"frameworks": [{"framework_id": "enisa", "controls": [
            {"control_id": "Table 3:", "description": "poisoning"},
            {"control_id": "Table 3:", "description": "data disclosure"},
            {"control_id": "4.1", "description": "something else"},
        ]}]}
        baseline = build_baseline(corpus)
        assert baseline["n_keys"] == 2
        assert baseline["n_records"] == 3
        assert len(baseline["digests"]["enisa:Table 3:"]) == 2

    def test_two_records_under_one_key_are_two_units(self) -> None:
        first = {"control_id": "Table 3:", "description": "poisoning"}
        second = {"control_id": "Table 3:", "description": "data disclosure"}
        report = diff_against_baseline(
            {"enisa": [first, second]},
            _baseline(("enisa:Table 3:", first), ("enisa:Table 3:", second)),
        )
        assert report.unchanged == 2
        assert report.changed == []

    def test_losing_one_of_two_shadowed_records_is_visible(self) -> None:
        first = {"control_id": "Table 3:", "description": "poisoning"}
        second = {"control_id": "Table 3:", "description": "data disclosure"}
        report = diff_against_baseline(
            {"enisa": [first]},
            _baseline(("enisa:Table 3:", first), ("enisa:Table 3:", second)),
        )
        assert report.unchanged == 1
        assert report.removed == ["enisa:Table 3:"]


class TestRenamesAreNotLosses:
    def test_the_same_content_under_a_new_id_is_a_rename(self) -> None:
        old = {"control_id": "c1", "description": "validate every input"}
        new = {"control_id": "C1", "description": "validate every input"}
        report = diff_against_baseline(
            {"owasp_proactive_controls": [new]},
            _baseline(("owasp_proactive_controls:c1", old)),
        )
        assert report.renamed == [
            ("owasp_proactive_controls:c1", "owasp_proactive_controls:C1"),
        ]
        assert report.removed == []
        assert report.added == []

    def test_a_rename_does_not_cross_frameworks(self) -> None:
        old = {"control_id": "c1", "description": "validate every input"}
        new = {"control_id": "C1", "description": "validate every input"}
        report = diff_against_baseline(
            {"wstg": [new]}, _baseline(("owasp_proactive_controls:c1", old)),
        )
        assert report.renamed == []
        assert report.removed == ["owasp_proactive_controls:c1"]
        assert report.added == ["wstg:C1"]


class TestTheStopRuleIsAnAssertion:
    """Step 6 of the previous version was prose an autonomous worker reads past."""

    def test_an_unexpected_framework_halts_the_run(self) -> None:
        report = RebuildReport(changed=["capec:125"])
        with pytest.raises(SystemExit, match="capec"):
            assert_expected_frameworks_only(report)

    def test_a_framework_that_did_not_move_halts_the_run(self) -> None:
        """A parser that silently no-ops leaves the previous artifact in place."""
        report = RebuildReport(changed=[f"{f}:x" for f in (
            "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
            "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021", "samm",
        )])
        with pytest.raises(SystemExit, match="wstg"):
            assert_expected_frameworks_only(report)

    def test_the_expected_shape_passes(self) -> None:
        report = RebuildReport(
            changed=[f"{f}:x" for f in (
                "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
                "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021",
                "samm", "wstg",
            )],
            added=["owasp_llm_top10_2026:LLM01"],
            unchanged=3786,
        )
        assert_expected_frameworks_only(report) is None


class TestTheSnapshotIsARollback:
    """etsi.json, iso_27001.json and licensed/all_controls.json are untracked.

    .gitignore lines 37, 38 and 39. scripts/fetch_frameworks.py has no
    iso_27001 entry at all [measured], so ISO's raw source is hand-staged and
    its output is re-derivable from no scripted path. ISO is the corpus's only
    high-prose fold.
    """

    def test_a_snapshot_restores_byte_for_byte(self, tmp_path: Path) -> None:
        source = tmp_path / "processed"
        source.mkdir()
        original = '{\n  "a": 1\n}\n'
        (source / "etsi.json").write_text(original, encoding="utf-8")

        snapshot = snapshot_processed(tmp_path / "snapshots",
                                      members=[source / "etsi.json"])
        (source / "etsi.json").write_text('{"a": 2}', encoding="utf-8")
        assert restore_snapshot(snapshot) == 1
        assert (source / "etsi.json").read_text(encoding="utf-8") == original

    def test_a_tampered_snapshot_refuses_to_restore(self, tmp_path: Path) -> None:
        source = tmp_path / "processed"
        source.mkdir()
        (source / "etsi.json").write_text("{}\n", encoding="utf-8")
        snapshot = snapshot_processed(tmp_path / "snapshots",
                                      members=[source / "etsi.json"])
        member = next(p for p in snapshot.rglob("etsi.json"))
        member.write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="does not match its manifest"):
            restore_snapshot(snapshot)


class TestTheRegeneratedBaselineAgreesWithTheCommittedOne:
    """Changing the instrument must not change the answer it already gave."""

    def test_description_only_hashes_reproduce_on_non_colliding_keys(self) -> None:
        import hashlib

        from tract.text_selection import merged_corpus_path

        committed = json.loads(
            (REPO_ROOT / "data/processed/pre_rebuild_control_hashes.json")
            .read_text(encoding="utf-8")
        )
        if "sha256_of_description" not in committed:
            pytest.skip("baseline already regenerated by Step 2")
        old = committed["sha256_of_description"]
        corpus = json.loads(merged_corpus_path().read_text(encoding="utf-8"))
        first_seen: dict[str, str] = {}
        counts: dict[str, int] = {}
        for record in corpus["frameworks"]:
            for control in record.get("controls") or []:
                key = f"{record['framework_id']}:{control['control_id']}"
                counts[key] = counts.get(key, 0) + 1
                first_seen.setdefault(
                    key,
                    hashlib.sha256(
                        str(control.get("description") or "").encode("utf-8")
                    ).hexdigest(),
                )
        singles = [k for k, n in counts.items() if n == 1]
        assert len(singles) == 4213
        assert all(old[k] == first_seen[k] for k in singles)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_rebuild_corpus.py -q
```

Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.rebuild_corpus'`.

- [ ] **Step 3: Write the rebuild script**

```python
# scripts/rebuild_corpus.py: create

"""Re-run every parser into a scratch directory and diff the anchor fields.

The point is not to rebuild. It is to be able to say, per control record, what
changed, and to be able to put it back.

Three properties the previous version did not have.

Reversible. data/processed/frameworks/etsi.json, iso_27001.json and
licensed/all_controls.json are untracked (.gitignore 37-39), and
scripts/fetch_frameworks.py has no iso_27001 entry at all, so ISO's output is
re-derivable from no scripted path. --commit snapshots every overwritable file
first and --restore puts them back.

Blind to nothing. ProseIndex prefers full_text over description
unconditionally, and alt_ids and alt_titles decide which control a link
resolves to, so the digest covers all five fields. A description-only digest
could re-point every wstg, top10, proactive and nist_ssdf anchor and report
0 changed.

Enforcing. Nine baseline keys hold 39 extra records with distinct text, so the
value is a multiset of digests rather than one digest, and an unexpected
framework raises SystemExit rather than logging at INFO while --commit copies
anyway.

    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --dry-run
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --commit
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --list-snapshots
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --restore <snapshot-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

from tract.config import (
    PARSERS_DIR,
    PROCESSED_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    PROCESSED_LICENSED_DIR,
    PROJECT_ROOT,
)
from tract.io import atomic_write_json, atomic_write_text, load_json
from tract.parsers.base import BaseParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_PATH: Final[Path] = PROCESSED_DIR / "pre_rebuild_control_hashes.json"
SNAPSHOT_ROOT: Final[Path] = PROJECT_ROOT / "build" / "corpus_snapshots"

# The eleven frameworks Tasks 3-13 give a parser. Every one of their 436
# baseline keys is an OpenCRE-derived stub whose description equals its title,
# so every one MUST move. [measured]
EXPECTED_CHANGED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63", "nist_ssdf",
    "owasp_proactive_controls", "owasp_top10_2021", "samm", "wstg",
})
# Has a parser and no corpus entry: it landed after the baseline was taken, so
# its 10 controls are additions rather than changes. [measured]
EXPECTED_ADDED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "owasp_llm_top10_2026",
})
# Baseline keys outside the eleven. 1,897 from the 19 importable parsers plus
# capec 558 and cwe 1,331, each reproducing with 0 mismatch. [measured]
EXPECTED_UNCHANGED_RECORDS: Final[int] = 3786


def content_digest(control: Mapping[str, Any]) -> str:
    """Hash every field that decides which text a link resolves to.

    Hashing `description` alone, which is what the committed baseline does, is
    blind to the field the model reads. ProseIndex prefers `full_text`
    unconditionally and BaseParser._sanitize_control writes it behind the
    parser's back for any description over 2,000 characters. `title` and the
    two alternate lists decide WHICH control a link resolves to, so a change
    there re-points the link as surely.
    """
    metadata = control.get("metadata") or {}

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return sorted(str(v) for v in value)

    payload = {
        "description": str(control.get("description") or ""),
        "full_text": str(control.get("full_text") or ""),
        "title": str(control.get("title") or ""),
        "alt_ids": as_list(metadata.get("alt_ids")),
        "alt_titles": as_list(metadata.get("alt_titles")),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def build_baseline(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Digest every control record in a merged corpus, collisions included.

    The committed baseline maps one key to one digest, so nine keys holding 48
    records recorded 9 digests and shadowed 39 records with distinct text, all
    of them inside the two frameworks this rebuild touches. The value is a
    sorted list, so a key that loses one of its records is visible. [measured]
    """
    digests: dict[str, list[str]] = {}
    n_records = 0
    for record in corpus["frameworks"]:
        framework_id = record["framework_id"]
        for control in record.get("controls") or []:
            key = f"{framework_id}:{control['control_id']}"
            digests.setdefault(key, []).append(content_digest(control))
            n_records += 1
    return {
        "content_digest_fields": [
            "description", "full_text", "title", "alt_ids", "alt_titles",
        ],
        "digests": {key: sorted(values) for key, values in sorted(digests.items())},
        "n_keys": len(digests),
        "n_records": n_records,
    }


def _snapshot_members() -> list[Path]:
    """Every file --commit can overwrite."""
    members = sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json"))
    for extra in (
        PROCESSED_DIR / "all_controls.json",
        PROCESSED_LICENSED_DIR / "all_controls.json",
        PROCESSED_DIR / "stopwords.json",
        BASELINE_PATH,
    ):
        if extra.exists():
            members.append(extra)
    return members


def _member_key(path: Path) -> str:
    """Where a snapshot member came from, so restore can put it back there.

    Relative to the repo for the real artifacts, absolute otherwise, because
    the tests snapshot a tmp_path. Recording only the file name would restore
    every member to the repo root.
    """
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def _member_target(key: str) -> Path:
    candidate = Path(key)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def snapshot_processed(
    root: Path = SNAPSHOT_ROOT, members: list[Path] | None = None,
) -> Path:
    """Copy every overwritable artifact into a content-addressed directory.

    git checkout recovers 29 of the 31 per-framework files. It cannot recover
    etsi.json, iso_27001.json or licensed/all_controls.json, and ISO has no
    scripted re-fetch path. Overwriting them without a copy is the one
    irreversible act in this plan.

    The directory is named by the digest of its own manifest rather than by a
    clock. Two runs over identical inputs land in one directory, so a second
    --commit cannot bury the pristine copy under a fresher timestamp, and no
    written artifact carries a clock read.

    Copies go through atomic_write_text rather than atomic_write_json: a
    rollback that re-serialises what it restores is not a rollback.
    """
    sources = members if members is not None else _snapshot_members()
    # Every member is read before anything is written, so a read failure
    # leaves no half-written snapshot for --restore to trust.
    payload = {
        _member_key(path): path.read_text(encoding="utf-8") for path in sources
    }
    manifest = {
        "files": {
            key: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for key, text in sorted(payload.items())
        },
        "n_files": len(payload),
    }
    name = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    destination = root / name
    for key, text in sorted(payload.items()):
        atomic_write_text(text, destination / "files" / key.lstrip("/"))
    atomic_write_json(manifest, destination / "manifest.json")
    logger.info("snapshot: %d file(s) -> %s", len(payload), destination)
    return destination


def restore_snapshot(snapshot: Path) -> int:
    """Put every file in `snapshot` back, after verifying it against the manifest.

    Raises:
        ValueError: If a snapshot member's digest does not match the manifest.
            A rollback that restores corrupted bytes is worse than none.
    """
    manifest = load_json(snapshot / "manifest.json")
    restored = 0
    for key, expected in sorted(manifest["files"].items()):
        member = snapshot / "files" / key.lstrip("/")
        text = member.read_text(encoding="utf-8")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{member} does not match its manifest digest "
                f"({actual[:16]} against {expected[:16]}). Refusing to restore."
            )
        atomic_write_text(text, _member_target(key))
        restored += 1
    logger.info("restored %d file(s) from %s", restored, snapshot)
    return restored


@dataclass
class RebuildReport:
    changed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    unchanged: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def touched_frameworks(self) -> set[str]:
        keys = self.changed + self.added + self.removed
        keys += [old for old, _ in self.renamed] + [new for _, new in self.renamed]
        return {key.split(":", 1)[0] for key in keys}


def _parser_classes() -> dict[str, type[BaseParser]]:
    """Every concrete parser class, keyed by framework_id.

    Raises:
        ValueError: If a parse_*.py module defines no BaseParser subclass.
    """
    classes: dict[str, type[BaseParser]] = {}
    for path in sorted(PARSERS_DIR.glob("parse_*.py")):
        module = importlib.import_module(f"parsers.{path.stem}")
        found = [
            value for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, BaseParser)
            and value is not BaseParser
            and value.__module__ == module.__name__
        ]
        if not found:
            raise ValueError(
                f"{path.name} defines no BaseParser subclass. Every parser "
                f"module must, or the rebuild silently skips a framework and "
                f"reports its controls as unchanged because it never ran one."
            )
        classes[found[0].framework_id] = found[0]
    return classes


def run_all(
    output_dir: Path, audit_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Run every parser into `output_dir`. Returns (controls, failures).

    audit_dir is required. BaseParser.__init__ defaults it to
    PROCESSED_REPAIR_AUDIT_DIR, so the previous version let a --dry-run write
    repair audits into the real data/processed/repair_audit/ while claiming to
    touch nothing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    parsed: dict[str, list[dict[str, Any]]] = {}
    failed: dict[str, str] = {}
    for framework_id, parser_class in sorted(_parser_classes().items()):
        try:
            result = parser_class(output_dir=output_dir, audit_dir=audit_dir).run()
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            failed[framework_id] = f"{type(error).__name__}: {error}"
            logger.error("%s FAILED: %s", framework_id, failed[framework_id])
            continue
        parsed[framework_id] = [
            control.model_dump(mode="json") for control in result.controls
        ]
    return parsed, failed


def diff_against_baseline(
    parsed: dict[str, list[dict[str, Any]]], baseline: dict[str, list[str]],
) -> RebuildReport:
    """Which control records changed anchor text, were added, moved id, or went.

    Comparison is per key on a MULTISET of digests, so a key holding several
    records is compared record by record rather than collapsing to its first
    writer.
    """
    report = RebuildReport()
    new: dict[str, Counter[str]] = {}
    for framework_id, controls in sorted(parsed.items()):
        for control in controls:
            key = f"{framework_id}:{control['control_id']}"
            new.setdefault(key, Counter())[content_digest(control)] += 1
    old = {key: Counter(values) for key, values in baseline.items()}

    surplus_new: dict[str, Counter[str]] = {}
    surplus_old: dict[str, Counter[str]] = {}
    for key in sorted(set(new) | set(old)):
        mine, theirs = new.get(key, Counter()), old.get(key, Counter())
        report.unchanged += sum((mine & theirs).values())
        left_new, left_old = mine - theirs, theirs - mine
        if left_new:
            surplus_new[key] = left_new
        if left_old:
            surplus_old[key] = left_old

    # A rename is content that survived under a different id, within one
    # framework. For the 111 id-shape changes in the eleven it finds nothing,
    # because the old record is a stub and the new one is prose, and that is
    # the honest answer rather than a failure. It exists so `removed` means
    # "content gone" for every framework where a stub is not the before state.
    for old_key in sorted(surplus_old):
        framework = old_key.split(":", 1)[0]
        for digest in list(surplus_old[old_key]):
            match = next(
                (k for k in sorted(surplus_new)
                 if k.split(":", 1)[0] == framework and surplus_new[k][digest]),
                None,
            )
            if match is None:
                continue
            surplus_new[match][digest] -= 1
            surplus_old[old_key][digest] -= 1
            report.renamed.append((old_key, match))

    for key, counter in sorted(surplus_new.items()):
        if sum(counter.values()) and key in surplus_old and sum(
            surplus_old[key].values()
        ):
            report.changed.append(key)
        elif sum(counter.values()):
            report.added.append(key)
    for key, counter in sorted(surplus_old.items()):
        if sum(counter.values()) and not sum(surplus_new.get(key, Counter()).values()):
            report.removed.append(key)

    report.changed.sort()
    report.added.sort()
    report.removed.sort()
    report.renamed.sort()
    return report


def assert_expected_frameworks_only(report: RebuildReport) -> None:
    """Halt on a framework that moved when it should not, or did not when it should.

    The previous version said "if capec, cwe, asvs, owasp_cheat_sheets,
    nist_800_53, mitre_atlas or any other framework appears in that list,
    stop". That is an instruction, and this plan's header sends execution to an
    autonomous runner. main() raised only on a parser exception. An unexpected
    change was logged at INFO and --commit copied regardless. A control whose
    only enforcement is prose is decorative (ledger lesson 4).

    Raises:
        SystemExit: On any unexpected framework, any missing framework, or an
            unchanged count that is not exactly the pre-measured 3,786.
    """
    allowed = EXPECTED_CHANGED_FRAMEWORK_IDS | EXPECTED_ADDED_FRAMEWORK_IDS
    touched = report.touched_frameworks()
    unexpected = sorted(touched - allowed)
    if unexpected:
        raise SystemExit(
            f"these frameworks moved and their parsers were not touched: "
            f"{unexpected}. Their sources are pinned and 3,786 of their control "
            f"records were pre-measured as reproducing byte-identically, so a "
            f"change here is a defect this plan introduced, not a source change."
        )
    silent = sorted(EXPECTED_CHANGED_FRAMEWORK_IDS - touched)
    if silent:
        raise SystemExit(
            f"these frameworks got a parser in Tasks 3-13 and produced no change: "
            f"{silent}. Every one of their baseline records is a stub whose "
            f"description equals its title, so every one must move. A parser "
            f"that silently no-ops leaves the previous artifact in place while "
            f"the run reports success."
        )
    if report.unchanged != EXPECTED_UNCHANGED_RECORDS:
        raise SystemExit(
            f"{report.unchanged} unchanged records against the pre-measured "
            f"{EXPECTED_UNCHANGED_RECORDS}. Below it, a framework outside the "
            f"eleven stopped reproducing. Above it, a new parser reproduced a "
            f"stub, which means it emitted the OpenCRE section name instead of "
            f"the source's prose."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("build/rebuild"))
    parser.add_argument("--audit-dir", type=Path, default=Path("build/rebuild_audit"))
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", type=Path, default=None)
    parser.add_argument("--list-snapshots", action="store_true")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_ROOT / "results/corpus/rebuild_diff.json")
    args = parser.parse_args()

    if args.list_snapshots:
        for path in sorted(SNAPSHOT_ROOT.glob("*/manifest.json")):
            manifest = load_json(path)
            logger.info("%s  %d file(s)", path.parent.name, manifest["n_files"])
        return
    if args.restore is not None:
        restore_snapshot(args.restore)
        return
    if args.commit and args.dry_run:
        raise SystemExit("--commit and --dry-run are mutually exclusive.")

    baseline = load_json(BASELINE_PATH)["digests"]
    parsed, failed = run_all(args.scratch, args.audit_dir)
    if failed:
        raise SystemExit(
            f"{len(failed)} parser(s) failed: {sorted(failed)}. A rebuild that "
            f"skips a framework leaves the previous artifact in place while "
            f"reporting success."
        )
    report = diff_against_baseline(parsed, baseline)

    logger.info(
        "rebuild: %d frameworks, %d unchanged records, %d changed, %d added, "
        "%d removed, %d renamed",
        len(parsed), report.unchanged, len(report.changed), len(report.added),
        len(report.removed), len(report.renamed),
    )
    for bucket, keys in (("changed", report.changed), ("added", report.added),
                         ("removed", report.removed)):
        counts: Counter[str] = Counter(key.split(":", 1)[0] for key in keys)
        for framework_id, count in sorted(counts.items()):
            logger.info("  %-8s %-26s %d", bucket, framework_id, count)

    atomic_write_json(
        {
            "changed": report.changed, "added": report.added,
            "removed": report.removed,
            "renamed": [list(pair) for pair in report.renamed],
            "unchanged": report.unchanged,
        },
        args.report,
    )
    assert_expected_frameworks_only(report)

    if args.commit:
        snapshot = snapshot_processed()
        logger.info("rollback: --restore %s", snapshot)
        for source in sorted(args.scratch.glob("*.json")):
            atomic_write_json(
                load_json(source), PROCESSED_FRAMEWORKS_DIR / source.name,
            )
        logger.info("committed %d artifact(s) into %s",
                    len(list(args.scratch.glob("*.json"))),
                    PROCESSED_FRAMEWORKS_DIR)


if __name__ == "__main__":
    main()
```

The commit path uses `atomic_write_json(load_json(source), ...)` rather than
`shutil.copy2`, per the plan's own Global Constraint. It is byte-identical:
`BaseParser.run` writes its output through the same helper at `base.py:346`, and
`atomic_write_json` is deterministic (sorted keys, 2-space indent,
`ensure_ascii=False`, trailing newline). **[measured]**

- [ ] **Step 4: Regenerate the baseline and prove it agrees with the old one**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json

from scripts.rebuild_corpus import BASELINE_PATH, build_baseline
from tract.io import atomic_write_json, load_json
from tract.text_selection import merged_corpus_path

corpus = load_json(merged_corpus_path())
baseline = build_baseline(corpus)
baseline["generated_from"] = str(merged_corpus_path())
print("keys:", baseline["n_keys"], "records:", baseline["n_records"])
atomic_write_json(baseline, BASELINE_PATH)
PYEOF
"$PY" -m pytest tests/test_rebuild_corpus.py -q
"$PY" -m mypy scripts/rebuild_corpus.py --strict
```

Expected: `keys: 4222 records: 4261`. **[measured, orchestrator]** The old file
declared `n_controls: 4222` against those same 4,261 records, so this step
recovers the 39 shadowed ones. Run
`tests/test_rebuild_corpus.py::TestTheRegeneratedBaselineAgreesWithTheCommittedOne`
**before** overwriting the file, or it skips.

`git add data/processed/pre_rebuild_control_hashes.json` in Step 9's commit. Note
that the corpus this is derived from is the overlay, so a checkout without the
overlay produces 3,905 keys instead of 4,222 and Step 5 will report the missing
frameworks as `removed`. Run this task on a checkout that holds the overlay.

- [ ] **Step 5: Dry run, and read the diff before touching anything**

```bash
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --dry-run
```

Every line must be explainable before proceeding, and the script now halts on
its own if it is not:

- `0 parser failure(s)`, enforced by the `SystemExit` above the diff.
- `unchanged` exactly **3,786**, enforced by `assert_expected_frameworks_only`.
  Anything else means a framework outside the eleven stopped reproducing, or a
  new parser reproduced a stub.
- `changed` and `removed` reported only for the eleven. `capec`, `cwe`, `asvs`,
  `owasp_cheat_sheets`, `nist_800_53` and `mitre_atlas` appearing raises
  `SystemExit` rather than printing a warning the runner reads past.
- `removed` of roughly **111**, and the assertion is on framework membership,
  not on that number. Composition: wstg 59, nist_800_63 25, owasp_proactive_controls
  10, enisa 10, csa_ccm 7. **[measured]** Each is an OpenCRE-derived stub id the
  new parser replaces with the source's own id.
- `renamed` of **0** for the eleven, expected and stated: the old record is a
  stub whose description equals its title, so no prose control can content-match
  one. The bucket exists so `removed` still means "content gone" elsewhere.
- `added` covering the controls the new parsers emit beyond the stubs, plus
  `owasp_llm_top10_2026`'s 10. Expected magnitudes from the parser tasks:
  dsomm 194 against 183 stubs, wstg 115 against 59, csa_ccm 224 against 29,
  iso 27001 unchanged at 93.

Record `unchanged`, and the per-framework `changed` / `added` / `removed` counts
from `results/corpus/rebuild_diff.json`, in the run ledger before Step 6.

- [ ] **Step 6: Snapshot and commit the rebuild**

```bash
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --commit 2>&1 | tee /tmp/rebuild.log
grep "rollback: --restore" /tmp/rebuild.log
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --list-snapshots
```

Copy the `--restore` line into the run ledger before running anything else. That
directory is the only recovery path for
`data/processed/frameworks/etsi.json`, `iso_27001.json` and
`data/processed/licensed/all_controls.json`, which `git checkout` cannot restore
and `scripts/fetch_frameworks.py` cannot refetch. **[measured]**

```bash
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" parsers/validate_all.py
```

- [ ] **Step 7: Regenerate the stop word list**

`data/processed/stopwords.json` is derived from the corpus, committed, applied
to every control and hub text, and hashed into every fold record at
`tract/training/orchestrate.py:351`. The rebuild replaces 436 stub records with
prose. The v2 plan mentioned `stopwords` zero times in 6,987 lines. **[measured]**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
from pathlib import Path
before = json.loads(Path("data/processed/stopwords.json").read_text(encoding="utf-8"))
print("before:", before["count"], "words from", before["n_documents"], "documents")
Path("/tmp/stopwords_before.json").write_text(json.dumps(before, sort_keys=True))
PYEOF
PYTHONPATH=. "$PY" -m scripts.build_stopwords
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
from pathlib import Path
before = json.loads(Path("/tmp/stopwords_before.json").read_text(encoding="utf-8"))
after = json.loads(Path("data/processed/stopwords.json").read_text(encoding="utf-8"))
print("after:", after["count"], "words from", after["n_documents"], "documents")
print("added:", sorted(set(after["stopwords"]) - set(before["stopwords"])))
print("removed:", sorted(set(before["stopwords"]) - set(after["stopwords"])))
assert after["min_doc_freq"] == before["min_doc_freq"] == 0.05
PYEOF
"$PY" -m pytest tests/test_stopword_filtering.py tests/test_stopword_protection.py -q
```

Before: 78 words from 4,783 documents at `min_doc_freq = 0.05`. **[measured]**
After: **[unmeasured]** until the eleven parsers exist. Record both counts and
the full added and removed word lists in the run ledger.

One consequence to state rather than discover later. `scripts/build_stopwords.py:37`
reads `PROCESSED_DIR / "all_controls.json"`, the tracked corpus, not
`merged_corpus_path()`. **[measured]** So the list is identical on a machine with
the overlay and one without, which is the right property for a committed artifact
that is hashed into every run record. The cost is that the overlay frameworks'
boilerplate does not vote on the list. Under Contract Rule 3 that is nine
frameworks. Leave it, and record it in "What this plan does not close".

```python
# tests/test_rebuild_corpus.py: append

def test_the_committed_stopword_list_reproduces_from_the_committed_corpus() -> None:
    """Catches the staleness directly rather than by remembering to rerun it.

    The list is applied to every control and hub text and hashed into every
    fold record. A list built for a corpus that no longer exists is invisible
    in the metrics and changes every one of them.
    """
    import json

    from scripts.build_stopwords import collect_documents
    from tract.stopwords import STOPWORDS_PATH, generate_stopwords

    committed = json.loads(STOPWORDS_PATH.read_text(encoding="utf-8"))
    documents, protected = collect_documents()
    words = generate_stopwords(
        documents,
        min_doc_freq=committed["min_doc_freq"],
        max_words=committed["max_words"],
        protect=protected,
    )
    assert sorted(words) == committed["stopwords"]
    assert len(documents) == committed["n_documents"]
```

- [ ] **Step 8: Record the 63 published assignments this rebuild orphans**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json

from tract.io import atomic_write_json, load_json
from tract.text_selection import merged_corpus_path

rows = [json.loads(l) for l in
        open("build/dataset/crosswalk_v1.0.jsonl", encoding="utf-8") if l.strip()]
corpus = load_json(merged_corpus_path())
live = {
    f"{record['framework_id']}:{control['control_id']}"
    for record in corpus["frameworks"] for control in record.get("controls") or []
}
orphans = [
    {
        "control_id": row["control_id"], "framework_id": row["framework_id"],
        "hub_id": row["hub_id"], "review_status": row["review_status"],
        "section_id": row.get("section_id", ""),
    }
    for row in rows
    if row["control_id"].split(":", 1)[1] not in
    {k.split(":", 1)[1] for k in live if k.startswith(row["framework_id"] + ":")}
]
atomic_write_json(
    {
        "note": (
            "Published rows in crosswalk_v1.0.jsonl whose control_id the "
            "corpus rebuild retires. All carry review_status='ground_truth'. "
            "tract/export/canonical.py:76 filters on review_status='accepted', "
            "so diff_snapshots never sees them and no UPDATE_CONTROL or "
            "DELETE_CONTROL changeset will mention them. Republication is "
            "banned, so this file is the record until that ban lifts."
        ),
        "n_rows": len(orphans),
        "rows": sorted(orphans, key=lambda r: (r["framework_id"], r["control_id"],
                                               r["hub_id"])),
    },
    "results/corpus/retired_control_ids.json",
)
print("orphaned published rows:", len(orphans))
PYEOF
```

Expected: **63**. **[measured, orchestrator]** 56 `enisa:enisa:Table 5:` (38) and
`Table 3:` (18), plus 7 `csa_ccm:csa_ccm:IVS-0{1,2,4,5,6,8,9}`. The file carries
control ids and hub ids and no control text, so it is safe to track for the
overlay frameworks as well.

```python
# tests/test_rebuild_corpus.py: append

def test_every_retired_published_id_is_named() -> None:
    """63 published assignments lose their control identity. [measured]

    The export path cannot see them: all 63 carry review_status='ground_truth'
    and tract/export/canonical.py filters on 'accepted', so no changeset will
    ever mention them. This file is the only record.
    """
    import json

    retired = json.loads(
        (REPO_ROOT / "results/corpus/retired_control_ids.json")
        .read_text(encoding="utf-8")
    )
    assert retired["n_rows"] == 63
    assert {row["framework_id"] for row in retired["rows"]} == {"enisa", "csa_ccm"}
    assert {row["review_status"] for row in retired["rows"]} == {"ground_truth"}
```

- [ ] **Step 9: Verify the licence channel before staging anything**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
git add scripts/rebuild_corpus.py tests/test_rebuild_corpus.py \
        data/processed/pre_rebuild_control_hashes.json \
        data/processed/stopwords.json \
        data/processed/all_controls.json \
        results/corpus/rebuild_diff.json \
        results/corpus/retired_control_ids.json
git add data/processed/frameworks/
PYTHONPATH=. "$PY" - <<'PYEOF'
import subprocess
import sys

from tract.config import OVERLAY_FRAMEWORK_IDS

staged = subprocess.run(
    ["git", "diff", "--cached", "--name-only"],
    capture_output=True, text=True, check=True,
).stdout.split()
leaked = [
    path for path in staged
    if path.startswith("data/processed/frameworks/")
    and path.rsplit("/", 1)[1].removesuffix(".json") in OVERLAY_FRAMEWORK_IDS
]
if leaked:
    sys.exit(
        f"these paths carry text on terms a CC0 grant cannot carry and are "
        f"staged: {leaked}. .gitignore is not covering them in this checkout. "
        f"git push is the publication event regardless of what any publish-path "
        f"filter does."
    )
print(f"licence channel clear: {len(staged)} staged path(s), 0 overlay frameworks")
PYEOF
```

The v2 version checked this with `git status --porcelain` and a paragraph telling
the operator to look. `git add data/processed/frameworks/` exits 1 on the ignored
members while staging the rest, so the exit code says nothing useful about which
files landed. The check above reads what git staged and exits non-zero.

The `OVERLAY_FRAMEWORK_IDS` import is the ordering dependency stated in the task
header. An `ImportError` here means Contract Rule 3's licence tiering has not
landed, and the correct response is to land it, not to fall back to
`RESTRICTED_FRAMEWORK_IDS` and stage seven conditional frameworks' prose.

```bash
"$PY" -m pytest tests/test_rebuild_corpus.py tests/test_corpus_invariants.py \
      tests/test_licensed_text_not_tracked.py tests/test_holdout_framework.py \
      tests/test_framework_licenses.py tests/test_parser_manifest_coverage.py \
      tests/test_prose_reachability.py -q
git commit -m "chore: rebuild the corpus from pinned sources, with the per-record diff

3,786 control records outside the eleven reproduce unchanged, pre-measured at
1,897 from the 19 importable parsers plus capec 558 and cwe 1,331 with 0
mismatch. The eleven's 436 records were OpenCRE-derived stubs whose description
equalled their title, so every one moves: roughly 111 through an id-shape change
and the rest in place. The baseline is regenerated with a five-field content
digest and a per-key digest multiset, which recovers 39 records that nine
colliding keys had shadowed. The stop word list is rebuilt from the new corpus.
Every overwritten file is snapshotted first, because git cannot restore three of
them and no script can refetch ISO 27001."
```

- [ ] **Step 10: Regenerate the training file against the corpus that now exists**

Task 14 made `hub_links_training.jsonl` a function of the corpus. This step is
what closes ledger lesson 6, and the sidecar test from Task 14 Step 8 is red
until it runs.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.text_selection import merged_corpus_sha256
from tract.training.data_quality import curated_link_filter_report, save_training_links

report, raw_hash = curated_link_filter_report()
print("corpus:", report.corpus_path)
print("training links:", len(report.kept))
print("dropped unresolved:", report.dropped_unresolved)
print("dropped thin anchor:", report.dropped_thin_anchor)
save_training_links(report.kept, raw_hash, merged_corpus_sha256())
PYEOF
"$PY" -m pytest tests/test_data_quality.py tests/test_ceiling_study.py -q
git add data/training/hub_links_training.jsonl \
        data/training/hub_links_training.meta.json
git commit -m "chore: rebuild the training links against the rebuilt corpus

hub_links_training.jsonl resolves every link through ProseIndex, so it is a
function of the corpus, and Step 6 changed the corpus. The sidecar records the
sha256 of the corpus this file was built from, and the test that compares the
two is what makes the ordering enforceable rather than a claim."
```

Expected: `training links: 4389`, sixteen named unresolved drops, zero thin
anchors, and the corpus digest matching `data/processed/licensed/all_controls.json`.
**[derived]** A different number here than in Task 14 Step 9 means the rebuild
changed which links resolve, which is information the run ledger needs and the
Task 14 commit message does not carry. Record both.

---
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
