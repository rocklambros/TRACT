# L3 — implementation report

Branch `semantic-rebuild`. Two commits, oldest first.

| commit | scope |
|---|---|
| `3235e0c` | Part A — the tier decision and the copyleft gate's new demand |
| `e950b49` | Part B — the second export leak, `opencre_export/` |

`tract/text_selection.py` was not touched. `results/corpus/before.json` and
`results/corpus/link_resolution_before.jsonl` are unmodified. No `git add -f`,
no `git push`, no model loaded.

---

## Part A — the tier decision

### Order of operations, as instructed

1. Five `.gitignore` lines removed (`biml`, `samm`, `wstg`,
   `owasp_top10_2021`, `owasp_proactive_controls`). Verified with
   `git check-ignore` per file before proceeding: all five reported unignored.
2. `git add` on the five artifacts, plain add.
3. `CONDITIONAL_FRAMEWORK_IDS` shrunk to `{dsomm, csa_ccm}`.
4. Tests updated.

The order matters because `test_the_registry_names_no_framework_that_does_not_exist`
exempts `OVERLAY_FRAMEWORK_IDS` by name. Shrinking the set before the artifacts
are tracked reports the five as stale registry entries on a fresh CI checkout,
which points at the wrong problem.

### `data/processed/all_controls.json` did NOT move

Recorded before the change and reproduced after it by rebuilding the corpus
into a scratch directory under the new tiers:

```
before  7106642cb3a7995355d25db6b356217a4f7f2d5eb70889af70ed2d6844f287e6
after   7106642cb3a7995355d25db6b356217a4f7f2d5eb70889af70ed2d6844f287e6
        29 frameworks, 4141 controls, unchanged
```

Byte-identical, and by construction rather than by luck. Measured on all seven
former conditional members: every control carries `description == title` and no
`full_text`, so the redaction the five no longer receive was already a no-op.
`_redact_prose` returns the same object when there is nothing to redact.

The overlay artifact does move, from 7 unpublishable frameworks to 4. It is
gitignored, so nothing tracked changes.

### The reasoning, recorded in `tract/config.py`

Against the constant, not in a commit message, because the next reader will
ask there. Three paragraphs: the parity argument for the five, the GPL-3.0
section 5 argument for `dsomm`, and the "this reverses nothing" argument for
`csa_ccm`. The `.gitignore` comment block above the two surviving lines now
says two rather than seven and points at the config comment.

`NOTICE` gains a short paragraph under "Licence texts" recording that thirteen
frameworks are copyleft, that twelve have their text tracked, and that the
share-alike obligations are discharged through NOTICE and `LICENSES/` rather
than by withholding. That is the record a downstream reader acts on, and it was
the one place the decision would otherwise be invisible.

### The copyleft gate: demand changed, gate kept

`test_every_copyleft_framework_is_conditional` is gone and
`TestCopyleftObligationsAreDischarged` replaces it. Four assertions plus two
constructed cases:

| condition | clause | assertion |
|---|---|---|
| 1 | CC BY-SA 4.0 3(a)(1), GPL-3.0 §4 | a row in NOTICE |
| 2 | GPL-3.0 §4, CC BY-SA 4.0 3(a)(1)(A) | `LICENSES/<id>.txt` for every recorded SPDX identifier |
| 3 | GPL-3.0 §5(a), CC BY-SA 4.0 3(a)(1)(B) | NOTICE's modification statement, present and stating its scope |

Condition 3 is corpus-wide where 1 and 2 are per framework, and the docstring
says so rather than dressing it up. The transforms are corpus-wide: every
control statement that reaches `data/processed/` goes through the same
`tract/sanitize.py` path, so one notice covers thirteen frameworks and losing it
loses all thirteen at once. Its failure message names them.

A fourth test asserts `_copyleft()` is non-empty, because the three conditions
above are conjunctive loops and the likeliest way a conjunctive gate fails is
by looping over nothing.

`_copyleft()` itself is unchanged and still substring-based on `GPL` and
`CC-BY-SA`. Fixing that blindness would pull `csa_aicm` into a tier, which is
the owner question NOTICE escalates and a test currently holds open. Out of
scope here, still open, listed under concerns.

### `PRE_EXISTING_EXPOSURE` deleted, with the three tests around it

The set existed as the exception carve-out for "copyleft implies overlay".
Under the new demand nothing subtracts it: the three conditions apply uniformly
to all thirteen copyleft sources with no member exempt. A list no gate consults
is not a ratchet, it is a comment with an `assert` under it, and keeping it
would have meant three tests asserting facts about a constant that no longer
influences any outcome.

The ratchet that now carries the weight is
`test_no_overlay_framework_is_still_tracked`, which is unchanged and covers all
four overlay members. Mutation M6 below confirms it is live.

### Mutations run for Part A

Each applied, suite run, reverted, and the tree diffed clean afterwards.

| # | mutation | result |
|---|---|---|
| M1 | delete the `samm` row from NOTICE | condition 1 red, plus `test_notice_lists_exactly_the_registry` |
| M2 | remove `LICENSES/CC-BY-SA-4.0.txt` | condition 2 red |
| M3 | rename the "Modifications to framework text" heading | condition 3 red on `statement is not None` |
| M4 | modification statement stops naming `tract/sanitize.py` | condition 3 red on the scope claim |
| M5 | `_copyleft()` matches nothing | non-vacuity test red, plus both constructed cases |
| M6 | put the five back in CONDITIONAL while their files are tracked | `test_no_overlay_framework_is_still_tracked` and `test_every_overlay_framework_has_a_gitignore_line` red |

One assertion was rewritten after M1: the constructed cases originally asserted
the offender list equalled exactly `[new_id]`, so M1 killed them for an
unrelated reason and reported the constructed case as broken instead of
reporting the real regression. They now assert the DELTA the injection causes,
which isolates them. Re-ran M1 and M2 against the delta form: each now kills
only the condition it targets.

The two constructed cases are the "prove it fails" the brief asked for. A
CC-BY-SA-4.0 source injected into `FRAMEWORK_LICENSES` with no NOTICE row is
rejected by condition 1. A CC-BY-SA-2.5 source, share-alike and with no shipped
text, is rejected by condition 2. Two cases rather than one because a single
injection cannot tell the two omissions apart.

---

## Part B — `opencre_export/`

### What was leaking

`tract export --opencre` → `_cmd_export_opencre` → `write_opencre_csv` →
`generate_opencre_csv`, which wrote `row["description"]` into the
`<Standard>|description` column with no licence check at all. The default
output directory was the literal `"./opencre_export"` at two CLI call sites,
and `git check-ignore` reported it not ignored with seven files tracked.

### The fix

`exportable_description` moved from `tract/export/canonical.py` to
`tract/licensing.py`. Both exporters now import the one implementation.
`tract/export/canonical.py` re-exports the name at its old import path, so
`tests/test_canonical_export.py` and every other caller are unchanged. The
alternative, importing `tract.export.canonical` from `tract.export.opencre_csv`,
would drag the pydantic snapshot schema into the CSV path for one function.

Applied per row in `generate_opencre_csv`, keyed on `row["framework_id"]` and
not on the `framework_id` argument. The two agree at every call site today
because `_cmd_export_opencre` filters the query per framework before writing.
Keying on the argument would let a withheld framework's row ride out under a
publishable one's name the first time a caller passes a mixed list, and
`generate_opencre_csv` is a public function.

Withheld rows keep their `CRE 0`, name, id and hyperlink columns, and get
`withheld_control_text` in the description. Same shape and same reasoning as the
canonical export: an empty cell says "this control has no description", which
is a different claim from "the publisher's terms do not permit us to hand you
the text". The count is logged per framework.

`PHASE5_OPENCRE_EXPORT_DIR` added to `tract/config.py` and both CLI call sites
read it. The gitignore gate and the CLI now cannot disagree about which
directory has to be ignored.

`opencre_export/` added to `.gitignore`, with a comment saying explicitly that
it stops the next run's output and does not untrack the seven files already in
git.

### The five existing CSVs are UNTOUCHED, and why

Not deleted, not rewritten, not re-exported. `git status opencre_export/` is
empty after both commits.

`csa_aicm` is already tracked and already published. Removing
`CSA_AI_Controls_Matrix.csv` now un-publishes nothing, moves metrics that other
artifacts quote, and pre-empts an owner decision that is escalated and recorded
in NOTICE's open-questions section. The job here was to stop the NEXT export
from writing prose it may not redistribute, not to relitigate what shipped.

Left alone is not the same as unchecked.
`test_no_tracked_export_csv_carries_an_overlay_framework_description` reads each
tracked CSV back, resolves its description column header through
`TRACT_TO_OPENCRE_NAME` to the framework that wrote it, and fails if that
framework is in `OVERLAY_FRAMEWORK_IDS` with a populated description column. It
asserts it found at least one tracked CSV and at least one populated column, so
it cannot pass by reading nothing. A CSV whose OpenCRE name maps to no framework
fails loudly rather than being skipped.

### Reachability, and why the fixture registers a name

`TRACT_TO_OPENCRE_NAME` holds six frameworks and none of them is in a licence
tier, so nothing has leaked through this path yet. That is what makes it worth a
gate rather than a note: the exposure is one name-map entry away, and that entry
is a one-line change nobody would recognise as a licensing decision.

The test fixture registers the entry with `monkeypatch.setitem` for
`sorted(OVERLAY_FRAMEWORK_IDS)[0]`, taken from the live constant so the fixture
cannot drift out of step with the tier. It asserts the framework does not
already have a real OpenCRE name, so the day one is added the fixture stops
masking it. With the entry registered, every assertion in the class is reachable
in both directions: without the filter these rows write the publisher's
statement into the description column.

### Mutations run for Part B

| # | mutation | result |
|---|---|---|
| N1 | no filter, the pre-fix implementation | 6 tests red, including the written-file one |
| N2 | filter keyed on the argument instead of the row | mixed-list test red, `ValueError` test red |
| N3 | filter withholds every framework | publishable-direction test red, plus the pre-existing `test_standard_columns_populated` |
| N4 | withheld text replaced by an empty cell | placeholder test red |
| N5 | `opencre_export/` removed from `.gitignore` | gitignore probe red |
| N6 | a tracked CSV whose description column belongs to an overlay framework | the tracked-file gate's logic rejects it |

N6 was run against the gate's logic on a constructed row rather than by writing
a file into `opencre_export/`, because staging a file there is the one thing
this work exists to prevent and a mistimed `git add -A` from a concurrent task
would have committed it.

N3 is the direction that matters most for a filter: a gate that withheld
everything would pass every "no prose leaked" assertion and destroy the
deliverable. `test_a_publishable_framework_keeps_its_control_text` is the
assertion that kills it.

---

## Verification

Baseline before this work, measured on this machine rather than taken from the
brief: **13 failed, 1490 passed, 22 skipped, 3 xfailed**. The brief said 1,476;
a concurrent task has added tests since it was written.

After both commits: **13 failed, 1502 passed, 22 skipped, 3 xfailed**. The
failure SET is byte-identical to baseline, all 13 the model-loading and missing
`anthropic` paths. Net +12 passing: Part A removed 3 tests and added 5, Part B
added 10.

Confirmed under both `-p no:randomly` and the default random ordering.

`mypy --strict` over `tract/ parsers/ scripts/phase1a/ scripts/phase1b/
scripts/phase0/runpod_provision.py`: 26 errors, all the pre-existing
`huggingface_hub` missing-stubs class, unchanged in count and kind.
`tests/test_framework_licenses.py` and `tests/test_opencre_csv.py` are both
`--strict` clean. The second one was not: `_make_row` returned a bare `dict` and
produced 12 errors at HEAD. It now returns `ExportableAssignment` with all ten
fields, which is a stronger fixture as well as a clean one, because a column the
exporter reads can no longer be dropped from it without mypy saying so.

---

## Concerns

1. **`csa_aicm` is still unresolved and its prose is still in git, in two
   places.** 243 controls at a 176-character median in
   `data/processed/frameworks/csa_aicm.json`, and 184 rows at up to 485
   characters in `opencre_export/CSA_AI_Controls_Matrix.csv`. Both tracked,
   both under a notice reserving redistribution, in no licence tier. Part B
   stops the next export from adding to it and deliberately does not remove it.
   Owner decision, escalated, recorded in NOTICE.

2. **The tier derivation is still substring-based and still blind.**
   `_copyleft()` matches `GPL` and `CC-BY-SA`, so a publisher who reserves
   rights outright produces no tier membership at all. That is the structural
   cause of concern 1 and it is unchanged here. Fixing it means moving
   `csa_aicm` into a tier, which is the owner decision itself rather than a
   test change.

3. **Condition 3 of the new gate overlaps
   `tests/test_notice_completeness.py::test_the_section_exists`.** Deleting
   NOTICE's modification section turns both red. The overlap is deliberate and
   the framings differ, one asserting the section states the transforms and the
   other asserting it covers the copyleft corpus, but a reader should know a
   single deletion fires two files.

4. **Nothing enforces that a framework tracked under CC BY-SA keeps its
   attribution in a DERIVED artifact.** The three conditions are checked
   against the repository. The dataset bundle and model repo ship NOTICE and
   `LICENSES/` under L1.5, which is the mechanism, but no test ties a published
   bundle's framework list back to `_copyleft()`. A framework added to the
   corpus and published without reaching NOTICE would fail condition 1 here,
   so the gap is narrow, and it is a gap.

5. **`opencre_export/` is now ignored while seven files inside it stay
   tracked.** That is the intended state and it is an odd one to read: editing
   a tracked file there still shows in `git status`, adding a new one does not.
   Anyone re-running the export and expecting to commit the result has to
   `git add -f`, which is the friction the ignore line is for.

6. **The five newly tracked artifacts carry no prose today and nothing
   guarantees they will not tomorrow.** They are title-only stubs because their
   parsers have not landed. When those parsers write real CC BY-SA control
   statements, the text enters git on the next `git add`, which is exactly what
   this change decided to allow. The decision is recorded; the consequence is
   the point rather than a surprise.
