# Detector B applicability and the alt_titles validator

Status: DONE_WITH_CONCERNS. Two commits, `0d95868` (Part A) and `e4988f6` (Part B).
Tests 1,520 -> 1,546 passing (+26), same 13 pre-existing environmental failures.

## Part A — detector B declared inapplicable where the link file names a coarser level

### What landed

`tract/corpus_report.py`:

- `COARSE_NAME_RATIO: Final[float] = 2.0` — the threshold on
  `distinct(section_id) / distinct(section_name)`.
- `_coarse_name_frameworks(grouped)` / `coarse_name_frameworks(links_path=None)` — the
  derivation. Guards division by zero by skipping a framework with no non-empty
  `section_name`, which is also the right answer on the merits: detector B never reads a
  name it was not given, so it is already inert there and there is nothing to declare.
- `DETECTOR_B_INAPPLICABLE: Final[frozenset[str]] = frozenset({"dsomm"})` — the declared
  set. Runtime reads this rather than re-deriving per run, so the exemption arrives
  through a reviewed edit and not through a link-file change nobody looked at.
- `_wrong_anchor(..., *, detector_b: bool)` — keyword-only and required, so the single
  call site has to state which way it is going.

### The measurement, re-derived rather than trusted

I re-ran the owner's measurement over all 22 frameworks carrying curated links
(2026-08-19) and it reproduces exactly:

| framework | distinct ids | distinct names | ratio |
|---|---|---|---|
| dsomm | 183 | 18 | **10.167** |
| biml | 20 | 17 | 1.176 |
| wstg, samm, nist_800_53, capec, asvs, ... | — | — | 1.000 |
| enisa (lowest) | 10 | 33 | 0.303 |

dsomm is the only framework at or above 2.0. The threshold has a factor of 8.6 of
headroom below dsomm and a factor of 1.7 above biml, so neither a name repair nor a
handful of new links moves a framework across it by accident. This is asserted in
`test_the_real_link_file_puts_only_dsomm_over_the_ratio`, not merely cited in a comment.

### Result

`dsomm wrong_anchor_risk 198 of 213` becomes **`0 of 3`**.

The three survivors are detector C's, and they are the uuid-suffixed WAF ids the ledger
named: `f0e01814-...-d20b`, `...-d20b-medium`, `...-d20b-advanced`. C checks them and
flags none. Detector A never applied to dsomm (`by_title == 0`).

The denominator moves with the detector, which is requirement 6. I verified the split
directly rather than inferring it:

- 210 rows go checked -> unchecked. Detector B was the sole applicable check on each.
- 3 rows stay checked. Detector C reaches each.
- 0 rows are flagged-but-unchecked after the change.

Only the dsomm row moves. I ran the whole report with the exemption disabled and enabled
and diffed all 22 framework rows and all 4,405 resolution rows: the change touches
dsomm and nothing else.

### Mutation audit — 10 mutations, 10 killed

| # | Plausible wrong implementation | Killed by |
|---|---|---|
| A1 | Exempt the member from all three detectors, not just B | `test_detector_c_still_fires_for_a_member`, `test_detector_a_still_fires_for_a_member`, real-row test |
| A2 | Skip B's verdict but keep counting it in the denominator | `test_detector_b_is_skipped_...`, real-row test (would read 0 of 213) |
| A3 | `>` instead of `>=` at the ratio | `test_the_ratio_decides_membership_at_the_boundary` |
| A4 | Declare `biml` alongside `dsomm` without the property | `test_the_declared_set_equals_the_derived_set` |
| A5 | Loosen the ratio to 1.1 so a second framework qualifies | ratchet + boundary test |
| A6 | Invert the ratio (names over ids) | ratchet + boundary test |
| A7 | Drop the empty-name guard (divide by zero) | `test_a_link_file_with_no_names_does_not_divide_by_zero` |
| A8 | Switch B off for everyone, not only members | pre-existing `test_detector_b_id_hit_...` + the new A/B test |
| A9 | Count empty `section_name`s as a distinct name | nameless-file test |
| A10 | Derive from raw link counts instead of distinct values | ratchet + boundary + nameless tests |

The strongest of the new tests is `test_detector_b_is_skipped_for_a_member_and_not_for_anyone_else`:
identical link content under two `framework_id`s, `dsomm` and `demo`, against one corpus.
Membership is the only variable, so a pass cannot come from the corpus or the link shape.
It kills A2 and A8 from opposite directions.

## Part B — the alt_titles validator, ruling P2

### Step 1: report-only sweep, run before any code changed

```
=== licensed overlay (data/processed/licensed/all_controls.json) ===
  controls carrying the key      70
  carrying a non-empty value     30 {'NIST AI 100-2': 5, 'OWASP Cheat Sheets': 25}
  carrying an empty list         40
  rejected by the validator      0
  VERDICT: CLEAN

=== tracked corpus (data/processed/all_controls.json) ===
  (identical figures)
  VERDICT: CLEAN
```

This reconciles exactly with the owner's measurement: `owasp_cheat_sheets` 25,
`nist_ai_100_2` 5, 30 in total. The 40 extra `nist_ai_100_2` controls carry
`alt_titles: []`, an empty list, which the validator accepts as a legal declaration of
zero alternates. Every declaration is a list, all parser-generated. Nothing to rule on,
so step 3 proceeded.

### Step 3: the raise

The Task 2 helper's shape did allow reuse, with one edit. Validation is identical between
the two fields and only the downstream normaliser differs, so I lifted the check into
`_declared_strings(declared, framework, control_id, field)` and left two thin callers:
`_alternate_ids` runs `normalize_section_id`, `_alternate_titles` lowercases. Two copies
of the check would drift and the weaker copy would be the one nobody notices. The field
name is a parameter so the message sends a reader to the right parser.

The `or []` is gone on the title side as it already was on the id side, so `0` and
`False` reach the type check instead of being read as "the author wrote nothing".

### Mutation audit — 7 mutations, 7 killed

| # | Plausible wrong implementation | Killed by |
|---|---|---|
| B1 | Coerce entries with `str()` instead of raising | entry-type tests, both fields |
| B2 | Restore `or []`, folding 0 and False into an absent field | `test_a_falsy_non_string_is_not_read_as_an_absent_field` |
| B3 | Accept any field shape, so a dict or a number passes | field-type tests, both fields |
| B4 | Drop the list position from the message | position tests, both fields |
| B5 | Index empty-string alternates instead of skipping them | `test_an_empty_entry_is_skipped_rather_than_indexed` |
| B6 | Validate correctly and then index nothing | `test_a_well_formed_alternate_is_still_indexed` |
| B7 | Shared validator hardcodes `alt_ids` for both fields | `test_the_message_names_the_field_not_just_the_other_one` |

**The mutation audit found a blind assertion of mine.** My first version of the
empty-entry test probed `index.lookup("Demo", None, "")`. `lookup` skips the title branch
on a falsy name, so it returns `None` whether or not the empty key was indexed. B5
survived against it. I proved the blindness rather than assuming it, by forcing the key
into `_by_title` and showing `lookup` still answers `None` while `by_title` answers with
a hit, then rewrote the probe to use `by_title`. B5 then died. This is the third task in
the run where mutation testing found something a green suite could not.

## Baselines

`results/corpus/before.json` and `results/corpus/link_resolution_before.jsonl` are
**unmoved**. `git status --porcelain results/corpus/` is empty at both commits.

Part B is proved unmoved empirically: I rebuilt the report with the pre-Part-B
`text_selection.py` and with the new one and compared. The summary JSON is identical and
all 4,405 resolution rows are identical.

Part A is proved unmoved by construction and by inspection: all 214 dsomm rows in the
committed BEFORE JSONL are `channel: unresolved` with `wrong_anchor_checked: False`, and
`_wrong_anchor` never runs on an unresolved link, so the exemption cannot reach a single
committed row whichever corpus the baseline is rebuilt from. The committed dsomm summary
row already reads `wrong_anchor_risk: 0`.

### CONCERN — `--tag before` can no longer reproduce the committed baseline

I ran the instructed regeneration and it moved both files, so per the instruction I
stopped, restored from git, and did not commit a regenerated baseline. **The cause is not
this change.** The corpus itself moved when the DSOMM parser landed at `d8ad0c9`:

```
corpus_sha256 recorded in before.json  2440d7c062055f66
data/processed/licensed/all_controls.json on disk  5b0a428958e9e10a
```

The only row that differs is dsomm, and every differing column is a join column the
parser produced: `by_id` 0 -> 213, `unresolved` 214 -> 1, `distinct_anchors` 0 -> 182,
`dropped_by_prose_rule` 183 -> 1, `resolution_rate` 0.0 -> 0.9953. `wrong_anchor_risk` is
**not** in the differing set, because it reads 0 both before (no resolved links) and
after (this change). Without this change it would have regenerated as 198, so the change
is the reason the regenerated file is closer to the baseline rather than further from it.

The practical consequence for whoever owns the corpus evidence: `--tag before` is no
longer a reproduction of the BEFORE state, it is a rebuild of the current state under the
BEFORE name. Anyone running it will silently replace the baseline, which is ledger lesson
5. It needs either a pinned corpus argument, or an AFTER tag for the post-parser state,
or a guard on `corpus_sha256` in the tagged write path alongside the existing
`require_full_corpus` and `require_portable_paths` checks. That is an owner decision, not
mine, so I made no change to the write path.

### Second concern, smaller

`csa_ccm` and `etsi` both read `0 of 0` on the current corpus, while
`JOIN_WRONG_ANCHOR_BUDGET` pre-registers 1 for each. This predates both commits (both
rows are identical in `before.json`) and is untouched by this work, but it means the two
frameworks whose entries exist specifically so that the wrong-anchor gate "can fail in
both directions" are currently blind. Whatever gate Task 16 builds on that mapping will
not do what its comment claims until that is reconciled.

## Environmental note

Two mutation runs produced a false result from a stale `__pycache__`: the B7 mutation
replaced `{field!r}` with `'alt_ids'`, which is the same 9 characters, and the restore
landed inside the same one-second mtime granularity window, so Python reused the mutated
bytecode against restored source. I re-ran every mutation with `PYTHONDONTWRITEBYTECODE=1`
after clearing `__pycache__`. All 17 verdicts above are from that clean re-run. Anyone
building a mutation harness on this repo should set that variable from the start.

---

# Ruling R12 — refuse to replace a tagged artifact built from another corpus

Commit `6b7de3e`. Tests 1,577 -> 1,590 (+13), same 13 environmental failures.
`mypy --strict` clean on `tract/corpus_report.py`, `tract/text_selection.py` and
`scripts/corpus_report.py`.

## What landed

`tract/corpus_report.py` gains `require_unmoved_corpus(report, summary_path)`. It reads
the `corpus_sha256` recorded in an existing tagged artifact and compares it against this
run's. Four outcomes, all reachable:

| existing artifact | verdict |
|---|---|
| absent | write (first capture is not a replacement) |
| records the same digest | write (reproduction, not replacement) |
| records a different digest | RAISE, naming both digests and the file |
| unreadable or missing the digest | RAISE (unreadable provenance is not permission) |

`scripts/corpus_report.py` gains `--replace-baseline` and a `build_parser()` split so the
CLI surface can be read by a test without being run. `require_full_corpus` sits outside
the override branch, so the census guard binds either way.

## The drift is real and it moved twice during this session

```
committed results/corpus/before.json          2440d7c0...   31 frameworks
after the DSOMM parser landed (d8ad0c9)       5b0a4289...   31 frameworks
after the SAMM parser landed (217ee73)        880a0bd5...   31 frameworks
```

Two parsers moved the corpus digest inside one working session and the framework count
never budged, so `require_full_corpus` passed every time. That is the whole case for the
guard, observed rather than argued.

Live behaviour on this checkout, with `before.json` byte-identical afterwards:

```
$ PYTHONPATH=. "$PY" scripts/corpus_report.py --tag before
ValueError: refusing to overwrite results/corpus/before.json: it was built from a
different corpus.
  recorded in the existing artifact  2440d7c062055f66...
  this run                           880a0bd5b6435188...
Both hold 31 frameworks, which is why require_full_corpus passes and cannot catch this.
[...] If you mean to re-baseline, say so with --replace-baseline.

$ ... --tag before --replace-baseline --corpus data/processed/all_controls.json
ValueError: refusing to write tagged evidence from a corpus of 29 frameworks against 31
in the full set.
```

The second command is requirement 3: the override does not buy a way past the census
guard.

## Mutation audit — 13 mutations, 13 killed, and one found a defect in my own test

| # | Plausible wrong implementation | Killed by |
|---|---|---|
| R1 | Guard checks nothing | 7 tests |
| R2 | Guard blocks an identical re-run | `test_the_same_corpus_reproduces_the_tag_without_an_override` |
| R3 | Guard refuses a tag that does not exist yet | first-capture test |
| R4 | Compare framework COUNT instead of the digest | reproduction test + live-baseline test |
| R5 | Refusal omits the recorded digest | `test_the_refusal_names_both_digests_and_the_file` |
| R6 | Refusal omits this run's digest | same |
| R7 | Missing `corpus_sha256` read as permission | `test_an_artifact_with_no_recorded_digest_is_refused` |
| R8 | Unparseable artifact read as permission | `test_an_unparseable_artifact_is_refused` |
| R9 | `--replace-baseline` also bypasses `require_full_corpus` | `test_the_override_does_not_bypass_the_census_guard` (after repair) |
| R10 | Guard runs after the write | `test_the_override_reaches_the_write_and_the_guard_does_not_run` |
| R11 | Override on by default | flag-default test + override test |
| R12 | Help text drops what it destroys | `test_the_help_text_says_what_it_destroys` |
| R13 | Guard applied to the scratch `--out` path | `test_an_out_write_stays_unguarded` |

**R9 initially SURVIVED, and the reason is a trap worth recording.**
`require_full_corpus` and `require_portable_paths` both open their message with
`"refusing to write tagged evidence"`. My test matched on that shared prefix. Under the
mutation the census guard was skipped, `require_portable_paths` fired instead on the
tmp_path corpus, the message still matched, and the test passed while asserting nothing
about the guard it was named for. Repaired to match `"from a corpus of 1 frameworks
against"` — text unique to the census guard — plus an explicit assertion that the
portable-paths message is NOT what fired. R9 then died. A shared error-message prefix
across two guards is a live hazard for any future test that matches on it.

## Two harness defects found, both worth carrying forward

**Timeout-killed mutation runs leave the tree dirty.** My first R12 harness run was
SIGTERM'd at the 2-minute tool timeout. Python does not run `finally` on SIGTERM, so the
mutation was left applied in the working tree, and the next run measured every later
verdict against a mutant. I caught it because R9's failure list contained five tests it
had no business touching and ran faster than its neighbours. Rewritten to restore from a
pristine on-disk snapshot before *and* after every mutation, and to run in the background
so no timeout can kill it mid-write. Every verdict above is from the hardened harness on a
verified-clean tree, and the tree was diffed against the snapshot afterwards.

**Stale `__pycache__` (carried from the earlier section).** Same-length edits restored
inside the one-second mtime window make Python reuse mutated bytecode. Set
`PYTHONDONTWRITEBYTECODE=1` and clear `__pycache__` from the start.

## Correction to the test count I previously reported

I reported 1,546 passing after Part B. The correct pre-R12 figure on this branch is
**1,577**, reproduced four times with and without bytecode caching. The gap is not a
measurement artifact: the SAMM parser (`217ee73`, Task 4) landed on `semantic-rebuild`
while I was working and brought `tests/test_parse_samm.py` with it. My commits sit on top
of it and my Part A and Part B commits are intact in history. The R12 delta is exactly
+13, the number of tests added. The failure SET is unchanged across every measurement.

## Baselines

`results/corpus/before.json` and `results/corpus/link_resolution_before.jsonl` are
untouched. `git status --porcelain results/corpus/` is empty, and `before.json` is
byte-identical after both live refusal attempts above. Nothing was regenerated or
committed.
