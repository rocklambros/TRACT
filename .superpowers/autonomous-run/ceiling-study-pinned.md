# Ruling R22: the annotated ceiling study is pinned

Status: COMPLETE. No evidence file was modified.

## The problem, restated as a measurement

`build_ceiling_study()` draws from the live curated-link pool plus a seed. Task 14 moved that pool.
The pinned artifact and a fresh draw are now two different studies, and nothing said so.

Measured against the tracked corpus at `7a8465b`
(`data/processed/all_controls.json` sha256 `ebaed987`,
`data/training/hub_links_curated.jsonl` sha256 `3d42cbd3`, seed 42):

| count | value |
|---|---|
| item positions that hold | 82 of 250 |
| item positions carrying a different control | **168 of 250** |
| pinned controls absent from the fresh sample altogether | 77 of 250 |

The ledger and the module docstring both reported "82 hold, 77 replaced". Those are two different
measurements of two different things, and 82 + 77 is not 250. Every answer file in the directory
keys on `item_index`, so **168** is the number that governs whether an answer still lines up. 77 is
the smaller question: how many annotated controls a fresh draw drops, ignoring where the rest land.
Both are now recorded, named, and separated by a test.

The licensed overlay changes neither figure. All seven eligible frameworks are unlicensed, and the
measurement was taken with the overlay present and absent, with the same result.

## What changed

`tract/ceiling_study.py`

- `load_ceiling_items()` is the single entry point to the study of record. Validates the schema and
  refuses a non-contiguous `item_index`, because an answer file keyed on it would mis-join.
- `require_pinned_study_unmodified()` compares `ceiling_items.json` against the digest recorded in
  the provenance artifact, on every read through `load_ceiling_items()`. This is the live tripwire
  on "do not modify the evidence".
- `require_unmoved_ceiling_study()` refuses to overwrite the pinned artifact with a draw that moved,
  and names how far. Shaped after `require_unmoved_corpus`, including its both-directions property:
  a draw that reproduces the artifact exactly is allowed, because byte-identical regeneration is the
  check being protected.
- `require_new_study_destination()` refuses any destination inside `results/ceiling_study/`, not
  only `ceiling_items.json`. The key and six answer files sit beside it under names a fresh draw
  also wants.
- `new_study_dir(name)` puts a new study under `results/ceiling_study/studies/<name>/`. The name
  allowlist is the traversal defence.
- `ceiling_study_divergence()` reports both counts and never raises. The refusal lives in the guard.
- `load_pinned_study_provenance()` validates the record and refuses to let `recovery:
  "unrecoverable"` sit beside a digest. A provenance that is a guess is worse than an absent one.

`scripts/build_ceiling_study.py` gained three routes with one guard each: default (regenerate the
study of record, gated on reproducing it), `--study-name` (a NEW study under `studies/`), and
`--out-dir` (a scratch draw). The default route refuses today, which is the point.

`scripts/score_ceiling_study.py`, `scripts/analyze_panel_agreement.py` and `scripts/run_panel.py`
now reach the study through the validated loader or the tripwire.

## Provenance, and what it cost to recover

`results/ceiling_study/ceiling_study_provenance.json`, new and tracked.

Everything is recoverable, and it is recorded as `recovery: "reproduced"` rather than inferred,
because it was confirmed by replay rather than read off a commit:

- **seed 42** — carried in the artifact itself.
- **curated links `3d42cbd396f26cc7...`** — `data/training/hub_links_curated.jsonl`, byte-identical
  at `62afd39`, at HEAD, and on disk today.
- **corpus `ceef7fc6dc586f68...`** — `data/processed/all_controls.json` at `62afd39`.
- **code `62afd39d4a6809c9...`** — the commit that pinned the study.

The replay: the repository tree at `62afd39` was extracted to a scratch directory and
`build_ceiling_study()` run against it. It returned all 250 pinned items in their pinned positions,
250 of 250. The replay was possible only because `merged_corpus_path()` and its gitignored licensed
overlay did not exist at that commit, so `ProseIndex.load()` read the tracked corpus directly and
there is no unrecoverable input to guess at. Had the study been drawn a week later, the corpus
digest would have been unrecoverable and the record would say so.

The measured divergence is recorded beside the digests it was measured against, not beside a date.
The pool keeps moving, and a count without its inputs goes stale silently.

## Tests

79 tests in `tests/test_ceiling_study.py`, up from 25. Full suite 2,161 passing against a baseline
of 2,107, with the same 9 environmental model-loading failures and no new ones. CI-simulated
(`-m "not integration"`, `data/raw` tests deselected) 1,604 passing against a baseline of 1,550,
same 9 failures. `mypy --strict` and `ruff` clean on the CI scope.

The one test against live data is a biconditional, not a pinned number: the guard must refuse
exactly when the draw moved. Pinning today's 168 in a test would break when Task 15 lands and would
say nothing about whether the guard works.

## Mutations: 26 written, 26 killed, one real defect found

Run with `PYTHONDONTWRITEBYTECODE=1` against a pristine snapshot, restored before AND after each
mutation, verified in a full run and with `data/raw` tests deselected.

| id | mutation | result |
|---|---|---|
| M1 | `_coerce_items` drops the contiguity check | killed |
| M2 | contiguity check sorts first, accepting a reordered file | killed |
| M3 | `isinstance(index, int)` without the bool exclusion | killed |
| M4 | string fields coerced instead of checked | killed |
| M5 | `load_ceiling_items` never calls the tripwire | killed |
| M6 | tripwire always returns | killed |
| M7 | unmoved guard returns before comparing | killed |
| M8 | unmoved guard drops the length clause | **survived, then killed** |
| M9 | unmoved guard refuses even an identical redraw | killed |
| M10 | absence counted by `control_id` instead of anchor text | killed |
| M11 | `positions_held` counted by set membership, not by index | killed |
| M12 | new-study guard drops the sibling check | killed |
| M13 | new-study guard refuses every destination | killed |
| M14 | study-name alphabet widened to dots, slashes and capitals | killed |
| M15 | name length check off by one | killed |
| M16 | provenance allows `unrecoverable` beside a digest | killed |
| M17 | `unrecoverable` removed from the allowed states | killed |
| M18 | provenance records 77 as `positions_replaced` | killed |
| M19 | recorded `pinned_artifact.sha256` off by one byte | killed |
| M20 | recorded seed 43 | killed |
| M21 | recorded path points at the answer key | killed |
| M22 | provenance names no answer file | killed |
| M23 | scorer reverts to a raw `load_json` | killed |
| M24 | panel analysis reverts to a raw `load_json` | killed |
| M25 | `run_panel` skips the tripwire | killed |
| M26 | `describe()` hides a size change | killed |

**M8 exposed a real defect.** `require_unmoved_ceiling_study` refuses when
`positions_replaced != 0` OR the lengths differ. My first test for the length clause used a fresh
draw SHORTER than the pinned one, which a length-blind guard still catches, because a dropped tail
shows up as a replaced position. The case only the length clause catches is a fresh draw LONGER than
the pinned one with the same prefix: no position moves, no anchor is dropped, both divergence counts
read zero, and a guard looking only at `positions_replaced` would let a 251-item draw replace the
250-item study. `test_a_longer_draw_is_refused_and_says_why` now covers it, and `describe()` leads
with the size when it differs, because a message opening "0 of 250 positions now carry a different
control" reads as a match. M26 was added to cover that message, and it dies.

The evidence files were never mutated. `ceiling_items.json` and the answer files are read-only in
this work, so the items-side of the tripwire is covered by a real test that writes and edits a
study in `tmp_path` rather than by a mutation of tracked evidence.

## Evidence integrity

Digests unchanged across the whole task:

```
2a83b6f7...  ceiling_items.json
d9aa5a7c...  ceiling_answer_key.json
03c93b27...  answers_human_rock.json
62574319... 2d6b803f... 5f79914b... c6d4a1ea... 7a035a05...  answers_panel_*.json
```

`git status results/ceiling_study/` reports one change: the new untracked provenance file.

## Concerns

1. **The divergence will move again.** Task 15 rebuilds the corpus. The recorded 168/82/77 is
   identified by the corpus digest it was measured against, so it will read as historical rather
   than current, but nothing recomputes it. A follow-up task that re-measures should update
   `divergence_when_pinned` and its digests together, never one without the other.
2. **The alpha-1 = 0.181 CAPEC figure is safe but not regenerable.** Every anchor the owner scored
   still exists in the pool, which an existing test asserts, so the number can be recomputed against
   the pinned text. What cannot be done is regenerating the sample from code. Any new study is not
   comparable to the scored 250 without saying so, and the build script now says so in its output.
3. **A new study needs its own provenance record, and nothing enforces that yet.** The build script
   prints the requirement. Wiring `--study-name` to emit a provenance stub would close it, and was
   left out as scope.
4. **`run_panel.py` serves two item files.** `contamination_control_items.json` uses negative
   `item_index` by design, so it cannot go through `load_ceiling_items`. The tripwire there is
   conditional on the path being the pinned artifact. The condition is tested (M25), but it is a
   condition, and a third items file added later would need the same thought.
5. **A concurrent agent was mid-rebuild of `data/processed/` throughout.** The divergence was
   measured against the tracked corpus at HEAD extracted to a scratch directory, not against the
   working tree, so the figure is reproducible from git. The 7 new failures in
   `tests/test_rebuild_corpus.py` are that agent's untracked in-progress work and are excluded from
   every count above.
