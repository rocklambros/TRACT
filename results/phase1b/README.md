# What is in this directory, and which of it may be quoted

Twelve result directories sit here as flat siblings and nothing in the layout
says which are current. The prefix convention (`c2r_` / `c2_` / `lofo_`) is not
self-explaining, and at least one superseded number — the withdrawn Phase 1B
headline — is still cited elsewhere as a comparator. This table is the answer to
"which of these is the result?".

Currency is decided by `inputs.all_controls_sha256` in each fold record, not by
the directory name. The corpus was rebuilt during Campaign 2; anything reading
`776be12eb542…` was measured against text that no longer exists in the tree.

| directory | folds | corpus digest | status | quotable |
|---|---|---|---|---|
| `c2r_TEST_A3_prose_sw_qwen06b` | 5 | `c69b06e14796` | **CURRENT — the reported test round** | yes, with `docs/campaign2-results.md` |
| `c2r_A3_prose_sw_qwen06b` | 5 | `c69b06e14796` | CURRENT — validation, selected arm | as validation only |
| `c2r_A1_prose_sw_bge` | 5 | `c69b06e14796` | CURRENT — validation, primary arm | as validation only |
| `c2r_A5_title_bge` | 5 | *(none recorded)* | CURRENT — validation, title-only arm | as validation only |
| `c2_A1_prose_sw_bge` | 5 | `776be12eb542` | SUPERSEDED — pre-rebuild corpus | no |
| `c2_A2_prose_sw_bge_bal3` | 5 | `776be12eb542` | SUPERSEDED — arm dropped by Amendment 1 | no |
| `c2_canary_qwen` | 1 | `776be12eb542` | SUPERSEDED — single-fold canary | no |
| `lofo_prose` | 5 | `776be12eb542` | SUPERSEDED — campaign 1 | no |
| `lofo_prose_stopwords` | 5 | `776be12eb542` | SUPERSEDED — campaign 1 | no |
| `lofo_prose_desconly` | 5 | `776be12eb542` | SUPERSEDED — campaign 1 | no |
| `lofo_canary_prose` | 1 | `776be12eb542` | SUPERSEDED — single-fold canary | no |
| `lofo_title_only` | 5 | *(none recorded)* | **SUPERSEDED — this is the WITHDRAWN headline** | no |

## Two directories record no corpus digest, and that is not the same as fresh

`c2r_A5_title_bge` and `lofo_title_only` are title-only arms. They run with
`--no-prose`, so they never open the merged corpus and correctly record
`all_controls_sha256: null`.

The consequence is easy to misread. `tract.staleness` marks a field stale only
when the recorded digest differs from the file on disk, so a `null` cannot
differ and these two are **structurally unable to be flagged**. `lofo_title_only`
therefore reports as not-stale while its same-era sibling `lofo_prose`, measured
against the same superseded corpus on the same day, reports as stale. The
difference is which flags each run happened to use, not which is current.

`lofo_title_only` is the withdrawn Phase 1B headline (micro delta +0.1293
[0.0408, 0.2177], n=147, `point_estimate_pass` true and `ci_low_pass` false).
`PRD.md` §6.4 records the withdrawal. It is quoted in `CAMPAIGN2.md` as
campaign-1 context, which is legitimate; it is not a current result.

## What is missing from `aggregate_metrics.json`

The file that carries each arm's headline records no `inputs` block and no
`git_sha` — those live only in the per-fold `fold_result.json`. So
`tract.staleness`, which globs `**/fold_result.json`, does not scan the one file
a reader is most likely to open and quote. Checking currency means opening a
fold record, or reading this table.

## Nothing here is deleted or moved

`PREINPUTS-ARCHIVE.md` records the deliberate decision to keep superseded runs
tracked: they are the evidence for what was measured when, and several are cited
by path from `CAMPAIGN2.md`. Marking them is the fix; removing them would break
those citations and destroy the record. That archive covers six of these twelve
directories and predates the `c2r_*` arms, so this table supersedes it as the
complete list.
