# Pre-`inputs` archive: six directories that `tract.staleness` cannot see

Six of the directories beside this file are an **archive, not results**. They are
tracked so a `git clean -fdx` cannot destroy the only copy, and this file exists
because tracking them costs something that has to be paid back in writing.

    ablation_a6_descriptions        5 folds    12 files
    phase1b_corrected               5 folds    15 files
    phase1b_nist_determinism        1 fold      3 files
    phase1b_primary                 5 folds    17 files
    phase1b_textaware               5 folds    16 files
    zero_shot_firewalled_baseline   5 folds    11 files
                                   26 folds    74 files, 294 KiB

Written 2026-04-29 to 2026-05-02. Committed 2026-08-26.

## Why they need a sign

They sit in this directory as siblings of `c2_A1_prose_sw_bge` and
`c2_A2_prose_sw_bge_bal3`, and in a directory listing nothing distinguishes an
April archive from a live Campaign 2 arm. The usual guard against quoting a
stale number does not cover them:

`tract.staleness` globs `**/fold_result.json`. These six predate that schema and
use the older `metrics.json` / `predictions.json` / `fold_*_summary.json` layout
with no `inputs` block, so the reporter never opens them. Its summary line reads

    32 fold results, 27 stale, 0 recording no input digest at all

and all 32 belong to the tracked `c2_*` and `lofo_*` directories. **These 26
folds are not among the 32, and their absence from the stale list is not
freshness — it is invisibility.** The reassurance "0 recording no input digest at
all" describes only the files the scanner can see.

## What they record, and what they do not

Grepped across all 74: zero `all_controls_sha256`, zero `stopwords_sha256`, zero
`curated_links_sha256`. The only provenance field present anywhere is `git_sha`,
carrying three distinct values across the set, one of which is the literal string
`"unknown"` — the orchestrator's fallback when it cannot resolve a commit.

Since they were written, the corpus moved `776be12eb542 -> 1a4a9676fe2b` and the
stopword list moved `2ac10af14bb1` and `6a0f0a0d9202 -> be5cbb35b721`. Both are
inputs to every number in these files.

So these are stale in the ordinary sense and worse off than the 27 results that
`tract.staleness` does flag. A flagged result may at least be compared against
its own recorded inputs. **These recorded none, so they cannot be compared
against anything — not against each other, not against a Campaign 2 arm, and not
against a published figure.** Read them as a record that a run happened and
roughly where it landed. Do not quote a number from them.

## What is here and what is not

Tracking covers the 74 metric and summary JSON files. It does **not** cover the
796 files and 1.08 GiB of `checkpoint-*/` and `model/` artifacts alongside them —
96 `.pt`, 53 `.safetensors`, 32 `.pth` and the rest — which `.gitignore` excludes
by design so that a new binary artifact type defaults to staying out of git.
Those remain on this machine only and a `git clean -fdx` still removes them. They
are also stale against the rebuilt corpus, and `load_fold_model` could not open
any of them until the repair in `957d245`, whose output is itself gitignored.

If the weights matter, copy them off this machine. Git is not protecting them.

## Telling an archive directory from a new fold result

After a `collect`, the new evidence appears as individual file paths ending in
`fold_result.json` inside a tracked `c2_*` directory. Nothing in this archive
matches that shape — these six contain zero `fold_result.json` between them.

    git status --porcelain -uall -- 'results/phase1b/c2_*'
    find results/phase1b -name fold_result.json | wc -l
