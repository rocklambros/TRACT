# Running TRACT on another machine

Written for the case this repository is built around: a workstation clones the
repo from GitHub and drives a RunPod fleet, while the licensed sources stay off
GitHub entirely.

Read this before the first training run. The failure it prevents is silent, not
loud: a fresh clone trains on **4,019** of the 4,389 links and produces output
the same shape as a complete run.

## What a fresh clone has

Everything the pipeline needs except the licensed prose:

| artifact | tracked | note |
|---|---|---|
| `data/training/hub_links_curated.jsonl` | yes | 4,405 curated links |
| `data/training/hub_links_training.jsonl` | yes | 4,389 filtered links, identity only, no anchor text |
| `data/training/hub_links_training.meta.json` | yes | records the corpus digest the links were built from |
| `data/processed/all_controls.json` | yes | **29** frameworks, overlay prose withheld |
| `data/processed/frameworks/*.json` | 28 of 32 | four are gitignored, see below |
| `data/processed/stopwords.json` | yes | derived from the corpus, versioned on purpose |
| `data/processed/hub_descriptions.json` | yes | |
| `results/corpus/*.json` | yes | the join evidence |
| `data/raw/` | **no** | every publisher's source bytes |
| `data/processed/licensed/all_controls.json` | **no** | the 31-framework overlay |

## Why four frameworks are missing, and what it costs

`tract.config.OVERLAY_FRAMEWORK_IDS` holds four:

| framework | terms | training links |
|---|---|---|
| `etsi` | reproduction only by written permission | 36 |
| `iso_27001` | single-user store licence, no reproduction | 92 |
| `csa_ccm` | all rights reserved, no redistribution | 29 |
| `dsomm` | GPL-3.0-only | 213 |
| | | **370** |

Their prose is written to `data/processed/licensed/` and never enters git.
`parsers/merge_all_controls.py` withholds it from the tracked corpus while
keeping titles and identifiers, so the tracked corpus indexes **4,135** controls
against the overlay's **4,667**.

So a clone without the overlay resolves 370 fewer training links: **8.4% of the
training set**, weighted toward DSOMM, which is the plan's single largest anchor
gain.

## The guard

`tract.training.data_quality.assert_corpus_matches_training_links()` compares
the digest of the corpus this run reads against the one recorded in
`hub_links_training.meta.json`, and refuses when they differ, naming both.

It checks the DIGEST rather than file existence, because both files exist on a
fresh clone and existence cannot tell a complete corpus from a partial one.

Call it before training. It is cheap and it is the difference between a run that
is 8.4% short and a run that says so.

## Staging the licensed sources

Two options. Neither goes through GitHub.

**Option A, transfer the raw sources and re-parse.** Copy `data/raw/` from a
machine that has it, then:

```bash
PY=/path/to/python3
for f in parsers/parse_*.py; do PYTHONPATH=. "$PY" "$f"; done
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" -c "
from tract.training.data_quality import assert_corpus_matches_training_links
print(assert_corpus_matches_training_links())
"
```

This reproduces the overlay from pinned bytes and is the option that keeps the
digest verifiable end to end. `data/raw/` is immutable; parsers read it and never
write it.

**Option B, transfer the overlay directly.** Copy
`data/processed/licensed/all_controls.json` and the four gitignored
`data/processed/frameworks/*.json`. Faster, and it skips the parse, so the
digest is only as trustworthy as the transfer.

Under either option, `data/raw/` and `data/processed/licensed/` must stay out of
git on the receiving machine too. Both are already in `.gitignore`; do not force
them in. A tree-wide fingerprint gate carries 21,158 n-grams from ETSI, ISO 27001
and DSOMM and fails any tracked file reproducing twelve consecutive words.

**Option C, accept the shortfall.** Train on 4,019 links deliberately. Legitimate
for a smoke test, and it is not comparable to any figure measured on 4,389. Say
so wherever the result is quoted, and record the corpus digest with it.

## Credentials

Not in the repository, by policy. `pass` holds them:
`pass runpod/api-key`, `pass huggingface/token`, `pass wandb/api-key`. Export
them into the environment on the driving machine; the pods receive only what the
provisioning code sends.

## Before spending on GPUs

`scripts/phase1b/runpod_parallel.py` runs `_preflight_training_stack()` as the
first statement of `provision()`. It reads the pin from `requirements-train.txt`,
which is what the pods install, and refuses a sentence-transformers version
absent from `TESTED_VERSIONS`. That check exists because the training modules
import three submodules the library reorganised in 5.7.0, and a failure there
would otherwise land after the fleet is already billing.

The RunPod orchestrator has NOT been through its own premortem yet. Do not treat
a green preflight as clearance for an unattended run.

## Getting a run's output back into the repository

`python -m scripts.phase1b.runpod_parallel collect` rsyncs each pod's output
into `results/phase1b/<config_name>/`. Until 2026-08-20 every file it wrote
there was gitignored: `results/*` excluded the directory, and the forty-five
fold results already tracked kept `git status` looking clean, so a fleet could
finish with its evidence on disk and nothing to push.

What is stageable now, and what is not:

| path | in git | why |
|---|---|---|
| `results/phase1b/**/*.json`, `*.md` | yes | fold results, metrics, predictions, aggregates |
| `results/phase1b/**/checkpoint-*/` | no | optimizer state and adapter tensors |
| `results/phase1b/**/model/` | no | saved backbones, 35 MB of tokenizers alone |
| `results/bridge/` | yes | `bridge_report.json` |
| `results/phase1c/calibration/` | yes | `ece_gate.json`, a gate verdict |
| `results/phase1c/similarities/`, `deployment_model/`, `crosswalk.db` | no | arrays, weights, a database |

The rule is an allowlist, so a new binary artifact type defaults to excluded
rather than landing in git. `tests/test_results_reachable_by_git.py` derives the
list from `tract.config` and fails if a directory a run writes to stops being
stageable, or if one of the deliberate exclusions stops being excluded.

After `collect`, `git status` should show the new fold results. If it shows
nothing, that is the bug above returning, not a run that produced nothing.

Weights stay on the pod. Pull a checkpoint down deliberately if you need one,
and put it somewhere outside `results/`.

## Before quoting any recorded number

```bash
PYTHONPATH=. python3 -m tract.staleness
```

Every fold records the digests of the three files it read: the curated links,
the merged corpus and the stopword list. The report compares each against the
file today and names what moved.

At the time of writing, **27 of 32 recorded fold results are stale**, because
the corpus rebuild and the stopword regeneration both moved under them. That is
expected and correct. A stale result may be kept, and may be compared against
its own recorded inputs. It may not be quoted as a current measurement without
re-running it.

The suite asserts that every result records at least one digest, so staleness
stays detectable. It does not fail on staleness itself, because a suite that
went red for the whole of a rebuild would be silenced rather than heeded.
