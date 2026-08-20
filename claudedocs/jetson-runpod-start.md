# Jetson session start: Campaign 2 on RunPod

Written on the Mac, 2026-08-20, at branch `semantic-rebuild` / PR #62.
Read this before provisioning anything.

## The prompt

Paste this into a fresh Claude Code session on the Jetson, from the repository
root:

> Read `claudedocs/jetson-runpod-start.md` in full, then `PRD.md`,
> `results/phase1b/CAMPAIGN2.md` and `docs/RUNNING_ELSEWHERE.md`. You are
> driving a RunPod fleet from this machine to run Campaign 2. All training and
> inference happens on the pods, never on this Jetson.
>
> Work in this order and do not skip ahead:
>
> 1. Verify the environment against the "Before you provision" checklist in the
>    briefing. Report each item as pass or fail with the command output that
>    settled it. Stop if any fails.
> 2. Run `/adversarial-premortem-complete` against
>    `scripts/phase1b/runpod_parallel.py` and the Campaign 2 design. This has
>    never been done and no GPU money is spent until it is. Fix what it finds
>    at Plausible or above, then tell me what you fixed and what you parked.
> 3. Run the campaign per `results/phase1b/CAMPAIGN2.md`. Five arms on
>    validation, then the test set once with the winner. All five arms re-run.
>    Do not reuse the A1 or A2 results already in the repository.
> 4. After each `collect`, confirm `git status` shows the new fold results,
>    run the licensed-text gate, commit, and push.
>
> Rules that hold the whole way: no `git push --force`, no `git add -f`, no
> republishing to HuggingFace without asking me. Report what the numbers say
> even when they say no. If you hit an ambiguity, decide it, write down why,
> and keep going. Stop only for something irreversible, a security decision,
> or an owner decision named in the briefing.

## What this campaign is

`results/phase1b/CAMPAIGN2.md` is the pre-registration and it is binding. It
was committed before any Campaign 2 result existed, which is the point of it.
The short version:

Five arms, held fixed at batch 32, 20 epochs, LoRA rank 16, seed 42, three hard
negatives, `max_seq_length` 512.

| arm | anchor | encoder | branch balance |
|---|---|---|---|
| A1 | prose+stopwords | `BAAI/bge-large-en-v1.5` | 0.0 |
| A2 | prose+stopwords | `BAAI/bge-large-en-v1.5` | 3.0 |
| A3 | prose+stopwords | `Qwen/Qwen3-Embedding-0.6B` | 0.0 |
| A4 | prose+stopwords | `Qwen/Qwen3-Embedding-0.6B` | 3.0 |
| A5 | title-only | `BAAI/bge-large-en-v1.5` | 0.0 |

Arms are selected on the validation split (five non-AI frameworks, 1,265 eval
items). The test split (five AI frameworks, 147 items, the PRD 6.4 population)
runs **once**, with the validation winner. Campaign 1 used the same 147 items
for both, which is the mistake this design exists to avoid.

Derive the exact flags from `python -m scripts.phase1b.runpod_parallel --help`
against the table above rather than from any command written here. The flags
that matter are `--config-name`, `--split`, `--stopwords`, `--no-prose`,
`--base-model`, `--branch-balance` and `--n-configurations`.

Budget: 30 folds, roughly 3.5 hours and $90 at five parallel pods. Around $80
of the authorized $2000 is already spent.

## Three things that will go wrong quietly

These do not announce themselves. Each produces output the same shape as a
correct run.

**1. A fresh clone has no licensed prose, and trains on 4,019 links instead of
4,389.** The four overlay frameworks (ETSI, ISO 27001, CSA CCM, DSOMM) keep
their prose out of git by design, so 370 training links resolve to nothing and
the run reports normally. `docs/RUNNING_ELSEWHERE.md` covers staging them.
Before any training, run:

```bash
PYTHONPATH=. python3 -c "
from tract.training.data_quality import assert_corpus_matches_training_links
print(assert_corpus_matches_training_links())
"
```

It compares the corpus digest against the one recorded in
`hub_links_training.meta.json` and refuses when they differ. It checks the
digest rather than file existence, because both files exist on a fresh clone
and existence cannot tell a complete corpus from a partial one.

**2. Every Campaign 2 result already in the repository is stale.** The corpus
rebuild and the stopword regeneration both moved under them. `python -m
tract.staleness` names all 27. A1 and A2 have five validation folds each and
they look complete, but they were measured against a different corpus and are
not comparable to anything a new arm produces. Re-run all five arms. A stale
result may be kept and compared against its own recorded inputs. It may not be
quoted as a current measurement.

**3. The RunPod orchestrator has never had its own premortem.** A green
`_preflight_training_stack()` is not clearance for an unattended run. It checks
that the sentence-transformers pin resolves, nothing more. Everything else
about the fleet path is unaudited: teardown on failure, partial-collect
recovery, budget enforcement, what happens when one pod of five dies mid-fold.
Do the premortem first.

## Before you provision

Every item is a command with an answer, not a judgment call.

| check | how |
|---|---|
| on the right branch | `git status` shows `semantic-rebuild`, clean tree |
| corpus is complete | the `assert_corpus_matches_training_links()` snippet above returns without raising |
| stopwords present | `data/processed/stopwords.json` exists and is tracked |
| credentials load | `pass runpod/api-key`, `pass huggingface/token`, `pass wandb/api-key` each return a value |
| HF token is read-scope | it fetches the base model and nothing else. A write token on a rented host is a published-model compromise |
| SSH key registered | `~/.ssh/tract_runpod` exists and its `.pub` is on the RunPod account |
| price sanity | `python -m scripts.phase1b.runpod_parallel price` creates nothing and prints the estimate |
| suite is green | `pytest tests/ -q -m "not integration"` |
| results are stageable | `pytest tests/test_results_reachable_by_git.py -q` passes, 25 tests |

That last one guards a defect fixed on 2026-08-20 and worth understanding
before you rely on it. `results/*` in `.gitignore` also excluded
`results/phase1b`, which is where the orchestrator writes and where `collect`
rsyncs the fleet's output. Forty-five fold results were already tracked, so
`git status` stayed clean while every new result was unstageable. A fleet could
finish with its evidence on disk and nothing to push.

So: **after every `collect`, `git status` must show the new fold results.** If
it shows nothing, that is this bug returning, not a run that produced nothing.

What is stageable now and what is not:

| path | in git |
|---|---|
| `results/phase1b/**/*.json`, `*.md` | yes |
| `results/phase1b/**/checkpoint-*/`, `**/model/` | no, weights |
| `results/bridge/`, `results/phase1c/calibration/` | yes |
| `results/phase1c/similarities/`, `deployment_model/`, `crosswalk.db` | no |

Weights stay on the pod. Pull a checkpoint down deliberately if you need one
and put it somewhere outside `results/`.

## Rules that do not bend

**All training and inference on RunPod.** The Jetson drives the fleet. It does
not load a model. Writing code, running unit tests that do not allocate a
model, linting and type checking are fine locally.

**Run the licensed-text gate before every push.** Licensed text has escaped
this repository four times, each escape through a channel a previous fix did
not cover. The gate is `pytest tests/test_licensed_text_not_tracked.py`, it
carries 21,158 salted n-grams from ETSI, ISO 27001 and DSOMM, and it fails any
tracked file reproducing twelve consecutive words. Call
`tract.licensing.fingerprint_ngrams` if you need the primitive. Never
reimplement it. A hand-rolled version returned a false negative on a file the
real gate caught seconds later, because the salt is prefixed and
`normalise_for_fingerprint` does work a naive tokenizer does not.

**Never `git add -f`, never `git push --force`.** `data/raw/` and
`data/processed/licensed/` stay out of git on this machine too.

**Never republish to HuggingFace without asking.** Both the model and the
dataset repositories are live.

**Measure runtime behaviour by running it.** Four hypotheses were refuted
during the last session, every one inferred from a static artifact: stored JSON
shape read as a clobber, truncated output read as source, a wheel namelist read
as import behaviour, a default argument read as what executes. Each was refuted
by running the thing. The most recent instance is instructive: a test passed on
the Mac and failed in CI because `Path.resolve()` followed a symlink out of the
repository, silently dropping two constants from a parametrization. Twenty-two
passing tests looked exactly like twenty-five.

**Mutation-test anything you add.** It found a real defect eighteen times last
session, including tests that passed while asserting nothing. Two mutations
survived the first draft of the newest test.

## Reporting rules, fixed in advance

From the pre-registration, and not negotiable after the fact:

1. `gate_decision` is called with `n_configurations=5`. Report the
   Šidák-corrected family-wise interval alongside the nominal one and mark the
   point estimate selection-optimistic.
2. The headline is the **test-set** number for the one arm chosen on
   validation. Validation numbers are reported as validation.
3. The metric is micro-averaged hit@1 delta over the paired zero-shot baseline,
   threshold 0.10. Macro delta and worst-fold delta are diagnostics. Reporting
   macro because it is larger is metric substitution and is not permitted.
4. **If no arm clears the gate, that is the result.** `prose 0.4354` was
   already a defensible no and this campaign is allowed to produce another one.

Campaign 1 context, so a new number is not mistaken for a regression: title-only
scored 0.5306 and reproduces the published 0.531, but 79% of its lead is lexical
echo. On the 115 non-echo items title and prose tie exactly at 0.4174, McNemar
p=1.000. Nine of 147 test items appeared verbatim as a training anchor for their
own answer under title anchors, and zero do under prose.

## Owner decisions, open

Do not resolve these. Surface them and wait.

- **`csa_aicm` licensing.** 243 tracked controls under a no-redistribution
  notice, 138 of them byte-identical to a CSA CCM specification. Widening the
  fingerprint corpus to `csa_ccm` waits on this.
- **98 unopenable checkpoints** under `results/`, all adapter-only with no
  `config.json`. The fix makes the failure explicit rather than repairing them.
  Re-saving artifacts whose provenance nobody has checked is a decision.
- **PR #62 is a draft.** Converting it to ready-for-review and merging are the
  owner's call.
- **Publisher-acronym stripping** is an unmeasured ablation. The toggle defaults
  off. Whether it helps is unknown, not assumed.

## Where things are

| what | where |
|---|---|
| the spec | `PRD.md` |
| the campaign pre-registration | `results/phase1b/CAMPAIGN2.md` |
| running off the Mac | `docs/RUNNING_ELSEWHERE.md` |
| the orchestrator | `scripts/phase1b/runpod_parallel.py` |
| the last session's rulings | `.superpowers/autonomous-run/RUN-LEDGER.md` |
| staleness report | `python -m tract.staleness` |
| credentials | `pass`, never a file |
