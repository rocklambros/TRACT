# Jetson session start: Campaign 2 on RunPod

Written on the Mac, 2026-08-20. **Both pre-provisioning gates were closed on
the Mac on 2026-08-26**, so this file now describes a run that is cleared to
provision. Read it before provisioning anything.

**Start here: `git checkout campaign-2`.** It is cut from `main` at `753f614`,
which is the merge of PR #62. `semantic-rebuild` is merged and finished.

The two gates and their state:

1. **Four owner decisions answered and committed.** DONE 2026-08-26. The
   decision record at the end of "Owner decisions" has four filled rows.
   **Do not re-ask them.** D3 was answered (a), so the corpus rebuild is
   merged and Campaign 2 runs on `campaign-2`.
2. **P2 and P3 from the premortem fixed.** DONE 2026-08-26 in commit
   `90a5f15`, with tests in `tests/test_runpod_safety.py`. P2 could abandon
   four healthy folds and leave five GPUs billing. P3 could destroy a
   paid-for result permanently. Verify the tests pass on the Jetson rather
   than trusting this line.

D2 was implemented on the Mac the same day (`957d245`): all 98 checkpoints now
carry a base config and pass `assert_loadable_checkpoint`. They are still
stale against the rebuilt corpus, so loadable is not useful — do not draw a
conclusion from one. The repaired `config.json` files sit inside gitignored
`checkpoint-*/` directories, so they do not travel with a clone. **Re-run
`python -m scripts.repair_adapter_checkpoints` on the Jetson** if you need
them there.

One thing is still open, and it does not block a pod: D1's implementation, the
`csa_aicm` fingerprint question. See the note under the decision record.

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
> 1. Confirm the two pre-provisioning gates are closed rather than re-opening
>    them. The four owner decisions were answered on 2026-08-26 and the
>    decision record at the end of "Owner decisions" has four filled rows —
>    read them, do not ask me again. P2 and P3 were fixed in commit `90a5f15`;
>    run `pytest tests/test_runpod_safety.py -q` and confirm it passes here.
>    Work on `campaign-2`, already cut from `main` at `753f614`. Do not use
>    `semantic-rebuild`; it is merged and finished.
> 2. Verify the environment against the "Before you provision" checklist.
>    Report each item as pass or fail with the command output that settled it.
>    Stop if any fails.
> 3. Read "Adversarial premortem: the orchestrator". Confirm or refute P1, P5
>    and P6 on this machine by running the code, not by reading it, and apply
>    the P1 and P5 mitigations. P2, P3 and P4 are already fixed; re-verify
>    them by running their tests rather than by reading the diff. Then run
>    your own `/adversarial-premortem-complete` pass over
>    `scripts/phase1b/runpod_parallel.py`, because mine was one reviewer and
>    the skill uses six. Tell me what you fixed and what you parked.
> 4. Run the campaign per `results/phase1b/CAMPAIGN2.md`. Five arms on
>    validation, then the test set once with the winner. All five arms re-run.
>    Do not reuse the A1 or A2 results already in the repository.
> 5. After each `collect`, confirm `git status` shows the new fold results,
>    run the licensed-text gate, commit, and push.
> 6. After the test round only, run the agentic smoke test once on the winning
>    arm, against the pass condition already committed in
>    `data/eval/agentic_smoke_test.json`. It is six items. Report it in prose
>    as "n of 6". Do not turn it into a metric and do not re-select an arm on
>    it, whatever it says.
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

**3. Nothing stops a pod except a live orchestrator or a person.** A green
`_preflight_training_stack()` is not clearance for an unattended run. It checks
that the sentence-transformers pin resolves, nothing more. The next section is
the premortem, and its first finding is the one to internalise: the create
payload carries no server-side stop, and `full_pipeline` leaves the fleet
running on every failure path by design.

## Adversarial premortem: the orchestrator

Run on the Mac against `scripts/phase1b/runpod_parallel.py` at `38555da`, by
one reviewer reading the code. Treat it as a head start, not as the finished
job. The `/adversarial-premortem-complete` skill runs six perspectives across
five rounds and will find things a single pass did not. Confirm or refute each
finding below by running the code on the Jetson, because four hypotheses were
refuted during the last session and every one of them came from reading an
artifact instead of executing it.

The premortem narrative: it is six weeks from now, the campaign produced no
usable number, and the RunPod bill is four figures. Working backward, here is
how that happened.

| id | finding | impact | confidence |
|---|---|---|---|
| P1 | No server-side stop. A dead orchestrator bills until a human acts | High | Confirmed |
| P2 | A `pass` timeout at fold launch aborts the fleet and abandons in-flight folds | High | Likely |
| P3 | `collect` verifies transport, not payload | High | Plausible |
| P4 | The documented budget default is half the enforced one | Low | Confirmed |
| P5 | The extension cap permits about $970 for a $90 campaign | Medium | Confirmed |
| P6 | The deadline warns and does not stop | Medium | Confirmed |

### P1. The only thing between a crashed orchestrator and an open-ended bill is you

`create_pod` in `scripts/phase0/runpod_provision.py:242` sends no TTL, no
auto-stop and no idle timeout. `reap`'s own docstring says it plainly at
`runpod_parallel.py:1312`. `full_pipeline`'s `finally` at lines 1424 to 1437
then leaves the fleet running on **every** failure path, deliberately, because
terminating a pod whose results have not been collected destroys paid-for work.
That trade is correct. It also means a failure at minute five bills for as long
as nobody looks.

Five pods at roughly $2.70 an hour is $13.50 an hour. An eight-hour overnight
gap is about $108, which exceeds this campaign's entire $90 budget.

The Jetson makes this worse than a laptop would, because the run is unattended
by design. Mitigations, all of which you apply before provisioning:

- Run the orchestrator under `tmux` or `nohup` so a dropped SSH session to the
  Jetson does not kill it. A killed orchestrator is the exact scenario with no
  bound.
- Schedule an independent reaper. A cron or `at` job at T+8h that runs
  `python -m scripts.phase1b.runpod_parallel reap --confirm` costs nothing when
  the run finished cleanly, because `reap` finds no targets and exits. It is
  the only bound that survives the orchestrator dying.
- Make `reap --confirm` the first command of any session that resumes after an
  unexplained gap. Run it before you read logs and before you form a theory.

### P2. One `pass` timeout takes down four healthy folds — FIXED 2026-08-26

**Fixed in `90a5f15`.** Both halves, as prescribed below. `run_folds` reads the
credential once on the main thread and hands the dict to the workers, so
nothing is left to race, and the fold loop now catches. The same hoist covers
the bootstrap loop, which raced the same five `pass` calls; catching there had
only converted the race into "every pod failed to bootstrap". Guarded by
`TestOneFoldFailureDoesNotAbortTheFleet` in `tests/test_runpod_safety.py`.
The diagnosis below is kept because it explains what the tests are for.


`_run_fold_on_pod` calls `_get_pod_env()` at line 911, one line **above** the
`try` that starts at 912. `_get_pod_env` calls `_get_hf_read_token`, which
shells out to `subprocess.run(["pass", ...], timeout=10)` at line 228 and
raises `RuntimeError` on any failure. So `_run_fold_on_pod` can raise instead
of returning a status dict.

`run_folds` then calls `result = f.result()` at line 1058 with no guard. The
bootstrap loop twenty lines above it, at 1027 to 1031, does catch, and its
comment explains exactly why: "one bad pod aborted the whole fleet while the
other four kept billing." The fold loop did not get the same treatment.

The trigger: five worker threads call `pass` at the same instant when the folds
launch. The GPG agent serialises decryption, and it cannot run pinentry from a
non-tty worker thread, so an expired agent cache makes all five race the same
ten-second timeout. One raise ends the `as_completed` loop, discards the other
four futures' results, and reaches `full_pipeline`'s `finally` with
`results_are_safe` still False. Five GPUs keep billing and four folds that were
about to succeed are abandoned.

Fix both halves before provisioning:

- Hoist `_get_pod_env()` out of `_run_fold_on_pod` and call it once on the main
  thread in `run_folds`, passing the dict in. One `pass` invocation instead of
  five concurrent ones removes the race entirely.
- Wrap line 1058 the way line 1027 is wrapped, so a fold that raises becomes a
  failed fold rather than a failed fleet.

Warm the agent as well with `gpg-connect-agent 'keepalive' /bye` before the
run. That is a mitigation and not the fix, because it narrows the window
without closing it.

### P3. A collected fold and an empty directory look the same — FIXED 2026-08-26

**Fixed in `90a5f15`.** `collect` now verifies the payload: the fold record has
to exist and parse as JSON before a role counts as collected, and a role that
fails the check is logged with "do NOT tear this pod down". Guarded by
`TestCollectVerifiesThePayload` in `tests/test_runpod_safety.py`, including the
two cases that must NOT become failures: a clean fleet, and a genuine rsync
error. The diagnosis below is kept because it explains what the tests are for.


`collect` at lines 1094 to 1110 records a role as failed only when
`_rsync_from` raises. An rsync against a directory that exists and holds no
fold record exits 0 and is counted as collected.

The path to unrecoverable loss: a fold exits 0 without writing
`fold_result.json`, so `failed_folds` is empty and `uncollected` is empty.
`full_pipeline` sets `results_are_safe = True` at line 1423, `teardown()` runs,
and the pods are destroyed. `aggregate` then fails with nothing left to re-run
and the GPU hours are already spent.

Fix: after each rsync, assert that
`results/phase1b/<config>/fold_<role>/fold_result.json` exists and parses as
JSON, and append the role to `failed` when it does not. Verifying the payload
rather than the transport is the whole point of the function.

### P4. The docstring understates the budget by half — FIXED 2026-08-26

The docstring said `TRACT_RUNPOD_BUDGET_USD (default 1000)` while the code read
`"2000"`, so anyone sizing a run against the documented figure had half the
real ceiling. Corrected in `90a5f15`. **Set the variable explicitly for this
campaign regardless**: the default is a backstop, not a plan.

### P5. A cap that permits ten times the expected spend is not a cap

`MAX_DEADLINE_EXTENSIONS` defaults to 12 at line 109 and `MAX_RUN_HOURS`
defaults to 6. That is 72 hours of fleet time before the extension refusal
fires. At five pods and roughly $2.70 an hour, about $970. Campaign 2 needs
$90.

Set both for this campaign before you provision:

```bash
export TRACT_RUNPOD_BUDGET_USD=200
export TRACT_RUNPOD_MAX_ARMS=6
```

Six arms covers five validation arms plus the single test round, which is
exactly what the pre-registration calls for. Anything beyond that is a signal
to stop and think rather than a window to extend into.

### P6. The mid-run deadline reports, it does not enforce

The `_check_deadline()` call inside the fold loop at lines 1065 to 1068 is
wrapped and only logs `DEADLINE EXCEEDED`. The comment states the reasoning:
in-flight folds are already paid for, so aborting buys nothing back. That is
defensible on its own. Read together with P1 it means `MAX_RUN_HOURS` is a log
line, not a spend control, and the scheduled reaper in P1 is the thing actually
holding the wall.

### What the orchestrator already does well

Calibration matters, so here is where not to spend premortem effort. This
module has been through several hardening rounds and the scars are visible in
its comments. `teardown` terminates only this run's pods rather than every pod
on the account. `reap` falls back to matching the account's running pods by
name when the state file is missing or stale, which is the exact situation it
exists for. Bootstrap failures are isolated per pod. Folds run detached under
`setsid nohup`, so a dropped SSH session no longer kills an hour of training.
A fleet provisioned for one split refuses to run another before spending
anything. The budget check prices the wall time the timeouts actually permit
rather than the wall time the code intends. Poll errors retry instead of being
read as fold failures.

The gap is not carelessness. It is that every fix so far was written after an
incident, and the failure modes above have not had their incident yet.

## Before you provision

Every item is a command with an answer, not a judgment call.

| check | how |
|---|---|
| four decisions answered | DONE. Read the four filled rows; do not re-ask |
| P2 and P3 fixed | DONE in `90a5f15`. `pytest tests/test_runpod_safety.py -q` passes, 53 tests |
| spend bounds set | `TRACT_RUNPOD_BUDGET_USD=200` and `TRACT_RUNPOD_MAX_ARMS=6` exported |
| orchestrator survives a dropped session | launched under `tmux` or `nohup`, not a bare SSH foreground |
| independent reaper scheduled | a cron or `at` job at T+8h runs `reap --confirm` |
| GPG agent warm | `gpg-connect-agent 'keepalive' /bye` returns OK |
| on the right branch | `git status` shows `campaign-2`, cut from `main` at `753f614`, clean tree. NOT `semantic-rebuild` |
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

### The agentic smoke test, and why it is not a sixth number

`data/eval/agentic_smoke_test.json` holds six held-out items: the six OWASP
Agentic Top 10 controls the owner hand-mapped to CRE hubs, derived from a
39-row bridge CSV that fans each item out across several ATLAS techniques.
Collapsed to distinct control-to-hub pairs it is six items, not thirty-nine.

It is genuinely held out. `hub_links_curated.jsonl` carries **zero** links for
`owasp_agentic_top10`, so no item here has ever been a training anchor. The
four hubs do appear in training through 23 links from eight other frameworks,
so this asks whether agentic control text routes to hubs the model already
knows. That is a text generalisation test and not a hub generalisation test.

**It is not a metric and it may not become one.** The arithmetic, which is also
recorded inside the fixture:

- Three of the six items answer hub `220-442`, so always guessing that one hub
  scores 0.500.
- Against that baseline only 6/6 clears p<0.05 by a one-sided binomial
  (p=0.016). 5/6 gives p=0.109.
- The Wilson 95% interval at 4/6 is [0.30, 0.90], a width of 0.60. The
  campaign's own eval sets give 0.159 at 147 items and 0.055 at 1,265.
- Two arms compared on six items produce 1.5 expected discordant pairs, so
  McNemar cannot run at all.

Six items detect a catastrophe and nothing smaller. That is the entire job.

**Pre-declared pass condition.** Declared here and committed before any
Campaign 2 arm runs, which is the only thing that stops it being rewritten
around whatever result appears:

| outcome | reading |
|---|---|
| 4, 5 or 6 of 6 correct at rank 1 | pass |
| 2 or 3 correct | investigate, report as an open question |
| 0 or 1 correct, or any top-1 in a different `branch_root_id` than its answer | fail |

**On a fail, do not re-select the arm and do not discard the campaign result.**
Record the failure beside the headline number and open it as its own question.
Re-selecting on a six-item set is exactly the selection optimism the whole
split design exists to prevent, and doing it here would undo the campaign's
main methodological improvement over campaign 1.

**How to run it.** Once, on the single winning arm, after the test round has
already produced the headline. Never on all five arms, because comparing arms
on six items is the thing the numbers above rule out. Report it in prose with
the count stated as "n of 6" and never as `hit@1`. It does not enter
`gate_decision`, it does not count toward `n_configurations`, and it is not
quoted as a result.

Campaign 1 context, so a new number is not mistaken for a regression: title-only
scored 0.5306 and reproduces the published 0.531, but 79% of its lead is lexical
echo. On the 115 non-echo items title and prose tie exactly at 0.4174, McNemar
p=1.000. Nine of 147 test items appeared verbatim as a training anchor for their
own answer under title anchors, and zero do under prose.

## Owner decisions: answer these before provisioning

These have been carried across three sessions without an answer, which is
itself a decision, taken by default and by nobody. Two of them change what the
campaign does. Two do not, and they are here because carrying them costs more
than closing them.

**ANSWERED 2026-08-26 on the Mac. The record at the end of this section is
filled and committed, and this gate is closed. Do not re-ask these.** The
options are kept below because the record's answers are meaningless without
them, and because D1's answer has an open implementation question that only
makes sense against option (b) as written.

Each option below carries a recommendation. A recommendation is not an answer.

### D1. `csa_aicm` licensing. Blocks the fingerprint corpus, not the campaign

243 `csa_aicm` controls are tracked under a no-redistribution notice, and 138
of them are byte-identical to a CSA CCM specification. The licensed-text gate
excludes both `csa_aicm` and `csa_ccm`, so neither is protected by it today.
CSA CCM and CSA AICM are different frameworks and the ruling on one does not
carry to the other.

- **(a) Treat `csa_aicm` as an overlay.** Withhold its prose from git the way
  ETSI, ISO 27001, CSA CCM and DSOMM already are, keeping titles and
  identifiers. Costs those controls' prose on a fresh clone.
- **(b) Confirm redistribution is permitted, then add both to the fingerprint
  corpus and keep them tracked.** *Recommended if the CSA membership terms
  allow it.* CCM was already ruled redistributable and 138 of the 243 are the
  same bytes as CCM, so the two rulings should not diverge.
- **(c) Leave as-is.** Tracked, ungated, undeclared. This is the status quo and
  it is the option that has failed for three sessions.

Only the owner can read the CSA membership agreement. If that reading does not
happen, the answer is (a), because an unverified (b) is (c) wearing a hat.

### D2. 98 unopenable checkpoints — ANSWERED (b), DONE 2026-08-26

Repaired in `957d245` by `scripts/repair_adapter_checkpoints.py`, which copies
the backbone's `config.json` in beside each adapter rather than re-serialising
weights. Two things a reader should know before trusting it. The base-model
guard matches on the repo id the config was fetched for, not on
`_name_or_path` inside the config, because BAAI shipped `bge-large-en-v1.5`
carrying a path from their own build machine and reading that field refused 95
of the 98 on the first run. And passing `assert_loadable_checkpoint` proves the
directory is self-describing, not that sentence-transformers opens it —
proving that needs a model allocation, so it happens on a pod or not at all.


Every checkpoint under `results/` is adapter-only with no `config.json`, so
`load_fold_model` cannot open any of them. `assert_loadable_checkpoint` now
makes that failure explicit rather than silent.

- **(a) Leave them and move on.** *Recommended.* They are also stale against
  the rebuilt corpus, so nothing that matters would load them even if they
  opened. Revisit after Campaign 2 lands.
- **(b) Re-save all 98 with the backbone config.** Makes them loadable and
  rewrites 98 artifacts whose provenance nobody has audited.
- **(c) Delete them.** Reclaims the disk and destroys the provenance
  permanently. Tempting, and irreversible, which is why it is not the
  recommendation while a campaign is about to produce replacements.

### D3. PR #62 — ANSWERED (a), MERGED 2026-08-26 as `753f614`


The branch is 210-plus commits ahead of `main`, zero behind, with all eight CI
jobs green. It is still a draft.

- **(a) Mark ready and merge the corpus rebuild now, then run Campaign 2 from
  `main` on a fresh branch.** *Recommended.* The rebuild is independently
  correct and does not depend on any campaign result. Keeping it draft couples
  an infrastructure change to an experiment's outcome, and a long-lived branch
  rots a little more each day it waits.
- **(b) Keep it draft until Campaign 2 produces a number, and push campaign
  results onto `semantic-rebuild`.** One merge instead of two, at the cost of a
  branch that keeps growing and a rebuild that stays unmerged for no reason of
  its own.

The answer decides where results go. Under (a) the Jetson branches from `main`.
Under (b) it pushes to `semantic-rebuild`. Do not guess this one, because
guessing wrong means moving commits later.

### D4. Publisher-acronym stripping. Changes the arm count

Whether stripping publisher acronyms from anchors helps is unmeasured. The
toggle defaults off.

- **(a) Leave it off. Five arms exactly as pre-registered.** *Recommended.*
  `CAMPAIGN2.md` fixes K=5 and shows why: the minimum detectable effect is 11.4
  hit@1 points on the 147-item test set and 3.5 on validation. Adding a sixth
  arm after seeing campaign 1's results is the selection optimism the Šidák
  correction exists to bound, and it would make the pre-registered
  `n_configurations=5` wrong.
- **(b) Add it as a sixth arm.** Requires amending the pre-registration in a
  commit that lands before any arm runs, and calling `gate_decision` with
  `n_configurations=6`. Amending a pre-registration after seeing results is
  legitimate only when the amendment is dated and committed first.
- **(c) Run it as its own campaign later,** pre-registered on its own terms.

### Decision record

Answered by the owner on 2026-08-26 and committed. This gate is closed.

| id | decision | answer | date | note |
|---|---|---|---|---|
| D1 | `csa_aicm` licensing | (b) redistribution permitted, keep tracked | 2026-08-26 | Rests on the owner's reading of the CSA membership terms. Implementation NOT done — see the note below |
| D2 | 98 unopenable checkpoints | (b) re-save with the backbone config | 2026-08-26 | DONE in `957d245`. 98 of 98 pass `assert_loadable_checkpoint`. Repaired configs are gitignored, so nothing entered the repository |
| D3 | PR #62 merge timing | (a) merge now, branch from `main` | 2026-08-26 | DONE. Merged as `753f614` with a MERGE COMMIT, not a squash, because this file cites `90a5f15` and `957d245` by SHA. Campaign 2 runs on branch `campaign-2`, cut from `main` at `753f614` |
| D4 | publisher-acronym arm | (a) five arms as pre-registered | 2026-08-26 | No code change. `results/phase1b/CAMPAIGN2.md` already sets `n_configurations=5` |

**D1's answer does not translate into a change yet, and the Jetson must not
invent one.** Option (b) reads "add both to the fingerprint corpus and keep
them tracked", and those two halves contradict each other against how the gate
actually works. `fingerprinted_framework_ids()` returns
`OVERLAY_FRAMEWORK_IDS - FINGERPRINT_EXCLUDED_FRAMEWORK_IDS`, and the gate
fails when fingerprinted text appears in a tracked file. Measured on
2026-08-26:

```
overlay              csa_ccm, dsomm, etsi, iso_27001
csa_aicm in overlay  False   (its prose is tracked today)
csa_ccm  in overlay  True    (its prose is withheld today)
fingerprinted        dsomm, etsi, iso_27001
```

So fingerprinting `csa_aicm` while its prose stays tracked would fail the gate
on its own tracked prose, and fingerprinting `csa_ccm` reds the six tracked
AICM-derived artifacts that share CCM's bytes — which is exactly the deferral
already recorded in `tract/licensing.py`.

The coherent reading of "redistribution is permitted" is the opposite move:
`csa_ccm` comes OUT of `CONDITIONAL_FRAMEWORK_IDS` so its prose is tracked
too, `csa_aicm` stays tracked, both deferral entries in
`FINGERPRINT_EXCLUDED_FRAMEWORK_IDS` are deleted as dead rather than switched
on, and the licence declaration is updated to record the owner's reading and
its date. That is a licence-declaration change, not a gate change.

Confirm which of those two the owner meant before touching
`tract/licensing.py`. Getting it wrong in the permissive direction is the
fifth escape in a sequence of four, so the default while it is unconfirmed is
to change nothing. See `[[licensed-text-keeps-escaping]]` and
`[[licence-tier-is-publication-state]]`.

## Where things are

| what | where |
|---|---|
| the spec | `PRD.md` |
| the campaign pre-registration | `results/phase1b/CAMPAIGN2.md` |
| running off the Mac | `docs/RUNNING_ELSEWHERE.md` |
| the orchestrator | `scripts/phase1b/runpod_parallel.py` |
| the agentic smoke test | `data/eval/agentic_smoke_test.json` |
| the last session's rulings | `.superpowers/autonomous-run/RUN-LEDGER.md` |
| staleness report | `python -m tract.staleness` |
| credentials | `pass`, never a file |
