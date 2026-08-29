# Campaign 2 — pre-registration

Written **before** provisioning, committed **before** any result exists. The
first campaign's design lived in conversation and an untracked shell script,
which is why nobody could say afterwards how many configurations had been
tried. PRD.md:377 pre-registers the gate *metric*; it does not pre-register an
arm count or a stopping rule, and this file supplies both.

Date: 2026-08-15. Branch `lofo-rederivation`.

**Amended 2026-08-27, before any Campaign 2 arm ran.** Arms A2 and A4 dropped,
`n_configurations` 5 -> 3, the selection statistic named, config names bound,
the budget ceiling corrected, the governing metric settled, and the
licensed-overlay transfer to rented hosts recorded as an owner decision. The
amendments are set out in full at the end of this file. Where one reverses
something this document previously said, the old text is quoted rather than
deleted — a pre-registration that edits itself silently is worth nothing.

## What campaign 1 established

Four anchor arms, 5 AI folds, 147 eval items:

```
title-only        0.5306 [0.456, 0.605]   delta +0.1293   PASS/FAIL
prose+stopwords   0.4490 [0.374, 0.524]   delta +0.0748   FAIL/FAIL
prose+desconly    0.4422 [0.367, 0.517]   delta +0.0884   FAIL/FAIL
prose             0.4354 [0.361, 0.510]   delta +0.0816   FAIL/FAIL
```

Follow-up measurement, reproduced independently twice:

- 79% of title-only's lead is **lexical echo** — 26 of 147 eval items have a
  title character-identical to their own hub name, where title-only scores
  0.923 and prose 0.654. On the 115 non-echo items the two arms tie exactly
  (0.4174 each, McNemar p=1.000). Excluding MITRE ATLAS techniques as well,
  prose+stopwords leads 0.4653 to 0.4455.
- The title arm additionally leaks: 9 of 147 eval items appear verbatim as a
  training anchor for their own answer. Under prose anchors that is 0 of 147.
- The one genuine prose deficit is 17 MITRE ATLAS technique items, where the
  prose-trained model scores *below* its own zero-shot baseline. Cause is a
  training imbalance, not the anchor: 72.1% of links point at "Technical
  application security controls" and 3.3% at the threat branch, and none of
  CAPEC's 702 adversary-as-subject anchors point at a threat hub.

## Why this campaign is not 16 configurations

The eval set cannot support it. Measured on the committed per-item
indicators, mean inter-arm discordance 0.244:

| stratum | n | minimum detectable effect |
|---|---|---|
| AI test set | 147 | **11.4 hit@1 points** |
| non-AI validation | 1,265 | **4.0 points** |

**Corrected 2026-08-27, before any Campaign 2 arm ran.** This row said
**1,614** items and **3.5 points**. The non-AI validation stratum is **1,265**
items — the figure the Design section below, `runpod_parallel.py:159` and the
`--split` CLI help all carry, and the sum of its five folds
(277 + 349 + 246 + 93 + 300). The MDE was never recomputed against it. Minimum
detectable effect scales as 1/sqrt(n), so the true value is
3.5 x sqrt(1614/1265) = **3.95 points**, stated here as **4.0**. Rounding UP is
the conservative direction: it makes displacing the primary arm harder, not
easier. The tie-break below depends on this number, and at 3.5 the bar was
about 13% too low — a challenger arm could have displaced A1 on a lead that is
inside the noise floor.

Detecting a 3-point difference on the test set needs 2,126 items. Five of the
six pairwise comparisons among the campaign-1 arms already include zero.
Simulated at this eval size, 16 configurations under a null where every one
has a true delta of 0.08 produce at least one point estimate above the 0.10
gate about 73% of the time.

So: **arms are selected on validation, and the test set runs once.**

## Design

**Split.** Selection and reporting are separated, because campaign 1 used the
same 147 items for both.

- *Validation* — LOFO over the 5 largest non-AI frameworks (CAPEC, NIST
  800-53, ASVS, CWE, ISO 27001), 1,265 eval items. Fits one 5-pod fleet per
  arm. Every arm runs here.
- *Test* — LOFO over the 5 AI frameworks, 147 items, the pre-registered PRD
  6.4 population. Runs **once**, with the validation winner.

**Arms (K = 3).** Held fixed across arms: batch 32, 20 epochs, LoRA rank 16,
seed 42, 3 hard negatives, `max_seq_length` 512.

| # | anchor | encoder | branch balance |
|---|---|---|---|
| A1 | prose+stopwords | `BAAI/bge-large-en-v1.5` | 0.0 |
| A3 | prose+stopwords | `Qwen/Qwen3-Embedding-0.6B` | 0.0 |
| A5 | title-only | `BAAI/bge-large-en-v1.5` | 0.0 |

A1 is the primary. A5 is the control that reproduces campaign 1's winning
condition on the new split. A3 tests whether a stronger encoder helps at all —
the question that had never been asked, as distinct from whether *longer
context* helps, which is measured and does not.

The numbering is deliberately not resequenced. A2 and A4 were the
branch-balance arms, dropped 2026-08-27 for the reason recorded under
*Amendment 1*. Keeping the gaps means a reader who finds
`c2_A2_prose_sw_bge_bal3` sitting in this directory can tell what it was and
why nothing here refers to it.

**Config names, bound here as part of the pre-registration.** Fold records
carry no arm label, so the results directory is the only thing separating one
arm from another and a mixed directory aggregates into a number describing no
single configuration.

| # | `--config-name` | arm flags |
|---|---|---|
| A1 | `c2r_A1_prose_sw_bge` | `--stopwords` |
| A3 | `c2r_A3_prose_sw_qwen06b` | `--stopwords --base-model Qwen/Qwen3-Embedding-0.6B` |
| A5 | `c2r_A5_title_bge` | `--no-prose` |

`--branch-balance` is passed by no arm; 0.0 is off. Validation runs add
`--split validation`, the single test round `--split test`.

**The names are fresh on purpose, and that is not tidiness.**
`results/phase1b/` already holds `c2_A1_prose_sw_bge` and
`c2_A2_prose_sw_bge_bal3` from a prior run that is stale against the rebuilt
corpus. `collect` rsyncs each pod's output into `RESULTS_DIR / config_name`
with `rsync -rltz --safe-links --partial` and **no `--delete`**, so re-using
either name merges fresh folds on top of stale ones and the aggregate then
describes a mixture of two corpora. Every other way of getting this wrong
raises; this one returns a plausible number.

**Not run, and why.** Qwen3-Embedding-4B is refused by the memory pre-flight
at batch 32 (3.75x BGE's activation cost per token-slot); running it at a
smaller batch changes the in-batch negatives MultipleNegativesRankingLoss
draws on, so it would not be comparable to the other arms. Long-context arms
(8192+) are omitted: MITRE ATLAS has zero items over 512 tokens, and on the
clean-121 subset eliminating truncation measures **negative**.

## How the winning arm is chosen

Added 2026-08-27. Until today no document said this, which meant the campaign
could run to completion and the choice of winner would be made after the
numbers were visible.

**Arms are ranked on absolute micro-averaged validation hit@1. Not on delta.**

Delta is measured against a *per-arm* paired zero-shot baseline, and that
baseline moves with both the anchor and the encoder: A3's Qwen baseline is not
A1's BGE baseline, and A5's title-only baseline is neither. A delta therefore
states how far an arm travelled from its own starting point, and ranking on it
rewards whichever arm started worst. Absolute hit@1 is the only quantity on
this eval that means the same thing across all three arms.

The two statistics do different jobs and this document does not merge them:

- **Selection** — absolute micro-averaged validation hit@1. Decides *which arm
  advances* to the test round.
- **Gate** — micro-averaged hit@1 delta over that arm's own paired zero-shot
  baseline, threshold 0.10, PRD 6.4. Decides *whether the arm that advanced is
  better than its own baseline*.

An arm can win selection and fail the gate. That is a coherent outcome, not a
contradiction, and it is reported as one. Conflating the two statistics is the
defect this amendment closes.

**Tie-break.** A1 is the primary, and **A1 advances unless another arm beats it
by more than the minimum detectable effect on the validation split** — 4.0
hit@1 points, from the table above. A lead narrower than the MDE is not a lead
this eval can resolve, and breaking such a tie toward the pre-registered
primary is a rule fixed while the numbers do not yet exist rather than a
preference discovered once they do. If two non-primary arms both clear A1 by
more than the MDE, the higher absolute validation hit@1 advances.

## Reporting rules, fixed in advance

1. `gate_decision` is called with **`n_configurations=3`** on the **validation
   aggregates**, because validation is where selection happens. The
   Šidák-corrected family-wise interval is reported alongside the nominal one,
   and the point estimate is marked selection-optimistic.
2. **Dropping two arms does not weaken the gate.** Šidák sets
   `alpha_effective = 1 - 0.95^(1/n)`: 0.0102 at n=5, 0.0170 at n=3. The
   corrected interval is narrower at n=3, which reads like a concession and is
   not one — the correction exists to price the raffle, and two fewer arms is
   a genuinely smaller raffle. The quantity being controlled is the chance that
   *some* arm clears 0.10 on noise alone, and it falls with the arm count.
   Reporting the n=5 interval for a three-arm campaign would not be
   conservative, it would be wrong. The failure to guard against is the
   reverse: running more than three arms and still reporting n=3.
3. The **single uncontaminated test round is reported with
   `n_configurations=1`**. One arm reaches it, chosen on a disjoint split, so
   at that point there is no selection left to correct for. The correction
   belongs where the selection happened and nowhere else.
4. The reported headline is the **test-set** number for the single arm chosen
   on validation. Validation numbers are reported as validation.
5. The pre-registered metric is unchanged: micro-averaged hit@1 delta over the
   paired zero-shot baseline, threshold 0.10. Macro delta and worst-fold delta
   remain diagnostics. Reporting macro because it is larger would be metric
   substitution and is not permitted.
6. **The pre-registered verdict is `point_estimate_pass`** — PRD 6.4 as
   written. `ci_low_pass` and `familywise_pass` are **diagnostics**. They are
   reported every time and substituted for the verdict in neither direction.
   Their thresholds are recorded here because they are reachable only in
   principle: on the 147-item test split `ci_low_pass` needs a micro delta
   above **0.188** and `familywise_pass` above **0.216**, against a best-ever
   measured delta of **+0.129** (title-only, campaign 1). Shifting that arm's
   committed per-item indicators upward, the two first turn true at +0.1905
   and +0.2177. Neither is within reach of any result this campaign can
   plausibly produce. Stating it in advance is what stops a failed diagnostic
   being presented afterwards as a failed gate — or a passing one being
   presented as a stronger result than the pre-registration ever asked for.
7. If no arm clears the gate, that is the result. `prose 0.4354` was already a
   defensible "no" and this campaign is allowed to produce another one.

## Budget

3 arms x 5 validation folds + 1 test round of 5 folds = 20 folds. At ~28
min/fold across 5 parallel pods and ~$3.29/pod-hour, roughly 2.5 hours and
**$60**, against **$1000 authorized** and ~$80 already spent.

**Corrected 2026-08-27.** This line read "$2000 authorized" until today. The
owner lowered the ceiling to $1000 on 2026-08-26 and the figure here was left
stale. The campaign fits comfortably under either number, so nothing about the
design changes — but a budget line nobody reconciles is exactly how the wrong
ceiling gets quoted at the moment someone is deciding whether a re-run is
affordable. The fold count also fell from 30 to 20 with A2 and A4.

## Known limitations, stated before the result

- No early stopping. 20 fixed epochs, last checkpoint taken; `eval_dataset` is
  never passed so `load_best_model_at_end` is inert.
- One seed per arm. Within-arm run-to-run variance is unmeasured and assumed
  zero, which it is not.
- The validation frameworks are traditional-security, not AI. An arm that wins
  there may not be the best arm for AI framework text; this buys statistical
  power at the cost of population match, and that trade is the reason the test
  set still exists.

## Amendments, 2026-08-27

All made **before any Campaign 2 arm ran**. Each records what this document
said, what it says now, and why.

### Amendment 1 — arms A2 and A4 dropped

**Was:** "**Arms (K = 5).**", with A2 and A4 as prose+stopwords at branch
balance 3.0, and "A2/A4 test whether branch rebalancing repairs the ATLAS
regression."

**Now:** three arms — A1, A3, A5. The branch-balance question is **deferred to
its own campaign**, to be pre-registered separately on a split that actually
contains threat-branch items.

Two independent reasons, either one sufficient.

**1. This split cannot see the arm.** Branch balancing manipulates the
Cross-cutting concerns branch. Measured on the committed curated links:

| split | curated links | Cross-cutting | share |
|---|---|---|---|
| test (5 AI frameworks) | 197 | 83 | **42.13%** |
| validation (CAPEC, NIST 800-53, ASVS, CWE, ISO 27001) | 3,083 | 10 | **0.32%** |

Three of the five validation folds — ASVS, CAPEC and CWE — contain **zero**
Cross-cutting items. Arms are selected on validation. A2 and A4 would therefore
have been selected on a split that is 0.32% the thing they manipulate, while
the one split where the manipulation matters is the split that runs once, after
selection is already over. An arm whose effect is invisible to the statistic
that picks it is not an experiment.

**2. The arm does not do what this document said it did.** A2/A4 were described
above as testing "whether branch rebalancing repairs the ATLAS regression",
which reads as a change to what the model trains on. It is not. The
`_order_by_strata` docstring at `tract/training/data.py:363-366` states the
mechanism:

> p^(1/T): T=1 leaves the natural distribution, larger T flattens
> toward uniform. Sampling is without replacement -- each example still
> appears exactly once per epoch, only the ORDER changes, so no example
> is duplicated or dropped and epoch size is unchanged.

Every example appears exactly once per epoch at every temperature. The sampler
reorders an epoch; it does not rebalance one. What `--branch-balance 3.0`
actually varies is which examples land in a batch together — hence the in-batch
negatives `MultipleNegativesRankingLoss` draws on, and the order in which
gradients arrive — and not the composition of the training set. That may well
be a real effect and it is worth measuring. It is not the effect this document
claimed to be measuring, and a pre-registration that mis-states its own
mechanism cannot adjudicate its own result.

Deferring costs two arms and buys back the Šidák budget they were spending on a
question this split cannot answer.

### Amendment 2 — which metric governs Campaign 2

`docs/superpowers/specs/2026-08-15-semantic-rebuild-design.md` Part 0.2 names
**hit@5** as its primary metric ("The operative metric is hit@5, not hit@1").
This document names **micro-averaged hit@1 delta**, threshold 0.10, per PRD
6.4. Both are committed, and they disagree.

**CAMPAIGN2.md governs Campaign 2.** It is the campaign-scoped
pre-registration, committed before any Campaign 2 result existed, and a
campaign is adjudicated by the document written for it. hit@5 remains the
rebuild spec's metric for the rebuild spec's own questions — ceiling analysis,
multi-label adequacy, whether the label set can support hit@1 at all — and
nothing here withdraws it.

This is settled in advance for one reason. Two live primary metrics plus a
result in hand is precisely the situation where whichever number looks better
becomes the one that was "always" primary. Deciding now costs nothing and
removes the argument entirely. Campaign 2 reports hit@1 delta as its verdict;
hit@5 may be reported alongside it and may not be substituted for it.

### Amendment 3 — licensed-overlay transfer to rented hosts

**Owner decision, 2026-08-27. Decided, not defaulted.** The fleet bootstrap
ships the licensed overlay to rented RunPod hosts. That was already true and
was nowhere recorded as a choice: `docs/RUNNING_ELSEWHERE.md` says the licensed
sources stay off GitHub and says nothing at all about pods. This entry closes
that gap.

**What transfers.** `_rsync_to` in `scripts/phase1b/runpod_parallel.py` sends
the working tree under an exclude list that covers `data/raw` but **not**
`data/processed/licensed`. Each pod therefore receives:

| file | size |
|---|---|
| `data/processed/licensed/all_controls.json` | 13 MB |
| `data/processed/frameworks/etsi.json` | 200 KB |
| `data/processed/frameworks/iso_27001.json` | 24 KB |
| `data/processed/frameworks/dsomm.json` | 168 KB |

All four are gitignored and untracked — they are the overlay. Five hosts per
fleet, four fleets across this campaign (three validation arms plus the single
test round), so twenty pod-instances receive it. That figure was six fleets
before A2 and A4 were dropped.

**Why it must.** `assert_corpus_matches_training_links` compares the corpus
digest against the one recorded in `hub_links_training.meta.json` and refuses
when they differ; since 2026-08-26 it runs at `provision`, at `run_folds`, and
again on the pod inside `run_fold.py`. Excluding the overlay from the rsync
does not train a smaller model — it fails every pod of every fleet at
`CorpusMismatchError` before a single step runs. The alternative that appears
to "work", regenerating the sidecar on a host without the overlay, records the
4,048-link corpus as the reference, passes the gate, and trains 7.8% short
while emitting output of exactly the same shape. That is the worse outcome, and
it is why excluding the overlay is not available as a fix.

**How this is characterised.** Transient processing on rented compute under the
operator's control, for the operator's own training run. Not redistribution: no
third party receives the text, the pods run only this operator's job, and what
leaves them is model weights and per-item indicators, not prose.

**Residual risk, stated rather than mitigated.** Nothing wipes the pod before
`teardown()` terminates it — `teardown` calls `terminate_pods` on the pod ids
this run created and touches the filesystem not at all beforehand. Destruction
of the overlay therefore relies entirely on RunPod's termination of the
underlying volume. The owner accepts that. It is written down so that accepting
it is a decision on the record rather than an assumption nobody examined.
