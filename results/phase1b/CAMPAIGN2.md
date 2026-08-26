# Campaign 2 — pre-registration

Written **before** provisioning, committed **before** any result exists. The
first campaign's design lived in conversation and an untracked shell script,
which is why nobody could say afterwards how many configurations had been
tried. PRD.md:377 pre-registers the gate *metric*; it does not pre-register an
arm count or a stopping rule, and this file supplies both.

Date: 2026-08-15. Branch `lofo-rederivation`.

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
| non-AI validation | 1,614 | **3.5 points** |

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

**Arms (K = 5).** Held fixed across arms: batch 32, 20 epochs, LoRA rank 16,
seed 42, 3 hard negatives, `max_seq_length` 512.

| # | anchor | encoder | branch balance |
|---|---|---|---|
| A1 | prose+stopwords | BGE-large | 0.0 |
| A2 | prose+stopwords | BGE-large | 3.0 |
| A3 | prose+stopwords | Qwen3-Embedding-0.6B | 0.0 |
| A4 | prose+stopwords | Qwen3-Embedding-0.6B | 3.0 |
| A5 | title-only | BGE-large | 0.0 |

A1 is the primary. A5 is the control that reproduces campaign 1's winning
condition on the new split. A2/A4 test whether branch rebalancing repairs the
ATLAS regression. A3/A4 test whether a stronger encoder helps at all — the
question that had never been asked, as distinct from whether *longer context*
helps, which is measured and does not.

**Not run, and why.** Qwen3-Embedding-4B is refused by the memory pre-flight
at batch 32 (3.75x BGE's activation cost per token-slot); running it at a
smaller batch changes the in-batch negatives MultipleNegativesRankingLoss
draws on, so it would not be comparable to the other arms. Long-context arms
(8192+) are omitted: MITRE ATLAS has zero items over 512 tokens, and on the
clean-121 subset eliminating truncation measures **negative**.

## Reporting rules, fixed in advance

1. `gate_decision` is called with `n_configurations=5`. The Šidák-corrected
   family-wise interval is reported alongside the nominal one, and the point
   estimate is marked selection-optimistic.
2. The reported headline is the **test-set** number for the single arm chosen
   on validation. Validation numbers are reported as validation.
3. The pre-registered metric is unchanged: micro-averaged hit@1 delta over the
   paired zero-shot baseline, threshold 0.10. Macro delta and worst-fold delta
   remain diagnostics. Reporting macro because it is larger would be metric
   substitution and is not permitted.
4. If no arm clears the gate, that is the result. `prose 0.4354` was already a
   defensible "no" and this campaign is allowed to produce another one.

## Budget

5 arms x 5 validation folds + 1 test round = 30 folds. At ~28 min/fold across
5 parallel pods and ~$2.70/pod-hour, roughly 3.5 hours and **$90**, against
$2000 authorized and ~$80 already spent.

## Known limitations, stated before the result

- No early stopping. 20 fixed epochs, last checkpoint taken; `eval_dataset` is
  never passed so `load_best_model_at_end` is inert.
- One seed per arm. Within-arm run-to-run variance is unmeasured and assumed
  zero, which it is not.
- The validation frameworks are traditional-security, not AI. An arm that wins
  there may not be the best arm for AI framework text; this buys statistical
  power at the cost of population match, and that trade is the reason the test
  set still exists.
