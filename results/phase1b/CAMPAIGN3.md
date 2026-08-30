# Campaign 3 — pre-registration

Written 2026-08-29 against `main`, **before any Campaign 3 arm has run**. Binding
on every arm that follows. Amendments must be dated, committed, and must state
what was already known when they were made — Campaign 2's authorising clause was
added eight days after its arm results existed, and that is why its headline
does not carry the weight it appears to.

Read `docs/campaign2-results.md` and `docs/campaign3-premortem.md` first. This
document assumes both.

---

## 0. What changed before this was written

Two findings from the Campaign 3 premortem, both verified against the
repository, set the terms here.

**The domain shortcut is refuted.** Handing the zero-shot encoder the whole
78-hub AI region for free — deleting 444 of 522 candidates — moves exactly one
item in 147 (+0.0068). The Campaign 2 gain is not candidate-set narrowing. The
pre-registered condition for funding curation is met.

**25% of the Campaign 2 test gold was rewritten by TRACT's own link audit**, and
the gain concentrates there: +0.2432 on the 37 touched items against +0.1000 on
the 110 untouched. The untouched figure is the honest baseline for anything
Campaign 3 compares against.

---

## 1. Synthetic data: NONE. Not in training, not in evaluation.

This is a reversal of the coordinator's earlier proposal, which would have
permitted synthetic paraphrase of Tier-1-labelled items at up to 10% of training
pairs. The probe is what reversed it.

**No synthetic control text in any evaluation set, at any ratio, ever.** Both
natural generator designs recreate a documented disaster. Conditioning on the
hub name manufactures lexical echo, which is what withdrew the Campaign 1
headline. Generating one control per hub manufactures the ASVS bijection — 277
links across 277 distinct hubs, zero-shot 0.6282, trained 0.2347, the worst fold
in Campaign 2 at **−0.3935**.

**No synthetic paraphrase in training either.** The earlier case for it was
regularisation against a corpus with a 22× anchor-length range. That case is
weaker than it looked: the probe shows the gain is genuine within-domain
discrimination, so the binding constraint is **evaluation size, not training
signal**, and paraphrase does not add an evaluation item. It would buy an
unmeasured benefit at the cost of an `is_synthetic` column, an epoch-share cap
that becomes a pre-registered parameter, and a bootstrap that no longer matches
its own data.

**If synthetic items are ever admitted, the bootstrap must resample generation
clusters, not items.** `paired_bootstrap_delta` (`tract/training/evaluate.py`)
resamples i.i.d. within folds with no cluster level. Measured ICC on real items
is 0.031 (design effect 1.06). At an *assumed* synthetic ICC of 0.20 with ~32
items per hub the design effect is ~7.2 — the code would report n=2,500 while
the effective n was ~347. That 0.20 is an assumption, not a measurement, and the
gap between it and the measured 0.031 is exactly why a 50-item pilot must
measure it before any synthetic item enters a denominator.

**Revisit condition:** a measured synthetic pilot showing ICC < 0.05 and a
demonstrated training gain on a held-out Tier-1 stratum. Absent both, the answer
stays no.

## 2. Label provenance tiers

The line is **who produced the label**, not whether text was generated.

| tier | definition | may sit in a gate denominator |
|---|---|---|
| **1** | Curated by OpenCRE independently of TRACT | **Yes** |
| **2** | Independently human-authored, blind to model output | Separate study, own n, own delta — **never** continues a Campaign 2 comparison |
| **3** | Produced by, or ratified in the presence of, a model or LLM | **No. At any ratio.** |

**`data/training/hub_links_curated.jsonl` is NOT purely Tier 1**, and the
earlier definition calling it "labels OpenCRE curated before TRACT existed" was
factually false. 56 of its AI links carry gold TRACT's own audit rewrote — Tier
2, since the audit CSV has no prediction, score or ranking column and is a human
relabel rather than model-seeded. The Tier-1 stratum is therefore the
**audit-untouched** subset, and that is what §3's primary is computed on.

**`results/review/review_export.json` is Tier 3 and quarantined.** See
`results/review/PROVENANCE.md` and `tests/test_tier3_quarantine.py`.

## 3. The gate — thresholds, committed now

Campaign 2 pre-registered a *report*. This pre-registers a *decision*.

**Primary.** Micro-averaged hit@1 delta over the paired zero-shot baseline, on
**Tier-1 (audit-untouched) items only**, fold-stratified paired bootstrap,
10,000 resamples, seed 42.

> **PASS iff `P(true delta ≤ 0.10) < 0.05`.**

That is the one-sided 95% lower bound clearing 0.10. Not the point estimate.
Campaign 2 passed on the point estimate alone with `P(δ ≤ 0.10) = 0.203`, and
that is the failure this threshold exists to prevent.

**Binding side condition.** The non-echo stratum delta must satisfy **point
estimate ≥ 0.10 and CI low > 0**. This makes "the gains are not string-matching"
a property the design *enforces* rather than one noticed afterward.

**Echo is a property of the item, fixed once.** Computed at corpus-build time
from frozen canonical text — the **union of title and full prose** — and
recorded in the eval corpus. Campaign 2 recomputed it per arm against that arm's
own anchors: 38 echo items under prose, 32 under titles, symmetric difference
28. A partition that moves with the arm being measured cannot support a binding
condition.

> ⚠️ **The paragraph below is SUPERSEDED by Amendment 1 §1.3.** "Full prose"
> here meant *truncated* full prose, so the partition still moved with the
> anchor budget. The corrected non-echo figure is **+0.1538 on n=91**, computed
> against untruncated text by `tract/training/echo.py`.

The floor is 0.10, not the +0.1743 Campaign 2 reported. That figure was computed
under the prose-only definition; under the frozen union definition the same
campaign gives **+0.1531 on n=98**. A floor calibrated on the wrong definition
is miscalibrated on arrival.

**Outcome table — every combination decided in advance.**

| primary | side condition | verdict |
|---|---|---|
| pass | pass | **PASS.** Report both. |
| pass | fail | **FAIL.** A gain that does not survive the non-echo stratum is the Campaign 1 failure mode. |
| fail | pass | **FAIL.** Report the side condition as a diagnostic, never as the result. |
| fail | fail | **FAIL.** |

No arm may be re-selected on the side condition. No metric substitution.

## 4. What this design can and cannot resolve — stated up front

> ⚠️ **This whole section is SUPERSEDED by Amendment 1 §1.1.** The n=940 row is
> unreachable: the four frameworks §6 permits hold 500 controls, so the ceiling
> is 721 and the realistic figure is 521–596. **Real power is 43–54%, not 63%.**
> The table below is retained as the record of what was decided against.

Per-item delta SD is **0.5577**, measured on the Campaign 2 test round. Power to
clear `P(δ ≤ 0.10) < 0.05`:

| n | SE | true δ = 0.136 | 0.150 | 0.175 |
|---|---|---|---|---|
| 221 (Tier-1 today) | 0.0375 | 25% | 38% | 64% |
| 500 | 0.0249 | 42% | 64% | 91% |
| **940 (full curation of unlinked frameworks)** | 0.0182 | **63%** | 87% | 99% |
| 1,484 | 0.0145 | 80% | 96% | 100% |

**Read the 940 row before authorising the curation spend.** If the true effect
is the +0.136 Campaign 2 estimated, a fully curated Campaign 3 resolves it
roughly **five times in eight**. It is adequately powered only if the true
effect is ≥ ~0.15.

**MDE at n=940 is 0.145.** A true effect of 0.136 will more likely than not
produce an ambiguous result even after every reachable item is curated. That is
not a reason to skip the campaign; it is a reason to not be surprised, and to
have decided in advance — as §3 does — what an ambiguous result means.

**Owner sign-off required before spend:** that a ~37% chance of an
undecided outcome at full curation is an acceptable return on ~25 expert-hours.

## 5. Open, and blocking

These are not engineering tasks and are not resolved here.

1. **~~`R4.1` versus comparability.~~ DECIDED 2026-08-30: fix it and rebaseline.**
   53 of 147 test anchors (36%) truncate at the 2,150-char budget. Text
   selection is being fixed, and both the paired zero-shot and the A3 recipe
   are being re-run to establish a new baseline. **Every figure in §0 and §4 of
   this document is stated against the OLD anchors and must be re-derived
   against the new ones before any Campaign 3 arm runs.** The +0.1361 and
   +0.1000 comparisons are retired as forward targets; they remain the
   historical record of what the old anchors produced.
2. **~~Audit provenance.~~ RESOLVED 2026-08-30 by the owner: no model output was
   visible to the reviewer.** The 56 corrections are **Tier 2**, not Tier 3.
   Nothing model-derived is downstream in published artifacts and the OpenCRE
   RFC may cite these links. The Tier-1/Tier-2 stratification in §2 and §3 is
   unchanged: Tier 2 is legitimate but is not OpenCRE's taxonomy, so the primary
   is still computed on the audit-untouched stratum.
3. **The forge blob.** Licensed ISO text reachable via `refs/pull/73/head`,
   removable only by GitHub Support. It gets worse with external attention, and
   the RFC submission is exactly that — so the request belongs **before** the
   RFC. **Still open; still an owner action.**

## 6. Curation, if funded

Order, with measured estimates. Blind to model output, or the result is Tier 3
and worthless for §3.

1. **Fix `csa_aicm` text selection first** (~0.5 engineer-day). Median
   `full_text` 17,115 chars against a 2,150-char budget: reviewing before fixing
   this means curating against text the model never sees.
2. `csa_aicm` — 243 controls, ~190 items, ~6 reviewer-hours.
3. `cosai` (55) + `aiuc_1` (130) + `nist_ai_rmf` (72) — ~7.5 hours combined.
4. **Do not curate** `eu_ai_act` (governance prose, only ~26 of 126 articles
   CRE-mappable) or `owasp_agentic_top10` (10 controls; linking it entrenches
   the risk→threat convention that broke its own smoke test).
5. External: **UK DSIT AI Cyber Security Code of Practice** only — 72
   provisions, Open Government Licence v3.0, published after the CRE tree so no
   bijection risk. Singapore CSA Companion Guide is second and its licence is
   **undetermined**; that is an owner decision, not an engineering one. Reject
   MITRE D3FEND (~140 bytes/row — titles, not prose) and AIC4 (its claimed
   coverage gap does not exist: zero hubs match bias/fairness).

---

## AMENDMENT 1 — 2026-08-30

**What was known when this was written: no Campaign 3 arm has run, and no
curation has been commissioned.** The only new measurements are forensics on
Campaign 2's committed artifacts and on the corpus. This is blind
pre-registration; §2 of `docs/campaign2-results.md` explains why that
distinction is the one this project keeps getting wrong.

Three things in the original are wrong. They are corrected here rather than
edited away.

### 1.1 The n=940 the power table is justified against is unreachable

§6 permits curating four frameworks. They hold **500 controls** — `csa_aicm`
243, `aiuc_1` 130, `nist_ai_rmf` 72, `cosai` 55. Even if every single control
yielded one eval item, the ceiling is **221 + 500 = 721**. Adding the two
frameworks §6.4 forbids reaches only 857. **No combination of the planned work
reaches 940.**

Corrected power, at the measured per-item delta SD of 0.5577:

| scenario | n | power at δ=0.136 | at δ=0.150 | MDE at 80% |
|---|---|---|---|---|
| today, Tier-1 only | 221 | 25% | 38% | 0.193 |
| 60% yield (300 new) | 521 | 43% | 66% | 0.161 |
| 75% yield (375 new) | 596 | 47% | 71% | 0.157 |
| 100% yield (500 new) | 721 | 54% | 78% | 0.152 |
| ~~the original plan~~ | ~~940~~ | ~~63%~~ | ~~87%~~ | ~~0.145~~ |

**The realistic figure is 43–54%, not 63%.** A curation round that costs ~25
expert-hours resolves a true effect of 0.136 slightly less often than a coin.
The owner has authorised the spend knowing the 63% figure; this correction must
reach them before the recruiting starts, and it is the single most important
line in this amendment.

That SD is itself provisional: it was measured on the OLD anchors, and the
rebaseline changes them. Re-derive it from the rebaselined test round and
restate this table before the curated items are scored.

### 1.2 Tier-2 items and the primary denominator — the rule, made explicit

§2 says Tier 2 gets its "own stratum, own n, own delta — never continues a
Campaign 2 comparison", and §4's power table assumes curated items join the
primary. Curation produces Tier 2 by construction: it cannot produce Tier 1,
which is defined as labels OpenCRE curated independently of TRACT. As written,
the document forbids the thing it budgets for.

The rebaseline resolves half of it. Once the anchor budget changes, +0.1361 is
no longer a forward target and there is no Campaign 2 comparison left to
continue; the prohibition is satisfied trivially. What survives is the real
question: may Tier-1 and Tier-2 items share one denominator?

**They may, under one pre-registered condition.** The metric is paired —
trained against its own zero-shot on the *same* items — so label provenance
affects whether the gold is a good target, not whether the pairing is valid, and
a wrong label costs both arms equally in expectation. But the two tiers are two
taxonomies, and a model trained on OpenCRE-shaped links may do systematically
better on Tier-1 items, in which case a pooled number moves with the
COMPOSITION of the eval set rather than with model quality.

So, decided now:

> Report the Tier-1 delta, the Tier-2 delta, and the pooled delta, always, all
> three. **The gate is decided on the pooled estimate only if the two strata are
> consistent — their 95% intervals must overlap.** If they do not overlap, the
> pooled figure is a composition artifact, the gate is UNDECIDED, and the two
> strata are reported separately with that stated.

This spends the extra n when it is safe to spend and refuses when it is not.

### 1.3 The echo partition was not frozen, and the floor was set against the wrong number

§3 specifies echo as "the union of title and full prose" and quotes +0.1531 on
n=98. That figure was computed against the **truncated** prose anchor, so the
partition moves with `max_seq_length` — measured on the Campaign 2 corpus: 38
echo items at 2,150 chars, 41 at 4,300, 44 at 8,601 and above. Six items become
echo purely because a longer budget restores the tail that names their hub.

The five items that move under the owner's just-approved budget change carry
trained hit == zero-shot hit, so **re-partitioning alone shifts the
side-condition metric with no change in model behaviour.** A ruler that moves
when the experiment moves cannot bind the experiment.

Corrected. Echo is now computed by `tract/training/echo.py` from the maximum
text an item could ever present — title unioned with full **untruncated** prose
— making it a property of the item and the hub tree alone, budget-independent by
construction and verified so by `tests/test_frozen_echo.py`. It is deliberately
the most generous definition available, which makes the non-echo stratum the
conservative one to claim on.

Restated against Campaign 2's committed indicators under the frozen partition:

| stratum | n | delta | 95% CI |
|---|---|---|---|
| echo (frozen) | 56 | +0.1071 | [−0.0357, +0.2500] |
| **non-echo (frozen)** | **91** | **+0.1538** | **[+0.0440, +0.2637]** |

The binding floor stays at **point estimate ≥ 0.10 and CI low > 0**, now
evaluated on this partition. The historical +0.1743/n=109 and +0.1531/n=98
figures are retired; both were computed against anchors that no longer exist.

### 1.4 Two hazards recorded for the curation round

Neither changes the gate, and both change how the round must be run. They are
specified in `claudedocs/curation-package.md`.

- **The text fix makes `csa_aicm` an echo contributor.** 96.3% of its controls
  admit at least one hub whose name is a content-word subset of the fixed
  full-prose anchor (median 5 such hubs), against 30.9% and median 0 on the
  description field. Curating it grows the echo stratum the side condition
  exists to protect against.
- **`hub_reference.md` is model-derived.** 400 of its 522 hub descriptions were
  LLM-authored conditioned on the existing gold links. It is the obvious
  navigation aid to hand an annotator, and handing it over would make the round
  Tier 3 by §2's own rule.

### 1.5 The rebaseline is a SECOND run on the held-out test split

Recorded before it runs, not after.

The whole point of a validation/test separation is that the test split is
touched once. `scripts/phase1b/await_capacity.py` says so about the Campaign 2
test round in as many words: *"Runs ONCE on the 147-item AI split and is
unrecoverable: a second run would contaminate the split that the whole
validation/test separation exists to protect."*

The rebaseline runs that split again. That is a real cost and it is being paid
deliberately.

**Why it is sanctioned here.** The anchors change, so this is not a second draw
on the same measurement — it is a different measurement on the same items. The
old figure is being *retired* rather than competed against: +0.1361 stops being
a forward target the moment the anchor budget moves, so there is no
"best of two runs" to be had. And no arm selection occurs: one recipe,
`n_configurations=1`, decided before the run.

**What it costs anyway, stated plainly.** The 147 AI items have now been scored
twice by the same recipe family. A third run would be much harder to justify,
and any future claim on this split must disclose that it is not a
never-before-seen set. If the rebaselined number comes out *higher* than
+0.1361, that fact alone is not evidence of improvement — it is one of two
draws, and the honest reading is the one the pre-registered thresholds in §3
give, not the comparison to the retired figure.

**The one variable.** `max_seq_length` 512 → 1024, moving the anchor budget from
2,150 to 4,300 characters and cutting eval truncation from 55 of 147 items.
Batch size stays 32 deliberately: 2,048 tokens would force batch 24, and
changing the batch changes MNRL's in-batch negatives, so a shift could not be
attributed to context rather than to negatives. Everything else is byte-identical
to `c2r_TEST_A3_prose_sw_qwen06b`.

**Provisioning constraint discovered while running it.** SECURE-tier capacity
would only yield 3 of the 5 pods on repeated attempts, and the natural
workaround — running the folds in two batches with `--folds` — is refused by
`run_folds`, correctly: a partial fleet produces a scoped result that would
aggregate as though it were cross-validation. The fleet therefore waits for
five-pod SECURE capacity rather than degrading. SECURE specifically, because
the working tree carries the licensed ISO 27001 and ETSI corpus and `_rsync_to`
ships it to whichever host answers.
