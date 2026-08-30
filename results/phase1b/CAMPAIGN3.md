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
