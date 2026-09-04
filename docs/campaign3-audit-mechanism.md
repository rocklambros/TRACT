# The audit effect is real. The published explanation for it is not.

Run 2026-08-30 against `c3_TEST_A3_prose_sw_qwen06b_seq1024`, the committed
C3TEST artifacts. Read-only, no model, no pods.

**Verdict: the degree mechanism fails all three of its own predictions.** The
audit effect it was invented to explain survives; the explanation does not. That
matters because the explanation, not the effect, is what made paid curation look
safe for the gate.

Reproduce:

```bash
python -m scripts.analysis.audit_mechanism_probe \
    --run c3_TEST_A3_prose_sw_qwen06b_seq1024
```

---

## 1. What was claimed, and what rested on it

Three artifacts carry the same explanation in the same words —
`docs/campaign2-results.md` §13, `docs/campaign3-rebaseline.md` §4, and the
docstring of `scripts/analysis/audit_stratified_delta.py`:

> "49 of 56 move the gold label from a less-linked hub to a more-linked one
> (median link degree 3.0 → 7.5) [...] Fine-tuning learns high-degree hubs best
> — they carry more positives and appear in more batches — while a zero-shot
> encoder has no reason to prefer them. So relabelling toward high-degree hubs
> mechanically widens a *paired* delta."

From that followed the conclusion recorded in the Campaign 3 wrap-up: the
inflation is **"arithmetic, not provenance"** — it widens a paired delta
"whoever picked the destination." And from *that* followed the decision to treat
human curation as safe for the gate: if the inflation is a property of hub
degree rather than of who chose the hub, then annotators cannot introduce a new
bias, only the one already understood.

That chain is load-bearing for a spending decision, and its first link was never
tested. It was inferred from two facts observed *inside* the touched stratum —
degree went up, zero-shot is low there — which is consistent with the degree
story and equally consistent with several others.

> **CORRECTED 2026-08-30, after the adversarial premortem. The premise did not
> need refuting; it was never true.** "Degree went up" was measured on the
> **post-audit** graph. 56 corrections land on 26 destination hubs, so each
> destination is credited with its own arrivals while each source is drained.
> On the pre-audit graph: median **4.0 → 3.0**, **20 of 56** move to a
> higher-degree hub — the direction reverses.
> (`tests/test_degree_claim_corrected.py`.)
>
> So §3's three hypothesis tests are the expensive route to a conclusion one
> line of arithmetic delivers. Worse, all three return
> `established_at_alpha: False` against this file's own `ALPHA = 0.05`
> (H1 P=0.463, H2 P=0.587, H3 P=0.066), yet §4 below says *"Overturned: the
> explanation"* flatly and installs a replacement reading on the P=0.066
> contrast the probe itself labels SUGGESTIVE ONLY.
>
> **The conclusion stands; the warrant given for it does not.** Read §3 as
> corroboration of an arithmetic result, never as its basis. Full accounting in
> `docs/campaign3-premortem-round1.md` §A2.

## 2. The three predictions

The degree story is not merely a description of the touched stratum. It is a
causal claim about how fine-tuning and hub degree interact, and it makes
predictions nothing else does.

| | prediction | why it follows |
|---|---|---|
| **H1** | Degree predicts the delta **generally**, including on items the audit never touched | If fine-tuning learns high-degree hubs and zero-shot does not prefer them, that is true of every item, not only relabelled ones |
| **H2** | The effect is **dose-dependent** | A correction raising degree by 10 should widen the delta more than one raising it by 1 |
| **H3** | The effect is **indifferent to verdict** | Degree is arithmetic; it cannot care whether the auditor called the old link "wrong" or merely "weak" |

H1 is decisive: it is measured entirely on the 110 audit-untouched items, so
nothing about the audit can confound it.

## 3. Results

### H1 — degree does not predict the delta on untouched items

| stratum | n | zero-shot | trained | delta |
|---|---|---|---|---|
| untouched, **high**-degree gold (>4.5) | 55 | 0.5091 | 0.6182 | +0.1091 [−0.0364, +0.2545] |
| untouched, **low**-degree gold (≤4.5) | 55 | 0.5455 | 0.6364 | +0.0909 [−0.0364, +0.2182] |
| **difference** | | | | **+0.0182 [−0.1636, +0.2000]** |

Unchanged under the alternative degree definition (`gold_degree_primary`:
+0.0218). Terciles are non-monotonic — the highest-degree third has a *smaller*
delta (+0.0645) than the middle third (+0.1667).

**The cleanest single number is the baseline.** The mechanism requires that
high-degree gold *depresses zero-shot*, because that is the whole engine of the
inflation. Measured:

- untouched, high-degree gold → zero-shot **0.5091**
- untouched, low-degree gold → zero-shot **0.5455**
- **touched → zero-shot 0.1892**

Degree moves the zero-shot baseline by 0.036. The touched stratum's baseline
sits **0.32 below** the high-degree untouched items it is supposed to resemble.
Whatever collapses the baseline on touched items, it is not hub degree.

### H2 — no dose-response

| stratum | n | delta |
|---|---|---|
| touched, **big** degree increase (>+3.0) | 16 | +0.2500 [+0.0000, +0.5000] |
| touched, **small/none** (≤+3.0) | 21 | +0.2857 [+0.0476, +0.4762] |

Pearson *r*(degree change, per-item delta) = **+0.1464** (n=37). The point
estimate runs **against** the prediction: items whose gold moved least show the
larger delta.

### H3 — the effect tracks the auditor's judgement, not arithmetic

| stratum | n | zero-shot | trained | delta | median degree change |
|---|---|---|---|---|---|
| verdict = **wrong** (genuine error) | 20 | 0.2000 | 0.3500 | +0.1500 [−0.0500, +0.3500] | +4.0 |
| verdict = **weak** (discretionary) | 17 | 0.1765 | 0.5882 | **+0.4118** [+0.1176, +0.7059] | +2.0 |
| **difference (weak − wrong)** | | | | **+0.2618 [−0.0735, +0.5971]**, P(≤0)=0.066 | |

Doubly damning for the arithmetic story: the stratum with the **larger delta**
had the **smaller degree change**.

**This separation is suggestive, not established** — n=17 and n=20, and the
interval on the difference crosses zero. It is reported because of its
direction, not its significance.

## 4. What this does and does not overturn

**Overturned:** the explanation. No prediction of the degree mechanism is
supported; two have point estimates at zero or against the predicted direction.
Individually each test is underpowered, but they fail *jointly and
consistently*, and H1's baseline comparison is not a close call.

**Not overturned:** the effect. The touched stratum really does show a larger
delta, and it is not a framework-composition artifact — fold-matching the
comparison (dropping OWASP AI Exchange, which contributes no touched items) does
not shrink it:

| | delta |
|---|---|
| touched | +0.2703 [+0.1081, +0.4324] |
| untouched, as published (all folds) | +0.1000 [+0.0000, +0.2000] |
| untouched, fold-matched to touched | +0.0851 [−0.0851, +0.2553] |
| **touched − untouched, fold-matched** | **+0.1852 [−0.0523, +0.4267]**, P(≤0)=0.063 |

So the audit stratification remains the right thing to report. What changes is
what it means.

**The best-supported reading is the one that was ruled out.** The effect
concentrates in *discretionary* reassignments — links the auditor judged
defensible but suboptimal — rather than in corrections of genuine errors, and it
does not scale with the arithmetic property it was attributed to. That is the
signature of *judgement* entering the gold labels, not arithmetic. The owner
decision of 2026-08-30 established that no model output was visible during the
audit (Tier 2), so this is not circularity through the model. It is a human
re-reading links and preferring hubs that fine-tuning, for reasons this probe
does not identify, is much better positioned to hit than a zero-shot encoder.

## 5. A separate finding: what the primary is made of

The published primary — the 110 audit-untouched items — is not a
representative sample of the test corpus:

| fold | n | share of the primary |
|---|---|---|
| OWASP AI Exchange | 63 | **57.3%** |
| MITRE ATLAS | 30 | 27.3% |
| NIST AI 100-2 | 11 | 10.0% |
| OWASP Top10 for ML | 4 | 3.6% |
| OWASP Top10 for LLM | 2 | 1.8% |

One framework supplies **57.3%** of the gate's denominator and **zero** touched
items — the audit applied no *correction* to OWASP AI Exchange.

> **CORRECTED.** An earlier version of this sentence read "the audit never
> touched OWASP AI Exchange at all." That is false. The audit's `exclusions`
> list deletes one OWASP AI Exchange link (`547-824`, "AI model bias testing",
> verdict `wrong`, on the grounds that no suitable CRE hub exists for AI bias
> testing). It removed an item from that fold's denominator rather than
> relabelling one, which is why the corrections-only stratification cannot see
> it. The log records 65 decisions in three lists — 56 corrections, 1 exclusion,
> 8 kept-weak — and `audit_touched` was built from the first list alone. Five
> kept-weak items therefore sit inside the Tier-1 "untouched" primary despite
> having been inspected and affirmed by the same auditor.
>
> The magnitude is one item, so §4's fold-matched robustness check stands. The
> definition gap does not: a future audit that excluded twenty links would leave
> this stratification unchanged. `load_audit_index` now reconciles all three
> lists and refuses a log that disagrees with itself
> (`tests/test_audit_mechanism_probe.py::TestEveryAuditDecisionIsAccountedFor`).

OWASP AI Exchange is also the easiest
fold (zero-shot 0.6190 against 0.4043 across the folds the audit did touch).
Stripping it drops the untouched delta from +0.1000 to +0.0851 and widens the
interval through zero.

The primary is therefore substantially a measurement of one framework, and its
stability across the 512→1024 rebaseline should be read with that in mind.

## 6. What this means for curation

Curation is, by construction, a large-scale exercise in exactly the category
that carries the effect here: discretionary "this hub is a better fit"
judgements over links that are not outright wrong. The reassurance that made it
safe for the gate — inflation is arithmetic, so whoever picks the destination is
irrelevant — is the specific claim this probe fails to support.

That does not mean curation is worthless; better labels are still better
labels. It means **the current gate cannot measure it.** A paired
trained-minus-zero-shot delta over curated gold would move for reasons that
include the annotators' judgement, and this corpus gives no way to separate that
from model skill after the fact.

The gate design has to be settled **before** annotation begins, not after. Once
the labels are rewritten there is no untouched stratum left to check against.

## 6b. The pre-registered safeguard does not catch this

`results/phase1b/CAMPAIGN3.md` §1.2 already anticipates that curated (Tier-2)
items might distort a pooled figure, and pre-registers a guard:

> "Report the Tier-1 delta, the Tier-2 delta, and the pooled delta, always, all
> three. **The gate is decided on the pooled estimate only if the two strata are
> consistent — their 95% intervals must overlap.**"

Its stated justification is:

> "The metric is paired [...] so label provenance affects whether the gold is a
> good target, not whether the pairing is valid, and **a wrong label costs both
> arms equally in expectation.**"

**That premise is measurably false on this corpus.** Relative to untouched
items, relabelling cost the two arms very differently:

| arm | untouched | touched | cost of relabelling |
|---|---|---|---|
| zero-shot | 0.5273 | 0.1892 | **−0.3381** |
| trained | 0.6273 | 0.4595 | −0.1678 |

The relabelling cost the zero-shot arm almost exactly **twice** what it cost the
trained arm. That asymmetry *is* the inflation, and it is the thing §1.2 assumes
away.

**And the guard would not have caught it.** The audit-touched stratum is a
natural experiment for what a curated Tier-2 stratum looks like. Applying §1.2's
rule to it:

| | delta | 95% CI |
|---|---|---|
| Tier-1 (audit-untouched) | +0.1000 | [+0.0000, +0.2000] |
| relabelled stratum (Tier-2-like) | +0.2703 | [+0.1081, +0.4324] |

The intervals **overlap** on [+0.1081, +0.2000]. The guard passes, the gate is
decided on the pooled estimate — **+0.1429 against a Tier-1 truth of +0.1000** —
and `P(δ ≤ 0.10)` moves from 0.535 to 0.161, more than three times closer to
passing, with no change in model behaviour.

The verdict recorded for C3TEST was FAIL either way, so nothing published is
affected. But a guard that admits a 0.043 inflation on the only relabelled
stratum available to test it is not a guard that should be trusted with a
curated stratum several times larger.

**This is the specific thing to fix before annotation starts.** Overlapping
intervals on small strata are weak evidence of consistency; two strata with
n=110 and n=37 will overlap under most realistic inflations. Whatever replaces
it should be pre-registered against this worked example — a rule that pools the
audit-touched stratum with Tier-1 is a rule already known to fail.

## 6c. Candidate replacement rules, scored

Reproduce: `python -m scripts.analysis.gate_rule_candidates`

A replacement must be judged in **both** directions. A rule that refuses
everything is safe and worthless — it discards exactly the extra *n* curation is
being funded to buy. So each candidate is scored on:

- **Worked example** — does it refuse the audit-relabelled contrast? (must)
- **Null permit** — does it still pool two strata drawn from the *same*
  population? Simulated by 200 random 73/37 splits of the audit-untouched
  items, the small half matched to the real Tier-2 *n*. (must stay high)

| id | rule | worked ex. | null permit | verdict |
|---|---|---|---|---|
| **R0** | *status quo*: 95% intervals overlap | pooled | 98% | **FAILS** — admits the inflation |
| **R1** | difference interval covers zero | pooled | 92% | **FAILS** — admits the inflation |
| **R2** | equivalence: difference within ±0.05 | REFUSED | **0%** | too strict |
| **R3** | never pool (Tier-1 only) | REFUSED | 0% | too strict, by construction |
| **R4** | baseline symmetry within 0.10 | REFUSED | 52% | too strict at this margin |
| **R5** | R2 ∧ R4 | REFUSED | 0% | too strict |
| **R6** | **baseline difference n.s. at p ≥ 0.05** | **REFUSED** | **94%** | **USABLE** |

**Why R1 fails is the instructive part.** Testing the *difference* looks like the
obvious repair, but at n=37 the difference interval is [−0.0283, +0.3693] — wide
enough to cover zero while the true gap is +0.1703. Failing to prove a
difference is not evidence of consistency, and R0 and R1 both take pooling as
the default when the test is uninformative.

**Why R2 cannot work here.** Equivalence testing inverts that burden correctly,
but the difference interval spans 0.40 and can never fit inside ±0.05. At n=37
**no delta-based consistency test is feasible** — the delta is simply not
measured precisely enough to demonstrate agreement.

**Why R6 works.** It keeps R1's weak burden of proof but applies it to the
*baseline* instead of the delta. That is not a trick: the relabelling moves the
baseline (−0.3381) about **twice** as far as it moves the delta (+0.1703), so the
same test that is underpowered on the delta (z ≈ 1.7) is decisive on the
baseline (z ≈ 3.3). Its specificity is set by α rather than by a margin fitted
to this dataset — which is why it needs no tuning and R4 does.

R4 is viable at a wider margin (0.20 → refuses the example, permits 97%), but
that margin was chosen by looking at this data. R6 gets the same protection from
a stated α.

### R6's blind spot, stated

Detection curve — baseline drop injected into a synthetic Tier-2 stratum of
n=37, refusal rate over 100 replicates:

| baseline drop | R6 refuses | R0 refuses |
|---|---|---|
| 0.05 | 16% | 0% |
| 0.10 | 30% | 8% |
| 0.15 | 39% | 11% |
| 0.20 | 55% | 28% |
| 0.25 | 71% | 28% |
| 0.30 | 85% | 46% |
| 0.35 | 97% | 67% |

R6 is roughly **50% sensitive at a 0.19 baseline drop** and reliable only above
~0.30. The real audit sits at 0.3381, comfortably inside its range — but a
*subtler* curated inflation would slip through. R6 is a large improvement on R0
at every point on this curve, not a guarantee.

This weakness is a function of n=37. A curated Tier-2 stratum of 200+ items
would tighten it substantially, so R6 gets stronger exactly as curation scales.

## 6d. The pooling rule is the wrong place to fix this

### R6 has a second blind spot, and it is the one that matters here

R6 refuses the observed composition shift (OWASP AI Exchange vs the rest:
baseline gap 0.2148 → refused, correctly). But it catches that *incidentally*,
because that particular shift happens to move the baseline. A composition shift
that preserves the baseline goes straight through. This corpus contains one:

| | n | zero-shot | delta |
|---|---|---|---|
| MITRE ATLAS (untouched) | 30 | 0.4000 | **−0.0333** |
| NIST AI 100-2 (untouched) | 11 | 0.4545 | **+0.2727** |
| | | gap **0.0545** | gap **+0.3061** |

Both strata carry original gold. No relabelling at all. The baselines are
nearly identical (p = 0.780) and the deltas differ by **three times the gate
threshold**. R0, R1, R4 and **R6 all pool them**. Only R2, R3 and R5 refuse —
and those permit 0% of true nulls.

**No candidate rule both permits real nulls and catches composition shift.**

### Why: the four curation targets share no framework with any test fold

`csa_aicm`, `cosai`, `aiuc_1`, `nist_ai_rmf` are disjoint from MITRE ATLAS,
NIST AI 100-2, OWASP AI Exchange, OWASP Top10 for LLM and OWASP Top10 for ML.
(NIST AI RMF is a different document from NIST AI 100-2.) So a curated Tier-2
stratum is **100% composition-shifted by construction** — the adversarial case
above is not a hypothetical, it is the curation plan's design.

And composition has room to dominate. Per-framework delta on untouched items,
folds with n ≥ 10:

| fold | n | zero-shot | delta |
|---|---|---|---|
| MITRE ATLAS | 30 | 0.4000 | −0.0333 [−0.2333, +0.1667] |
| NIST AI 100-2 | 11 | 0.4545 | +0.2727 [−0.0909, +0.6364] |
| OWASP AI Exchange | 63 | 0.6190 | +0.1111 [+0.0000, +0.2222] |

Swing **0.3061 — 3.1× the 0.10 gate.** Which frameworks are in the denominator
matters more than anything else measured in this campaign.

### The estimator, not the rule, is what is wrong

The published interval treats the five folds as **fixed**. That answers "how
would this delta vary on more items from *these* frameworks?" The gate's actual
claim — and the whole point of LOFO — is generalisation to *unseen* frameworks,
which makes framework a **random effect**. Re-running the primary as a cluster
bootstrap that resamples frameworks as well as items:

| framing | 95% CI | P(δ ≤ 0.10) |
|---|---|---|
| folds fixed *(published)* | [+0.0000, +0.2000] | 0.535 |
| **framework as random effect** | **[−0.0521, +0.3750]** | 0.479 |

**2.1× wider, and it crosses zero.** Asked the question the gate is actually
about, the primary does not exclude zero improvement.

This subsumes the pooling problem. Pooling a composition-shifted stratum is only
dangerous because the estimator treats composition as free of uncertainty; an
estimator that propagates between-framework variance prices it automatically,
and no §1.2-style rule is needed.

It also changes the reading of curation. With five frameworks, between-framework
variance is estimated from five points and the cluster interval is correspondingly
unstable. Curation's four new frameworks take that to nine.

> **CORRECTED.** An earlier draft of this section argued from the 1/√k scaling
> law that "the value of curation is more frameworks, not more items." The
> simulation in §6e was then run to check it and **does not support that
> claim**: separating the two levers, more items and more frameworks contribute
> comparably, with items slightly ahead at low τ. The scaling-law shortcut
> ignored that a cluster bootstrap over k=5 is unstable in its own right, not
> merely wide. Both levers help; neither rescues the design when τ is large.
> §6e supersedes the projection.

> **This has not been pre-registered and no verdict is restated on it.** The
> published C3TEST FAIL stands on the published estimator. Changing the estimator
> after seeing results would be exactly the move this project refuses; it is
> raised as a design question for the *next* pre-registration, and the owner's
> call.

## 6e. Power under the random-effects estimator

Reproduce: `python -m scripts.analysis.gate_power_simulation`

The gate is a decision rule (`P(δ ≤ 0.10) < 0.05`) on a cluster bootstrap of a
paired binary outcome over few clusters. No closed form exists, so power is
simulated end to end: draw frameworks, draw items, run the bootstrap, apply the
rule, count passes. 400 studies × 800 bootstrap resamples per cell.

**Validity check:** at μ = 0.10, exactly the threshold, measured power is 2–6%
across every cell — the rule is calibrated at its nominal 5%.

### τ is not identified, so power is a surface

| estimator | τ |
|---|---|
| all 5 folds | **0.3702** |
| folds with n ≥ 10 (k=3) | **0.0782** |

A 4.7× range. The high estimate is driven by a degenerate n=2 fold whose two
items both flipped — delta exactly +1.0000, sample within-variance exactly 0, so
method-of-moments reads it as a *precisely measured* framework at +1.0. With
five frameworks, two of them n ≤ 4, **τ cannot be estimated.** Quoting one power
number here would repeat the error Amendment 1 corrected.

### Power at μ = 0.20 (double the gate threshold)

| scenario | items | τ=0.00 | τ=0.05 | τ=0.08 | τ=0.12 | τ=0.16 | τ=0.20 |
|---|---|---|---|---|---|---|---|
| now (k=5, 22/fw) | 110 | 44% | 41% | 40% | 32% | 26% | 26% |
| +items (k=5, 68/fw) | 340 | 88% | 78% | 63% | 50% | 41% | 39% |
| +frameworks (k=9, 22/fw) | 198 | 73% | 70% | 57% | 48% | 37% | 32% |
| **curated (k=9, 68/fw)** | 612 | **99%** | **94%** | **87%** | **66%** | **48%** | **40%** |

### What it says

1. **The current design is unpowered.** k=5 with 110 items never reaches 80% for
   any μ ≤ 0.20 at any τ. At the pilot's own observed +0.1000 it cannot reach the
   gate at all — that is the definition of the threshold, not a failure.
2. **80% power needs a true effect near 0.175–0.20** — roughly double the 0.10
   gate — **and τ ≤ 0.08.** At τ ≥ 0.12 it needs μ ≥ 0.25.
3. **Both levers matter, comparably.** Separating them, +items and +frameworks
   give similar gains, items slightly ahead at low τ. The earlier 1/√k argument
   that frameworks dominate is not supported (see the correction in §6d).
4. **Nothing rescues a large τ.** At τ = 0.20 even the full curated design tops
   out near 40% at μ = 0.20. If frameworks really do differ that much, this gate
   is not answerable at any feasible scale, and the design question is the gate,
   not the sample.

**The single most valuable next measurement is τ**, because it separates cell 2
from cell 4 and every funding decision runs through it. It cannot be had from
five frameworks — which is the one thing curation would fix, whatever else it
does.

### Assumptions

- Framework effects Normal(μ, τ²); items within a framework i.i.d. McNemar
  cells with the discordant rate fixed at the observed 0.30.
- Balanced n per framework. The real design is severely unbalanced (63/30/11/4/2),
  which is worse than modelled — an unbalanced k=5 behaves like a smaller k.
- Curated frameworks assumed to behave like the observed ones. They are
  different documents and may not.
- Simulation error ≈ ±2.5pp per cell at 400 studies.

### 6e-corrected-again — three errors in 6e-corrected, found at checkpoint 1

**The section below is itself wrong in three places.** Premortem checkpoint 1
measured each one. Read this block first; the section after it is kept for the
record, not for its conclusions.

**1. The surface prices an estimator no verdict uses, and the header says the
opposite.** `results/phase1b/CAMPAIGN3.md` §3 binds the primary as a
*fold-stratified* paired bootstrap — `_build_fold_index_matrix(fold_sizes, …)`
in `tract/training/evaluate.py:154` holds fold sizes **fixed** and resamples
items. `gate_power_simulation._pass_probability` resamples **frameworks**. That
is the §6d random-effects rule, which §6d itself records as *not*
pre-registered, and which Amendment 2 — withdrawn — proposed.

The header below says the recomputation is "on the **right** estimand" and
justifies it by *"the gate reports the item-weighted mean."* Two things were
swapped in one edit and only one was disclosed: the **estimand** (macro → micro,
correct and real) and the **estimator** (fixed-fold → random-effects,
undisclosed). Measured at k=5, 110 items, 1,500 studies per cell:

| μ | τ | §3 as pre-registered | what the surface reports |
|---|---|---|---|
| 0.20 | 0.00 | **65.4%** | 38.4% |
| 0.20 | 0.37 | **48.9%** | 19.5% |
| 0.25 | 0.00 | **94.3%** | 74.5% |

So conclusion 2 below — *"80% power now needs μ ≈ 0.25 and τ ≤ 0.08 even at k=8
with 552 items"* — **is false for the instrument the campaign is bound to**:
94.3% at k=5 with 110 items, μ=0.25, τ=0.

**2. The "anti-conservative" correction is backwards.** μ = 0.10 is not a null.
The gate threshold applies to a study's *realised* micro delta, `Σnᵢδᵢ/Σnᵢ`, not
to μ. At the observed fold weights:

| τ | P(true micro delta > 0.10) |
|---|---|
| 0.00 | 0.000 |
| 0.08 | **0.498** |
| 0.20 | **0.500** |
| 0.37 | **0.502** |

At every τ > 0, **half the simulated studies have a genuinely true effect above
the gate**. An 11.5% pass rate there is a *power* number, not a type-I rate, and
a rule firing on 11.5% of studies when 50% are true positives is grossly
**conservative**. At τ = 0 — the actual null — the surface reads **0.0275**,
below the nominal 5%. The claim that "a PASS in that regime is more likely to be
spurious" is the opposite of what was measured, and it was stated under a bold
heading as the unsafe direction.

**3. The τ null p-value is an artifact of a homogeneous discordant rate.** The
null simulation imposes 0.30 on every fold. The observed rates are
0.2381 / 0.3000 / 0.4545 / 0.5000 / **1.0000** — and the file's own docstring
already records that fixing the pooled value *"understates within-fold variance
on four of five folds."* Applying that caveat to the null as well as to power,
20,000 replicates:

| null | P(estimate ≥ 0.3702 \| true τ = 0) |
|---|---|
| pooled 0.30 *(as published)* | 0.043 |
| **each fold at its own observed rate** | **0.391** |

So *"sits at the 95.2nd percentile of the null"* becomes roughly the 60th. The
honest statement is not that τ is significantly non-zero, nor that it is
distinguishable from zero — it is that **the observed value is uninformative in
both directions**, which is a stronger version of the same conclusion and does
not rest on a p-value quoted to three decimals with a Monte-Carlo error of
0.005.

**Consequences.** `results/analysis/power_surface.json` is stale — it carries
`realised_tau`/`realised_mu` keys the current code no longer emits, so it cannot
have come from the committed script. It is regenerated, and the tables below are
not to be quoted until they are re-derived under both estimators.

**What survives from below:** the macro→micro estimand fix (real, and the fold
sizes now match the design), the τ leave-one-fold-out span 0.0000–0.4446 (real,
and the reason no design can be planned on τ), and the statement that Phase 2C
does not fix τ.

---

### 6e-corrected — recomputed 2026-09-01, on the right estimand

Everything above in §6e was computed with an identical item count per fold,
which makes the statistic an **unweighted mean over frameworks**. The gate
reports the **item-weighted** mean. On the real primary those are +0.2701 and
+0.1000 — either side of the 0.10 threshold. The surface was sizing a design for
a number nobody reports.

Recomputed with `pooled_delta` (micro), the observed fold sizes 63/30/11/4/2,
τ swept to 0.37, and the clamp's effect reported. Machine-readable at
`results/analysis/power_surface.json`.

**Power at μ = 0.20:**

| scenario | items | τ=0.00 | τ=0.05 | τ=0.08 | τ=0.12 | τ=0.16 | τ=0.20 | τ=0.28 | τ=0.37 |
|---|---|---|---|---|---|---|---|---|---|
| now (k=5) | 110 | 38% | 36% | 33% | 30% | 28% | 28% | 26% | 20% |
| +items (k=5) | 340 | 78% | 69% | 55% | 44% | 35% | 33% | 29% | 24% |
| +frameworks (k=8) | 184 | 62% | 63% | 50% | 39% | 38% | 29% | 26% | 18% |
| +both (k=8) | 552 | 97% | 92% | 77% | 55% | 44% | 36% | 24% | 22% |

**Three corrections to what §6e concluded.**

1. **The calibration claim was wrong, and in the unsafe direction.** §6e reported
   2–6% at μ = 0.10 and called the rule "calibrated at its nominal 5%."
   Recomputed, the range is **1–12%**, reaching 12% at k=5, τ=0.37. The gate is
   **anti-conservative when τ is large** — a PASS in that regime is more likely
   to be spurious than the stated α implies.
2. **80% power is further away than stated.** It now needs μ ≈ 0.25 *and*
   τ ≤ 0.08 even at k=8 with 552 items. At μ = 0.20 only the largest design
   reaches it, and only at τ ≤ 0.05.
3. **Nothing reaches 80% at τ ≥ 0.16**, for any design, at any μ ≤ 0.25. The
   ceiling there is 68%.

### τ is not merely unidentified — it is barely distinguishable from zero

Simulating at a **true τ of 0** with the real fold sizes and running the same
method-of-moments estimator the data was measured with, 2,000 replicates:

| | value |
|---|---|
| median estimate under true τ = 0 | **0.0000** |
| fraction returning exactly 0.0000 | **52.5%** |
| 95th percentile | **0.3641** |
| **the corpus's observed estimate** | **0.3702** |
| **P(estimate ≥ 0.3702 \| true τ = 0)** | **0.048** |

The observed value sits at the 95.2nd percentile of the null. It clears p < 0.05
by four thousandths, on an estimator that returns exactly zero half the time when
the truth is zero, at fold sizes including one of n=2 and one of n=4.

**No design can be planned on that.** The τ axis of the table above spans the
entire plausible range and the data cannot narrow it.

> **This is the limitation Phase 2C does not fix.** Bridge links add *training*
> supervision; they do not add evaluation folds, so they leave τ exactly where it
> is. Narrowing τ needs more evaluation frameworks with n ≥ 10 — a different
> deliverable, and one the roster rotation only half-provides (§6g: three folds
> of 33, 17 and 24, of which 91.9% of the gold hubs are already incumbent).

## 6f. Is the pooled paired-delta gate the right instrument?

Four structural problems have accumulated, three of them measured here:

1. **Power collapses if τ is large** (§6e). At τ ≥ 0.16 no feasible design
   reaches 80%, curated or not.
2. **The paired delta is inflatable by relabelling** (§6b) — 2:1 baseline
   asymmetry, and the pre-registered guard does not catch it.
3. **The curation targets have no within-framework control.** `csa_aicm`,
   `cosai`, `aiuc_1` and `nist_ai_rmf` carry **zero** existing CRE links, so
   annotator-induced inflation is undetectable there by any method here.
4. **hit@1 discards 70% of the eval set.** The measured discordant rate is
   0.30; the other 70% of items are ties carrying no information.

### Option A — absolute accuracy as a co-primary

Attractive in principle: a relabelling shifts *both* arms, so an absolute score
should be far less sensitive to it than a paired difference.
`docs/campaign2-results.md` §13 quantifies this as the audit being worth
**+0.0068** absolute — essentially immunity.

**That figure does not reproduce.** Rescoring the committed Campaign 2 test
predictions against gold reconstructed from `audit_corrections_log.json`
(positional join validated on all 147 items; post-audit score reproduces the
published 0.5918 exactly):

| pre-audit gold reconstruction | score | audit worth |
|---|---|---|
| **swap** corrected link old-for-new *(the faithful one)* | 0.5170 | **+0.0748** |
| union of old and new *(lenient)* | 0.6122 | −0.0204 |
| **as documented in §13** | **0.5850** | **+0.0068** |

The documented value sits between the two natural reconstructions and matches
neither. **This is reported as an unresolved discrepancy, not as an error** —
§13 may have rescored against a pre-curation links file rather than replaying
the log, and that file would be authoritative where this replay is not.

What it does mean: the immunity claim cannot currently be relied on. On the
faithful reconstruction, absolute accuracy is about **2.3× less**
relabelling-sensitive than the paired delta (+0.0748 vs +0.1703) — a real
advantage, but not the ~25× the documented figure implies. Option A is worth
having as a co-primary; it is not a solution on its own, and the discrepancy
should be settled first.

### Option B — a rank-aware metric (NDCG@10 / MRR)

On the C3TEST round, NDCG@10 has a much better cluster-level effect-to-noise
ratio than hit@1 (t = 3.94 vs 2.93; between-framework SD 0.1071 vs 0.1840),
worth roughly as much as raising k from 5 to 9 — for free.

**But that is post-hoc metric selection on the test round.** Tested on the
effect-independent property (between-framework SD) across all nine complete
runs, NDCG@10 beats hit@1 in only **6 of 9** (p ≈ 0.25). The validation arms
cannot arbitrate: their effects are ~0, so comparing detection there compares
two nulls. Promising lead; not adoption-grade. Pre-register prospectively
(Amendment 2 §6).

### Option C — Tier-1 frameworks carry the verdict; curated frameworks report separately

Follows directly from problem 3. Honest, and needs no new statistics.

**Cost:** curation then cannot move the gate at all, which removes the stated
reason for funding it. It would still buy training data, a τ estimate, and a
separately-reported generalisation figure.

### Option D — per-framework conjunctive gate

Require the delta to clear the threshold in a majority of frameworks. Immune to
composition by construction. **Cost:** brutal power at k=5; only becomes
plausible at k=9+.

### What the evidence supports

No option is clean. The measured position is:

- The instrument is not obviously wrong, but it is **fragile in three
  independent ways**, and problem 3 is structural rather than statistical —
  no estimator fixes a stratum with no control.
- **τ is the pivot.** At τ ≤ 0.08 the current instrument works once k reaches 9.
  At τ ≥ 0.16 nothing does, and the gate should be replaced rather than
  re-powered.
- τ cannot be measured at k=5. **Every path runs through getting more
  frameworks**, whatever the gate ends up being — which is the one thing all
  four options agree on.

## 6g. The fold roster: three free AI frameworks, and a leak that cannot be closed

`docs/campaign2-results.md` §6 already records this, and line 335 lists it as
open before Campaign 3 — it is not a new discovery here:

> "of the 71 distinct ground-truth hubs in the test set, **56 are supervised by
> ENISA, ETSI, or BIML** — AI-security frameworks that are never held out on
> either split. `AI_FRAMEWORK_NAMES` contains only the five that rotate."

Three AI-security frameworks carry existing OpenCRE gold and never rotate:

| | links | dedup eval items | licensed? |
|---|---|---|---|
| ENISA | 68 | 33 | no |
| BIML | 21 | 17 | no |
| ETSI | 36 | 24 | **yes — needs overlay** |
| **total** | **125** | **74** | k: 5 → 8 |

### Would they be fair folds? Yes.

Two checks. First, **hub text cannot go empty.** `build_firewalled_hub_text`
emits `"{hierarchy_path} | {hub_name}"` from the CRE hierarchy; framework
sections enter only under `include_standards` (ablation A3), which the primary
does not use. Holding a framework out does not touch hub text.

Second, LOFO **training supervision** — after holding a framework out, do its
own gold hubs still have positives from elsewhere?

| fold | gold hubs | orphaned | median remaining |
|---|---|---|---|
| MITRE ATLAS | 41 | 2.4% | 3 |
| NIST AI 100-2 | 27 | 3.7% | 3 |
| OWASP AI Exchange | 64 | **9.4%** | 3 |
| OWASP Top10 for LLM | 10 | 0.0% | 6 |
| OWASP Top10 for ML | 7 | 0.0% | 7 |
| **ENISA** | 56 | **7.1%** | 3 |
| **BIML** | 11 | **0.0%** | 5 |
| **ETSI** | 29 | **0.0%** | 3 |

The candidates are **no harder than the incumbents** — ENISA's 7.1% is below
OWASP AI Exchange's 9.4%, and BIML and ETSI orphan nothing. Rotating them in is
sound.

### But it does NOT close the leak — and nothing can

An earlier draft of this analysis claimed rotating them in would fix the
supervision leak. **That is wrong.** Rotating subjects them to LOFO in turn; it
does not stop sibling AI frameworks supervising the answer hubs when some other
AI framework is held out.

The version that would close it — hold out **all eight** AI frameworks whenever
evaluating any one of them — was measured:

| firewall | orphaned AI gold hubs |
|---|---|
| rotate (hold out the evaluated framework only) | 12 / 245 = **4.9%** |
| strict (hold out all 8 AI frameworks) | 245 / 245 = **100%** |

**Every AI gold hub is supervised exclusively by AI frameworks.** Not one has a
training positive from a non-AI framework. That is the same structural fact
already recorded in PRD.md §58 ("Zero hubs currently bridge AI to traditional")
and §14's two disconnected components — and it means the strict firewall leaves
no trainable task at all.

> **So the leak is not a bug to be fixed; it is a permanent property of this
> corpus.** Any "generalises to an unseen framework" claim must be qualified:
> unseen *framework*, inside an AI hub region that sibling AI frameworks
> supervise. Rotating the roster makes that treatment symmetric across all eight
> rather than privileging three as permanent donors — an honesty gain and a
> power gain, but not a repair.

## 7. Limits

- The three tests are underpowered individually; the argument rests on their
  joint direction plus H1's baseline comparison.
- This probe identifies what the mechanism is **not**. It does not establish
  what it **is**; "discretionary judgement" is the best-supported reading, not
  a demonstrated one.
- Several strata are small (n=16–21). Every interval is reported.
- The tests were chosen after the effect was known. They are predictions of a
  published mechanism rather than pre-registered hypotheses, and H3 in
  particular should be re-tested on any future audited corpus before being
  relied on.
- **The rule comparison in §6c selects on a single adversarial case.** R6 beats
  R0 on this worked example and on the injected-shift curve, but one real
  relabelling is n=1. R6 is chosen because its specificity comes from a stated
  α rather than from a margin fitted to this data — that property, not the
  scoreboard, is the reason to prefer it.
- The null simulation splits one framework mix into two. A real Tier-2 stratum
  will also differ in framework composition, which R6 does not test for. The
  57.3% OWASP AI Exchange concentration in §5 is a separate exposure and needs
  its own guard.

---

## 8. Estimator correction, 2026-09-04 — every interval in this file was recomputed

**The point estimates did not move. Five intervals and p-values did, in the
third decimal. No verdict in this file changed.**

`audit_mechanism_probe.py` produced every figure here by threading **one**
`np.random.Generator` through each `score()` and `contrast()` call in sequence.
A stratum's draws therefore depended on how many draws every prior stratum had
consumed, so a published interval moved with the order the strata happened to
be computed in and with any unrelated upstream draw.

This is the identical defect that was found and fixed in the sibling module
`gate_rule_candidates.py`, where it was *measured*: it printed **+0.4595** for a
contrast whose 500,000-resample reference value is **+0.4324**. The fix there —
deriving each stratum's stream from a SHA-256 hash of its own contents, so a
stratum's interval is a property of that stratum — was not carried across at the
time. It is now (`_stratum_rng`), together with the resample count: 10,000, the
setting under which the sibling's artifact was measured, reproduced its
reference on 11 of 12 seeds; 100,000 reproduced it on 12 of 12. This module
publishes into a results document, so it takes the count shown to be stable.

| figure | as first published | corrected |
|---|---|---|
| H1 contrast, P(≤0) | 0.459 | **0.463** |
| H2 contrast, P(≤0) | 0.581 | **0.587** |
| H3 contrast, P(≤0) | 0.064 | **0.066** |
| untouched, fold-matched, CI | [−0.0644, +0.2340] | **[−0.0851, +0.2553]** |
| touched − untouched fold-matched | [−0.0523, +0.4209], P=0.061 | **[−0.0523, +0.4267], P=0.063** |

The headline primary is **unchanged and exact**: untouched n=110,
**+0.1000 [+0.0000, +0.2000]**. H1, H2 and H3 all remain `SUGGESTIVE ONLY`
against `ALPHA = 0.05`, so §3's reading and §4's conclusion stand as written.

`tests/test_probe_order_independence.py` pins the property. Note for anyone
extending those tests: the first version of its three `score()` tests compared
`ci_low`/`ci_high` and **passed against the unfixed code** — on a binary
fixture the 2.5th percentile is discrete and lands on the same value whichever
stream produced it. They discriminate only when asserting on the full resample
distribution. A test of a randomness property that compares two summary
percentiles is probably not testing anything.
