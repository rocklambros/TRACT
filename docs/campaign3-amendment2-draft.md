# Campaign 3, Amendment 2 — DRAFT for owner decision

**Status: DRAFT. Not in force. Not a pre-registration until the owner approves
it and it is committed before the next scoring run.**

This drafts the two things flagged in `docs/campaign3-audit-mechanism.md`: the
estimator (§6d) and the §1.2 pooling rule (§6b). It changes what a future gate
can conclude, so it is the owner's call, and it must land **before** any curated
item is scored — after is HARKing.

Nothing here restates a verdict. C3TEST's FAIL stands on the estimator it was
run under.

---

## 1. Primary estimator: framework as a random effect

**Current.** Fold-stratified bootstrap: resample items *within* each fold, folds
held fixed. Answers "how would this delta vary on more items from **these** five
frameworks?"

**Problem.** The gate's claim, and the entire justification for LOFO, is
generalisation to **unseen** frameworks. That makes framework a random effect.
Measured on the C3TEST primary:

| framing | 95% CI | P(δ ≤ 0.10) |
|---|---|---|
| folds fixed *(current)* | [+0.0000, +0.2000] | 0.535 |
| framework as random effect | **[−0.0521, +0.3750]** | 0.479 |

2.1× wider, and it crosses zero.

> **Proposed.** The primary estimator resamples **frameworks with replacement,
> then items within each drawn framework**. The gate rule is unchanged in form:
> `P(δ ≤ 0.10) < 0.05`. The fixed-fold interval may still be reported, labelled
> as within-framework precision, never as the primary.

**Consequence, stated plainly:** this makes the gate harder to pass, and it makes
every historical figure computed the old way non-comparable. That is the point —
the old framing answers a question nobody is asking.

## 2. The §1.2 pooling rule is retired, not replaced

§1.2 currently pools Tier-1 and Tier-2 "only if their 95% intervals overlap".
`campaign3-audit-mechanism.md` §6b shows that admits the one relabelling
available to test it, and §6c shows no candidate replacement both catches
composition shift and still permits genuine nulls.

**The rule is unnecessary under §1.** Once frameworks are resampled, a curated
framework is simply another cluster and its composition difference is priced
into the interval automatically. The overlap test was compensating for an
estimator that treated composition as free.

> **Proposed.** Delete the §1.2 overlap rule. Tier-2 frameworks enter the
> primary as additional clusters. Report Tier-1-only, Tier-2-only and combined
> estimates always, all three, as §1.2 already required.

## 3. The relabelling guard, and where it cannot reach

Composition is handled by §1. Relabelling is not: a stratum whose gold was
rewritten can carry an inflated *within-framework* delta, via the 2:1 baseline
asymmetry measured in §6b (zero-shot −0.3381, trained −0.1678).

The best available guard is **R6** — refuse to combine strata whose zero-shot
baselines differ significantly (p < 0.05). It refuses the audit worked example
and permits 94% of true nulls (§6c).

> **Proposed.** Where a framework contains both original and rewritten gold,
> R6 must pass before the two are combined within that framework.

### The gap this leaves — and it is large

**R6 needs a within-framework control, and the four curation targets have none.**
Checked against the curated link set: of 22 frameworks carrying CRE links, the
five AI ones are exactly the current test folds. `csa_aicm`, `cosai`, `aiuc_1`
and `nist_ai_rmf` have **zero** existing CRE links between them. (Cloud Controls
Matrix's 29 links are CSA **CCM**, a different framework.)

So every curated label in those frameworks is annotator-produced, there is no
original-gold stratum to compare against, and **annotator-induced inflation is
not statistically detectable there by any method in this document.**

Process controls — blind packet, no model output visible, two annotators with
15–20% overlap — are necessary but demonstrably not sufficient: the link audit
was itself Tier 2 with no model output visible, and still produced the largest
inflation measured in this project.

> **Proposed, pending §4.** Curated frameworks are reported as their own stratum
> and **cannot carry a gate verdict alone**. A PASS requires the
> OpenCRE-original frameworks to support it independently.

## 4. Minimum cluster size

Two current folds have n ≤ 7 (OWASP Top10 for LLM n=6, for ML n=7). On the
untouched stratum they fall to n=2 and n=4. The n=2 fold has delta exactly
+1.0000 and sample within-variance exactly 0, which is what makes τ
unidentifiable (0.0782 vs 0.3702 depending on inclusion).

> **Proposed.** Frameworks contributing fewer than **10** eval items are
> reported but excluded from the primary estimator, and the exclusion is stated
> with the result. Trade-off acknowledged: this lowers k from 5 to 3 today,
> which is worse for the random-effects estimate. It is proposed anyway because
> a cluster whose variance estimate is exactly zero corrupts τ, and τ drives
> every power calculation. **If the owner prefers to keep them, that must be
> chosen explicitly and τ reported both ways.**

## 5. τ must be reported, and its status stated

> **Proposed.** Every gate decision reports the between-framework SD τ and the
> k it was estimated from. Where τ is unidentified — as it is at k=5 — the
> report says so rather than quoting the more convenient estimate.

## 6. Metric: hit@1 stays primary; NDCG@10 is a prospective secondary

hit@1 discards 70% of eval items as ties (measured discordant rate 0.30). On
the C3TEST round, NDCG@10 has a markedly better cluster-level effect-to-noise
ratio than hit@1 (t = 3.94 vs 2.93; between-framework SD 0.1071 vs 0.1840) —
worth roughly as much as raising k from 5 to 9, for free.

**But that comparison was made on the test round, which is post-hoc metric
selection.** Checked against the effect-independent property (between-framework
SD) across all nine complete runs, NDCG@10 beats hit@1 in only **6 of 9**
(p ≈ 0.25 against a coin flip). The validation arms cannot arbitrate it: their
effects are ~0, so comparing detection there compares two nulls.

> **Proposed.** hit@1 remains the primary. NDCG@10 is pre-registered **now**, in
> advance, as a secondary reported alongside it for the next two runs, with no
> gate authority. If its SD advantage replicates prospectively, a later
> amendment may promote it. Adopting it today on 6-of-9 would be exactly the
> pattern this project has retracted three times.

## 6b. Fold roster: rotate ENISA, BIML and ETSI in

`campaign3-audit-mechanism.md` §6g measures this. Three AI-security frameworks
carry existing OpenCRE Tier-1 gold and never rotate, contributing 74 dedup eval
items and taking k from 5 to 8 at no annotation cost. On the LOFO supervision
check they are no harder than the incumbents (ENISA orphans 7.1% of its gold
hubs against OWASP AI Exchange's 9.4%; BIML and ETSI orphan none), and hub text
is unaffected because it is built from the hierarchy, not from framework links.

> **Proposed.** Add ENISA, BIML and ETSI to `AI_FRAMEWORK_NAMES` and rotate them
> through the LOFO roster on the same terms as the existing five. ETSI runs only
> where the licensed overlay is staged, so pods stay SECURE-tier.

**Expected consequence, stated in advance so it is not read as a surprise:** the
measured delta will probably **fall**. Those three currently act as permanent
supervision donors for the AI hub region, and rotating them means each takes a
turn without that support. A drop is the correction, not a regression.

**What this does not do:** it does not close the supervision leak. Every AI gold
hub is supervised exclusively by AI frameworks (245/245), so a strict all-AI
firewall would orphan 100% of them and leave no trainable task. The leak is a
structural property of the corpus and must be **disclosed as a limitation on the
LOFO claim**, not repaired.

> **Proposed.** Every report of a LOFO result states: unseen *framework*, within
> an AI hub region that sibling AI frameworks supervise.

## 7. What is NOT proposed here

- No change to the 0.10 threshold, the LOFO design, or the hub firewall.
- No restatement of any past verdict.
- No decision on funding curation — see the §4 gap, which is an input to that
  decision, not a resolution of it.

## 8. Open question this amendment does not close

If τ is genuinely ≥ 0.16, no feasible design reaches 80% power (§6e), and no
amendment to the estimator fixes that. In that case the pooled paired-delta gate
is the wrong instrument and the question becomes which instrument replaces it.
That is deliberately left open here.
