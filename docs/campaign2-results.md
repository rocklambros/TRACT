# Campaign 2 — results

Written 2026-08-28, after the campaign closed and after an adversarial premortem
that attacked these numbers from six independent perspectives. Several claims
made in the campaign's own commit messages did not survive that pass and are
corrected here. Where a commit message and this document disagree, **this
document is right and the commit message is superseded** — the corrections are
listed in §11 rather than quietly applied.

The binding pre-registration is `results/phase1b/CAMPAIGN2.md`. Every number
below is recomputed from committed artifacts; nothing is quoted from memory.

> **Amended 2026-08-29.** A second premortem, run against the *Campaign 3* plan,
> found something in *this* campaign that neither this document nor the campaign
> disclosed: TRACT rewrote the gold label on 25% of the test corpus before
> scoring it. The headline in §1 is unchanged and correct as computed, but it is
> no longer the only number that has to be reported. See **§13**, which is the
> most important section in this file.

---

## 1. Headline

On the five held-out AI frameworks, contrastive fine-tuning of
Qwen3-Embedding-0.6B (arm A3, prose anchors + stop-word filtering) improves
micro-averaged hit@1 over its own paired zero-shot baseline:

| quantity | value |
|---|---|
| trained micro hit@1 | **0.5918** [0.5170, 0.6667] |
| paired zero-shot micro hit@1 | 0.4558 |
| **micro delta** | **+0.1361** [+0.0476, +0.2245] |
| n | 147, paired, fold-stratified bootstrap, 10,000 resamples |
| macro delta (diagnostic, not the metric) | +0.1834 |
| **co-primary, audit-untouched items only (§13)** | **+0.1000** [+0.000, +0.200], n=110 |

**Gate 1 verdict table — all four booleans, because reporting one is how the
last headline was withdrawn:**

| verdict | value |
|---|---|
| `point_estimate_pass` | **true** |
| `ci_low_pass` | false |
| `familywise_pass` | false |
| `verdicts_agree` | **false** |

`CAMPAIGN2.md:196` designates `point_estimate_pass` as the verdict and names the
other two diagnostics. **That clause does not carry the weight it appears to —
see §2.** The honest one-line statement of this result is:

> Fine-tuning improves hit@1 by 13.6 points over its own zero-shot baseline on
> held-out AI frameworks. The improvement over **zero** is decisive. The
> evidence does **not** establish that it exceeds the pre-registered 0.10 gate.

## 2. Why "PASS" is not the word for this

Three facts, each verified against the repository rather than asserted.

**The authorizing clause postdates the arm results.** `CAMPAIGN2.md` has exactly
two commits: `cf92cbf` (2026-08-15, 119 lines) and `ef8ca5d` (2026-08-27). The
original contains **zero** occurrences of `point_estimate_pass`; clause 6 is in
the amendment. Campaign 2 arm results — `c2_A1_prose_sw_bge`,
`c2_A2_prose_sw_bge_bal3`, `c2_canary_qwen` — were committed in `7b03b75` on
**2026-08-19**, eight days before. The amendment asserts twice that it was made
"before any Campaign 2 arm ran"; `CAMPAIGN2.md:120` acknowledges those
directories exist. Clause 6 additionally derives its unreachability arithmetic
(0.188 / 0.216) from per-item indicators on this same test set. **It is not
blind pre-registration and must not be cited as if it were.**

**`p = 0.0016` does not test the gate.** `tract/training/evaluate.py:202`
computes `np.mean(boot_delta_means <= 0)` — the probability the true effect is
at or below **zero**, not 0.10. Recomputed from the committed indicators at
200,000 resamples:

| quantity | value |
|---|---|
| P(true delta ≤ 0.00) | 0.0018 |
| P(true delta ≤ 0.05) | 0.030 |
| **P(true delta ≤ 0.10)** — the gate | **0.203** |
| one-sided 95% lower bound | **+0.0612** |

Roughly a one-in-five chance the true effect is at or below the threshold, and
even the one-sided lower bound falls short.

**The statistical situation is indistinguishable from the withdrawn headline.**

| | Campaign 1 `lofo_title_only` (WITHDRAWN) | Campaign 2 `c2r_TEST_A3` |
|---|---|---|
| micro delta | +0.1293 | +0.1361 |
| 95% CI | [0.0408, 0.2177] | [0.0476, 0.2245] |
| n | 147 | 147 |
| point / ci_low / agree | true / false / false | true / false / false |

`PRD.md:383-388` withdrew the first for exactly this pattern.

**What genuinely distinguishes this result** is §5, not clause 6.

## 3. What the campaign measured, per fold

**Test split** — 5 AI frameworks, run once, `n_configurations=1`:

| fold | n | zero-shot | trained | delta |
|---|---|---|---|---|
| OWASP AI Exchange | 63 | 0.6349 | 0.6984 | +0.0635 |
| MITRE ATLAS | 43 | 0.3023 | 0.4651 | +0.1628 |
| NIST AI 100-2 | 28 | 0.3214 | 0.5357 | +0.2143 |
| OWASP Top10 for ML | 7 | 0.4286 | 0.5714 | +0.1429 |
| OWASP Top10 for LLM | 6 | 0.3333 | 0.6667 | +0.3333 |

No negative folds. But four of five per-fold CIs include zero, so "zero negative
folds" is not corroboration and is not quoted as such.

**Robustness.** Leave-one-fold-out: +0.1176, +0.1250, +0.1277, +0.1357, +0.1905.
Dropping **both** small folds (n=6 and n=7): +0.1269, n=134. The point estimate
never crosses 0.10 under any single-fold removal.

## 4. Validation, and the negative finding

Validation is 5 general-security frameworks, n=1,265, `n_configurations=3`. It
decides **selection only**; it is not the reported population.

| arm | absolute micro hit@1 | delta | 95% CI |
|---|---|---|---|
| A3 Qwen3-0.6B, prose+sw | **0.2680** [0.2443, 0.2925] | −0.0150 | [−0.0419, +0.0126] |
| A5 BGE, title-only | 0.2553 [0.2316, 0.2783] | −0.0221 | [−0.0474, +0.0032] |
| A1 BGE, prose+sw (primary) | 0.1937 [0.1723, 0.2150] | **−0.0609** | [−0.0870, −0.0340] |

**No arm cleared Gate 1 on validation, and the primary arm A1 is significantly
negative** — its interval lies entirely below zero. That is the same shape as
withdrawal reason #4 in `PRD.md:396-398` and it is stated here rather than left
in a commit message.

**A3 won a tie-break, not a separation.** A3 − A5 = 1.26 points against a
pre-registered 4.0-point MDE. Both cleared A1 by more than the MDE, so
`CAMPAIGN2.md:168-169` selected the higher absolute — A3. **"Qwen3-0.6B is the
better encoder" is not established by this campaign.** McNemar p = 0.405; LOFO
flips the winner in 2 of 5 folds.

### 4.1 The sign flip is one fold, not a population effect

The campaign's own commit `3d586c2` framed the validation/test gap as "an
AI-security crosswalk model that helps on AI frameworks and hurts on general
ones." **That framing is wrong.** Per-fold validation deltas for A3:

| fold | n | zero-shot | trained | delta |
|---|---|---|---|---|
| ASVS | 277 | 0.6282 | 0.2347 | **−0.3935** |
| ISO 27001 | 93 | 0.2581 | 0.4409 | +0.1828 |
| CWE | 246 | 0.2602 | 0.3577 | +0.0976 |
| CAPEC | 349 | 0.1089 | 0.1920 | +0.0831 |
| NIST 800-53 v5 | 300 | 0.1933 | 0.2600 | +0.0667 |

**Four of five validation folds are positive.** Excluding ASVS, validation gives
**+0.0911 [+0.0628, +0.1194]** on n=988 — overlapping the test interval. The two
splits agree once one fold is removed.

This is a **post-hoc diagnostic**, chosen after seeing which fold was negative.
It is reported as a diagnostic and never substituted for the headline.

### 4.2 Why ASVS is pathological

ASVS is a 1:1 bijection with the hub tree — the only framework in the corpus
with one distinct hub per link:

| framework | links | distinct hubs | hubs/link |
|---|---|---|---|
| capec | 1799 | 194 | 0.108 |
| cwe | 613 | 268 | 0.437 |
| nist_800_53 | 300 | 66 | 0.220 |
| **asvs** | **277** | **277** | **1.000** |

Its zero-shot 0.6282 is closer to an identity lookup than a semantic mapping,
and fine-tuning degrading an identity lookup is not straightforwardly a model
failure. Whether ASVS is a valid LOFO fold at all is **open** and should be
settled before Campaign 3. `PRD.md:377` requires any negative fold to be flagged
before deployment; this is that flag.

## 5. The result's strongest evidence: the gain is not lexical

Campaign 1's headline was withdrawn in part because most of its lead came from
items whose anchor already contained the answer. This campaign is the inverse:

| subset | n | zero-shot | trained | delta | 95% CI |
|---|---|---|---|---|---|
| lexical echo | 38 | 0.7895 | 0.8158 | +0.0263 | [−0.132, +0.184] |
| **non-echo** | **109** | **0.3394** | **0.5138** | **+0.1743** | **[+0.073, +0.275]** |

**Essentially the entire gain lives on items whose text does not contain its own
answer.** This is a post-hoc decomposition and is not a gate substitute
(`CAMPAIGN2.md:192-195` forbids metric substitution), but it is the single
strongest thing this result has, and it is the axis on which the previous
headline failed.

One consequence: the **absolute** 0.5918 is the fragile number. The
semantic-mapping figure is **0.5138**.

## 6. Limitation the design did not disclose: LOFO is nominal on the test split

Holding out an AI framework removes 10–65 of 4,405 curated links
(**0.23%–1.48%**). Holding out a validation framework removes 94–1,799
(**2.13%–40.84%**; CAPEC alone is 41%). The five test folds share ≥98.5% of
their training data, so they are closer to one model evaluated five ways than to
five independent generalization tests. This is not among the three limitations
listed at `CAMPAIGN2.md:230-233`, and it is a mundane mechanical candidate for
the split difference in §4.1.

**Related and more serious:** of the 71 distinct ground-truth hubs in the test
set, **56 are supervised by ENISA, ETSI, or BIML** — AI-security frameworks that
are never held out on either split. `AI_FRAMEWORK_NAMES` contains only the five
that rotate. So "5 held-out AI frameworks" describes 5 of 8 AI-security
frameworks in the corpus, and the answer hubs remain supervised in training even
when their framework is held out.

## 7. The agentic smoke test failed, and what that does and does not mean

Six OWASP Agentic Top 10 items (zero curated links, genuinely held out), scored
against all five test-round checkpoints: **1, 1, 2, 1, 1 of 6**. Fails both
pre-declared clauses — ≤1 correct, and top-1 in a different `branch_root_id` on
3–4 items per fold. A constant predictor of hub `220-442` scores **3 of 6**, so
this is below a constant predictor.

`is_a_metric: false`. No arm was re-selected on it.

**The campaign's published explanation for this failure was wrong.** Commit
`553ccc7` attributed it to the model being "trained overwhelmingly on control
text" and unable to map risk-level text. The corpus says otherwise:

| link hierarchy_level | fraction pointing into the 803-457 threat subtree |
|---|---|
| **risk** | **34/36 = 94.4%** |
| technique | 23/31 = 74.2% |
| control | 18/500 = 3.6% |
| requirement | 0/277 = 0.0% |
| mitigation | 0/42 = 0.0% |
| *corpus base rate* | *126/4262 = 3.0%* |

Risk-level supervision **is** present, it is highly consistent, and it says
risk → threat hub. The model learned that convention and applied it. The fixture
maps 5 of its 6 risk-level items to *countermeasure* hubs, inverting a 94%
convention. **What the smoke test measured is a disagreement between the
hand-mapping and OpenCRE's own curation convention, not a model deficit.**

Two instrument defects compound it, both of which would produce this FAIL
against a perfect assignment engine:

- **Every one of the six anchors is truncated before the "Prevention and
  Mitigation Guidelines" section.** No countermeasure text reached the encoder.
  The natural experiment is inside the fixture: `privilege` appears six times
  *before* the Prevention heading in ASI02 (correct on all five models) and
  exactly once *after* it in ASI10 (missed on all five) — same hub, same
  framework, opposite outcome.
- **The parser drops the Prevention section entirely for ASI03, ASI04 and
  ASI10.** ASI10's parsed record is roughly half its source. This is upstream of
  the encoder budget; a larger context window would not recover it.

The one item that still supports the original reading is **ASI09**: its anchor
contains "human", "AI" and "oversight", the truth hub is *Human AI oversight*,
and the model answers a threat hub anyway.

## 8. Reproducibility

The headline **reproduces bit-for-bit**. Replaying `gate_decision()` on the
committed indicator arrays yields zero differing keys against
`aggregate_metrics.json` — `ci_low`, `ci_high`, `p_value` all exact. The
bootstrap is seeded (`PHASE1B_BOOTSTRAP_SEED = 42`, 10,000 resamples). Trained
indicators recompute element-by-element from `predictions.json`, an independent
committed source, on all five folds.

**Provenance.** `git_sha 9321ded` uniform across all five test folds;
`all_controls_sha256 c69b06e14796…`, `curated_links_sha256 3d42cbd396f2…`,
`stopwords_sha256 be5cbb35b721…`. All 20 folds across all four arms agree on
these, and `git diff` over `tract/training/`, `tract/text_selection.py`,
`scripts/phase1b/run_fold.py` and `tract/config.py` across the four arm SHAs is
**empty**. The arms are data- and code-comparable.

**Model.** `Qwen/Qwen3-Embedding-0.6B` rev `97b0c614be4d…`, LoRA r=16 α=32
dropout=0.1, lr 5e-4, batch 32, 20 epochs, max_seq 512, seed 42,
`sampling_temperature` 2.0, 3 hard negatives, `hub_rep_format path+name`.

**What an outside reviewer cannot verify.** Per-item **zero-shot** predictions
are not committed — only their hit@1 indicators — so the denominator of the
delta must be taken on trust. `valid_hub_ids` is absent from `predictions.json`,
so label semantics cannot be checked without rebuilding the corpus, which
requires the gitignored licensed overlay. And `aggregate_metrics.json` — the
file carrying the headline — records no `inputs` block and no `git_sha`, so the
freshness instrument does not scan it.

## 9. Text regime

Test anchors: 82 `description`, 65 `full_text`, **0 `title`**, `prose_fraction`
1.0; 39 of 147 (26.5%) truncated at the 512-token encoder budget.

**Caveat on that block:** `tract/training/orchestrate.py:286-297` re-derives the
anchor source by calling `prose_index.lookup(...)` with `control_text` that has
*already* been replaced by prose, against a title-first lookup. The
`text_selection` block in every committed fold record is therefore unreliable in
both directions. A previously circulated claim that "NIST 800-53 v5 and ASVS
fell back to title-only anchors" is an artifact of this defect and is
**withdrawn**.

## 10. Cost

≈ **$204** against $1,000 authorized (range $171–$226). Training accounted for
13.78 GPU-hours of roughly 44.3 billed pod-hours — 31%. The remainder is
bootstrap, four fleets that produced no folds before the pipeline worked, and
two incidents (a stranded `collect`, and a retried launch that put two trainers
on one GPU).

## 11. Corrections to the campaign's own commit messages

| claim | commit | status |
|---|---|---|
| "clears Gate 1 on the pre-registered verdict" | `3d586c2` | **Superseded.** Clause 6 postdates the arm results (§2). |
| `p=0.0016` printed under "threshold 0.10" | `3d586c2` | **Misleading.** Tests against zero; P(≤0.10) = 0.203. |
| `familywise_pass` "above 0.216" | `3d586c2` | **Wrong for this run.** At `n_configurations=1` familywise is nominal; the threshold is 0.188. |
| "helps on AI frameworks and hurts on general ones" | `3d586c2` | **Not supported.** One fold (ASVS); 4 of 5 validation folds positive (§4.1). |
| "147 control-level items" | `3d586c2` | **False.** The 147 span several hierarchy levels. |
| smoke failure = risk-vs-control training gap | `553ccc7` | **Falsified** by the corpus it appeals to (§7). |
| "ISO 27001 Annex A verbatim" | session report | **Accurate.** A later retraction of this description was itself wrong; 52 of 93 rows match, 30 fully, longest run 35 words. |

## 12. What this result does and does not license

**Does:** a claim that contrastive fine-tuning improves CRE hub assignment on
AI-framework text, over its own zero-shot baseline, by an amount whose interval
is [+0.048, +0.225] and whose lower bound does not reach the pre-registered
gate. The gain is concentrated in non-lexical items.

**Does not:** any claim of general improvement in hub assignment — validation is
flat-to-negative and the primary arm is significantly negative. Any claim that
Qwen3-0.6B beats BGE — that was a tie-break inside the MDE. Any claim about
risk-level or agentic text — the only held-out framework of that kind failed.
Any claim about the shipped model: `TRACT_MODEL_PINNED_REVISION` still resolves
to the campaign-1-era BGE model, and **A3 is not the shipped model**.

**Open before Campaign 3:** whether ASVS is a valid fold; whether ENISA/ETSI/BIML
should rotate through the roster; whether the eval needs ~890 AI items for the
gate to be decidable as an interval criterion; and the OWASP Agentic parser's
dropped Prevention sections.

## 13. Undisclosed: TRACT rewrote 25% of the test gold before scoring it

Found 2026-08-29 by the Campaign 3 premortem, not by this campaign. Every number
here is reproducible with `python -m scripts.analysis.audit_stratified_delta`
and locked by `tests/test_audit_disclosure.py`.

### What was done

`data/training/ai_link_audit.csv` is a link-by-link review of all 198 AI
framework links — verdicts `correct` (133), `weak` (40), `wrong` (25).
`scripts/phase0/curate_links.py` parses a replacement hub id out of each
review's free-text `notes` column and applies it. The result is
`data/training/audit_corrections_log.json`: **56 gold labels rewritten, one
link excluded, 197 AI links surviving.**

All 56 fall inside the four frameworks that make up the test split — `ATLAS`
(22), `NIST AI 100-2` (24), `OWASP Top10 LLM` (5), `OWASP Top10 ML` (5). After
the corpus deduplicates to items, **37 of the 147 test items (25.2%)** carry a
gold label TRACT wrote rather than one OpenCRE curated.

**The 4,208 general-security links received no such pass.** The test split got a
gold-quality treatment the training corpus did not. That asymmetry is the
disclosure; it appeared in no `.md` file and had no test through an entire
campaign.

### What it does to the gate

| stratum | n | zero-shot | trained | delta | 95% CI | P(δ ≤ 0.10) |
|---|---|---|---|---|---|---|
| pooled (published headline) | 147 | 0.4558 | 0.5918 | +0.1361 | [+0.048, +0.225] | 0.205 |
| audit-**touched** | 37 | 0.2162 | 0.4595 | **+0.2432** | [+0.081, +0.405] | 0.038 |
| audit-**untouched** | 110 | 0.5364 | 0.6364 | **+0.1000** | [+0.000, +0.200] | **0.531** |

A quarter of the items carry 45% of the headline. On the items TRACT did not
relabel, **the delta is exactly the gate value and a coin flip on clearing it.**

### The mechanism, and why the obvious objection fails

> **SUPERSEDED 2026-08-30 — the mechanism below is refuted; the effect is not.**
> The degree explanation in this subsection makes three testable predictions and
> fails all three, measured on the committed C3TEST artifacts. Chiefly: among
> audit-**untouched** items, high-degree gold does *not* depress the zero-shot
> baseline (0.5091 vs 0.5455 for low-degree), so degree cannot be what drives
> the touched stratum's 0.1892. See `docs/campaign3-audit-mechanism.md`;
> reproduce with `python -m scripts.analysis.audit_mechanism_probe`.
>
> **The stratification and the co-primary +0.1000 stand unchanged.** What does
> not stand is the inference drawn from this mechanism elsewhere — that the
> inflation is "arithmetic, not provenance" and therefore that human curation
> cannot introduce a new bias into the gate. It can.

The obvious objection is that the audit simply made the task easier. It did not,
and that is the interesting part. Rescoring the *same* trained predictions
against pristine pre-audit gold gives **0.5850 against 0.5918** — the audit is
worth **+0.0068** in absolute accuracy, essentially nothing.

It is the *baseline* the audit moves. 49 of the 56 corrections relocate gold
from a sparsely-linked hub to a densely-linked one (median link degree
**3.0 → 7.5**, mean 3.50 → 8.05), collapsing 56 links onto 26 distinct hubs.

> **CORRECTED 2026-08-30 — that degree statistic is an artifact of when it was
> measured.** Degree was counted over `hub_links_curated.jsonl`, the file the
> corrections had *already been applied to*. Because 56 corrections land on 26
> destination hubs, each destination is credited with the corrections that
> arrived there and each source is drained by them. Recomputed on the
> **pre-audit** graph the direction reverses:
>
> | degree basis | median old → new | moved to higher |
> |---|---|---|
> | post-audit (as published above) | 3.0 → 7.5 | 49 of 56 |
> | **pre-audit (correct)** | **4.0 → 3.0** | **20 of 56** |
>
> `scripts/analysis/audit_stratified_delta.py` disclosed the contamination but
> priced it at "+1 per correction"; 56/26 ≈ 2.15 corrections per destination,
> so the true factor is larger and two-sided. Pinned by
> `tests/test_degree_claim_corrected.py`.
>
> **The stratified deltas in this section are unaffected** — none of them was
> computed from degree. What falls is the *explanation*, which
> `docs/campaign3-audit-mechanism.md` then spent three underpowered tests
> refuting when one recomputation would have done it.
High-degree hubs carry more positives and appear in more training batches, so a
fine-tuned model learns them well while a zero-shot encoder has no reason to
prefer them. On touched items the zero-shot scores **0.2162** against 0.5364
elsewhere.

So the audit barely raises the numerator and substantially lowers the baseline.
That inflates a **paired improvement** metric without inflating accuracy — which
is precisely the metric Gate 1 is defined on.

### What this does and does not establish

**Does:** that the pooled +0.1361 cannot be reported alone. The audit-untouched
+0.1000 is a co-primary and belongs beside it wherever the headline appears.

**Does not:** that the audit was wrong, or that it was performed to move the
metric. 24 of 56 corrections carry verdict `wrong`, and re-reading a link and
finding a better hub is legitimate work. The audit CSV carries **no model
prediction, score, or ranking column** — the reviewer was shown the existing
OpenCRE link and judged it, so this is a human relabel, not model-seeded
circularity. Under the Campaign 3 provenance tiers that makes it **Tier 2**
(independently human-authored), not Tier 3.

But Tier 2 is still not Tier 1, and the Campaign 3 design's Tier-1 definition —
"labels OpenCRE curated before TRACT existed" — is **factually false** for
`hub_links_curated.jsonl`. Any gate denominator drawn from that file inherits
the error.

**RESOLVED 2026-08-30, by the owner: no model output was visible to the reviewer
while the audit was performed.** The 56 corrections are **Tier 2** —
independently human-authored — not Tier 3. Nothing model-derived is downstream
in the published dataset, the published model, the 411 assignments imported into
the OpenCRE fork, or the Phase 5B canonical export. The tail risk that those
artifacts were unrecallably contaminated is closed, and the OpenCRE RFC may cite
these links.

**It does not change the number.** Every figure in this section holds exactly as
computed, and the audit-untouched co-primary of **+0.1000 [0.000, 0.200]** stays
the figure to report beside the pooled +0.1361.

> **CORRECTED 2026-08-30.** This paragraph originally justified that conclusion
> with the degree mechanism — "inflates a paired delta whether a human, a model,
> or a coin chose the destination [...] arithmetic rather than provenance."
> **That justification is refuted** (`docs/campaign3-audit-mechanism.md`). The
> effect concentrates in *discretionary* reassignments (verdict `weak`, delta
> **+0.4118**) rather than in genuine error corrections (verdict `wrong`,
> **+0.1500**), and does not scale with the degree change it was attributed to.
> Who picks the destination is exactly what is *not* ruled out.
>
> The **numbers** in this section are unaffected — they never depended on the
> mechanism. What is withdrawn is the downstream inference that human curation
> is therefore safe for a paired gate.

What the answer settles is which rule applies. Tier 2 is legitimate work that
may be published and cited; it simply is not OpenCRE's taxonomy, so it cannot
silently continue a comparison defined against OpenCRE's labels. That is why the
test split is stratified rather than discarded, and why §3 of
`results/phase1b/CAMPAIGN3.md` computes its primary on the untouched stratum.

## 14. The domain-shortcut hypothesis, tested and refuted

Campaign 3's premortem proposed that the +0.1361 might not be semantic at all.
The reasoning was structural and checked out: build the framework-hub bipartite
graph over all 4,405 curated links, enumerate connected components **supplying no
AI/general labels**, and exactly two fall out — **380 hubs / 14 frameworks** and
**78 hubs / 8 frameworks** (the five rotating AI frameworks plus ENISA, ETSI,
BIML). Intersection empty. All 147 test golds land in the 78-hub component.

Worse, the domain is written into the text being ranked. A bare `\bAI\b` regex
over exactly what `build_firewalled_hub_text` emits matches **78/78** AI-component
hubs and **0/380** general ones — `"Technical AI security controls > Secure AI
inference"` versus `"Session management > Session token generation"`.

So a model could score here by learning "AI text → answer in the AI region",
collapsing 522 candidates to 78 with no semantic mapping whatsoever. Nothing in
Campaign 2 excluded it.

### The measurement

`scripts/phase1b/domain_shortcut_probe.py`, run on one pod for 10 minutes at
$1.59/hr (**≈$0.27**). Hand the *zero-shot* encoder the AI region for free —
restrict its ranking pool to those 78 hubs — and see how much of the trained
model's advantage that alone buys.

| quantity | hit@1 |
|---|---|
| full-pool zero-shot (**control**, campaign says 0.4558) | **0.4558**, drift **0.0000** |
| AI-restricted zero-shot (522 → 78 candidates) | 0.4626 |
| **free domain-oracle gain** | **+0.0068** |
| trained model | 0.5918 |

| fold | n | full | restricted | gain |
|---|---|---|---|---|
| MITRE ATLAS | 43 | 0.3023 | 0.3256 | +0.0233 |
| NIST AI 100-2 | 28 | 0.3214 | 0.3214 | +0.0000 |
| OWASP AI Exchange | 63 | 0.6349 | 0.6349 | +0.0000 |
| OWASP Top10 for LLM | 6 | 0.3333 | 0.3333 | +0.0000 |
| OWASP Top10 for ML | 7 | 0.4286 | 0.4286 | +0.0000 |

The control reproduced the campaign's paired zero-shot to **four decimal places**,
which is what licenses reading the restricted figure at all.

### The verdict

**Deleting 444 of 522 candidate hubs — a 6.7× collapse of the label space, chance
rising from 0.0019 to 0.0128 — moves exactly one item in 147.** Four of five
folds do not move at all.

**The domain shortcut explains +0.0068 of the +0.1361. It is refuted as an
explanation of this result.** The base encoder was already ranking effectively
inside the AI region; the 380 general hubs were never meaningful distractors, so
there was no distractor-rejection gain available for fine-tuning to capture.

Two hypotheses died here, and the second is worth recording because it was the
more careful one. The premortem's cross-attack estimated a perfect domain oracle
would recover **70–114%** of the headline, using the BGE zero-shot in
`zero_shot_firewalled_baseline` as a stand-in for the Qwen baseline that actually
produced the delta. It flagged that substitution as the weakest link in its own
argument. It was right to: measured on the real paired baseline, the oracle
recovers **5%**. A stand-in encoder was not a usable proxy, and no amount of
further offline reasoning would have found that out.

### What this changes

**+0.1292 of the +0.1361 is not domain detection.** Combined with §5 — the gain
concentrates on items whose text does not contain its own answer — the
non-lexical, non-structural reading of this result is now the surviving one.

It does **not** rescue §13. The audit-untouched co-primary is still +0.1000, and
these are independent: one says the gain is not a candidate-set artifact, the
other says part of it is a labelling artifact. Both hold.

The pre-registered rule was: if the shortcut is not excluded, fund no curation.
**It is excluded, so curation is justified** — on the audit-untouched reading of
the effect, not the pooled one.
