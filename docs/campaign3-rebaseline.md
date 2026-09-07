# Campaign 3 — the anchor-budget rebaseline

Run 2026-08-30 on five SECURE-tier H100 pods. Arm `C3TEST`, config
`c3_TEST_A3_prose_sw_qwen06b_seq1024`. Pre-registration:
`results/phase1b/CAMPAIGN3.md` plus Amendment 1, both committed before this ran.

**Verdict: FAIL, on the pre-registered rule.** Details in §3.

---

## 1. What was changed, and why

One variable: `max_seq_length` **512 → 1024**, which moves the anchor character
budget from 2,150 to 4,300 and cut eval truncation from **55 of 147 items to
28**. Batch size stayed at 32 deliberately — 2,048 tokens would have forced
batch 24, and changing the batch changes MNRL's in-batch negatives, so a shift
could not have been attributed to context rather than to negatives. Everything
else is byte-identical to `c2r_TEST_A3_prose_sw_qwen06b`.

Three code defects had to be fixed first, or the intervention would not have
been the intervention:

- `build_training_pairs` declared `max_chars` and never forwarded it, so every
  training anchor was pinned at 2,150 characters whatever the config said.
  Raising the budget would have lengthened **eval** anchors while **training**
  anchors stayed put — train/eval skew introduced by the flag meant to remove
  one.
- `run_experiment` built the eval corpus with no budget at all, so it disagreed
  with `run_fold.py` for any configuration that was not 512.
- The fold record's truncation count was a length heuristic that undercounts,
  because `prepare_anchor` rstrips after cutting. It reported 39 where the truth
  was 55.

All three are covered by `tests/test_anchor_budget_contract.py`. The third is
verified end-to-end by this run: every fold record's count matches the
authoritative `SelectionStats` figure exactly.

## 2. The result

Per fold:

| fold | n | zero-shot | trained | delta | truncated |
|---|---|---|---|---|---|
| MITRE ATLAS | 43 | 0.3023 | 0.3256 | +0.0233 | 0 |
| NIST AI 100-2 | 28 | 0.3214 | 0.6071 | +0.2857 | 5 |
| OWASP AI Exchange | 63 | 0.6190 | 0.7302 | +0.1111 | 18 |
| OWASP Top10 for LLM | 6 | 0.1667 | 0.6667 | +0.5000 | 3 |
| OWASP Top10 for ML | 7 | 0.4286 | 0.7143 | +0.2857 | 2 |

Against Campaign 2, on **identical strata** — the frozen echo partition and the
audit stratification are properties of the items, so these are like-for-like:

| stratum | n | C2 delta (512) | **C3 delta (1024)** |
|---|---|---|---|
| pooled | 147 | +0.1361 | **+0.1429** |
| **Tier-1, audit-untouched — THE PRIMARY** | 110 | **+0.1000** | **+0.1000** |
| non-echo, frozen — the side condition | 91 | +0.1538 | **+0.1319** |
| echo, frozen | 56 | +0.1071 | **+0.1607** |

## 3. The gate

| criterion | required | measured | verdict |
|---|---|---|---|
| **Primary** — Tier-1 delta | `P(δ ≤ 0.10) < 0.05` | **0.535** | **FAIL** |
| **Side condition** — non-echo delta | point ≥ 0.10 **and** CI low > 0 | +0.1319, CI [+0.0220, +0.2418] | **PASS** |

The pre-registered outcome table gives `fail` / `pass` → **FAIL**, with the side
condition reported as a diagnostic and never as the result.

Primary detail: +0.1000, 95% CI [+0.0000, +0.2000], `P(δ ≤ 0.10) = 0.535`. The
interval's lower bound sits on zero.

## 4. What this establishes

**Doubling the context budget moved the primary by exactly nothing.** +0.1000 at
512 tokens, +0.1000 at 1024, on the same 110 items. Truncation halved, so the
intervention took effect; the primary did not notice.

That is a clean negative result and it retires a live hypothesis. The Campaign 3
premortem established that the discarded tail was substantive prose rather than
boilerplate — 47% of all eval prose was being thrown away, and in 0 of 53
truncated items did the description finish inside the kept head. It was
reasonable to expect that restoring it would help. It did not.

The premortem also recorded the one measurement that predicted this, and it
deserves the credit: the ground-truth hub name's content words already appeared
in the *kept head* for 53 of 53 truncated items, 0 tail-only. The model was
never blind to the topic — only to the discriminating detail — and it turns out
the discriminating detail is not what was limiting it.

**The extra context helped where the answer is in the text.** The echo stratum
went **+0.1071 → +0.1607** while non-echo went **+0.1538 → +0.1319**. Longer
anchors pull in more text, some of which names the hub: measured on this corpus,
the echo partition grows from 38 items at 2,150 characters to 41 at 4,300 and 44
at 8,601. That is precisely why the partition was frozen against untruncated
text before this ran — had it been recomputed per arm, five items would have
moved between strata and the side-condition metric would have shifted with no
change in model behaviour at all.

**Absolute accuracy fell slightly on both arms** — zero-shot 0.4558 → 0.4422,
trained 0.5918 → 0.5850. More context made the task marginally harder for both,
which is consistent with additional text adding distractors faster than signal.

**The audit effect persists and is larger.** The audit-touched stratum moved
+0.2432 → +0.2703 while the untouched stratum stayed at +0.1000. Nothing about
the anchor budget touches the mechanism described in
`docs/campaign2-results.md` §13.

> **AMENDED 2026-08-30.** The *effect* above is confirmed and survives
> fold-matching (+0.1852 [−0.0523, +0.4209] against untouched items drawn from
> the same folds). The *mechanism* referred to here is refuted — degree fails
> all three of its own predictions on these very artifacts. See
> `docs/campaign3-audit-mechanism.md`.
>
> That probe also found something this section should have reported: the
> published primary is **57.3% OWASP AI Exchange** (63 of 110 items), the
> easiest fold and the only one the audit never touched. Dropping it moves the
> untouched delta +0.1000 → +0.0851 with the interval crossing zero. The
> primary's stability across 512→1024 should be read with that composition in
> view.

## 5. What it does not establish

It does not show that context length is irrelevant in general — only that going
from 512 to 1024 tokens does not move this metric on this corpus. 28 of 147
anchors still truncate. A 2,048-token arm would take that to near zero, and is
**not** ruled out by this result; it was ruled out of *this* run because it
forces batch 24 and would have confounded context with in-batch negatives.

It does not license comparing +0.1429 to Campaign 2's +0.1361 as an improvement.
Both are pooled figures over a test split that has now been scored twice by the
same recipe family (Amendment 1 §1.5), and the pooled figure is not the
pre-registered primary in either campaign.

## 6. Cost and provenance

Five SECURE-tier H100 pods at $3.29/hr, roughly 1h45m wall including bootstrap
and collection, plus about $6 of aborted provisioning attempts. Under $40 total.

SECURE specifically: `_rsync_to` ships the working tree — `data/processed/licensed`
included — to whichever host answers, and the first provisioning attempt put four
of five folds on COMMUNITY third-party hosts. That run was torn down before
bootstrap, so no licensed corpus left the machine, and the tier is now
restricted at pod-creation time whenever the licensed overlay is staged.

Reproduce the analysis from committed artifacts:

```bash
python -m scripts.analysis.audit_stratified_delta \
    --run c3_TEST_A3_prose_sw_qwen06b_seq1024
```
