# Adversarial premortem, checkpoint 1 — Phase R and the Phase 2C design

Four perspectives (Red Teamer, Governance/Risk, Data Scientist, ML Engineer),
independent contexts, run 2026-09-01 against branch `campaign3-premortem-fixes`.
Security Architect and MLOps skipped with reason: Phase R added no pods, no data
flows and no trust boundaries. Their round-1 findings (`B5`–`B8`) remain open and
are scheduled for checkpoint 2, where the annotator packet gives them surface.

**Verdict: Phase R fixed six real defects and introduced four more, three of them
inside the corrections themselves. The Phase 2C design was unpassable by
construction on both gates.** Everything below was re-measured by the
orchestrator before being accepted.

---

## A. Fixed in response to this checkpoint

### A1 — Both Phase 2C gates were pre-registered to fail
**Found independently by Red Teamer and Governance.**

Gate 1 required 23 AI hubs de-orphaned (78 → ≤55). D1 scoped the packet to the
**top 20 hubs by eval weight**. A link carries one `cre_id`, so a flawless
annotator accepting every hub on the sheet reaches 20 and fails. The design
terminated its own funding path, and the FAIL would have read as *"the two
domains are less connected than the product assumes"* — a substantive conclusion
manufactured by a counting error.

The 20-hub scope was a second defect. "Eval weight" counts how often a hub
appears as gold in the held-out 147-item split that Gate 2 then scored — a
**selection rule derived from the test set**, which is the leakage shape that
withdrew two prior campaigns, entering through the sampling frame rather than
the corpus.

Gate 2 required the τ leave-one-fold-out swing under 0.15. Bridge links add
training supervision, not evaluation folds, so fold sizes are unchanged by
construction and the measured swing is **0.3702 → 0.0000**. This project's own
document states that limitation forty lines from the criterion that ignored it.
The Data Scientist measured the criterion's pass probability as **12–39% on noise
alone**, moving 4pp across a μ range of 0.00–0.30 — essentially blind to model
skill.

**Fixed:** all 78 hubs, unranked, no test-set statistic in the packet. Gate 2
moved to the **validation** split under the strict all-AI firewall, τ criterion
dropped. Four pre-registered quality conditions added, since Gate 1 counts hubs
and was gameable by volume.

### A2 — CI executed 41 tests out of ~2,850
**Governance.** `tests/test_ai_framework_sets.py` imports `tract.training.data`
→ torch, absent from CI's `requirements.txt`. With `pytest -x`, the ImportError
aborted the run: CI at `c817b5b` reported *"1 failed, 40 passed"* and read as an
ordinary red while **2,690 tests silently did not execute**. Every verification
claim on this branch had been a local run.

This is open issue #55's failure mode, recurring inside work meant to raise
rigour. **Fixed:** the containment guards import only torch-free modules; `-x`
removed from the CI invocation.

### A3 — Three errors inside `6e-corrected`, the correction written to fix `6e`
**Data Scientist.**

1. **Undisclosed estimator swap.** `CAMPAIGN3.md` §3 binds a *fold-stratified*
   bootstrap — folds fixed, items resampled. The simulation resamples
   **frameworks**, which is the §6d random-effects rule that §6d itself records
   as not pre-registered and withdrawn Amendment 2 proposed. The header claimed
   the recomputation was "on the right estimand" and justified it by *"the gate
   reports the item-weighted mean"* — presenting an estimator replacement as an
   estimand match. Measured: the pre-registered rule gives **94.3%** power at
   k=5/110 items/μ=0.25/τ=0 where the surface reported 74.5%.
2. **The anti-conservatism claim is backwards.** μ=0.10 is not a null. The gate
   threshold applies to a study's *realised micro delta*, and at the observed
   fold weights every τ>0 puts **~50%** of studies genuinely above it. At the
   real null (τ=0) the surface reads **0.0275** — conservative.
3. **The τ p-value is an artifact.** The null imposed one discordant rate on
   every fold; observed rates run 0.2381 to **1.0000**. Per-fold:
   P(estimate ≥ 0.3702 | τ=0) = **0.391**, not 0.048.

**Fixed:** `6e-corrected-again` records all three; `power_surface.json` deleted
as a schema mismatch with the committed code.

### A4 — The gate returned PASS on an empty fold set
**ML Engineer.** `_pass_probability` divided `sums / counts` unguarded. With no
folds, `nan <= threshold` is False, `P(δ ≤ 0.10)` reads 0.0, and
`0.0 < GATE_ALPHA` returns **True**. In a project whose record is three
withdrawn passes, a silent PASS is the worst available direction. **Fixed** with
the guard the sibling module already had.

### A5 — Three more mutation survivors
**ML Engineer, 18 mutations, 12 survived.**

- `pooled_delta` — named in the docstring as "the estimator" — was **dead code**.
  The live statistic is the vectorised form inside `_pass_probability`; mutating
  *that* to macro left the suite green while moving power **20.5pp** at μ=0.25.
- The calibration test asserted `power < 3 * GATE_ALPHA`, which scales with the
  constant it validates. Inflating `GATE_ALPHA` tenfold passed it unchanged.
- `TestTheGuardCatchesADesync` — written specifically to prove the guards could
  fail — **could not fail**. Two tests asserted `x not in (S - {x})`, true for
  every S. The third was an **inverted tripwire**: adding UK DSIT to the packet
  exclusion, the remediation its own comment prescribes, turned it red. An
  engineer would have reverted the fix and leaked the answer key for the
  framework under test.

**Fixed:** all three, with the guard bodies extracted so the desync tests run
real code against perturbed constants.

---

## B. Confirmed as genuinely fixed

- **Deleting the cluster resampling is now killed by two tests** (`M3b`). Phase
  R's headline claim holds.
- The bridge-links-do-not-touch-the-eval-corpus property **verifies**: 147 items,
  fingerprint identical before and after.
- Under the CI pins on a clean checkout, ruff and mypy are both clean. The 7
  local mypy errors are phantoms from a gitignored `wandb/` directory.
- `audit_mechanism_probe` reproduces every published figure exactly.

---

## C. Open, carried to checkpoint 2

| | finding | source |
|---|---|---|
| C1 | **The model card falsehood is still live on HuggingFace.** Only the generator is fixed, on an unmerged PR. Nobody owns the republish. | Governance |
| C2 | `merge_for_training` drops the Tier-2 tag; `data_quality` then tiers bridge links **T1**, defeating D3's file-boundary rationale after one function call | Red Teamer |
| C3 | No quarantine test for `hub_links_curated.jsonl`; 20+ call sites would inherit a breach | Red Teamer |
| C4 | The packet guard checks column *names*, not data. `cre_hierarchy.json` carries 51 hubs with model-proposed `related_hub_ids` | Red Teamer |
| C5 | 78 vs 83 AI-hub counts from two different link sources, both committed as measured | Governance |
| C6 | `hub_rep_format="path+name+standards"` puts NIST 800-53 section names into AI hub text at eval time — a real bridge-reaches-evaluation path | Red Teamer |
| C7 | `gate_decision()` implements `P(δ≤0.10) < 0.025`, not §3's `< 0.05`, and reports `P(δ≤0)` | Data Scientist |
| C8 | The order-independence fix landed in `gate_rule_candidates` only; `audit_mechanism_probe` still threads one RNG at 10,000 | Data Scientist |
| C9 | Round-1 items E9–E14 unstarted: echo key collision, multi-label covariate, dead guardrail path, `results/bridge/PROVENANCE.md`, the κ claim, `_require_secure_cloud` call-site coverage | Governance |
| C10 | 6 dependabot advisories on `main` (4 high, 2 moderate) | — |

---

## D. The pattern, stated plainly

Three rounds, and each has found errors in the previous round's fixes:

| round | found | introduced |
|---|---|---|
| 1 | 9 confirmed defects in the analysis | — |
| Phase R | fixed 6 | 4 new, 3 inside the corrections |
| checkpoint 1 | found those 4 + 2 fatal design errors | *(pending)* |

The defects are not random. They are overwhelmingly **diagnostics that flatter
the design** and **prose invariants that no test holds**. A fix gets written to
demonstrate a repair rather than to test it, so nobody asks what the new
instrument reports when the answer is already known. Measuring each new
diagnostic under a known-zero condition would have caught all four.

Every perspective independently reached the same structural conclusion:
`.github/CODEOWNERS` is one name, all four PRs carry zero reviews, and the
premortems are commissioned, run and adjudicated by the same party. Six
perspectives with clean contexts is a good instrument and it is not
independence.

**The cheapest external check, costed by Governance at ~2 hours:** hand one
outside reader three numbers and no context — Gate 1's 78/55/23 against the
20-hub packet, Gate 2's 0.15 τ threshold against the committed 0.3702→0.0000,
and the 78-vs-83 hub counts. All three fall out without domain knowledge, and
two of them were this checkpoint's most expensive findings.
