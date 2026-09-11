# Phase 2C Gate 2 — execution plan (UNDER REVIEW, not yet binding)

**Deliverable, as stated by the owner 2026-09-11: the bridge corpus, and
whether it helps the model.** Not the `NONE` rate, not a claim about how
connected the two domains are. That narrows what this plan has to establish.

This document is the thing an adversarial premortem is being run against.
Nothing here is binding until that review closes.

---

## 1. What is already decided and not reopened

- **Gate 1 passed**, twice, on 28 and 31 de-orphaned hubs against a threshold of
  23. Stage 2 funding is not in question.
- **No round 3.** A third annotation round gates nothing the deliverable needs,
  and costs volunteer time these two have already given twice.
- **Gate 2's criterion** is `docs/phase2c-preregistration.md` §3, restated
  2026-09-04: retrain under the strict all-AI firewall, score on **ENISA +
  BIML** (50 items, 32 AI-only gold hubs, no test-split draw spent), PASS iff a
  fold-stratified paired bootstrap gives `ci_low > 0` against a **bridge-free**
  arm, not against zero-shot.

## 2. The corpus choice, declared before running

Round 1 is **strictly nested inside** round 2: every round-1 link appears in
round 2 with an identical hub, and no link was ever withdrawn. Verified.

| | links (counting, confidence ≥ 2) | κ |
|---|---|---|
| round 1 | 80 | 0.597 |
| round 2 | 173 | 0.508 |

Round 2 adds only the hedged band — **zero of the 106 additions is confidence
3** — and agreement fell as it grew.

**Declared now, before any retrain:**

- **Primary: round 1.** The higher-agreement subset and the conservative claim.
- **Secondary: round 2**, pre-declared, not chosen after seeing results.

Picking whichever wins afterwards is the outcome-switching this project has
withdrawn three headlines for.

## 3. Arms

| arm | corpus | purpose |
|---|---|---|
| A0 | none | the comparator |
| A1 | round 1 (80 links) | primary |
| A2 | round 2 (173 links) | secondary; the nesting makes A2 − A1 the marginal band's contribution |

Three retrains. The pre-registration says "one retrain, ~$40"; this is three,
and that is a deliberate departure to be justified or cut during review.

## 4. Known blockers, carried in rather than discovered

- **The strict all-AI firewall has no implementation.** `run_single_fold` takes
  `held_out_framework: str`, singular; `run_fold.py` exposes `--framework` with
  one value. Holding out eight frameworks at once does not exist.
- **Power is low.** At n=50 with the observed discordant rate, MDE at 80% power
  is δ ≈ 0.21; power at δ=0.10 is 0.22. A FAIL is weak evidence of absence and
  the pre-registration already says so.
- **ENISA and BIML are AI frameworks in the eval roster sense but have never
  been scored as folds.** Nothing has run against them.
- **All inference runs on RunPod**, never locally. Every arm is pod work.

## 5. Sequence

1. Implement multi-framework holdout.
2. Dry-run the fold construction locally (no model) to confirm 50 items and 32
   AI gold hubs.
3. Provision one pod, run A0, A1, A2 in sequence on it.
4. Score, report, tear down.

## 6. What will be reported

The delta and its interval for A1 (primary) and A2 (secondary), the power the
design had, and a statement that a null at this n is weak. The corpus and its
provenance ship regardless of the verdict.
