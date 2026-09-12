# Phase 2C Gate 2 — results

**Run 2026-09-11. Four arms, one pod, 196 minutes, ~$12.**

**Verdict: NO VERDICT.** Not a PASS and not a FAIL. The noise-floor guard fired:
a contrast whose true effect is zero by construction produced a larger
difference-in-differences than the treatment did. The instrument could not
resolve an effect of this size, and that is a pre-declared outcome
(`docs/phase2c-gate2-plan.md` §5), not an interpretation reached afterwards.

---

## 1. The arms

All four hold out the same eight AI frameworks, score the same 74 items with one
model each, and differ from one another in exactly one thing.

| arm | corpus | seed | pairs | hit@1 | items correct |
|---|---|---|---|---|---|
| **A0** | none | 42 | 4,053 | 0.1486 | 11 / 74 |
| **A1** | round-2 union, conf ≥ 2 | 42 | 4,177 | 0.1216 | 9 |
| **A0R** | placebo, hub-matched | 42 | 4,177 | 0.0676 | 5 |
| **A0P** | none | 43 | 4,053 | 0.2162 | 16 |

Read the first and last rows together before anything else. **A0 and A0P differ
only in the random seed, and they are five items apart** — 11 against 16, a 45%
relative swing on a bridge-free configuration where the true difference is zero.

## 2. The three contrasts

| contrast | DiD | 95% CI | exposed (n=27) | unexposed (n=47) | pooled |
|---|---|---|---|---|---|
| **PRIMARY** A1 − A0 | **+0.1009** | [−0.0426, +0.2600] | +0.0370 | −0.0638 | −0.0270 |
| **NOISE** A0P − A0 | −0.1064 | [−0.2553, +0.0473] | +0.0000 | +0.1064 | +0.0676 |
| **SECONDARY** A1 − A0R | +0.1481 | [−0.0055, +0.3121] | +0.1481 | +0.0000 | +0.0541 |

## 3. The outcome table, applied

`docs/phase2c-gate2-plan.md` §5, decided before the run:

| check | result | |
|---|---|---|
| noise-floor guard: `\|DiD(A0′,A0)\| ≥ DiD(A1,A0)` | 0.1064 ≥ 0.1009 | **FIRES** |
| negative control: `delta_BIML ≥ delta_exposed` | −0.0588 < +0.0370 | passes |
| primary: `ci_low > 0` | −0.0426 | fails |
| secondary: `ci_low > 0` | −0.0055 | fails |

The noise-floor guard takes precedence, so the round returns **NO VERDICT**.

Reproduce:

```bash
python -m scripts.analysis.gate2_delta \
  --comparator results/phase1b/gate2_A0 --treatment results/phase1b/gate2_A1 \
  --strata results/phase2c/gate2_strata.json --label "A1 vs A0 (PRIMARY)"
```

---

## 4. What this does and does not establish

**It does not establish that the bridge corpus fails to help.** A FAIL would
have been a claim about the corpus. This is a claim about the *measurement*: at
74 items with 27 exposed, the seed alone moves the statistic as far as the
treatment does, so the two cannot be told apart. Reporting this as "bridges
don't work" would be exactly the error the noise-floor arm was added to prevent.

**It does not establish that the corpus helps either.** The primary DiD is
+0.1009, positive and in the predicted direction — and that number is not
usable, for the same reason. A positive point estimate inside a noise band is
not evidence.

**It does establish, with a measurement rather than an argument, that a
single-seed arm comparison on this evaluation cannot answer the question.** That
is a real finding and it is the one this round bought. The premortem predicted
it from two committed same-arm runs (19.1% per-item discordance); the dedicated
arm confirms it at 0.1064 on the gate's own statistic.

### The one number worth looking at again

The secondary is the most suggestive result in the round and it is still not a
verdict. `A1 − A0R` holds the supervision *volume* constant — same count, same
source framework, same hubs at the same multiplicities — and varies only which
control the annotators judged to belong to each hub. On the exposed stratum the
treatment beat the placebo by **+0.1481 (4 items of 27)**, and on the unexposed
stratum the two were **exactly identical (+0.0000)**, which is what a real
localised effect looks like rather than drift.

Its interval is [−0.0055, +0.3121]. The lower bound misses zero by 0.0055. Under
the outcome table a secondary cannot rescue a failed primary and the noise-floor
guard supersedes both, so this is **descriptive and nothing more**. It is
recorded because it is the most informative thing the round produced, not
because it licenses a claim.

### The DiD earned its place

The pooled `A1 − A0` delta is **−0.0270**: negative. The DiD on the same data is
**+0.1009**: positive. They disagree in sign because the unexposed stratum fell
further than the exposed one, which is precisely the global drift the
difference-in-differences exists to cancel and the pooled figure cannot see.
Had the plan kept the pooled contrast as its primary, this round would have
reported a negative delta and called it evidence against the corpus.

---

## 5. What may and may not be reported

**Report:** NO VERDICT, with the noise floor (0.1064) beside the primary DiD
(+0.1009) so the reason is visible. Report the arm table — the 11-vs-16 seed
swing is the finding.

**Do not report** the primary DiD, the pooled delta, or the secondary as a
result about the bridge corpus. None of them is separable from seed variance at
this n.

**Do not report** anything here as bearing on the `NONE` rate, the link rate, or
how connected the two hub regions are. `docs/phase2c-results.md` §4 governs
those and is not superseded by any number in this document.

**The corpus ships regardless**, and did — `results/phase2c/phase2c_bridge_corpus.json`
was committed before the first pod, which is the only ordering under which that
sentence means anything.

---

## 6. What would answer the question

In rough order of value per pound:

1. **Multiple seeds per arm — at least 3, ideally 5.** The estimand becomes the
   seed-mean delta and the interval carries both seed and item variance. This
   round shows a single-seed comparison is not an instrument here, and the fix
   is not subtle. At the measured cost of **$2.45 per arm** (49 minutes on a
   SECURE H100), five seeds across A0/A1/A0R is 15 arms for ~$37 — still less
   than the pre-registration's estimate for *one*.
2. **More exposure.** 27 of 74 items is the binding constraint on power. That
   means annotating a framework whose gold hubs the corpus reaches, not
   re-annotating NIST 800-53.
3. **Fix the sampler's AI upweighting**, which is currently inert for exactly
   this experiment: `is_ai` keys on the SOURCE framework, and every bridge link
   names a traditional standard, so under the strict firewall `n_ai == 0` and
   the branch-balance interleave degenerates to a plain shuffle. The one lever
   aimed at the AI region is switched off precisely when it is needed.

**What is not worth doing:** a third annotation round. Nothing in this result
implicates the corpus. The instrument is what failed.

---

## 7. Cost, recorded against the estimate

Amendment 1 budgeted "four retrains at ~$160", inheriting the pre-registration's
"one retrain, ~$40". The measured figure is **~$12 for all four** — 49 minutes
per arm on a SECURE H100. The estimate was high by more than an order of
magnitude, which matters because it is the number that made a four-arm design
look expensive enough to argue about. Corrected in Amendment 1 §1.4.

Two earlier attempts failed and are recorded rather than quietly re-run:

- **Attempt 1** died instantly. `POD_RSYNC_EXCLUDES` excludes `results/`
  wholesale — correctly — so `/workspace/tract/results` did not exist and the
  arm's log `tee` could not open. It trained 44 minutes and evaluated nothing.
  Also exposed a worse latent bug: a shell pipeline returns `tee`'s exit status,
  so the mirror case (training dies, `tee` exits 0) would have reported SUCCESS
  and collected nothing. `set -o pipefail` was the more valuable half of the fix.
- **Attempt 2** got A0 home and lost A1 at 95% of training to
  `OSError: [Errno 5]` on `/tmp`. That is the 20GB *container* disk, not the
  50GB volume; RunPod's overlay reports exhaustion as EIO rather than ENOSPC, so
  it presents as a disk failing rather than a disk filling. Fixed by moving
  `TMPDIR` to the volume, enlarging the container disk, and clearing each arm's
  checkpoints once it is safely collected.

Neither attempt left a pod billing. The `try/finally` teardown added on this
branch fired both times; before it, the same crashes would have stranded an
H100 the reaper could not see.
