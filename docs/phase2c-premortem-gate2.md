# Adversarial premortem — Phase 2C Gate 2 execution plan

**Artifact:** `docs/phase2c-gate2-plan.md` @ `66d9287` (drafted at `21a5e81`)
**Date:** 2026-09-11
**Lens, set by the owner:** *the deliverable is the bridge corpus and whether it
helps the model.* Not the `NONE` rate. Not a claim about how connected the two
domains are.
**Premise:** it is six months from now, Gate 2 ran exactly as the plan said, and
the result was worthless, misleading, or cost far more than the budgeted ~$40.

Six perspectives, each in a clean context, read the artifacts independently and
returned evidence-anchored findings. No perspective was skipped. This document
is the round report; the remediation it produced is in the rewritten plan and in
Amendment 1 to the pre-registration.

---

## The one-paragraph answer

The plan would have failed for a reason that had nothing to do with the pod, the
firewall, or the statistics as such. **The treatment cannot reach the
measurement.** Under the strict all-AI firewall the bridge corpus supervises 18
of the 50 evaluation items, all 18 of them ENISA — the BIML stratum, 34% of the
denominator, has *zero* exposure and can contribute only noise. The comparator
it is measured against has zero training positives for any of the 56 scored
hubs, so a PASS would have licensed only "some supervision beats none." And
run-to-run drift between two fine-tunes, measured on this repository's own
committed artifacts at 19.1% per-item discordance, is the same magnitude as the
effect being looked for, so the reported interval would not have contained the
dominant source of variance. Every one of those numbers was obtainable from
committed artifacts in under an hour, before renting anything.

---

## Findings that survived cross-attack

Merged across perspectives; duplicates credited to the strongest anchor. Prior →
posterior with the evidence that moved it.

### C1 — The treatment reaches 18 of 50 eval items, and none of them are BIML
**Impact: Critical · Prior: Likely → Posterior: Confirmed (measured three times, reconciled)**

Raised independently by Red Teamer (15/50), ML Engineer (16/50) and Data
Scientist (16/50, all ENISA). The three disagreed because they counted different
denominators. Re-derived directly:

```
eval: 50 items (ENISA 33, BIML 17); 32 ground_truth hubs; 56 valid_hub_ids (SCORED)
round 1 conf>=2: 16/50 exposed   by framework {'ENISA': 16}
round 2 conf>=2: 18/50 exposed   by framework {'ENISA': 18}
round 2 all-conf: 22/50          by framework {'ENISA': 19, 'BIML': 3}
```

`evaluate.py:355-365` credits hit@1 against `valid_hub_ids`, so 56 is the scored
set and 16/18 is the right exposure count. **BIML reaches 3 items only through
confidence-1 links** that `docs/phase2c-preregistration.md:78` classifies as
"data, not evidence."

*Cross-attack:* the ML Engineer's count is the correct one and supersedes the Red
Teamer's; the Data Scientist's framework breakdown is the finding that matters
and neither of the others reported it. The plan's own "32 AI-only gold hubs" is
neither number.

**Failure mode.** A1−A0 returns +0.04 [−0.06, +0.14], FAIL, written up as
"bridges did not help." What happened is that 32 of 50 items were structurally
incapable of moving and diluted a real effect on the other 18 by a factor of 2.8.

### C2 — The comparator is degenerate: A0 has zero positives for all 56 scored hubs
**Impact: Critical · Prior: Plausible → Posterior: Confirmed**

ML Engineer, measured:

```
strict ALL-EIGHT firewall:  4,083 links,  0/56 ENISA+BIML gold hubs supervised
five-AI firewall (current): 4,208 links, 56/56
ENISA-only holdout:         4,337 links, 52/56
strict + bridge r1:         4,163 links, 18/56
```

The AI and traditional hub regions are disjoint by construction, so excluding all
eight AI frameworks removes every positive pointing into the AI region. A0 has
never been pushed toward any answer it is scored on.

*Cross-attack:* checked whether A0 is actively *suppressed* rather than merely
starved — it is not. Only 1 of 56 gold hubs ever appears as a hard negative,
8 slots total, because `mine_hard_negatives` walks hierarchy siblings and the AI
region is isolated. Starved, not suppressed. That keeps the finding at Critical
rather than Fatal.

**Failure mode.** A PASS is near-tautological and does not survive a skeptical
reader; a FAIL means supervision aimed directly at the label set failed to move
four items. Neither outcome answers the deliverable's question.

### C3 — Run-to-run drift is the same magnitude as the effect, and the interval does not contain it
**Impact: Critical · Prior: Likely → Posterior: Confirmed on estimand, Plausible on magnitude**

`paired_bootstrap_delta` (`evaluate.py:170-219`) resamples *items* within folds
and nothing else. Its variance is item-sampling variance conditional on the two
arrays handed to it. Two trained arms are two draws from a training procedure,
and that draw-level variance is absent from the interval.

Data Scientist measured it on two committed same-arm runs (`c2_A1_prose_sw_bge`
vs `c2r_A1_prose_sw_bge`), whose zero-shot indicators are byte-identical, so
every flip is attributable to the trained model:

| fold | n | hit@1 run 1 → run 2 | per-item discordance |
|---|---|---|---|
| ASVS | 277 | 0.2816 → 0.0939 | 23.1% |
| CWE | 246 | 0.2561 → 0.2846 | 19.9% |
| NIST 800-53 | 300 | 0.1667 → 0.2267 | 14.7% |
| **pooled** | **823** | **−3.3pp** | **19.1%** |

Null calibration at the measured fold-drift SD of 13.5pp: **P(false PASS) =
0.134**, against a nominal 0.025.

*Cross-attack:* the magnitude estimate is confounded — that pair also grew its
training pool ~11%, a *larger* perturbation than A1's +2%. Posterior on the
estimand being mis-specified is Confirmed (it is a property of the code);
posterior on σ ≈ 0.135 specifically is held at Plausible and the remediation does
not depend on the exact value, only on it being non-negligible.

**Failure mode.** The dangerous outcome is the PASS, not the FAIL. `ci_low` just
clears zero, Stage 2 funds, the corpus ships as "shown to help," and a different
seed six months later returns +0.02. That is the fourth withdrawn headline, for
Campaign 2's exact reason.

### C4 — Gate 2's criterion has no implementation, and the comparison it forbids has one, under the key `gate`
**Impact: Critical · Prior: Likely → Posterior: Confirmed**

Raised by Red Teamer, ML Engineer, Data Scientist and Governance independently.
`gate_decision` (`orchestrate.py:839-892`) hardwires `baseline =
r["zero_shot"]["hit1_indicators"]` and raises without it. Gate 2's criterion is
A1−A0. No function anywhere pairs two training arms. Meanwhile every arm writes
an `aggregate_metrics.json` carrying `gate.preregistered_pass`,
`gate.ci_low_pass`, `gate.familywise_pass` — three fields answering a different
question, at threshold 0.10, against the comparator
`docs/phase2c-preregistration.md:129-139` retired in writing, and which 73% of
*bridge-free* arms already satisfy.

*Cross-attack:* four perspectives converging is correlated evidence, not four
confirmations — they all read the same function. The band is set by the code,
not by the count.

### C5 — The two corpora the plan names do not exist as files, the confidence floor never reaches training, and the arm is 86 rows over 61 edges
**Impact: Critical · Prior: Likely → Posterior: Confirmed**

`load_bridge_links` requires `path.is_file()` (`tract/bridge/links.py:72`); the
rounds on disk are per-annotator pairs. The Q3 confidence floor lives only in
`scripts/analysis/gate1_report.py:286-290`; `bridge_training_records`
(`links.py:121-131`) drops the confidence field entirely. Re-derived:

| corpus | plan says | rows ingested | conf-1 ingested | distinct edges | dup (both annotators) |
|---|---|---|---|---|---|
| round 1 | "80 links" | **86** | 6 | **61** | 25 |
| round 2 | "173 links" | **192** | 19 | **132** | 60 |

The 25 duplicated edges are exactly the both-annotators-agreed subset, so
duplication silently double-weights agreement — defensible as a design, but
undeclared, and the duplication *rate differs between rounds*, which lands
directly on the A2−A1 contrast the plan called "the marginal band's
contribution."

### C6 — The plan's primary-corpus declaration rests on a contaminated statistic and is contradicted by the clean one
**Impact: Critical · Prior: Plausible → Posterior: Confirmed (measured)**

The plan picked round 1 as primary on κ 0.597 vs 0.508 — the **link/no-link** κ,
which `docs/phase2c-results.md:32-52` establishes the instruction wording moved
(106 `NONE`→link, 0 reversals, p<0.0001). On **hub agreement**, the decision that
determines whether a link is *correct*, Governance measured round 2 equal or
better on more than twice the n: **0.8846 (n=26) vs 0.8909 (n=55)**.

And the plan's stated ground — "round 2 adds only the hedged band, zero of the
106 additions is confidence 3" — is true of *additions* and blind to *retained*
links. The transition matrix on the 86 carried links:

```
2 -> 3:  7        3 -> 2:  1
confidence-3 population:  round 1: 16  ->  round 2: 22   (+37.5%)
```

Seven links the annotators called "defensible" under the biased framing became
"confident" under the neutral one. Round 1 did not contain every confident
mapping.

*Cross-attack:* this is the premortem finding a claim made by the document under
review, using the document under review's own governing rule
(`phase2c-results.md:117-120`, "do not report κ without saying which
denominator"). The rule was four days old and the plan violated it anyway.

### C7 — Three arms, no multiplicity control, no outcome table
**Impact: High · Prior: Likely → Posterior: Confirmed**

Family-wise error under the global null, simulated: 0.069 over three contrasts at
zero drift, **0.163 at σ=0.10**. The plan never states an alpha, never mentions
`n_configurations` (which `gate_decision` takes and `runpod_parallel` makes a
required no-default flag precisely because a wrong value "produces an interval
that looks correct"), and never says what verdict *primary FAIL / secondary PASS*
produces — a likely cell at the stated power. `CAMPAIGN3.md:121-130` requires an
outcome table with "every combination decided in advance." The plan has the
pre-declaration of which arm is primary but not of which result is the verdict.

### C8 — A2 − A1 is 91% a re-weighting manipulation and buys 2 eval items
**Impact: High · Prior: Plausible → Posterior: Confirmed**

Of the 106 round-2 additions, 91 are counting links; **83 land on hubs round 1
already covered and 8 on new hubs**. New hubs: exactly 3, of which 2 are in the
scored set. **Eval items newly supervised by the marginal band alone: 2 of 50.**
So the contrast confounds three hubs of coverage with a 2.3× density increase on
already-covered hubs, while κ fell — i.e. noisier labels on the same hubs — and
it carries the *sum* of two independent training-draw variances, not a
difference.

### C9 — `A2 − A1` is an unpre-registered route to the conclusion the results document forbids
**Impact: High · Prior: Plausible → Posterior: Likely**

At n=50 and this power, A2−A1 is a guaranteed null whose only readable meaning is
"round 2's additions were noise" → "round 1's answer set was the real one" →
"the round-1 `NONE` rate was about right" → "the domains are disconnected." That
is the forbidden claim, reconstructed through the model instead of through the
annotation rate. The plan leans that way in prose already, calling round 1 "the
conservative claim" and relabelling the project's own counting tier as "hedged."

### C10 — The volunteers' promises have no implementation, and training is the irreversible step
**Impact: High · Prior: Plausible → Posterior: Confirmed**

The handbook promises publication with credit, upstream proposal to OpenCRE, and
**"a right to withdraw their contribution before publication."** `grep -rn -i
"withdraw" tract/ scripts/` returns no deletion path. The corpora are gitignored
with no publication command. `docs/phase2c-preregistration.md:180` conditions
Stage 2 — which is where publication lives — on both gates passing.

Round 1 is nested in round 2, so one withdrawal changes the composition of every
scored arm. A dataset can be re-cut; a checkpoint cannot. **This makes closing
the withdrawal window a precondition of the retrain, not a cleanup item**, and
the promise is the project's to keep, not mine to waive.

Measured, on the plan as written: declaring round 1 primary strands 82 of
vol-01's 122 links (67%) and 24 of vol-02's 70 (34%) outside the headline — and
vol-01 is the annotator whose round 2 most resembles genuine re-judgement.

### C11 — Cost controls absent on the path the plan's shape selects
**Impact: High · Prior: Likely → Posterior: Confirmed**

"Provision one pod, run in sequence" is `scripts/phase1c/runpod_retrain.py`'s
shape. It has no `try/finally` (`grep -n finally` returns nothing), no
`max_usd_per_hour` (`:148`), no wall clock, no retry ladder — and its pod name
`tract-p1c-retrain` is outside the reaper's sweep, which covers only
`tract-p1b-fold*` / `tract-p1b-val-fold*`. Worse, `_is_orchestrator_argv` matches
only `runpod_parallel`, so a live retrain reads as *no orchestrator*,
`running_pod_count()` returns 0, and the guard **disarms after 3 quiet checks
(~6h)** while the pod trains. At the last recorded SECURE H100 rate that is
$79/day. Three arms is three independent draws on it.

Independently: no reaper timer is armed on this machine right now
(`systemctl --user list-timers 'tract-reaper*'` → none listed).

### C12 — The strict firewall has no implementation, and a wrong-firewall shortcut runs today and looks identical
**Impact: Critical · Prior: Plausible → Posterior: Confirmed**

`run_single_fold` takes `held_out_framework: str`. `validation_frameworks()`
returns everything minus the **five**-name AI roster — which *includes* ENISA,
ETSI and BIML. So `run_fold.py --split validation --framework ENISA` passes every
guard today, holds out ENISA alone, leaves the other seven AI frameworks in
training (**52 of 56 gold hubs supervised**), and writes a `fold_result.json`
indistinguishable in shape from a firewalled one. Nothing in the record names the
firewall.

ML Engineer costed the real change at **one engineer-day** for the firewall and
two to a runnable Gate 2 — six signatures, 18 edit sites, three out-of-tree
callers that `mypy --strict` will catch (and it is a genuine net here: a set
silently passed to a `str` parameter makes `standard_name == {...}` permanently
False, training on everything, scoring ~0.9, with no guard noticing).

### C13 — Arms are not separable in the artifacts
**Impact: High · Prior: Likely → Posterior: Confirmed**

`bridge_links_path` is absent from `ARM_DEFINING_KEYS`, so `load_fold_results`
would aggregate an A0 fold beside an A1 fold without complaint — saved only
accidentally by the `inputs` digest check. `_campaign_label` ignores bridge, so
all three arms share a WandB run id and an output directory
(`results/phase1b/<config-name>`), and `orchestrate.py:293` `mkdir(exist_ok=True)`
+ `atomic_write_json` means A1 overwrites A0. The launch lock is keyed on
framework alone (`runpod_parallel.py:1403`), so arm 2 on the same pod refuses
with "stale launch lock."

And `staleness.py:73` maps `bridge_links_sha256` to a **constant** path
(`data/training/hub_links_bridge.jsonl`) that does not exist — `_digest()` returns
`None`, `check_result` skips the comparison, and the bridge staleness check is
today a no-op.

### C14 — `_build_fold_index_matrix` order dependence is open and lands on a boundary test
**Impact: High · Prior: Plausible → Posterior: Confirmed**

`campaign3-premortem-round3.md:402` still lists C1 as open: per-fold indices are
drawn sequentially from one RNG, so every fold's draw depends on the sizes and
order of the folds before it, with measured spread 0.0162 in p. Gate 2's
criterion is `ci_low > 0` — a *boundary* test — which makes that spread decisive
in a way it was not for a `P(δ≤0.10)=0.535` verdict. The same table's B6
(`preregistered_pass` uncorrected for selection) becomes live with three arms.

### C15 — The declaration that makes this a pre-registration was untracked
**Impact: High · Prior: Likely → Posterior: Confirmed, remediated during the review**

At the time of the review `git status` showed `?? docs/phase2c-gate2-plan.md`,
`grep -rln "phase2c-gate2-plan"` returned nothing, and there was no
`PHASE2C_GATE2_*` constant of any kind — against Gate 1, which has six named
constants and a test asserting each one's literal text appears in the
pre-registration. Editing the markdown after results existed would have left no
diff and failed no test. *Remediated mid-review at `66d9287`, before any pod.*

### C16 — Data handling: the guard is keyed on the wrong corpus and the exclude lists do not cover the annotator text
**Impact: High · Prior: Plausible → Posterior: Confirmed**

`require_secure_cloud()` returns True **only** when `"licensed" in
merged_corpus_path().parts`. The bridge corpora are gitignored, so a fresh clone
has no overlay, the guard returns False, COMMUNITY returns to the preference
list — while the operator hand-copies the annotator corpus in, because the run
cannot start without it. Neither rsync exclude list names any `data/training`
path, so both rounds and both sidecars ship, carrying `annotator_id` and verbatim
rationales. The retrain path additionally excludes only `results/phase0` and
`results/phase1b`, so `results/review/review_export.json` (Tier 3, quarantined)
and `results/ceiling_study/hub_reference.md` (400 LLM-written descriptions) go to
the pod.

*Cross-attack:* checked whether the pseudonym↔person key is anywhere in the tree.
It is not — `claudedocs/curation-package.md` contains no emails, names, or
mappings. That holds the impact at "pseudonymised judgements on a rented host"
rather than "identified volunteers," and the protection is currently accidental.

### C17 — ETSI cannot be added to the eval without a redaction fix
**Impact: High · Prior: — (found in cross-attack) → Posterior: Confirmed**

Not raised by any single perspective. The Data Scientist proposed adding ETSI to
raise exposure and argued correctly that the licensing constraint is on
*redistribution*, not scoring. The Security Architect never examined the eval set
because the plan did not name ETSI. Composing the two: ETSI is in
`RESTRICTED_FRAMEWORK_IDS`, `orchestrate.py:371` writes `"control_text":
item.control_text` into `predictions.json`, and `.gitignore:43` negates
`results/phase1b/**/*.json` back into tracking. Adding ETSI would commit ETSI
verbatim control statements to a CC0 repository.

Verified, and the fix is an improvement: `predictions.json` needs `control_text`
for exactly one purpose — asserting item alignment between two arms — and a
`control_text_sha256` serves that better and leaks nothing.

### C18 — The annotator hub sheet was built from the eval's own gold links
**Impact: Medium · Prior: Plausible → Posterior: Confirmed**

`build_bridge_packet.ai_hub_ids()` derives the 78-hub sheet from
`BRIDGE_AI_FRAMEWORK_IDS`, which includes `enisa` and `biml`. Those curated links
*are* the Gate 2 eval gold. No text leaked and no model output was shown — the
corpus is genuinely Tier 2 — but the candidate universe was constructed from the
evaluation labels, so the treatment's support set could not have missed the gold.
Open item C6 from checkpoint 2, still unremediated. A PASS licenses "bridges
aimed at this region help on this region," not "bridges generalize."

---

## Dropped ledger

| finding | band | reason |
|---|---|---|
| Secrets handled differently across three arms | Remote | One read-only HF token, written to piped stdin while only the un-augmented cmd is logged. Verified clean; the arm count changes nothing. |
| Supply-chain advisory blocks the run | Unlikely | All five `AUDIT_SUPPRESSIONS` expire 2026-11-17, 67 days out. Schedule risk only if the round slips past mid-November. |
| Bridge corpus licensing exposure | Unlikely | Both rounds are 100% `nist_800_53`, a US Government work not subject to domestic copyright, in none of the three restriction sets. |
| Packet leaked model output to annotators | Remote | `build_bridge_packet` emits hub id, name, hierarchy path, branch and empty answer columns. `related_hub_ids` (Tier 3) is never read. Tier 2 holds. |
| Hub firewall degenerates under 8-way holdout | Unlikely | Default hub text is `"{hierarchy_path} \| {hub_name}"`, CRE-native, so AI hub representations do not depend on the held-out frameworks. |
| Anchor truncation repeat of Campaign 3 | Unlikely | ENISA+BIML resolve 100% to prose, 0 fallbacks, 0 truncations at 512, median 384 chars. |
| Eval credits leak to the traditional region | Remote | All 56 `valid_hub_ids` are AI-only; 0 items admit a traditional hub. |

**Parked tail risk (Critical, below Plausible, not deleted):** a volunteer
requests withdrawal after training. Probability of the request is Unlikely
(~0.10–0.20); probability it could not be honoured if made is ~1.0, because the
supervision is in the weights. **Trigger that raises it:** any annotator contact
about publication, credit, or attribution. Mitigated by making the withdrawal
window an explicit pre-retrain gate rather than by reducing the probability.

---

## Cross-attack log

- **Exposure count, three values (15/16/16).** Reconciled by re-derivation: 16 is
  correct at the counting floor because `evaluate.py` credits `valid_hub_ids`.
  The Red Teamer's 15 counted `ground_truth_hub_id` only. The Data Scientist's
  framework breakdown — all ENISA, BIML zero — is the load-bearing part and only
  one perspective reported it.
- **"A0 is starved" vs "A0 is suppressed."** Tested the stronger claim; only 1 of
  56 gold hubs appears as a hard negative, 8 slots. Held at starved.
- **Drift magnitude.** The Data Scientist's own caveat (the comparison pair grew
  its training pool ~11%) was accepted and the posterior split: estimand
  mis-specification Confirmed, σ≈0.135 held at Plausible. The remediation does
  not depend on the value.
- **Four perspectives on C4.** Treated as correlated, not confirmatory — all four
  read the same function. No band inflation.
- **ETSI (C17).** The Data Scientist's fix and the Security Architect's trust
  boundary were both correct and jointly produced a finding neither reported.
- **Round 1 vs round 2 as primary.** Governance argued round 2 on hub agreement
  and confidence drift; the Data Scientist independently showed the two corpora
  differ by only 2 eval items of exposure. Together these void the statistical
  case for either and hand the decision to governance — which favours the union.

---

## Convergence

Round 1 surfaced findings at Critical and High across all six perspectives, so
the stop signal was not met on count. It was met on **kind**: every surviving
Critical finding traces to one of three roots — the treatment cannot reach the
measurement (C1, C2, C8, C18), the comparison is unimplemented or unseparable
(C4, C5, C12, C13, C14), or a commitment has no binding home (C10, C15, C16). A
second round against the same artifact would re-derive the same roots, because
the artifact is being replaced rather than patched. The premortem therefore
converges here, and the rewritten plan is the object a future round should
attack.

---

## Remediation, ordered by expected cost reduction

**Gate-blocking, before any pod:**

1. Rebuild the design so the treatment can reach the measurement — exposure
   stratification as the pre-registered primary, ETSI added, BIML declared a
   negative control (C1, C2, C8).
2. Close the withdrawal window in writing, and commit the publication artifact
   *before* the run so a FAIL cannot shelve it (C10).
3. Amendment 1 to the pre-registration, dated, recording what was known when
   written (C6, C7, C9, C15).
4. `scripts/build_gate2_corpus.py` — deterministic merge, confidence floor,
   declared dedup rule, committed manifest (C5).
5. `scripts/analysis/gate2_delta.py` — arm-vs-arm DiD and pooled, with item
   alignment asserted (C4).
6. Multi-framework firewall, and refuse the wrong-firewall shortcut by name
   (C12).
7. Arm separability: `ARM_DEFINING_KEYS`, campaign label, staleness resolution,
   launch lock key (C13).
8. `_build_fold_index_matrix` per-fold seeding (C14).
9. `predictions.json` redaction for restricted frameworks (C17).
10. Cost and data-handling controls: `try/finally`, price ceiling, reaper
    coverage, `require_secure_cloud` over the bridge corpus, rsync excludes
    (C11, C16).

**Disclosed, not fixed this round:** C18 (the hub sheet was built from the eval's
gold links). Unfixable without re-annotating against a sheet rebuilt with the
eval frameworks excluded. Recorded as a limit on what a PASS licenses, and as the
design for any future round.

---

## Residual risk after remediation

- **Power remains below convention.** Exposure stratification and ETSI raise the
  reachable signal substantially, but the design will not reach 80% power for
  plausible effects. The honest position is that Gate 2 reports an estimate with
  its interval, and the pre-registration must say what a null does and does not
  license.
- **C18 stands.** A PASS licenses a claim about this hub region, not about
  bridging in general.
- **k=2 or k=3 folds.** The interval generalizes to new items, not to new
  frameworks; `gate_power_simulation.py` already argues τ is unidentified even at
  k=5.
- **The premortem is still self-commissioned.** Checkpoint 1 costed the cheapest
  external check at ~2 hours and it has not been done.
