# Adversarial premortem, round 1 — Amendment 2 and the curation decision

Artifacts under review: `docs/campaign3-amendment2-draft.md`,
`docs/campaign3-audit-mechanism.md`, `results/phase1b/CAMPAIGN3.md`.
Six perspectives, independent contexts, run 2026-08-30. Orchestrator
cross-attacked and calibrated; every posterior below names what moved it.

**Verdict: Amendment 2 is not fit to be pre-registered, and neither the
curation spend nor the roster rotation should proceed on the evidence in
`campaign3-audit-mechanism.md`. Both documents are the orchestrator's own work
from earlier the same day; the findings below are overwhelmingly against them.**

---

## A. Confirmed by independent re-measurement (Very likely)

Each was re-derived by the orchestrator after the perspective reported it.

### A1 — A false claim is live on the published HuggingFace model card
**Impact: Critical.** `tract/publish/model_card.py:455` publishes
`| Naturally bridged (both) | 60 | "Data poisoning" (linked by both ATLAS and CWE) |`.
Measured: **MITRE ATLAS hubs ∩ CWE hubs = 0**. The worked example does not exist.

The 60 arises because `BRIDGE_AI_FRAMEWORK_IDS` (`tract/config.py:716`) defines AI
as the **five rotating** frameworks, so ENISA/ETSI/BIML are counted as
"traditional." The traditional side of all 57 bridged hubs I measure comes
entirely from those three (ENISA 51, ETSI 28, BIML 11; zero from any other
framework). Under the eight-framework definition every current analysis uses,
**bridged = 0**, reproducing `PRD.md:58`.

*Prior Critical/High (Governance) → posterior Very likely: orchestrator
reproduced the intersection directly.*

### A2 — The degree mechanism was an arithmetic artifact before anyone refuted it
**Impact: Critical.** The claim carried in three documents — *"49 of 56 move the
gold label to a more-linked hub, median degree 3.0 → 7.5"* — counts degree on the
**post-audit** graph. 56 corrections collapse onto 26 destination hubs, so each
destination is credited with the corrections that landed on it.

| degree basis | median old → new | moved to higher |
|---|---|---|
| post-audit (as used) | 3.0 → 7.5 | 49/56 |
| **pre-audit (correct)** | **4.0 → 3.0** | **20/56** |

The direction reverses. `audit_stratified_delta.py:241` disclosed the
contamination but priced it at "+1 per correction"; the true factor is ~2.15
destinations-per-hub plus source drainage.

Consequence: `campaign3-audit-mechanism.md` spends a probe, three hypothesis
tests and a documentation pass refuting a claim that one line of arithmetic
dissolves — and then installs a replacement reading ("judgement entering the
gold labels") on H3's `P(≤0)=0.064`, which the probe's own
`established_at_alpha` flag marks **False**. All three tests are below the
project's own α (H1 P=0.459, H2 P=0.581, H3 P=0.064), yet §4 states
*"Overturned: the explanation"* flatly.

*Prior Very likely (Red Teamer) → posterior Very likely: exact recomputation.*

### A3 — Amendment 2 §1 is contradicted by its own table
**Impact: High.** §1 states *"this makes the gate harder to pass — that is the
point."* The table directly beneath: `P(δ ≤ 0.10)` fixed **0.535** →
random-effects **0.479**. The gate fires *below* 0.05, so the proposed
estimator moves the primary **0.056 closer to passing**. Widening an interval
does not tighten a one-sided threshold gate when the point estimate sits on the
threshold.

*Prior Very likely (Red Teamer) → posterior Very likely: arithmetic, in the
document.*

### A4 — The power simulation measures a different estimand than the gate
**Impact: Critical.** `gate_power_simulation.py` uses identical `n_per` for
every fold, so `drawn.mean(axis=(1,2))` is an **unweighted mean over
frameworks (macro)**. The gate's primary is the **item-weighted micro** average.
Measured on the untouched stratum:

| estimator | value |
|---|---|
| micro (the published primary) | **+0.1000** |
| macro (what the simulation computes) | **+0.2701** |

Difference **0.1701 = 1.7× the gate threshold**. Real fold weights are
0.573/0.273/0.100/0.036/0.018; the simulation assigns 0.200 each. §6e is
therefore power for a statistic nobody computes.

*Prior Very likely (Data Scientist) → posterior Very likely: orchestrator
reproduced both estimators.*

### A5 — τ is a single two-item fold, and the swept grid excludes the point estimate
**Impact: Critical.** Leave-one-fold-out on the shipped estimator:

| dropped fold | τ |
|---|---|
| OWASP Top10 for LLM (**n=2**) | **0.0000** |
| MITRE ATLAS | 0.3880 |
| NIST AI 100-2 | 0.4426 |
| OWASP AI Exchange | 0.4204 |
| OWASP Top10 for ML | 0.4446 |
| *(none — all five)* | 0.3702 |

The alternative to 0.3702 is **0.0000**, not the 0.0782 the document reports —
that figure is `0.023443 − 0.017321`, sitting on the truncation floor. And
`TAU_GRID` tops out at **0.20** while the document's own point estimate is
**0.3702**, so the branch that says *"do not fund; replace the instrument"* was
never simulated.

*Prior Very likely (Data Scientist) → posterior Very likely: LOFO recomputation.*

### A6 — Path C adds five hubs, not three independent clusters
**Impact: High.**

| | value |
|---|---|
| incumbent gold hubs | 73 |
| ENISA + BIML + ETSI gold hubs | 62 |
| overlap | **57 (91.9%)** |
| **genuinely new hubs** | **5** |

BIML 11/11 redundant (100%), ETSI 28/29 (96.6%), ENISA 51/56 (91.1%).
Jaccard(ENISA, OWASP AI Exchange) = **0.600**, higher than the highest
incumbent–incumbent pair (ATLAS/AIX = 0.522).

A cluster bootstrap treats frameworks as exchangeable draws. Adding three
clusters answering the same 57 hubs pushes **measured τ down and k up
simultaneously** — moving the design into §6e's "works" region on redundancy
rather than evidence. Combined with A3, both proposed changes loosen the gate
while the document claims the opposite.

*Prior Very likely (Red Teamer) → posterior Very likely: set arithmetic.*

### A7 — The new tests do not constrain the claims, and one pins a wrong number
**Impact: Critical.** Six of six core-logic mutations survive
`tests/test_gate_power_simulation.py` and `tests/test_audit_mechanism_probe.py`,
including **deleting framework resampling entirely** from
`cluster_bootstrap_pass` — the property the estimator exists for.
`test_between_framework_spread_can_sink_a_high_mean` asserts
`tight_pass >= spread_pass`, satisfied by **5 ≥ 5**, and passes identically with
cluster resampling removed (orchestrator-verified).

Separately, `gate_rule_candidates.make_contrast` threads one generator through
two `bootstrap_deltas` calls, so the reported CI depends on call order:

| | 97.5th percentile |
|---|---|
| shipped order | 0.45946 |
| swapped order | 0.43243 |
| **500,000-draw reference** | **0.43243** |

`tests/test_gate_rule_candidates.py:54` pins **0.4595** under a comment claiming
it comes from `campaign3-audit-mechanism.md` §6b — which, with CAMPAIGN3.md and
the script's own docstring, says **0.4324**. The same RNG defect was identified
and fixed in `audit_mechanism_probe.py` and then reintroduced here.

`tests/test_audit_mechanism_probe.py` also states that `build_rows` "is guarded
by an assertion… tested here." `build_rows` is never imported; coverage is zero.

*Prior High (ML Engineer) → posterior Very likely: orchestrator reproduced the
mutation survival and the 500k reference.*

### A8 — The spend bound does not survive a reboot, and is absent right now
**Impact: Critical.** `reaper_guard` is armed only as a **transient** systemd
unit (`systemd-run --user --collect --on-active=…`), which a reboot deletes.
Verified live: `systemctl --user list-timers 'tract-reaper*'` → 0 timers;
`/run/user/1000/tract-reaper/` → absent; no crontab; `~/.config/systemd/user/`
empty. The docstring's reassurance that `Linger=yes` "survives logout" does not
address power loss. The 2026-08-30 outage landed 72 minutes after teardown, so
nothing was lost; earlier and it would have been unbounded.

*Prior Critical (MLOps) → posterior Very likely: verified on this host.*

### A9 — `245/245` is a pair count
**Impact: Low, but diagnostic.** Distinct AI gold hubs = **78**; 245 is the sum
of per-fold counts. The 100% conclusion holds at 78/78; the denominator was
inflated 3.1× — the same defect class the document catalogues in others.

---

## B. Surviving, not independently re-measured (Likely / Plausible)

- **B1 (High, Likely) — R6 can never fire on any curation target.** §3 requires a
  framework containing both original and rewritten gold; all four targets have
  **zero** original CRE links. R6's "USABLE" verdict is also set by a hardcoded
  `BASELINE_MARGIN`; R4 at margin 0.15 clears the same bar (90% null permit).
  Its 94% specificity is measured against a null `split_null` builds
  composition-matched by construction — the one configuration curation
  guarantees will not occur. *This is the strongest argument for the (NEW)
  bridge option: bridge links land in frameworks that DO have original gold and
  therefore DO admit the control R6 needs.*
- **B2 (High, Likely) — the audit log's `exclusions` and `kept_weak` are read by
  no code.** The audit *excluded* an OWASP AI Exchange link outright
  (`547-824`, verdict `wrong`), so *"the audit never touched OWASP AI Exchange
  at all"* (§5) is false — and AIX is 57.3% of the primary. 5 kept-weak items sit
  inside the Tier-1 stratum despite having been inspected.
- **B3 (High, Likely) — rotating the roster is confounded with a 4× multi-label
  shift.** ~~Multi-label share: incumbents 8.8%, candidates 36.5% (ENISA alone
  51.5%). Single-label delta +0.1165 vs multi-label −0.1429;
  difference **P(≤0)=0.044** — better established than the H3 split the whole~~
  **PARTIALLY RETRACTED 2026-09-04 — see the correction at the end of this
  file. The composition figures hold; the delta split does not reproduce on any
  committed run and its sign reverses on four of six.** Original text struck
  through, not deleted, because remediation item 10 was written from it. The
  strikethrough continues to the end of the bullet: the H3 split the whole
  document rests on. §6b pre-registers "the delta will probably fall" and
  attributes it to supervision donors; label density predicts it equally well and
  the two are not separable after the fact.
- **B4 (High, Likely) — the frozen echo key is not unique.** `(framework,
  section_id)` collides on 13/147 items today (8.8%) and **54/221 (24.4%)** under
  the proposed roster — `('ENISA','Table 5:')` ×21, `('ENISA','Table 3:')` ×8.
  The **binding side condition** — the one criterion C3TEST passed — would be
  evaluated on a partition mislabelling a quarter of the corpus.
- **B5 (High, Likely) — `_require_secure_cloud` guards 1 of 5 pod-creation call
  sites.** `runpod_retrain.py:151`, `smoke_on_pod.py:81`, `probe_on_pod.py:74`
  and `runpod_orchestrate.py:382` take the default `(SECURE, COMMUNITY)`
  preference, and three rsync `PROJECT_ROOT` without excluding
  `data/processed/licensed`. Two of them are campaign instruments, not adjacent
  tooling.
- **B6 (High, Likely) — every contamination warning names a path that does not
  resolve.** Thirteen files including `CLAUDE.md` name
  `results/review/hub_reference.md`. That path is empty; the real file is
  `results/ceiling_study/hub_reference.md` (422 KB, orchestrator-verified). A
  guardrail spelled against a non-resolving path is indistinguishable from one
  that is working. *Re-scoped from the Security Architect's "the file does not
  exist," which is wrong.*
- **B7 (High, Likely) — the 46 Phase 2B bridges are 100% of `related_hub_ids`
  with no provenance marker**, model-proposed by cosine top-k (accept rate
  46/63 = 73.0%; a single threshold at 0.4485 reproduces 59 of 63 human
  decisions). `results/bridge/` has no `PROVENANCE.md` and
  `test_tier3_quarantine.py` does not cover it. Any bridge-curation round
  inherits them as taxonomy.
- **B8 (High, Likely) — the recruiting package's validity premise is
  misdescribed.** `claudedocs/curation-package.md` cites "two independent blind
  annotators agree at Cohen's κ ≈ 0.71–0.73"; the source
  (`results/ceiling_study/panel_agreement.md`) is an **LLM judge panel** with one
  human (the owner) and five LLM judges, reporting raw agreement proportions,
  not κ.
- **B9 (Medium-High, Likely) — the estimator Amendment 2 makes primary does not
  exist in the code path that emits the verdict.** `grep -rn "cluster_bootstrap"
  tract/ scripts/phase1b/` returns nothing; `gate_decision` still calls
  `paired_bootstrap_delta`. Adopting §1 as documentation would leave the
  pipeline emitting a verdict the pre-registration no longer authorises.
- **B10 (Medium-High, Likely) — the roster change costs two pod runs and two more
  draws on the non-renewable 147-item test split.** Amendment 1 §1.5 already
  spent draw two and recorded that a third "would be much harder to justify."
  §6b spends draws three and four and does not mention the split.
- **B11 (Medium-High, Likely) — the simulation's clamp shrinks delivered μ by up
  to 12% and τ by up to 18%**, biting on 18–40% of draws in the cells the
  decision reads. At τ=0.3702 realised τ is 0.2851 and μ falls 0.20 → 0.156.
- **B12 (Medium, Likely) — the calibration claim is wrong on its own output.**
  §6e says "2–6% across every cell"; the shipped script prints **2–9%**, with 9%
  at τ=0.16. The guarding test permits up to **3α = 15%**. Un-clamped
  re-simulation gives 8.5% at τ=0.20, and the small-cluster bootstrap is
  anti-conservative in a way that **worsens as items per framework grow** — the
  lever curation buys.
- **B13 (Medium, Likely) — Amendment 2 §4 destroys §1.** Excluding n<10 folds
  leaves k=3, i.e. **ten** distinct cluster multisets; the resulting interval
  [−0.0704, +0.2706] is *narrower* than the k=5 version §1 adopted for being
  wider.
- **B14 (Medium, Likely) — Amendment 2 carries no "what was known when this was
  written" block**, which `CAMPAIGN3.md:4-7` makes binding on amendments and
  Amendment 1 supplies. Every §-level proposal is traceable to a measurement
  taken on the round it now governs. §1's random-effects argument *is* derivable
  from LOFO's own rationale without reference to any outcome; §2, §4 and §6 are
  not, and the draft does not separate them.
- **B15 (Medium, Likely) — `AI_FRAMEWORK_NAMES` is defined three times**
  (`scripts/phase0/common.py:42`, `tract/training/data.py:34`,
  `tract/training/data_quality.py:38`) with no test binding them, plus
  `BRIDGE_AI_FRAMEWORK_IDS` and `EXCLUDED_ILLUSTRATION_FRAMEWORKS` as a fourth
  and fifth population. §6b's instruction names one.
- **B16 (Medium, Likely) — the evidence base is untracked and writes nothing by
  default.** All three analysis scripts default `--out=None`; the documented
  reproduce commands omit `--out`; `results/analysis/` does not exist; every
  figure was transcribed by hand. `scripts/analysis/` is in neither CI's lint
  nor its mypy path. `audit_stratified_delta.py` is modified-but-uncommitted
  while `campaign3-rebaseline.md:149` cites it as the C3TEST reproduction path.
- **B17 (Medium, Plausible) — 100% of the primary's measured gain sits on
  donor-supervised hubs.** Splitting the 110 untouched items on whether any gold
  hub carries an ENISA/BIML/ETSI link: donor-supervised n=93 delta **+0.1183
  [+0.0108, +0.2258]**; not donor-supervised n=17 delta **+0.0000
  [−0.1765, +0.1765]**. Contrast P(≤0)=0.144 — *not* established, reported at the
  standard demanded of H3.
- **B18 (Medium, Plausible) — H1's predictor is post-audit and un-firewalled.**
  50% of untouched items change `gold_degree_max` if the audit is undone; **100%**
  change under the LOFO firewall. Re-running three ways sustains the conclusion
  (no zero-shot depression in any), so the refutation holds — for a reason the
  document does not state.

---

## C. Dropped ledger

| claim | anchor | posterior | why dropped |
|---|---|---|---|
| Incoming folds have far shorter anchors than incumbents | MLOps F10 | Unlikely | Overstated. ENISA 330 / BIML 388 chars, but incumbent **MITRE ATLAS is 476** with 1/202 full_text; incumbents already span 476→7858. |
| `results/review/hub_reference.md` does not exist | Security F2 | — | Re-scoped, not dropped → **B6**. The file exists at `results/ceiling_study/`. |
| Per-fold delta tracks anchor length (r=+0.889, p=0.044) | orchestrator | Unlikely | Collapses on robustness: k=4 r=0.67 p=0.33; k=3 r=0.45 p=0.70. Carried by the two folds with 6 and 7 items. Hypothesis only. |
| `rescore_predictions.py` mis-join contaminates the new analysis | ML Engineer F6 | Unlikely *(for this scope)* | The mis-join is real (93.2% key miss) but the three new scripts build corpus and predictions the same way; positional join verified valid on 147/147. |
| Budget gate refusal at k=8 is a defect | MLOps F1 | Unlikely | It is the control working. Retained instead as an **operational cost** input (B10). |

**Tail risk (Critical, below Plausible, parked with trigger):**
`PRD.md:465-467` states nothing model-derived is downstream in "the OpenCRE fork
import," while `tract/export/filters.py:64` selects `review_status = 'accepted'`
from a pool `results/review/PROVENANCE.md` classifies **Tier 3** (898
model-proposed, human-ratified; 74.6% agreement with independent gold).
Not re-measured here. **Trigger to raise: any move toward RFC submission.**

---

## D. Convergence

Round 1 did **not** converge — it returned Critical findings at Very likely
against the artifact's central claims, its estimator, its power analysis, its
tests, and a published external artifact. Rounds 2–5 are not the next step,
because the round-1 findings invalidate the inputs those rounds would attack.
**The correct next step is remediation, then a fresh round 1 against the
corrected documents.**

---

## E. Remediation plan, ordered by expected cost reduction per unit of effort

### Do now, independent of every decision
1. **Correct the model card's bridge table** (A1). A live external falsehood.
   Reconcile `BRIDGE_AI_FRAMEWORK_IDS` with the eight-framework definition,
   restate AI-only / bridged counts, and disclose that the Phase 2B bridge
   analysis ran over 21 AI-only hubs when the current definition gives 83.
2. **Re-arm the reaper persistently** (A8). One user unit with
   `WantedBy=default.target`. The spend bound is absent right now.
3. **Correct the degree claim in three documents** (A2) — it is arithmetically
   wrong, and `docs/campaign3-audit-mechanism.md` should state that the
   mechanism was dissolved by recomputation, not by its three underpowered tests.
4. **Commit or delete the evidence base** (B16). A spending decision currently
   rests on untracked files; add `scripts/analysis/` to CI lint and mypy.

### Before any pre-registration
5. **Withdraw Amendment 2 as drafted.** A3 (its rationale is backwards), B13
   (§4 destroys §1), B14 (no disclosure block), B9 (the estimator is not in the
   code path). §1's random-effects argument is worth keeping; it needs a correct
   rationale — *the interval crosses zero*, not *the gate gets harder*.
6. **Redo the power analysis** (A4, A5, B11, B12): micro estimator, real fold
   sizes, a τ grid covering 0.3702, no silent clamp, and a coverage check of the
   proposed primary.
7. **Fix the tests** (A7): a mutation that deletes cluster resampling must fail;
   correct the pinned CI to 0.4324; give `make_contrast` independent generators;
   cover `build_rows`.
8. **Reconcile the AI-framework definitions** (B15) with one constant and one
   test asserting the sets agree.

### Before the roster rotation
9. **Fix the echo key** (B4) — 24.4% collision under the proposed roster would
   corrupt the one criterion C3TEST passed.
10. **Pre-register multi-label density as a covariate** (B3), or the predicted
    delta drop is uninterpretable.
11. **Price the test-split draws** (B10) explicitly; Amendment 1 §1.5 makes this
    an owner decision, not an operational detail.

### Before any annotation round
12. **Fix the contamination guardrail path** (B6) and add a `PROVENANCE.md` plus
    quarantine coverage to `results/bridge/` (B7).
13. **Correct the recruiting package's κ claim** (B8).
14. **Extend `_require_secure_cloud` to all five call sites** (B5).

---

## F. Residual risk

Even with every item above closed, three things remain unresolved and are not
fixable by remediation:

- **τ cannot be estimated at k=5.** A5 shows the estimate is one two-item fold.
  Every funding decision runs through a parameter the design cannot measure, and
  the cheapest purchase of it is the roster rotation — which A6 shows buys five
  hubs, not three independent clusters.
- **The four curation targets admit no within-framework control** (B1). No
  estimator fixes a stratum with no control; this is structural.
- **The supervision leak cannot be closed by firewalling** (78 AI hubs / 380
  general / intersection 0, orchestrator-verified). It *can* be closed by
  creating traditional→AI links, which is the (NEW) option — and B1 is the
  strongest argument for it, because those frameworks have original gold.


---

## Correction, 2026-09-04 — B3's delta split does not reproduce

**The composition half of B3 is correct.** Multi-label share among incumbent
frameworks is 13 of 147 = **8.8%**, exactly as stated, and the candidate-roster
figures are a property of those frameworks rather than of any run.

**The delta half is not.** B3 reports single-label +0.1165 against multi-label
**−0.1429**, difference **P(≤0)=0.044**, and calls it "better established than
the H3 split the whole document rests on". Measured over every committed run
that carries a complete five-fold set:

| run | single | multi (n=13) | P(≤0) |
|---|---|---|---|
| `c2r_TEST_A3_prose_sw_qwen06b` | +0.1493 | +0.0000 | 0.073 |
| `c3_TEST_A3_prose_sw_qwen06b_seq1024` | +0.1567 | +0.0000 | 0.167 |
| `lofo_prose` | +0.0672 | **+0.2308** | 0.955 |
| `lofo_prose_desconly` | +0.0746 | **+0.2308** | 0.937 |
| `lofo_prose_stopwords` | +0.0522 | **+0.3077** | 0.987 |
| `lofo_title_only` | +0.1119 | **+0.3077** | 0.957 |

The multi-label delta is **never negative**. On four of six runs it is *larger*
than the single-label delta — the opposite of the claimed direction — and no run
comes near P(≤0)=0.044. The claimed −0.1429 is exactly −1/7, a value an n=13
stratum cannot produce.

### What this changes

**Remediation item 10** — "pre-register multi-label density as a covariate, or
the predicted delta drop is uninterpretable" — rests on the refuted half. There
is no evidential basis for predicting that roster rotation depresses the delta,
and pre-registering a directional covariate on a number that does not reproduce
would repeat the error the pre-registration exists to prevent.

### What survives, and it is worth keeping

The confound is real and **does not depend on sign**. If the roster moves
multi-label density from 8.8% to 36.5%, and multi-label items behave differently
in *either* direction, the pooled delta moves for compositional reasons. On the
campaign's own test run the two strata differ by more than 0.15.

So multi-label density should be **disclosed and stratified**, not predicted:
report the multi-label share of any new roster alongside the stratified deltas,
and treat a pooled movement as uninterpretable until the stratified figures are
shown. That is the defensible form of item 10 and it costs nothing to run.

`tests/test_multilabel_covariate_claim.py` holds this correction from both
sides — it re-measures every qualifying run and goes red if any of them ever
does reproduce the negative split, at which point item 10 should be re-opened
rather than the test deleted.

### Note on this correction

This is the third round in which a round's own findings contained an error, and
the second in which the error was a diagnostic that flattered the argument being
made. B3's number was more favourable to its conclusion than any measurement
supports, and it was quoted as *better established* than the finding it was
being compared to. Nothing in the round-1 process would have caught it, because
nobody re-ran it.
