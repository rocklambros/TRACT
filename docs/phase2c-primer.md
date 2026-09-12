# Phase 2C primer — cold pickup

**Last updated 2026-09-12, at `a11d4d0` (PR #87 merged).** Read this before
touching Phase 2C. It exists because the phase looks over-engineered until you
know which measurement forced each piece.

## Purpose

Phase 2C buys human supervision for a structural hole: the AI and traditional
CRE hub regions are **disjoint** — 78 AI hubs, 380 traditional, intersection
exactly 0 — so nothing in the curated gold ever positions a traditional control
against an AI hub, and the PRD's bridging capability has no training signal
behind it. Two anonymous volunteers judged 300 NIST SP 800-53 controls against
all 78 AI hubs, producing a Tier-2 corpus. The binding constraint is that this
project has withdrawn three headline numbers, so every claim here is
pre-registered before it can be measured.

## Founding principles

1. **Pre-register, then measure — and commit the plan before the review.** The
   Gate 2 plan was committed at `66d9287` *before* its premortem so findings
   anchored to a fixed blob and remediation was a visible second commit.
   Rejected: editing the plan as findings arrived, which leaves no diff and is
   how Campaign 2's authorising clause came to be written after its results.

2. **The primary estimand is a difference-in-differences over exposure, not a
   pooled delta.** Only 27 of 74 eval items can respond to the corpus at all;
   the other 47 have no bridge positive on any scored hub in either arm. On the
   real run the two disagreed *in sign* (pooled −0.0270, DiD +0.1009). Rejected:
   the pooled contrast, which would have published a negative delta as evidence
   against the corpus.

3. **A comparator with zero supervision is not a control.** Under the strict
   all-AI firewall the bridge-free arm has **0 of 62** scored hubs supervised,
   so `A1 − A0` asks "does any supervision beat none". The placebo arm (A0R)
   holds count, source framework and hub distribution fixed and varies only
   *which* control was judged to belong to each hub. Rejected: three arms with
   a second corpus instead, which buys 2 eval items.

4. **A delta without a noise floor is unreadable.** A0P is the comparator at
   seed+1, true effect zero by construction. It cost $2.45 and it is what
   converted an ambiguous FAIL into a defensible NO VERDICT. Rejected: trusting
   that a fixed seed makes arms comparable — `fp16=True` and no
   `use_deterministic_algorithms` mean they are not bit-reproducible.

5. **Both DiD strata must come from one trained model.** `--split gate2
   --framework ALL` scores all 74 items with one model per arm. Rejected:
   one model per framework, which puts training-draw variance *between* the
   strata — precisely what the DiD exists to cancel.

6. **Publish the corpus before the verdict exists.** `results/phase2c/phase2c_bridge_corpus.json`
   was committed before the first pod. Rejected: shipping after, which makes the
   commitment exactly as independent of the result as the result allows.

7. **Anonymity is honoured mechanically, not by asking.** The annotators are
   anonymous by request, so there is no channel for a withdrawal
   acknowledgement. `build_gate2_corpus --exclude-annotator` re-cuts the corpus
   instead. What cannot be undone is recorded: a trained checkpoint cannot be
   un-trained.

## Where to start

1. **If you read only one thing:** `docs/phase2c-gate2-results.md` — the run,
   the verdict, what may and may not be reported, and what would answer the
   question.
2. `docs/phase2c-preregistration.md` **including Amendment 1** — binding. §3 is
   superseded by the amendment in four places; read both or you will cite a
   retired criterion.
3. Reproduce the verdict without a GPU:
   ```bash
   python -m scripts.analysis.gate2_delta \
     --comparator results/phase1b/gate2_A0 --treatment results/phase1b/gate2_A1 \
     --strata results/phase2c/gate2_strata.json --label "A1 vs A0 (PRIMARY)"
   ```
4. `docs/phase2c-premortem-gate2.md` — the six-perspective review that rebuilt
   the design. Its C1/C2/C3 are the three measurements behind principles 2–4.
5. Rebuild every input from scratch:
   ```bash
   python -m scripts.build_gate2_corpus --round-dir data/training/bridge-r2
   python -m scripts.build_placebo_corpus
   python -m scripts.analysis.gate2_strata --corpus data/training/hub_links_bridge.r2.jsonl
   ```

## Defense-in-depth that will mislead you

Each of these looks like layered protection. Each has a single component all
the layers depend on.

- **The firewall has two checks** — `assert_firewall` (no held-out text leaked
  into hub representations) and `assert_exclusion_fired` (the exclusion removed
  what it named). **Neither catches a framework name that does not match the
  corpus spelling.** `"OWASP LLM Top 10"` vs `"OWASP Top10 for LLM"` excludes
  nothing, trains on everything, scores ~0.9. Only
  `tests/test_gate2_constants.py::TestTheNamesResolve` catches it.
- **Provenance looks triple-guarded** — `bridge_links_sha256`,
  `staleness.check_result`, and `bridge_links_path` in `ARM_DEFINING_KEYS`.
  **All three read the fold record's `config` block.** A record written before
  that field existed reads as *unrecorded*, not *wrong*, and unrecorded is not
  a failure.
- **Cost looks triple-guarded** — reaper timer, `try/finally` teardown, price
  ceiling. **The reaper only protects a pod whose name is in
  `expected_pod_names()` AND whose driver module is in `ORCHESTRATOR_MODULES`.**
  Miss the first and a dead orchestrator strands a billing GPU; miss the second
  and the guard *reaps your live run*. A new runner needs both entries.
- **Gate 1 passed twice**, which reads as robustness. **Both rounds share the
  same two annotators and the same hub sheet**, and that sheet was built from
  `BRIDGE_AI_FRAMEWORK_IDS` — which includes the frameworks supplying Gate 2's
  eval gold (open item C6). A pass licenses a claim about *this hub region*.
- **The bootstrap reports a 95% interval**, which reads as full uncertainty.
  **It resamples items only.** Training-draw variance is invisible to it; the
  noise-floor arm is the sole instrument for that and it is a separate manual
  contrast, not part of the gate's own output.

## Glossary

- **Exposed / unexposed** — an eval item is *exposed* when ≥1 hub in its
  `valid_hub_ids` receives a bridge positive. 27 / 47 on this eval. Fixed in
  `results/phase2c/gate2_strata.json` before any arm ran.
- **DiD** — `mean(delta | exposed) − mean(delta | unexposed)`. The Gate 2
  primary.
- **Strict all-AI firewall** — all eight `PHASE2C_GATE2_HELD_OUT` frameworks
  held out at once, as opposed to the five-name LOFO eval roster
  (`AI_FRAMEWORK_NAMES`). Confusing these is the single most expensive mistake
  available here.
- **Scored hubs (62) vs gold hubs (43)** — `hit@1` credits `valid_hub_ids`, not
  `ground_truth_hub_id`. Quote 62.
- **A0 / A1 / A0R / A0P** — comparator / treatment / placebo / noise floor.
- **Tier 2** — independently human-authored, no model output shown to the
  annotator. Weaker than Tier 1 (OpenCRE-curated independently of TRACT),
  far stronger than Tier 3.
- **NO VERDICT** — the pre-declared outcome when the noise floor equals or
  exceeds the treatment effect. Not a FAIL.

## What's NOT here

- **No round 3, deliberately.** Nothing in the result implicates the corpus;
  the instrument failed. Re-annotating NIST 800-53 with the same two people
  measures memory.
- **No multi-seed run yet.** That is the next work: 3–5 seeds across A0/A1/A0R,
  ~$37 at the measured $2.45/arm, estimand becomes the seed-mean delta.
- **`gate_decision()` is NOT Gate 2.** It is hardwired to a zero-shot baseline
  at threshold 0.10 — the comparator the pre-registration retires, and one that
  73% of *bridge-free* arms already satisfy. Its output lands in
  `aggregate_metrics.json` under the key `gate`, exactly where a reader looks
  for a verdict. Use `scripts/analysis/gate2_delta.py`.
- **`runpod_retrain` does not run `run_fold`.** It runs the Phase 1C
  active-learning scripts. `scripts/phase2c/run_gate2.py` is the Gate 2 driver;
  it reuses only the pod lifecycle.
- **The sampler's AI upweighting is inert here.** `is_ai` keys on the *source*
  framework and every bridge link names a traditional standard, so under the
  strict firewall `n_ai == 0` and the branch-balance interleave degenerates to a
  plain shuffle. Known, unfixed, and worth fixing before the multi-seed run.
- **No withdrawal acknowledgements.** Anonymity by request; mechanism instead.
- **C6 is disclosed, not fixed.** The hub sheet was built from the eval's own
  gold links. Fixing it means re-annotating against a sheet rebuilt with the
  eval frameworks excluded — the right design for a future round.

## Owner

Rock Lambros. Binding documents: `docs/phase2c-preregistration.md` (+ Amendment
1), `results/phase1b/CAMPAIGN3.md`. Reporting rules that survive any Gate 2
result: `docs/phase2c-results.md` §4.
