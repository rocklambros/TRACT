# Phase R + 2C-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the defects the round-1 premortem confirmed in the Campaign 3 analysis code, then build the tooling for a traditional→AI bridge curation round that closes the supervision leak without touching the evaluation corpus.

**Architecture:** Phase R repairs four measurement defects in `scripts/analysis/`. Phase 2C-1 adds a tier-separated bridge corpus (`hub_links_bridge.jsonl`), a model-free annotator packet generator, an importer with provenance guards, and the free Gate-1 orphan-rate check. Bridge links merge into training only; the evaluation build never sees them.

**Tech Stack:** Python 3.11/3.12, numpy, pytest, ruff, mypy --strict. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-01-phase2c-bridge-curation-design.md`

## Global Constraints

- **No model output may reach an annotator.** Enforced by test, not by review.
- **Bridge links never enter `hub_links_curated.jsonl` and never enter `build_evaluation_corpus`.** The 147-item eval corpus must be byte-identical before and after.
- **Tier 2 tagging is mandatory** on every bridge link: `tier`, `annotator_id`, `created_at`.
- `USE_TF=0` for every command that imports anything under `tract/`.
- **No `# type: ignore`** in new code. CI runs mypy 2.2.0; this machine runs 1.11.2, and a suppression one version needs the other rejects as unused.
- ruff and mypy `--strict` clean on every file touched.
- Commit messages carry no AI/tool attribution (project rule).

---

## Phase R — repair the confirmed measurement defects

### Task R1: Give `make_contrast` independent generators

The reported CI depends on the order the two strata are drawn in. Shipped order prints `[+0.1081, +0.4595]`; a 500,000-draw reference gives `+0.4324`, which is what every document states.

**Files:**
- Modify: `scripts/analysis/gate_rule_candidates.py` (`make_contrast`)
- Test: `tests/test_gate_rule_candidates.py`

**Interfaces:**
- Consumes: `bootstrap_deltas(rows, n_resamples, rng)`, `bootstrap_baselines(...)`
- Produces: `make_contrast(a, b, n_resamples, rng)` unchanged in signature; results become order-invariant.

- [ ] **Step 1: Write the failing test**

```python
def test_contrast_is_invariant_to_argument_order(self) -> None:
    """The CI must not depend on which stratum is drawn first."""
    a = [_row("f", 1, 0)] * 20 + [_row("f", 0, 0)] * 17
    b = [_row("f", 1, 0)] * 11 + [_row("f", 0, 0)] * 99
    fwd = make_contrast(a, b, 4000, np.random.default_rng(42))
    rev = make_contrast(b, a, 4000, np.random.default_rng(42))
    assert fwd["ci_a"] == pytest.approx(rev["ci_b"], abs=1e-12)
    assert fwd["ci_b"] == pytest.approx(rev["ci_a"], abs=1e-12)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `USE_TF=0 python -m pytest tests/test_gate_rule_candidates.py::TestContrast::test_contrast_is_invariant_to_argument_order -q`
Expected: FAIL — the two intervals differ because one generator is threaded through both draws.

- [ ] **Step 3: Derive a per-stratum generator from the caller's**

```python
def _child(rng: np.random.Generator, tag: int) -> np.random.Generator:
    """A generator whose stream depends on the stratum, not on call order."""
    return np.random.default_rng(rng.bit_generator.seed_seq.spawn(tag + 1)[tag])
```

In `make_contrast`, replace the shared `rng` with `_child(rng, 0)` for stratum
`a` and `_child(rng, 1)` for stratum `b`, and the same for the baselines.

- [ ] **Step 4: Run and watch it pass**

Run: `USE_TF=0 python -m pytest tests/test_gate_rule_candidates.py -q`
Expected: PASS.

- [ ] **Step 5: Correct the pinned bound**

In `tests/test_gate_rule_candidates.py`, `WORKED_EXAMPLE` currently pins
`ci_b=(0.1081, 0.4595)`. Change to `(0.1081, 0.4324)` and update the comment to
say the value is the 500,000-draw reference, not a 10,000-draw sample.

- [ ] **Step 6: Re-run the script and confirm the docs now agree**

Run: `USE_TF=0 python -m scripts.analysis.gate_rule_candidates`
Expected: the relabelled stratum prints `[+0.1081, +0.4324]`, matching
`docs/campaign3-audit-mechanism.md` §6b and `results/phase1b/CAMPAIGN3.md`.

- [ ] **Step 7: Commit**

```bash
git add scripts/analysis/gate_rule_candidates.py tests/test_gate_rule_candidates.py
git commit -m "make bootstrap contrasts independent of argument order"
```

---

### Task R2: Make the power simulation measure the gate's estimand

`gate_power_simulation` uses identical `n_per` for every fold, so it computes an unweighted mean over frameworks (macro). The gate reports the item-weighted mean (micro). On the real primary those differ by 0.1701 — 1.7× the gate threshold.

**Files:**
- Modify: `scripts/analysis/gate_power_simulation.py`
- Test: `tests/test_gate_power_simulation.py`

**Interfaces:**
- Produces: `simulate_study(k, n_per_fold, mu, tau, discordant, rng) -> list[np.ndarray]` where `n_per_fold` becomes `Sequence[int]`, one size per framework.
- Produces: `cluster_bootstrap_pass(folds, n_bootstrap, rng)` weights by item count.

- [ ] **Step 1: Write the failing test**

```python
def test_uses_the_item_weighted_mean_not_the_fold_mean(self) -> None:
    """The gate reports micro; a balanced simulation silently reports macro."""
    # 90 items at delta 0, 10 items at delta 1. micro = 0.10, macro = 0.50.
    folds = [np.zeros(90), np.ones(10)]
    stat = pooled_delta(folds)
    assert stat == pytest.approx(0.10)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `USE_TF=0 python -m pytest tests/test_gate_power_simulation.py -k item_weighted -q`
Expected: FAIL — `pooled_delta` does not exist.

- [ ] **Step 3: Add the estimator and use it everywhere**

```python
def pooled_delta(
    folds: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]],
) -> float:
    """Item-weighted (micro) mean, matching the gate's primary."""
    return float(np.concatenate(folds).mean())
```

Change `simulate_study` to accept `n_per_fold: Sequence[int]`, and
`cluster_bootstrap_pass` to resample each drawn fold at that fold's own size and
pool with `pooled_delta` rather than `drawn.mean(axis=(1, 2))`.

- [ ] **Step 4: Run and watch it pass**

Run: `USE_TF=0 python -m pytest tests/test_gate_power_simulation.py -q`
Expected: PASS.

- [ ] **Step 5: Use the real fold sizes**

Replace the balanced scenarios with the observed sizes
`(63, 30, 11, 4, 2)` for k=5, and `(63, 30, 11, 4, 2, 33, 17, 24)` for the k=8
roster, so the surface describes a design that exists.

- [ ] **Step 6: Commit**

```bash
git add scripts/analysis/gate_power_simulation.py tests/test_gate_power_simulation.py
git commit -m "simulate the gate's own estimand, at the fold sizes it runs on"
```

---

### Task R3: Cover the τ point estimate, and stop the clamp biasing the axes

`TAU_GRID` stops at 0.20 while the data's own point estimate is 0.3702, so the branch that says "replace the instrument" was never simulated. Separately the `np.clip` in `simulate_study` shrinks delivered μ by up to 12% and τ by up to 18% at the high end, unlogged.

**Files:**
- Modify: `scripts/analysis/gate_power_simulation.py`
- Test: `tests/test_gate_power_simulation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_reports_the_realised_mu_and_tau_not_the_requested_ones(self) -> None:
    """The clamp binds on a third of draws at high tau; the surface must say so."""
    folds = simulate_study(400, [200] * 400, 0.25, 0.20,
                           DISCORDANT_RATE, np.random.default_rng(0))
    realised = realised_parameters(folds)
    assert realised["mu"] < 0.25
    assert realised["clamped_fraction"] > 0.0
```

- [ ] **Step 2: Run it and watch it fail**

Run: `USE_TF=0 python -m pytest tests/test_gate_power_simulation.py -k realised -q`
Expected: FAIL — `realised_parameters` does not exist.

- [ ] **Step 3: Implement it**

```python
class RealisedParameters(TypedDict):
    mu: float
    tau: float
    clamped_fraction: float


def realised_parameters(
    folds: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]],
) -> RealisedParameters:
    """What the simulation actually delivered, after clamping."""
    means = np.array([f.mean() for f in folds])
    return RealisedParameters(
        mu=float(means.mean()),
        tau=float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        clamped_fraction=float((np.abs(means) > DISCORDANT_RATE).mean()),
    )
```

- [ ] **Step 4: Extend the grid and log the realised values**

```python
TAU_GRID: Final[tuple[float, ...]] = (0.00, 0.05, 0.08, 0.12, 0.16, 0.20, 0.28, 0.37)
```

Log realised μ and τ beside each requested cell so the axes are honest.

- [ ] **Step 5: Run and confirm**

Run: `USE_TF=0 python -m pytest tests/test_gate_power_simulation.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/analysis/gate_power_simulation.py tests/test_gate_power_simulation.py
git commit -m "sweep tau past the corpus's own estimate and report realised parameters"
```

---

### Task R4: Make the tests fail when the cluster bootstrap is removed

Six of six core-logic mutations survive the current suite, including deleting framework resampling outright — the property the estimator exists for. `test_between_framework_spread_can_sink_a_high_mean` asserts `tight_pass >= spread_pass`, satisfied by 5 ≥ 5.

**Files:**
- Modify: `tests/test_gate_power_simulation.py`
- Test: itself

- [ ] **Step 1: Write the failing test**

```python
def test_cluster_resampling_changes_the_result(self) -> None:
    """Deleting framework resampling must break something. It currently does not."""
    # Nine folds, four at delta 1.0 and five at 0.0. Item resampling alone
    # cannot move the fold means; cluster resampling can.
    folds = [np.ones(40) if i < 4 else np.zeros(40) for i in range(9)]
    rng = np.random.default_rng(0)
    with_clusters = _pass_probability(folds, 2000, rng, resample_frameworks=True)
    without = _pass_probability(folds, 2000, rng, resample_frameworks=False)
    assert abs(with_clusters - without) > 0.10, (
        "cluster resampling made no difference; the estimator is an "
        "item bootstrap wearing its name"
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `USE_TF=0 python -m pytest tests/test_gate_power_simulation.py -k cluster_resampling_changes -q`
Expected: FAIL — `_pass_probability` does not exist.

- [ ] **Step 3: Expose the switch the test needs**

```python
def _pass_probability(
    folds: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]],
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    resample_frameworks: bool = True,
) -> float:
    """Fraction of bootstrap draws at or below the gate threshold."""
    k = len(folds)
    picks = (rng.integers(0, k, (n_bootstrap, k)) if resample_frameworks
             else np.tile(np.arange(k), (n_bootstrap, 1)))
    ...
```

`cluster_bootstrap_pass` calls it with the default.

- [ ] **Step 4: Run and confirm the mutation is now caught**

Run: `USE_TF=0 python -m pytest tests/test_gate_power_simulation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/gate_power_simulation.py tests/test_gate_power_simulation.py
git commit -m "fail the suite when framework resampling is removed"
```

---

### Task R5: Read every audit decision, not only `corrections`

`audit_corrections_log.json` carries `corrections` (56), `exclusions` (1) and `kept_weak` (8) — 65 decisions. Both probes read `corrections` alone, so an OWASP AI Exchange link the audit *deleted* is counted as untouched, and 5 kept-weak items sit inside the Tier-1 stratum.

**Files:**
- Modify: `scripts/analysis/audit_mechanism_probe.py` (`load_audit_index`)
- Modify: `docs/campaign3-audit-mechanism.md` §5
- Test: `tests/test_audit_mechanism_probe.py`

- [ ] **Step 1: Write the failing test**

```python
def test_reconciles_every_decision_in_the_log(self) -> None:
    """corrections + exclusions + kept_weak must account for the whole log."""
    log = json.loads(AUDIT_LOG.read_text(encoding="utf-8"))
    counted = (len(log["corrections"]) + len(log["exclusions"])
               + len(log["kept_weak"]))
    assert counted == 65
    assert log["links_excluded"] == len(log["exclusions"])
```

- [ ] **Step 2: Run it and watch it fail**

Run: `USE_TF=0 python -m pytest tests/test_audit_mechanism_probe.py -k reconciles -q`
Expected: FAIL — the test does not exist yet; write it, confirm it passes on the
arithmetic, then add the assertion that `load_audit_index` raises when the three
counts do not reconcile.

- [ ] **Step 3: Reconcile in `load_audit_index`**

```python
    applied = log["corrections_applied"]
    excluded = log["links_excluded"]
    if len(corrections) != applied or len(log["exclusions"]) != excluded:
        raise ValueError(
            f"audit log does not reconcile: {len(corrections)} corrections "
            f"against corrections_applied={applied}, {len(log['exclusions'])} "
            f"exclusions against links_excluded={excluded}. Refusing to "
            "stratify against a log that disagrees with itself."
        )
```

- [ ] **Step 4: Correct the claim in the document**

`docs/campaign3-audit-mechanism.md` §5 states *"the audit never touched OWASP AI
Exchange at all."* It excluded link `547-824` from that fold. Replace with the
measured statement and note that the exclusion removes an item from the
denominator rather than relabelling one.

- [ ] **Step 5: Run and commit**

```bash
USE_TF=0 python -m pytest tests/test_audit_mechanism_probe.py -q
git add scripts/analysis/audit_mechanism_probe.py tests/test_audit_mechanism_probe.py docs/campaign3-audit-mechanism.md
git commit -m "reconcile all 65 audit decisions, not the 56 corrections alone"
```

---

### Task R6: One AI-framework definition, with a test binding the copies

`AI_FRAMEWORK_NAMES` is defined in three modules, `BRIDGE_AI_FRAMEWORK_IDS` is a fourth population and `EXCLUDED_ILLUSTRATION_FRAMEWORKS` a fifth. No test relates them. This is what produced the published bridge falsehood.

**Files:**
- Create: `tests/test_ai_framework_sets.py`
- Modify: `tract/training/data.py`, `tract/training/data_quality.py` to import from `scripts/phase0/common.py`'s definition if they already agree; otherwise document why they differ.

- [ ] **Step 1: Write the failing test**

```python
def test_every_ai_framework_name_has_a_bridge_id(self) -> None:
    from scripts.phase0.common import AI_FRAMEWORK_ID_MAP, AI_FRAMEWORK_NAMES
    from tract.config import BRIDGE_AI_FRAMEWORK_IDS
    ids = {AI_FRAMEWORK_ID_MAP[n] for n in AI_FRAMEWORK_NAMES}
    assert ids <= BRIDGE_AI_FRAMEWORK_IDS, (
        f"{ids - BRIDGE_AI_FRAMEWORK_IDS} are AI frameworks for the LOFO "
        "roster but count as traditional for bridge classification"
    )


def test_the_three_copies_of_ai_framework_names_agree(self) -> None:
    from scripts.phase0.common import AI_FRAMEWORK_NAMES as a
    from tract.training.data import AI_FRAMEWORK_NAMES as b
    from tract.training.data_quality import AI_FRAMEWORK_NAMES as c
    assert a == b == c
```

- [ ] **Step 2: Run and watch it fail or pass**

Run: `USE_TF=0 python -m pytest tests/test_ai_framework_sets.py -q`
If it passes, the constants happen to agree today and the test is the guard that
keeps them agreeing. If it fails, reconcile before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ai_framework_sets.py
git commit -m "bind the five AI-framework definitions with a test"
```

---

## Premortem checkpoint 1

Run `/adversarial-premortem-complete` over Phase R's diff and
`docs/campaign3-audit-mechanism.md` as amended. Carry confirmed findings into
Phase 2C-1 before writing any of it. Record the round at
`docs/campaign3-premortem-round2.md`.

---

## Phase 2C-1 — bridge tooling

### Task C1: The bridge link type and loader

**Files:**
- Create: `tract/bridge/links.py`
- Test: `tests/test_bridge_links.py`

**Interfaces:**
- Produces: `BridgeLink` (frozen dataclass: `framework_id`, `standard_name`, `section_id`, `section_name`, `cre_id`, `tier`, `annotator_id`, `created_at`, `confidence`, `rationale`)
- Produces: `load_bridge_links(path: Path) -> list[BridgeLink]`
- Produces: `merge_for_training(curated: list[HubStandardLink], bridge: list[BridgeLink]) -> list[HubStandardLink]`

- [ ] **Step 1: Write the failing test**

```python
def test_bridge_links_never_reach_the_evaluation_corpus(tmp_path) -> None:
    """The 147-item eval corpus must be byte-identical with bridges merged."""
    from scripts.phase0.common import (AI_FRAMEWORK_NAMES,
                                       build_evaluation_corpus,
                                       load_curated_links)
    from tract.bridge.links import BridgeLink, merge_for_training
    curated = load_curated_links()
    before = build_evaluation_corpus(curated, AI_FRAMEWORK_NAMES, {})
    bridge = [BridgeLink(
        framework_id="nist_800_53", standard_name="NIST 800-53 v5",
        section_id="AC-3", section_name="Access Enforcement",
        cre_id="342-641", tier=2, annotator_id="a1",
        created_at="2026-09-01T00:00:00Z", confidence=3, rationale="test",
    )]
    after = build_evaluation_corpus(
        merge_for_training(curated, bridge), AI_FRAMEWORK_NAMES, {})
    assert len(after) == len(before)
    assert [i.control_text for i in after] == [i.control_text for i in before]
```

- [ ] **Step 2: Run and watch it fail**

Run: `USE_TF=0 python -m pytest tests/test_bridge_links.py -q`
Expected: FAIL — `tract.bridge.links` does not exist.

- [ ] **Step 3: Implement the module**

```python
@dataclass(frozen=True)
class BridgeLink:
    framework_id: str
    standard_name: str
    section_id: str
    section_name: str
    cre_id: str
    tier: int
    annotator_id: str
    created_at: str
    confidence: int
    rationale: str


def merge_for_training(
    curated: list[HubStandardLink], bridge: list[BridgeLink],
) -> list[HubStandardLink]:
    """Training corpus only. Never pass the result to the evaluation build."""
    return list(curated) + [
        # HubStandardLink fields are exactly: cre_id, cre_name,
        # standard_name, section_id, section_name. There is no link_type.
        HubStandardLink(
            cre_id=b.cre_id, cre_name="", standard_name=b.standard_name,
            section_id=b.section_id, section_name=b.section_name,
        )
        for b in bridge
    ]
```

- [ ] **Step 4: Run and confirm, then commit**

```bash
USE_TF=0 python -m pytest tests/test_bridge_links.py -q
git add tract/bridge/links.py tests/test_bridge_links.py
git commit -m "add the tier-2 bridge link corpus, separate from curated links"
```

---

### Task C2: Gate 1 — the orphan-rate check

**Files:**
- Create: `scripts/analysis/orphan_rate.py`
- Test: `tests/test_orphan_rate.py`

**Interfaces:**
- Produces: `strict_firewall_orphans(links) -> tuple[int, int]` returning (orphaned, total) distinct AI gold hubs.

- [ ] **Step 1: Write the failing test**

```python
def test_reproduces_the_measured_baseline(self) -> None:
    """78 of 78 today, per docs/campaign3-audit-mechanism.md section 6g."""
    from scripts.analysis.orphan_rate import strict_firewall_orphans
    from scripts.phase0.common import load_curated_links
    orphaned, total = strict_firewall_orphans(load_curated_links())
    assert (orphaned, total) == (78, 78)


def test_a_traditional_link_rescues_its_hub(self) -> None:
    from scripts.analysis.orphan_rate import strict_firewall_orphans
    from scripts.phase0.common import load_curated_links
    from tract.bridge.links import BridgeLink, merge_for_training
    links = load_curated_links()
    base, total = strict_firewall_orphans(links)
    rescued = merge_for_training(links, [BridgeLink(
        framework_id="nist_800_53", standard_name="NIST 800-53 v5",
        section_id="AC-3", section_name="Access Enforcement",
        cre_id="342-641", tier=2, annotator_id="a1",
        created_at="2026-09-01T00:00:00Z", confidence=3, rationale="t")])
    after, _ = strict_firewall_orphans(rescued)
    assert after == base - 1
```

- [ ] **Step 2: Run, watch fail, implement, run again**

Run: `USE_TF=0 python -m pytest tests/test_orphan_rate.py -q`

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/orphan_rate.py tests/test_orphan_rate.py
git commit -m "add the free Gate 1 orphan-rate check"
```

---

### Task C3: The annotator packet, provably model-free

**Files:**
- Create: `scripts/build_bridge_packet.py`
- Test: `tests/test_bridge_packet.py`

- [ ] **Step 1: Write the failing tests**

```python
FORBIDDEN = ("similarity", "cosine", "rank", "top_k", "suggested",
             "candidate", "model", "predict", "score", "related_hub")


def test_no_model_derived_column_anywhere(tmp_path) -> None:
    build_bridge_packet(tmp_path, top_n_hubs=20, framework_id="nist_800_53")
    for csv_path in tmp_path.glob("*.csv"):
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].lower()
        for term in FORBIDDEN:
            assert term not in header, f"{csv_path.name} leaks {term}"


def test_no_related_hub_ids_leak(tmp_path) -> None:
    """cre_hierarchy.json carries Phase 2B's 46 model-proposed edges."""
    build_bridge_packet(tmp_path, top_n_hubs=20, framework_id="nist_800_53")
    blob = " ".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.csv"))
    assert "related_hub" not in blob.lower()


def test_refuses_a_restricted_framework(tmp_path) -> None:
    with pytest.raises(ValueError, match="restricted"):
        build_bridge_packet(tmp_path, top_n_hubs=20, framework_id="etsi")
```

- [ ] **Step 2: Run, watch fail, implement, run again**

The hub sheet carries `hub_id, hub_name, hierarchy_path, branch`. The control
sheet carries `control_id, control_title, control_text`. Neither carries
`related_hub_ids`, any similarity, or any TRACT output. `framework_id` is checked
against `RESTRICTED_FRAMEWORK_IDS` before any text is read.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_bridge_packet.py tests/test_bridge_packet.py
git commit -m "generate the bridge annotation packet from non-model sources only"
```

---

### Task C4: The importer

**Files:**
- Create: `scripts/import_bridge_links.py`
- Test: `tests/test_import_bridge_links.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_refuses_an_unknown_hub_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown hub"):
        import_bridge_links(_csv(tmp_path, cre_id="000-000"), tmp_path / "o.jsonl")


def test_refuses_a_duplicate_row(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        import_bridge_links(_csv(tmp_path, duplicate=True), tmp_path / "o.jsonl")


def test_every_row_carries_tier_2_and_provenance(tmp_path) -> None:
    out = tmp_path / "o.jsonl"
    import_bridge_links(_csv(tmp_path), out)
    for line in out.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["tier"] == 2
        assert row["annotator_id"]
        assert row["created_at"]
```

- [ ] **Step 2: Run, watch fail, implement with `atomic_write_text`, run again**

- [ ] **Step 3: Commit**

```bash
git add scripts/import_bridge_links.py tests/test_import_bridge_links.py
git commit -m "import reviewed bridge links with provenance enforced at the boundary"
```

---

### Task C5: The pre-registration

**Files:**
- Create: `docs/phase2c-preregistration.md`

- [ ] **Step 1: Write it, committed before any annotation begins**

Contents: D4's two gates verbatim (Gate 1 `≤ 55/78`; Gate 2 trainable task and
τ leave-one-fold-out swing `≤ 0.15`), the measured baseline `78/78`, the
statement that the primary delta is explicitly **not** a gate, the annotator
exclusion list including framework authorship, and what was known on the day it
was written.

- [ ] **Step 2: Commit before the packet is sent**

```bash
git add docs/phase2c-preregistration.md
git commit -m "pre-register the Phase 2C gates before any control is read"
```

---

## Premortem checkpoint 2

Run `/adversarial-premortem-complete` over Phase 2C-1 and the pre-registration.
Record at `docs/campaign3-premortem-round3.md`. Fix confirmed findings before the
packet is generated.

---

## Phase S — security and quality, before merge

- [ ] Resolve the 6 dependabot advisories on `main` (4 high, 2 moderate)
- [ ] `/security-review` over the branch diff
- [ ] `code-simplifier` over the new modules
- [ ] Full suite, ruff, mypy `--strict`; confirm the failure set still matches the documented 28
- [ ] Squash-free merge of PR #82 with a summary that states what is *not* delivered: the annotation round and everything behind it

---

## Blocked, and stated plainly

**Phase 2C-2 (the NIST 800-53 sweep) cannot be done by an agent.** Generating
those links from a model would make them Tier 3 by
`results/phase1b/CAMPAIGN3.md` §2 — *"Produced by, or ratified in the presence
of, a model or LLM. No. At any ratio."* — which is the circularity this whole
phase exists to remove. The tooling, the gates and the packet are the
deliverable; the judgement is a person's.

**Phase 2C-3 (retrain, measure, premortem)** waits behind it. Gate 1 is free and
runs the moment links exist; Gate 2 is one ~$40 retrain.
