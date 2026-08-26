# Re-deriving Phase 1B LOFO hit@1 under a pinned, digest-locked environment

Status: approved to implement (owner delegated execution 2026-08-14)
Owner: rock@rockcyber.com

## Problem

`hit@1=0.531` (PRD.md:378, Phase 1B Gate 1) cannot be reproduced. Three
independent gaps caused that:

1. **The GPU image was a mutable tag.** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   carries no digest. The published model records `pytorch 2.8.0a0+gitba56102`,
   a nightly built from a git commit, which the "2.4.0" tag cannot explain. The
   tag moved under the project.
2. **The pod bootstrap installed ranges.** `pip install -e '.[phase0]'` let
   `transformers` float; it landed on 4.49.0. `peft`, `datasets` and
   `accelerate` were installed with no constraint at all. Only
   `sentence-transformers==5.3.0` was pinned, which is precisely why it is the
   one version the model artifact records correctly.
3. **No serving pin existed.** The model is fetched by revision with sha256
   verification, but the runtime interpreting it was whatever pip resolved.

The consequence: the weights are reproducible and the number attached to them
is not.

## Goal

Produce a `hit@1` measured under a fully specified environment, and land the
environment pin, the number, and the provenance linking them as one change.

Explicit non-goal: reproducing `0.531` exactly. That is not achievable — the
original torch nightly is not retrievable from any index. If the new number
differs, the difference is a finding, not a failure.

## Stack decision

**Selected: torch 2.13.0 / transformers 4.57.6 / sentence-transformers 5.7.0 /
tokenizers 0.22.2 / peft 0.20.0 / datasets 3.6.0, with huggingface_hub held
below 1.0.**

Four candidates were audited with `pip-audit`:

| Candidate | torch | transformers | Known vulns |
|---|---|---|---|
| A: current project env | 2.4.1 | 4.45.1 | 51 |
| B: model build env | 2.8.0 | 4.49.0 | 31 |
| **C: selected** | **2.13.0** | **4.57.6** | **5** |
| D: fully clean | 2.13.0 | 5.8.0 | 0 |

D was the original instruction and was rejected on evidence. `transformers`
5.x requires `huggingface-hub>=1.5.0`, while TRACT declares
`huggingface_hub>=0.24,<1` as a **core** dependency. The transformers 4→5
boundary is exactly the hf_hub 0.x→1.x boundary. Adopting D therefore forces a
major migration of six call sites, including `tract/model_resolver.py`, which
is the sha256-verified download path that gates `tract assign` for every end
user. Performing that migration unsupervised, in the same change as a metrics
re-derivation, would put the shipped CLI at risk to remove findings that are
not reachable.

C retains D's security benefit where it matters:

- **All torch findings are eliminated**, including `PYSEC-2025-41` (RCE via
  `torch.load(weights_only=True)`) and `PYSEC-2026-2286` (weights_only
  unpickler memory corruption). These are the two that map onto what TRACT
  actually does: download an artifact and load it.
- The 5 residual findings are all `transformers`, in X-CLIP, MobileViT,
  MaskFormer and Trax checkpoint-conversion paths. A search of `tract/` and
  `scripts/` returns **zero references** to any of them. They are not
  reachable from this codebase.

C also lands on sentence-transformers 5.7.0, the same version D would use, so
the model (built under 5.3.0) is loaded by a forward-compatible library rather
than the backward-compatible 3.2.0 currently installed locally.

### Validation already performed (no GPU cost)

On a local CPU venv running the exact C pin:

- every import in `tract/training/loop.py` and `tract/training/data.py` resolves
- `model[0].auto_model = get_peft_model(...)` (loop.py:49, an internal reach
  into sentence-transformers) still works
- `SentenceTransformerTrainer.train()` completes real steps with decreasing loss
- `encode()` returns L2-normalised vectors

Three sentence-transformers import paths emit DeprecationWarnings
(`losses`, `training_args`, `sampler` have moved under `sentence_transformer.`
and `base.`). They still function. They are left alone: changing them would
break compatibility with the older ST the project currently runs, and that is
not this change's job.

## Design

### Component 1 — Image digest pin

Resolve the RunPod image to `runpod/pytorch@sha256:<digest>` and pin it in
`scripts/phase1b/runpod_parallel.py`. The image must ship a CUDA runtime
compatible with torch 2.13.0; if the current tag's CUDA is too old, select a
newer tag first and pin *its* digest.

Applies equally to `scripts/phase1c/runpod_retrain.py` and
`scripts/phase0/runpod_orchestrate.py` if they carry the same pattern.

### Component 2 — Bootstrap repoint

Replace:

```
pip install -e '.[phase0]' && pip install sentence-transformers==5.3.0 peft datasets accelerate
```

with an install driven by `requirements.txt` + `requirements-ml.txt`, so the
pod runs the same pinned stack the repo declares. `accelerate` must be added
to `requirements-ml.txt` with a pin, since the bootstrap needs it and nothing
declares it today.

### Component 3 — Provenance capture

Each fold emits, alongside its metrics: image digest, `pip freeze` of the
realised environment, git SHA, seed, data hash, GPU model, and the resolved
versions of torch / transformers / sentence-transformers / tokenizers / peft.
Without this the new number inherits the old number's problem.

### Component 4 — The run

Five LOFO folds (MITRE ATLAS, OWASP AI Exchange, NIST AI 100-2, OWASP Top10
for LLM, OWASP Top10 for ML) plus the zero-shot baseline. Folds are
independent and run concurrently. Zero-shot needs inference only.

The Gate 1 claim is a **delta** (+0.132 over 0.399). Re-deriving only the
fine-tuned number would measure it against a baseline from a different
environment, so the baseline is re-run too.

### Component 5 — Landing

`requirements-ml.txt` updated to the C pin; the measured `hit@1` and its
provenance committed; PRD.md, CLAUDE.md and the HF model card reconciled with
whatever the number turns out to be. All onto PR #61.

## Determinism

`tract/training/loop.py:75-79` sets seed 42, `cudnn.deterministic = True`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. PRD records a NIST-fold determinism re-run
matching exactly under the old stack.

Bit-exactness is **not** expected across a torch 2.4 → 2.13 jump. Success is
judged on whether the fold-level deltas hold and the aggregate lands within the
bootstrap CI, not on exact equality.

## Budget and safety

Ceiling $1000, speed prioritised over cost. Fastest available GPU; all folds
parallel. Controls:

- a single cheap pod validates CUDA + stack + one real training step before
  the fleet is provisioned
- spend is checked against the ceiling before provisioning and monitored during
- teardown is unconditional, including on failure
- the first fold to report is sanity-checked before the others are allowed to
  burn full duration

## Risks

| Risk | Handling |
|---|---|
| Image CUDA too old for torch 2.13 | Phase B catches it on one cheap pod; fall back to a newer digest-pinned tag |
| Training code breaks on the new stack | Already disproved locally on CPU |
| Number differs materially from 0.531 | Expected outcome, not failure; report with provenance and let the owner decide on republication |
| Pods leak and burn budget | Unconditional teardown; verify zero running pods at the end |
| RunPod capacity for 5 fast GPUs | Provisioner already falls back through an H100/A100 preference list |
