# CI dependency fixes: scipy, the pod interpreter, and the serving stack RCEs

Branch `semantic-rebuild`, PR #62. Written 2026-08-19.

## Corrections to the brief

Two claims in the task did not survive re-derivation. Both change the fix.

**The pods were never going to die at dependency install.** The brief says
`pip install -r requirements-train.txt` on a RunPod pod fails the same way CI
does, citing the `py3.11` image defaults at `scripts/phase0/runpod_provision.py`
lines 245 and 363. Those defaults are never used by the training path.
`scripts/phase1b/runpod_parallel.py:721` passes `image=DOCKER_IMAGE`
explicitly, and that digest is
`1.1.0-cu1300-torch291-ubuntu2404-cluster`, whose system interpreter is Python
**3.12** (`runpod_parallel.py:773` names it, and the 2026-08-14 canary confirmed
it by tripping PEP 668). `requirements-train.txt` is installed in exactly one
place, `_bootstrap_pod`, on that image. Measured: the pre-fix file resolves
cleanly under `pip install --dry-run --python-version 3.12`. The two scripts
that do use the 3.11 image defaults, `scripts/phase1c/runpod_retrain.py` and
`scripts/phase0/runpod_orchestrate.py`, install their own unpinned sets and
never read `requirements-train.txt`.

So no GPU money was at risk from this pin. The broken environment was the CI
job, and the real defect is that the job claiming to prove the pod install ran
an interpreter no pod has.

**The audit findings were understated by 15.** The quoted `pip-audit` log lists
transformers and datasets only. Re-running it against the pre-fix
`requirements-ml.txt` returns **41** findings, of which **15** are torch 2.4.1.
One of those is `PYSEC-2026-2286`, arbitrary code execution in `torch.load`'s
`weights_only` unpickler, which maps directly onto what TRACT does: download an
artifact from the Hub and load it. That is a more reachable flaw than any of the
three transformers RCEs the brief centres on.

## Failure 1: scipy

**Pinned to `scipy==1.17.1`** in both `requirements-train.txt` and
`requirements-ml.txt`, not to a Python move.

Re-derived from PyPI: `scipy==1.18.0` declares `Requires-Python: >=3.12`,
`1.17.1` declares `>=3.11`, and 1.17.1 is the last release with that floor.
`pyproject.toml` declares `requires-python = ">=3.11"`, and the CI test matrix
runs 3.11. A serving requirements file that cannot install on a Python the
package claims to support is a defect on its own, independent of any CI job, so
`requirements-ml.txt` was forced to 1.17.1 regardless. Holding both files at one
scipy also keeps calibration numerics identical between the stack that fits a
temperature and the stack that applies it.

Moving the pod image was rejected because the pod image is already 3.12 and
already correct for `torch==2.13.0`; there was nothing to move. Verified rather
than reasoned:

- `pip install --dry-run --python-version 3.11 --only-binary=:all: -r requirements-train.txt` → exit 0
- the same on 3.12 → exit 0
- the same file with `scipy==1.18.0` restored, on 3.11 → reproduces the CI error verbatim, listing availability up to 1.17.1
- `torch==2.13.0` publishes cp311 through cp315 wheels and `scikit-learn==1.9.0` declares `>=3.11`, so scipy was the only 3.11 blocker

**The CI `training-stack` job moved 3.11 → 3.12**, because the pods run 3.12.
That is the fix for the actual defect, not for scipy.

## What the preflight now catches

`_preflight_training_stack()` gained an INSTALL check that runs ahead of the two
import checks. It parses every exact pin out of `requirements-train.txt`, reads
each release's `Requires-Python` from PyPI, and refuses to provision when any
floor excludes `POD_PYTHON_VERSION`. New constant `POD_PYTHON_VERSION = "3.12"`
sits beside `DOCKER_IMAGE`, because the interpreter is a property of the digest.

It catches the class "a pin in the training file cannot install on the pod's
interpreter", reporting every violation at once rather than one per run. Against
the pre-fix tree it would have reported **nothing**, correctly: the pods run
3.12 and `scipy==1.18.0` installs there. The check that would have caught this
specific defect is the second one, a test asserting the `training-stack` job's
`python-version` equals `POD_PYTHON_VERSION`. That is what was actually wrong:
the job's PASS and its FAIL both said nothing about the fleet.

Fails closed. A PyPI outage refuses the provision rather than assuming the pins
are fine, which is the right trade in front of a billable resource.

## Failure 2: transformers and datasets

Established from distribution metadata, not memory.

**Cleared by bumping, in `requirements-ml.txt`:**

| pin | from | to | effect |
|---|---|---|---|
| torch | 2.4.1 | 2.13.0 | −15 findings, including `PYSEC-2026-2286` (ACE in `torch.load`) |
| transformers | 4.45.1 | 4.57.6 | −14, the whole ReDoS set |
| tokenizers | 0.20.0 | 0.22.2 | forced by `transformers 4.57.6` (`tokenizers<=0.23.0,>=0.22.0`) |
| datasets | 3.6.0 | 5.0.1 | −1, `PYSEC-2026-3716` path traversal |

41 findings → 5. torch 2.13.0 is the *lowest* release clearing every torch
finding: 2.10.0 leaves two, 2.12.1 leaves one plus two transitive setuptools.

**Not clearable. The specific constraint:** `sentence-transformers==3.2.0`
declares `transformers<5.0.0,>=4.41.0`, so the three RCE fixes (5.0.0, 5.3.0,
5.5.0) are out of reach without moving sentence-transformers itself. Moving it
is not sufficient either: `transformers>=5.0.0` requires
`huggingface-hub>=1.3.0` and `>=5.5.0` requires `>=1.5.0`, while
`pyproject.toml` declares `huggingface_hub>=0.24,<1` as a **core** dependency of
tract. The 4→5 boundary is exactly the hub 0.x→1.x boundary.

**Upgrade path**, in order, as one change and not inside a pin repair:
migrate the six `huggingface_hub` call sites including `tract/model_resolver.py`
(the sha256-verified download behind `tract assign`) to hub 1.x → bump
sentence-transformers past its `transformers<5` cap and add the new layout to
`tract/training/st_compat.py:SYMBOL_PATHS` → bump transformers to ≥5.5.0 → drop
all four suppressions.

**Residual, suppressed with a named argument and an expiry of 2026-11-17**, in
`tract/supply_chain.py:AUDIT_SUPPRESSIONS`:

- `PYSEC-2025-217` — X-CLIP checkpoint conversion. No fix exists in any release. No `x_clip` reference in `tract/` or `scripts/`.
- `PYSEC-2026-2288` — `Trainer._load_rng_state`. Needs `resume_from_checkpoint`, which appears nowhere; the advisory's own precondition is torch < 2.6 and both stacks now carry 2.13.0.
- `PYSEC-2026-2289` — **mitigated, not unreachable.** `from_pretrained` is reached through sentence-transformers, so the path exists. `tract/model_resolver.py` verifies a recorded sha256 per file, so `config.json` cannot change on the default repo and revision. The mitigation lapses under a `TRACT_MODEL_REPO` / `TRACT_MODEL_REVISION` override, which model_resolver logs as revision-trust only. This is the one that should drive the upgrade.
- `PYSEC-2026-2290` — LightGlue. `LightGlueConfig` is instantiated only for `model_type: lightglue`; every encoder here is bert, modernbert or xlm-roberta.

The expiry is enforced, not documented: `expired_suppressions(date.today())` is
asserted empty by `tests/test_supply_chain.py`, and a second test asserts the
workflow's `--ignore-vuln` set equals the registry exactly, so neither a blanket
ignore nor silent drift passes.

## Verification

- `pip-audit --desc -r requirements-ml.txt` with the four flags → "No known vulnerabilities found, 5 ignored", exit 0
- `pip-audit --desc -r requirements.txt` → clean with the new `packaging==26.3` pin
- isolated 3.12 venv: torch 2.13.0 + transformers 4.57.6 + sentence-transformers 3.2.0 + tokenizers 0.22.2 + hf_hub 0.36.2 + datasets 5.0.1 import together, and `Dataset.from_list` round-trips. Nothing installed into the working environment; no model loaded.
- `mypy --strict` over the CI scope: 157 files, clean. `ruff`: clean.
- suite: **2,407 passed**, 9 failed, 23 skipped, 3 xpassed. Baseline before the change was 2,358 passed with the same 9 `datasets` / model-loading environmental failures.

## Mutation testing

16 mutants, 16 killed, no survivors. Each is a plausible wrong implementation of
one new assertion: parser skips bad lines rather than raising; parser drops PEP
503 name normalisation; conflicting duplicate pins accepted; an empty
`Requires-Python` treated as admitting nothing; only the first specifier clause
honoured; violation scan returns after the first hit; violations reported
unsorted; the admission test inverted; expiry compared with `>` instead of
`>=`; network failure swallowed into "no floor"; the install check reordered
behind the layout check; the refusal downgraded to a warning; the CI job put
back on 3.11; one suppression flag dropped from the workflow; `|| true` appended
to the audit; scipy reverted to 1.18.0.

## What the install fix uncovered, and why it is not patched here

`training-stack` is still red, on a different failure. Fixing the install moved
the job from dying at `pip install` to running the regression tests for the
first time, and two of them fail:

- `tests/test_training_loop.py::TestLoRACheckpointPersistence::test_adapter_survives_save_and_reload`
- `tests/test_publish_merge.py::TestMergeRealAdapter::test_merges_adapter_only_checkpoint`

Both raise the same error: `ValueError: Unrecognized model in <dir>. Should have
a model_type key in its config.json`.

**Not caused by this change.** `main` carries neither the `training-stack` job
nor `requirements-train.txt`, so these two tests have never executed in CI
anywhere. On this branch the previous run's "Run the training regression tests"
step is recorded as `skipped`, because the job died at install. The repins in
this commit were scipy and datasets; the failing call is
`SentenceTransformer(<adapter-only dir>)` resolving a transformers `AutoConfig`,
which neither touches.

**One root cause, two call sites.** A LoRA checkpoint written by `model.save()`
is adapter-only: `adapter_config.json` plus `adapter_model.safetensors`, no base
config and no base weights. `tract/training/loop.py:315:_reload_saved_model`
already documents this and works around it by loading the base named inside the
adapter config and attaching the adapter. Two places bypass that routine and
call the constructor directly:

- `tract/publish/merge.py:95` — `model = SentenceTransformer(str(model_dir))`, the first line of `merge_lora_adapters`. It dies before ever reaching its own adapter-only branch at lines 118-128, which was written for exactly this case.
- `tests/test_training_loop.py:200` — asserts a naked `SentenceTransformer(saved)` reproduces the adapted model, which is the contract `loop.py`'s docstring says sentence-transformers does not offer for an adapter-only directory.

**The workaround is proven to work under this exact stack.**
`test_verify_checkpoint_roundtrip_passes_on_match` passed in the same CI run,
and it goes through `verify_checkpoint_roundtrip` → `_reload_saved_model` on an
adapter-only checkpoint. So the reload strategy is sound and only the two call
sites above are wrong.

**Fix shape**, not applied here. `merge_lora_adapters` needs the *unmerged*
adapter model to compute its pre-merge reference embeddings, so it cannot simply
call `_reload_saved_model`, which merges eagerly. It needs the same load
stopping one step earlier: build the SentenceTransformer from
`base_model_name_or_path`, attach `PeftModel.from_pretrained(...)` to the
backbone with `_set_backbone` without merging, then let the existing
`hasattr(inner, "merge_and_unload")` branch do the merge and the verification.

**Why it is not patched here.** It is a production change to the LoRA publish
path, which project memory flags as the defect that "corrupts artifacts
independently of everything else". Verifying it needs a real model load, which
the standing rules put on a pod and off this machine, so any patch would be a
blind push into the artifact path. It is a distinct defect from the three in the
brief and wants its own change with a pod behind it.

## Concerns

1. **The serving stack's reproduction claim is now unproven, and the file says
   so.** `tokenizers` moved 0.20.0 → 0.22.2 because transformers 4.57.6 requires
   it. By `requirements-ml.txt`'s own argument a tokenizer change can move
   embeddings and therefore the confidences published in the crosswalk dataset.
   Re-deriving hit@1 on this stack needs a GPU and was not done. The trade taken
   was: an unproven numerical equivalence beats a live ACE in the load path.
   Owner decision if that is wrong.
2. **`PYSEC-2026-2289` is a mitigation, not unreachability.** It is suppressed
   on the strength of a sha256 pin that a documented environment variable turns
   off.
3. `sentence-transformers==3.2.0` with `transformers==4.57.6` resolves and
   imports, verified in a clean venv. It has not been exercised against a real
   model load, which needs a pod.
4. The expiry lands on 2026-11-17 and will fail the suite that day by design.
5. **`training-stack` is still red**, on the LoRA checkpoint defect above rather
   than on anything in this commit. Seven of eight checks pass: `lint`,
   `model-pins`, `test (3.11)`, `test (3.12)`, `audit`, `CodeQL` and
   `Analyze Python`. Inside `training-stack` the install, the pin assertion and
   the import proof all pass now, and 157 of 159 regression tests pass.
