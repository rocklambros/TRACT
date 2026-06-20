# Checkpoint — validate `tract assign` lazy auto-download end-to-end (run on Jetson Orin AGX)

Date: 2026-06-17
Branch: `feature/lazy-model-autodownload` (HEAD `088fd40` at checkpoint time)
Purpose: the feature is implemented and unit-tested, but the **live model load + prediction** could not be validated on the dev Mac due to an environment-level `sentence_transformers` import deadlock (details below). Run the live end-to-end on the Jetson, which has a working CUDA ML stack.

## What this feature does

Makes `tract assign` work after a plain `pip install` with zero manual model placement: a lazy resolver downloads the pinned HuggingFace model snapshot into the HF cache on first use, verifies recorded sha256 hashes once, and `TRACTPredictor` tolerates the published flat layout. Also fixes a silent `assign --file` shuffle/truncate bug and adds `tract --version`. Design: `docs/design-lazy-model-autodownload.md`. Plan: `docs/plan-lazy-model-autodownload.md`.

## What is already validated (on the Mac)

- **All changed-code unit tests pass**: clean isolated run = **66 passed, 0 failed** across `test_config_pins`, `test_model_resolver`, `test_assign_e2e`, `test_cli`, `test_export_manifest`, `test_canonical_export` (plus the Task 6 `model_io` test). Command used:
  ```
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest \
    tests/test_config_pins.py tests/test_model_resolver.py tests/test_assign_e2e.py \
    tests/test_cli.py tests/test_export_manifest.py tests/test_canonical_export.py \
    -p no:cacheprovider -m "not integration" -q
  ```
- **Real model download works**: `snapshot_download` of pinned revision `2d2095518428b4ae88566bad43e57c9b370eba0c` pulled the full **1.3 GB** snapshot into `~/.cache/huggingface/hub/models--rockCO78--tract-cre-assignment` (no `.incomplete` files). The download half of the feature is proven.
- **mypy**: no new real type errors from our code. The repo has a large pre-existing untyped-numpy / missing-stub debt (241 errors in the documented `mypy parsers/ scripts/ --strict` gate, all pre-existing); our additions only add the benign `huggingface_hub` `import-untyped` note (that library ships no type stubs), consistent with existing usages.

## What is NOT yet validated (the reason for this checkpoint)

The live **model load + prediction** never ran on the Mac. `import sentence_transformers` **deadlocks** on this machine — an abseil mutex (`[mutex.cc:452] RAW: Lock blocking`) while loading a native extension during `sentence_transformers/cross_encoder/__init__.py`. Isolation probe results on the Mac:

| import | result |
|---|---|
| `import torch` | OK |
| `import transformers` | OK |
| `import sentence_transformers` | **DEADLOCK** (hangs in `cross_encoder/__init__.py`, abseil mutex) |

This is a broken/conflicting native install on the Mac (abseil/grpc/protobuf or libomp), **independent of our feature code** — our code runs right up to `inference.py:147` (`from tract.active_learning.model_io import load_deployment_model`), and the deadlock is inside that third-party import. It is also what stalled the full test suite (the pre-existing `test_active_learning_model_io.py::TestLoadFoldModel::test_loads_from_valid_path` loads a real model and hits the same wedge).

A Jetson with a working CUDA `sentence_transformers` should not hit this.

## Jetson validation steps

### 0. Get the branch
The branch must be pushed to GitHub first (the Mac will push it). On the Jetson:
```
cd <your TRACT checkout>            # or: git clone https://github.com/rocklambros/TRACT && cd TRACT
git fetch origin
git checkout feature/lazy-model-autodownload
git pull --ff-only origin feature/lazy-model-autodownload
```

### 1. Sanity: confirm the ML stack imports cleanly (the Mac's failure point)
```
python -c "import torch, transformers, sentence_transformers; print('ML stack OK', torch.cuda.is_available())"
```
Expect: `ML stack OK True` (no hang). If THIS hangs on the Jetson too, stop — it's the same env problem, not the feature.

### 2. Install the package (so `tract` and `huggingface_hub` are present)
Jetson torch must come from NVIDIA, not pip. Do NOT let pip reinstall torch. Either:
- If the container already has torch + transformers + sentence_transformers: `pip install -e . huggingface_hub` (installs `tract` + the now-default `huggingface_hub`; does not touch torch).
- Only if you want the pure acceptance path in a throwaway venv where pip-installed torch is acceptable: `pip install '.[phase0]'`.

### 3. THE acceptance test — real download + real prediction
Clear any prior local model dir so the lazy path is exercised, then run:
```
rm -rf results/phase1c/deployment_model        # force the resolver's download path (step 2)
tract --version                                # prints: tract <ver> \n model: rockCO78/tract-cre-assignment@2d20955...
tract assign "ensure access to systems requires multi-factor authentication"
```
Expect: first run prints `Downloading model (~1.3 GB) on first use; cached for next time…`, downloads, runs the sha256 integrity check, loads the flat-layout model, and prints a table of top-5 CRE hub assignments with calibrated confidences. Capture that output — that is the proof.

### 4. Integrity + offline behavior
```
# Re-run: should use the cache, no re-download, no re-hash (sentinel-gated)
tract assign "audit logging must capture authentication events"
# Offline with a cold cache should give the actionable error (exit code 3):
HF_HUB_OFFLINE=1 python -c "import shutil,glob,os; [shutil.rmtree(p) for p in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--rockCO78--tract-cre-assignment'))]"
HF_HUB_OFFLINE=1 tract assign "test access control"; echo "exit=$?"   # expect EXIT_OFFLINE=3 + actionable message
```

### 5. assign --file correctness (workstream B)
```
printf 'access control\n\nencryption at rest\n' > /tmp/controls.txt
tract assign --file /tmp/controls.txt --output /tmp/out.jsonl
cat /tmp/out.jsonl   # expect records in INPUT order, full untruncated text, input_index = 1 then 3 (gap at blank line 2)
```

### 6. Full test suite (install the optional deps the Mac lacked: datasets, einops)
```
pip install "datasets>=2.18,<4" "einops>=0.7,<1"
python -m pytest tests/ -q -m "not integration"          # full unit suite — expect all green
python -m pytest tests/ -q -m integration                # real-model acceptance + pin-consistency (network + GPU)
```
Note: `tests/test_model_pins_consistency.py` (integration) hashes the four pinned files from HF and asserts they match the recorded constants — proves the pin is correct.

## Report back

Paste the `tract assign` prediction table, the `tract --version` output, the `assign --file` JSONL, the offline-error exit code, and the two pytest summary lines. If anything fails, capture the traceback.

## Exact prompt to paste into Claude Code on the Jetson

> Read `docs/checkpoint-jetson-validation.md` in this repo first — it is your full context. We are on branch `feature/lazy-model-autodownload`. The lazy-model-autodownload feature was implemented and unit-tested on a Mac, but the live model load + prediction could not be validated there because `import sentence_transformers` deadlocks on that machine (an environment issue, not the code). Validate the live end-to-end here on the Jetson Orin AGX: (1) confirm `torch`+`transformers`+`sentence_transformers` import cleanly with CUDA; (2) ensure `tract` and `huggingface_hub` are installed without reinstalling the Jetson's torch; (3) run the validation steps 1–6 in the checkpoint exactly, capturing the real `tract assign` prediction output, `tract --version`, the `assign --file` JSONL ordering/index, the `HF_HUB_OFFLINE` cold-cache exit code, and the full pytest results (`-m "not integration"` and `-m integration`, after `pip install datasets einops`). Report each result with evidence (actual output/tracebacks), and tell me whether the end-to-end acceptance — fresh download → integrity check → flat-layout load → real prediction — passes. Do not modify the feature code unless a test reveals a genuine bug in our code (as opposed to an environment/dependency issue); if you find a code bug, fix it on this branch with a test and report it.
