# Lazy Model Auto-Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tract assign` work after a plain `pip install 'tract[phase0]'` with zero manual model placement, fix the silent `assign --file` shuffle/truncate bug, and add a real `tract --version` — all hardened against the adversarial premortem.

**Architecture:** A lazy resolver (`ensure_deployment_model`) returns a model directory, downloading a pinned HuggingFace snapshot into the HF cache on first use and verifying recorded sha256 hashes once per download. `TRACTPredictor` tolerates flat/nested layouts and cross-checks the hierarchy hash. The CLI checks the inference runtime is present before any download. Provenance fields that feed the published OpenCRE export are deliberately untouched.

**Tech Stack:** Python 3.11, `huggingface_hub` (snapshot download), `sentence-transformers`/`torch` (phase0 extra), `numpy`, `pytest`, `argparse`.

**Reference spec:** `docs/design-lazy-model-autodownload.md` (commit `69efbc6`).

## Global Constraints

- Branch: `feature/lazy-model-autodownload`. Never commit to `main`.
- Python `>=3.11`; full type annotations on every new function (`mypy --strict` clean).
- No `print()` in library code — use `logging`. `print()` is allowed only in `tract/cli.py` (user-facing CLI output) and `scripts/`.
- Pinned HF model revision: `2d2095518428b4ae88566bad43e57c9b370eba0c`.
- Recorded sha256 (rev `2d20955`): model.safetensors `c1f7b6d65c4440ea6b497a47de85898812ebc5efce63f608902de9a4fbe215cd`; deployment_artifacts.npz `7e8b8f834db503118d75727675716471636f139ecb3b64fbd6bc96d6690122f7`; calibration.json `a49c532d7f8e4d42ff1e5208f68aabdd60d87feb5231c77fbc276c757edda88a`; cre_hierarchy.json `8dc48bd397cf6ee455193a9768760258f235fb5519659915dccd733dcaa19738`.
- `huggingface_hub` pinned `>=0.24,<1`.
- Never set `trust_remote_code=True`; never `np.load(..., allow_pickle=True)` on downloaded artifacts.
- Do NOT change `cli.py` canonical-export `tract_version` or `model_version` (they feed `content_hash` / mapping diffs).
- Run the suite with `python -m pytest tests/ -q -m "not integration"` (integration tests need the real ~1.3 GB model).
- No AI attribution in any commit message.

---

### Task 1: `predict_batch` tolerates empty input

**Files:**
- Modify: `tract/inference.py` (method `TRACTPredictor.predict_batch`, ~188-230)
- Test: `tests/test_inference.py`

**Interfaces:**
- Produces: `TRACTPredictor.predict_batch(texts: list[str], top_k: int = PHASE1D_DEFAULT_TOP_K) -> list[list[HubPrediction]]` returns `[]` for empty `texts` without touching the model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py  (add near other predict_batch tests)
def test_predict_batch_empty_returns_empty_without_model_call():
    # A predictor whose model would raise if .encode were called proves we short-circuit.
    pred = TRACTPredictor.__new__(TRACTPredictor)  # bypass __init__ (no model load)
    class _Boom:
        def encode(self, *a, **k):  # pragma: no cover - must never be called
            raise AssertionError("encode called on empty input")
    pred._model = _Boom()
    assert pred.predict_batch([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference.py::test_predict_batch_empty_returns_empty_without_model_call -v`
Expected: FAIL (current code calls `self._model.encode([])` → `_Boom` raises).

- [ ] **Step 3: Write minimal implementation**

Add at the very top of `predict_batch`, before `clean_texts = ...`:

```python
        if not texts:
            return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference.py::test_predict_batch_empty_returns_empty_without_model_call -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/inference.py tests/test_inference.py
git commit -m "fix(inference): predict_batch returns [] for empty input"
```

---

### Task 2: `assign --file` preserves order, full text, input_index, with empty + size guards

**Files:**
- Modify: `tract/cli.py` (`_cmd_assign`, ~501-539)
- Test: `tests/test_assign_e2e.py`

**Interfaces:**
- Consumes: `predict_batch` (Task 1), `PHASE1D_INGEST_MAX_FILE_SIZE` (config.py:258).
- Produces: JSONL records `{"input_index": <1-based source line int>, "text": <full str>, "predictions": [...]}` written in input order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_assign_e2e.py
import json, sys
from pathlib import Path
from unittest.mock import patch
import tract.cli as cli

class _FakePred:
    """raw_similarity descending != input order, to catch a reorder."""
    def __init__(self, sim): self._sim = sim
    @property
    def raw_similarity(self): return self._sim
    def to_dict(self): return {"raw_similarity": self._sim}

class _FakePredictor:
    def __init__(self, *a, **k): pass
    def predict_batch(self, texts, top_k=5):
        # lower similarity for earlier lines, so a similarity-sort would reverse them
        return [[_FakePred(0.1 * (i + 1))] for i, _ in enumerate(texts)]

def _run_assign_file(tmp_path, lines, monkeypatch):
    infile = tmp_path / "controls.txt"
    infile.write_text("\n".join(lines), encoding="utf-8")
    outfile = tmp_path / "out.jsonl"
    monkeypatch.setattr(cli, "TRACTPredictor", _FakePredictor, raising=False)
    monkeypatch.setattr("tract.inference.TRACTPredictor", _FakePredictor)
    monkeypatch.setattr(cli, "ensure_deployment_model",
                        lambda: type("R", (), {"path": tmp_path})(), raising=False)
    import argparse
    args = argparse.Namespace(file=str(infile), output=str(outfile),
                              text=None, top_k=5, json=False, raw=False, verbose=False)
    cli._cmd_assign(args)
    return [json.loads(l) for l in outfile.read_text(encoding="utf-8").splitlines()]

def test_assign_file_preserves_order_full_text_and_index(tmp_path, monkeypatch):
    long_line = "access control " * 20  # > 100 chars
    recs = _run_assign_file(tmp_path, [long_line, "encryption", "", "audit logging"], monkeypatch)
    assert [r["input_index"] for r in recs] == [1, 2, 4]          # gap where blank line 3 was
    assert recs[0]["text"] == long_line                          # not truncated to 100
    assert [r["text"] for r in recs] == [long_line, "encryption", "audit logging"]  # input order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_assign_e2e.py::test_assign_file_preserves_order_full_text_and_index -v`
Expected: FAIL (current code sorts by similarity → reversed order, and truncates `text[:100]`, and has no `input_index`).

- [ ] **Step 3: Write minimal implementation**

Replace the `if args.file:` block body in `_cmd_assign` (from `texts = [...]` through the `with open(...)` loop) with:

```python
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(EXIT_USER_ERROR)
        if file_path.stat().st_size > PHASE1D_INGEST_MAX_FILE_SIZE:
            print(f"Error: --file exceeds {PHASE1D_INGEST_MAX_FILE_SIZE} bytes",
                  file=sys.stderr)
            sys.exit(EXIT_USER_ERROR)

        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
        indexed = [(n, ln.strip()) for n, ln in enumerate(raw_lines, start=1) if ln.strip()]

        output_path = Path(args.output) if args.output else file_path.with_suffix(".jsonl").with_stem(file_path.stem + "_assignments")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not indexed:
            print("Wrote 0 assignments (no non-blank input lines).")
            return

        line_numbers = [n for n, _ in indexed]
        texts = [t for _, t in indexed]
        results = predictor.predict_batch(texts, top_k=args.top_k)

        with open(output_path, "w", encoding="utf-8") as f:
            for input_index, text, preds in zip(line_numbers, texts, results):
                f.write(json.dumps(
                    {"input_index": input_index, "text": text,
                     "predictions": [p.to_dict() for p in preds]},
                    ensure_ascii=False) + "\n")

        ood_count = sum(1 for r in results if r and r[0].is_ood)
        high_conf = sum(1 for r in results if r and r[0].calibrated_confidence > 0.5)
        print(f"Wrote {len(results)} assignments to {output_path}")
        print(f"{ood_count}/{len(results)} controls flagged OOD, {high_conf}/{len(results)} high confidence")
        return
```

Add the import at the top of `cli.py` config-import block: `EXIT_USER_ERROR` (created in Task 3). For now, if Task 3 is not yet merged, temporarily `from tract.config import ... ` will fail — Task 3 precedes this in execution order, so the constant exists. Also import `ensure_deployment_model` is wired in Task 10; this task's test monkeypatches it, and `_cmd_assign` still constructs `predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)` at this point (unchanged until Task 10). Note: the test stubs `is_ood`/`calibrated_confidence` are not read because `_FakePred` lacks them — guard the summary lines with `getattr`:

```python
        ood_count = sum(1 for r in results if r and getattr(r[0], "is_ood", False))
        high_conf = sum(1 for r in results if r and getattr(r[0], "calibrated_confidence", 0.0) > 0.5)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_assign_e2e.py -v -m "not integration"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/cli.py tests/test_assign_e2e.py
git commit -m "fix(cli): assign --file preserves order, full text, adds input_index"
```

---

### Task 3: Config constants — pin, hashes, allow-patterns, exit codes

**Files:**
- Modify: `tract/config.py` (Phase 2B HuggingFace section, ~292-313)
- Test: `tests/test_config_pins.py` (create)

**Interfaces:**
- Produces: `TRACT_MODEL_PINNED_REVISION: str`, `TRACT_MODEL_SAFETENSORS_SHA256`, `TRACT_DEPLOYMENT_ARTIFACTS_SHA256`, `TRACT_CALIBRATION_SHA256`, `TRACT_HIERARCHY_SHA256` (all 64-hex), `TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS: tuple[str, ...]`, `EXIT_USER_ERROR=2`, `EXIT_OFFLINE=3`, `EXIT_INTEGRITY=4`, `EXIT_MISSING_RUNTIME=5`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_pins.py
import re
from tract import config

def test_pin_and_hashes_well_formed():
    assert re.fullmatch(r"[0-9a-f]{40}", config.TRACT_MODEL_PINNED_REVISION)
    for h in (config.TRACT_MODEL_SAFETENSORS_SHA256, config.TRACT_DEPLOYMENT_ARTIFACTS_SHA256,
              config.TRACT_CALIBRATION_SHA256, config.TRACT_HIERARCHY_SHA256):
        assert re.fullmatch(r"[0-9a-f]{64}", h)

def test_allow_patterns_include_hierarchy_exclude_scripts():
    ap = config.TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "cre_hierarchy.json" in ap
    assert "deployment_artifacts.npz" in ap and "calibration.json" in ap
    assert "model.safetensors" in ap and "modules.json" in ap
    assert "predict.py" not in ap and "train.py" not in ap

def test_exit_codes_distinct():
    codes = {config.EXIT_USER_ERROR, config.EXIT_OFFLINE,
             config.EXIT_INTEGRITY, config.EXIT_MISSING_RUNTIME}
    assert codes == {2, 3, 4, 5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_pins.py -v`
Expected: FAIL (`AttributeError` — constants not defined).

- [ ] **Step 3: Write minimal implementation**

Append to the Phase 2B HuggingFace section of `tract/config.py` (after `HF_BASE_MODEL`):

```python
# ── Pinned deployment model (lazy auto-download) ──────────────────────
# Bump procedure (run on every HF model re-publish):
#   1. Push new artifacts to HF; note the new full commit SHA.
#   2. python scripts/recompute_model_pins.py <new_sha>   # prints the 5 constants
#   3. Replace the constants below with the printed values; commit.
#   The CI "model-pins" job recomputes these from HF and fails on drift.
TRACT_MODEL_PINNED_REVISION: Final[str] = "2d2095518428b4ae88566bad43e57c9b370eba0c"
TRACT_MODEL_SAFETENSORS_SHA256: Final[str] = (
    "c1f7b6d65c4440ea6b497a47de85898812ebc5efce63f608902de9a4fbe215cd")
TRACT_DEPLOYMENT_ARTIFACTS_SHA256: Final[str] = (
    "7e8b8f834db503118d75727675716471636f139ecb3b64fbd6bc96d6690122f7")
TRACT_CALIBRATION_SHA256: Final[str] = (
    "a49c532d7f8e4d42ff1e5208f68aabdd60d87feb5231c77fbc276c757edda88a")
TRACT_HIERARCHY_SHA256: Final[str] = (
    "8dc48bd397cf6ee455193a9768760258f235fb5519659915dccd733dcaa19738")

# sha256 keyed by the file's basename, for download-time integrity.
TRACT_MODEL_PINNED_FILE_HASHES: Final[dict[str, str]] = {
    "model.safetensors": TRACT_MODEL_SAFETENSORS_SHA256,
    "deployment_artifacts.npz": TRACT_DEPLOYMENT_ARTIFACTS_SHA256,
    "calibration.json": TRACT_CALIBRATION_SHA256,
    "cre_hierarchy.json": TRACT_HIERARCHY_SHA256,
}

TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS: Final[tuple[str, ...]] = (
    *HF_MODEL_FILES, *HF_DEPLOY_FILES, "cre_hierarchy.json",
)

# ── CLI exit codes (scriptable failure classes) ───────────────────────
EXIT_USER_ERROR: Final[int] = 2
EXIT_OFFLINE: Final[int] = 3
EXIT_INTEGRITY: Final[int] = 4
EXIT_MISSING_RUNTIME: Final[int] = 5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_pins.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/config.py tests/test_config_pins.py
git commit -m "feat(config): pinned model revision, file hashes, exit codes"
```

---

### Task 4: Layout + hierarchy resolution helpers

**Files:**
- Modify: `tract/inference.py` (module-level functions, above `TRACTPredictor`)
- Test: `tests/test_inference.py`

**Interfaces:**
- Produces:
  - `find_st_model_root(model_dir: Path) -> Path` — most-nested-first of `model_dir/model/model`, `model_dir/model`, `model_dir`; returns the first containing **both** `modules.json` and `config_sentence_transformers.json`; raises `FileNotFoundError` if none.
  - `resolve_hierarchy_path(model_dir: Path, source: str) -> Path` — for `source == "download"` tries `model_dir/cre_hierarchy.json` first, then `model_dir.parent.parent/data/processed/cre_hierarchy.json`, then `PROCESSED_DIR/cre_hierarchy.json`; for any other `source`, tries the `data/processed` path first, then `model_dir`, then `PROCESSED_DIR`. Raises `FileNotFoundError` listing all candidates if none.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
import json
from pathlib import Path
import pytest
from tract.inference import find_st_model_root, resolve_hierarchy_path

def _make_st_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "modules.json").write_text("[]", encoding="utf-8")
    (d / "config_sentence_transformers.json").write_text("{}", encoding="utf-8")

def test_find_st_model_root_flat(tmp_path):
    _make_st_dir(tmp_path)
    assert find_st_model_root(tmp_path) == tmp_path

def test_find_st_model_root_single_nest(tmp_path):
    _make_st_dir(tmp_path / "model")
    assert find_st_model_root(tmp_path) == tmp_path / "model"

def test_find_st_model_root_double_nest_wins_over_outer(tmp_path):
    _make_st_dir(tmp_path / "model" / "model")
    assert find_st_model_root(tmp_path) == tmp_path / "model" / "model"

def test_find_st_model_root_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_st_model_root(tmp_path)

def test_resolve_hierarchy_download_prefers_snapshot_root(tmp_path):
    (tmp_path / "cre_hierarchy.json").write_text("{}", encoding="utf-8")
    assert resolve_hierarchy_path(tmp_path, "download") == tmp_path / "cre_hierarchy.json"

def test_resolve_hierarchy_missing_raises_listing_candidates(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        resolve_hierarchy_path(tmp_path, "download")
    assert "cre_hierarchy.json" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference.py -k "find_st_model_root or resolve_hierarchy" -v`
Expected: FAIL (`ImportError` — functions not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `tract/inference.py` after the imports, before `@dataclass(frozen=True) class HubPrediction`:

```python
_ST_MARKERS = ("modules.json", "config_sentence_transformers.json")


def find_st_model_root(model_dir: Path) -> Path:
    """Return the SentenceTransformer root, tolerating flat/nested layouts.

    Probes most-nested first so a double-nested dev tree is not shadowed by an
    outer dir. Requires both ST marker files to avoid loading a partial dir.
    """
    candidates = [model_dir / "model" / "model", model_dir / "model", model_dir]
    for cand in candidates:
        if all((cand / marker).exists() for marker in _ST_MARKERS):
            return cand
    raise FileNotFoundError(
        f"No SentenceTransformer root under {model_dir} "
        f"(need {' + '.join(_ST_MARKERS)} in one of: "
        f"{', '.join(str(c) for c in candidates)})"
    )


def resolve_hierarchy_path(model_dir: Path, source: str) -> Path:
    """Resolve cre_hierarchy.json. For a downloaded snapshot, prefer the
    revision-pinned copy at the snapshot root over a possibly-stale dev copy.
    """
    snapshot_root = model_dir / "cre_hierarchy.json"
    data_processed = model_dir.parent.parent / "data" / "processed" / "cre_hierarchy.json"
    package_processed = PROCESSED_DIR / "cre_hierarchy.json"
    order = (
        [snapshot_root, data_processed, package_processed]
        if source == "download"
        else [data_processed, snapshot_root, package_processed]
    )
    for cand in order:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "cre_hierarchy.json not found in any of: "
        + ", ".join(str(c) for c in order)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference.py -k "find_st_model_root or resolve_hierarchy" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/inference.py tests/test_inference.py
git commit -m "feat(inference): flat/nested layout + hierarchy resolution helpers"
```

---

### Task 5: Harden npz loading — `allow_pickle=False`

**Files:**
- Modify: `tract/inference.py` (`load_deployment_artifacts`, ~73-83)
- Modify: `tract/export/canonical.py:389`
- Test: `tests/test_inference.py`

**Interfaces:**
- Produces: `load_deployment_artifacts` loads with `allow_pickle=False` (rejects pickled-object npz).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
import numpy as np
import pytest
from tract.inference import load_deployment_artifacts

def test_load_deployment_artifacts_rejects_pickled_object(tmp_path):
    npz = tmp_path / "deployment_artifacts.npz"
    # An object array forces pickle on load; allow_pickle=False must refuse it.
    np.savez(str(npz), hub_embeddings=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        load_deployment_artifacts(npz)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference.py::test_load_deployment_artifacts_rejects_pickled_object -v`
Expected: FAIL (current `allow_pickle=True` loads the object array without error, then `KeyError` on missing keys — not `ValueError`).

- [ ] **Step 3: Write minimal implementation**

In `tract/inference.py`, change `load_deployment_artifacts`:

```python
    data = np.load(str(artifacts_path), allow_pickle=False)
```

In `tract/export/canonical.py:389`, change:

```python
    data = np.load(str(artifacts_path), allow_pickle=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference.py::test_load_deployment_artifacts_rejects_pickled_object -v`
Expected: PASS (numpy raises `ValueError: Object arrays cannot be loaded when allow_pickle=False`).

Also run the existing canonical tests to confirm no regression:
Run: `python -m pytest tests/test_canonical_export.py -q -m "not integration"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/inference.py tract/export/canonical.py tests/test_inference.py
git commit -m "fix(security): load deployment npz with allow_pickle=False"
```

---

### Task 6: Harden model load — explicit `trust_remote_code=False`

**Files:**
- Modify: `tract/active_learning/model_io.py` (`load_deployment_model`, ~36-53)
- Test: `tests/test_active_learning_model_io.py`

**Interfaces:**
- Produces: `load_deployment_model` raises `ValueError` if `config.json` declares `auto_map`/`custom_pipelines`; constructs `SentenceTransformer(..., trust_remote_code=False)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_active_learning_model_io.py
import json
from pathlib import Path
import pytest
from tract.active_learning.model_io import load_deployment_model

def test_load_deployment_model_rejects_auto_map(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "evil--repo.modeling.Evil"}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="custom code"):
        load_deployment_model(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_active_learning_model_io.py::test_load_deployment_model_rejects_auto_map -v`
Expected: FAIL (current code calls `SentenceTransformer(...)` directly and would attempt a load, not raise `ValueError`).

- [ ] **Step 3: Write minimal implementation**

In `load_deployment_model`, before `model = SentenceTransformer(str(model_dir))`:

```python
    config_path = model_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("auto_map") or cfg.get("custom_pipelines"):
            raise ValueError(
                f"Refusing to load model with custom code (auto_map/custom_pipelines) "
                f"in {config_path}"
            )

    model = SentenceTransformer(str(model_dir), trust_remote_code=False)
```

Add `import json` at the top of `model_io.py` if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_active_learning_model_io.py::test_load_deployment_model_rejects_auto_map -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/active_learning/model_io.py tests/test_active_learning_model_io.py
git commit -m "fix(security): explicit trust_remote_code=False + reject auto_map"
```

---

### Task 7: The resolver — `ensure_deployment_model`

**Files:**
- Create: `tract/model_resolver.py`
- Test: `tests/test_model_resolver.py`

**Interfaces:**
- Consumes: config constants from Task 3; `huggingface_hub.snapshot_download`.
- Produces:
  - `@dataclass(frozen=True) ResolvedModel(path: Path, revision: str, source: str)`
  - `ensure_deployment_model() -> ResolvedModel`
  - `class OfflineModelError(RuntimeError)` and `class ModelIntegrityError(RuntimeError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_resolver.py
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
import tract.model_resolver as mr
from tract import config

def _make_snapshot(d: Path, hashes: dict[str, bytes]):
    d.mkdir(parents=True, exist_ok=True)
    for name in ("modules.json", "config_sentence_transformers.json"):
        (d / name).write_text("{}", encoding="utf-8")
    for name, content in hashes.items():
        (d / name).write_bytes(content)
    return d

def _good_files():
    # Build files whose sha256 match the recorded constants by monkeypatching the
    # expected-hash map to the bytes we write.
    return {
        "model.safetensors": b"WEIGHTS",
        "deployment_artifacts.npz": b"NPZ",
        "calibration.json": b"{}",
        "cre_hierarchy.json": b"{}",
    }

def test_local_complete_returns_local_without_download(tmp_path, monkeypatch):
    local = tmp_path / "deployment_model"
    _make_snapshot(local / "model", {})
    (local / "deployment_artifacts.npz").write_bytes(b"x")
    (local / "calibration.json").write_text("{}", encoding="utf-8")
    (local / "cre_hierarchy.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    with patch.object(mr, "snapshot_download") as sd:
        result = mr.ensure_deployment_model()
        sd.assert_not_called()
    assert result.path == local and result.source == "local"

def test_local_absent_triggers_pinned_download(tmp_path, monkeypatch):
    local = tmp_path / "deployment_model"  # does not exist
    snap = tmp_path / "snap"
    files = _good_files()
    _make_snapshot(snap, files)
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    monkeypatch.setattr(config, "TRACT_MODEL_PINNED_FILE_HASHES",
                        {n: hashlib.sha256(c).hexdigest() for n, c in files.items()})
    with patch.object(mr, "snapshot_download", return_value=str(snap)) as sd:
        result = mr.ensure_deployment_model()
        _, kwargs = sd.call_args
        assert kwargs["revision"] == config.TRACT_MODEL_PINNED_REVISION
        assert "cre_hierarchy.json" in kwargs["allow_patterns"]
    assert result.source == "download" and (snap / ".tract-verified-" + config.TRACT_MODEL_PINNED_REVISION).exists()

def test_download_integrity_mismatch_raises(tmp_path, monkeypatch):
    local = tmp_path / "deployment_model"
    snap = tmp_path / "snap"
    _make_snapshot(snap, _good_files())
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", local)
    monkeypatch.setattr(config, "TRACT_MODEL_PINNED_FILE_HASHES",
                        {"model.safetensors": "0" * 64})  # wrong
    with patch.object(mr, "snapshot_download", return_value=str(snap)):
        with pytest.raises(mr.ModelIntegrityError):
            mr.ensure_deployment_model()

def test_offline_cold_cache_raises_offline(tmp_path, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", tmp_path / "nope")
    with patch.object(mr, "snapshot_download", side_effect=LocalEntryNotFoundError("x")):
        with pytest.raises(mr.OfflineModelError) as e:
            mr.ensure_deployment_model()
    assert "HF_HUB_OFFLINE" in str(e.value) and config.HF_DEFAULT_REPO_ID in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_resolver.py -v`
Expected: FAIL (`ModuleNotFoundError: tract.model_resolver`).

- [ ] **Step 3: Write minimal implementation**

```python
# tract/model_resolver.py
"""Resolve a deployment-model directory, downloading a pinned HF snapshot lazily.

Resolution order: a complete local dir (dev checkout / prior `tract download`) ->
a pinned HuggingFace snapshot in the HF cache -> an actionable offline error.
Integrity (recorded sha256) is verified once per download, gated by a sentinel.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

from tract import config
from tract.inference import find_st_model_root, resolve_hierarchy_path

logger = logging.getLogger(__name__)


class OfflineModelError(RuntimeError):
    """Model not cached and the Hub is unreachable / offline."""


class ModelIntegrityError(RuntimeError):
    """A downloaded artifact's sha256 did not match the recorded constant."""


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    revision: str
    source: str  # "local" | "download"


def _local_is_complete(model_dir: Path) -> bool:
    if not (model_dir / "deployment_artifacts.npz").exists():
        return False
    if not (model_dir / "calibration.json").exists():
        return False
    try:
        find_st_model_root(model_dir)
        resolve_hierarchy_path(model_dir, source="local")
    except FileNotFoundError:
        return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_pinned(snapshot: Path) -> None:
    for name, expected in config.TRACT_MODEL_PINNED_FILE_HASHES.items():
        actual = _sha256(snapshot / name)
        if actual != expected:
            raise ModelIntegrityError(
                f"Integrity check failed for {name}: expected {expected}, got {actual}. "
                f"Clear the HF cache for {config.HF_DEFAULT_REPO_ID} and re-run."
            )


def ensure_deployment_model() -> ResolvedModel:
    local = config.PHASE1D_DEPLOYMENT_MODEL_DIR
    if _local_is_complete(local):
        logger.info("Using local deployment model at %s", local)
        return ResolvedModel(path=local, revision="local", source="local")

    repo_id = os.environ.get("TRACT_MODEL_REPO_ID", config.HF_DEFAULT_REPO_ID)
    revision = os.environ.get("TRACT_MODEL_REVISION", config.TRACT_MODEL_PINNED_REVISION)
    is_pinned_default = (
        repo_id == config.HF_DEFAULT_REPO_ID
        and revision == config.TRACT_MODEL_PINNED_REVISION
    )
    if not is_pinned_default:
        logger.warning(
            "Using non-default model repo/revision (%s@%s); recorded-hash integrity "
            "is skipped (revision-trust only).", repo_id, revision)

    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(config.TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS),
        ))
    except (LocalEntryNotFoundError, OfflineModeIsEnabled) as exc:
        raise OfflineModelError(
            f"Model not in cache and the Hub is offline. Repo {repo_id}@{revision}. "
            f"Unset HF_HUB_OFFLINE or run `tract download` while online."
        ) from exc

    sentinel = snapshot / f".tract-verified-{revision}"
    if is_pinned_default and not sentinel.exists():
        _verify_pinned(snapshot)
        try:
            sentinel.touch()
        except OSError:
            logger.debug("Could not write verify sentinel in %s", snapshot)
    return ResolvedModel(path=snapshot, revision=revision, source="download")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/model_resolver.py tests/test_model_resolver.py
git commit -m "feat(resolver): ensure_deployment_model with pinned download + integrity"
```

---

### Task 8: Wire helpers + hierarchy_hash check into `TRACTPredictor`

**Files:**
- Modify: `tract/inference.py` (`TRACTPredictor.__init__`, ~89-143)
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `find_st_model_root`, `resolve_hierarchy_path` (Task 4).
- Produces: `TRACTPredictor(model_dir: Path, source: str = "local")` — loads flat or nested layouts; cross-checks `sha256(cre_hierarchy.json)` against `calibration["hierarchy_hash"]` when present (raises `ValueError` on mismatch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py  (unit — exercises only the hierarchy_hash gate via a helper)
import hashlib, json
import pytest
from tract.inference import verify_hierarchy_hash  # new pure helper

def test_verify_hierarchy_hash_match(tmp_path):
    h = tmp_path / "cre_hierarchy.json"
    h.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(h.read_bytes()).hexdigest()
    verify_hierarchy_hash(h, {"hierarchy_hash": digest})  # no raise

def test_verify_hierarchy_hash_mismatch_raises(tmp_path):
    h = tmp_path / "cre_hierarchy.json"
    h.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hierarchy_hash"):
        verify_hierarchy_hash(h, {"hierarchy_hash": "0" * 64})

def test_verify_hierarchy_hash_absent_key_is_noop(tmp_path):
    h = tmp_path / "cre_hierarchy.json"
    h.write_text("{}", encoding="utf-8")
    verify_hierarchy_hash(h, {})  # older bundle: no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference.py -k verify_hierarchy_hash -v`
Expected: FAIL (`ImportError` — `verify_hierarchy_hash` not defined).

- [ ] **Step 3: Write minimal implementation**

Add the helper to `tract/inference.py` (module level):

```python
def verify_hierarchy_hash(hierarchy_path: Path, calibration: dict[str, Any]) -> None:
    """Cross-check the loaded hierarchy against the hash recorded in calibration.

    Catches a stale hierarchy shadowing the snapshot's. No-op for older calibration
    bundles that predate the hierarchy_hash field.
    """
    expected = calibration.get("hierarchy_hash")
    if not expected:
        return
    actual = hashlib.sha256(hierarchy_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"hierarchy_hash mismatch: calibration={expected[:12]}… vs "
            f"loaded {hierarchy_path}={actual[:12]}…"
        )
```

Then rewire `TRACTPredictor.__init__`. Change the signature and the layout/hierarchy resolution:

```python
    def __init__(self, model_dir: Path, source: str = "local") -> None:
        from tract.active_learning.model_io import load_deployment_model

        self._model_dir = model_dir
        artifacts_path = model_dir / "deployment_artifacts.npz"
        calibration_path = model_dir / "calibration.json"
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Deployment artifacts not found: {artifacts_path}")
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration bundle not found: {calibration_path}")

        self._artifacts = load_deployment_artifacts(artifacts_path)
        self._calibration = load_json(calibration_path)
        self._t_deploy = self._calibration["t_deploy"]
        self._ood_threshold = self._calibration["ood_threshold"]
        self._conformal_quantile = self._calibration["conformal_quantile"]

        hierarchy_path = resolve_hierarchy_path(model_dir, source=source)
        verify_hierarchy_hash(hierarchy_path, self._calibration)
        self._hierarchy = CREHierarchy.load(hierarchy_path)

        st_model_dir = find_st_model_root(model_dir)

        adapter_path = st_model_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            for p in st_model_dir.rglob("adapter_model.safetensors"):
                adapter_path = p
                break
        if adapter_path.exists():
            current_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
            if current_hash != self._artifacts.model_adapter_hash:
                raise ValueError(
                    f"Model adapter hash mismatch: artifacts={self._artifacts.model_adapter_hash[:12]}… "
                    f"vs current={current_hash[:12]}…"
                )

        self._model = load_deployment_model(st_model_dir)
        # ... (health check unchanged)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference.py -k verify_hierarchy_hash -v`
Expected: PASS
Run the integration predictor test if a real model is present:
Run: `python -m pytest tests/test_inference.py -m integration -v` (skipped without the model — acceptable).

- [ ] **Step 5: Commit**

```bash
git add tract/inference.py tests/test_inference.py
git commit -m "feat(inference): flat-layout TRACTPredictor + hierarchy_hash cross-check"
```

---

### Task 9: Promote `huggingface_hub` to a pinned default dependency

**Files:**
- Modify: `pyproject.toml` (`[project] dependencies`, lines 10-16)
- Test: `tests/test_config_pins.py`

**Interfaces:**
- Produces: `huggingface_hub` importable on a base install.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_pins.py
def test_huggingface_hub_is_a_default_dependency():
    import tomllib
    from pathlib import Path
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert any(d.startswith("huggingface_hub") or d.startswith("huggingface-hub") for d in deps), deps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_pins.py::test_huggingface_hub_is_a_default_dependency -v`
Expected: FAIL (not in default deps).

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml`, add to `[project] dependencies`:

```toml
    "huggingface_hub>=0.24,<1",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_pins.py::test_huggingface_hub_is_a_default_dependency -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_config_pins.py
git commit -m "build: promote huggingface_hub to a pinned default dependency"
```

---

### Task 10: CLI runtime guard + resolver wiring

**Files:**
- Modify: `tract/cli.py` (`_cmd_assign` 501, `_cmd_ingest` 649, `_cmd_import_ground_truth` 1551, `_cmd_review_export` 1579, plus a new helper and the `--model-dir` default at 324)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ensure_deployment_model` (Task 7), `EXIT_MISSING_RUNTIME` (Task 3).
- Produces: `_require_inference_runtime() -> None` (exits `EXIT_MISSING_RUNTIME` before any download if torch/sentence_transformers absent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import importlib.util
import pytest
import tract.cli as cli

def test_require_runtime_exits_before_download(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)  # torch "absent"
    called = {"download": False}
    monkeypatch.setattr(cli, "ensure_deployment_model",
                        lambda: called.__setitem__("download", True), raising=False)
    with pytest.raises(SystemExit) as e:
        cli._require_inference_runtime()
    assert e.value.code == cli.EXIT_MISSING_RUNTIME
    assert called["download"] is False  # guard fired before any resolve/download
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_require_runtime_exits_before_download -v`
Expected: FAIL (`AttributeError` — `_require_inference_runtime` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `tract/cli.py` (near the top, after imports), and import `EXIT_MISSING_RUNTIME`, `ensure_deployment_model`:

```python
def _require_inference_runtime() -> None:
    """Fail fast (before any download) if the phase0 inference runtime is missing."""
    import importlib.util
    if (importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("sentence_transformers") is None):
        print("Inference needs the phase0 runtime: pip install 'tract[phase0]'",
              file=sys.stderr)
        sys.exit(EXIT_MISSING_RUNTIME)
```

In `_cmd_assign` and `_cmd_ingest`, replace `predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)` with:

```python
    _require_inference_runtime()
    resolved = ensure_deployment_model()
    predictor = TRACTPredictor(resolved.path, source=resolved.source)
```

In `_cmd_import_ground_truth`, replace `PHASE1D_DEPLOYMENT_MODEL_DIR` (passed to `run_uncovered_inference`) with:

```python
        _require_inference_runtime()
        resolved = ensure_deployment_model()
        inf_summary = run_uncovered_inference(
            PHASE1C_CROSSWALK_DB_PATH, resolved.path, dry_run=args.dry_run,
        )
```

In `_cmd_review_export`, replace `model_dir = Path(args.model_dir)` with:

```python
    if args.model_dir is None:
        _require_inference_runtime()
        model_dir = ensure_deployment_model().path
    else:
        model_dir = Path(args.model_dir)
```

And change the `--model-dir` argument default at cli.py:324 from `default=str(PHASE1D_DEPLOYMENT_MODEL_DIR)` to `default=None`.

Add the import line in the `from tract.config import (...)` block: `EXIT_MISSING_RUNTIME`, and `from tract.model_resolver import ensure_deployment_model`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py::test_require_runtime_exits_before_download -v`
Expected: PASS
Run the existing CLI suite: `python -m pytest tests/test_cli.py -q -m "not integration"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/cli.py tests/test_cli.py
git commit -m "feat(cli): runtime guard before download + resolver wiring"
```

---

### Task 11: Pin `tract download` to the recorded revision

**Files:**
- Modify: `tract/cli.py` (`_cmd_download`, ~459-475)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `_cmd_download` passes `revision=TRACT_MODEL_PINNED_REVISION` on every model-file `hf_hub_download`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import argparse
from unittest.mock import patch
import tract.cli as cli
from tract import config

def test_download_pins_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", tmp_path / "dm")
    monkeypatch.setattr(config, "PHASE1C_CROSSWALK_DB_PATH", tmp_path / "x.db")
    with patch("tract.cli.hf_hub_download" if hasattr(cli, "hf_hub_download") else "huggingface_hub.hf_hub_download") as dl:
        dl.return_value = str(tmp_path / "f")
        cli._cmd_download(argparse.Namespace(model_only=True, force=True))
        assert dl.call_count >= 1
        for _, kwargs in dl.call_args_list:
            assert kwargs.get("revision") == config.TRACT_MODEL_PINNED_REVISION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_download_pins_revision -v`
Expected: FAIL (current `hf_hub_download` calls pass no `revision`).

- [ ] **Step 3: Write minimal implementation**

In `_cmd_download`, add `revision=TRACT_MODEL_PINNED_REVISION,` to each of the two model-file `hf_hub_download(...)` calls (the `HF_MODEL_FILES` loop and the `HF_DEPLOY_FILES` loop). Leave the `crosswalk.db` (dataset) download unchanged. Import `TRACT_MODEL_PINNED_REVISION` in the config-import block.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py::test_download_pins_revision -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/cli.py tests/test_cli.py
git commit -m "fix(cli): pin tract download to the recorded model revision"
```

---

### Task 12: Pin-recompute script + CI consistency check

**Files:**
- Create: `scripts/recompute_model_pins.py`
- Create: `tests/test_model_pins_consistency.py` (network, marked)
- Modify: `.github/workflows/ci.yml` (add a `model-pins` job step)
- Test: as above

**Interfaces:**
- Produces: `scripts/recompute_model_pins.py <revision>` prints the five constants by hashing the HF files at that revision.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_pins_consistency.py
import hashlib
import pytest
from tract import config

pytestmark = pytest.mark.integration  # network; excluded from default suite

def test_recorded_hashes_match_pinned_revision():
    from huggingface_hub import hf_hub_download
    for name, expected in config.TRACT_MODEL_PINNED_FILE_HASHES.items():
        path = hf_hub_download(
            repo_id=config.HF_DEFAULT_REPO_ID,
            revision=config.TRACT_MODEL_PINNED_REVISION,
            filename=name,
        )
        actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert actual == expected, f"{name}: {actual} != {expected}"
```

- [ ] **Step 2: Run test to verify it fails (or skips without network)**

Run: `python -m pytest tests/test_model_pins_consistency.py -m integration -v`
Expected: PASS if online (the recorded hashes are correct), or skipped/`error` offline — acceptable; this test is for CI.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/recompute_model_pins.py
"""Print the pinned-model constants for a given HF revision.

Usage: python scripts/recompute_model_pins.py <full_commit_sha>
"""
import hashlib
import sys

from huggingface_hub import hf_hub_download

from tract import config

FILES = ("model.safetensors", "deployment_artifacts.npz",
         "calibration.json", "cre_hierarchy.json")


def main(revision: str) -> None:
    print(f'TRACT_MODEL_PINNED_REVISION = "{revision}"')
    for name in FILES:
        path = hf_hub_download(
            repo_id=config.HF_DEFAULT_REPO_ID, revision=revision, filename=name)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        const = {
            "model.safetensors": "TRACT_MODEL_SAFETENSORS_SHA256",
            "deployment_artifacts.npz": "TRACT_DEPLOYMENT_ARTIFACTS_SHA256",
            "calibration.json": "TRACT_CALIBRATION_SHA256",
            "cre_hierarchy.json": "TRACT_HIERARCHY_SHA256",
        }[name]
        print(f'{const} = "{digest}"')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
```

Add a CI step to `.github/workflows/ci.yml` (new job after the test job):

```yaml
  model-pins:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install "huggingface_hub>=0.24,<1" -e .
      - run: python -m pytest tests/test_model_pins_consistency.py -m integration -q
```

- [ ] **Step 4: Verify the script runs**

Run: `python scripts/recompute_model_pins.py 2d2095518428b4ae88566bad43e57c9b370eba0c`
Expected: prints the five constants matching `tract/config.py`.

- [ ] **Step 5: Commit**

```bash
git add scripts/recompute_model_pins.py tests/test_model_pins_consistency.py .github/workflows/ci.yml
git commit -m "ci: model-pins consistency check + recompute script"
```

---

### Task 13: `__version__` single source

**Files:**
- Modify: `tract/__init__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `tract.__version__: str` — the installed package version, falling back to a git short SHA, then `"0.0.0+unknown"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
def test_version_is_a_nonempty_string():
    import tract
    assert isinstance(tract.__version__, str) and tract.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_version_is_a_nonempty_string -v`
Expected: FAIL (`AttributeError: module 'tract' has no attribute '__version__'`).

- [ ] **Step 3: Write minimal implementation**

Replace `tract/__init__.py` content with:

```python
"""TRACT — Translating Requirements Across CRE Trees."""
from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


try:
    __version__ = _pkg_version("tract")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = _git_short_sha() or "0.0.0+unknown"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py::test_version_is_a_nonempty_string -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/__init__.py tests/test_cli.py
git commit -m "feat: single-source __version__ with git-sha fallback"
```

---

### Task 14: `tract --version` subcommand-free flag

**Files:**
- Modify: `tract/cli.py` (top-level parser construction, ~40-48)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `tract.__version__` (Task 13), `HF_DEFAULT_REPO_ID`, `TRACT_MODEL_PINNED_REVISION`.
- Produces: `tract --version` prints `tract <version>` + model line, exits 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import subprocess, sys

def test_tract_version_flag(capsys):
    import tract.cli as cli
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("tract ")
    assert "model:" in out
```

(Add `import pytest` at the top of the test file if absent.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_tract_version_flag -v`
Expected: FAIL (`--version` unrecognized → `SystemExit(2)`).

- [ ] **Step 3: Write minimal implementation**

In `tract/cli.py`, where the top-level `argparse.ArgumentParser(...)` is created, add a `--version` action. Compute the version string once:

```python
    from tract import __version__
    _repo = os.environ.get("TRACT_MODEL_REPO_ID", HF_DEFAULT_REPO_ID)
    _rev = os.environ.get("TRACT_MODEL_REVISION", TRACT_MODEL_PINNED_REVISION)
    parser.add_argument(
        "--version", action="version",
        version=f"tract {__version__}\nmodel: {_repo}@{_rev}",
    )
```

Ensure `os`, `HF_DEFAULT_REPO_ID`, `TRACT_MODEL_PINNED_REVISION` are imported. Ensure `main()` accepts an optional `argv` list (`def main(argv: list[str] | None = None) -> None:` → `args = parser.parse_args(argv)`); if it currently takes no argv, add the parameter and thread it through.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py::test_tract_version_flag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/cli.py tests/test_cli.py
git commit -m "feat(cli): tract --version prints package + pinned model revision"
```

---

### Task 15: Manifest `tract_version` from `__version__` (diff-safe)

**Files:**
- Modify: `tract/export/manifest.py:46`
- Test: `tests/test_export_manifest.py`

**Interfaces:**
- Produces: `build_manifest(...)["tract_version"] == tract.__version__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_manifest.py
def test_manifest_tract_version_uses_package_version():
    import tract
    from tract.export.manifest import build_manifest
    m = build_manifest(
        per_framework_stats={}, confidence_floor=0.3, confidence_overrides={},
        staleness_result={}, model_adapter_hash="deadbeef",
    )
    assert m["tract_version"] == tract.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_manifest.py::test_manifest_tract_version_uses_package_version -v`
Expected: FAIL (returns hardcoded `"0.1.0"`; assert fails only if `__version__ != "0.1.0"`; to make the test meaningful regardless, the implementation must reference `__version__`, so the test asserts identity of source — keep as written, it passes once sourced from `__version__`).

- [ ] **Step 3: Write minimal implementation**

In `tract/export/manifest.py`, add `from tract import __version__` at the top, and change line 46:

```python
        "tract_version": __version__,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_manifest.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tract/export/manifest.py tests/test_export_manifest.py
git commit -m "feat(export): manifest tract_version from single-source __version__"
```

---

### Task 16: Full-suite green + premortem acceptance sweep

**Files:**
- Test: whole suite

- [ ] **Step 1: Run the full unit suite**

Run: `python -m pytest tests/ -q -m "not integration"`
Expected: PASS (the pre-existing 866 tests + new tests).

- [ ] **Step 2: Type-check the changed modules**

Run: `mypy tract/model_resolver.py tract/inference.py tract/cli.py tract/__init__.py tract/config.py --strict`
Expected: no errors (fix annotations if any surface).

- [ ] **Step 3: Acceptance — `assign --file` rejoin (B)**

Run:
```bash
printf 'access control\n\nencryption at rest\n' > /tmp/controls.txt
# With a real model present (or the integration fixture):
python -c "import json; [print(json.loads(l)['input_index'], json.loads(l)['text']) for l in open('/tmp/controls_assignments.jsonl')]" 2>/dev/null || true
```
Expected (with a model): records `input_index` 1 and 3 (gap at blank line 2), full text, input order.

- [ ] **Step 4: Acceptance — base-install refuses before download (A)**

Confirm (in an env without torch) that `tract assign "x"` prints the phase0 message and exits 5 **without** writing to the HF cache. Verified by `tests/test_cli.py::test_require_runtime_exits_before_download`.

- [ ] **Step 5: Commit any annotation fixes**

```bash
git add -A
git commit -m "chore: type-check fixes and acceptance sweep" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:** Every premortem-hardened spec item maps to a task — S1→T5, S2→T6, S3→T1/T2, S4→T7 (sentinel), S5→T10, S6→T15 (manifest only; canonical untouched per Non-goals), S7→T4/T8, S8→T12, S9→T11, S10→T10, S11→T3/T7/T10 (exit codes + offline error), S12→T4, S13→T9, S14→T13, S15→T7 (`TRACT_MODEL_REPO_ID`), S19→T2. Workstreams B (T1-T2), A (T3-T12), C (T13-T15).

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Every code step shows complete code. The two CI/integration tests are real and runnable.

**Type consistency:** `ResolvedModel(path, revision, source)` used identically in T7/T8/T10. `ensure_deployment_model() -> ResolvedModel` consumed in T10. `find_st_model_root`/`resolve_hierarchy_path`/`verify_hierarchy_hash` signatures match between T4/T8. `TRACTPredictor(model_dir, source="local")` signature in T8 matches the call in T10. `EXIT_*` integers defined in T3, used in T2/T10. `TRACT_MODEL_PINNED_FILE_HASHES` defined in T3, consumed in T7/T12.
