# Design: make `tract assign` work end-to-end after a plain install

Date: 2026-06-16
Status: approved (architecture), pending spec review
Owner: rocklambros

Three workstreams on one branch, committed in this order:

- **B. `tract assign --file` correctness** — preserve input order, stop truncating
  echoed text, add a stable rejoin key. Silent-correctness bug; lands first as its own
  revertable commit.
- **A. Lazy model auto-download** — `tract assign` works on first use with zero manual
  model placement, downloading once to the HuggingFace cache.
- **C. `tract --version` + real manifest version** — kill the hardcoded/`"unknown"`
  version drift; the auto-download resolver supplies the authoritative model revision.

---

## Workstream B — `tract assign --file` correctness

### Problem

The `--file` batch branch silently shuffles and truncates output records. Downstream
consumers that rejoin predictions to inputs by position get wrong answers with no error.

### Verified gaps (read in code)

1. **Reorder.** `cli.py:519` writes records via
   `sorted(zip(texts, results), key=lambda tp: tp[1][0].raw_similarity …)` — output is
   sorted ascending by the top prediction's similarity, not input order. This is the
   only record-level reorder.
2. **`predict_batch` already preserves order.** `inference.py:212-230` builds `results`
   over `for i in range(len(texts))` with `results.append(preds)`, so `results[i] ↔
   texts[i]`. The `np.argsort` at `inference.py:170/214` is *within-record* top-k
   ranking and must stay. So the fix keeps `predict_batch` (batch encoding,
   `batch_size=128`) and removes only the write-time sort — tighter and faster than
   switching to per-item `predict()`.
3. **Truncation.** `cli.py:520` echoes `text[:100]`. Only occurrence in the JSON-record
   path (the single-text branch uses the table/JSON formatters, which write no record).
4. **No rejoin key.** Records are `{"text", "predictions"}` with no positional field.

### Change

Rewrite the `--file` write block in `_cmd_assign`:

```python
raw_lines = file_path.read_text(encoding="utf-8").splitlines()
indexed = [(n, ln.strip()) for n, ln in enumerate(raw_lines, start=1) if ln.strip()]
line_numbers = [n for n, _ in indexed]
texts = [t for _, t in indexed]
results = predictor.predict_batch(texts, top_k=args.top_k)   # input-parallel
with open(output_path, "w", encoding="utf-8") as f:
    for input_index, text, preds in zip(line_numbers, texts, results):
        f.write(json.dumps(
            {"input_index": input_index, "text": text,
             "predictions": [p.to_dict() for p in preds]},
            ensure_ascii=False) + "\n")
```

- **`input_index` = 1-based source-file line number** (gaps where blank lines were
  filtered). Invariant to the blank-filtering logic: a consumer can recompute it from
  the file. Blank lines stay filtered (no inference on empty text).
- Output records are written in input order; full untruncated text is echoed.
- Summary stats (`ood_count`, `high_conf`) are order-independent and unchanged.

### Tests (`tests/test_assign_e2e.py`, predictor mocked — no torch)

- Order: input whose similarity order differs from input order → output line order
  equals input order.
- Index: a blank line mid-file → `input_index` shows the gap (e.g. 1, 2, 4) and each
  record maps to the correct text.
- No truncation: a >100-char line → full text in the record.
- Empty / all-blank file → zero records, clean "Wrote 0 assignments".

---

## Workstream A — lazy model auto-download

### Problem

On a non-editable `pip install 'tract[phase0]'` (the documented install),
`tract assign` fails before doing any work: `TRACTPredictor.__init__` requires a
populated deployment-model directory and never fetches it.

### Verified gaps

1. **`model/` subdir mandatory.** `inference.py:113` unconditionally sets
   `st_model_dir = model_dir / "model"`; a flat snapshot cannot load.
2. **No `model_dir/cre_hierarchy.json` fallback.** `inference.py:108-110` tries only the
   `data/processed` paths, absent on a non-editable install.
3. **Integrity silently skipped.** `inference.py:122` runs the hash check only when an
   `adapter_model.safetensors` exists; the published repo ships merged
   `model.safetensors`, so it never runs. The NPZ's `model_adapter_hash` is `sha256` of
   the **LoRA adapter** (`t5_finalize_crosswalk.py:319`) — a file the published repo
   does not even contain, so that stored hash can never validate published weights.
4. **`PROJECT_ROOT` lands in site-packages.** `config.py:13` derives it from
   `__file__.parent.parent`; non-editable that is `site-packages/` — ephemeral, often
   read-only.
5. **No packaged data.** `pyproject.toml` packages only `tract*`; `cre_hierarchy.json`
   is absent from the wheel.
6. **`huggingface_hub` undeclared.** Imported in three places but only transitively
   present via the phase0 stack. `torch` is correctly phase0-only.

### Stale premise, reconciled

`tract download` (merged 2026-06-16, `39712a9`) already fetches the model and reshapes
the flat repo into the nested `model/` layout, but it is **manual**, writes into the
site-packages tree, and never fetches `cre_hierarchy.json`. This design adds the
**lazy, automatic** path and leaves `tract download` untouched; the resolver honors its
output when complete.

### Decisions (locked)

- **Lazy resolver fetches via `snapshot_download` into the HF cache**, not the per-file
  `local_dir` loop. The HF cache is always user-writable, shared across venvs, survives
  `pip install --upgrade`, and gives revision pinning, `HF_HUB_OFFLINE` handling, and
  content-addressed blob validation for free — less code than a writable-directory
  workaround, and it fixes the site-packages problem instead of dodging it.
- **`tract download` is not refactored.** Resolver honors complete output, supersedes
  incomplete output.
- **Integrity = pinned revision + recorded sha256** of the two large artifacts, enforced
  on the download only (a trusted local dev checkout is not hash-checked).
- **`huggingface_hub` promoted to default deps.** `torch`/`sentence-transformers` stay in
  `phase0`.
- **`predict.py` / `train.py` never fetched or executed.** `allow_patterns` limits the
  snapshot; `trust_remote_code` is never set.

### New module `tract/model_resolver.py`

```python
@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    revision: str          # HF commit SHA, or "local" for a dev/`tract download` dir
    source: str            # "local" | "download"

def ensure_deployment_model() -> ResolvedModel:
    """Resolution order:
      1. Local dir (PHASE1D_DEPLOYMENT_MODEL_DIR) if COMPLETE — dev checkout or a prior
         `tract download`. Trusted; revision="local", source="local".
      2. snapshot_download(repo, revision=resolved_revision, allow_patterns=…) into the
         HF cache. Integrity-checked. source="download".
      3. HF_HUB_OFFLINE + not cached -> actionable ValueError naming repo + env var.
    """
```

- **COMPLETE** = `deployment_artifacts.npz` + `calibration.json` at root, a loadable
  SentenceTransformer root (`model/`, `model/model/`, or flat), **and**
  `cre_hierarchy.json` resolvable. An incomplete local dir (e.g. a non-editable
  `tract download` that omitted the hierarchy) falls through to step 2. Self-healing.
- `resolved_revision = os.environ.get("TRACT_MODEL_REVISION", TRACT_MODEL_PINNED_REVISION)`.
- `huggingface_hub` imported lazily inside the function.
- `.path` is what `TRACTPredictor` consumes; `.revision` feeds manifests (workstream C).

### `TRACTPredictor` changes (`tract/inference.py`)

Factor two pure, unit-testable helpers (no model load, no torch):

- `_find_st_model_root(model_dir) -> Path`: first of `model_dir`, `model_dir/model`,
  `model_dir/model/model` containing `modules.json`. Handles flat (snapshot),
  single-nest (`tract download`), double-nest (dev) in one rule; raises if none.
- `_resolve_hierarchy_path(model_dir) -> Path`: first existing of
  `model_dir.parent.parent/data/processed/cre_hierarchy.json`,
  `model_dir/cre_hierarchy.json`, `PROCESSED_DIR/cre_hierarchy.json`; raises naming all
  three if none.

The existing adapter-hash check stays (meaningful only for a dev checkout shipping a
real adapter). Integrity of the **downloaded** model is the resolver's job, not the
predictor's — a deliberate refinement of the original spec wording: only the resolver
knows the revision and whether it downloaded; the predictor stays a pure loader.

### Integrity (in the resolver)

After step 2 resolves a snapshot, and only on the pinned default revision, verify:

- `sha256(model.safetensors) == TRACT_MODEL_SAFETENSORS_SHA256`
- `sha256(deployment_artifacts.npz) == TRACT_DEPLOYMENT_ARTIFACTS_SHA256`

Mismatch -> `ValueError` naming file + both hashes. A `TRACT_MODEL_REVISION` override
relaxes to revision-trust (HF validated blob sha256 on download) with a WARNING. The
local dir (step 1) is trusted and not hash-checked, so dev artifacts that differ from
the published model do not break.

### New `tract/config.py` constants

```python
TRACT_MODEL_PINNED_REVISION: Final[str] = "2d2095518428b4ae88566bad43e57c9b370eba0c"
TRACT_MODEL_SAFETENSORS_SHA256: Final[str] = (
    "c1f7b6d65c4440ea6b497a47de85898812ebc5efce63f608902de9a4fbe215cd")
TRACT_DEPLOYMENT_ARTIFACTS_SHA256: Final[str] = (
    "7e8b8f834db503118d75727675716471636f139ecb3b64fbd6bc96d6690122f7")
TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS: Final[tuple[str, ...]] = (
    *HF_MODEL_FILES, *HF_DEPLOY_FILES, "cre_hierarchy.json",
)
```

(SHA values resolved live from the HF API against revision `2d20955`. `allow_patterns`
excludes `predict.py`, `train.py`, `README.md`, `bridge_report.json`, redundant `hub_*`.)

### Tests (`tests/test_model_resolver.py`, mock `snapshot_download`, no network)

- local-present → returns local, `snapshot_download` not called.
- local-absent → called once with `revision=PIN` and `allow_patterns` including
  `cre_hierarchy.json`; returns the mocked path; `source="download"`.
- `TRACT_MODEL_REVISION` override → called with that revision; hash enforcement skipped
  with a WARNING.
- offline + cold cache (`LocalEntryNotFoundError`) → actionable `ValueError`.
- integrity tamper (wrong bytes, pinned rev) → `ValueError`.
- incomplete local dir (missing hierarchy) → falls through to download.

`tests/test_inference.py` (unit, no model load): `_find_st_model_root` flat/single/double;
`_resolve_hierarchy_path` ordering. Integration (`@pytest.mark.integration`, real
~1.3 GB model): flat-layout fixture loads through `TRACTPredictor` with no `model/`.

---

## Workstream C — `tract --version` + real manifest version

### Problem

There is no single source of truth for the package version. `tract --version` is
unsupported (errors `unrecognized arguments: --version`). Manifests record the version
inconsistently: `manifest.py:46` hardcodes `"0.1.0"`; `cli.py:1144-1150` shells out to
`git rev-parse` and falls back to `"unknown"` (the common case on a pip install with no
git repo).

### Verified gaps

1. `tract/__init__.py` has no `__version__`.
2. Top-level parser has no `--version`; only `prepare` has an unrelated `--version`
   framework-string arg.
3. `manifest.py:46` hardcodes `"tract_version": "0.1.0"` (drifts from `pyproject`).
4. `cli.py:1144-1150` derives `tract_version` from `git rev-parse --short HEAD` →
   `"unknown"` off a checkout.
5. `manifest.py` already separates `tract_git_sha` (best-effort git) from
   `tract_version` — so git provenance stays best-effort; only `tract_version` must
   become real.

### Changes

- **`tract/__init__.py`:** single source of truth.
  ```python
  from importlib.metadata import PackageNotFoundError, version as _v
  try:
      __version__ = _v("tract")
  except PackageNotFoundError:
      __version__ = "0.0.0+unknown"
  ```
- **`tract --version`:** top-level argparse `action="version"`, computed once from
  config (no download triggered):
  ```
  tract {__version__}
  model: {HF_DEFAULT_REPO_ID}@{resolved_revision}
  ```
  where `resolved_revision = TRACT_MODEL_REVISION env or TRACT_MODEL_PINNED_REVISION`.
- **Manifest `tract_version`:** `manifest.py:46` and `cli.py:1144-1150` both source
  `__version__` (drop the `git rev-parse` shell-out for the version; `tract_git_sha`
  stays best-effort). Kills the hardcode and the `"unknown"`.
- **Manifest `model_version`:** in the predictor-backed manifest paths
  (`review/export.py:310`, `crosswalk/ground_truth.py:279`), record the resolver's
  `.revision` instead of `model_adapter_hash[:12]` (which hashes a LoRA file absent from
  the published model). Threaded from the CLI handler that calls
  `ensure_deployment_model()`. Where no resolver is in play, the existing npz-hash is
  retained.

### Tests

- `tract --version` exits 0 and prints `tract <version>` + the model line (CLI invocation
  test in `tests/test_cli.py`).
- `__version__` resolves to the installed package version, not the literal `"0.1.0"`.
- `build_manifest` records `tract_version == __version__` (update
  `tests/test_export_manifest.py`).

---

## CLI wiring (`tract/cli.py`)

Route the four inference entrypoints through the resolver; interior functions keep their
`model_dir` parameter and trust the caller.

- `_cmd_assign` (504): `TRACTPredictor(ensure_deployment_model().path)`.
- `_cmd_ingest` (649): same.
- `_cmd_import_ground_truth` (1569): pass `resolved.path` (+ `resolved.revision` for
  `model_version`) to `run_uncovered_inference`.
- `_cmd_review_export` (1583): default `--model-dir` to `None`; when `None`, resolve via
  `ensure_deployment_model()`, else honor the explicit override.

Add a friendly dependency guard at the inference entrypoints: catch `ImportError` from
`from tract.inference import TRACTPredictor` and print
`Inference needs the phase0 runtime: pip install 'tract[phase0]'`, exit non-zero. (No
central dep-check exists today; `prepare/llm_extractor.py` is the precedent.) Print an
informational line on first download (`Downloading model (~1.3 GB) on first use; cached
for next time…`).

## `pyproject.toml`

- Add `huggingface_hub>=0.20,<1` to default `dependencies`.
- `torch`/`sentence-transformers` stay in `phase0`. Tradeoff (PR description): default
  install stays small and can fetch artifacts; `tract assign` needs `phase0` for the
  runtime, surfaced by the friendly guard.

## Data flow (assign)

```
tract assign "..."           tract assign --file f
  └─ ensure_deployment_model()        (B) preserve order + full text + input_index
       ├─ local COMPLETE? → it          read lines (keep 1-based line #s)
       └─ else snapshot(pin) → cache    filter blanks, predict_batch (input-parallel)
       └─ offline+cold → ValueError     write records in input order, no [:100]
  └─ TRACTPredictor(resolved.path)
       ├─ _find_st_model_root: flat | model/ | model/model/
       ├─ _resolve_hierarchy_path: data/processed | model_dir | PROCESSED_DIR
       └─ load + health check
```

## Layout matrix

| Source | model root | hierarchy | resolver step |
|---|---|---|---|
| Dev checkout | `…/model/model/` | repo `data/processed` | 1 |
| `tract download` (editable) | `…/model/` | repo `data/processed` | 1 |
| `tract download` (non-editable) | `…/model/` (site-packages) | absent | falls to 2 |
| Lazy snapshot | flat (cache) | alongside (cache) | 2 |

## Error handling

- Offline + cold cache → `ValueError` naming repo id, revision, `HF_HUB_OFFLINE`.
- Missing phase0 runtime → friendly `pip install 'tract[phase0]'` message.
- Integrity mismatch → `ValueError` naming file, expected, actual hash.
- Hierarchy unresolvable → `FileNotFoundError` listing all three candidates.
- Embedding-dim mismatch → existing 1024 check, unchanged.

## Sequencing & packaging

One feature branch, one PR, commits in order:

1. **B** — `assign --file` correctness (independent, revertable, highest priority).
2. **A** — lazy auto-download (resolver, predictor, integrity, config, deps).
3. **C** — `__version__`, `tract --version`, manifest version (consumes A's resolver for
   the model revision).

## Acceptance

- **B:** `tract assign --file controls.txt` → JSONL records in input order, full text,
  `input_index` = source line number; rejoin by `input_index` is exact.
- **A:** fresh venv, `pip install '<path>[phase0]'` then
  `tract assign "test access control"` works with no pre-existing model dir and no manual
  placement — downloads once to the HF cache, prints predictions; re-run uses the cache;
  `HF_HUB_OFFLINE=1` cold cache prints the actionable error.
- **C:** `tract --version` prints `tract <version>` + model line; export manifests carry
  the real `tract_version` and the resolved model revision.

## Non-goals

- No refactor of `tract download` (resolver honors complete output).
- No `--revision` flag on `tract download` (env var covers override).
- No change to zero-residence semantics: this is TRACT's own model, orthogonal to
  source-data residence. All changes stay inside the TRACT repo.

## Risks / maintenance

- **Pinned hashes are revision-bound.** Re-publishing the HF model repo requires bumping
  `TRACT_MODEL_PINNED_REVISION` and recomputing the two sha256 constants. Document the
  bump procedure beside the constants.
- **First-use latency:** ~1.36 GB download, mitigated by the informational message and
  `tract download` as an optional CI/airgapped warm step.
- **Optional follow-up:** add `cre_hierarchy.json` to `tract download`'s file list so the
  manual and lazy paths fully converge (not required; the resolver self-heals).
