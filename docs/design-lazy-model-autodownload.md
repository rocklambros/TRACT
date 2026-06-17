# Design: lazy model auto-download for `tract assign`

Date: 2026-06-16
Status: approved (architecture), pending spec review
Owner: rocklambros

## Problem

On a non-editable `pip install 'tract[phase0]'` (the documented install command),
`tract assign` fails before doing any work. `TRACTPredictor.__init__` requires a
populated deployment-model directory and never fetches it. The user has to hand-place
files, or run a manual command, before the first prediction works. Goal: `tract assign`
works on first use with zero manual placement, downloading the model once.

## Verified gap (against current `main`, post-merge of `tract download`)

Each item below was read in the code, not assumed.

1. **`model/` subdir is mandatory.** `inference.py:113` unconditionally sets
   `st_model_dir = model_dir / "model"`. A raw flat snapshot (no `model/`) cannot load.
2. **No `model_dir/cre_hierarchy.json` fallback.** `inference.py:108-110` tries
   `model_dir.parent.parent/data/processed/cre_hierarchy.json` then `PROCESSED_DIR`.
   Neither exists on a non-editable install.
3. **Integrity is silently skipped for the published model.** `inference.py:122` only
   runs the hash check when an `adapter_model.safetensors` exists. The published repo
   ships a merged `model.safetensors`, so the check never runs. Worse, the NPZ's stored
   `model_adapter_hash` is `sha256` of the **LoRA adapter** (`t5_finalize_crosswalk.py:319`),
   a file the published repo does not even contain — so that stored hash can never
   validate the published weights.
4. **`PROJECT_ROOT` lands in site-packages.** `config.py:13` derives it from
   `__file__.parent.parent`; on a non-editable install that is `site-packages/`, an
   ephemeral and sometimes read-only location. `PHASE1D_DEPLOYMENT_MODEL_DIR` inherits this.
5. **Packaging ships no data.** `pyproject.toml` packages only `tract*`; `data/` (and
   thus `cre_hierarchy.json`) is absent from the wheel.
6. **`huggingface_hub` is undeclared.** It is imported in three places
   (`cli.py:441`, `dataset/publish.py:36`, `publish/__init__.py:201`) but only present
   transitively via the phase0 stack. `torch` is correctly phase0-only.

## Stale premise, now reconciled

`tract download` (merged 2026-06-16, commit `39712a9`) already fetches the model and
reshapes the flat HF repo into the nested `model/` layout. Its own commit message claims
to close "the fresh-clone gap." It does not: it is a **manual** step, it writes into the
site-packages tree, and it never fetches `cre_hierarchy.json`. This design adds the
**lazy, automatic** path and leaves `tract download` untouched. The resolver honors
`tract download`'s output when it is complete (see resolution order).

## Decisions (locked)

- **Lazy resolver fetches via `snapshot_download` into the HuggingFace cache**, not the
  per-file `local_dir` loop. The HF cache is always user-writable, shared across venvs,
  survives `pip install --upgrade`, and gives revision pinning, `HF_HUB_OFFLINE`
  handling, and content-addressed blob validation for free. This is less code than a
  writable-directory-resolution workaround, and it fixes the site-packages problem
  instead of dodging it.
- **`tract download` is not refactored.** It stays per-file → results tree. The resolver
  honors its output when complete.
- **Integrity = pinned revision + recorded sha256 of the two large artifacts**, enforced
  on the downloaded snapshot (not on a trusted local dev checkout).
- **`huggingface_hub` promoted to default deps.** `torch`/`sentence-transformers` stay
  in the `phase0` extra (multi-GB; a base install can fetch artifacts but not run
  inference).
- **`predict.py` / `train.py` are never fetched or executed.** `allow_patterns` limits
  the snapshot to the files we need. `trust_remote_code` is never set.

## Components

### New: `tract/model_resolver.py`

```python
def ensure_deployment_model() -> Path:
    """Return a directory that satisfies TRACTPredictor's contract.

    Resolution order:
      1. Local dir (PHASE1D_DEPLOYMENT_MODEL_DIR) if COMPLETE — honors dev
         checkout and a prior `tract download`. Trusted; no hash enforcement.
      2. snapshot_download(repo, revision=resolved_revision, allow_patterns=...)
         into the HF cache. Integrity-checked. Returns the flat snapshot path.
      3. HF_HUB_OFFLINE + not cached -> actionable ValueError.
    """
```

"COMPLETE" for step 1 means: `deployment_artifacts.npz` and `calibration.json` at root,
a loadable SentenceTransformer root (`model/`, `model/model/`, or flat), **and**
`cre_hierarchy.json` resolvable (repo `data/processed` or alongside the dir). An
incomplete local dir (e.g. a non-editable `tract download` that omitted the hierarchy)
falls through to step 2, which fetches a complete snapshot. Self-healing.

`resolved_revision = os.environ.get("TRACT_MODEL_REVISION", TRACT_MODEL_PINNED_REVISION)`.

`huggingface_hub` is imported lazily inside the function to keep CLI startup fast.

### Changed: `tract/inference.py` (`TRACTPredictor`)

Two layout helpers, factored as pure functions so they are unit-testable without loading
a model or importing torch:

- `_find_st_model_root(model_dir) -> Path`: first of `model_dir`, `model_dir/model`,
  `model_dir/model/model` that contains `modules.json`. Handles flat (snapshot),
  single-nest (`tract download`), and double-nest (dev) in one rule. Raises if none.
- `_resolve_hierarchy_path(model_dir) -> Path`: first existing of
  `model_dir.parent.parent/data/processed/cre_hierarchy.json`,
  `model_dir/cre_hierarchy.json`, `PROCESSED_DIR/cre_hierarchy.json`. Raises a clear
  error naming all three if none.

The existing adapter-hash check stays (it is meaningful only for a dev checkout that
ships an actual adapter). Integrity of the **downloaded** model is the resolver's job,
not the predictor's — this is a deliberate refinement of the original spec wording,
which put a positive check inside `TRACTPredictor`. Rationale: only the resolver knows
the revision and whether it just downloaded; the predictor should stay a pure loader.

### Changed: integrity in the resolver

After step 2 resolves a snapshot path, and only when running the pinned default
revision, verify:

- `sha256(model.safetensors) == TRACT_MODEL_SAFETENSORS_SHA256`
- `sha256(deployment_artifacts.npz) == TRACT_DEPLOYMENT_ARTIFACTS_SHA256`

Mismatch raises `ValueError` naming the file and both hashes. When `TRACT_MODEL_REVISION`
overrides the default, the recorded hashes do not apply: relax to revision-trust (HF
already validated blob sha256 on download) and log a WARNING that integrity is
revision-only. The local-dir path (step 1) is trusted and not hash-checked, so dev
artifacts that differ from the published model do not break.

### Changed: `tract/config.py` (new constants)

```python
# Pinned for reproducibility; bump when the HF model repo is re-published.
TRACT_MODEL_PINNED_REVISION: Final[str] = "2d2095518428b4ae88566bad43e57c9b370eba0c"
TRACT_MODEL_SAFETENSORS_SHA256: Final[str] = (
    "c1f7b6d65c4440ea6b497a47de85898812ebc5efce63f608902de9a4fbe215cd")
TRACT_DEPLOYMENT_ARTIFACTS_SHA256: Final[str] = (
    "7e8b8f834db503118d75727675716471636f139ecb3b64fbd6bc96d6690122f7")
TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS: Final[tuple[str, ...]] = (
    *HF_MODEL_FILES, *HF_DEPLOY_FILES, "cre_hierarchy.json",
)
```

(SHA values resolved live from the HF API at design time against revision
`2d20955`. `allow_patterns` excludes `predict.py`, `train.py`, `README.md`,
`bridge_report.json`, and the redundant `hub_*` files.)

### Changed: CLI wiring (`tract/cli.py`)

Route the four inference entrypoints through the resolver; interior functions keep their
`model_dir` parameter unchanged.

- `_cmd_assign` (504): `TRACTPredictor(ensure_deployment_model())`.
- `_cmd_ingest` (649): same.
- `_cmd_import_ground_truth` (1569): pass `ensure_deployment_model()` to
  `run_uncovered_inference`.
- `_cmd_review_export` (1583): default `--model-dir` to `None`; when `None`, use
  `ensure_deployment_model()`, else honor the explicit override.

Add a friendly dependency guard at the inference entrypoints: catch `ImportError` from
`from tract.inference import TRACTPredictor` and print
`Inference needs the phase0 runtime: pip install 'tract[phase0]'` then exit non-zero.
(There is no central dep-check today; `prepare/llm_extractor.py` is the precedent.)

Print an informational line on first download (`Downloading model (~1.3 GB) on first
use; cached for next time…`).

### Changed: `pyproject.toml`

- Add `huggingface_hub>=0.20,<1` to default `dependencies`.
- `torch`/`sentence-transformers` stay in `phase0`. Tradeoff (state in PR description):
  default install stays small and can fetch artifacts; `tract assign` needs `phase0` for
  the runtime, surfaced by the friendly dependency guard.

## Data flow

```
tract assign "..."
  └─ ensure_deployment_model()
       ├─ local dir COMPLETE? ──► return it (dev / prior `tract download`)
       └─ else snapshot_download(repo, rev=PIN, allow_patterns) ─► HF cache
              └─ integrity (sha256 model.safetensors + npz) on pinned rev
       └─ HF_HUB_OFFLINE + cold cache ─► actionable ValueError
  └─ TRACTPredictor(model_dir)
       ├─ _find_st_model_root: flat | model/ | model/model/
       ├─ _resolve_hierarchy_path: data/processed | model_dir | PROCESSED_DIR
       ├─ load npz + calibration (root)
       ├─ adapter-hash check (dev only; no-op for published model)
       └─ SentenceTransformer load + health check
```

## Layout matrix

| Source | model root | hierarchy | resolver step |
|---|---|---|---|
| Dev checkout | `…/model/model/` | repo `data/processed` | 1 |
| `tract download` (editable) | `…/model/` | repo `data/processed` | 1 |
| `tract download` (non-editable) | `…/model/` (site-packages) | absent | falls to 2 |
| Lazy snapshot | flat (cache) | alongside (cache) | 2 |

## Error handling

- **Offline + cold cache:** `ValueError` naming repo id, revision, and the
  `HF_HUB_OFFLINE` env var, suggesting `tract download` while online.
- **Missing phase0 runtime:** friendly message with the exact `pip install` line.
- **Integrity mismatch:** `ValueError` naming file, expected, and actual hash.
- **Hierarchy unresolvable:** `FileNotFoundError` listing all three candidate paths.
- **Embedding-dim mismatch:** existing `load_deployment_model` check (1024) unchanged.

## Testing

`tests/test_model_resolver.py` (unit, mock `snapshot_download`, no network):

- local-present → returns local, `snapshot_download` not called.
- local-absent → `snapshot_download` called once with `revision=PIN` and
  `allow_patterns` including `cre_hierarchy.json`; returns the mocked path.
- `TRACT_MODEL_REVISION` override → called with that revision; hash enforcement skipped
  with a WARNING.
- offline + cold cache (simulate `LocalEntryNotFoundError`) → actionable `ValueError`.
- integrity tamper (wrong bytes in resolved snapshot, pinned rev) → `ValueError`.
- incomplete local dir (missing hierarchy) → falls through to download.

`tests/test_inference.py` additions (unit, no model load):

- `_find_st_model_root` for flat / single-nest / double-nest fixtures.
- `_resolve_hierarchy_path` fallback ordering.

Integration (`@pytest.mark.integration`, real ~1.3 GB model, excluded from default CI):

- flat-layout fixture loads through `TRACTPredictor` with no `model/` subdir.

## Acceptance

In a fresh venv:

```
pip install '<path-to-TRACT>[phase0]'
tract assign "test access control"
```

works with no pre-existing `deployment_model` dir and no manual placement: downloads once
to the HF cache, prints predictions. Re-running uses the cache (no re-download).
`HF_HUB_OFFLINE=1` with a cold cache prints the actionable error.

## Non-goals

- No refactor of `tract download` (stays as-is; resolver honors complete output).
- No `--revision` flag on `tract download` (the env var covers override).
- No change to zero-residence semantics: this is TRACT's own model, orthogonal to
  source-data residence. All changes stay inside the TRACT repo.

## Risks / maintenance

- **Pinned hashes are revision-bound.** Re-publishing the HF model repo (as happened
  2026-06-16) requires bumping `TRACT_MODEL_PINNED_REVISION` and recomputing the two
  sha256 constants. Document the bump procedure beside the constants.
- **First-use latency:** ~1.36 GB download. Mitigated by the informational message and
  `tract download` as an optional warm-the-cache step for CI/airgapped deploys.
- **Optional follow-up:** add `cre_hierarchy.json` to `tract download`'s file list so the
  manual and lazy paths fully converge (not required; the resolver self-heals).
