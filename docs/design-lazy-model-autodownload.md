# Design: make `tract assign` work end-to-end after a plain install

Date: 2026-06-16
Status: approved (architecture), hardened by adversarial premortem, pending spec review
Owner: rocklambros

Three workstreams on one branch, committed in this order:

- **B. `tract assign --file` correctness** — preserve input order, stop truncating
  echoed text, add a stable rejoin key. Silent-correctness bug; lands first.
- **A. Lazy model auto-download** — `tract assign` works on first use with zero manual
  model placement, downloading once to the HuggingFace cache.
- **C. `tract --version`** — a real, single-source package version (human-facing only).

> **Premortem note (2026-06-16).** A six-perspective adversarial premortem hardened this
> spec. The biggest correction: workstream C was **de-scoped** to the human-facing
> `--version` only, because changing the canonical-export provenance fields
> (`tract_version`, `model_version`) would phantom-diff the published OpenCRE export. All
> premortem remediations are folded into the sections below; a consolidated list is in
> "Premortem hardening."

---

## Workstream B — `tract assign --file` correctness

### Problem

The `--file` batch branch silently shuffles and truncates output records. Downstream
consumers that rejoin predictions to inputs by position get wrong answers with no error.

### Verified gaps

1. **Reorder.** `cli.py:519` writes via `sorted(zip(texts, results), key=…raw_similarity)`
   — sorted by similarity, not input order. The only record-level reorder.
2. **`predict_batch` already preserves order** (`inference.py:212-230`), so the fix keeps
   batch encoding and removes only the sort. The `np.argsort` (170/214) is within-record
   top-k ranking — keep it.
3. **Truncation.** `cli.py:520` echoes `text[:100]`. Only occurrence in the record path.
4. **No rejoin key.** Records are `{"text", "predictions"}`.

### Change

```python
file_path = Path(args.file)
if not file_path.exists():
    print(f"Error: File not found: {file_path}", file=sys.stderr); sys.exit(EXIT_USER_ERROR)
if file_path.stat().st_size > PHASE1D_INGEST_MAX_FILE_SIZE:        # S19: mirror ingest cap
    print(f"Error: --file exceeds {PHASE1D_INGEST_MAX_FILE_SIZE} bytes", file=sys.stderr); sys.exit(EXIT_USER_ERROR)
raw_lines = file_path.read_text(encoding="utf-8").splitlines()
indexed = [(n, ln.strip()) for n, ln in enumerate(raw_lines, start=1) if ln.strip()]
if not indexed:                                                    # S3: empty/all-blank guard
    print("Wrote 0 assignments (no non-blank input lines)."); return
line_numbers = [n for n, _ in indexed]
texts = [t for _, t in indexed]
results = predictor.predict_batch(texts, top_k=args.top_k)        # input-parallel
with open(output_path, "w", encoding="utf-8") as f:
    for input_index, text, preds in zip(line_numbers, texts, results):
        f.write(json.dumps(
            {"input_index": input_index, "text": text,
             "predictions": [p.to_dict() for p in preds]},
            ensure_ascii=False) + "\n")
```

- **`input_index` = 1-based source-file line number** (gaps where blank lines were
  filtered). Invariant to the blank filter; maps back to the file on independent re-read.
- Input order; full untruncated text; size cap reused from `PHASE1D_INGEST_MAX_FILE_SIZE`
  (which `_cmd_ingest` already enforces but `_cmd_assign` does not).
- `predict_batch` also early-returns `[]` for empty input (defensive; S3).

### Tests (`tests/test_assign_e2e.py`, predictor mocked — no torch)

- Order: input whose similarity order ≠ input order → output is in input order.
- Index: blank line mid-file → `input_index` shows the gap (1, 2, 4).
- No truncation: >100-char line → full text.
- Empty / all-blank file → zero records, clean message, no crash.
- Oversized file → rejected with the size-cap error.

---

## Workstream A — lazy model auto-download

### Problem

On a non-editable `pip install 'tract[phase0]'`, `tract assign` fails before doing any
work: `TRACTPredictor.__init__` requires a populated model dir and never fetches it.

### Verified gaps

1. `model/` subdir mandatory (`inference.py:113`); a flat snapshot cannot load.
2. No `model_dir/cre_hierarchy.json` fallback (`inference.py:108-110`).
3. Integrity silently skipped (`inference.py:122`); the npz's `model_adapter_hash` hashes
   a LoRA absent from the published repo (`t5_finalize_crosswalk.py:319`).
4. `PROJECT_ROOT` lands in site-packages (`config.py:13`).
5. `cre_hierarchy.json` absent from the wheel (`pyproject.toml` packages only `tract*`).
6. `huggingface_hub` undeclared (imported transitively).

### Stale premise, reconciled

`tract download` (merged `39712a9`) already fetches+reshapes but is manual, writes
site-packages, omits `cre_hierarchy.json`, **and is unpinned** (see Premortem S9). The
resolver adds the lazy path and honors `tract download`'s output when complete.

### Decisions (locked)

- **`snapshot_download` into the HF cache** (revision-pinned, `HF_HUB_OFFLINE`-aware,
  content-addressed blob validation, user-writable, shared across venvs).
- **`tract download` is not refactored**, but **is pinned** to the same revision (S9).
- **Integrity = pin + recorded sha256, enforced ONCE on download** (S4), covering the
  weights, npz, calibration, and hierarchy.
- **`huggingface_hub` is a default dep, version-pinned tightly** (S13).
- **`predict.py`/`train.py` never fetched; `trust_remote_code=False` explicit** (S2).

### New module `tract/model_resolver.py`

```python
@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    revision: str          # HF commit SHA, or "local" for a dev/`tract download` dir
    source: str            # "local" | "download"

def ensure_deployment_model() -> ResolvedModel:
    """1. Local dir if COMPLETE -> trusted (revision="local").
       2. else snapshot_download(repo, revision=resolved_revision, allow_patterns=…)
          -> HF cache; integrity verified ONCE (sentinel-gated); source="download".
       3. HF_HUB_OFFLINE + not cached -> actionable ValueError (exit code EXIT_OFFLINE).
    """
```

- `repo = os.environ.get("TRACT_MODEL_REPO_ID", HF_DEFAULT_REPO_ID)` — mirror override
  for account continuity (S15G); a non-default repo logs a WARNING.
- `resolved_revision = os.environ.get("TRACT_MODEL_REVISION", TRACT_MODEL_PINNED_REVISION)`.
- **COMPLETE** = npz + calibration at root, an ST root resolvable, **and**
  `cre_hierarchy.json` resolvable; else fall through to download (self-healing).
- `huggingface_hub` imported lazily, wrapped in an actionable `ImportError` message (S13).

### Integrity (resolver, download path only — S4)

`snapshot_download` returns the cached path on every call, so hashing there would re-hash
1.34 GB per invocation. Instead:

1. After a download (cache miss), verify, on the **pinned default revision** only:
   `sha256(model.safetensors) == TRACT_MODEL_SAFETENSORS_SHA256`,
   `sha256(deployment_artifacts.npz) == TRACT_DEPLOYMENT_ARTIFACTS_SHA256`,
   `sha256(calibration.json) == TRACT_CALIBRATION_SHA256`,
   `sha256(cre_hierarchy.json) == TRACT_HIERARCHY_SHA256`.
2. On success, write a sentinel `<<snapshot>>/.tract-verified-<revision>`. On subsequent
   calls, if the sentinel exists, **skip** hashing (warm path is free).
3. Mismatch → `ValueError` (exit `EXIT_INTEGRITY`) naming the file, both hashes, and the
   remediation (`clear the HF cache for this repo and re-run`).
4. A `TRACT_MODEL_REVISION` or `TRACT_MODEL_REPO_ID` override skips the recorded-hash gate
   (the constants are revision/repo-specific) and logs a WARNING that integrity is
   revision-trust only (HF blob validation). The local dir (step 1) is trusted.

For all four files at the pinned revision, the pin + HF blob validation is the primary
provenance; the recorded hashes are the cache-tamper backstop.

### Cross-version skew check (S7) — in `TRACTPredictor`

The calibration bundle already records `hierarchy_hash = sha256(cre_hierarchy.json)`
(`t5_finalize_crosswalk.py:347/354`) and nothing reads it. Add: after loading calibration
and the hierarchy, assert `sha256(loaded cre_hierarchy.json) == calibration["hierarchy_hash"]`
(when the key is present; warn-and-continue if an older bundle lacks it). Catches a stale
`data/processed/cre_hierarchy.json` shadowing the snapshot's.

### `TRACTPredictor` changes (`tract/inference.py`)

Pure, unit-testable helpers (no model load, no torch):

- `_find_st_model_root(model_dir)`: probe **most-nested first**
  (`model_dir/model/model` → `model_dir/model` → `model_dir`); accept the first dir
  containing **both** `modules.json` and `config_sentence_transformers.json`; raise if
  none, error if ambiguous (S12).
- `_resolve_hierarchy_path(model_dir, source)`: for `source=="download"`, try the
  snapshot root (`model_dir/cre_hierarchy.json`) **first** (revision-pinned, S7); then
  `model_dir.parent.parent/data/processed`; then `PROCESSED_DIR`. For `source=="local"`,
  keep the dev order (`data/processed` first). Raise naming all candidates if none.

Security: `load_deployment_artifacts` uses **`np.load(allow_pickle=False)`** (S1; the npz
holds only float arrays + string arrays — no pickle needed; matches `bridge`/`bundle`).
Apply the same to `canonical.py:389`. `load_deployment_model` passes
**`trust_remote_code=False`** explicitly and asserts `config.json` has no `auto_map` /
custom architecture before load (S2). The existing adapter-hash check stays (dev only).

### New `tract/config.py` constants

```python
TRACT_MODEL_PINNED_REVISION: Final[str] = "2d2095518428b4ae88566bad43e57c9b370eba0c"
TRACT_MODEL_SAFETENSORS_SHA256: Final[str] = (
    "c1f7b6d65c4440ea6b497a47de85898812ebc5efce63f608902de9a4fbe215cd")
TRACT_DEPLOYMENT_ARTIFACTS_SHA256: Final[str] = (
    "7e8b8f834db503118d75727675716471636f139ecb3b64fbd6bc96d6690122f7")
TRACT_CALIBRATION_SHA256: Final[str]  = "a49c532d7f8e4d42ff1e5208f68aabdd60d87feb5231c77fbc276c757edda88a"
TRACT_HIERARCHY_SHA256: Final[str]    = "8dc48bd397cf6ee455193a9768760258f235fb5519659915dccd733dcaa19738"
TRACT_MODEL_SNAPSHOT_ALLOW_PATTERNS: Final[tuple[str, ...]] = (
    *HF_MODEL_FILES, *HF_DEPLOY_FILES, "cre_hierarchy.json",
)
# Exit codes for scriptable failure classes (S11)
EXIT_USER_ERROR, EXIT_OFFLINE, EXIT_INTEGRITY, EXIT_MISSING_RUNTIME = 2, 3, 4, 5
```

`allow_patterns` excludes `predict.py`, `train.py`, `README.md`, `bridge_report.json`,
redundant `hub_*`. **A `# bump procedure:` comment block sits beside the constants** with
the exact recompute steps (S8).

### Tests (`tests/test_model_resolver.py`, mock `snapshot_download`, no network)

- local-present → returns local; `snapshot_download` not called.
- local-absent → called once with `revision=PIN`, `allow_patterns` incl. `cre_hierarchy.json`;
  `source="download"`; integrity runs; sentinel written.
- warm cache (sentinel present) → integrity NOT re-run (S4).
- `TRACT_MODEL_REVISION` / `TRACT_MODEL_REPO_ID` override → hash gate skipped + WARNING.
- offline + cold cache (`LocalEntryNotFoundError`) → `ValueError`, exit `EXIT_OFFLINE`.
- integrity tamper (wrong bytes, pinned rev, no sentinel) → `ValueError`, exit `EXIT_INTEGRITY`.
- incomplete local dir (missing hierarchy) → falls through to download.

`tests/test_inference.py` (no model load): `_find_st_model_root` flat/single/double +
ambiguity; `_resolve_hierarchy_path` ordering per `source`; `hierarchy_hash` mismatch
raises; `np.load` rejects a pickled-object npz. Integration (`@pytest.mark.integration`,
real ~1.3 GB, **excluded from default CI**): flat-layout fixture loads end-to-end.

---

## Workstream C — `tract --version` (human-facing only)

### Scope (de-scoped by premortem S6)

`tract --version` is unsupported (`unrecognized arguments: --version`). Add it from a
single-source `__version__`. **Do not** change any export-manifest provenance field that
feeds a `content_hash` or a mapping diff — see Non-goals.

### Changes

- **`tract/__init__.py`:**
  ```python
  from importlib.metadata import PackageNotFoundError, version as _v
  try:
      __version__ = _v("tract")
  except PackageNotFoundError:
      __version__ = _git_short_sha_or("0.0.0+unknown")   # S14: prefer a git SHA off a checkout
  ```
- **`tract --version`:** top-level argparse `action="version"`, computed once from config
  (no download):
  ```
  tract {__version__}
  model: {repo}@{resolved_revision}
  ```
- **`manifest.py:46`:** replace the hardcoded `"0.1.0"` with `__version__`. This is
  value-identical today (`version("tract") == "0.1.0"`) and the Phase 5 OpenCRE export
  manifest is **not** part of a `content_hash`, so it is diff-safe. `tract_git_sha`
  (manifest.py:51) stays the per-build discriminator.

### Explicitly NOT changed (S6)

- `cli.py:1144-1150` canonical-export `tract_version` stays the git-SHA derivation —
  it is hashed into `StandardSnapshot.content_hash` (`canonical_schema.py:90` excludes
  only `content_hash`+`export_date`).
- `review/export.py:310` / `crosswalk/ground_truth.py:279` `model_version` stays
  `model_adapter_hash[:12]` — it is a `_MAPPING_MUTABLE_FIELDS` member
  (`canonical.py:165`); changing it diffs all 5,238 mappings.
- `ResolvedModel.revision` is used for `--version` display and logging only, never written
  into an export provenance field in this change.

### Tests

- `tract --version` exits 0, prints `tract <version>` + model line.
- `build_manifest` records `tract_version == __version__`.
- A `export-canonical` run with no underlying change produces an **empty** diff
  (guards against accidental provenance-field churn).

---

## CLI wiring (`tract/cli.py`)

Route the four inference entrypoints through the resolver; interior functions keep their
`model_dir` param. **Before** the resolver runs, check the runtime is present (S5):

```python
import importlib.util
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("sentence_transformers") is None:
    print("Inference needs the phase0 runtime: pip install 'tract[phase0]'", file=sys.stderr)
    sys.exit(EXIT_MISSING_RUNTIME)                     # fires BEFORE any 1.36 GB download
resolved = ensure_deployment_model()
predictor = TRACTPredictor(resolved.path)
```

- `_cmd_assign` (504) and `_cmd_ingest` (649): as above.
- `_cmd_import_ground_truth` (1569): pass `resolved.path` to `run_uncovered_inference`.
- `_cmd_review_export` (1583): `model_dir = ensure_deployment_model().path if args.model_dir
  is None else Path(args.model_dir)` (S10; default `--model-dir` becomes `None`).
- `tract download` (`_cmd_download`, 461-491): add `revision=TRACT_MODEL_PINNED_REVISION`
  to the model-file `hf_hub_download` calls (S9; light touch, no refactor).

Print `Downloading model (~1.3 GB) on first use; cached for next time…` on a cache miss.

## `pyproject.toml`

- Add `huggingface_hub>=0.24,<1` to default `dependencies` (S13: tighter than the original
  `>=0.20`; 0.24+ guarantees the cache filelock + `allow_patterns` + offline behavior the
  resolver relies on). `torch`/`sentence-transformers` stay in `phase0`. Tradeoff in PR
  description: base install can fetch artifacts; inference needs `phase0`, surfaced by the
  pre-resolver runtime check.

## Error handling & operability (S11)

- Distinct exit codes per failure class: `EXIT_USER_ERROR` 2, `EXIT_OFFLINE` 3,
  `EXIT_INTEGRITY` 4, `EXIT_MISSING_RUNTIME` 5.
- Offline error names repo id, revision, **and** the cache-location env precedence
  (`HF_HUB_CACHE` > `HF_HOME/hub` > `XDG_CACHE_HOME` > `~/.cache/huggingface/hub`).
- Integrity/corruption error includes the purge-and-refetch remediation; a corrupted cache
  blob is not a permanent dead-end.
- Concurrency: rely on `huggingface_hub`'s cache filelock for the download; run integrity
  after `snapshot_download` returns (post atomic-rename), so no partial-read race.

## CI / maintenance (S8)

- New CI job: recompute `sha256` of the four pinned files from `TRACT_MODEL_PINNED_REVISION`
  via the HF API and assert equality with the four constants — fails the build on any
  pin/hash drift (catches the typo-causes-global-outage failure at merge).
- A `scripts/recompute_model_pins.py` helper prints the constants for a given revision.
- Bump runbook + named owner committed beside the constants.
- The integration test is excluded from the default suite (`-m "not integration"`), so the
  60 s CI timeout never meets a 1.36 GB fetch.

## Data flow (assign)

```
tract assign "..."                         tract assign --file f
  └─ runtime check (torch/ST present?) ──► no -> EXIT_MISSING_RUNTIME (before download)
  └─ ensure_deployment_model()                (B) size-cap; keep 1-based line #s;
       ├─ local COMPLETE? -> it                   filter blanks; empty -> 0 records;
       └─ else snapshot(pin) -> cache             predict_batch (input-parallel);
            └─ download? verify once + sentinel    write in input order, full text,
       └─ offline+cold -> EXIT_OFFLINE            input_index = source line #
  └─ TRACTPredictor(resolved.path)
       ├─ _find_st_model_root (nested-first, modules.json + config_sentence_transformers.json)
       ├─ _resolve_hierarchy_path (snapshot-root first when source==download)
       ├─ np.load(allow_pickle=False); hierarchy_hash cross-check
       └─ SentenceTransformer(trust_remote_code=False) + health check
```

## Layout matrix

| Source | model root | hierarchy | resolver step |
|---|---|---|---|
| Dev checkout | `…/model/model/` | repo `data/processed` | 1 |
| `tract download` (editable) | `…/model/` | repo `data/processed` | 1 |
| `tract download` (non-editable) | `…/model/` (site-packages) | absent | falls to 2 |
| Lazy snapshot | flat (cache) | snapshot root | 2 |

## Sequencing & packaging

One branch, one PR, commits in order:

1. **B** — `assign --file` correctness (independent, highest priority).
2. **A** — lazy auto-download + all Round 1-4 security/ops hardening.
3. **C** — `__version__` + `tract --version` (human-facing only).

## Acceptance

- **B:** records in input order, full text, `input_index` = source line; empty/oversized
  files handled; rejoin by `input_index` exact.
- **A:** fresh venv, `pip install '<path>[phase0]'` → `tract assign "test access control"`
  works with no pre-existing model dir, no manual placement; downloads once; warm re-run
  does no 1.3 GB re-hash; base install (no phase0) refuses **before** downloading;
  `HF_HUB_OFFLINE=1` cold cache exits `EXIT_OFFLINE`; a tampered cached blob exits
  `EXIT_INTEGRITY` with a purge hint.
- **C:** `tract --version` prints `tract <version>` + model line; `export-canonical`
  produces an empty diff when nothing changed.

## Non-goals

- No refactor of `tract download` (only a `revision=` pin added).
- No `--revision` flag on `tract download` (env var covers override).
- **No change to canonical-export provenance fields** (`tract_version` in `content_hash`,
  `model_version` in mapping diffs) — would phantom-diff the published OpenCRE export.
- No change to zero-residence semantics (TRACT's own model; orthogonal to source-data
  residence).
- No sigstore/HF-commit-signing in this change (see Residual risk).

## Premortem hardening (consolidated)

| ID | Fix |
|---|---|
| S1 | `np.load(allow_pickle=False)` at `inference.py:75` + `canonical.py:389` |
| S2 | explicit `trust_remote_code=False` + assert no `auto_map` in `config.json` |
| S3 | empty/all-blank `--file` guard + `predict_batch([])` early-return |
| S4 | integrity verified once on download, sentinel-gated (no per-call 1.3 GB hash) |
| S5 | runtime (`torch`/ST) `find_spec` check **before** the resolver/download |
| S6 | de-scope C: do not change canonical `tract_version`/`model_version` |
| S7 | snapshot-root hierarchy first + `hierarchy_hash` cross-check + hash calib/hierarchy |
| S8 | CI hash-consistency check + recompute script + bump runbook/owner |
| S9 | pin `tract download` to `TRACT_MODEL_PINNED_REVISION` |
| S10 | `review-export --model-dir` None-handling |
| S11 | distinct exit codes; cache-env + purge remediation in errors |
| S12 | `_find_st_model_root` most-nested-first + require two ST marker files |
| S13 | `huggingface_hub` default dep, pinned `>=0.24,<1`, import-guarded |
| S14 | `__version__` git-SHA fallback (not `0.0.0+unknown`); `tract_git_sha` primary |
| S15 | `TRACT_MODEL_REPO_ID` mirror override + continuity note |
| S19 | reuse `PHASE1D_INGEST_MAX_FILE_SIZE` cap on `assign --file` |

## Residual risk (accepted, documented)

- **Provenance ≠ integrity (S15):** the pin + hashes bind to values a contributor PR can
  also edit. Interim control: pin + CI hash-consistency check. Full fix (sigstore / HF
  commit signing) deferred — heavyweight for a research tool; the
  `verifying-sigstore-signatures` skill is the path when warranted.
- **Bus-factor (S15G):** single personal HF account is now a runtime dependency.
  Mitigated by `TRACT_MODEL_REPO_ID` mirror + documented continuity; an org-account move
  is a separate publication-governance decision.
- **Model_version honesty (S16):** the npz's adapter-hash is meaningless for the published
  merged model; the honest fix (record the revision) is entangled with `content_hash`/diff
  idempotency and is deferred to a diff-safe follow-up. The pin binds npz + weights from
  one commit in the interim.
- **Disclosure (S17):** the published crosswalk was generated via the direct
  `predict_batch` paths (`ground_truth.py`, `review/export.py`), **not** `assign --file`,
  so the sort/truncate bug did not contaminate it — confirm during implementation; no
  erratum expected.
- **Reproducibility (S18):** record which model revision produced the published dataset.
- **Tail risk:** `--output` path traversal would become High if `tract` is ever wrapped as
  a network service; out of scope for the local CLI threat model today.
- **PII in binaries:** the publish-time secret scan does not cover `.npz`/`.safetensors`;
  a follow-up to extend it.
