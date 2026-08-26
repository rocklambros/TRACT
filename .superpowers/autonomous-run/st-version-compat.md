# sentence-transformers version compatibility

Status: shipped. The reported defect does not exist. The guard rails around it now do.

## The premise was wrong, and the evidence is direct

The task stated that four imports raise `ModuleNotFoundError` under
`sentence-transformers==5.7.0`, so `scripts/phase1b/runpod_parallel.py` would
provision a GPU fleet and die at import while the pods bill.

That is false. I built an isolated venv under the scratch directory carrying the
exact training pin (sentence-transformers 5.7.0, torch 2.13.0, transformers
4.57.6) and ran the original import lines:

```
from sentence_transformers.sampler import DefaultBatchSampler          -> OK
from sentence_transformers.losses import MultipleNegativesRankingLoss  -> OK
from sentence_transformers.training_args import BatchSamplers          -> OK
```

All three succeed. They emit `DeprecationWarning` and resolve.

The mechanism: 5.7.0 installs `_DeprecatedModuleFinder` on `sys.meta_path`, from
`sentence_transformers/util/deprecated_import.py`. Its `DEPRECATED_MODULE_PATHS`
dict maps every old path to its new home, including all three above. The
briefing's evidence (the wheel's top-level directory listing) is accurate about
the files on disk and simply does not capture a runtime meta-path alias.

Nothing in the repo turns warnings into errors. `pyproject.toml` sets no
`filterwarnings`, and neither CI nor `runpod_parallel.py` passes `-W error` or
`PYTHONWARNINGS`. So the warning is not fatal on any path.

The `training-stack` CI job would also have caught a real breakage: it runs
`tests/test_training_loop.py`, which imports `tract.training.loop` inside test
bodies. That catch was indirect and fragile, and it missed
`scripts/phase1b/alpha.py` entirely, which no test imports.

## What is real

The aliases are documented as temporary ("will be removed in a future version").
When they go, the pods do die at import, after the fleet is billing. The
remaining work is therefore about the class of defect rather than a live
instance of it.

## Compatibility matrix

Read from the published wheels for each version, then confirmed by importing the
real 3.2.0 (installed here) and the real 5.7.0 (isolated scratch venv).

| symbol | 3.2.0 (serving pin) | 5.3.0 (published build stack) | 5.7.0 (training pin) |
|---|---|---|---|
| `DefaultBatchSampler` | `sentence_transformers.sampler` | `sentence_transformers.sampler` | `sentence_transformers.base.sampler` |
| `MultipleNegativesRankingLoss` | `sentence_transformers.losses` | `sentence_transformers.losses` | `sentence_transformers.sentence_transformer.losses` |
| `BatchSamplers` | `sentence_transformers.training_args` | `sentence_transformers.training_args` | `sentence_transformers.sentence_transformer.training_args` |

Notes that shaped the ladder:

- No single literal path covers all three versions, so a shim is required rather
  than a rewrite to the new paths.
- The top-level package re-exports `DefaultBatchSampler` on 5.3.0 and 5.7.0 but
  not on 3.2.0, and re-exports neither `MultipleNegativesRankingLoss` nor
  `BatchSamplers` on any version. The package root is not a usable path.
- `SentenceTransformer`, `SentenceTransformerTrainer` and
  `SentenceTransformerTrainingArguments` are top-level in all three versions, so
  `loop.py:19` and `alpha.py:23` needed no change. That answers the briefing's
  open question about the bare-package imports.
- `MultipleNegativesRankingLoss` exists twice on 5.7.0, under `cross_encoder`
  and under `sentence_transformer`. TRACT trains a bi-encoder, so only the
  `sentence_transformer` copy is in the ladder.
- On 5.7.0 `BatchSamplers` is *defined* in `base/sampler.py`. The ladder targets
  `sentence_transformer/training_args`, which re-exports it in `__all__`, because
  that is the module the old path aliases to.

## Import strategy

One compatibility shim, `tract/training/st_compat.py`, resolving each symbol
through an ordered `(module, attribute)` ladder that fails loud with the
installed version and every path tried.

Trade-offs weighed. A try/except ladder at each of the three call sites was
rejected because the sites sit in two packages and would drift apart, and
`alpha.py` is imported by no test. Version-gated imports (`if version >= 5.4`)
were rejected because they encode a guess about the boundary rather than a fact
read from a distribution. The shim costs one indirection and type precision:
mypy now sees the three symbols as `Any` rather than real classes, so the
constructor calls at `loop.py:213` and `alpha.py:141` are no longer checked.
That is the main cost of the design and it is deliberate.

Two constraints honoured. There is no bare `except ImportError: pass` anywhere:
an exhausted ladder raises `SentenceTransformersLayoutError` naming the installed
version, every candidate tried, and the reason each failed.
`ModuleNotFoundError` is caught only when the missing module is the candidate
itself or one of its parent packages (`_covers`), so a missing torch or datasets
underneath a module that does exist propagates instead of masquerading as a
layout change. An untested version is never silently accepted: resolution warns,
and `require_tested_version` raises for callers about to spend money.

A side benefit worth recording: the sampler base class is now `Any` in every
environment rather than a real class on a dev machine and `Any` in the lint job,
so the mypy verdict for `tract/training/data.py:203` no longer depends on which
extras are installed. Verified by running `mypy --strict` twice, once here with
ST 3.2.0 and torch present and once in a scratch venv built from
`requirements-lint.txt` with neither. Both report `Success: no issues found in
152 source files`.

## Does the preflight block provisioning

Yes. `_preflight_training_stack()` is the first statement in
`scripts/phase1b/runpod_parallel.py:provision()`, ahead of the existing
`_preflight_tracking()`. This is the right place: it is the single function that
every provisioning path reaches, including `full_pipeline()`.

It does two things. The load-bearing one reads the
`sentence-transformers==` pin out of `requirements-train.txt`, which is what the
pods install, and refuses when that version is absent from `TESTED_VERSIONS`.
The weaker one resolves all three symbols locally and logs the version. The
local resolve proves little on its own, because the provisioning host normally
carries the serving pin rather than the training pin, and the code says so.

Ordering is asserted by a test, not assumed: the deterministic offline check must
not sit behind a network call to WandB.

## Would CI now catch it

Yes, at two levels.

The default `test` job (no ML stack) runs the pin-parsing, version-gate,
`_covers`, resolver control-flow and provisioning-block tests on every PR. That
layer catches a pin bump to an unverified version.

The `training-stack` job gains a step, "Prove the training modules import under
the pinned stack", placed before the regression tests. It imports
`tract.training.data`, `tract.training.loop` and `scripts.phase1b.alpha` under
the real pinned stack and prints which module each symbol resolved from. It
costs seconds and it covers `alpha.py`, which the previous test selection missed.
`tests/test_st_compat.py` is added to that job's pytest selection so the layout
matrix is asserted against a real 5.7.0.

## Tests

45 new tests in `tests/test_st_compat.py`: 44 pass and 1 skips here (it needs
`datasets`), and all 45 pass under the real 5.7.0 scratch venv. Suite went from
1,907 passing to 1,951, with the same 9 pre-existing environmental failures
(missing `datasets` and `peft`) before and after.

## Mutations

Run with `PYTHONDONTWRITEBYTECODE=1` against a pristine snapshot, restored before
and after each.

| id | wrong implementation | verdict |
|---|---|---|
| M1 | `_covers` uses a bare `startswith` with no package boundary | KILLED |
| M2 | `_covers` always returns True | KILLED |
| M3 | pin regex accepts `>=` and `~=` as a pin | KILLED |
| M4 | pin regex matches mid-line, so a commented pin counts | KILLED |
| M5 | conflicting duplicate pins accepted, first one wins | KILLED |
| M6 | `require_tested_version` warns instead of raising | KILLED |
| M7 | MNRL candidate order swapped, 3.x path first | SURVIVED locally, KILLED under 5.7.0 |
| M8 | a missing dependency treated as a layout mismatch | KILLED |
| M9 | exhausted ladder returns `(None, "")` instead of raising | KILLED |
| M10 | preflight never called from `provision()` | KILLED |
| M11 | preflight runs after the tracking preflight | KILLED |
| M12 | preflight computes the pin then ignores the verdict | KILLED |
| M13 | `resolve_symbol_source` reports `obj.__module__` | KILLED |

M7 is the interesting one and the result was predicted before running it. Under
ST 3.2.0 both candidate orderings resolve from `sentence_transformers.losses`,
because the 5.7 path does not exist, so no local test can distinguish them. It
dies under a real 5.7.0, where the swapped order resolves through the deprecated
alias and `test_resolution_matches_the_layout_matrix_for_this_version` catches
the wrong source. That is precisely the defect class the `training-stack` job
exists to cover, and it is evidence the cross-version job earns its runtime.

## Concerns

1. The premise was wrong, so this change buys future-proofing and a spend gate,
   not a rescued campaign. Anyone reading the commit as "fixed a crash" is being
   misled. There was no crash at 5.7.0.
2. Type precision at three call sites is now lower. The two loss constructors and
   the `BatchSamplers` member access are unchecked by mypy. The sampler base class
   was already unchecked.
3. The local half of the preflight validates the provisioning host's stack, which
   is usually 3.2.0 and not what the pods run. Only the pin check constrains the
   pod. A pod-side import failure under a version that is in `TESTED_VERSIONS` but
   broken for another reason still costs money.
4. `TESTED_VERSIONS` holds exact versions, so a patch bump to 5.7.1 blocks
   provisioning until someone reads that wheel. That is deliberate, and it will
   read as friction the first time it fires.
5. The 5.7.0 verification venv lives in scratch and is not reproducible from the
   repo. The matrix in `tests/test_st_compat.py` encodes the finding, and CI
   re-checks the 5.7.0 row on every run, but the 5.3.0 row is asserted by no
   running environment.
