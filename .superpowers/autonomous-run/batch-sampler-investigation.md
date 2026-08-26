# Batch sampler investigation

Question: does `sentence-transformers` ignore the `batch_sampler` value that
`tract/training/loop.py` passes, so that hub-aware temperature sampling has
never run?

**Verdict: (b). The sampler runs. The annotation is the only defect at that
line.** The hypothesis is refuted.

A second, unrelated defect surfaced during the investigation and is recorded at
the end. It is more serious than the one I was sent to find.

---

## Step 1 — what `batch_sampler` accepts, read from the library

Three versions matter, because this repo pins three different ones:

| pin | file | version | flat `sentence_transformers.sampler`? |
|---|---|---|---|
| serving | `requirements-ml.txt:39` | 3.2.0 | yes |
| pod retrain | `scripts/phase1c/runpod_retrain.py:172` | 5.3.0 | yes |
| training | `requirements-train.txt:45` | 5.7.0 | **no** |

### sentence-transformers 5.3.0 — the version the pods install

`sentence_transformers/training_args.py:294`

```python
self.batch_sampler = (
    BatchSamplers(self.batch_sampler) if isinstance(self.batch_sampler, str) else self.batch_sampler
)
```

Coercion is guarded by `isinstance(..., str)`. A class passes through untouched.
The field is declared at `training_args.py:231` as

```python
batch_sampler: Union[BatchSamplers, str, DefaultBatchSampler, Callable[..., DefaultBatchSampler]]
```

so a `DefaultBatchSampler` subclass is a documented, first-class value.

`sentence_transformers/trainer.py:672`

```python
# If the batch sampler is a DefaultBatchSampler subclass, initialize it
if inspect.isclass(self.args.batch_sampler) and issubclass(self.args.batch_sampler, DefaultBatchSampler):
    return self.args.batch_sampler(dataset, **batch_sampler_kwargs)

# If it's a callable, call it
if callable(self.args.batch_sampler):
    return self.args.batch_sampler(dataset, **batch_sampler_kwargs)
```

There is no silent fallback reachable by a class. The first branch matches; even
if it did not, a class is callable and the second branch would match. The enum
comparisons below those two branches are only reached by a non-callable value.

5.7.0 is identical in behaviour, relocated: `sentence_transformers/base/trainer.py:748`.

### sentence-transformers 3.2.0 — the serving pin, which mypy resolves locally

`training_args.py:170` coerces unconditionally:

```python
self.batch_sampler = BatchSamplers(self.batch_sampler)
```

Passing the class here raises `ValueError: ... is not a valid BatchSamplers`.
Loud, not silent. 3.2.0 also predates the `DefaultBatchSampler(dataset, ...)`
signature this sampler calls `super().__init__` with, so the class cannot be
constructed there either. No run has ever used 3.2.0 for training.

`self.generator`, `self.seed` and `self.epoch` all exist at runtime from 5.3.0
on: `sampler.py:76-77` assigns the first two, `SetEpochMixin.__init__`
(`sampler.py:40`) assigns the third.

## Step 2 — empirical, no model loaded

Built an isolated venv carrying sentence-transformers 5.3.0 and drove the
library's own `get_batch_sampler` with a stub `self` that holds only `.args`.
`get_batch_sampler` reads nothing else, so no trainer and no model is
constructed. Result:

```
issubclass(HubAware, DefaultBatchSampler) = True
args.batch_sampler after __post_init__ = <class 'tract.training.data.HubAwareTemperatureSampler'>
get_batch_sampler returned  = HubAwareTemperatureSampler
is HubAware instance        = True
__iter__ invocations        = 1
batches produced            = 5
full batches with hub collision = 0
temperature on instance     = 2.0

enum default returned       = DefaultBatchSampler
__iter__ invocations (enum) = 0
```

The sampler's own `__iter__` body executed and produced collision-free batches.
The negative control shows the enum arm does not reach it, so the positive
result is not vacuous.

**Single strongest piece of evidence:** `get_batch_sampler` returned a
`HubAwareTemperatureSampler` instance whose `__iter__` ran, driven through the
library's real dispatch rather than a reimplementation of it.

Hub-aware temperature sampling is therefore not a candidate explanation for the
recorded net-zero fine-tuning result. That finding needs a different cause.

## Step 3 — why the type errors appeared at all

The six mypy errors are **local-only**. Verified by replicating the CI lint job
exactly (fresh venv, `requirements-lint.txt` only, mypy 2.2.0):

```
Success: no issues found in 150 source files
```

`requirements-lint.txt` deliberately excludes the ML stack, and
`pyproject.toml` silences `sentence_transformers.*`, `torch.*` and `wandb.*`
with `ignore_missing_imports = true`. That setting applies only when the import
is **missing**. On a developer machine that installed the optional `phase0`
extra, mypy resolves the real, py.typed `sentence_transformers` — at the
**serving** pin, 3.2.0 — and type-checks training code against a library
version it never runs on. Four of the six errors are that mismatch. The fifth
(`unused-ignore`) is the same mechanism inverted. The sixth is a genuine wandb
protocol error, visible only because wandb is installed locally and absent from
the lint job.

The premise that CI is red was wrong. The real defect is that **the gate's
verdict depends on which optional packages happen to be installed**, so a
developer and CI disagree about whether the repo type-checks, and neither is
told. That is the same class of problem `requirements-lint.txt` was created to
fix for unpinned ruff and mypy, running in the other direction.

## Step 4 and 5 — fixes

All six now clear under **both** resolutions, local and CI-identical.

| error | fix | ignore added |
|---|---|---|
| `loop.py:267` arg-type | comment citing the library dispatch, narrowed ignore | `[arg-type, unused-ignore]` |
| `data.py:386-389` generator/seed | `__init__` now assigns both itself | none |
| `data.py:190` unused ignore | narrowed | `[misc, unused-ignore]` |
| `tracking.py:222` Run vs WandbRun | protocol `url` is a read-only property returning `str \| None`, matching `wandb.sdk.wandb_run.Run.url` | none |

`unused-ignore` is listed alongside the real code rather than used bare,
because the same line is checked under two resolutions and is an error under
exactly one of them. Both sites carry a comment saying so.

### Tests

Added to `tests/test_training_data.py`:

- `TestSamplerAttributeContract` — three tests. The attributes exist after
  construction, the seed changes the emitted order, and the generator both
  reproduces and overrides the seed. An attribute that is present but ignored
  is the same silent no-op as one that is missing, so presence alone is not
  asserted.
- `TestTrainerReachesTheCustomSampler` — three tests.
  `test_library_dispatch_instantiates_and_iterates_the_class` drives the real
  `get_batch_sampler` and observes `__iter__` execute.
  `test_enum_batch_sampler_does_not_reach_the_custom_sampler` is the negative
  control. `test_train_model_wires_the_class_into_the_training_arguments` runs
  the real `train_model` with the model, loss and trainer stubbed, so the
  training arguments are built by production code, then asks the library which
  sampler that configuration selects. That last one is what fails if
  `loop.py` stops passing the class.

### Mutation testing

Pristine snapshot restored before AND after each mutation,
`PYTHONDONTWRITEBYTECODE=1`. Baseline 31 passed, post-restore 31 passed.

| id | mutation | outcome | killed by |
|---|---|---|---|
| M1 | `loop.py` always hands the trainer the library default | KILLED | `test_train_model_wires_the_class_into_the_training_arguments` |
| M2 | `__init__` stops owning `generator`/`seed` | **SURVIVED (expected)** | — |
| M3 | `__iter__` ignores `self.seed` and `self.epoch` | KILLED | `test_seed_selects_the_order_when_no_generator_is_given`, `test_epoch_changes_order` |
| M4 | `__iter__` ignores `self.generator` | KILLED | `test_generator_overrides_the_seed_when_given` |
| M5 | sampler stops being a `DefaultBatchSampler` subclass | KILLED | 16 tests, including both dispatch tests |
| M6 | `__iter__` drops the trailing partial batch | KILLED | `test_library_dispatch_instantiates_and_iterates_the_class`, `test_set_metadata_works_without_hub_id_column` |
| M7 | harness check: dispatch always returns the custom sampler | KILLED | `test_enum_batch_sampler_does_not_reach_the_custom_sampler` |
| M8 | `__init__` pins `seed` to 0 | KILLED | `test_generator_and_seed_survive_construction`, `test_seed_selects_the_order_when_no_generator_is_given` |
| M9 | `__init__` discards the `generator` argument | KILLED | `test_generator_and_seed_survive_construction`, `test_generator_overrides_the_seed_when_given` |

**M2 is a declared survivor, not a test gap.** Deleting the two assignments is
masked because `DefaultBatchSampler.__init__` sets the same values on every
version this code can run on. That is precisely the condition under which
deleting them is harmless at runtime, so a test that killed M2 would be
asserting something false. The lines earn their place statically, and M8 and M9
show the assertions fail when the values are wrong in either direction.

**M6 found a real weakness in a first draft of the test.** The dispatch test
used 40 examples at batch size 8, so the trailing-partial-batch path was never
taken and the completeness assertion could not fail through it. Changed to 45,
after which M6 kills that test. This is recorded because the first version of
the test looked correct and asserted less than it appeared to.

---

## Separate finding, higher severity: the pinned training stack cannot import

`requirements-train.txt` pins `sentence-transformers==5.7.0`, and
`scripts/phase1b/runpod_parallel.py:739` installs that file on every pod.
5.7.0 restructured the package: `sentence_transformers/sampler.py`,
`losses/` and `training_args.py` no longer exist at the top level, there is no
`__getattr__` shim and no `sys.modules` alias. Verified against the 5.7.0
sdist. Three modules import paths that were deleted:

- `tract/training/data.py:20` — `from sentence_transformers.sampler import DefaultBatchSampler`
- `tract/training/loop.py:24` — `from sentence_transformers.losses import MultipleNegativesRankingLoss`
- `tract/training/loop.py:25` — `from sentence_transformers.training_args import BatchSamplers`
- `scripts/phase1b/alpha.py:28` — `from sentence_transformers.losses import MultipleNegativesRankingLoss`

Under the pinned training stack all four raise `ModuleNotFoundError`, so the
next `runpod_parallel.py` run fails at import after the pod is already paid for.
This was never caught because the `training-stack` CI job that installs 5.7.0
is new on this unpushed branch and has never run.

New locations in 5.7.0:

- `DefaultBatchSampler` — top level, `from sentence_transformers import DefaultBatchSampler`. Also top level in 5.3.0, absent from 3.2.0.
- `BatchSamplers` — `sentence_transformers.base.sampler`
- `MultipleNegativesRankingLoss` — `sentence_transformers.sentence_transformer.losses`

`SentenceTransformer`, `SentenceTransformerTrainer` and
`SentenceTransformerTrainingArguments` are top level in all three versions and
need no change.

Not fixed here. It is a distinct defect from the one under investigation, it
cannot be verified locally without installing torch 2.13 and transformers
4.57.6, and the right shape of the fix depends on an owner decision about
whether the lint environment should keep resolving the serving pin. Fixing the
imports would also let `tests/test_training_data.py` be added to the
`training-stack` CI job, where the new dispatch tests would run for real. They
are dormant everywhere in CI today.

## Verification

- ruff: clean on the CI paths.
- mypy `--strict`, local (ML stack present, ST 3.2.0): `Success: no issues found in 151 source files`.
- mypy `--strict`, CI-identical venv (`requirements-lint.txt` only): `Success: no issues found in 151 source files`.
- `tests/test_training_data.py` + `tests/test_branch_balancing.py` under ST 5.3.0: 40 passed.
- Full local suite: 1,893 passed, 9 failed, 23 skipped. The 9 are the documented
  environmental model-loading failures in `test_training_loop.py` and
  `test_publish_merge.py`, all `ModuleNotFoundError` on the absent optional
  stack. No regressions. The passing count is above the carried-forward 1,841
  because a concurrent task added parser tests.
- No model was loaded on this machine.
