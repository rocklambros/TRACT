# LoRA checkpoint fix — PR #62, branch `semantic-rebuild`

Status: **done**. All eight jobs on PR #62 pass. `training-stack` reports
`162 passed, 3 warnings` (was `2 failed, 157 passed`).

Commit: `3bc5239` — `fix: write a base config so a LoRA checkpoint can be loaded back`

## The failing tests were not the ones named in the brief

The brief named `TestLoRACheckpointPersistence::test_verify_checkpoint_roundtrip_passes_on_match`
and `TestSaveCheckpoint::test_saves_metadata`. Both of those were **passing** at
`d3ccae2`. The two that failed were:

- `tests/test_training_loop.py::TestLoRACheckpointPersistence::test_adapter_survives_save_and_reload`
- `tests/test_publish_merge.py::TestMergeRealAdapter::test_merges_adapter_only_checkpoint`

The brief's list appears to predate commit `96cb938`, which reworked that test
class. The two tests it named passed precisely because they went through
`_reload_saved_model`, the bespoke loader that hid the defect.

## Root cause

Both failures raised the same error from the same frame:

```
AutoProcessor.from_pretrained(<checkpoint dir>)      transformer.py:671
  -> AutoConfig.from_pretrained(<checkpoint dir>)    processing_auto.py:363
     ValueError: Unrecognized model in <dir>
     config_dict = {}
```

`config_dict = {}` in the CI traceback is the decisive fact: the checkpoint has
**no `config.json` at all**.

Read from the pinned wheels:

1. `transformers.PreTrainedModel.save_pretrained` guards the config write with
   `if not _hf_peft_config_loaded` (`modeling_utils.py:3914`, 4.57.6). A model
   carrying an injected PEFT adapter therefore saves `adapter_config.json` and
   the adapter weights and **no base `config.json`**, on the reasoning that the
   adapter already names its base.
2. sentence-transformers 5.7 resolves the *model* correctly through that:
   `Transformer._load_config` calls `find_adapter_config_file` first and returns
   a `PeftConfig` (`base/modules/transformer.py:1747`). `_load_model` succeeds.
3. `Transformer.__init__` then calls `AutoProcessor.from_pretrained` on the same
   directory (`transformer.py:671`). AutoProcessor walks processor config,
   image-processor config, video processor, feature extractor, then
   `processor_class` inside `tokenizer_config.json` — an ordinary BERT tokenizer
   has none of them — and falls through to `AutoConfig.from_pretrained(dir)`.
   No `config.json`, so it raises.

The checkpoint's weights are all correct and nothing can open it. This is the
same shape as the other defects on this branch: well-formed output that is
wrong.

Every production consumer of a training checkpoint uses the plain path and hits
this: `tract/active_learning/model_io.py:26` (`load_fold_model`),
`tract/publish/merge.py:95` (`merge_lora_adapters`), and anyone loading the
published artifact.

### Against the prior diagnosis

The prior agent reported "one root cause with two call sites bypassing
`loop.py:_reload_saved_model`". It identified the right two call sites and the
wrong conclusion. Routing them through `_reload_saved_model` would have made the
tests green while leaving every checkpoint on disk unopenable — the tests assert
that a plain `SentenceTransformer(dir)` reproduces the model, and that assertion
is the requirement, not an inconvenience.

`_reload_saved_model`'s own docstring also stated the mechanism incorrectly: it
claimed `SentenceTransformer(saved_dir)` fails because the directory cannot
describe an architecture. sentence-transformers handles that fine. The break is
in the processor, one call later.

## What changed

- **`tract/training/checkpoint.py`** (new). `save_sentence_transformer` saves the
  model then writes the backbone's own config beside the adapter when
  `config.json` is absent, and fails loud if it cannot. `assert_loadable_checkpoint`
  is the reader-side guard. The module imports no ML stack, which is what makes
  its logic testable off a GPU host.
- **`tract/training/loop.py`**. `save_checkpoint` uses the new save path.
  `_reload_saved_model` is deleted; `verify_checkpoint_roundtrip` now reloads
  with a plain `SentenceTransformer(saved_dir)`. A guard that loads differently
  from the consumer can pass on an artifact no consumer can open, and this one
  did — that is why the two tests the brief named were green.
- **`tract/publish/merge.py`**. Calls `assert_loadable_checkpoint` before
  loading, so a pre-fix checkpoint is named rather than reported as a corrupt
  model. Code change only; no publish was run.
- **Tests**. Assertions unchanged. Fixtures now build checkpoints through the
  save path under test instead of a bare `model.save()`. Four tests added:
  the checkpoint is self-describing; verification rejects an adapter-only
  directory; merge refuses one; plus 11 unit tests for the new module.

## Mutation testing

Ten mutants against `tract/training/checkpoint.py`, all killed:

| # | Mutant | Killed by |
|---|--------|-----------|
| M1 | skip config completion entirely | `test_completes_an_adapter_only_save` (+3) |
| M2 | always rewrite config.json from the live config | `test_leaves_an_existing_config_untouched` |
| M3 | drop the post-write existence check | `test_raises_when_the_config_write_produces_nothing` |
| M4 | drop the missing-backbone guard | `test_raises_when_the_module_exposes_no_backbone` |
| M5 | drop the missing-config guard | `test_raises_when_the_backbone_carries_no_config` |
| M6 | do not create the output directory | `test_creates_the_output_directory` (+5) |
| M7 | `is_file()` -> `exists()` in the guard | `test_a_directory_named_config_json_is_not_a_config` |
| M8 | guard passes everything | `test_rejects_an_adapter_only_checkpoint` (+2) |
| M9 | adapter-only reported as the generic error | `test_rejects_an_adapter_only_checkpoint` (+1) |
| M10 | complete the config by deleting the adapter | `test_completes_an_adapter_only_save` |

M11, removing `assert_loadable_checkpoint` from `merge.py`, killed by
`test_refuses_an_adapter_only_checkpoint`. That mutant survived the first pass
and is why that test exists.

No survivors.

## What is not verified without a GPU

The save/reload round trip is verified end to end on real weights: the
`training-stack` runner installs the pinned stack and exercises
`all-MiniLM-L6-v2` on CPU. What that does not cover:

- The fix inside a real RunPod fold — BGE-large or Qwen3-Embedding, fp16,
  gradient checkpointing, `save_checkpoint` at the end of training. The written
  config comes from the live backbone and is architecture-independent, and
  BGE-large is the same BERT family as the CI model, so the residual risk is
  low but it is not zero for the Qwen arm.
- Numerical equality of a merged publish artifact built from a repaired
  checkpoint. `merge_lora_adapters` is covered by the CI test on MiniLM; no
  publish was run.

## Operational consequence the owner should see

Every fold checkpoint already on disk is adapter-only with no `config.json`:

```
results/phase1b/c2_A2_prose_sw_bge_bal3/fold_CWE/model/model      adapter=y config=n
results/phase1b/lofo_prose_stopwords/fold_NIST_AI_100-2/model/model  adapter=y config=n
... 98 adapter_config.json across results/
```

None of them can be loaded by `load_fold_model` today. This change does not
repair them — it makes the failure explicit and names the fix instead of raising
a transformers error about an unrecognised model. Any downstream step that needs
those fold models (active learning, calibration against a fold, re-publishing)
needs them re-saved, or the base `config.json` copied in beside each adapter.
That is an owner decision, not something to do silently.

## Local test summary

`2419 passed, 11 failed` (baseline `2407 passed, 9 failed`). The 11 are the 9
pre-existing model-loading failures plus the 2 model-loading tests added here;
all 11 fail on `ModuleNotFoundError: No module named 'datasets'` in this
environment, and all 11 pass in CI. `ruff` and `mypy --strict` clean on the CI
scope.
