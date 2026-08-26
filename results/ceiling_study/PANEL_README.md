# LLM judge panel over the ceiling study

Five model families answer the same blind annotation question the human
annotator answered, so that two explanations for CAPEC's alpha-1 of 0.181 can
be told apart:

  (a) OpenCRE's CAPEC links are poor.
  (b) the single human annotator's reading of CAPEC is idiosyncratic.

They predict opposite things. Under (b), independent judges land where OpenCRE
landed and the human is the outlier. Under (a), judges that never saw the
human's answers land near the human and away from OpenCRE.

## The panel

| model | lab | route model id | pinned backend | quantization |
|---|---|---|---|---|
| `moonshotai/Kimi-K3` | Moonshot (CN) | `moonshotai/kimi-k3` | DeepInfra | bf16 |
| `z-ai/GLM-5.3` | Zhipu (CN) | `z-ai/glm-5.3` | Z.AI | fp8 |
| `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek (CN) | `deepseek/deepseek-v4-pro` | StreamLake | fp8 |
| `meta-llama/Llama-4-Maverick` | Meta (US) | `meta-llama/llama-4-maverick` | DeepInfra | fp8 |
| `x-ai/Grok-4.20` | xAI (US) | `x-ai/grok-4.20` | xAI | first party |

Five distinct labs, not five checkpoints of one. Correlated pretraining would
make agreement between two judges evidence of shared lineage rather than
evidence about the label.

Five is odd on purpose. An even panel can split 2-2 on exactly the contested
CAPEC items the study turns on, and a tie there leaves no majority to compare
the human against. Three Chinese labs and two American ones, so convergence
cannot be dismissed as a single training-data monoculture agreeing with itself.

Each model is pinned to one serving backend with `allow_fallbacks: false`.
OpenRouter otherwise load-balances a single model id across backends that
differ in quantization, and the same prompt served at fp4 and at bf16 is not
the same judge. DeepSeek's own first-party endpoint returns 404 for this
account at every token budget, so that one is pinned to the best third party
rather than to the vendor.

## Running it

Dry run is the default and spends nothing. It resolves the route, renders the
exact prompts, counts tokens, and prints a cost estimate:

```bash
python -m scripts.run_panel --model z-ai/GLM-5.3
```

Spending requires `--execute`:

```bash
for m in moonshotai/Kimi-K3 z-ai/GLM-5.3 deepseek-ai/DeepSeek-V4-Pro \
         meta-llama/Llama-4-Maverick x-ai/Grok-4.20; do
    python -m scripts.run_panel --model "$m" --execute --resume
done
```

Then the analysis, which needs no key:

```bash
python -m scripts.analyze_panel_agreement
```

### Credentials

`scripts/run_panel.py` takes the first route that has a credential, in this
order. Every one of them is OpenAI-compatible, so the client is the same.

| route | credential | covers |
|---|---|---|
| `openrouter` | `pass openrouter/api-key` or `$OPENROUTER_API_KEY` | all five |
| `moonshot` | `pass moonshot/api-key` or `$MOONSHOT_API_KEY` | Kimi only |
| `deepseek` | `pass deepseek/api-key` or `$DEEPSEEK_API_KEY` | DeepSeek only |
| `zhipu` | `pass zhipu/api-key` or `$ZHIPU_API_KEY` | GLM only |
| `hf_router` | `pass huggingface/read-token` or `$HF_TOKEN` | Kimi, DeepSeek |

Force one with `--route`. OpenRouter is the only route that reaches all five.
The HuggingFace router works and needs no new vendor relationship, but its free
monthly allowance is small: on 2026-08-18 it served one batch per model and then
returned HTTP 402 with included credits exhausted, resetting on the first of the
month.

If no route has a credential, the run refuses rather than falling back to a
model nobody asked for.

### Cost

The dry run estimates from a measured 3.81 characters per token. The executed
run records OpenRouter's own `usage.cost` per response, which is what was
actually billed, and sums failed attempts into the total.

| model | est. cost, 250 items |
|---|---|
| Kimi K3 | ~$2.60 |
| GLM-5.3 | ~$1.70 |
| DeepSeek V4 Pro | ~$0.80 |
| Llama 4 Maverick | ~$0.25 |
| Grok 4.20 | ~$1.50 |

The hub reference is sent on every batch and is roughly 114k tokens, which is
where nearly all the input cost goes. Prompt caching would cut it, but no route
granted a meaningful cache hit on this shape of request, so the estimates
assume none.

### Three parameters that fail quietly

**Reasoning effort must be the object form.** OpenRouter's unified parameter is
`reasoning: {"effort": "low"}`. The OpenAI-style `reasoning_effort: "low"`
string is accepted and silently ignored: a control call sent that way spent all
3,000 of its allowed tokens on reasoning, while the object form spent 560.
Every panel member is a thinking model and reasoning bills at the output rate,
so the wrong spelling runs at full effort and charges for it.

**`max_tokens` is capped by the tightest pinned endpoint,** 16,384 for DeepInfra.
Requesting more does not raise a parameter error. OpenRouter filters out every
backend that cannot honour the request and returns `No endpoints found for
<model>`, which reads exactly like the model not existing. A 25-item batch
measured 2,662 completion tokens including reasoning, so 16,384 is ample.

**A truncated answer is not an empty answer.** `finish_reason: "length"` returns
non-empty text that parses to nothing, and would otherwise be recorded as 25
items the judge declined to answer, quietly dropping them from the denominator.
Both truncation and a null content are treated as failed calls and retried.

## Files

| file | what it is |
|---|---|
| `ceiling_items.json` | the 250 items, no gold, no shortlist |
| `hub_reference.md` | the 522-hub taxonomy the annotator reads |
| `answers_human_rock.json` | the human's answers. **Never modify** |
| `ceiling_answer_key.json` | OpenCRE's labels. **Never modify** |
| `answers_panel_<slug>.json` | one judge's answers, plus its `run` pin block |
| `answers_panel_<slug>.partial.json` | mid-run checkpoint, removed on success |
| `contamination_probe_<slug>.json` | closed-book recall, contaminable arm |
| `contamination_control_<slug>.json` | closed-book recall, post-cutoff control arm |
| `contamination_control_items.json` | the 10 post-cutoff control items |
| `panel_agreement.md` | the analysis |

Each `answers_panel_*.json` matches the human worksheet's schema, so the
unmodified scorer works on it directly:

```bash
python -m scripts.score_ceiling_study --answers results/ceiling_study/answers_panel_z_ai_glm_5_3.json
```

The extra `run` key records model id, route, served snapshot, temperature,
reasoning effort, batch size, per-request token usage, failed attempts, and
cost. An unpinned judge is an unreproducible judge, and these numbers will be
cited against the human ceiling.

## Contamination probe

OpenCRE's mappings are public. If a judge memorised them, its agreement with
OpenCRE measures recall rather than judgement.

```bash
python -m scripts.build_contamination_control          # builds the control items
for m in moonshotai/Kimi-K3 z-ai/GLM-5.3 deepseek-ai/DeepSeek-V4-Pro \
         meta-llama/Llama-4-Maverick x-ai/Grok-4.20; do
    python -m scripts.run_panel --model "$m" --execute --closed-book
    python -m scripts.run_panel --model "$m" --execute --closed-book \
        --items results/ceiling_study/contamination_control_items.json \
        --out results/ceiling_study/contamination_control_$(echo "$m" | tr 'A-Z/.' 'a-z__').json
done
```

This withholds the hub taxonomy and asks the model to state the hub id OpenCRE
publishes for each control. Chance is 1 in 522. The negative control is the
2026 edition of the OWASP LLM Top 10, whose source document is dated August
2026 while every panel member predates it,
so it postdates every training cutoff and OpenCRE has never mapped it. Every
hub id a model emits for those ten controls is confabulation by construction,
which is the base rate needed to read the contaminable arm.

The probe bounds the direct memorisation channel. It does not establish that a
model never saw OpenCRE, because verbatim recall is a much stronger property
than exposure, and it does not address contamination running through the
frameworks themselves.

## Rules the panel run must not break

- Judges never see the human's answers, each other's answers, or the key.
- Items go in `item_index` order, so every prefix stays a valid stratified
  sample.
- A judge that cannot be reached is reported as not run. It is never replaced
  by a model that happens to be reachable, and its answers are never invented.
- An invented hub id is recorded as the judge gave it. Blanking it would drop
  the item from the denominator and inflate that judge's score.
