# LLM judge panel: three-way agreement over the ceiling study

Version 1.0. Owner: Rock Lambros.

## What this is

The human ceiling study returned pooled alpha-1 of 0.572, and CAPEC's
alpha-1 of 0.181 on 83 items against a framework that is 42.8% of
the training graph. This panel exists to discriminate two explanations:

  (a) OpenCRE's CAPEC links are poor.
  (b) the single human annotator's reading of CAPEC is idiosyncratic.

Each judge answered the runbook prompt verbatim, in `item_index` order,
with no sight of the human's answers, the other judges' answers, or the
key.

## Panel roster and pins

| model | ran | route | served snapshot | temp | reasoning | cost USD |
|---|---|---|---|---|---|---|
| `moonshotai/Kimi-K3` | yes | openrouter (`moonshotai/kimi-k3`) | `moonshotai/kimi-k3` | 0.0 | low | $1.3131 |
| `z-ai/GLM-5.3` | yes | openrouter (`z-ai/glm-5.3`) | `z-ai/glm-5.3` | 0.0 | low | $0.3919 |
| `deepseek-ai/DeepSeek-V4-Pro` | yes | openrouter (`deepseek/deepseek-v4-pro`) | `deepseek/deepseek-v4-pro` | 0.0 | low | $0.2314 |
| `meta-llama/Llama-4-Maverick` | yes | openrouter (`meta-llama/llama-4-maverick`) | `meta-llama/llama-4-maverick` | 0.0 | low | $0.1626 |
| `x-ai/Grok-4.20` | yes | openrouter (`x-ai/grok-4.20`) | `x-ai/grok-4.20` | 0.0 | low | $0.7819 |

Total panel cost $2.8809 across 5 models, including 10 failed or retried requests.

### Judge competence

Not every panel member is an equally credible judge, and a weak one should not carry a majority vote unexamined. Two signals, both independent of the answer key:

| model | invented hub ids | unanswered | alpha-5 minus alpha-1 |
|---|---|---|---|
| moonshotai/Kimi-K3 | 0 | 0 | +0.040 |
| z-ai/GLM-5.3 | 1 | 0 | +0.072 |
| deepseek-ai/DeepSeek-V4-Pro | 0 | 0 | +0.040 |
| meta-llama/Llama-4-Maverick | 17 | 0 | +0.016 |
| x-ai/Grok-4.20 | 0 | 2 | +0.040 |

An invented hub id is one the reference does not contain, so the judge was not reading the taxonomy it was given. These are recorded as the judge gave them rather than blanked, because blanking one would drop the item from the denominator and inflate that model's score.

A near-zero alpha-5 minus alpha-1 spread means the judge's shortlist adds nothing to its first choice, which is what a judge that is guessing looks like. The human's spread is +0.088.

## 1. Agreement with OpenCRE (alpha-1 and alpha-5)

Wilson 95% intervals, from `tract.stats.wilson_interval`. Scored by
`scripts.score_ceiling_study.score_items`, unmodified, so these numbers
are produced by the same code that produced the human's 0.572.

| annotator | scope | alpha-1 | alpha-5 |
|---|---|---|---|
| human (Rock) | pooled | 0.572 [0.510, 0.632] (n=250) | 0.660 [0.599, 0.716] (n=250) |
| human (Rock) | stratum: test | 0.848 [0.775, 0.900] (n=125) | 0.912 [0.849, 0.950] (n=125) |
| human (Rock) | stratum: validation | 0.296 [0.223, 0.381] (n=125) | 0.408 [0.326, 0.496] (n=125) |
| moonshotai/Kimi-K3 | pooled | 0.588 [0.526, 0.647] (n=250) | 0.628 [0.567, 0.686] (n=250) |
| moonshotai/Kimi-K3 | stratum: test | 0.864 [0.793, 0.913] (n=125) | 0.896 [0.830, 0.938] (n=125) |
| moonshotai/Kimi-K3 | stratum: validation | 0.312 [0.237, 0.398] (n=125) | 0.360 [0.281, 0.447] (n=125) |
| z-ai/GLM-5.3 | pooled | 0.576 [0.514, 0.636] (n=250) | 0.648 [0.587, 0.705] (n=250) |
| z-ai/GLM-5.3 | stratum: test | 0.840 [0.766, 0.894] (n=125) | 0.888 [0.821, 0.932] (n=125) |
| z-ai/GLM-5.3 | stratum: validation | 0.312 [0.237, 0.398] (n=125) | 0.408 [0.326, 0.496] (n=125) |
| deepseek-ai/DeepSeek-V4-Pro | pooled | 0.580 [0.518, 0.640] (n=250) | 0.620 [0.558, 0.678] (n=250) |
| deepseek-ai/DeepSeek-V4-Pro | stratum: test | 0.824 [0.748, 0.881] (n=125) | 0.888 [0.821, 0.932] (n=125) |
| deepseek-ai/DeepSeek-V4-Pro | stratum: validation | 0.336 [0.259, 0.423] (n=125) | 0.352 [0.274, 0.439] (n=125) |
| meta-llama/Llama-4-Maverick | pooled | 0.256 [0.206, 0.314] (n=250) | 0.272 [0.221, 0.330] (n=250) |
| meta-llama/Llama-4-Maverick | stratum: test | 0.384 [0.303, 0.472] (n=125) | 0.408 [0.326, 0.496] (n=125) |
| meta-llama/Llama-4-Maverick | stratum: validation | 0.128 [0.080, 0.198] (n=125) | 0.136 [0.087, 0.207] (n=125) |
| x-ai/Grok-4.20 | pooled | 0.484 [0.422, 0.546] (n=248) | 0.524 [0.462, 0.586] (n=248) |
| x-ai/Grok-4.20 | stratum: test | 0.728 [0.644, 0.798] (n=125) | 0.768 [0.687, 0.833] (n=125) |
| x-ai/Grok-4.20 | stratum: validation | 0.236 [0.169, 0.318] (n=123) | 0.276 [0.205, 0.361] (n=123) |

### Per framework, alpha-1

| annotator | capec | cwe | mitre_atlas | nist_800_53 | nist_ai_100_2 | owasp_ai_exchange | owasp_llm_top10 |
|---|---|---|---|---|---|---|---|
| human (Rock) | 0.181 [0.113, 0.277] (n=83) | 0.464 [0.295, 0.642] (n=28) | 0.721 [0.573, 0.833] (n=43) | 0.643 [0.388, 0.837] (n=14) | 0.773 [0.566, 0.899] (n=22) | 0.981 [0.902, 0.997] (n=54) | 0.833 [0.436, 0.970] (n=6) |
| moonshotai/Kimi-K3 | 0.205 [0.132, 0.304] (n=83) | 0.464 [0.295, 0.642] (n=28) | 0.744 [0.598, 0.851] (n=43) | 0.643 [0.388, 0.837] (n=14) | 0.818 [0.615, 0.927] (n=22) | 1.000 [0.934, 1.000] (n=54) | 0.667 [0.300, 0.903] (n=6) |
| z-ai/GLM-5.3 | 0.229 [0.152, 0.330] (n=83) | 0.429 [0.265, 0.609] (n=28) | 0.721 [0.573, 0.833] (n=43) | 0.571 [0.326, 0.786] (n=14) | 0.818 [0.615, 0.927] (n=22) | 0.944 [0.849, 0.981] (n=54) | 0.833 [0.436, 0.970] (n=6) |
| deepseek-ai/DeepSeek-V4-Pro | 0.229 [0.152, 0.330] (n=83) | 0.500 [0.326, 0.674] (n=28) | 0.721 [0.573, 0.833] (n=43) | 0.643 [0.388, 0.837] (n=14) | 0.818 [0.615, 0.927] (n=22) | 0.944 [0.849, 0.981] (n=54) | 0.500 [0.188, 0.812] (n=6) |
| meta-llama/Llama-4-Maverick | 0.060 [0.026, 0.133] (n=83) | 0.143 [0.057, 0.315] (n=28) | 0.302 [0.186, 0.451] (n=43) | 0.500 [0.268, 0.732] (n=14) | 0.409 [0.233, 0.613] (n=22) | 0.389 [0.270, 0.522] (n=54) | 0.833 [0.436, 0.970] (n=6) |
| x-ai/Grok-4.20 | 0.146 [0.086, 0.239] (n=82) | 0.370 [0.215, 0.558] (n=27) | 0.721 [0.573, 0.833] (n=43) | 0.500 [0.268, 0.732] (n=14) | 0.500 [0.307, 0.693] (n=22) | 0.833 [0.713, 0.910] (n=54) | 0.667 [0.300, 0.903] (n=6) |

### Per framework, alpha-5

| annotator | capec | cwe | mitre_atlas | nist_800_53 | nist_ai_100_2 | owasp_ai_exchange | owasp_llm_top10 |
|---|---|---|---|---|---|---|---|
| human (Rock) | 0.337 [0.245, 0.444] (n=83) | 0.464 [0.295, 0.642] (n=28) | 0.860 [0.727, 0.934] (n=43) | 0.714 [0.454, 0.883] (n=14) | 0.818 [0.615, 0.927] (n=22) | 0.981 [0.902, 0.997] (n=54) | 1.000 [0.610, 1.000] (n=6) |
| moonshotai/Kimi-K3 | 0.265 [0.182, 0.369] (n=83) | 0.500 [0.326, 0.674] (n=28) | 0.791 [0.648, 0.886] (n=43) | 0.643 [0.388, 0.837] (n=14) | 0.864 [0.667, 0.953] (n=22) | 1.000 [0.934, 1.000] (n=54) | 0.833 [0.436, 0.970] (n=6) |
| z-ai/GLM-5.3 | 0.325 [0.234, 0.432] (n=83) | 0.500 [0.326, 0.674] (n=28) | 0.814 [0.674, 0.903] (n=43) | 0.714 [0.454, 0.883] (n=14) | 0.818 [0.615, 0.927] (n=22) | 0.981 [0.902, 0.997] (n=54) | 0.833 [0.436, 0.970] (n=6) |
| deepseek-ai/DeepSeek-V4-Pro | 0.253 [0.172, 0.356] (n=83) | 0.500 [0.326, 0.674] (n=28) | 0.791 [0.648, 0.886] (n=43) | 0.643 [0.388, 0.837] (n=14) | 0.909 [0.722, 0.975] (n=22) | 0.963 [0.875, 0.990] (n=54) | 0.833 [0.436, 0.970] (n=6) |
| meta-llama/Llama-4-Maverick | 0.072 [0.034, 0.149] (n=83) | 0.143 [0.057, 0.315] (n=28) | 0.326 [0.205, 0.475] (n=43) | 0.500 [0.268, 0.732] (n=14) | 0.409 [0.233, 0.613] (n=22) | 0.426 [0.303, 0.558] (n=54) | 0.833 [0.436, 0.970] (n=6) |
| x-ai/Grok-4.20 | 0.195 [0.124, 0.294] (n=82) | 0.407 [0.245, 0.593] (n=27) | 0.767 [0.623, 0.868] (n=43) | 0.500 [0.268, 0.732] (n=14) | 0.545 [0.347, 0.731] (n=22) | 0.852 [0.734, 0.923] (n=54) | 0.833 [0.436, 0.970] (n=6) |

## 2. Human versus panel

The least contaminated comparison in this report. Neither side saw the
other's answers, and neither is derived from OpenCRE. Two annotators
converging on a hub that OpenCRE did not choose is evidence about the
label, not about either annotator.

| panel member | agreement with human, pooled | on CAPEC |
|---|---|---|
| moonshotai/Kimi-K3 | 0.772 [0.716, 0.820] (n=250) | 0.687 [0.581, 0.776] (n=83) |
| z-ai/GLM-5.3 | 0.768 [0.712, 0.816] (n=250) | 0.735 [0.631, 0.818] (n=83) |
| deepseek-ai/DeepSeek-V4-Pro | 0.772 [0.716, 0.820] (n=250) | 0.711 [0.606, 0.797] (n=83) |
| meta-llama/Llama-4-Maverick | 0.312 [0.258, 0.372] (n=250) | 0.241 [0.162, 0.343] (n=83) |
| x-ai/Grok-4.20 | 0.669 [0.609, 0.725] (n=248) | 0.646 [0.538, 0.741] (n=82) |
| **panel majority** | **0.780 [0.725, 0.827] (n=250)** | **0.711 [0.606, 0.797] (n=83)** |

## 3. Panel versus panel

Whether disagreement is structural or model-specific. Families that
disagree with OpenCRE in the same direction, while agreeing with each
other, are describing a property of the labels. Families that disagree
with OpenCRE in different directions are describing several different
confusions, and the majority vote over them means much less.

| pair | pooled | on CAPEC |
|---|---|---|
| moonshotai/Kimi-K3 vs z-ai/GLM-5.3 | 0.808 [0.755, 0.852] (n=250) | 0.711 [0.606, 0.797] (n=83) |
| moonshotai/Kimi-K3 vs deepseek-ai/DeepSeek-V4-Pro | 0.788 [0.733, 0.834] (n=250) | 0.735 [0.631, 0.818] (n=83) |
| moonshotai/Kimi-K3 vs meta-llama/Llama-4-Maverick | 0.336 [0.280, 0.397] (n=250) | 0.277 [0.192, 0.382] (n=83) |
| moonshotai/Kimi-K3 vs x-ai/Grok-4.20 | 0.685 [0.625, 0.740] (n=248) | 0.646 [0.538, 0.741] (n=82) |
| z-ai/GLM-5.3 vs deepseek-ai/DeepSeek-V4-Pro | 0.740 [0.682, 0.790] (n=250) | 0.651 [0.543, 0.744] (n=83) |
| z-ai/GLM-5.3 vs meta-llama/Llama-4-Maverick | 0.344 [0.288, 0.405] (n=250) | 0.253 [0.172, 0.356] (n=83) |
| z-ai/GLM-5.3 vs x-ai/Grok-4.20 | 0.685 [0.625, 0.740] (n=248) | 0.646 [0.538, 0.741] (n=82) |
| deepseek-ai/DeepSeek-V4-Pro vs meta-llama/Llama-4-Maverick | 0.316 [0.262, 0.376] (n=250) | 0.265 [0.182, 0.369] (n=83) |
| deepseek-ai/DeepSeek-V4-Pro vs x-ai/Grok-4.20 | 0.657 [0.596, 0.714] (n=248) | 0.610 [0.502, 0.708] (n=82) |
| meta-llama/Llama-4-Maverick vs x-ai/Grok-4.20 | 0.310 [0.256, 0.371] (n=248) | 0.232 [0.154, 0.334] (n=82) |

All 5 panel members chose the identical hub on 15 of 83 CAPEC items (18.1%).

## 4. The CAPEC three-way contingency

All 83 CAPEC items, 83 of them answered by both
the human and at least one panel member. Agreement with OpenCRE is
membership in the item's valid gold set; agreement between human and
panel is an identical primary hub.

| cell | n | share of scorable |
|---|---|---|
| both match OpenCRE | 15 | 18.1% |
| human matches OpenCRE, panel does not | 0 | 0.0% |
| panel matches OpenCRE, human does not | 1 | 1.2% |
| **human and panel agree, both differ from OpenCRE** | **44** | 53.0% |
| all three differ | 23 | 27.7% |

**Headline: 44 of 83 CAPEC items where the human and the panel majority chose the same hub and OpenCRE chose a different one.**

Read it against the two hypotheses. Under (b), a single human's idiosyncratic reading, this cell should be near zero: 5 unrelated model families have no reason to reproduce one person's private confusion. Every item in it is an item where two independent readings converged and the published label did not.

## 5. Where the disagreements land in the hierarchy

Same four categories the human ceiling analysis used, measured the same
way. Only alpha-1 misses are categorised, that is items where the
annotator's primary hub is not in the item's valid gold set, and the
comparison is against the key's `primary_gold_hub_id`. Both details
matter: including items that hit a non-primary gold hub, or comparing
against the whole valid set, changes the split. Measured this way the
human's row below is 9.3 / 3.7 / 17.8 / 69.2, which is what the human
ceiling analysis published, which is what makes the panel rows
comparable to it.

| annotator | scope | disagreements categorised |
|---|---|---|
| human (Rock) | pooled | ancestor/descendant 10 (9.3%), sibling 4 (3.7%), same branch 19 (17.8%), different branch 74 (69.2%) |
| human (Rock) | CAPEC | ancestor/descendant 1 (1.5%), sibling 1 (1.5%), same branch 9 (13.2%), different branch 57 (83.8%) |
| moonshotai/Kimi-K3 | pooled | ancestor/descendant 12 (11.7%), sibling 5 (4.9%), same branch 14 (13.6%), different branch 72 (69.9%) |
| moonshotai/Kimi-K3 | CAPEC | ancestor/descendant 0 (0.0%), sibling 1 (1.5%), same branch 7 (10.6%), different branch 58 (87.9%) |
| z-ai/GLM-5.3 | pooled | ancestor/descendant 11 (10.4%), sibling 8 (7.5%), same branch 17 (16.0%), different branch 69 (65.1%), unknown 1 |
| z-ai/GLM-5.3 | CAPEC | ancestor/descendant 1 (1.6%), sibling 1 (1.6%), same branch 6 (9.4%), different branch 56 (87.5%) |
| deepseek-ai/DeepSeek-V4-Pro | pooled | ancestor/descendant 10 (9.5%), sibling 5 (4.8%), same branch 17 (16.2%), different branch 73 (69.5%) |
| deepseek-ai/DeepSeek-V4-Pro | CAPEC | ancestor/descendant 0 (0.0%), sibling 1 (1.6%), same branch 9 (14.1%), different branch 54 (84.4%) |
| meta-llama/Llama-4-Maverick | pooled | ancestor/descendant 6 (3.2%), sibling 11 (5.9%), same branch 22 (11.8%), different branch 130 (69.9%), unknown 17 |
| meta-llama/Llama-4-Maverick | CAPEC | ancestor/descendant 1 (1.3%), sibling 1 (1.3%), same branch 5 (6.4%), different branch 65 (83.3%), unknown 6 |
| x-ai/Grok-4.20 | pooled | ancestor/descendant 11 (8.6%), sibling 6 (4.7%), same branch 27 (21.1%), different branch 84 (65.6%) |
| x-ai/Grok-4.20 | CAPEC | ancestor/descendant 1 (1.4%), sibling 1 (1.4%), same branch 9 (12.9%), different branch 59 (84.3%) |
| panel majority | pooled | ancestor/descendant 12 (11.5%), sibling 5 (4.8%), same branch 16 (15.4%), different branch 71 (68.3%) |
| panel majority | CAPEC | ancestor/descendant 0 (0.0%), sibling 1 (1.5%), same branch 8 (11.9%), different branch 58 (86.6%) |

A disagreement in a different branch is not a near miss. It means the two readings disagree about what kind of thing the control is, not about how finely to file it.

## 6. Contamination probe

### What it tests

OpenCRE's mappings are published on opencre.org and in a public GitHub
repository. If a panel member memorised them, its agreement with
OpenCRE measures recall, not judgement, and every number in section 1
is inflated. The probe asks each model, with the hub taxonomy withheld,
to state the hub id OpenCRE publishes for each control. Chance is 1 in
522 hubs, so a memorising model is easy to see.

The negative control is `owasp_llm_top10`'s 2026
edition (`data/processed/frameworks/owasp_llm_top10_2026.json`, fetched
2026-08-16, source document dated August 2026). All three panel members
were released between April and June 2026, so that document postdates
every one of their training cutoffs, and OpenCRE has never mapped it.
Any hub id a model produces for it is confabulation by construction,
which calibrates how readily each model emits a plausible-looking id
when it cannot possibly know.

### Result

| model | scope | items | exact-id recall | ids emitted |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-V4-Pro | contaminable study items | 250 | 0.000 [0.000, 0.015] (n=250) | 4 |
| deepseek-ai/DeepSeek-V4-Pro | CAPEC only | 83 | 0.000 [0.000, 0.044] (n=83) | 2 |
| meta-llama/Llama-4-Maverick | contaminable study items | 250 | 0.000 [0.000, 0.015] (n=250) | 100 |
| meta-llama/Llama-4-Maverick | CAPEC only | 83 | 0.000 [0.000, 0.044] (n=83) | 59 |
| moonshotai/Kimi-K3 | contaminable study items | 250 | 0.000 [0.000, 0.015] (n=250) | 0 |
| moonshotai/Kimi-K3 | CAPEC only | 83 | 0.000 [0.000, 0.044] (n=83) | 0 |
| x-ai/Grok-4.20 | contaminable study items | 250 | 0.000 [0.000, 0.015] (n=250) | 1 |
| x-ai/Grok-4.20 | CAPEC only | 83 | 0.000 [0.000, 0.044] (n=83) | 0 |
| z-ai/GLM-5.3 | contaminable study items | 250 | 0.000 [0.000, 0.015] (n=250) | 0 |
| z-ai/GLM-5.3 | CAPEC only | 83 | 0.000 [0.000, 0.044] (n=83) | 0 |

Chance recall is 1/522 = 0.0019. A rate indistinguishable from chance means the panel could not recite OpenCRE's mapping when asked directly.

### Exposure control: can a judge name a hub it is given the id of?

A judge that emits no hub id above has either never memorised OpenCRE or is simply obeying the instruction not to guess, and the mapping arm alone cannot tell those apart. This asks for the NAME of a hub whose id is supplied, which is a far weaker memory than a mapping and is checkable against `hub_reference.md`. A judge that can name hubs but cannot recall mappings saw the taxonomy and not the links. A judge that can do neither never saw OpenCRE in a form it retained, and contamination is moot for it.

| model | hubs asked | names emitted | names correct |
|---|---|---|---|
| deepseek-ai/DeepSeek-V4-Pro | 50 | 0 | 0 |
| meta-llama/Llama-4-Maverick | 50 | 2 | 0 |
| moonshotai/Kimi-K3 | 50 | 0 | 0 |
| x-ai/Grok-4.20 | 50 | 0 | 0 |
| z-ai/GLM-5.3 | 50 | 0 | 0 |

### What this does and does not establish

It establishes that the models cannot reproduce OpenCRE's mapping on
demand from the control text, and that they cannot name a hub from its
id either. Those two together are hard to reconcile with a memorised
copy of the mapping table, which is the form of contamination that
would directly inflate section 1.

It does not establish that the models never saw OpenCRE. Verbatim recall
is a much stronger property than having read a document once, and a
model can be influenced by training exposure it cannot recite. Nor does
it rule out contamination that runs through the frameworks themselves:
CAPEC and CWE entries are heavily represented on the public web, and a
model's sense of what a CAPEC entry is 'about' is shaped by that
exposure whether or not OpenCRE was in the corpus. That channel would
push the panel toward the same reading of a CAPEC entry the human
reached, and this probe cannot bound it.

The probe also cannot distinguish a model that never saw OpenCRE from
one that saw it and has no addressable memory of six-digit ids, which
are exactly the kind of token string models retain badly. The negative
result is therefore weaker than it looks in one specific way: it is
evidence against id-level memorisation, not against topic-level
familiarity with the taxonomy's shape.

