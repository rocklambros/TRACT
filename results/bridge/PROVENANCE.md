# Provenance: the Phase 2B bridge set

**Tier 3.** Model-proposed, human-ratified in the model's presence. Not
admissible in a gate denominator at any ratio, per
`results/phase1b/CAMPAIGN3.md` §2.

This file exists because `results/review/PROVENANCE.md` covers
`review_export.json` and nothing covered this directory, while the artifacts
here are published — into `data/processed/cre_hierarchy.json` as
`related_hub_ids`, into the HuggingFace dataset, and onto the model card with a
worked code example teaching consumers to iterate the field.

## What is here

| file | what it is |
|---|---|
| `bridge_candidates.json` | 63 AI-hub → traditional-hub pairs, ranked by cosine similarity, top-3 per AI hub |
| `bridge_report.json` | the 46 a reviewer accepted, 17 rejected, plus similarity statistics and the hub classification counts |

## How they were produced

`tract/bridge/similarity.py` embedded every hub with the TRACT model, took the
top 3 traditional hubs per AI-only hub by cosine similarity
(`"method": "top_k_per_ai_hub", "top_k": 3`), and `tract/bridge/describe.py`
generated an LLM rationale for each pair. One human reviewer then accepted or
rejected each of the 63.

So the reviewer saw a **model-ranked shortlist annotated with model-written
rationales**. `CAMPAIGN3.md` §2 defines Tier 3 as *"Produced by, or ratified in
the presence of, a model or LLM — No. At any ratio."* This is that, on its face.

The separability is measurable and worth recording rather than arguing about:

- accepted mean cosine **0.5502** (min 0.4512); rejected mean **0.4497**
  (max 0.5371)
- a single threshold at **0.4512** — the accepted minimum — reproduces
  **59 of 63** human decisions
- acceptance by presented rank: **19/21, 15/21, 12/21**

That is the same signature `results/review/PROVENANCE.md` quarantined
`review_export.json` for (77.2% accepted exactly as proposed); here the rate is
46/63 = **73.0%**.

## Where they went

- `data/processed/cre_hierarchy.json` v1.1 — **51 hubs carry
  `related_hub_ids`, 92 endpoints, 46 edges. That field is 100% this set.**
  There is no OpenCRE-native content in it and no per-edge provenance marker.
- The published HuggingFace dataset, via `tract/dataset/bundle.py`.
- The published model card, which documents the field and shows how to read it.

## Known scope error

The round ran over **21 AI-only hubs**, the count produced by the superseded
five-framework definition of "AI framework" that also put a false bridge count
on the model card. Under the corrected eight-framework definition there are
**83**. The completed round therefore covers roughly a quarter of the AI region.
The accepted bridges stand on their own terms; their coverage does not.

## What this means for Phase 2C

Phase 2C produces **framework→hub** links by human judgement, deliberately not
hub→hub, and deliberately without a model-ranked shortlist — see
`docs/superpowers/specs/2026-09-01-phase2c-bridge-curation-design.md` §1.2 and
D2. Two consequences follow for whoever runs it:

1. **Nothing in this directory may seed a Phase 2C packet.** A shortlist derived
   from it inherits its tier.
2. **`related_hub_ids` must not appear in any annotator sheet**, and the packet
   builder must be checked against hub-id *values*, not only column names —
   a column called `see_also` carrying bare ids would pass a header check.

## Not resolved here

Whether these 46 edges should remain in a published hierarchy at all is an owner
decision, not a provenance one. This file records what they are so the decision
is made with the tier visible.
