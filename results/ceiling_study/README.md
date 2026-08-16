# Ceiling study: blind expert-agreement review

## What this measures

Whether two qualified annotators agree on which CRE hub a control belongs
to. Nobody has measured this before at a size that means anything -- the
only prior evidence is 13 of 20 hidden calibration items from Phase 3, a
point estimate whose Wilson 95% interval is [0.433, 0.819]. This study
replaces that with 250 items, powered to a Wilson half-width of 0.059.

Validation stratum (125 items): capec 83, cwe 28, nist_800_53 14. Test stratum (125 items): mitre_atlas 43, owasp_ai_exchange 54, nist_ai_100_2 22, owasp_llm_top10 6.

## What to do

1. Open `ceiling_items.json`. Work through the items **in item_index order**.
   Do not skip around -- the order is shuffled so that stopping anywhere is
   still a valid random sample, and working out of order defeats that.
2. For each item, use `hub_reference.md` to find the CRE hub you believe the
   control belongs to. Search by keyword, or browse by branch -- the file is
   organized as an outline (5 top branches, then depth-first by name).
3. Fill in `ceiling_answers_TEMPLATE.json` in place, one entry per item:
   - `primary_hub_id`: your single best hub id. This measures alpha-1
     (agreement at rank 1, the ceiling on hit@1).
   - `acceptable_hub_ids`: up to 5 hub ids
     you would also accept as correct, including the primary one if it
     belongs in the set. This measures alpha-5 (agreement within a
     shortlist, the ceiling on hit@5).
   - `confidence`: "high", "medium", or "low".
   - `notes`: optional, free text.
4. **Do not open `ceiling_answer_key.json`.** It has a warning at the top of
   the file for the same reason.

## Stopping partway is fine

The 250 items are shuffled so that every prefix -- the first 20, the first
83, whatever you actually get through -- is itself a valid stratified
sample across both strata. `scripts/score_ceiling_study.py` scores whatever
fraction of `primary_hub_id` fields are non-empty and reports how many of
250 were completed. There is no requirement to finish before scoring.

## Time budget

Expect roughly one to three minutes per item. A short CAPEC or CWE entry
reads faster than a longer NIST 800-53 control body. Nothing here is timed,
this is only so you can plan the session.

## Scoring

Once you have done as many items as you intend to, from the repository root:

    python -m scripts.score_ceiling_study

Reads your filled-in `ceiling_answers_TEMPLATE.json` against
`ceiling_answer_key.json`, and reports alpha-1 and alpha-5 with Wilson 95%
intervals, pooled and per stratum and per framework, against the 13/20
Phase 3 datum, and states plainly whether the interval is narrow enough to
decide anything.
