# Erratum — accuracy figures in this directory are historical

**Every hit@1 figure in these documents is the state of belief on the day the
document was written, not a current result.** They are kept unedited on purpose:
these are design records, and rewriting a superseded plan to make it agree with
later measurements destroys the record of what was actually planned and why.

Two figures in particular recur here and are both **WITHDRAWN**:

- `hit@1 = 0.531, delta = +0.132` — the Phase 1B "Gate 1 CLEAN PASS" headline.
- `hit@1 = 0.537 [0.463, 0.612]` — an earlier variant of the same claim.

`PRD.md` section 6.4 records the withdrawal and its four reasons: the stated
verdict was one of two the code computed and the other failed; the line mixed
two runs; the interval was arithmetic on the point estimate rather than the
bootstrap it claimed to be; and it did not generalize.

**For the current measured result, read `docs/campaign2-results.md`.** It also
lists which claims made during Campaign 2 itself were later superseded, so it is
the one place where the numbers and their caveats travel together.

Do not quote an accuracy figure out of this directory.
