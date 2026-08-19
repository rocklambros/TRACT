### Task 14: Retire both link gates onto the resolved anchor

`assign_quality_tier` drops a link two ways and both test a section title.
`PHASE1B_DROPPED_FRAMEWORKS` names `nist_800_63` and `owasp_proactive_controls`
outright, and `_has_descriptive_text` drops any link whose `section_name` is
shorter than `PHASE1B_MIN_SECTION_TEXT_LENGTH = 10`. Reproduced exactly:
**278 of 4,405 curated links are dropped, 155 by the framework list and 123 by
the short title.** **[measured, orchestrator, `data/training/hub_links_curated.jsonl`]**

Per framework, the 123 short-title drops: capec 44, dsomm 38, cwe 17, enisa 9,
biml 7, iso_27001 2, nist_800_53 2, owasp_ai_exchange 2, etsi 1,
owasp_top10_2021 1. **[measured, orchestrator]**

**Sixty-four of those 123 already resolve to prose in today's corpus.**
**[measured, orchestrator, `ProseIndex.load()` over the 31-framework overlay]**
They are dropped for having a short title while the pipeline holds a paragraph
for them.

**The gate leaks in the other direction too, and no one counted it.** Of the
4,127 links that reach the trainer today, **525 resolve to nothing and train on
their section title**, spread over **251 distinct anchor strings**:
dsomm 176, wstg 118, enisa 59, nist_ssdf 46, etsi 35, samm 30, csa_ccm 29,
owasp_top10_2021 16, biml 14, iso_27001 2. **[measured, orchestrator]** That is
12.7% of the training file training on labels while production serves prose,
against CLAUDE.md's standing rule that title fallback is a last resort. The
count that matters for this task is not 4,127 rising, it is 525 falling.

**Retiring the framework list alone changes nothing for those two
frameworks.** Every one of `nist_800_63`'s 79 `section_name` values is a
section number of 3 to 7 characters and every one of
`owasp_proactive_controls`' 76 is `C1`..`C10`. **[measured, orchestrator]**
Remove the framework list and the short-title gate drops all 155 anyway. Both
must move together, which is why this is one task and why it lands after the
parsers that give those links a resolved anchor.

#### The gate requires a resolved anchor, and does not fall back to the title

The v2 gate was `text = (resolved_text or link.get("section_name", "")).strip()`.
That falls back to `section_name`, the exact field the task's own commit message
says it is moving away from, and twelve links clear the ten-character floor on a
string the model should never see:

| framework | links | anchor they would train on | length |
|---|---|---|---|
| wstg | 9 | `WSTG-BUSL-$$` (3), `WSTG-INPV-00` (3), `WSTG-APPE-D` (2), `WSTG-INFO-##` (1) | 11-12 chars |
| iso_27001 | 2 | `Security of assets off-premises`, `Equipment siting and protection` | 31 chars each |
| dsomm | 1 | the activity name for the one activity whose statement is 11 characters, which `_is_prose` refuses to index | long |

**[measured, orchestrator]** `section_name == section_id` for all 118 wstg rows,
and the four bogus ids appear in no WSTG archive file (Contract Rule 0: the
parser reaches 109 of 118). Those nine links would pair a literal control id
with a real CRE hub and call it training data.

The gate therefore requires `resolved_text is not None`. The cost is stated
rather than hidden: two real ISO 27001 links whose `section_name` is a genuine
descriptive title get dropped because the link's id and name match no parsed
control. Keeping them would mean shipping title anchors that nothing downstream
can distinguish from prose anchors, which is the defect this task exists to
close. Two links out of 4,405 is the price.

#### Derived outcome

| quantity | before | after | source |
|---|---|---|---|
| curated links | 4,405 | 4,405 | **[measured]** |
| training links | 4,127 | **4,389** | **[derived]**, see arithmetic |
| anchors that are section titles | 525 | **0** | **[derived]**, by construction of the gate |
| distinct title-fallback anchor strings | 251 | 0 | **[derived]** |

Arithmetic, every term measured or taken from a `JOIN_FLOORS` entry committed in
Task 16 before any parser existed:

```
4,127  training links today                                    [measured]
  +154  framework deny list retired: proactive 76 + nist_800_63 78 of 79
  +120  short-title drops that Tasks 3-13 give a resolved anchor
    -9  wstg links whose id is absent from the archive          [measured]
    -2  iso_27001 links that resolve to no parsed control       [measured]
    -1  dsomm link whose control statement is 11 characters     [JOIN_FLOORS]
=4,389                                                          [derived]
```

The sixteen links that stay dropped, named, so the acceptance test asserts
identity rather than a count:

| framework | links | reason | source |
|---|---|---|---|
| wstg | 9 | ids absent from the archive | `JOIN_FLOORS["wstg"] = 0.92` |
| nist_800_53 | 2 | `SC-23(1)`, `SC-23(3)` match no parsed control | **[measured]** |
| iso_27001 | 2 | `7.8`, `7.9` match no parsed control | **[measured]** |
| dsomm | 1 | statement is 11 characters, `_is_prose` skips it | `JOIN_FLOORS["dsomm"] = 0.99` |
| nist_800_63 | 1 | `section_id == section_name == "are g"`, a corrupt OpenCRE row | **[measured]** |
| cwe | 1 | `937` matches no parsed control | **[measured]** |

**The v2 plan's `4,402 of 4,405` was wrong twice.** Under its own
fallback-to-title gate the answer is **4,401**, because a fourth link falls under
the floor (`nist_800_63` with `section_name == "are g"`, 5 characters) and the
plan enumerated only three. Under the gate this task ships, the answer is
**4,389**, and the 12-link difference between 4,401 and 4,389 is precisely the
set that would have trained on a title. The wrong number was hard-coded into a
commit message and copied into the run ledger with no test behind it. Step 8 adds
the test.

#### The count depends on whether this checkout holds the licensed overlay

`ProseIndex.load()` calls `merged_corpus_path()`, which returns the gitignored
overlay when it exists and the tracked corpus otherwise. Measured against both
files today: the overlay resolves **3,666** of 4,405 curated links, the tracked
corpus resolves **3,574**. **[measured, orchestrator]** The whole 92-link gap is
`iso_27001`, whose text is licensed and is not in git.

That makes 4,389 reachable only where the overlay is present. Where it is not, the
expected count falls by every link belonging to a framework the corpus does not
carry. Under Contract Rule 3's `OVERLAY_FRAMEWORK_IDS` that is nine frameworks
holding 635 links, of which 623 would otherwise resolve, giving **3,766**.
**[derived]** The test computes the expectation from the corpus it read
rather than hard-coding either literal, so it asserts in both environments and
skips in neither.

Worse, the mechanism that was supposed to make the two runs distinguishable does
not work. `merged_corpus_path`'s docstring states "the fold metadata records the
corpus sha256". It does not: `tract/training/orchestrate.py:347` hashes
`PROCESSED_DIR / "all_controls.json"` while `ProseIndex.load()` at line 183 reads
`merged_corpus_path()`. **[measured, orchestrator]** Two runs 92 links apart
record the same digest. Step 6 fixes it.

#### The training file becomes a function of the corpus, and must say so

After this task, `filter_training_links` resolves every link through
`ProseIndex.load()`, so `hub_links_training.jsonl` depends on
`merged_corpus_path()`. Today `save_training_links(links, raw_hash)` records only
the curated-links hash, so two runs over different corpora produce the same
`raw_hash`. Task 15 then rewrites the corpus.

The v2 self-review claimed "Task 14 precedes Task 15" discharged ledger lesson 6.
It does the opposite: ordering a task before the thing that invalidates its output
is the lesson, not the remedy. Two fixes, neither of which depends on task
numbering:

1. `save_training_links` takes `corpus_sha256` as a required positional argument
   and writes `data/training/hub_links_training.meta.json` beside the JSONL.
2. `tests/test_data_quality.py` asserts the sidecar's `corpus_sha256` equals the
   digest of the corpus on disk. Any task that rewrites the corpus without
   regenerating the training file turns that test red. Task 15 Step 10
   regenerates it.

#### CAPEC and CWE: a lever, stated rather than assumed

This change restores every contested link in both frameworks. CAPEC training links
move 1,755 to 1,799 (all 44 recovered), CWE moves 596 to 612 (16 of 17 recovered,
`937` resolves to nothing). **[measured, orchestrator]** The recovered links are
the terse ones: `UDP Ping`, `Fuzzing`, `Pharming`, `HTTP DoS`, `XML Flood`.

The v2 self-review stated "CAPEC and CWE are untouched and remain 57.3% of the
training graph. Nothing here improves that." Both halves are wrong. They are
touched, and their combined share falls from **56.97%** (2,351 of 4,127) to
**54.94%** (2,411 of 4,389) because the eleven frameworks add more than CAPEC and
CWE do. **[derived from measured counts]**

The reason to make it a lever rather than a default: the human ceiling study
measured CAPEC's agreement with OpenCRE at **alpha-1 = 0.181 [0.113, 0.277] on
n=83**. **[measured, `results/ceiling_study/panel_agreement.md:8,77`]** A domain
expert and OpenCRE's curators pick the same best hub fewer than one time in five
on that framework. Recovering its shortest-labelled links is not self-evidently
progress. Ten new CAPEC items and six new CWE items also enter the validation
roster (1,244 to 1,264) **[measured, premortem Data Scientist and Governance]**,
drawn from the least-agreed stratum, after the ceiling was measured on a roster
without them.

`filter_training_links` therefore takes `recover_contested: bool = True`. The
default ships the recovery, the flag gives the later training-mix decision a lever
that is not entangled with the eleven frameworks' 274 legitimate recoveries, and
both values are asserted, so neither branch is dead code:

| `recover_contested` | training links | capec | cwe | capec+cwe share |
|---|---|---|---|---|
| `True` (default) | 4,389 | 1,799 | 612 | 54.94% |
| `False` | 4,329 | 1,755 | 596 | 54.26% |

**[derived from measured counts]**

#### The ceiling study stops mirroring training the moment this lands

`tract/ceiling_study.py:132` calls `assign_quality_tier(record)` with one argument,
and line 119 documents why: "Mirrors tract.training.data_quality.load_and_filter_curated_links
exactly (same assign_quality_tier call) so the study pool is the pool training
would use, not an approximation of it." Line 144 makes a second one-argument call.
**[measured, orchestrator]**

Give `resolved_text` a default and both calls keep compiling, the study pool keeps
the section-title gate, training moves to the anchor gate, nothing raises, and the
docstring becomes false. `resolved_text` therefore has **no default**, so both call
sites fail at import under `mypy --strict` and at runtime under pytest. Step 7
rewrites them to call the same function training calls.

**Files:**
- Modify: `tract/config.py`
- Modify: `tract/training/data_quality.py`
- Modify: `tract/text_selection.py`
- Modify: `tract/training/orchestrate.py`
- Modify: `tract/ceiling_study.py`
- Modify: `tests/test_data_quality.py`
- Modify: `tests/test_ceiling_study.py`
- Create: `data/training/hub_links_training.meta.json`

**Interfaces:**
- Consumes: `tract.text_selection.ProseIndex`, `merged_corpus_path`; the parsers from Tasks 3-13.
- Produces: `assign_quality_tier(link: dict[str, str], resolved_text: str | None) -> QualityTier` (no default on the second parameter); `FilterReport`; `filter_training_links(links, index, *, recover_contested: bool = True) -> FilterReport`; `curated_link_filter_report(path=None, index=None, *, recover_contested=True) -> tuple[FilterReport, str]`; `save_training_links(links, raw_hash, corpus_sha256, path=None) -> str`; `tract.text_selection.merged_corpus_sha256() -> str`; `PHASE1B_MIN_ANCHOR_TEXT_LENGTH` and `CONTESTED_RECOVERY_FRAMEWORK_IDS` replacing `PHASE1B_DROPPED_FRAMEWORKS` and `PHASE1B_MIN_SECTION_TEXT_LENGTH`.

**Invalidates:**
- `data/training/hub_links_training.jsonl` and every artifact derived from it: `results/phase1b/**/fold_result.json`, every `data_hash` and `curated_links_sha256` recorded before this commit, and the published `hit@1 = 0.531` whose training file this replaces.
- `tract/ceiling_study.py`'s anchor pool, and therefore the sampling frame of `results/ceiling_study/ceiling_items.json`. The 250 drawn items survive: zero of them fall in the eleven frameworks and CAPEC and CWE reproduce byte-identically **[measured, adjudication C-A and C-B]**. The frame they were drawn from gains 60 anchors, so any future sample is not comparable to the 250 without saying so.
- `tract/training/orchestrate.py`'s `inputs.all_controls_sha256` in every fold record written before Step 6, which named a file the run did not read.
- Itself, by Task 15. Task 15 Step 10 regenerates `hub_links_training.jsonl` and its sidecar, and the sidecar test in Step 8 fails until it does.

- [ ] **Step 1: Record the before state, with the column the v2 plan did not have**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
from tract.text_selection import ProseIndex, merged_corpus_path
from tract.training.data_quality import QualityTier, assign_quality_tier

index = ProseIndex.load()
links = [json.loads(l) for l in
         open("data/training/hub_links_curated.jsonl", encoding="utf-8") if l.strip()]
kept = [l for l in links if assign_quality_tier(l) is not QualityTier.DROPPED]


def resolves(link: dict[str, str]) -> bool:
    return index.lookup(link.get("standard_name", ""),
                        link.get("section_id"), link.get("section_name")) is not None


fallback = [l for l in kept if not resolves(l)]
print("corpus read:", merged_corpus_path())
print("curated links:", len(links))
print("training links before:", len(kept))
print("of which train on a section title:", len(fallback))
print("distinct title anchors:", len({
    (l["framework_id"], (l.get("section_name") or l.get("section_id") or "").strip().lower())
    for l in fallback
}))
PYEOF
```

Expected, with the overlay present:

```
curated links: 4405
training links before: 4127
of which train on a section title: 525
distinct title anchors: 251
```

**[measured, orchestrator]** Write all four numbers into the run ledger. Step 9
compares against them. If `training links before` is not 4,127 the curated file
changed since the premortem and nothing below is derived from the right base.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_data_quality.py — append

import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestGatesTestTheAnchorNotTheTitle:
    """Both drops used to test section_name, which the model never sees."""

    LONG = "A control statement long enough to be worth training on, twice. " * 3

    def _index(self) -> "ProseIndex":
        from tract.text_selection import ProseIndex

        return ProseIndex([{
            "framework_name": "OWASP Proactive Controls",
            "controls": [
                {"control_id": "C6", "title": "Use Secure Dependencies",
                 "description": self.LONG},
            ],
        }])

    def test_a_short_title_with_a_resolved_anchor_is_kept(self) -> None:
        from tract.training.data_quality import QualityTier, assign_quality_tier

        link = {
            "framework_id": "owasp_proactive_controls",
            "standard_name": "OWASP Proactive Controls",
            "section_id": "C6", "section_name": "C6",
            "link_type": "LinkedTo",
        }
        assert assign_quality_tier(link, self.LONG) is QualityTier.T1

    def test_an_unresolved_link_is_dropped_however_long_its_title(self) -> None:
        """The nine wstg links this closes carry 11 and 12 character ids.

        Falling back to section_name would train "WSTG-BUSL-$$" against a real
        CRE hub, because section_name == section_id for all 118 wstg rows and
        the four bogus ids clear the ten-character floor. [measured]
        """
        from tract.training.data_quality import QualityTier, assign_quality_tier

        for name in ("WSTG-BUSL-$$", "WSTG-INPV-00",
                     "Security of assets off-premises"):
            link = {
                "framework_id": "wstg", "standard_name": "OWASP WSTG",
                "section_id": name, "section_name": name,
                "link_type": "LinkedTo",
            }
            assert assign_quality_tier(link, None) is QualityTier.DROPPED, name

    def test_a_resolved_but_thin_anchor_is_dropped(self) -> None:
        from tract.training.data_quality import QualityTier, assign_quality_tier

        link = {
            "framework_id": "dsomm", "standard_name": "DSOMM",
            "section_id": "x", "section_name": "a long activity name here",
            "link_type": "AutomaticallyLinkedTo",
        }
        assert assign_quality_tier(link, "Do backups") is QualityTier.DROPPED

    def test_the_anchor_parameter_has_no_default(self) -> None:
        """A defaulted second parameter is how the ceiling study broke silently.

        tract/ceiling_study.py called assign_quality_tier(record) with one
        argument under a docstring promising it mirrored training. Give
        resolved_text a default and that call keeps compiling while the two
        pools diverge, and nothing raises.
        """
        from tract.training.data_quality import assign_quality_tier

        parameter = inspect.signature(assign_quality_tier).parameters["resolved_text"]
        assert parameter.default is inspect.Parameter.empty

    def test_the_framework_deny_list_is_gone(self) -> None:
        import tract.config as config

        assert not hasattr(config, "PHASE1B_DROPPED_FRAMEWORKS")
        assert not hasattr(config, "PHASE1B_MIN_SECTION_TEXT_LENGTH")

    def test_filter_reports_each_drop_reason_separately(self) -> None:
        from tract.training.data_quality import filter_training_links

        links = [
            {"framework_id": "owasp_proactive_controls",
             "standard_name": "OWASP Proactive Controls",
             "section_id": "C6", "section_name": "C6",
             "cre_id": "1", "link_type": "LinkedTo"},
            {"framework_id": "owasp_proactive_controls",
             "standard_name": "OWASP Proactive Controls",
             "section_id": "C9", "section_name": "C9",
             "cre_id": "2", "link_type": "LinkedTo"},
        ]
        report = filter_training_links(links, self._index())
        assert len(report.kept) == 1
        assert len(report.dropped_unresolved) == 1
        assert report.dropped_thin_anchor == []

    def test_contested_recovery_is_a_lever_with_both_values_live(self) -> None:
        """capec alpha-1 is 0.181, so restoring its terse links is a choice.

        [measured, results/ceiling_study/panel_agreement.md]
        """
        from tract.text_selection import ProseIndex
        from tract.training.data_quality import filter_training_links

        index = ProseIndex([{
            "framework_name": "CAPEC",
            "controls": [{"control_id": "125", "title": "Flooding",
                          "description": self.LONG}],
        }])
        link = {"framework_id": "capec", "standard_name": "CAPEC",
                "section_id": "125", "section_name": "Flooding",
                "cre_id": "1", "link_type": "LinkedTo"}
        assert len(filter_training_links([link], index).kept) == 1
        off = filter_training_links([link], index, recover_contested=False)
        assert off.kept == []
        assert len(off.dropped_contested) == 1
```

```python
# tests/test_data_quality.py — append, the staleness guard

class TestTrainingFileRecordsTheCorpusItRead:
    """hub_links_training.jsonl is a function of the corpus after this task.

    save_training_links previously recorded only the curated-links hash, so two
    runs over corpora 92 iso_27001 links apart produced the same raw_hash
    [measured]. Task 15 rewrites the corpus. This test is what makes that
    ordering enforceable rather than a claim in a self-review.
    """

    def test_the_sidecar_names_the_corpus_on_disk(self) -> None:
        import json

        from tract.text_selection import merged_corpus_path, merged_corpus_sha256
        from tract.training.data_quality import TRAINING_META_PATH

        meta = json.loads(TRAINING_META_PATH.read_text(encoding="utf-8"))
        assert meta["corpus_sha256"] == merged_corpus_sha256(), (
            "hub_links_training.jsonl was built against a different corpus than "
            f"{merged_corpus_path()}. Regenerate it before trusting any metric "
            "derived from it."
        )
        assert meta["n_links"] == sum(
            1 for line in
            (TRAINING_META_PATH.parent / "hub_links_training.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        )

    def test_save_requires_the_corpus_digest(self) -> None:
        from tract.training.data_quality import save_training_links

        parameter = inspect.signature(save_training_links).parameters["corpus_sha256"]
        assert parameter.default is inspect.Parameter.empty
```

The three existing classes in `tests/test_data_quality.py` call
`assign_quality_tier(link)` with one argument at lines 28, 40, 52, 64, 76, 88 and
107, and `filter_training_links(links)` at 134, 151 and 185. **[measured]** Update
all ten call sites in this step. Do not add a default to make them pass, which is
the defect the signature test exists to catch.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_data_quality.py -q
```

Expected: FAIL. `assign_quality_tier` takes one argument,
`PHASE1B_DROPPED_FRAMEWORKS` still exists, `filter_training_links` returns a list
rather than a `FilterReport`, and `TRAINING_META_PATH` does not exist.

- [ ] **Step 4: Change the config**

```python
# tract/config.py — replace the PHASE1B_DROPPED_FRAMEWORKS block

# A link is worth training on when the text the model sees is
# substantial. Both of the gates this replaces tested link["section_name"], a
# title the model never sees: a framework deny list naming nist_800_63 and
# owasp_proactive_controls, and a 10-character floor on the same field. Between
# them they dropped 278 of 4,405 curated links, 64 of which already had a
# resolved paragraph in the corpus, while letting 525 links through to train on
# a title. [measured]
#
# The threshold is unchanged at 10 characters. Only the field it is applied to
# moved, from the title to the anchor the encoder is handed.
PHASE1B_MIN_ANCHOR_TEXT_LENGTH: Final[int] = 10

# Frameworks whose recovered links are a decision rather than a repair. The
# anchor gate restores 44 capec and 16 cwe links that the title floor dropped,
# and those are the terse ones ("UDP Ping", "Fuzzing", "Pharming"). The human
# ceiling study measured capec agreement with OpenCRE at alpha-1 = 0.181
# [0.113, 0.277] on n=83 [measured, results/ceiling_study/panel_agreement.md],
# so recovering its least-agreed stratum is not self-evidently progress. The
# default recovers them; filter_training_links(recover_contested=False) is the
# lever the later training-mix decision needs, and it is not entangled with the
# eleven frameworks' 274 recoveries.
CONTESTED_RECOVERY_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "capec", "cwe",
})
```

Delete `PHASE1B_DROPPED_FRAMEWORKS` and `PHASE1B_MIN_SECTION_TEXT_LENGTH`
entirely. A constant left in place with no reader is the decorative-control
defect from ledger lesson 4.

- [ ] **Step 5: Add the corpus digest to `text_selection`**

```python
# tract/text_selection.py — append after merged_corpus_path

def merged_corpus_sha256(path: Path | None = None) -> str:
    """The digest of the corpus a run read.

    merged_corpus_path's docstring already claimed a run that used the overlay
    and a run that did not were distinguishable because "the fold metadata
    records the corpus sha256". They were not. orchestrate.py hashed
    PROCESSED_DIR / "all_controls.json" while ProseIndex.load() read
    merged_corpus_path(), so two runs 92 iso_27001 links apart recorded the
    same digest for two different corpora. [measured]
    """
    import hashlib

    source = path or merged_corpus_path()
    return hashlib.sha256(source.read_bytes()).hexdigest()
```

- [ ] **Step 6: Change the filter, and make the training file name its corpus**

```python
# tract/training/data_quality.py — replace the imports and the two functions

from tract.config import (
    CONTESTED_RECOVERY_FRAMEWORK_IDS,
    PHASE1B_MIN_ANCHOR_TEXT_LENGTH,
    TRAINING_DIR,
)
from tract.io import atomic_write_json
from tract.text_selection import ProseIndex, merged_corpus_path, merged_corpus_sha256

TRAINING_META_PATH: Final[Path] = TRAINING_DIR / "hub_links_training.meta.json"


def link_key(link: dict[str, str]) -> str:
    """Stable identity for one curated link, so a drop can be named.

    A count tells an operator that something moved. Only a name tells them
    whether the thing that moved is the thing they expected.
    """
    return "|".join((
        link.get("framework_id", ""),
        link.get("section_id", ""),
        link.get("section_name", ""),
        link.get("cre_id", ""),
    ))


@dataclass(frozen=True)
class FilterReport:
    """What the anchor gate kept, and why it dropped everything else.

    Three drop reasons, reported apart, because they call for three different
    responses: an unresolved link means a parser or a join is missing, a thin
    anchor means the source is that terse, and a contested drop is a
    deliberate exclusion this run chose.
    """

    kept: list[TieredLink]
    dropped_unresolved: list[str]
    dropped_thin_anchor: list[str]
    dropped_contested: list[str]
    corpus_path: str
    corpus_sha256: str

    @property
    def n_dropped(self) -> int:
        return (
            len(self.dropped_unresolved)
            + len(self.dropped_thin_anchor)
            + len(self.dropped_contested)
        )


def _is_contested_recovery(link: dict[str, str]) -> bool:
    """True for a link this change newly admits from capec or cwe.

    Exactly the links the retired title floor dropped: 44 capec and 17 cwe.
    [measured]
    """
    return (
        link.get("framework_id", "") in CONTESTED_RECOVERY_FRAMEWORK_IDS
        and len(link.get("section_name", "").strip()) < PHASE1B_MIN_ANCHOR_TEXT_LENGTH
    )


def assign_quality_tier(
    link: dict[str, str], resolved_text: str | None,
) -> QualityTier:
    """Assign a quality tier to a single hub link.

    `resolved_text` is the anchor the encoder will be handed for this link,
    from ProseIndex, or None when the link resolves to no parsed control. It
    has no default on purpose. tract/ceiling_study.py calls this function under
    a docstring promising it mirrors training, and a defaulted parameter would
    let that call keep compiling while the two pools silently diverged.

    A link with no resolved anchor is dropped rather than falling back to
    link["section_name"]. That fallback is the field this change exists to stop
    training on, and twelve links clear the ten-character floor on it: nine
    wstg ids absent from the archive, two iso_27001 titles, and one dsomm
    activity whose statement _is_prose refuses to index. [measured]
    """
    if resolved_text is None:
        return QualityTier.DROPPED

    if len(resolved_text.strip()) < PHASE1B_MIN_ANCHOR_TEXT_LENGTH:
        return QualityTier.DROPPED

    if link.get("standard_name", "") in AI_FRAMEWORK_NAMES:
        return QualityTier.T1_AI

    if link.get("link_type", "") == "AutomaticallyLinkedTo":
        return QualityTier.T3

    return QualityTier.T1


def filter_training_links(
    links: list[dict[str, str]],
    index: ProseIndex,
    *,
    recover_contested: bool = True,
) -> FilterReport:
    """Filter links by the resolved anchor and assign tier metadata."""
    kept: list[TieredLink] = []
    unresolved: list[str] = []
    thin: list[str] = []
    contested: list[str] = []
    tier_counts: dict[QualityTier, int] = {t: 0 for t in QualityTier}

    for link in links:
        if not recover_contested and _is_contested_recovery(link):
            contested.append(link_key(link))
            continue

        selection = index.lookup(
            link.get("standard_name", ""), link.get("section_id"),
            link.get("section_name"),
        )
        text = selection.text if selection else None
        tier = assign_quality_tier(link, text)
        tier_counts[tier] += 1
        if tier is not QualityTier.DROPPED:
            kept.append(TieredLink(link=link, tier=tier))
        elif text is None:
            unresolved.append(link_key(link))
        else:
            thin.append(link_key(link))

    for tier, count in tier_counts.items():
        logger.info("Quality tier %s: %d links", tier.value, count)
    logger.info(
        "Dropped %d unresolved, %d thin anchors, %d contested",
        len(unresolved), len(thin), len(contested),
    )

    return FilterReport(
        kept=kept,
        dropped_unresolved=sorted(unresolved),
        dropped_thin_anchor=sorted(thin),
        dropped_contested=sorted(contested),
        corpus_path=str(merged_corpus_path()),
        corpus_sha256=merged_corpus_sha256(),
    )


def curated_link_filter_report(
    path: Path | None = None,
    index: ProseIndex | None = None,
    *,
    recover_contested: bool = True,
) -> tuple[FilterReport, str]:
    """Load the curated links and run the anchor gate over them.

    The single implementation of the gate. tract/ceiling_study.py calls this
    rather than repeating the tier call beside its own loop, which is how the
    two pools stopped agreeing.

    Returns:
        (report, sha256 of the raw curated records).
    """
    p = path or CURATED_PATH
    raw_links: list[dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_links.append(json.loads(line))

    raw_hash = compute_data_hash(raw_links)
    logger.info("Loaded %d curated links (hash=%s)", len(raw_links), raw_hash[:16])

    report = filter_training_links(
        raw_links, index or ProseIndex.load(), recover_contested=recover_contested,
    )
    logger.info(
        "After the anchor gate: %d usable links (dropped %d) against %s",
        len(report.kept), report.n_dropped, report.corpus_path,
    )
    return report, raw_hash


def load_and_filter_curated_links(
    path: Path | None = None,
) -> tuple[list[TieredLink], str]:
    """Load curated links, filter by the resolved anchor, return with data hash.

    Kept at its original arity so the six existing callers in
    tract/training/orchestrate.py, scripts/phase1b/run_fold.py and
    scripts/phase1c/ do not change. Callers that need the drop reasons call
    curated_link_filter_report directly.
    """
    report, raw_hash = curated_link_filter_report(path)
    return report.kept, raw_hash
```

Delete `_has_descriptive_text`.

```python
# tract/training/data_quality.py — replace save_training_links' signature and tail

def save_training_links(
    links: list[TieredLink],
    raw_hash: str,
    corpus_sha256: str,
    path: Path | None = None,
) -> str:
    """Save filtered training links to JSONL, and record what produced them.

    corpus_sha256 has no default. After the anchor gate, this file is a
    function of the corpus as well as of the curated links, and recording only
    raw_hash made two runs over corpora 92 links apart indistinguishable.
    """
```

At the end of that function, after `os.replace(tmp, p)`:

```python
    atomic_write_json(
        {
            "corpus_path": str(merged_corpus_path()),
            "corpus_sha256": corpus_sha256,
            "curated_links_sha256": raw_hash,
            "n_links": len(output_records),
            "output_sha256": output_hash,
        },
        TRAINING_META_PATH,
    )
```

```python
# tract/training/orchestrate.py — line 347, hash the corpus the run read

        "all_controls_sha256": (
            merged_corpus_sha256() if prose_index is not None else None
        ),
```

Import `merged_corpus_sha256` from `tract.text_selection` in that module.

- [ ] **Step 7: Make the ceiling study call the same gate**

```python
# tract/ceiling_study.py — replace _load_eligible_links and _link_priority

def _load_eligible_links(allowed_framework_ids: frozenset[str]) -> list[dict[str, str]]:
    """Curated links, quality-filtered, restricted to the eligible frameworks.

    Calls tract.training.data_quality.curated_link_filter_report, the function
    training calls, rather than repeating the gate beside it. The previous
    version inlined an assign_quality_tier(record) call under a docstring
    claiming it mirrored training. When that function gained a resolved-anchor
    argument, the copy here would have kept compiling against the old contract
    and quietly stopped mirroring: the study pool would keep the section-title
    gate while training moved to the anchor gate, and nothing would raise.
    """
    report, _ = curated_link_filter_report()
    return [
        tiered.link for tiered in report.kept
        if tiered.link.get("framework_id") in allowed_framework_ids
    ]


def _link_priority(
    link: dict[str, str], prose_index: ProseIndex,
) -> tuple[int, str, str]:
    """Sort key preferring higher-quality tiers, then section id, then name.

    Takes the index because assign_quality_tier now needs the anchor. The one
    caller, build_anchor_pool, already holds it.
    """
    selection = prose_index.lookup(
        link.get("standard_name", ""), link.get("section_id"),
        link.get("section_name"),
    )
    tier = assign_quality_tier(
        link, selection.text if selection else None,
    ).value
    return (
        _TIER_PRIORITY.get(tier, 99),
        link.get("section_id", ""),
        link.get("section_name", ""),
    )
```

```python
# tract/ceiling_study.py — line 187, inside build_anchor_pool
        representative = min(members, key=lambda m: _link_priority(m, prose_index))
```

Change the import on line 43 to add `curated_link_filter_report`.

```python
# tests/test_ceiling_study.py — append

class TestTheStudyPoolIsTheTrainingPool:
    """The mirror the docstring promises, asserted rather than described."""

    def test_the_two_pools_hold_the_same_links(self) -> None:
        from tract.ceiling_study import _load_eligible_links, eligible_framework_ids
        from tract.training.data_quality import curated_link_filter_report, link_key

        eligible = eligible_framework_ids()
        report, _ = curated_link_filter_report()
        training = {
            link_key(t.link) for t in report.kept
            if t.link.get("framework_id") in eligible
        }
        study = {link_key(l) for l in _load_eligible_links(eligible)}
        assert study == training

    def test_no_anchor_in_the_pool_is_a_section_title(self) -> None:
        """The gate admits only resolved links, so the pool is prose throughout.

        build_anchor_pool calls select_control_text, which falls back to the
        title when the index misses. Before the anchor gate, 525 of the 4,127
        training links resolved to nothing [measured], and any of them landing
        in an eligible framework put a title into the pool that a reviewer
        would have scored as a control statement.
        """
        from tract.ceiling_study import (
            _load_eligible_links, build_anchor_pool, eligible_framework_ids,
        )
        from tract.text_selection import ProseIndex

        index = ProseIndex.load()
        pool = build_anchor_pool(_load_eligible_links(eligible_framework_ids()), index)
        titles = [
            record.anchor_key for records in pool.values() for record in records
            if record.text_source == "title"
        ]
        assert titles == []
```

- [ ] **Step 8: Add the acceptance test for the count itself**

The v2 plan compared 4,127 against 4,402 with a `print()` that an agent read.
No test asserted either number, and the wrong one reached a commit message and
the run ledger. This test computes its expectation from the corpus it read, so
it asserts with the overlay and without it, and skips in neither.

```python
# tests/test_data_quality.py — append

class TestTheAnchorGateReachesItsDerivedCount:
    """4,389 of 4,405, and the sixteen exceptions named rather than counted."""

    # Every link the gate is expected to drop after Tasks 3-13, keyed
    # (framework_id, section_id). Nine wstg and one dsomm come from the
    # JOIN_FLOORS entries committed in Task 16 before any parser existed. The
    # other six were measured against the corpus at 8cf44b3. [measured]
    EXPECTED_UNRESOLVED: frozenset[tuple[str, str]] = frozenset({
        ("wstg", "WSTG-BUSL-$$"), ("wstg", "WSTG-INPV-00"),
        ("wstg", "WSTG-APPE-D"), ("wstg", "WSTG-INFO-##"),
        ("nist_800_53", "SC-23(1)"), ("nist_800_53", "SC-23(3)"),
        ("iso_27001", "7.8"), ("iso_27001", "7.9"),
        ("nist_800_63", "are g"), ("cwe", "937"),
    })
    EXPECTED_KEPT_FULL_CORPUS = 4389   # [derived] 4,405 - 16 unresolved links
    CONTESTED_RECOVERED = 60           # capec 44 + cwe 16 [measured]

    def _report(self, **kwargs: object):  # type: ignore[no-untyped-def]
        from tract.training.data_quality import curated_link_filter_report

        report, _ = curated_link_filter_report(**kwargs)  # type: ignore[arg-type]
        return report

    def test_every_drop_is_one_this_plan_predicted(self) -> None:
        """Fails in both directions: an unexpected drop, or an unexpected keep."""
        report = self._report()
        surprises = sorted(
            key for key in report.dropped_unresolved
            if (key.split("|")[0], key.split("|")[1]) not in self.EXPECTED_UNRESOLVED
        )
        assert surprises == [], (
            "these links resolve to no parsed control and this plan did not "
            f"predict that: {surprises[:20]}"
        )
        assert report.dropped_thin_anchor == [], (
            "a control resolved to fewer than ten characters of text. No "
            "parser in Tasks 3-13 was expected to emit one, so this is a "
            f"parser defect, not a source limit: {report.dropped_thin_anchor}"
        )

    def test_the_count_matches_the_corpus_that_was_read(self) -> None:
        """4,389 needs all 31 frameworks. Derive, never hard-code one literal.

        merged_corpus_path returns the gitignored overlay when it exists and
        the tracked corpus otherwise, and the tracked corpus always exists, so
        an existence check never skips. Measured: the overlay resolves 3,666 of
        4,405 curated links and the tracked file resolves 3,574, a 92-link gap
        that is entirely iso_27001. [measured]
        """
        import json

        from tract.text_selection import ProseIndex, merged_corpus_path

        report = self._report()
        data = json.loads(merged_corpus_path().read_text(encoding="utf-8"))
        present = {
            record.get("framework_id") for record in data["frameworks"]
        }
        absent_drops = [
            key for key in report.dropped_unresolved
            if key.split("|")[0] not in present
        ]
        expected = self.EXPECTED_KEPT_FULL_CORPUS - len(absent_drops)
        assert len(report.kept) == expected, (
            f"{len(report.kept)} kept against {expected} expected for a corpus "
            f"of {len(present)} frameworks reading {merged_corpus_path()}"
        )

    def test_no_kept_link_trains_on_a_section_title(self) -> None:
        """525 links did before this change, on 251 distinct strings. [measured]"""
        from tract.text_selection import ProseIndex

        index = ProseIndex.load()
        report = self._report()
        titles = [
            t.link for t in report.kept
            if index.lookup(t.link.get("standard_name", ""),
                            t.link.get("section_id"),
                            t.link.get("section_name")) is None
        ]
        assert titles == []

    def test_the_contested_lever_moves_exactly_the_contested_links(self) -> None:
        full = self._report()
        without = self._report(recover_contested=False)
        assert len(full.kept) - len(without.kept) == self.CONTESTED_RECOVERED
        assert {key.split("|")[0] for key in without.dropped_contested} == {
            "capec", "cwe",
        }
```

- [ ] **Step 9: Run everything and record the after state**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_data_quality.py tests/test_ceiling_study.py -q
"$PY" -m mypy tract/training/data_quality.py tract/config.py \
      tract/text_selection.py tract/ceiling_study.py \
      tract/training/orchestrate.py --strict
grep -rn "PHASE1B_DROPPED_FRAMEWORKS\|PHASE1B_MIN_SECTION_TEXT_LENGTH" \
     tract/ parsers/ scripts/ tests/ || echo "no readers left"
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.training.data_quality import curated_link_filter_report
report, _ = curated_link_filter_report()
print("corpus:", report.corpus_path)
print("training links after:", len(report.kept))
print("dropped unresolved:", len(report.dropped_unresolved),
      report.dropped_unresolved)
print("dropped thin anchor:", len(report.dropped_thin_anchor))
PYEOF
```

Expected with the overlay present: `training links after: 4389`, sixteen
unresolved drops matching the named set, zero thin anchors. **[derived]**

Reading 4,401 means the fallback to `section_name` survived somewhere and twelve
links are training on a title. Reading 4,127 means the index is not reaching the
filter. Reading 3,766 means this checkout has no overlay, which is a legitimate
state that the test in Step 8 accepts and this print does not explain on its own.

- [ ] **Step 10: Regenerate the training file and commit, split by decision**

Two commits. The second is revertable on its own, so the later training-mix
decision has a lever that does not disturb the eleven frameworks' recoveries.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.text_selection import merged_corpus_sha256
from tract.training.data_quality import curated_link_filter_report, save_training_links

report, raw_hash = curated_link_filter_report(recover_contested=False)
print("wrote", len(report.kept), "links, output hash",
      save_training_links(report.kept, raw_hash, merged_corpus_sha256())[:16])
PYEOF
git add tract/config.py tract/training/data_quality.py tract/text_selection.py \
        tract/training/orchestrate.py tract/ceiling_study.py \
        tests/test_data_quality.py tests/test_ceiling_study.py \
        data/training/hub_links_training.jsonl \
        data/training/hub_links_training.meta.json
git commit -m "fix: drop a link on the text the model sees, not on its section title

Training links move from 4,127 to 4,329 with the contested capec and cwe
recoveries held back for the next commit. The 154 dropped by the framework deny
list and 120 of the 123 dropped by the short-title floor now resolve to parsed
prose, and the 525 links that used to train on a section title fall to zero. The
sixteen that stay dropped resolve to no parsed control: nine wstg ids absent from
the archive, two nist_800_53, two iso_27001, one dsomm, one nist_800_63 and one
cwe. The training file now records the sha256 of the corpus it was built from."
```

```bash
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.text_selection import merged_corpus_sha256
from tract.training.data_quality import curated_link_filter_report, save_training_links

report, raw_hash = curated_link_filter_report()
print("wrote", len(report.kept), "links, output hash",
      save_training_links(report.kept, raw_hash, merged_corpus_sha256())[:16])
PYEOF
"$PY" -m pytest tests/test_data_quality.py -q
git add data/training/hub_links_training.jsonl \
        data/training/hub_links_training.meta.json
git commit -m "feat: restore the 60 contested capec and cwe links the title floor dropped

capec goes 1,755 to 1,799 and cwe 596 to 612, taking training links to 4,389 of
4,405. The recovered links are the terse ones. The human ceiling study measured
capec agreement with OpenCRE at alpha-1 = 0.181 [0.113, 0.277] on n=83, so this
commit is a choice rather than a repair, and it reverts on its own without
touching the eleven frameworks' recoveries. Ten capec and six cwe items enter
the validation roster, taking it from 1,244 to 1,264, after the ceiling was
measured on a roster without them."
```

Record in the run ledger: 4,127 to 4,389, the 525-to-0 title-anchor figure, the
sixteen named drops, the two commit SHAs, and which of the two corpora produced
the number.

---

### Task 15: Rebuild the corpus, and prove only the eleven changed

The previous plan re-ran all 31 parsers and committed `data/processed/`
wholesale, silently re-running CAPEC and CWE, and its only mutation was
`shutil.copy2` into `data/processed/frameworks/` with no snapshot and no way
back.

#### Coverage: 89.7% of the baseline is already proven to reproduce

Every non-eleven framework has been test-rebuilt. The 19 importable parsers
reproduce **1,897 of 1,897** baseline keys with 0 mismatch **[measured, plan v2
at `8cf44b3`]**, plus 10 controls for `owasp_llm_top10_2026`, which has a parser
and no corpus entry because it landed after the baseline was taken. `defusedxml==0.7.1`
is now installed and both XML parsers reproduce byte-identically: **capec 558 of
558, cwe 1,331 of 1,331, 0 mismatch** **[measured, adjudication C-A]**.

1,897 + 1,889 = **3,786**, which is exactly the number of baseline keys outside
the eleven. **[measured, orchestrator]** Pre-measured rebuild coverage is
3,786 of 4,222 = **89.7%**, not the 45% the v2 plan assumed. The v2 Step 1 that
installs `defusedxml` and `openpyxl` is deleted: both are present, and
`openpyxl DEFUSEDXML` flipped `False` to `True` on the same install
**[measured, adjudication C-C]**.

The remaining 436 baseline keys are the eleven frameworks, and **every one of
them must change**. Their current corpus entries are OpenCRE-derived stubs where
`description == title` and `full_text` is empty:

```
nist_800_63:5-1-1-1   title '5.1.1.1'    description '5.1.1.1'    full_text None
wstg:wstg-appe-d      title 'WSTG-APPE-D' description 'WSTG-APPE-D' full_text None
csa_ccm:AIS-01        title 'Application and Interface Security Policy and Procedures'
                      description identical to the title, full_text None
```

**[measured, orchestrator]** That is why 0 of their 734 links resolve today:
`_is_prose` refuses to index a description that does not exceed its title by
`PROSE_MIN_EXTRA_CHARS`. The rebuild turns those 436 stubs into prose, so
`unchanged` must land on exactly 3,786 and no stub may survive.

#### The baseline is lossy and must be regenerated before it can gate anything

`data/processed/pre_rebuild_control_hashes.json` declares `n_controls: 4222`
against **4,261 control records on disk**. The gap is key collision, not missing
controls: **9 keys absorb 39 extra records, every one with a distinct
description**, and the stored hash is the **first** writer's every time.
**[measured, orchestrator]**

| key | records | distinct descriptions |
|---|---|---|
| `enisa:enisa:Table 5:` | 22 | 22 |
| `enisa:enisa:Table 3:` | 8 | 8 |
| `etsi:etsi:6.2.2` | 4 | 4 |
| `etsi:etsi:6.1`, `etsi:etsi:6.2.3` | 3 each | 3 each |
| `etsi:etsi:5.2.2`, `6.4.1`, `6.4.2`, `6.4.3` | 2 each | 2 each |

**[measured, orchestrator]** So `unchanged` was never a per-control count, and
38 of the 39 shadowed records were invisible to any comparison. All nine
collisions sit inside the two frameworks this task rebuilds, which is the one
place a blind spot is least affordable.

The baseline also hashes `description` alone:

```python
digest = hashlib.sha256(str(control.get("description") or "").encode("utf-8")).hexdigest()
```

This plan's own Contract Fact 1 states that `ProseIndex` prefers `full_text`
over `description` unconditionally, so whatever a parser puts in `full_text` **is
the anchor the model sees**. `full_text` is set by `parse_wstg`,
`parse_owasp_top10_2021`, `parse_owasp_proactive_controls`, `parse_nist_ssdf`,
and by `_sanitize_control` for any description over 2,000 characters. `title` is
a join channel, and `metadata["alt_ids"]` and `metadata["alt_titles"]` decide
which control a link resolves to. A rebuild can re-point every BIML and SSDF link
and report `0 changed`.

Step 2 regenerates the baseline with a collision-safe value type and a five-field
content digest, and proves the regeneration reproduces the old description-only
hashes on the 4,213 non-colliding keys, so the change of instrument does not
smuggle in a change of answer.

#### Sixty-three published assignments point at ids this rebuild retires

Five parsers change the control_id shape. Baseline keys affected, measured
exactly:

| framework | change | keys |
|---|---|---|
| wstg | `wstg-appe-d` to `WSTG-APPE-D` | 59 |
| nist_800_63 | `5-1-1-1` to `5.1.1.1` | 25 |
| owasp_proactive_controls | `c1` to `C1` | 10 |
| enisa | `Table 3:` to a per-row slug | 10 |
| csa_ccm | `IVS-*` to `I&S-*` | 7 |
| **total** | | **111** |

**[measured, orchestrator]** A rename is not a loss, so Step 4 emits a `renamed`
bucket keyed on matching content digest. For these 111 the bucket will be empty
and that is the honest answer, not a failure: the old record is a stub and the
new record is prose, so nothing can content-match. `removed` therefore means
"the stub this parser replaces" for the eleven and "content gone" everywhere
else, and Step 6's assertion is on framework membership rather than on a count
the operator has to judge.

Downstream, `build/dataset/crosswalk_v1.0.jsonl` carries **63 published rows**
whose control identity this rebuild dissolves: **56** with `control_id`
`enisa:enisa:Table 5:` (38) or `Table 3:` (18), and **7** with retired
`csa_ccm:csa_ccm:IVS-0{1,2,4,5,6,8,9}`. **[measured, orchestrator]** All 63 carry
`review_status = "ground_truth"`. `tract/export/canonical.py:76` filters on
`WHERE a.review_status = 'accepted'`, so `diff_snapshots` never sees them and
`compute_content_hash` emits no `UPDATE_CONTROL` or `DELETE_CONTROL` for any of
them. **[measured, orchestrator]** Step 8 writes the artifact that says so.

**Depends on:** Contract Rule 3's licence tiering (`OVERLAY_FRAMEWORK_IDS` in
`tract/config.py`). Step 9 imports it to decide which per-framework files may be
staged. If it is absent the step fails with `ImportError`, which is the correct
answer: committing parser output before the licence tiers exist is how licensed
prose escaped three times.

**Files:**
- Create: `scripts/rebuild_corpus.py`
- Create: `tests/test_rebuild_corpus.py`
- Create: `results/corpus/rebuild_diff.json`
- Create: `results/corpus/retired_control_ids.json`
- Modify: `data/processed/pre_rebuild_control_hashes.json`
- Modify: `data/processed/stopwords.json`
- Modify: `data/processed/frameworks/*.json`, `data/processed/all_controls.json`
- Modify: `data/training/hub_links_training.jsonl`, `data/training/hub_links_training.meta.json`

**Interfaces:**
- Consumes: every `parsers/parse_*.py`; `data/processed/pre_rebuild_control_hashes.json`; `tract.io.atomic_write_json`, `atomic_write_text`; `tract.config.OVERLAY_FRAMEWORK_IDS`.
- Produces: `content_digest(control) -> str`; `build_baseline(corpus) -> dict[str, Any]`; `snapshot_processed(root) -> Path`; `restore_snapshot(path) -> int`; `run_all(output_dir, audit_dir) -> tuple[dict[str, list[dict]], dict[str, str]]`; `diff_against_baseline(parsed, baseline) -> RebuildReport` with `RebuildReport(changed, added, removed, renamed, unchanged, failed)`; `assert_expected_frameworks_only(report) -> None`.

**Invalidates:**
- `data/processed/stopwords.json`. It is derived from the corpus, committed, applied to every control and hub text by `tract/text_selection.py`, `tract/training/data.py` and `tract/training/firewall.py`, and hashed into every fold record at `tract/training/orchestrate.py:351`. Eight modules and five test files read it. **[measured]** Step 7 regenerates and commits it. Without that, every post-rebuild metric uses a stopword list built for a corpus that no longer exists.
- `data/processed/all_controls.json` and `data/processed/licensed/all_controls.json`, and therefore every `all_controls_sha256` recorded in `results/phase1b/**/fold_result.json`.
- `data/processed/pre_rebuild_control_hashes.json`, which this task replaces with a collision-safe, five-field version.
- `data/training/hub_links_training.jsonl` and its sidecar, which Task 14 produced against the previous corpus. Step 10 regenerates them, and the sidecar test from Task 14 Step 8 stays red until it does.
- `build/dataset/crosswalk_v1.0.jsonl`'s 63 rows, and the published HuggingFace dataset built from it. No republication happens here. Step 8 records the debt.
- `results/ceiling_study/ceiling_items.json`'s sampling frame. The 250 drawn items survive intact, because none falls in the eleven and capec and cwe reproduce byte-identically. **[measured, adjudication C-B]**

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rebuild_corpus.py — create

"""A corpus rebuild must be reversible and must diff the field the model reads.

Three things the previous version could not do. It hashed `description` while
ProseIndex prefers `full_text` unconditionally, so it could re-point every link
and report 0 changed. It stored one digest per key while nine keys hold 39 extra
records with distinct text. Its only mutation was shutil.copy2 over three
files that git cannot restore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rebuild_corpus import (
    RebuildReport,
    assert_expected_frameworks_only,
    build_baseline,
    content_digest,
    diff_against_baseline,
    restore_snapshot,
    snapshot_processed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _baseline(*pairs: tuple[str, dict[str, object]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, control in pairs:
        out.setdefault(key, []).append(content_digest(control))
    return out


class TestTheDiffSeesEveryAnchorField:
    def test_identical_content_reports_no_change(self) -> None:
        control = {"control_id": "C-1", "description": "statement one"}
        report = diff_against_baseline(
            {"demo": [control]}, _baseline(("demo:C-1", control)),
        )
        assert report.changed == []
        assert report.unchanged == 1

    def test_a_moved_full_text_is_a_change(self) -> None:
        """The defect this replaces. description is equal, full_text is not.

        ProseIndex.__init__ takes full_text when it is non-empty and never
        looks at description, so this control's anchor moved entirely while a
        description-only hash reports nothing.
        """
        old = {"control_id": "C-1", "description": "same", "full_text": "before"}
        new = {"control_id": "C-1", "description": "same", "full_text": "after"}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]
        assert report.unchanged == 0

    def test_a_moved_alt_id_is_a_change(self) -> None:
        """alt_ids decides which control a link resolves to."""
        old = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_ids": ["PO.1.1"]}}
        new = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_ids": ["PO.1.2"]}}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.changed == ["demo:C-1"]

    def test_alt_lists_are_order_insensitive(self) -> None:
        old = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["b", "a"]}}
        new = {"control_id": "C-1", "description": "same",
               "metadata": {"alt_titles": ["a", "b"]}}
        report = diff_against_baseline({"demo": [new]}, _baseline(("demo:C-1", old)))
        assert report.unchanged == 1


class TestCollidingKeysAreCountedPerRecord:
    """Nine keys hold 39 extra records, all with distinct text. [measured]"""

    def test_two_records_under_one_key_are_two_units(self) -> None:
        first = {"control_id": "Table 3:", "description": "poisoning"}
        second = {"control_id": "Table 3:", "description": "data disclosure"}
        report = diff_against_baseline(
            {"enisa": [first, second]},
            _baseline(("enisa:Table 3:", first), ("enisa:Table 3:", second)),
        )
        assert report.unchanged == 2
        assert report.changed == []

    def test_losing_one_of_two_shadowed_records_is_visible(self) -> None:
        first = {"control_id": "Table 3:", "description": "poisoning"}
        second = {"control_id": "Table 3:", "description": "data disclosure"}
        report = diff_against_baseline(
            {"enisa": [first]},
            _baseline(("enisa:Table 3:", first), ("enisa:Table 3:", second)),
        )
        assert report.unchanged == 1
        assert report.removed == ["enisa:Table 3:"]


class TestRenamesAreNotLosses:
    def test_the_same_content_under_a_new_id_is_a_rename(self) -> None:
        old = {"control_id": "c1", "description": "validate every input"}
        new = {"control_id": "C1", "description": "validate every input"}
        report = diff_against_baseline(
            {"owasp_proactive_controls": [new]},
            _baseline(("owasp_proactive_controls:c1", old)),
        )
        assert report.renamed == [
            ("owasp_proactive_controls:c1", "owasp_proactive_controls:C1"),
        ]
        assert report.removed == []
        assert report.added == []

    def test_a_rename_does_not_cross_frameworks(self) -> None:
        old = {"control_id": "c1", "description": "validate every input"}
        new = {"control_id": "C1", "description": "validate every input"}
        report = diff_against_baseline(
            {"wstg": [new]}, _baseline(("owasp_proactive_controls:c1", old)),
        )
        assert report.renamed == []
        assert report.removed == ["owasp_proactive_controls:c1"]
        assert report.added == ["wstg:C1"]


class TestTheStopRuleIsAnAssertion:
    """Step 6 of the previous version was prose an autonomous worker reads past."""

    def test_an_unexpected_framework_halts_the_run(self) -> None:
        report = RebuildReport(changed=["capec:125"])
        with pytest.raises(SystemExit, match="capec"):
            assert_expected_frameworks_only(report)

    def test_a_framework_that_did_not_move_halts_the_run(self) -> None:
        """A parser that silently no-ops leaves the previous artifact in place."""
        report = RebuildReport(changed=[f"{f}:x" for f in (
            "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
            "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021", "samm",
        )])
        with pytest.raises(SystemExit, match="wstg"):
            assert_expected_frameworks_only(report)

    def test_the_expected_shape_passes(self) -> None:
        report = RebuildReport(
            changed=[f"{f}:x" for f in (
                "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63",
                "nist_ssdf", "owasp_proactive_controls", "owasp_top10_2021",
                "samm", "wstg",
            )],
            added=["owasp_llm_top10_2026:LLM01"],
            unchanged=3786,
        )
        assert_expected_frameworks_only(report) is None


class TestTheSnapshotIsARollback:
    """etsi.json, iso_27001.json and licensed/all_controls.json are untracked.

    .gitignore lines 37, 38 and 39. scripts/fetch_frameworks.py has no
    iso_27001 entry at all [measured], so ISO's raw source is hand-staged and
    its output is re-derivable from no scripted path. ISO is the corpus's only
    high-prose fold.
    """

    def test_a_snapshot_restores_byte_for_byte(self, tmp_path: Path) -> None:
        source = tmp_path / "processed"
        source.mkdir()
        original = '{\n  "a": 1\n}\n'
        (source / "etsi.json").write_text(original, encoding="utf-8")

        snapshot = snapshot_processed(tmp_path / "snapshots",
                                      members=[source / "etsi.json"])
        (source / "etsi.json").write_text('{"a": 2}', encoding="utf-8")
        assert restore_snapshot(snapshot) == 1
        assert (source / "etsi.json").read_text(encoding="utf-8") == original

    def test_a_tampered_snapshot_refuses_to_restore(self, tmp_path: Path) -> None:
        source = tmp_path / "processed"
        source.mkdir()
        (source / "etsi.json").write_text("{}\n", encoding="utf-8")
        snapshot = snapshot_processed(tmp_path / "snapshots",
                                      members=[source / "etsi.json"])
        member = next(p for p in snapshot.rglob("etsi.json"))
        member.write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="does not match its manifest"):
            restore_snapshot(snapshot)


class TestTheRegeneratedBaselineAgreesWithTheCommittedOne:
    """Changing the instrument must not change the answer it already gave."""

    def test_description_only_hashes_reproduce_on_non_colliding_keys(self) -> None:
        import hashlib

        from tract.text_selection import merged_corpus_path

        committed = json.loads(
            (REPO_ROOT / "data/processed/pre_rebuild_control_hashes.json")
            .read_text(encoding="utf-8")
        )
        if "sha256_of_description" not in committed:
            pytest.skip("baseline already regenerated by Step 2")
        old = committed["sha256_of_description"]
        corpus = json.loads(merged_corpus_path().read_text(encoding="utf-8"))
        first_seen: dict[str, str] = {}
        counts: dict[str, int] = {}
        for record in corpus["frameworks"]:
            for control in record.get("controls") or []:
                key = f"{record['framework_id']}:{control['control_id']}"
                counts[key] = counts.get(key, 0) + 1
                first_seen.setdefault(
                    key,
                    hashlib.sha256(
                        str(control.get("description") or "").encode("utf-8")
                    ).hexdigest(),
                )
        singles = [k for k, n in counts.items() if n == 1]
        assert len(singles) == 4213
        assert all(old[k] == first_seen[k] for k in singles)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"$PY" -m pytest tests/test_rebuild_corpus.py -q
```

Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.rebuild_corpus'`.

- [ ] **Step 3: Write the rebuild script**

```python
# scripts/rebuild_corpus.py — create

"""Re-run every parser into a scratch directory and diff the anchor fields.

The point is not to rebuild. It is to be able to say, per control record, what
changed, and to be able to put it back.

Three properties the previous version did not have.

Reversible. data/processed/frameworks/etsi.json, iso_27001.json and
licensed/all_controls.json are untracked (.gitignore 37-39), and
scripts/fetch_frameworks.py has no iso_27001 entry at all, so ISO's output is
re-derivable from no scripted path. --commit snapshots every overwritable file
first and --restore puts them back.

Blind to nothing. ProseIndex prefers full_text over description
unconditionally, and alt_ids and alt_titles decide which control a link
resolves to, so the digest covers all five fields. A description-only digest
could re-point every wstg, top10, proactive and nist_ssdf anchor and report
0 changed.

Enforcing. Nine baseline keys hold 39 extra records with distinct text, so the
value is a multiset of digests rather than one digest, and an unexpected
framework raises SystemExit rather than logging at INFO while --commit copies
anyway.

    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --dry-run
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --commit
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --list-snapshots
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --restore <snapshot-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

from tract.config import (
    PARSERS_DIR,
    PROCESSED_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    PROCESSED_LICENSED_DIR,
    PROJECT_ROOT,
)
from tract.io import atomic_write_json, atomic_write_text, load_json
from tract.parsers.base import BaseParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_PATH: Final[Path] = PROCESSED_DIR / "pre_rebuild_control_hashes.json"
SNAPSHOT_ROOT: Final[Path] = PROJECT_ROOT / "build" / "corpus_snapshots"

# The eleven frameworks Tasks 3-13 give a parser. Every one of their 436
# baseline keys is an OpenCRE-derived stub whose description equals its title,
# so every one MUST move. [measured]
EXPECTED_CHANGED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63", "nist_ssdf",
    "owasp_proactive_controls", "owasp_top10_2021", "samm", "wstg",
})
# Has a parser and no corpus entry: it landed after the baseline was taken, so
# its 10 controls are additions rather than changes. [measured]
EXPECTED_ADDED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "owasp_llm_top10_2026",
})
# Baseline keys outside the eleven. 1,897 from the 19 importable parsers plus
# capec 558 and cwe 1,331, each reproducing with 0 mismatch. [measured]
EXPECTED_UNCHANGED_RECORDS: Final[int] = 3786


def content_digest(control: Mapping[str, Any]) -> str:
    """Hash every field that decides which text a link resolves to.

    Hashing `description` alone, which is what the committed baseline does, is
    blind to the field the model reads. ProseIndex prefers `full_text`
    unconditionally and BaseParser._sanitize_control writes it behind the
    parser's back for any description over 2,000 characters. `title` and the
    two alternate lists decide WHICH control a link resolves to, so a change
    there re-points the link as surely.
    """
    metadata = control.get("metadata") or {}

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return sorted(str(v) for v in value)

    payload = {
        "description": str(control.get("description") or ""),
        "full_text": str(control.get("full_text") or ""),
        "title": str(control.get("title") or ""),
        "alt_ids": as_list(metadata.get("alt_ids")),
        "alt_titles": as_list(metadata.get("alt_titles")),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def build_baseline(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Digest every control record in a merged corpus, collisions included.

    The committed baseline maps one key to one digest, so nine keys holding 48
    records recorded 9 digests and shadowed 39 records with distinct text, all
    of them inside the two frameworks this rebuild touches. The value is a
    sorted list, so a key that loses one of its records is visible. [measured]
    """
    digests: dict[str, list[str]] = {}
    n_records = 0
    for record in corpus["frameworks"]:
        framework_id = record["framework_id"]
        for control in record.get("controls") or []:
            key = f"{framework_id}:{control['control_id']}"
            digests.setdefault(key, []).append(content_digest(control))
            n_records += 1
    return {
        "content_digest_fields": [
            "description", "full_text", "title", "alt_ids", "alt_titles",
        ],
        "digests": {key: sorted(values) for key, values in sorted(digests.items())},
        "n_keys": len(digests),
        "n_records": n_records,
    }


def _snapshot_members() -> list[Path]:
    """Every file --commit can overwrite."""
    members = sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json"))
    for extra in (
        PROCESSED_DIR / "all_controls.json",
        PROCESSED_LICENSED_DIR / "all_controls.json",
        PROCESSED_DIR / "stopwords.json",
        BASELINE_PATH,
    ):
        if extra.exists():
            members.append(extra)
    return members


def _member_key(path: Path) -> str:
    """Where a snapshot member came from, so restore can put it back there.

    Relative to the repo for the real artifacts, absolute otherwise, because
    the tests snapshot a tmp_path. Recording only the file name would restore
    every member to the repo root.
    """
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def _member_target(key: str) -> Path:
    candidate = Path(key)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def snapshot_processed(
    root: Path = SNAPSHOT_ROOT, members: list[Path] | None = None,
) -> Path:
    """Copy every overwritable artifact into a content-addressed directory.

    git checkout recovers 29 of the 31 per-framework files. It cannot recover
    etsi.json, iso_27001.json or licensed/all_controls.json, and ISO has no
    scripted re-fetch path. Overwriting them without a copy is the one
    irreversible act in this plan.

    The directory is named by the digest of its own manifest rather than by a
    clock. Two runs over identical inputs land in one directory, so a second
    --commit cannot bury the pristine copy under a fresher timestamp, and no
    written artifact carries a clock read.

    Copies go through atomic_write_text rather than atomic_write_json: a
    rollback that re-serialises what it restores is not a rollback.
    """
    sources = members if members is not None else _snapshot_members()
    payload = {
        str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT)
            else path.name): (
            path.read_text(encoding="utf-8")
        )
        for path in sources
    }
    manifest = {
        "files": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in sorted(payload.items())
        },
        "n_files": len(payload),
    }
    name = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    destination = root / name
    for relative, text in sorted(payload.items()):
        atomic_write_text(text, destination / "files" / relative)
    atomic_write_json(manifest, destination / "manifest.json")
    logger.info("snapshot: %d file(s) -> %s", len(payload), destination)
    return destination


def restore_snapshot(snapshot: Path) -> int:
    """Put every file in `snapshot` back, after verifying it against the manifest.

    Raises:
        ValueError: If a snapshot member's digest does not match the manifest.
            A rollback that restores corrupted bytes is worse than none.
    """
    manifest = load_json(snapshot / "manifest.json")
    restored = 0
    for relative, expected in sorted(manifest["files"].items()):
        member = snapshot / "files" / relative
        text = member.read_text(encoding="utf-8")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{member} does not match its manifest digest "
                f"({actual[:16]} against {expected[:16]}). Refusing to restore."
            )
        atomic_write_text(text, PROJECT_ROOT / relative)
        restored += 1
    logger.info("restored %d file(s) from %s", restored, snapshot)
    return restored


@dataclass
class RebuildReport:
    changed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    unchanged: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def touched_frameworks(self) -> set[str]:
        keys = self.changed + self.added + self.removed
        keys += [old for old, _ in self.renamed] + [new for _, new in self.renamed]
        return {key.split(":", 1)[0] for key in keys}


def _parser_classes() -> dict[str, type[BaseParser]]:
    """Every concrete parser class, keyed by framework_id.

    Raises:
        ValueError: If a parse_*.py module defines no BaseParser subclass.
    """
    classes: dict[str, type[BaseParser]] = {}
    for path in sorted(PARSERS_DIR.glob("parse_*.py")):
        module = importlib.import_module(f"parsers.{path.stem}")
        found = [
            value for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, BaseParser)
            and value is not BaseParser
            and value.__module__ == module.__name__
        ]
        if not found:
            raise ValueError(
                f"{path.name} defines no BaseParser subclass. Every parser "
                f"module must, or the rebuild silently skips a framework and "
                f"reports its controls as unchanged because it never ran one."
            )
        classes[found[0].framework_id] = found[0]
    return classes


def run_all(
    output_dir: Path, audit_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Run every parser into `output_dir`. Returns (controls, failures).

    audit_dir is required. BaseParser.__init__ defaults it to
    PROCESSED_REPAIR_AUDIT_DIR, so the previous version let a --dry-run write
    repair audits into the real data/processed/repair_audit/ while claiming to
    touch nothing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    parsed: dict[str, list[dict[str, Any]]] = {}
    failed: dict[str, str] = {}
    for framework_id, parser_class in sorted(_parser_classes().items()):
        try:
            result = parser_class(output_dir=output_dir, audit_dir=audit_dir).run()
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            failed[framework_id] = f"{type(error).__name__}: {error}"
            logger.error("%s FAILED: %s", framework_id, failed[framework_id])
            continue
        parsed[framework_id] = [
            control.model_dump(mode="json") for control in result.controls
        ]
    return parsed, failed


def diff_against_baseline(
    parsed: dict[str, list[dict[str, Any]]], baseline: dict[str, list[str]],
) -> RebuildReport:
    """Which control records changed anchor text, were added, moved id, or went.

    Comparison is per key on a MULTISET of digests, so a key holding several
    records is compared record by record rather than collapsing to its first
    writer.
    """
    report = RebuildReport()
    new: dict[str, Counter[str]] = {}
    for framework_id, controls in sorted(parsed.items()):
        for control in controls:
            key = f"{framework_id}:{control['control_id']}"
            new.setdefault(key, Counter())[content_digest(control)] += 1
    old = {key: Counter(values) for key, values in baseline.items()}

    surplus_new: dict[str, Counter[str]] = {}
    surplus_old: dict[str, Counter[str]] = {}
    for key in sorted(set(new) | set(old)):
        mine, theirs = new.get(key, Counter()), old.get(key, Counter())
        report.unchanged += sum((mine & theirs).values())
        left_new, left_old = mine - theirs, theirs - mine
        if left_new:
            surplus_new[key] = left_new
        if left_old:
            surplus_old[key] = left_old

    # A rename is content that survived under a different id, within one
    # framework. For the 111 id-shape changes in the eleven it finds nothing,
    # because the old record is a stub and the new one is prose, and that is
    # the honest answer rather than a failure. It exists so `removed` means
    # "content gone" for every framework where a stub is not the before state.
    for old_key in sorted(surplus_old):
        framework = old_key.split(":", 1)[0]
        for digest in list(surplus_old[old_key]):
            match = next(
                (k for k in sorted(surplus_new)
                 if k.split(":", 1)[0] == framework and surplus_new[k][digest]),
                None,
            )
            if match is None:
                continue
            surplus_new[match][digest] -= 1
            surplus_old[old_key][digest] -= 1
            report.renamed.append((old_key, match))

    for key, counter in sorted(surplus_new.items()):
        if sum(counter.values()) and key in surplus_old and sum(
            surplus_old[key].values()
        ):
            report.changed.append(key)
        elif sum(counter.values()):
            report.added.append(key)
    for key, counter in sorted(surplus_old.items()):
        if sum(counter.values()) and not sum(surplus_new.get(key, Counter()).values()):
            report.removed.append(key)

    report.changed.sort()
    report.added.sort()
    report.removed.sort()
    report.renamed.sort()
    return report


def assert_expected_frameworks_only(report: RebuildReport) -> None:
    """Halt on a framework that moved when it should not, or did not when it should.

    The previous version said "if capec, cwe, asvs, owasp_cheat_sheets,
    nist_800_53, mitre_atlas or any other framework appears in that list,
    stop". That is an instruction, and this plan's header sends execution to an
    autonomous runner. main() raised only on a parser exception. An unexpected
    change was logged at INFO and --commit copied regardless. A control whose
    only enforcement is prose is decorative (ledger lesson 4).

    Raises:
        SystemExit: On any unexpected framework, any missing framework, or an
            unchanged count that is not exactly the pre-measured 3,786.
    """
    allowed = EXPECTED_CHANGED_FRAMEWORK_IDS | EXPECTED_ADDED_FRAMEWORK_IDS
    touched = report.touched_frameworks()
    unexpected = sorted(touched - allowed)
    if unexpected:
        raise SystemExit(
            f"these frameworks moved and their parsers were not touched: "
            f"{unexpected}. Their sources are pinned and 3,786 of their control "
            f"records were pre-measured as reproducing byte-identically, so a "
            f"change here is a defect this plan introduced, not a source change."
        )
    silent = sorted(EXPECTED_CHANGED_FRAMEWORK_IDS - touched)
    if silent:
        raise SystemExit(
            f"these frameworks got a parser in Tasks 3-13 and produced no change: "
            f"{silent}. Every one of their baseline records is a stub whose "
            f"description equals its title, so every one must move. A parser "
            f"that silently no-ops leaves the previous artifact in place while "
            f"the run reports success."
        )
    if report.unchanged != EXPECTED_UNCHANGED_RECORDS:
        raise SystemExit(
            f"{report.unchanged} unchanged records against the pre-measured "
            f"{EXPECTED_UNCHANGED_RECORDS}. Below it, a framework outside the "
            f"eleven stopped reproducing. Above it, a new parser reproduced a "
            f"stub, which means it emitted the OpenCRE section name instead of "
            f"the source's prose."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("build/rebuild"))
    parser.add_argument("--audit-dir", type=Path, default=Path("build/rebuild_audit"))
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", type=Path, default=None)
    parser.add_argument("--list-snapshots", action="store_true")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_ROOT / "results/corpus/rebuild_diff.json")
    args = parser.parse_args()

    if args.list_snapshots:
        for path in sorted(SNAPSHOT_ROOT.glob("*/manifest.json")):
            manifest = load_json(path)
            logger.info("%s  %d file(s)", path.parent.name, manifest["n_files"])
        return
    if args.restore is not None:
        restore_snapshot(args.restore)
        return
    if args.commit and args.dry_run:
        raise SystemExit("--commit and --dry-run are mutually exclusive.")

    baseline = load_json(BASELINE_PATH)["digests"]
    parsed, failed = run_all(args.scratch, args.audit_dir)
    if failed:
        raise SystemExit(
            f"{len(failed)} parser(s) failed: {sorted(failed)}. A rebuild that "
            f"skips a framework leaves the previous artifact in place while "
            f"reporting success."
        )
    report = diff_against_baseline(parsed, baseline)

    logger.info(
        "rebuild: %d frameworks, %d unchanged records, %d changed, %d added, "
        "%d removed, %d renamed",
        len(parsed), report.unchanged, len(report.changed), len(report.added),
        len(report.removed), len(report.renamed),
    )
    for bucket, keys in (("changed", report.changed), ("added", report.added),
                         ("removed", report.removed)):
        counts: Counter[str] = Counter(key.split(":", 1)[0] for key in keys)
        for framework_id, count in sorted(counts.items()):
            logger.info("  %-8s %-26s %d", bucket, framework_id, count)

    atomic_write_json(
        {
            "changed": report.changed, "added": report.added,
            "removed": report.removed,
            "renamed": [list(pair) for pair in report.renamed],
            "unchanged": report.unchanged,
        },
        args.report,
    )
    assert_expected_frameworks_only(report)

    if args.commit:
        snapshot = snapshot_processed()
        logger.info("rollback: --restore %s", snapshot)
        for source in sorted(args.scratch.glob("*.json")):
            atomic_write_json(
                load_json(source), PROCESSED_FRAMEWORKS_DIR / source.name,
            )
        logger.info("committed %d artifact(s) into %s",
                    len(list(args.scratch.glob("*.json"))),
                    PROCESSED_FRAMEWORKS_DIR)


if __name__ == "__main__":
    main()
```

The commit path uses `atomic_write_json(load_json(source), ...)` rather than
`shutil.copy2`, per the plan's own Global Constraint. It is byte-identical:
`BaseParser.run` writes its output through the same helper at `base.py:346`, and
`atomic_write_json` is deterministic (sorted keys, 2-space indent,
`ensure_ascii=False`, trailing newline). **[measured]**

- [ ] **Step 4: Regenerate the baseline and prove it agrees with the old one**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json

from scripts.rebuild_corpus import BASELINE_PATH, build_baseline
from tract.io import atomic_write_json, load_json
from tract.text_selection import merged_corpus_path

corpus = load_json(merged_corpus_path())
baseline = build_baseline(corpus)
baseline["generated_from"] = str(merged_corpus_path())
print("keys:", baseline["n_keys"], "records:", baseline["n_records"])
atomic_write_json(baseline, BASELINE_PATH)
PYEOF
"$PY" -m pytest tests/test_rebuild_corpus.py -q
"$PY" -m mypy scripts/rebuild_corpus.py --strict
```

Expected: `keys: 4222 records: 4261`. **[measured, orchestrator]** The old file
declared `n_controls: 4222` against those same 4,261 records, so this step
recovers the 39 shadowed ones. Run
`tests/test_rebuild_corpus.py::TestTheRegeneratedBaselineAgreesWithTheCommittedOne`
**before** overwriting the file, or it skips.

`git add data/processed/pre_rebuild_control_hashes.json` in Step 9's commit. Note
that the corpus this is derived from is the overlay, so a checkout without the
overlay produces 3,905 keys instead of 4,222 and Step 5 will report the missing
frameworks as `removed`. Run this task on a checkout that holds the overlay.

- [ ] **Step 5: Dry run, and read the diff before touching anything**

```bash
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --dry-run
```

Every line must be explainable before proceeding, and the script now halts on
its own if it is not:

- `0 parser failure(s)`, enforced by the `SystemExit` above the diff.
- `unchanged` exactly **3,786**, enforced by `assert_expected_frameworks_only`.
  Anything else means a framework outside the eleven stopped reproducing, or a
  new parser reproduced a stub.
- `changed` and `removed` reported only for the eleven. `capec`, `cwe`, `asvs`,
  `owasp_cheat_sheets`, `nist_800_53` and `mitre_atlas` appearing raises
  `SystemExit` rather than printing a warning the runner reads past.
- `removed` of roughly **111**, and the assertion is on framework membership,
  not on that number. Composition: wstg 59, nist_800_63 25, owasp_proactive_controls
  10, enisa 10, csa_ccm 7. **[measured]** Each is an OpenCRE-derived stub id the
  new parser replaces with the source's own id.
- `renamed` of **0** for the eleven, expected and stated: the old record is a
  stub whose description equals its title, so no prose control can content-match
  one. The bucket exists so `removed` still means "content gone" elsewhere.
- `added` covering the controls the new parsers emit beyond the stubs, plus
  `owasp_llm_top10_2026`'s 10. Expected magnitudes from the parser tasks:
  dsomm 194 against 183 stubs, wstg 115 against 59, csa_ccm 224 against 29,
  iso 27001 unchanged at 93.

Record `unchanged`, and the per-framework `changed` / `added` / `removed` counts
from `results/corpus/rebuild_diff.json`, in the run ledger before Step 6.

- [ ] **Step 6: Snapshot and commit the rebuild**

```bash
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --commit 2>&1 | tee /tmp/rebuild.log
grep "rollback: --restore" /tmp/rebuild.log
PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --list-snapshots
```

Copy the `--restore` line into the run ledger before running anything else. That
directory is the only recovery path for
`data/processed/frameworks/etsi.json`, `iso_27001.json` and
`data/processed/licensed/all_controls.json`, which `git checkout` cannot restore
and `scripts/fetch_frameworks.py` cannot refetch. **[measured]**

```bash
PYTHONPATH=. "$PY" parsers/merge_all_controls.py
PYTHONPATH=. "$PY" parsers/validate_all.py
```

- [ ] **Step 7: Regenerate the stop word list**

`data/processed/stopwords.json` is derived from the corpus, committed, applied
to every control and hub text, and hashed into every fold record at
`tract/training/orchestrate.py:351`. The rebuild replaces 436 stub records with
prose. The v2 plan mentioned `stopwords` zero times in 6,987 lines. **[measured]**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
from pathlib import Path
before = json.loads(Path("data/processed/stopwords.json").read_text(encoding="utf-8"))
print("before:", before["count"], "words from", before["n_documents"], "documents")
Path("/tmp/stopwords_before.json").write_text(json.dumps(before, sort_keys=True))
PYEOF
PYTHONPATH=. "$PY" -m scripts.build_stopwords
PYTHONPATH=. "$PY" - <<'PYEOF'
import json
from pathlib import Path
before = json.loads(Path("/tmp/stopwords_before.json").read_text(encoding="utf-8"))
after = json.loads(Path("data/processed/stopwords.json").read_text(encoding="utf-8"))
print("after:", after["count"], "words from", after["n_documents"], "documents")
print("added:", sorted(set(after["stopwords"]) - set(before["stopwords"])))
print("removed:", sorted(set(before["stopwords"]) - set(after["stopwords"])))
assert after["min_doc_freq"] == before["min_doc_freq"] == 0.05
PYEOF
"$PY" -m pytest tests/test_stopword_filtering.py tests/test_stopword_protection.py -q
```

Before: 78 words from 4,783 documents at `min_doc_freq = 0.05`. **[measured]**
After: **[unmeasured]** until the eleven parsers exist. Record both counts and
the full added and removed word lists in the run ledger.

One consequence to state rather than discover later. `scripts/build_stopwords.py:37`
reads `PROCESSED_DIR / "all_controls.json"`, the tracked corpus, not
`merged_corpus_path()`. **[measured]** So the list is identical on a machine with
the overlay and one without, which is the right property for a committed artifact
that is hashed into every run record. The cost is that the overlay frameworks'
boilerplate does not vote on the list. Under Contract Rule 3 that is nine
frameworks. Leave it, and record it in "What this plan does not close".

```python
# tests/test_rebuild_corpus.py — append

def test_the_committed_stopword_list_reproduces_from_the_committed_corpus() -> None:
    """Catches the staleness directly rather than by remembering to rerun it.

    The list is applied to every control and hub text and hashed into every
    fold record. A list built for a corpus that no longer exists is invisible
    in the metrics and changes every one of them.
    """
    import json

    from scripts.build_stopwords import collect_documents
    from tract.stopwords import STOPWORDS_PATH, generate_stopwords

    committed = json.loads(STOPWORDS_PATH.read_text(encoding="utf-8"))
    documents, protected = collect_documents()
    words = generate_stopwords(
        documents,
        min_doc_freq=committed["min_doc_freq"],
        max_words=committed["max_words"],
        protect=protected,
    )
    assert sorted(words) == committed["stopwords"]
    assert len(documents) == committed["n_documents"]
```

- [ ] **Step 8: Record the 63 published assignments this rebuild orphans**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
import json

from tract.io import atomic_write_json, load_json
from tract.text_selection import merged_corpus_path

rows = [json.loads(l) for l in
        open("build/dataset/crosswalk_v1.0.jsonl", encoding="utf-8") if l.strip()]
corpus = load_json(merged_corpus_path())
live = {
    f"{record['framework_id']}:{control['control_id']}"
    for record in corpus["frameworks"] for control in record.get("controls") or []
}
orphans = [
    {
        "control_id": row["control_id"], "framework_id": row["framework_id"],
        "hub_id": row["hub_id"], "review_status": row["review_status"],
        "section_id": row.get("section_id", ""),
    }
    for row in rows
    if row["control_id"].split(":", 1)[1] not in
    {k.split(":", 1)[1] for k in live if k.startswith(row["framework_id"] + ":")}
]
atomic_write_json(
    {
        "note": (
            "Published rows in crosswalk_v1.0.jsonl whose control_id the "
            "corpus rebuild retires. All carry review_status='ground_truth'. "
            "tract/export/canonical.py:76 filters on review_status='accepted', "
            "so diff_snapshots never sees them and no UPDATE_CONTROL or "
            "DELETE_CONTROL changeset will mention them. Republication is "
            "banned, so this file is the record until that ban lifts."
        ),
        "n_rows": len(orphans),
        "rows": sorted(orphans, key=lambda r: (r["framework_id"], r["control_id"],
                                               r["hub_id"])),
    },
    "results/corpus/retired_control_ids.json",
)
print("orphaned published rows:", len(orphans))
PYEOF
```

Expected: **63**. **[measured, orchestrator]** 56 `enisa:enisa:Table 5:` (38) and
`Table 3:` (18), plus 7 `csa_ccm:csa_ccm:IVS-0{1,2,4,5,6,8,9}`. The file carries
control ids and hub ids and no control text, so it is safe to track for the
overlay frameworks as well.

```python
# tests/test_rebuild_corpus.py — append

def test_every_retired_published_id_is_named() -> None:
    """63 published assignments lose their control identity. [measured]

    The export path cannot see them: all 63 carry review_status='ground_truth'
    and tract/export/canonical.py filters on 'accepted', so no changeset will
    ever mention them. This file is the only record.
    """
    import json

    retired = json.loads(
        (REPO_ROOT / "results/corpus/retired_control_ids.json")
        .read_text(encoding="utf-8")
    )
    assert retired["n_rows"] == 63
    assert {row["framework_id"] for row in retired["rows"]} == {"enisa", "csa_ccm"}
    assert {row["review_status"] for row in retired["rows"]} == {"ground_truth"}
```

- [ ] **Step 9: Verify the licence channel before staging anything**

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
git add scripts/rebuild_corpus.py tests/test_rebuild_corpus.py \
        data/processed/pre_rebuild_control_hashes.json \
        data/processed/stopwords.json \
        data/processed/all_controls.json \
        results/corpus/rebuild_diff.json \
        results/corpus/retired_control_ids.json
git add data/processed/frameworks/
PYTHONPATH=. "$PY" - <<'PYEOF'
import subprocess
import sys

from tract.config import OVERLAY_FRAMEWORK_IDS

staged = subprocess.run(
    ["git", "diff", "--cached", "--name-only"],
    capture_output=True, text=True, check=True,
).stdout.split()
leaked = [
    path for path in staged
    if path.startswith("data/processed/frameworks/")
    and path.rsplit("/", 1)[1].removesuffix(".json") in OVERLAY_FRAMEWORK_IDS
]
if leaked:
    sys.exit(
        f"these paths carry text on terms a CC0 grant cannot carry and are "
        f"staged: {leaked}. .gitignore is not covering them in this checkout. "
        f"git push is the publication event regardless of what any publish-path "
        f"filter does."
    )
print(f"licence channel clear: {len(staged)} staged path(s), 0 overlay frameworks")
PYEOF
```

The v2 version checked this with `git status --porcelain` and a paragraph telling
the operator to look. `git add data/processed/frameworks/` exits 1 on the ignored
members while staging the rest, so the exit code says nothing useful about which
files landed. The check above reads what git staged and exits non-zero.

The `OVERLAY_FRAMEWORK_IDS` import is the ordering dependency stated in the task
header. An `ImportError` here means Contract Rule 3's licence tiering has not
landed, and the correct response is to land it, not to fall back to
`RESTRICTED_FRAMEWORK_IDS` and stage seven conditional frameworks' prose.

```bash
"$PY" -m pytest tests/test_rebuild_corpus.py tests/test_corpus_invariants.py \
      tests/test_licensed_text_not_tracked.py tests/test_holdout_framework.py \
      tests/test_framework_licenses.py tests/test_parser_manifest_coverage.py \
      tests/test_prose_reachability.py -q
git commit -m "chore: rebuild the corpus from pinned sources, with the per-record diff

3,786 control records outside the eleven reproduce unchanged, pre-measured at
1,897 from the 19 importable parsers plus capec 558 and cwe 1,331 with 0
mismatch. The eleven's 436 records were OpenCRE-derived stubs whose description
equalled their title, so every one moves: roughly 111 through an id-shape change
and the rest in place. The baseline is regenerated with a five-field content
digest and a per-key digest multiset, which recovers 39 records that nine
colliding keys had shadowed. The stop word list is rebuilt from the new corpus.
Every overwritten file is snapshotted first, because git cannot restore three of
them and no script can refetch ISO 27001."
```

- [ ] **Step 10: Regenerate the training file against the corpus that now exists**

Task 14 made `hub_links_training.jsonl` a function of the corpus. This step is
what closes ledger lesson 6, and the sidecar test from Task 14 Step 8 is red
until it runs.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
PYTHONPATH=. "$PY" - <<'PYEOF'
from tract.text_selection import merged_corpus_sha256
from tract.training.data_quality import curated_link_filter_report, save_training_links

report, raw_hash = curated_link_filter_report()
print("corpus:", report.corpus_path)
print("training links:", len(report.kept))
print("dropped unresolved:", report.dropped_unresolved)
print("dropped thin anchor:", report.dropped_thin_anchor)
save_training_links(report.kept, raw_hash, merged_corpus_sha256())
PYEOF
"$PY" -m pytest tests/test_data_quality.py tests/test_ceiling_study.py -q
git add data/training/hub_links_training.jsonl \
        data/training/hub_links_training.meta.json
git commit -m "chore: rebuild the training links against the rebuilt corpus

hub_links_training.jsonl resolves every link through ProseIndex, so it is a
function of the corpus, and Step 6 changed the corpus. The sidecar records the
sha256 of the corpus this file was built from, and the test that compares the
two is what makes the ordering enforceable rather than a claim."
```

Expected: `training links: 4389`, sixteen named unresolved drops, zero thin
anchors, and the corpus digest matching `data/processed/licensed/all_controls.json`.
**[derived]** A different number here than in Task 14 Step 9 means the rebuild
changed which links resolve, which is information the run ledger needs and the
Task 14 commit message does not carry. Record both.

---
