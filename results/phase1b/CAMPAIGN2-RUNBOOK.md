# Campaign 2 runbook

The commands that execute `CAMPAIGN2.md`, which is the binding pre-registration
and governs anything here that disagrees with it.

**Type from this file. Do not pipe it to a shell**, and do not copy commands out
of the orchestrator's printed retry message — that message omits `--split` and
every arm flag, so retrying A3 from it silently re-runs the arm as BGE-large.
It is markdown rather than a script for the same reason: nothing here should be
executable in one keystroke.

Written 2026-08-27 against `eb15f37`. Flags verified against
`runpod_parallel --help`, not copied from prose.

Run everything in the campaign tmux session, and nowhere else:

```bash
tmux attach -t tract-campaign2
```

## Gate 0 — before anything

Each line fails loudly rather than defaulting.

```bash
echo "budget=${TRACT_RUNPOD_BUDGET_USD:?NOT SET - provision would run at the 1000 default}"
python -m scripts.stage_licensed_overlay --verify     # must exit 0
python -m scripts.phase1b.runpod_parallel price       # read the budget line it prints
systemctl --user list-timers 'tract-reaper*'          # the reaper must be armed
git status --porcelain                                # must be empty, or git_sha lies
```

An unset budget is **permissive, not refusing**: `runpod_parallel.py:116`
defaults to `1000`. `:?` turns the missing export into a non-zero exit.

## The three validation arms

A2 and A4 were dropped on 2026-08-27; see `CAMPAIGN2.md` Amendment 1. Config
names are **bound by the pre-registration**. `results/phase1b/c2_A1_prose_sw_bge`
already holds stale folds and `collect` rsyncs without `--delete`, so reusing a
name merges two corpora into one plausible-looking aggregate.

| arm | `--config-name` | arm flags |
|---|---|---|
| A1 — BGE, prose+stopwords | `c2r_A1_prose_sw_bge` | `--stopwords` |
| A3 — Qwen3-0.6B, prose+stopwords | `c2r_A3_prose_sw_qwen06b` | `--stopwords --base-model Qwen/Qwen3-Embedding-0.6B` |
| A5 — BGE, title-only | `c2r_A5_title_bge` | `--no-prose` |

All three take `--split validation`.

## Per arm — drive the subcommands, never `full`

`full_pipeline` calls `teardown()` in its `finally` and `aggregate()` **after**,
so every structural refusal inside `load_fold_results` fires with the pods
already destroyed. Running the steps by hand inverts that ordering.

```bash
python -m scripts.phase1b.runpod_parallel provision \
    --config-name c2r_A1_prose_sw_bge --split validation --stopwords
```

**Gate A — which GPU did we actually get?**

```bash
python -c "import json;s=json.load(open('scripts/phase1b/.pod_state.json'));print([(p['name'],p.get('gpu_type'),p.get('cloud_type')) for p in s['pods']])"
```

`provision` falls through GPU types on a capacity error, and the candidate tail
past `GPU_PREFERENCE` is sorted by **VRAM descending, not speed**, with
`min_vram_gb=48` admitting A40/L40S-class parts. A 57-minute Qwen fold on a
0.47x part becomes ~120 minutes, which is exactly `FOLD_TIMEOUT_S` — all five
folds die at the wall. H100 or A100-80GB: proceed. Anything 48GB: `teardown`
and retry later. Non-negotiable for A3.

**Gate B — one real SSH handshake.**

Nothing in provisioning authenticates. `_wait_for_ssh` returns on a bare TCP
connect, so *"All 5 pods provisioned and SSH-ready"* means port 22 answered, not
that our key works. The first real contact is `check=False`, so
`Permission denied (publickey)` arrives as one WARNING among many and then
surfaces as an opaque rsync failure. This check costs ~$0.45; discovering it
overnight costs ~$108.

```bash
python -c "import json;s=json.load(open('scripts/phase1b/.pod_state.json'));p=s['pods'][0];print(p['ip'],p['port'])"

ssh -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=scripts/phase1b/.runpod_known_hosts \
    -o IdentitiesOnly=yes -o BatchMode=yes -i ~/.ssh/tract_runpod \
    -p <PORT> root@<IP> 'id && (command -v rsync || echo NO_RSYNC)'
```

`uid=0(root)` plus an rsync path: proceed. `Permission denied`: `teardown` now.

**Run, collect, and check the tripwire.**

```bash
python -m scripts.phase1b.runpod_parallel run     --config-name c2r_A1_prose_sw_bge --split validation --stopwords
python -m scripts.phase1b.runpod_parallel collect --config-name c2r_A1_prose_sw_bge --split validation --stopwords

git status --porcelain -- 'results/phase1b/c2r_*'
find results/phase1b -name fold_result.json | wc -l     # must increase by 5
```

If `git status` shows nothing after a collect, that is the `.gitignore` defect
returning — not a run that produced nothing.

**Aggregate before teardown.**

```bash
python -m scripts.phase1b.runpod_parallel aggregate \
    --config-name c2r_A1_prose_sw_bge --split validation --stopwords --n-configurations 3
python -m scripts.phase1b.runpod_parallel teardown
```

`--n-configurations` has no default and `aggregate` refuses without it. **3** on
validation, where selection happens; **1** on the test round.

**Then bank it**: licensed-text gate, commit, push, confirm 8/8 checks green.

```bash
python -m pytest tests/test_licensed_text_not_tracked.py -q
```

## Selection, after all three arms

Rank on **absolute micro-averaged validation hit@1, not delta**. The gate delta
is measured against a per-arm zero-shot baseline that moves with both anchor and
encoder, so ranking on delta rewards whichever arm started worst. A1 advances
unless another arm beats it by more than **4.0 hit@1 points** — the corrected
validation MDE. An arm can win selection and still fail the gate; that is the
design, not a defect.

## The test round — once

```bash
# --split test, winner's flags, --config-name c2r_TEST_<winner>
# aggregate with --n-configurations 1: selection already happened on a
# disjoint split, so there is no multiplicity left to price.
```

Then the agentic smoke test, once, on the winning arm only. Report as "n of 6"
in prose. It is not a metric, does not enter `gate_decision`, and no arm is
re-selected on it whatever it says.

## When the campaign is over

```bash
mkdir -p "${XDG_RUNTIME_DIR:-/tmp}/tract-reaper"
touch "${XDG_RUNTIME_DIR:-/tmp}/tract-reaper/campaign-complete"
systemctl --user stop 'tract-reaper*.timer'
```

Without the sentinel the guard keeps re-arming for about six hours before its
quiet streak decides the campaign is finished. That is the safe direction, but
say so explicitly rather than waiting it out.

## If pods are left up

`full_pipeline` leaves the fleet running on **every** failure path, deliberately,
so a failed fold can be retried on a pod that is still warm. Disarm the reaper
before touching such a fleet, or it will reap the recovery window.

```bash
systemctl --user stop 'tract-reaper*.timer'        # BEFORE any warm-pod retry
# ... retry or collect ...
python -m scripts.phase1b.runpod_parallel reap --confirm   # when genuinely done
```
