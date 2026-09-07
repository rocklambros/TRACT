# Annotation packets

Generated packets, committed so a coordinator can pull one down on another
machine without rebuilding it.

## What is here

| packet | framework | contents |
|---|---|---|
| `phase2c-nist_800_53/` | NIST SP 800-53 | 78 AI hubs, 300 controls, blank answer sheet |

Each packet carries a `manifest.json` recording its framework, build time, the
git SHA of the tree that produced it, and a sha256 of every file. A returning
annotator sheet can be tied back to the exact bytes that were sent.

## Sending one

Read `docs/phase2c-annotator-handbook.md` first. Part 1 is for the coordinator;
Part 2 and the three CSVs are what the annotator receives.

```bash
# On another machine
git clone https://github.com/rocklambros/TRACT.git
cd TRACT
ls packets/phase2c-nist_800_53/
```

Send **`ai_hubs.csv`, `controls.csv` and `annotate.csv`**, plus Part 2 of the
handbook. Nothing else — in particular never
`results/ceiling_study/hub_reference.md`, whose 400 LLM-written hub
descriptions would make every label Tier 3.

## Why these may be committed, and the next one may not

A packet in git is published to everyone who clones the repository, so a
committed packet is redistribution.

NIST SP 800-53 is recorded as a US Government work not subject to copyright,
adjudicated 2026-09-06, and that was checked before this packet was copied in.
**That is a property of NIST 800-53, not of packets.** A packet built from
`csa_aicm`, `csa_ccm`, `etsi`, `iso_27001` or `dsomm` may not be committed —
their prose is either proprietary or licence-restricted.

The fingerprint gate does not protect you here: it fingerprints `dsomm`, `etsi`
and `iso_27001` only, so a CSA packet would pass it unnoticed.

`tests/test_packet_manifest.py` is what actually guards this. It reads each
committed packet's manifest and refuses any whose framework is not
redistributable, refuses one with no manifest at all, and refuses one whose
files no longer match their recorded digests. It also refuses a packet
containing filled answers — a committed answer key would be worse than a
committed corpus.

## Rebuilding

```bash
python -m scripts.build_bridge_packet packets/phase2c-nist_800_53 \
  --framework-id nist_800_53
```

Deterministic apart from `built_at` and `git_sha`. If you rebuild, commit the
regenerated `manifest.json` with it or the digest test will fail — which is the
point.

## Building a packet you do NOT intend to commit

Write it outside the working tree:

```bash
python -m scripts.build_bridge_packet ~/tract-packets/scratch --framework-id <id>
```

That is the default advice in the handbook, and it stands for anything not
deliberately reviewed for redistribution the way this one was.
