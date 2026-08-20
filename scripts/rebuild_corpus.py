"""Re-run every parser into a scratch directory and diff the anchor fields.

The point is not to rebuild. It is to be able to say, per control record, what
changed, and to be able to put it back.

Three properties the previous version did not have.

Reversible. The overlay tier's per-framework files are gitignored, and
scripts/fetch_frameworks.py has no iso_27001 entry at all, so ISO's output is
re-derivable from no scripted path. --commit snapshots every overwritable file
first and --restore puts them back. The irrecoverable set is DERIVED from
tract.config.OVERLAY_FRAMEWORK_IDS rather than listed, because rulings R4 and
R10 added two members after the list was written and a list leaves every future
member with no rollback.

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
    PYTHONPATH=. "$PY" scripts/rebuild_corpus.py --build-baseline <framework-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from tract.config import (
    OVERLAY_FRAMEWORK_IDS,
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
# baseline keys is an OpenCRE-derived stub whose description equals its title
# and whose control_id carries a redundant `<framework_id>:` prefix, so every
# one MUST move. [measured]
EXPECTED_CHANGED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "biml", "csa_ccm", "dsomm", "enisa", "etsi", "nist_800_63", "nist_ssdf",
    "owasp_proactive_controls", "owasp_top10_2021", "samm", "wstg",
})
# Has a parser and no corpus entry: it landed after the baseline was taken, so
# its 10 controls are additions rather than changes. It is also a pretraining
# holdout, which is why merge_all_controls drops it from both merged corpora
# and it can never reappear in the baseline. [measured]
EXPECTED_ADDED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "owasp_llm_top10_2026",
})
# Control records outside the eleven that may move at all, named rather than
# counted. pdfplumber is pinned and pdfminer.six, which does the glyph layout
# under it, was not. Under pdfminer.six 20260107 the rotated axis labels of the
# two taxonomy figures in NIST AI 100-2 order differently from the build that
# produced the committed artifact. Only figure glyphs move; the taxonomy prose
# is byte-identical, and two consecutive fresh runs agree. requirements.txt now
# pins pdfminer.six and these two records are regenerated under that pin, so a
# third movement here is drift this pin was supposed to stop. [measured]
DECLARED_MOVED_KEYS: Final[frozenset[str]] = frozenset({
    "nist_ai_100_2:2.1", "nist_ai_100_2:3.1",
})

# Baseline records outside the eleven that must reproduce, derived from the
# baseline itself rather than from the run it gates: 4,261 pre-rebuild records
# minus the 475 inside the eleven is 3,786, minus the 2 declared above. capec
# 558 and cwe 1,331 reproduce byte-identically under defusedxml, which is half
# of this total on its own. [measured]
EXPECTED_UNCHANGED_RECORDS: Final[int] = 3784

# Fields that decide which text a link resolves to, and therefore what a
# rebuild can silently move. Order is fixed because it is serialised.
CONTENT_DIGEST_FIELDS: Final[tuple[str, ...]] = (
    "description", "full_text", "title", "alt_ids", "alt_titles",
)


def content_digest(control: Mapping[str, Any]) -> str:
    """Hash every field that decides which text a link resolves to.

    Hashing `description` alone, which is what the committed baseline did, is
    blind to the field the model reads. ProseIndex prefers `full_text`
    unconditionally and BaseParser._sanitize_control writes it behind the
    parser's back for any description over 2,000 characters. `title` and the
    two alternate lists decide WHICH control a link resolves to, so a change
    there re-points the link as surely.

    An absent field and an empty one hash the same, because BaseParser.run
    serialises with exclude_none and a digest that told them apart would
    report every such control as changed.
    """
    metadata = control.get("metadata") or {}

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return sorted(str(item) for item in value)

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

    The committed baseline mapped one key to one digest, so nine keys holding
    48 records recorded 9 digests and shadowed 39 records with distinct text,
    all of them inside the two frameworks this rebuild touches. The value is a
    sorted list, so a key that loses one of its records is visible. [measured]

    Raises:
        ValueError: If a framework record carries no framework_id, or a control
            carries no control_id. A digest under an unknown key is worse than
            none: it reports a key that no reader can join back to a control.
    """
    digests: dict[str, list[str]] = {}
    n_records = 0
    for record in corpus["frameworks"]:
        framework_id = record.get("framework_id")
        if not framework_id:
            raise ValueError(
                f"a framework record carries no framework_id: "
                f"{sorted(record)[:6]}. Every digest is keyed on it."
            )
        for control in record.get("controls") or []:
            control_id = control.get("control_id")
            if not control_id:
                raise ValueError(
                    f"{framework_id} has a control with no control_id: "
                    f"{json.dumps(control, sort_keys=True)[:200]}"
                )
            digests.setdefault(f"{framework_id}:{control_id}", []).append(
                content_digest(control)
            )
            n_records += 1
    return {
        "content_digest_fields": list(CONTENT_DIGEST_FIELDS),
        "digests": {key: sorted(values) for key, values in sorted(digests.items())},
        "n_keys": len(digests),
        "n_records": n_records,
    }


def corpus_from_framework_dir(directory: Path) -> dict[str, Any]:
    """Assemble a merged-corpus-shaped mapping from per-framework artifacts.

    The pre-rebuild corpus is reconstructible from git only as a set of
    per-framework files, and the merged overlay it was originally taken from
    is gitignored. Reading a directory is what makes the baseline
    regenerable by a reviewer rather than a number to be trusted.

    Raises:
        FileNotFoundError: If the directory holds no framework artifact.
    """
    frameworks = [load_json(path) for path in sorted(directory.glob("*.json"))]
    if not frameworks:
        raise FileNotFoundError(f"No framework artifacts in {directory}")
    return {"frameworks": frameworks}


def irrecoverable_members() -> list[Path]:
    """The artifacts `git checkout` cannot restore, derived from the licence tier.

    Every framework in OVERLAY_FRAMEWORK_IDS routes its per-framework file to
    the gitignore, and the merged overlay is gitignored with them. The brief
    named three files against .gitignore lines 37-39; rulings R4 and R10 then
    added csa_ccm and dsomm to the tier, which a hand-written list of three
    would have left with no rollback path at all.
    """
    members = [
        PROCESSED_FRAMEWORKS_DIR / f"{framework_id}.json"
        for framework_id in sorted(OVERLAY_FRAMEWORK_IDS)
    ]
    members.append(PROCESSED_LICENSED_DIR / "all_controls.json")
    return [path for path in members if path.exists()]


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


def resolve_required(
    members: list[Path] | None, require: Sequence[Path] | None,
) -> list[Path]:
    """Which artifacts this snapshot must contain before it may be written.

    Explicit `require` always wins. Otherwise the demand attaches to the
    DEFAULT member set only: a caller that names its own members is
    snapshotting some other tree, and demanding the repository's overlay files
    of it would refuse every call that is not the production one.
    """
    if require is not None:
        return list(require)
    return irrecoverable_members() if members is None else []


def snapshot_processed(
    root: Path = SNAPSHOT_ROOT,
    members: list[Path] | None = None,
    require: Sequence[Path] | None = None,
) -> Path:
    """Copy every overwritable artifact into a content-addressed directory.

    git checkout recovers the 28 tracked per-framework files. It cannot recover
    the overlay tier's four, or the merged overlay, and ISO has no scripted
    re-fetch path. Overwriting them without a copy is the one irreversible act
    in this task, so `require` refuses rather than logs.

    The directory is named by the digest of its own manifest rather than by a
    clock. Two runs over identical inputs land in one directory, so a second
    --commit cannot bury the pristine copy under a fresher timestamp, and no
    written artifact carries a clock read.

    Copies go through atomic_write_text rather than atomic_write_json: a
    rollback that re-serialises what it restores is not a rollback.

    Raises:
        ValueError: If any path in `require` is missing from `members`.
    """
    sources = members if members is not None else _snapshot_members()
    demanded = resolve_required(members, require)
    missing = sorted(str(path) for path in demanded if path not in sources)
    if missing:
        raise ValueError(
            f"these irrecoverable artifacts are not in the snapshot: {missing}. "
            f"They are gitignored under the licence tiering and no script "
            f"refetches ISO 27001, so overwriting them without a copy cannot "
            f"be undone."
        )

    # Every member is read before anything is written, so a read failure
    # leaves no half-written snapshot for --restore to trust.
    payload = {
        _member_key(path): path.read_text(encoding="utf-8") for path in sources
    }
    manifest = {
        "files": {
            key: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for key, text in sorted(payload.items())
        },
        "n_files": len(payload),
    }
    name = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    destination = root / name
    for key, text in sorted(payload.items()):
        atomic_write_text(text, destination / "files" / key.lstrip("/"))
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
    for key, expected in sorted(manifest["files"].items()):
        member = snapshot / "files" / key.lstrip("/")
        text = member.read_text(encoding="utf-8")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{member} does not match its manifest digest "
                f"({actual[:16]} against {expected[:16]}). Refusing to restore."
            )
        atomic_write_text(text, _member_target(key))
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


def _normalize_id(control_id: str) -> str:
    """Fold the separator and case differences five parsers changed.

    nist_800_63 `5-1-1-1` to `5.1.1.1`, wstg `wstg-appe-d` to `WSTG-APPE-D`,
    owasp_proactive_controls `c1` to `C1`, csa_ccm `IVS-01` to `I&S-01`. Used
    only to classify a removal, never to decide equality: two controls with
    the same normalised id and different text are still two controls.
    """
    folded = control_id.strip().casefold()
    for old, new in (("-", "."), ("_", "."), ("&", ""), (" ", "")):
        folded = folded.replace(old, new)
    return folded


def classify_removed(
    removed: Sequence[str], parsed: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[str]]:
    """Split removed keys by whether the control's identity survived.

    `removed` on its own cannot tell a retired key prefix from a lost control.
    Every pre-rebuild control_id in the eleven carries a redundant
    `<framework_id>:` prefix that the OpenCRE extraction wrote and no new
    parser reproduces, so all 436 baseline keys are literally removed while
    328 of them keep their identifier exactly. An operator cannot act on the
    unsplit number. [measured]

    Buckets, in order of decreasing certainty that nothing was lost:
        prefix_only  the identifier reappears verbatim as a control_id or an
                     alt_id in the same framework
        id_reshaped  it reappears once case and separators are folded
        gone         no successor identifier at all

    Lineage never crosses a framework: a samm identifier reappearing under
    wstg is a loss, not a rename.
    """
    live_exact: dict[str, set[str]] = {}
    live_normal: dict[str, set[str]] = {}
    for framework_id, controls in parsed.items():
        exact: set[str] = set()
        for control in controls:
            exact.add(str(control["control_id"]))
            metadata = control.get("metadata") or {}
            exact.update(str(alt) for alt in (metadata.get("alt_ids") or []))
        live_exact[framework_id] = exact
        live_normal[framework_id] = {_normalize_id(value) for value in exact}

    buckets: dict[str, list[str]] = {
        "prefix_only": [], "id_reshaped": [], "gone": [],
    }
    for key in removed:
        framework_id, control_id = key.split(":", 1)
        # The pre-rebuild extraction wrote `<framework_id>:<id>` into
        # control_id itself, so the baseline key holds the prefix twice.
        suffix = control_id.removeprefix(f"{framework_id}:")
        if suffix in live_exact.get(framework_id, set()):
            buckets["prefix_only"].append(key)
        elif _normalize_id(suffix) in live_normal.get(framework_id, set()):
            buckets["id_reshaped"].append(key)
        else:
            buckets["gone"].append(key)
    for values in buckets.values():
        values.sort()
    return buckets


def sole_parser_class(
    module_name: str, candidates: Sequence[type[BaseParser]],
) -> type[BaseParser]:
    """The one concrete parser a module defines.

    Split out from the discovery loop so the count can be exercised with two
    candidates. Inline, the branch was unreachable from any test: no module in
    the tree defines two parsers today, so `if not found` and
    `if len(found) != 1` behave identically and a mutation between them
    survives. The rule it enforces is not decorative, though. Taking the first
    of several would silently skip a framework and report its controls as
    unchanged because no parser ever ran for them.

    Raises:
        ValueError: If the module defines no concrete BaseParser subclass, or
            defines more than one.
    """
    if len(candidates) != 1:
        raise ValueError(
            f"{module_name} defines {len(candidates)} concrete BaseParser "
            f"subclass(es), expected exactly 1: "
            f"{sorted(cls.__name__ for cls in candidates)}. Every parser module "
            f"must define one, or the rebuild silently skips a framework and "
            f"reports its controls as unchanged because it never ran one."
        )
    return candidates[0]


def _parser_classes() -> dict[str, type[BaseParser]]:
    """Every concrete parser class, keyed by framework_id."""
    classes: dict[str, type[BaseParser]] = {}
    for path in sorted(PARSERS_DIR.glob("parse_*.py")):
        module = importlib.import_module(f"parsers.{path.stem}")
        found = [
            value for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, BaseParser)
            and value is not BaseParser
            and value.__module__ == module.__name__
            and not inspect.isabstract(value)
        ]
        chosen = sole_parser_class(path.name, found)
        classes[chosen.framework_id] = chosen
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


def compare_artifacts(scratch: Path, live: Path) -> dict[str, list[str]]:
    """Which per-framework artifacts a fresh run reproduces byte for byte.

    The strongest available evidence that a framework outside the eleven did
    not move, because it covers every field rather than the five the digest
    reads. A digest comparison cannot see source_url, fetched_date, version or
    source_files, and a parser that re-pointed its own provenance would pass it.
    """
    result: dict[str, list[str]] = {"identical": [], "differing": [], "absent": []}
    for path in sorted(scratch.glob("*.json")):
        target = live / path.name
        if not target.exists():
            result["absent"].append(path.name)
        elif path.read_bytes() == target.read_bytes():
            result["identical"].append(path.name)
        else:
            result["differing"].append(path.name)
    return result


def diff_against_baseline(
    parsed: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Mapping[str, Sequence[str]],
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
    # framework. For the eleven it finds nothing, because the old record is a
    # stub and the new one is prose, and that is the honest answer rather than
    # a failure. It exists so `removed` means "content gone" for every
    # framework where a stub is not the before state.
    for old_key in sorted(surplus_old):
        framework = old_key.split(":", 1)[0]
        for digest in list(surplus_old[old_key]):
            match = next(
                (k for k in sorted(surplus_new)
                 if k != old_key
                 and k.split(":", 1)[0] == framework and surplus_new[k][digest]),
                None,
            )
            if match is None:
                continue
            surplus_new[match][digest] -= 1
            surplus_old[old_key][digest] -= 1
            report.renamed.append((old_key, match))

    # Counter.total() rather than sum(counter.values()), which mypy --strict
    # resolves to the bool overload of sum and rejects.
    for key, counter in sorted(surplus_new.items()):
        if not counter.total():
            continue
        if key in surplus_old and surplus_old[key].total():
            report.changed.append(key)
        else:
            report.added.append(key)
    for key, counter in sorted(surplus_old.items()):
        if counter.total() and not surplus_new.get(key, Counter()).total():
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
        SystemExit: On any undeclared record outside the eleven, any missing
            framework, or an unchanged count that is not exactly the
            pre-measured total.
    """
    allowed = EXPECTED_CHANGED_FRAMEWORK_IDS | EXPECTED_ADDED_FRAMEWORK_IDS
    touched = report.touched_frameworks()
    moved = report.changed + report.added + report.removed
    moved += [old for old, _ in report.renamed] + [new for _, new in report.renamed]
    # Keyed on the record, not on the framework. Exempting a whole framework
    # because two of its records are explained would hide the other 64.
    unexpected = sorted(
        key for key in moved
        if key.split(":", 1)[0] not in allowed and key not in DECLARED_MOVED_KEYS
    )
    if unexpected:
        raise SystemExit(
            f"these control records moved and their parsers were not touched: "
            f"{unexpected[:20]} ({len(unexpected)} in total). Their sources are "
            f"pinned and {EXPECTED_UNCHANGED_RECORDS} of their control records "
            f"were pre-measured as reproducing byte-identically, so a change "
            f"here is a defect this plan introduced, not a source change."
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


def _log_report(report: RebuildReport, parsed_count: int) -> None:
    logger.info(
        "rebuild: %d frameworks, %d unchanged records, %d changed, %d added, "
        "%d removed, %d renamed",
        parsed_count, report.unchanged, len(report.changed), len(report.added),
        len(report.removed), len(report.renamed),
    )
    for bucket, keys in (("changed", report.changed), ("added", report.added),
                         ("removed", report.removed)):
        counts: Counter[str] = Counter(key.split(":", 1)[0] for key in keys)
        for framework_id, count in sorted(counts.items()):
            logger.info("  %-8s %-26s %d", bucket, framework_id, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("build/rebuild"))
    parser.add_argument("--audit-dir", type=Path, default=Path("build/rebuild_audit"))
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", type=Path, default=None)
    parser.add_argument("--list-snapshots", action="store_true")
    parser.add_argument("--build-baseline", type=Path, default=None,
                        help="Rebuild BASELINE_PATH from a per-framework directory")
    parser.add_argument("--baseline-provenance", default="",
                        help="How that directory was assembled, recorded verbatim")
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
    if args.build_baseline is not None:
        baseline = build_baseline(corpus_from_framework_dir(args.build_baseline))
        existing = load_json(BASELINE_PATH) if BASELINE_PATH.exists() else {}
        # The description-only map is the only surviving record of what the
        # committed instrument measured. Keeping it makes "the change of
        # instrument did not change the answer" a live test rather than a
        # one-shot check that skips the moment the file is rewritten.
        if "sha256_of_description" in existing:
            baseline["sha256_of_description"] = existing["sha256_of_description"]
        # A scratch path means nothing to a later reader. The provenance string
        # is the recipe that rebuilds that directory from git.
        baseline["generated_from"] = (
            args.baseline_provenance or str(args.build_baseline)
        )
        atomic_write_json(baseline, BASELINE_PATH)
        logger.info("baseline: %d keys, %d records -> %s",
                    baseline["n_keys"], baseline["n_records"], BASELINE_PATH)
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
    _log_report(report, len(parsed))

    artifacts = compare_artifacts(args.scratch, PROCESSED_FRAMEWORKS_DIR)
    logger.info("artifacts: %d identical, %d differing, %d absent from %s",
                len(artifacts["identical"]), len(artifacts["differing"]),
                len(artifacts["absent"]), PROCESSED_FRAMEWORKS_DIR)

    atomic_write_json(
        {
            "changed": report.changed,
            "added": report.added,
            "removed": report.removed,
            "removed_classification": classify_removed(report.removed, parsed),
            "renamed": [list(pair) for pair in report.renamed],
            "unchanged": report.unchanged,
            "live_artifact_comparison": artifacts,
        },
        args.report,
    )
    assert_expected_frameworks_only(report)

    if args.commit:
        snapshot = snapshot_processed()
        logger.info("rollback: --restore %s", snapshot)
        sources = sorted(args.scratch.glob("*.json"))
        for source in sources:
            atomic_write_json(
                load_json(source), PROCESSED_FRAMEWORKS_DIR / source.name,
            )
        logger.info("committed %d artifact(s) into %s",
                    len(sources), PROCESSED_FRAMEWORKS_DIR)


if __name__ == "__main__":
    main()
