"""Fetch primary-source framework documents into data/raw/frameworks/.

    python -m scripts.fetch_frameworks --list
    python -m scripts.fetch_frameworks capec cwe
    python -m scripts.fetch_frameworks --all --verify

Nineteen of the thirty-one frameworks in the corpus reach it through
parsers/fetch_opencre.py, which carries the link graph and each section's name
but never the standard's text. Those frameworks are anchored on three-word
titles, and CAPEC alone is 41% of the training set. This script pulls the real
documents so a parser has something to parse.

data/raw/ is gitignored and immutable once written: parsers read it, nothing
edits it. Every download's hash is checked at record time against
Source.expected_sha256, a value hardcoded below and reviewed like any other
code change. A source whose bytes changed upstream without a matching code
change makes _record raise rather than silently re-baseline -- the whole
point of pinning a hash is that it stops meaning anything the moment the tool
that checks it can also rewrite it unattended. --accept-new-hash is the one
escape hatch, and it does not touch this file: it lets a single run through
after an operator has looked at the diff and decided the new bytes are fine,
and the constant below still has to be updated by hand afterward or the next
run raises again.

Re-fetching an existing file requires --force. Re-baselining a changed hash
requires --accept-new-hash. Neither implies the other.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import requests

from tract.config import FRAMEWORK_LICENSES, PROCESSED_DIR, RAW_FRAMEWORKS_DIR
from tract.io import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Committed, unlike data/raw/ itself. The point of recording a sha256 is to make
# an upstream change visible later and on another machine, which a gitignored
# manifest cannot do.
MANIFEST_PATH: Final[Path] = PROCESSED_DIR / "framework_sources.json"
TIMEOUT_S: Final[int] = 300
CHUNK_BYTES: Final[int] = 1 << 16


class SourceHashMismatch(Exception):
    """Raised when a downloaded or on-disk file does not match its pinned hash.

    This is the control that makes trust-on-first-use mean something. A
    source's expected_sha256 is a value committed to this file by a human who
    looked at a diff, not a value the script derives from what it just
    downloaded. Catching this and moving on without --accept-new-hash defeats
    the reason it exists.
    """


@dataclass(frozen=True)
class Source:
    """One downloadable primary source.

    Two or more Source entries may share a framework_id (see biml below) when
    OpenCRE's links for that framework span more than one primary document.
    They land in the same data/raw/frameworks/<framework_id>/ directory under
    their own filenames.
    """

    framework_id: str
    filename: str
    # None marks a source this script cannot fetch itself -- gated behind
    # registration or otherwise hand-delivered. fetch() requires the file to
    # already be on disk and never attempts a network call for it.
    url: str | None
    note: str
    # Links in hub_links_curated.jsonl that gain prose from this source.
    training_links: int
    # The licence this document is published under, read off the staged
    # artifact itself: an SPDX identifier where one applies, a short quotation
    # of the source's own notice where none does, and "UNDETERMINED" where the
    # artifact states no terms at all. Required, with no default, so a new
    # source cannot be added without answering the question. This repository is
    # CC0, an affirmative grant, so "we never looked" is not a safe default.
    # See NOTICE and tract.config.FRAMEWORK_LICENSES.
    license: str
    # Hardcoded and reviewed, not derived from a prior run's manifest. See the
    # module docstring for why this has to live in code rather than in the
    # generated manifest. None means this source has never been through an
    # accepted fetch yet -- trust-on-first-use is still permitted, but every
    # fetch after the first requires this to be set and to match.
    expected_sha256: str | None = None
    # Per-source request headers. ETSI's edge rejects a bare curl/requests
    # user-agent with 403 and accepts a browser one; nothing else needs this.
    headers: dict[str, str] | None = None
    # GitHub sources only: the commit the url's archive/<sha>.zip is pinned
    # to, resolved once via the GitHub API and frozen here. Recorded
    # separately from the url so the pin is auditable without parsing it back
    # out of a path string, and checked against the url below at import time.
    resolved_commit_sha: str | None = None


SOURCES: Final[tuple[Source, ...]] = (
    Source(
        "capec", "capec_latest.xml",
        "https://capec.mitre.org/data/xml/capec_latest.xml",
        "MITRE CAPEC attack patterns, full descriptions", 1799,
        license=FRAMEWORK_LICENSES["capec"],
        expected_sha256="70279a2dff0cb0ad79e546adb07828335a704ad5210e047e09e986172fc9e34d",
    ),
    Source(
        "cwe", "cwec_latest.xml.zip",
        "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
        "MITRE CWE weaknesses, full descriptions", 613,
        license=FRAMEWORK_LICENSES["cwe"],
        expected_sha256="3976f599e5e5200219a3108bb896d06e2a88fbb293369e1883cb423a5e9d7d50",
    ),
    Source(
        "owasp_cheat_sheets", "cheatsheets.zip",
        "https://github.com/OWASP/CheatSheetSeries/archive/"
        "07111ee754e832e335377ac64fd0f8f848d9029c.zip",
        "OWASP Cheat Sheet Series markdown. Was pinned to "
        "archive/refs/heads/master.zip, a moving ref; now pinned to the "
        "commit that was master's HEAD on 2026-08-15. Its content hash "
        "changes as a result of the re-pin -- expected.", 391,
        license=FRAMEWORK_LICENSES["owasp_cheat_sheets"],
        expected_sha256="f2ede0212f2550c578d9ce65a71185c2c4937528b8496ca1d8e9611ff9e068f3",
        resolved_commit_sha="07111ee754e832e335377ac64fd0f8f848d9029c",
    ),
    Source(
        "nist_800_53", "nist_800_53_catalog.json",
        "https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/"
        "SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json",
        "NIST SP 800-53 rev5 OSCAL catalog, full control text", 300,
        license=FRAMEWORK_LICENSES["nist_800_53"],
        expected_sha256="01f37cf90ea99d92242c936cbfbdebcc338eef1f71454e2acac36cc56e9bc062",
    ),
    Source(
        "asvs", "asvs.zip",
        "https://github.com/OWASP/ASVS/archive/"
        "cdc8a0f68ac2a9f9e3739266acdac0e4a98badee.zip",
        "OWASP ASVS requirements. Was pinned to "
        "archive/refs/heads/master.zip, a moving ref; now pinned to the "
        "commit that was master's HEAD on 2026-08-15. Its content hash "
        "changes as a result of the re-pin -- expected.", 277,
        license=FRAMEWORK_LICENSES["asvs"],
        expected_sha256="b6c05edea5b9da9762b997e248da2246d06eee50c86c9864daed2599215585c3",
        resolved_commit_sha="cdc8a0f68ac2a9f9e3739266acdac0e4a98badee",
    ),
    Source(
        "nist_ai_100_2", "nist_ai_100_2_e2023.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf",
        "NIST AI 100-2e2023 adversarial ML taxonomy. LOFO eval fold.", 45,
        license=FRAMEWORK_LICENSES["nist_ai_100_2"],
        expected_sha256="d1086f53a1634d6787c59510c117b22bb7e1a242f920d2830a0d334058b0cb78",
    ),
    Source(
        "owasp_ml_top10", "owasp_ml_top10.zip",
        "https://github.com/OWASP/www-project-machine-learning-security-top-10/"
        "archive/f0b0ed240c4d367ce483ab2ed2edf3563a5d29b9.zip",
        "OWASP ML Security Top 10. LOFO eval fold. Was pinned to "
        "archive/refs/heads/master.zip, a moving ref; now pinned to the "
        "commit that was master's HEAD on 2026-08-15. Its content hash "
        "changes as a result of the re-pin -- expected.", 10,
        license=FRAMEWORK_LICENSES["owasp_ml_top10"],
        expected_sha256="42d169c33943e5c3168a6ee0d9a1f76739cca2f4606dcfba0bd66f1377e04052",
        resolved_commit_sha="f0b0ed240c4d367ce483ab2ed2edf3563a5d29b9",
    ),
    Source(
        "dsomm", "dsomm_data.zip",
        "https://github.com/devsecopsmaturitymodel/DevSecOps-MaturityModel-data/"
        "archive/ca6e5174aed85a7bdbb845cb7431fec21c224d06.zip",
        "DevSecOps Maturity Model data repo (branch main), pinned to a "
        "commit. Highest-value single addition in this batch.", 214,
        license=FRAMEWORK_LICENSES["dsomm"],
        expected_sha256="a6d773129591d59e7c0757651142c39a341400333f40c1555fb2481ae89f2c66",
        resolved_commit_sha="ca6e5174aed85a7bdbb845cb7431fec21c224d06",
    ),
    Source(
        "wstg", "wstg.zip",
        "https://github.com/OWASP/wstg/archive/"
        "95ce6cfe5d463bbde88aa52b3171b123a1ea9ada.zip",
        "OWASP Web Security Testing Guide (branch master), pinned to a commit.", 118,
        license=FRAMEWORK_LICENSES["wstg"],
        expected_sha256="e093f1648fbf4195f2a8fccac4f80315fb6b6281af85aa557edb34d0f9c58b33",
        resolved_commit_sha="95ce6cfe5d463bbde88aa52b3171b123a1ea9ada",
    ),
    Source(
        "samm", "samm_core.zip",
        "https://github.com/owaspsamm/core/archive/"
        "bc2b5474ab248effbc357c389bec372b0f5e200f.zip",
        "OWASP SAMM core model. owaspsamm/core has no master branch -- "
        "the repo's default branch is develop, which is what this is "
        "pinned to. Deviates from a master-branch instruction that did "
        "not match the live repository.", 30,
        license=FRAMEWORK_LICENSES["samm"],
        expected_sha256="16eb608b70bad3039b14ca4e3f300893d29bbc4205c737ac07fcbdfb4f7493a6",
        resolved_commit_sha="bc2b5474ab248effbc357c389bec372b0f5e200f",
    ),
    Source(
        "owasp_top10_2021", "owasp_top10_2021.zip",
        "https://github.com/OWASP/Top10/archive/"
        "66ebc4798d2ca72973967a20264bdeb70dcf0a13.zip",
        "OWASP Top 10 2021 (branch master), pinned to a commit.", 17,
        license=FRAMEWORK_LICENSES["owasp_top10_2021"],
        expected_sha256="7f4747a7d7958d58ae3a4c7f7329740b9363c4788655bc3f28da8fdbedf48b5d",
        resolved_commit_sha="66ebc4798d2ca72973967a20264bdeb70dcf0a13",
    ),
    Source(
        "owasp_proactive_controls", "owasp_proactive_controls.zip",
        "https://github.com/OWASP/www-project-proactive-controls/archive/"
        "4f5cb1081b4253bbccb314ef7855a1430fec8571.zip",
        "OWASP Proactive Controls (branch master), pinned to a commit. "
        "Restores links dropped for lack of a primary source.", 76,
        license=FRAMEWORK_LICENSES["owasp_proactive_controls"],
        expected_sha256="6db1aafd6ecd758f05cf6b4133ec7085eb95016ec41afc5f462b4683c603b19d",
        resolved_commit_sha="4f5cb1081b4253bbccb314ef7855a1430fec8571",
    ),
    Source(
        "enisa", "enisa_securing_ml_algorithms.pdf",
        "https://www.enisa.europa.eu/sites/default/files/publications/"
        "ENISA%20Report%20-%20Securing%20Machine%20Learning%20Algorithms.pdf",
        "ENISA Securing Machine Learning Algorithms report.", 68,
        license=FRAMEWORK_LICENSES["enisa"],
        expected_sha256="4de967bbdf92a01339ae449b7d305b8ff266d7f16ed0a7d92a711ca20e20f087",
    ),
    Source(
        "nist_ssdf", "nist_sp_800_218.pdf",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf",
        "NIST SP 800-218 Secure Software Development Framework.", 46,
        license=FRAMEWORK_LICENSES["nist_ssdf"],
        expected_sha256="617746e553a9e2da49bfbd4eef0dfc3094758a39b869314e4173ac36605cde22",
    ),
    Source(
        "nist_800_63", "sp800_63b.html",
        "https://pages.nist.gov/800-63-3/sp800-63b.html",
        "NIST SP 800-63B Digital Identity Guidelines, authentication. "
        "Restores links dropped for lack of a primary source. "
        "REVISION 3, NOT 4, AND THE DIFFERENCE IS LOAD-BEARING. OpenCRE's 79 "
        "links carry 25 bare section numbers from revision 3, and revision 4 "
        "renumbered the document. Measured: revision 3B contains 24 of the 25, "
        "revision 4B contains 0 of 25, and the one miss is 'are g', a parsing "
        "artifact in OpenCRE's own data rather than a real section. Fetching "
        "revision 4 would leave every one of those links unjoinable while "
        "looking like a successful fetch. "
        "DELIBERATELY UNPINNED: pages.nist.gov sits behind Cloudflare, "
        "which injects a per-response random bot-challenge token "
        "(window.__CF$cv$params, a fresh nonce and timestamp) into the "
        "HTML body on every fetch. Two fetches of the identical document "
        "seconds apart produced two different sha256 hashes with a "
        "one-line diff confined to that injected script tag. Pinning a "
        "hash here would make --accept-new-hash routine rather than an "
        "alert, which is worse than no pin.", 79,
        license=FRAMEWORK_LICENSES["nist_800_63"],
    ),
    Source(
        "etsi", "etsi_gr_sai005_v010101p.pdf",
        "https://www.etsi.org/deliver/etsi_gr/SAI/001_099/005/"
        "01.01.01_60/gr_SAI005v010101p.pdf",
        "ETSI GR SAI 005 Securing AI Problem Statement. The edge in front "
        "of etsi.org 403s a bare requests/curl user-agent and 200s a "
        "browser one, hence the header.", 36,
        license=FRAMEWORK_LICENSES["etsi"],
        expected_sha256="46c2b6b880928ffe2e763fbd6e0d0660a0aa7de0ff0071f5e0694582d91d5622",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        },
    ),
    # BIML's 21 curated links carry two distinct id prefixes -- "BIML-78(2020): "
    # and "BIML-24(LLM): " -- naming two different BIML reports directly, plus
    # a set of unprefixed legacy ids that predate the second report and, on
    # content inspection, resolve almost entirely to the 2020 document. No
    # single PDF covers all of OpenCRE's BIML anchors; both are required. See
    # source-structures.md for the anchor-by-anchor evidence.
    Source(
        "biml", "ara.pdf",
        "https://berryvilleiml.com/results/ara.pdf",
        "BIML-78(2020): An Architectural Risk Analysis of Machine Learning "
        "Systems (Jan 2020, 42pp). Covers the BIML-78(2020)-prefixed "
        "anchors plus most unprefixed legacy anchors.", 15,
        # ara.pdf page 1: "licensed under the Creative Commons
        # Attribution-Share Alike 3.0 License". The 2024 LLM report below
        # moved to 4.0, so BIML cannot carry one framework-level licence.
        license="CC-BY-SA-3.0",
        expected_sha256="247d7f06d8c768cc734dc84ab7004c6e4d645e91911af61002fd1743807ef312",
    ),
    Source(
        "biml", "BIML-LLM24.pdf",
        "https://berryvilleiml.com/results/BIML-LLM24.pdf",
        "BIML-24(LLM): An Architectural Risk Analysis of Large Language "
        "Models (Jan 2024, 28pp). Covers the BIML-24(LLM)-prefixed anchors "
        "plus two unprefixed anchors unique to this document.", 6,
        # BIML-LLM24.pdf: "licensed under the Creative Commons
        # Attribution-ShareAlike 4.0 International License".
        license="CC-BY-SA-4.0",
        expected_sha256="1a41ba1a9218e6aecdcab46d2cc6cf8a3b99f6cc1c98a3683bf3a6e4964e955f",
    ),
    Source(
        "csa_ccm", "CCMv4.1.0-generated_at_2026_01_13.xlsx",
        None,
        "CSA Cloud Controls Matrix v4.1.0. CSA gates this behind "
        "registration; it cannot be fetched by this script and must be "
        "staged on disk manually. Not to be confused with csa_aicm (AI "
        "Controls Matrix), a different framework with zero CRE links.", 29,
        license=FRAMEWORK_LICENSES["csa_ccm"],
        expected_sha256="5e721628c8ab297bdbd355afa4c01699971fcbb9cb16802ccb9d42c7176ab32b",
    ),
)

# Fail at import time, not at fetch time, if a pin is internally inconsistent:
# a resolved_commit_sha that doesn't actually appear in its own url is a
# copy-paste error in this file, not a runtime condition.
for _source in SOURCES:
    # An empty licence is the state this field exists to make impossible.
    # "UNDETERMINED" is a legitimate answer and says so out loud; a blank
    # string says nothing and reads as "not applicable" to the next person.
    if not _source.license.strip():
        raise ValueError(
            f"{_source.framework_id}/{_source.filename}: license is empty. "
            f"Record the licence from the source's own notice, or "
            f"'UNDETERMINED' if the staged artifact states no terms."
        )
    if _source.resolved_commit_sha is not None:
        _expected_fragment = f"archive/{_source.resolved_commit_sha}.zip"
        if _source.url is None or _expected_fragment not in _source.url:
            raise ValueError(
                f"{_source.framework_id}/{_source.filename}: resolved_commit_sha "
                f"{_source.resolved_commit_sha!r} is not reflected in url "
                f"{_source.url!r}"
            )
del _source

BY_ID: Final[dict[str, tuple[Source, ...]]] = {}
for _s in SOURCES:
    BY_ID[_s.framework_id] = BY_ID.get(_s.framework_id, ()) + (_s,)
del _s


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, dict[str, dict[str, str]]]:
    """Load the committed manifest: framework_id -> filename -> record.

    Nested by filename because biml has two sources sharing one
    framework_id; a flat framework_id -> record mapping would let the second
    fetch silently clobber the first's entry.
    """
    if not MANIFEST_PATH.exists():
        return {}
    data = load_json(MANIFEST_PATH)
    return data.get("sources", {}) if isinstance(data, dict) else {}


def fetch(source: Source, force: bool = False, accept_new_hash: bool = False) -> Path:
    """Download (or verify) one source and record its hash. Returns the path.

    Raises:
        FileNotFoundError: source.url is None and no file is staged at the
            target path. This script never invents bytes for a locally-
            supplied source.
        SourceHashMismatch: the file's sha256 does not match
            source.expected_sha256 and accept_new_hash is False.
    """
    target_dir = RAW_FRAMEWORKS_DIR / source.framework_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.filename

    if source.url is None:
        if not target.exists():
            raise FileNotFoundError(
                f"{source.framework_id}/{source.filename}: no url and no file "
                f"staged at {target}. This source must be placed on disk "
                "manually before fetch() can record it."
            )
        logger.info("%s/%s is locally supplied, no download attempted",
                    source.framework_id, source.filename)
        _record(source, target, accept_new_hash)
        return target

    if target.exists() and not force:
        # Still record it. The manifest has to describe what is on disk, or a
        # file fetched before the manifest existed stays permanently unhashed
        # and its drift undetectable.
        logger.info("%s already present (%s); recording hash, use --force to "
                    "re-fetch", source.framework_id, target.name)
        _record(source, target, accept_new_hash)
        return target

    logger.info("Fetching %s from %s", source.framework_id, source.url)
    response = requests.get(
        source.url, timeout=TIMEOUT_S, stream=True, headers=source.headers,
    )
    response.raise_for_status()

    # Write to a temp path then rename, so an interrupted download cannot leave
    # a truncated file that a parser would happily read.
    temp = target.with_suffix(target.suffix + ".part")
    try:
        with open(temp, "wb") as handle:
            for chunk in response.iter_content(CHUNK_BYTES):
                handle.write(chunk)
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

    _record(source, target, accept_new_hash)
    logger.info("Wrote %s (%d bytes)", target, target.stat().st_size)
    return target


def _record(source: Source, target: Path, accept_new_hash: bool = False) -> None:
    """Hash the file on disk, check it against the pinned baseline, and write
    the observation into the committed manifest.

    source.expected_sha256 is the trust baseline, hardcoded in this file and
    reviewed like any other code change -- not the manifest's previous entry.
    A manifest can be regenerated by anyone with fetch access; a baseline
    that lives there is not a baseline, it is a cache of whatever ran last.

    Raises:
        SourceHashMismatch: expected_sha256 is set, the observed digest
            differs, and accept_new_hash is False.
    """
    digest = _sha256(target)

    if source.expected_sha256 is not None and digest != source.expected_sha256:
        if not accept_new_hash:
            raise SourceHashMismatch(
                f"{source.framework_id}/{source.filename}: expected sha256 "
                f"{source.expected_sha256[:16]}... got {digest[:16]}.... "
                "The upstream source changed. Re-run with --accept-new-hash "
                "to accept this fetch, then update Source.expected_sha256 in "
                "scripts/fetch_frameworks.py to re-pin it."
            )
        logger.warning(
            "%s/%s changed upstream and was accepted via --accept-new-hash: "
            "%s -> %s. Source.expected_sha256 in scripts/fetch_frameworks.py "
            "is now STALE and must be updated by hand.",
            source.framework_id, source.filename,
            source.expected_sha256[:16], digest[:16],
        )
    elif source.expected_sha256 is None:
        logger.info(
            "%s/%s has no pinned expected_sha256 yet (trust-on-first-use); "
            "observed %s. Pin it in scripts/fetch_frameworks.py.",
            source.framework_id, source.filename, digest[:16],
        )

    manifest = _load_manifest()
    existing = manifest.get(source.framework_id, {})
    # A framework_id entry written under the pre-migration flat schema (one
    # record directly, no filename nesting) has string values, not dicts.
    # Discard it wholesale rather than merge into it: mixing the old flat
    # keys ("url", "sha256", ...) with the new filename-keyed ones under the
    # same object corrupts both, and verify() would try to stat a file
    # literally named "sha256".
    if not all(isinstance(value, dict) for value in existing.values()):
        existing = {}
    manifest[source.framework_id] = existing
    framework_entry = existing
    record: dict[str, str] = {
        "url": source.url if source.url is not None else "",
        "filename": source.filename,
        "sha256": digest,
        "expected_sha256": source.expected_sha256 or "",
        "bytes": str(target.stat().st_size),
        "fetched_date": date.today().isoformat(),
        "note": source.note,
        # Travels with the manifest so a reader of data/processed/ can see the
        # terms without opening this file or the archive it came from.
        "license": source.license,
    }
    if source.resolved_commit_sha is not None:
        record["resolved_commit_sha"] = source.resolved_commit_sha
    framework_entry[source.filename] = record
    atomic_write_json({"sources": manifest}, MANIFEST_PATH)


def verify() -> int:
    """Re-hash everything in the manifest. Returns the number of mismatches."""
    manifest = _load_manifest()
    if not manifest:
        logger.warning("No manifest at %s; nothing to verify", MANIFEST_PATH)
        return 0
    bad = 0
    for framework_id, files in sorted(manifest.items()):
        for filename, entry in sorted(files.items()):
            path = RAW_FRAMEWORKS_DIR / framework_id / filename
            if not path.exists():
                logger.error("%s/%s MISSING: %s", framework_id, filename, path)
                bad += 1
                continue
            actual = _sha256(path)
            status = "ok" if actual == entry["sha256"] else "MISMATCH"
            if actual != entry["sha256"]:
                bad += 1
            logger.info("%-20s %-40s %s  %s", framework_id, filename, status, actual[:16])
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch primary framework sources")
    parser.add_argument("frameworks", nargs="*", help="framework ids to fetch")
    parser.add_argument("--all", action="store_true", help="fetch every source")
    parser.add_argument("--list", action="store_true", help="list known sources")
    parser.add_argument("--force", action="store_true", help="re-fetch if present")
    parser.add_argument("--accept-new-hash", action="store_true",
                        help="accept a source whose hash no longer matches "
                             "Source.expected_sha256; the constant in this "
                             "file must still be updated by hand afterward")
    parser.add_argument("--verify", action="store_true",
                        help="re-hash what is on disk against the manifest")
    args = parser.parse_args()

    if args.list:
        print(f"{'framework_id':<22}{'filename':<38}{'links':>7}  source")
        for source in SOURCES:
            print(f"{source.framework_id:<22}{source.filename:<38}"
                  f"{source.training_links:>7}  {source.note}")
        return 0

    if args.verify and not (args.all or args.frameworks):
        return 1 if verify() else 0

    selected: tuple[Source, ...]
    if args.all:
        selected = SOURCES
    else:
        unknown = [n for n in args.frameworks if n not in BY_ID]
        if unknown:
            raise SystemExit(f"Unknown framework ids: {unknown}. Try --list.")
        picked: list[Source] = []
        for name in args.frameworks:
            picked.extend(BY_ID[name])
        selected = tuple(picked)
    if not selected:
        raise SystemExit("Nothing selected. Pass framework ids, --all, or --list.")

    failures: list[str] = []
    for source in selected:
        try:
            fetch(source, force=args.force, accept_new_hash=args.accept_new_hash)
        except Exception as exc:  # noqa: BLE001 - report all, stop for none
            logger.error("%s/%s FAILED: %s", source.framework_id, source.filename, exc)
            failures.append(f"{source.framework_id}/{source.filename}")

    if args.verify:
        verify()
    if failures:
        logger.error("Failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
