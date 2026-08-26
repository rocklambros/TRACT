"""Register an SSH public key with the RunPod account, without dropping others.

    python -m scripts.register_runpod_key --dry-run
    python -m scripts.register_runpod_key

RunPod stores every authorized public key in a single newline-separated
``pubKey`` field on the account. A naive write replaces that whole field, which
silently revokes every other key on the account, including ones belonging to
machines this repo knows nothing about. This account already carries a key
labelled "lambda" whose private half is not on this machine. So this script
reads the field, appends only when the key is absent, and verifies afterwards.
It never removes a key.

The private half is never read, never transmitted, and never logged.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

from scripts.phase0.runpod_provision import _gql

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PUBKEY: Final[Path] = Path.home() / ".ssh" / "tract_runpod.pub"

_QUERY: Final[str] = "query { myself { id pubKey } }"
_MUTATION: Final[str] = (
    "mutation updateUserSettings($input: UpdateUserSettingsInput) { "
    "  updateUserSettings(input: $input) { id pubKey } "
    "}"
)


def key_material(line: str) -> str:
    """The base64 body of an SSH public key line.

    Comparison is on the material alone. The comment field is free text that
    differs between machines, so matching on the whole line would register the
    same key twice under two names.
    """
    parts = line.split()
    return parts[1] if len(parts) > 1 else ""


def parse_keys(blob: str) -> list[str]:
    return [line.strip() for line in (blob or "").splitlines() if line.strip()]


def describe(line: str) -> str:
    parts = line.split()
    algorithm = parts[0] if parts else "?"
    comment = parts[2] if len(parts) > 2 else "(no comment)"
    return f"{algorithm} ...{key_material(line)[-24:]} {comment}"


def read_public_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"No public key at {path}. Generate the run-scoped pair in a "
            f"terminal:\n  ssh-keygen -t ed25519 -f {path.with_suffix('')} "
            f"-N '' -C tract-runpod"
        )
    line = path.read_text(encoding="utf-8").strip()
    if not line.startswith(("ssh-", "ecdsa-", "sk-")):
        raise ValueError(f"{path} does not look like an SSH public key.")
    if "PRIVATE KEY" in line:
        raise ValueError(f"{path} contains a PRIVATE key. Refusing to transmit it.")
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description="Register an SSH key with RunPod")
    parser.add_argument("--pubkey", type=Path, default=DEFAULT_PUBKEY)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args()

    new_key = read_public_key(args.pubkey)
    logger.info("Local key:  %s", describe(new_key))

    account = _gql(_QUERY).get("myself") or {}
    existing = parse_keys(account.get("pubKey", ""))
    logger.info("Account %s currently has %d key(s):", account.get("id"), len(existing))
    for line in existing:
        logger.info("   %s", describe(line))

    if key_material(new_key) in {key_material(line) for line in existing}:
        logger.info("Already registered. Nothing to do.")
        return 0

    merged = existing + [new_key]
    logger.info("Will APPEND, leaving %d existing key(s) in place -> %d total",
                len(existing), len(merged))

    if args.dry_run:
        logger.info("Dry run; account not modified.")
        return 0

    result = _gql(_MUTATION, {"input": {"pubKey": "\n".join(merged)}})
    updated = parse_keys((result.get("updateUserSettings") or {}).get("pubKey", ""))

    # Verify against the account rather than trusting the mutation's echo.
    confirmed = parse_keys((_gql(_QUERY).get("myself") or {}).get("pubKey", ""))
    materials = {key_material(line) for line in confirmed}

    if key_material(new_key) not in materials:
        logger.error("Key is not present after the write. Account unchanged?")
        return 1
    missing = [line for line in existing if key_material(line) not in materials]
    if missing:
        # The failure this script exists to prevent.
        logger.error("A pre-existing key was DROPPED: %s",
                     [describe(line) for line in missing])
        return 1

    logger.info("Registered. Account now has %d key(s), none removed.", len(confirmed))
    logger.debug("Mutation echoed %d key(s)", len(updated))
    return 0


if __name__ == "__main__":
    sys.exit(main())
