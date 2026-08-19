"""Run one LLM judge over the blind ceiling study, using the human's prompt.

    python -m scripts.run_panel --model zai-org/GLM-5.2               # dry run
    python -m scripts.run_panel --model zai-org/GLM-5.2 --execute
    python -m scripts.run_panel --model zai-org/GLM-5.2 --execute --closed-book

Three model families answer the same question the human annotator answered,
from the same prompt in claudedocs/ceiling-study-runbook.md, with no sight of
the human's answers or of the key. The point is to separate two explanations
for CAPEC's alpha-1 of 0.181: OpenCRE's CAPEC links are poor, or one human's
reading of CAPEC is idiosyncratic.

Dry run is the default and spends nothing: it resolves the route, renders the
exact prompts, counts tokens, and prints a per-model cost estimate. Spending
requires --execute, so an accidental invocation cannot bill anything.

Writes results/ceiling_study/answers_panel_<slug>.json in the schema
scripts/score_ceiling_study.py reads, plus a `run` block recording the exact
model id, route, served snapshot, temperature, reasoning effort, token usage,
and cost. An unpinned judge is an unreproducible judge.

Completed batches are checkpointed as they land, and --resume reuses them, so
a run that dies partway through does not have to be paid for twice.

The --closed-book mode drops the hub reference and asks the model to recall
OpenCRE's mapping from memory. That is the contamination probe, not an
annotation run, and it writes to a separate file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, TypedDict

import requests

from tract.config import (
    CEILING_STUDY_DIR,
    CEILING_STUDY_MAX_ACCEPTABLE_HUBS,
    CEILING_STUDY_SEED,
    EXIT_OFFLINE,
    EXIT_USER_ERROR,
    PANEL_BATCH_SIZE,
    PANEL_MAX_RETRIES,
    PANEL_MAX_TOKENS,
    PANEL_MODEL_IDS,
    PANEL_MODELS,
    PANEL_OPENROUTER_PROVIDER_PIN,
    PANEL_PRICING_USD_PER_MTOK,
    PANEL_REASONING_EFFORT,
    PANEL_RETRY_BASE_DELAY_S,
    PANEL_ROUTES,
    PANEL_TEMPERATURE,
    PANEL_TIMEOUT_S,
)
from tract.io import atomic_write_json, load_json
from tract.panel import (
    HUB_ID_RE,
    PanelAnswer,
    extract_json_array,
    model_slug,
    parse_hub_reference,
    parse_hub_names,
    parse_judge_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The item fields the annotator sees. framework_id is included because the
# runbook prompt explicitly tells the annotator to ignore it, and removing it
# here would be a different prompt than the human answered.
_ITEM_FIELDS: Final[tuple[str, ...]] = (
    "item_index",
    "framework_id",
    "framework_name",
    "control_id",
    "control_title",
    "control_text",
)

# Verbatim from claudedocs/ceiling-study-runbook.md, "The prompt". Two
# placeholders are substituted. Deliberately not improved: the panel has to
# answer the question the human answered, or the comparison is between two
# different tasks.
_ANNOTATOR_PROMPT: Final[str] = """\
You are a senior cybersecurity domain expert performing a blind annotation task.
Thirty years of practice across security architecture, GRC, and control
frameworks. You are annotating independently: you have not seen any model's
prediction for these items and you must not try to infer one.

# Task

For each security control below, decide which CRE hub it belongs to.

CRE (Common Requirement Enumeration) is a taxonomy of 522 hubs arranged in a
hierarchy under 5 top-level branches. It is the coordinate system that lets a
control from one framework be compared with a control from another. Assigning a
control to a hub is a claim that the hub is what the control is fundamentally
about.

# Reference

The complete hub taxonomy follows. Every hub has an id, a name, a hierarchy
path, and, for the 400 leaf hubs, an expert description stating what the hub
covers and, usually, what it explicitly does NOT cover. The exclusions are
load-bearing: read them.

<hub_reference>
{hub_reference}
</hub_reference>

# Items

<items>
{items}
</items>

# Decision procedure

Work each item independently and in the order given. For each one:

1. Read `control_text` in full. It is the control's own words. Ignore
   `framework_id` when deciding: a control about access logging is the same
   concept whether NIST or OWASP wrote it, and letting the framework steer you
   introduces exactly the bias this study exists to measure.

2. Identify what the control is fundamentally ABOUT. Not what it mentions in
   passing, not what it would be filed under in its own framework's structure,
   but the security concept a practitioner would say it addresses. A control
   that says "encrypt backups using approved algorithms" is about backup
   protection or about cryptographic standards depending on which the sentence
   is actually constraining. Decide which.

3. Search the hub reference for candidates. Match on the hub's description and
   its stated exclusions, not on keyword overlap with its name. Two hubs often
   share vocabulary and differ in scope, and the descriptions say so explicitly.

4. Choose `primary_hub_id`: the single hub you would defend as the best
   assignment. Exactly one. If two feel equally good, pick the one whose
   description's exclusions do NOT rule the control out, then say so in `notes`.

5. Choose `acceptable_hub_ids`: up to 5 hub ids, including the primary, that
   you would accept as correct if another expert chose them. This is not a
   ranked list and not a hedge. Include a hub only if you would genuinely
   defend it. If the control is unambiguous, a single-element list containing
   only the primary is the correct answer.

6. Set `confidence`:
   - "high": the control maps cleanly, the hub description fits, exclusions
     do not bite.
   - "medium": defensible but a reasonable expert could choose differently.
   - "low": the control is vague, spans several hubs, or the taxonomy has no
     good home for it.

7. `notes`: optional. Use it when the item is contested, when the control seems
   to belong to no hub, or when you rejected a near-miss for a specific reason.
   One sentence. Leave empty otherwise.

# Rules

- Choose only from hub ids present in the reference. Never invent one.
- `primary_hub_id` must appear in the reference. `acceptable_hub_ids` may be
  empty only if you also leave `primary_hub_id` empty, which means "I did not
  do this item" and is scored as incomplete rather than wrong.
- Do not skip items. Do not reorder them.
- Do not reason about which hub a model would pick, or which is most common in
  the corpus. Both defeat the measurement.
- If the correct answer seems obvious, say so with "high" confidence and move
  on. Deliberation is not a virtue here; agreement is what is being measured.
- Never guess at what the answer key contains. There is a key and you have not
  seen it.

# Output

Return a JSON array, one object per item, in the same order, and nothing else.
No prose before or after.

[
  {{
    "item_index": 1,
    "primary_hub_id": "028-727",
    "acceptable_hub_ids": ["028-727", "155-155"],
    "confidence": "high",
    "notes": ""
  }}
]"""

# The contamination probe. No hub reference: the only way to answer is to
# have memorised OpenCRE's published mapping. Chance is 1 in 522.
# The exposure control. A judge that emits no hub id in the closed-book arm
# has either never memorised OpenCRE's mappings or is simply obeying the
# instruction not to guess, and the mapping arm alone cannot tell those
# apart. This asks for the NAME of a hub whose id is given, which is a much
# weaker memory than a mapping and is checkable against hub_reference.md. A
# judge that can name hubs but cannot recall mappings has seen the taxonomy
# and not the mappings. A judge that can do neither never saw OpenCRE at
# all, and contamination is moot for it.
_HUB_NAME_PROMPT: Final[str] = """\
You are being asked to recall published data, not to reason about it.

OpenCRE (opencre.org) publishes the Common Requirement Enumeration, a public
taxonomy of security topics. Each CRE hub has an id of the form NNN-NNN and a
short name, for example "Cryptography" or "Session management".

For each hub id below, state the hub's name from memory. You are NOT given the
taxonomy. If you do not recall the hub, return an empty string for
`recalled_name` and set `recall` to "none".

Set `recall` to "confident", "vague", or "none".

<items>
{items}
</items>

Return a JSON array, one object per id, in the same order, and nothing else.
No prose before or after.

[
  {{
    "hub_id": "028-727",
    "recalled_name": "CSRF protection",
    "recall": "confident"
  }}
]"""

_CLOSED_BOOK_PROMPT: Final[str] = """\
You are being asked to recall published data, not to reason about it.

OpenCRE (opencre.org) is a public catalogue that links security controls from
many frameworks to Common Requirement Enumeration (CRE) hubs. Its mappings are
published on opencre.org and in its public GitHub repository. Each CRE hub has
an id of the form NNN-NNN, for example 028-727.

For each control below, state the CRE hub id that OpenCRE links it to, from
memory. You are NOT given the hub taxonomy and you are NOT being asked which
hub you think fits best. The question is only: which hub id does OpenCRE's
published mapping actually record for this control?

If you do not recall OpenCRE's mapping for a control, return an empty string
for `recalled_hub_id` and set `recall` to "none". Guessing a plausible-looking
id is worse than admitting you do not know, because this task measures recall
and a guess is indistinguishable from a memory only if you hide it.

Set `recall` to:
  - "confident": you specifically remember OpenCRE's mapping for this control.
  - "vague": you have some impression but would not stake anything on it.
  - "none": you do not recall a mapping. Leave `recalled_hub_id` empty.

<items>
{items}
</items>

Return a JSON array, one object per item, in the same order, and nothing else.
No prose before or after.

[
  {{
    "item_index": 1,
    "recalled_hub_id": "028-727",
    "recall": "confident"
  }}
]"""


class BatchUsage(TypedDict):
    """Token accounting for a single request, including failed attempts."""

    batch_index: int
    attempt: int
    ok: bool
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    served_model: str
    served_provider: str
    billed_cost_usd: float
    response_id: str
    finish_reason: str
    latency_s: float
    error: str


def _read_pass(entry: str) -> str:
    """First line of a pass entry, or empty if it does not exist."""
    try:
        result = subprocess.run(
            ["pass", entry],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else ""


def resolve_route(model: str, forced: str | None = None) -> tuple[str, str]:
    """Pick the first route that has a credential and can serve *model*.

    Returns (route_name, api_key). Preference order is the order of
    PANEL_ROUTES: an aggregator that covers all three, then the model's own
    vendor, then the HuggingFace router.

    The HF read token is used rather than `pass huggingface/token`, which
    carries write scope this process has no use for.

    Raises:
        RuntimeError: If no route with a credential can serve the model. The
            message names every route that was tried, because a silent
            fallback to a model the caller did not ask for is the one
            outcome this panel cannot tolerate.
    """
    candidates = [forced] if forced else list(PANEL_ROUTES)
    tried: list[str] = []
    for route in candidates:
        if route not in PANEL_ROUTES:
            raise RuntimeError(f"unknown route {route!r}")
        if route not in PANEL_MODEL_IDS[model]:
            tried.append(f"{route} (does not serve {model})")
            continue
        spec = PANEL_ROUTES[route]
        key = os.environ.get(spec["env_var"], "").strip() or _read_pass(spec["pass_entry"])
        if key:
            return route, key
        tried.append(f"{route} (no ${spec['env_var']}, no `pass {spec['pass_entry']}`)")
    raise RuntimeError(
        f"no usable route for {model}. Tried: " + "; ".join(tried)
    )


def _render_items(items: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    """Items as a JSON array, restricted to the fields the annotator sees."""
    trimmed = [{f: item[f] for f in fields if f in item} for item in items]
    return json.dumps(trimmed, indent=2, ensure_ascii=False)


def _call_model(
    session: requests.Session,
    api_key: str,
    model: str,
    route: str,
    prompt: str,
    batch_index: int,
) -> tuple[str, BatchUsage]:
    """One chat completion, retried with exponential backoff.

    Returns the assistant text and the usage record. Every attempt, including
    the ones that fail, produces a usage record so the reported cost includes
    retries rather than only the attempt that happened to work.
    """
    url = f"{PANEL_ROUTES[route]['base_url']}/chat/completions"
    payload: dict[str, Any] = {
        "model": PANEL_MODEL_IDS[model][route],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": PANEL_TEMPERATURE,
        "max_tokens": PANEL_MAX_TOKENS,
        # Object form, not the `reasoning_effort` string. See
        # PANEL_REASONING_EFFORT: the string spelling is accepted and ignored.
        "reasoning": {"effort": PANEL_REASONING_EFFORT},
    }
    if route == "openrouter":
        pin = PANEL_OPENROUTER_PROVIDER_PIN.get(model)
        if pin:
            # allow_fallbacks=False makes a busy backend a visible failure
            # rather than a silent reroute to different quantization.
            payload["provider"] = {"order": [pin], "allow_fallbacks": False}
        payload["usage"] = {"include": True}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error = ""
    for attempt in range(1, PANEL_MAX_RETRIES + 1):
        started = time.monotonic()
        try:
            response = session.post(
                url, headers=headers, json=payload, timeout=PANEL_TIMEOUT_S
            )
            elapsed = time.monotonic() - started
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                logger.warning(
                    "batch %d attempt %d failed: %s", batch_index, attempt, last_error
                )
                _RETRY_LOG.append(
                    _usage_record(batch_index, attempt, False, {}, elapsed, last_error)
                )
                time.sleep(PANEL_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
                continue

            body = json.loads(response.text, strict=False)
            if "error" in body:
                last_error = f"api error: {json.dumps(body['error'])[:300]}"
                logger.warning(
                    "batch %d attempt %d failed: %s", batch_index, attempt, last_error
                )
                _RETRY_LOG.append(
                    _usage_record(batch_index, attempt, False, body, elapsed, last_error)
                )
                time.sleep(PANEL_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
                continue

            choice = body["choices"][0]
            finish_reason = str(choice.get("finish_reason") or "")
            text = str(choice["message"].get("content") or "")
            usage = _usage_record(
                batch_index, attempt, True, body, elapsed, "", finish_reason
            )

            # Truncation is the quiet failure. A cut-off JSON array still
            # arrives as non-empty text, parses to nothing, and would be
            # recorded as 25 items the judge declined to answer, which drops
            # them from the denominator instead of showing up as an error.
            # Both truncation and a null content are retried.
            if finish_reason == "length" or not text.strip():
                last_error = (
                    f"unusable response, finish_reason={finish_reason!r}, "
                    f"{len(text)} chars, "
                    f"{usage['reasoning_tokens']} reasoning tokens spent"
                )
                usage["ok"] = False
                usage["error"] = last_error
                _RETRY_LOG.append(usage)
                logger.warning(
                    "batch %d attempt %d failed: %s", batch_index, attempt, last_error
                )
                time.sleep(PANEL_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
                continue
            return text, usage

        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            elapsed = time.monotonic() - started
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "batch %d attempt %d failed: %s", batch_index, attempt, last_error
            )
            _RETRY_LOG.append(
                _usage_record(batch_index, attempt, False, {}, elapsed, last_error)
            )
            time.sleep(PANEL_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"batch {batch_index}: all {PANEL_MAX_RETRIES} attempts failed. "
        f"Last error: {last_error}"
    )


# Module-level so failed attempts are still counted in the cost table. A
# failed batch costs real tokens and omitting it would understate the run.
_RETRY_LOG: list[BatchUsage] = []


def _usage_record(
    batch_index: int,
    attempt: int,
    ok: bool,
    body: dict[str, Any],
    latency_s: float,
    error: str,
    finish_reason: str = "",
) -> BatchUsage:
    """One row of token accounting, from a whole response body.

    `billed_cost_usd` is OpenRouter's own `usage.cost`, which is what the
    account was actually charged. Recomputing it from posted rates would
    drift the moment a backend prices differently from the model page.
    """
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "batch_index": batch_index,
        "attempt": attempt,
        "ok": ok,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "served_model": str(body.get("model") or ""),
        "served_provider": str(body.get("provider") or ""),
        "billed_cost_usd": float(usage.get("cost") or 0.0),
        "response_id": str(body.get("id") or ""),
        "finish_reason": finish_reason,
        "latency_s": round(latency_s, 2),
        "error": error,
    }


def cost_usd(model: str, route: str, usage_rows: list[BatchUsage]) -> float:
    """Total USD across every attempt, retries included.

    Prefers the provider's own billed figure when it is reported, and falls
    back to posted rates only for rows that carry none. Failed attempts are
    in *usage_rows* on purpose: a batch that burned 40k reasoning tokens and
    returned nothing was still charged for.
    """
    in_rate, out_rate = PANEL_PRICING_USD_PER_MTOK[model][route]
    total = 0.0
    for row in usage_rows:
        if row["billed_cost_usd"]:
            total += row["billed_cost_usd"]
        else:
            total += (row["prompt_tokens"] / 1e6) * in_rate
            total += (row["completion_tokens"] / 1e6) * out_rate
    return total


def _checkpoint_path(out_path: Path) -> Path:
    return out_path.with_suffix(".partial.json")


def _load_checkpoint(path: Path) -> dict[int, PanelAnswer]:
    """Completed answers from an earlier run, keyed by item_index."""
    if not path.exists():
        return {}
    data = load_json(path)
    done: dict[int, PanelAnswer] = {}
    for raw in data.get("items", []):
        if raw.get("primary_hub_id"):
            done[int(raw["item_index"])] = {
                "item_index": int(raw["item_index"]),
                "primary_hub_id": str(raw["primary_hub_id"]),
                "acceptable_hub_ids": [str(h) for h in raw.get("acceptable_hub_ids") or []],
                "confidence": str(raw.get("confidence") or ""),
                "notes": str(raw.get("notes") or ""),
            }
    return done


def _run_annotation(
    session: requests.Session,
    api_key: str,
    model: str,
    route: str,
    items: list[dict[str, Any]],
    hub_reference: str,
    valid_hub_ids: set[str],
    checkpoint: Path,
    resume: dict[int, PanelAnswer],
) -> tuple[list[PanelAnswer], list[BatchUsage], dict[str, int]]:
    """The open-book annotation run: the human's prompt, batched.

    Each batch is written to *checkpoint* as it lands. A run killed at batch
    7 of 10 keeps the first six, which matters because the failure mode
    actually seen here was a mid-run 402 that would otherwise have thrown
    away paid-for work.
    """
    answers: list[PanelAnswer] = []
    usage_rows: list[BatchUsage] = []
    counters = {"invented_primary": 0, "truncated_acceptable": 0, "missing": 0}

    for start in range(0, len(items), PANEL_BATCH_SIZE):
        batch = items[start : start + PANEL_BATCH_SIZE]
        batch_index = start // PANEL_BATCH_SIZE + 1
        expected = [int(item["item_index"]) for item in batch]

        if all(index in resume for index in expected):
            logger.info("%s: batch %d already done, reusing", model, batch_index)
            answers.extend(resume[index] for index in expected)
            continue

        logger.info(
            "%s: batch %d, items %d-%d", model, batch_index, expected[0], expected[-1]
        )
        prompt = _ANNOTATOR_PROMPT.format(
            hub_reference=hub_reference, items=_render_items(batch, _ITEM_FIELDS)
        )
        try:
            text, usage = _call_model(
                session, api_key, model, route, prompt, batch_index
            )
        except RuntimeError:
            atomic_write_json({"items": answers}, checkpoint)
            logger.error(
                "batch %d failed after all retries. %d answers checkpointed to %s",
                batch_index,
                len(answers),
                checkpoint,
            )
            raise
        usage_rows.append(usage)

        parsed, batch_counters = parse_judge_response(
            text, expected, valid_hub_ids, CEILING_STUDY_MAX_ACCEPTABLE_HUBS
        )
        for key, value in batch_counters.items():
            counters[key] += value
        answers.extend(parsed)
        atomic_write_json({"items": answers}, checkpoint)

    return answers, usage_rows, counters


def _run_hub_names(
    session: requests.Session,
    api_key: str,
    model: str,
    route: str,
    hub_names: dict[str, str],
    sample_size: int,
) -> tuple[list[dict[str, Any]], list[BatchUsage]]:
    """Exposure control: can the judge name a hub given its id?

    A fixed pseudo-random sample rather than the first N, so the probe is
    not concentrated in one branch of the taxonomy, and seeded so re-running
    it asks about the same hubs.
    """
    rng = random.Random(CEILING_STUDY_SEED)
    sampled = sorted(rng.sample(sorted(hub_names), min(sample_size, len(hub_names))))
    results: list[dict[str, Any]] = []
    usage_rows: list[BatchUsage] = []

    for start in range(0, len(sampled), PANEL_BATCH_SIZE):
        batch = sampled[start : start + PANEL_BATCH_SIZE]
        batch_index = start // PANEL_BATCH_SIZE + 1
        logger.info("%s: hub-name batch %d", model, batch_index)
        prompt = _HUB_NAME_PROMPT.format(
            items=json.dumps([{"hub_id": hub} for hub in batch], indent=2)
        )
        text, usage = _call_model(session, api_key, model, route, prompt, batch_index)
        usage_rows.append(usage)

        block = extract_json_array(text)
        by_id: dict[str, dict[str, Any]] = {}
        if block is not None:
            for entry in block:
                if isinstance(entry, dict) and "hub_id" in entry:
                    by_id[str(entry["hub_id"]).strip()] = entry
        for hub in batch:
            entry = by_id.get(hub, {})
            recalled = str(entry.get("recalled_name") or "").strip()
            expected = hub_names[hub]
            results.append({
                "hub_id": hub,
                "true_name": expected,
                "recalled_name": recalled,
                "exact_match": recalled.casefold() == expected.casefold(),
                "recall": str(entry.get("recall") or "none").strip().lower(),
            })
    return results, usage_rows


def _run_closed_book(
    session: requests.Session,
    api_key: str,
    model: str,
    route: str,
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[BatchUsage]]:
    """The contamination probe: recall OpenCRE's mapping with no reference."""
    recalls: list[dict[str, Any]] = []
    usage_rows: list[BatchUsage] = []

    for start in range(0, len(items), PANEL_BATCH_SIZE):
        batch = items[start : start + PANEL_BATCH_SIZE]
        batch_index = start // PANEL_BATCH_SIZE + 1
        logger.info("%s: closed-book batch %d", model, batch_index)
        prompt = _CLOSED_BOOK_PROMPT.format(items=_render_items(batch, _ITEM_FIELDS))
        text, usage = _call_model(session, api_key, model, route, prompt, batch_index)
        usage_rows.append(usage)

        block = extract_json_array(text)
        by_index: dict[int, dict[str, Any]] = {}
        if block is not None:
            for entry in block:
                if isinstance(entry, dict) and "item_index" in entry:
                    try:
                        by_index[int(str(entry["item_index"]))] = entry
                    except ValueError:
                        continue
        for item in batch:
            idx = int(item["item_index"])
            entry = by_index.get(idx, {})
            raw_id = str(entry.get("recalled_hub_id") or "").strip()
            recalls.append({
                "item_index": idx,
                "framework_id": str(item["framework_id"]),
                "recalled_hub_id": raw_id if HUB_ID_RE.fullmatch(raw_id) else "",
                "raw_recalled_hub_id": raw_id,
                "recall": str(entry.get("recall") or "none").strip().lower(),
            })
    return recalls, usage_rows


# Measured on this corpus against the HF router's own reported
# prompt_tokens: a 3-item batch carrying the full hub reference rendered to
# 275,845 characters and billed 72,334 prompt tokens, giving 3.81 characters
# per token. Used instead of a tokenizer because each vendor tokenises
# differently and installing three of them to produce an estimate that is
# still an estimate is not worth the dependency.
_CHARS_PER_TOKEN: Final[float] = 3.81

# Completion tokens per item, measured the same way: 2,475 completion tokens
# for 3 items with notes written, at reasoning_effort=low.
_COMPLETION_TOKENS_PER_ITEM: Final[int] = 825


def estimate_tokens(text: str) -> int:
    """Prompt tokens for *text*, from the measured characters-per-token rate."""
    return int(len(text) / _CHARS_PER_TOKEN)


def _report_dry_run(
    model: str,
    forced_route: str | None,
    items: list[dict[str, Any]],
    hub_reference: str,
    closed_book: bool,
) -> None:
    """Render the real prompts, count tokens, and price the run."""
    batches = [
        items[start : start + PANEL_BATCH_SIZE]
        for start in range(0, len(items), PANEL_BATCH_SIZE)
    ]
    prompt_tokens = 0
    for batch in batches:
        rendered = _render_items(batch, _ITEM_FIELDS)
        prompt = (
            _CLOSED_BOOK_PROMPT.format(items=rendered)
            if closed_book
            else _ANNOTATOR_PROMPT.format(hub_reference=hub_reference, items=rendered)
        )
        prompt_tokens += estimate_tokens(prompt)
    completion_tokens = len(items) * _COMPLETION_TOKENS_PER_ITEM

    try:
        route, _ = resolve_route(model, forced_route)
        route_status = f"{route} ({PANEL_ROUTES[route]['base_url']}), credential found"
    except RuntimeError as exc:
        route = forced_route or "hf_router"
        route_status = f"NO USABLE ROUTE. {exc}"

    in_rate, out_rate = PANEL_PRICING_USD_PER_MTOK[model][route]
    cost = (prompt_tokens / 1e6) * in_rate + (completion_tokens / 1e6) * out_rate

    print(f"\n=== dry run: {model} ===")
    print(f"  mode              {'closed-book recall' if closed_book else 'open-book annotation'}")
    print(f"  route             {route_status}")
    print(f"  route model id    {PANEL_MODEL_IDS[model].get(route, '(not served here)')}")
    print(f"  items             {len(items)} in {len(batches)} batches of {PANEL_BATCH_SIZE}")
    print(f"  temperature       {PANEL_TEMPERATURE}")
    print(f"  reasoning_effort  {PANEL_REASONING_EFFORT}")
    print(f"  max_tokens        {PANEL_MAX_TOKENS}")
    print(f"  est prompt tok    {prompt_tokens:,} at ${in_rate}/M")
    print(f"  est output tok    {completion_tokens:,} at ${out_rate}/M")
    print(f"  EST COST          ${cost:.2f}")
    print(
        "  note              output estimate excludes reasoning tokens, which "
        "bill at the\n                    output rate and are not capped by "
        "any setting on Kimi K3."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(PANEL_MODELS))
    parser.add_argument(
        "--items", type=Path, default=CEILING_STUDY_DIR / "ceiling_items.json"
    )
    parser.add_argument(
        "--hub-reference", type=Path, default=CEILING_STUDY_DIR / "hub_reference.md"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="First N items only, in item_index order."
    )
    parser.add_argument(
        "--closed-book",
        action="store_true",
        help="Contamination probe: recall OpenCRE's mapping with no hub reference.",
    )
    parser.add_argument(
        "--hub-name-probe",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Exposure control: ask the judge to name N hubs given their ids, "
            "with no reference. Distinguishes a judge that never saw OpenCRE "
            "from one that saw it and declines to guess."
        ),
    )
    parser.add_argument(
        "--route",
        default=None,
        choices=list(PANEL_ROUTES),
        help="Force a route instead of taking the first one with a credential.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed batches from the .partial.json checkpoint.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually call the API and spend money. Without it this is a dry "
            "run that renders the prompts, counts tokens, and writes nothing."
        ),
    )
    args = parser.parse_args()

    for path in (args.items, args.hub_reference):
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return EXIT_USER_ERROR

    _RETRY_LOG.clear()

    hub_reference = args.hub_reference.read_text(encoding="utf-8")
    valid_hub_ids = set(parse_hub_reference(hub_reference))
    logger.info("hub reference: %d hubs", len(valid_hub_ids))

    items_doc = load_json(args.items)
    items: list[dict[str, Any]] = sorted(
        items_doc["items"], key=lambda item: int(item["item_index"])
    )
    if args.limit is not None:
        items = items[: args.limit]
    logger.info("items: %d", len(items))

    slug = model_slug(args.model)
    if args.hub_name_probe:
        default_out = CEILING_STUDY_DIR / f"hub_name_probe_{slug}.json"
    elif args.closed_book:
        default_out = CEILING_STUDY_DIR / f"contamination_probe_{slug}.json"
    else:
        default_out = CEILING_STUDY_DIR / f"answers_panel_{slug}.json"
    out_path = args.out or default_out

    if not args.execute:
        _report_dry_run(args.model, args.route, items, hub_reference, args.closed_book)
        print(f"\nDRY RUN. Nothing was sent and nothing was written to {out_path}.")
        print("Re-run with --execute to spend.")
        return 0

    try:
        route, api_key = resolve_route(args.model, args.route)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR
    logger.info("route: %s (%s)", route, PANEL_ROUTES[route]["base_url"])

    checkpoint = _checkpoint_path(out_path)
    resume = _load_checkpoint(checkpoint) if args.resume else {}
    if resume:
        logger.info("resuming with %d answers already on disk", len(resume))

    started_at = datetime.now(timezone.utc).isoformat()
    session = requests.Session()

    payload_key: str
    payload_value: list[Any]
    counters: dict[str, int]
    try:
        if args.hub_name_probe:
            names, usage_rows = _run_hub_names(
                session,
                api_key,
                args.model,
                route,
                parse_hub_names(hub_reference),
                args.hub_name_probe,
            )
            payload_key, payload_value, counters = "hub_names", list(names), {}
        elif args.closed_book:
            recalls, usage_rows = _run_closed_book(
                session, api_key, args.model, route, items
            )
            payload_key, payload_value, counters = "recalls", list(recalls), {}
        else:
            answers, usage_rows, counters = _run_annotation(
                session,
                api_key,
                args.model,
                route,
                items,
                hub_reference,
                valid_hub_ids,
                checkpoint,
                resume,
            )
            payload_key, payload_value = "items", list(answers)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Partial answers, if any, are in "
            f"{_checkpoint_path(out_path)}. Re-run with --resume to continue.",
            file=sys.stderr,
        )
        return EXIT_OFFLINE
    finally:
        session.close()

    all_usage = usage_rows + _RETRY_LOG
    served = {row["served_model"] for row in usage_rows if row["served_model"]}
    providers = {row["served_provider"] for row in usage_rows if row["served_provider"]}
    in_rate, out_rate = PANEL_PRICING_USD_PER_MTOK[args.model][route]
    document: dict[str, Any] = {
        payload_key: payload_value,
        "run": {
            "model_id": args.model,
            "route": route,
            "base_url": PANEL_ROUTES[route]["base_url"],
            "route_model_id": PANEL_MODEL_IDS[args.model][route],
            "provider_pin": (
                PANEL_OPENROUTER_PROVIDER_PIN.get(args.model, "")
                if route == "openrouter"
                else ""
            ),
            "served_providers": sorted(providers),
            "served_model_snapshot": sorted(served),
            "temperature": PANEL_TEMPERATURE,
            "reasoning_effort": PANEL_REASONING_EFFORT,
            "max_tokens": PANEL_MAX_TOKENS,
            "batch_size": PANEL_BATCH_SIZE,
            "mode": (
                "hub_name_exposure_probe"
                if args.hub_name_probe
                else "closed_book_recall"
                if args.closed_book
                else "open_book_annotation"
            ),
            "n_items": len(items),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "prompt_source": "claudedocs/ceiling-study-runbook.md",
            "pricing_usd_per_mtok": {"input": in_rate, "output": out_rate},
            "cost_usd": round(cost_usd(args.model, route, all_usage), 4),
            "n_failed_attempts": len(_RETRY_LOG),
            "parse_counters": counters,
            "usage": all_usage,
        },
    }

    atomic_write_json(document, out_path)
    logger.info("wrote %s", out_path)
    logger.info(
        "cost $%.4f over %d successful and %d failed requests",
        document["run"]["cost_usd"],
        len(usage_rows),
        len(_RETRY_LOG),
    )
    if counters:
        logger.info("parse counters: %s", counters)
    checkpoint.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
