# Synthetic Top 10 for Beacon Relay Systems 2026

This fixture is invented. It reproduces the heading skeleton of the OWASP Top
10 for LLM Applications 2026 (entry headings, the four standard subsections,
the extra subsections some entries carry, and the Appendix A mapping tables)
with prose about a fictional beacon relay, so the parser is exercised without
tracking a licensed source in git. See tests/test_licensed_text_not_tracked.py.

## License and Usage

Invented text for a test fixture. No third-party content appears here.

## Revision History

| Version | Date | Note |
|---|---|---|
| 0.1 | [release date] | synthetic |

## Relay Top 10 at a Glance

Figure 1: the fictional relay attack surface, drawn as three concentric rings.

## LLM01:2026 Beacon Spoofing

## Description

A beacon spoofing weakness occurs when a relay accepts a positioning beacon
whose origin it cannot establish, and then treats the beacon payload as an
instruction rather than as an observation. The relay draws no boundary between
a bearing report and a steering command, so both arrive on the same channel.

Three deployment properties make this worse. Channel pooling means the relay
merges operator commands, neighbour reports, and cached almanac entries into a
single decision buffer. Almanac persistence means a spoofed entry written to
the cache steers every later fix that reads it. Autonomous steering means the
relay acts on the resulting bearing without an operator in the loop.

## Types of Beacon Spoofing

## Direct Spoofing

An operator, or an attacker holding the operator's console, transmits a bearing
the relay accepts without challenge.

## Relayed Spoofing

The relay ingests a bearing forwarded by a neighbour it has no way to
authenticate, and the operator never sees the forwarded payload.

## Common Examples of Risk

1. Console override: a crafted bearing overrides the standing route and the
   relay steers outside its permitted corridor.
2. Neighbour injection: a spoofed bearing rides in a forwarded almanac entry
   and fires when the cache is read.
3. Cache persistence: one tainted almanac entry reaches every later fix.

## Prevention and Mitigation Strategies

1. Sign every beacon at the transmitter and verify the signature at the relay
   before the payload reaches the decision buffer.
2. Hold steering authority in the relay controller rather than in the beacon
   parser, and grant least privilege per manoeuvre.

## Example Attack Scenarios

## Scenario #1: Console Override

An attacker with console access transmits a bearing that steers the relay off
its corridor and into a restricted lane.

## Scenario #2: Almanac Poisoning

An attacker contributes a poisoned almanac entry that every later fix reads.

## LLM02:2026 Almanac Disclosure

## Description

An almanac disclosure weakness exposes cached positioning records, operator
identifiers, or corridor geometry to a party with no right to read them. The
relay answers a bearing query with more of its cache than the query needed.

## Common Examples of Risk

1. Over-broad answers: the relay returns the whole corridor when asked for one
   segment of it.
2. Telemetry leakage: the diagnostic feed carries full almanac records.

## Prevention and Mitigation Strategies

## Tier 1: Foundational

Scope every answer to the segment the query named.

## Tier 2: Hardening

Redact operator identifiers from the diagnostic feed.

## Example Attack Scenarios

## Scenario #1

A crafted query walks the corridor segment by segment and reconstructs it.

## LLM03:2026 Excessive Steering Authority

## Description

Excessive steering authority is the damage a relay can do once a bearing it
accepted turns out to be wrong. The weakness is not the wrong bearing, it is
the breadth of the manoeuvres the relay is permitted to perform on one.

## Common Examples of Risk

1. Excessive function: the relay holds a docking manoeuvre it never needs.
2. Excessive permission: one shared credential drives every lane.

## Prevention and Mitigation Strategies

Remove manoeuvres the standing route does not need, and bind each remaining
manoeuvre to a per-operator credential.

## Example Attack Scenarios

## Scenario #1: Unattended Docking

A spoofed bearing triggers the docking manoeuvre with no operator present.

## LLM04:2026 Relay Supply Chain

## Description

A relay supply chain weakness reaches the fleet through the parts, firmware,
and almanac feeds a relay depends on rather than through its own channel. A
tampered almanac feed is indistinguishable from a healthy one at install time.

## Common Examples of Risk

1. Unsigned firmware: a relay accepts a firmware image resolved by a mutable
   tag rather than by digest.
2. Feed substitution: an almanac feed is replaced upstream of the fleet.

## Prevention and Mitigation Strategies

Pin every firmware image and almanac feed by digest, and verify the signature
before install.

## Example Attack Scenarios

## Scenario #1: Tampered Firmware

A firmware image published under a reused name reaches the whole fleet.

## LLM05:2026 Almanac Poisoning

## Description

Almanac poisoning writes attacker-chosen entries into the corpus a relay reads
its fixes from, so the relay produces a wrong bearing from a healthy parser.
The poisoned entry survives a restart because the cache is durable.

## Common Examples of Risk

1. Corpus contribution: a public feed accepts an unvetted entry.
2. Dormant trigger: the entry only steers when a rare bearing is requested.

## Prevention and Mitigation Strategies

Validate every entry at ingest, and record the provenance of each one.

## Example Attack Scenarios

## Scenario #1

A poisoned entry steers the relay only on a bearing the operator rarely asks
for, so routine testing never reaches it.

## LLM06:2026 Unbounded Relay Consumption

## Description

Unbounded relay consumption is any pattern that lets one request consume more
of the relay's power, bandwidth, or compute budget than the request is worth.
The relay has no ceiling on how much work a single bearing query may cause.

## Common Examples of Risk

1. Query flood: a burst of bearing queries drains the power budget.
2. Fan-out: one query causes a chain of neighbour queries.

## Prevention and Mitigation Strategies

Rate limit per operator, and cap the fan-out of a single query.

## Example Attack Scenarios

## Scenario #1: Power Drain

A sustained query burst leaves the relay unable to answer its own operator.

## LLM07:2026 Bearing Misinformation

## Description

Bearing misinformation is a confidently reported fix that is wrong, and that a
downstream consumer acts on because the relay reported it without a qualifier.
The relay states a bearing it cannot support from the observations it holds.

## Common Examples of Risk

1. Fabricated fix: the relay reports a bearing with no supporting observation.
2. Stale fix: the relay reports a cached bearing as though it were current.

## Prevention and Mitigation Strategies

Attach the supporting observation and its age to every reported bearing.

## Example Attack Scenarios

## Scenario #1: Stale Fix Accepted

A downstream planner treats a cached bearing as current and plots a bad route.

## LLM08:2026 Hidden Configuration Exposure

## Description

Hidden configuration exposure is the recovery of the relay's standing route,
corridor limits, or credential material by a party that was never given them.
The configuration was treated as secret because it was merely not displayed.

## Common Examples of Risk

1. Route disclosure: the standing route is recovered from error text.
2. Credential exposure: a credential embedded in the configuration leaks.

## Prevention and Mitigation Strategies

Keep credentials out of the configuration block, and enforce corridor limits in
the controller rather than in the configuration text.

## Example Attack Scenarios

## Scenario #1: Route Recovered from Errors

An attacker walks the error surface and reconstructs the standing route.

## LLM09:2026 Cache Geometry Weaknesses

## Description

Cache geometry weaknesses are the failures specific to storing almanac entries
as vectors and retrieving them by proximity. Proximity is not authorization,
and a shared index answers a query from whichever tenant is nearest.

## Common Examples of Risk

1. Cross-tenant retrieval: a shared index returns another operator's entry.
2. Entry inversion: the source record is reconstructed from its vector.

## Prevention and Mitigation Strategies

Enforce the tenant filter inside the index query, not after it returns.

## Example Attack Scenarios

## Scenario #1: Shared Index Leak

A query in one lane returns an entry belonging to another lane.

## LLM10:2026 Improper Bearing Handling

## Description

Improper bearing handling is the failure to validate a relay's own output
before a downstream component acts on it. The relay is trusted as a source of
commands by a planner that never checks what it received.

## Common Examples of Risk

1. Unescaped output: a bearing string reaches a command interpreter intact.
2. Unchecked schema: the planner accepts a bearing outside the valid range.

## Prevention and Mitigation Strategies

Validate every bearing against a schema in the planner before acting on it.

## Example Attack Scenarios

## Scenario #1

A bearing string carrying interpreter syntax reaches the planner unescaped.

## Appendix A: Related Framework Mappings

This appendix consolidates the mappings from the ten synthetic relay entries to
two invented reference taxonomies.

## How to read this appendix

Legend: ● primary · ○ supporting · -no applicable mapping.

## Coverage matrix

| Risk | REF | XYZ |
|---|---|---|
| LLM01 Beacon Spoofing | ● | ○ |
| LLM02 Almanac Disclosure | ● | - |
| LLM03 Excessive Steering Authority | ○ | - |
| LLM04 Relay Supply Chain | ● | - |
| LLM05 Almanac Poisoning | ● | - |
| LLM06 Unbounded Relay Consumption | ● | - |
| LLM07 Bearing Misinformation | ● | - |
| LLM08 Hidden Configuration Exposure | ● | - |
| LLM09 Cache Geometry Weaknesses | ● | - |
| LLM10 Improper Bearing Handling | ● | - |

## Synthetic Reference Taxonomy (REF) -v9.9

Each row maps a relay entry to the invented REF taxonomy.

| Risk | Element | Relevance |
|---|---|---|
| LLM01 Beacon Spoofing | ● REF-11 Unauthenticated Input | The relay accepts a bearing whose origin it cannot establish. |
|| ○ REF-22 Durable Cache Trust | A spoofed almanac entry survives a restart and steers later fixes. |
| LLM02 Almanac Disclosure | ● REF-33 Over-Broad Answer | The relay answers with more of its cache than the query named. |
| LLM03 Excessive Steering Authority | ○ REF-44 Excess Capability | The relay holds manoeuvres the standing route never needs. |
| LLM04 Relay Supply Chain | ● REF-55 Unpinned Dependency | Firmware resolved by a mutable tag rather than by |

| Risk | Element | Relevance |
|---|---|---|
||| digest reaches the whole fleet. |
| LLM05 Almanac Poisoning | ● REF-22 Durable Cache Trust | A poisoned entry persists in the corpus the relay reads. |
| LLM06 Unbounded Relay Consumption | ● REF-66 Unbounded Work | One query causes work with no declared ceiling. |
| LLM07 Bearing Misinformation | ● REF-77 Unsupported Assertion | A bearing is reported with no supporting observation. |
| LLM08 Hidden Configuration Exposure | ● REF-33 Over-Broad Answer | Error text discloses the standing route. |
| LLM09 Cache Geometry Weaknesses | ● REF-88 Proximity As Authorization | A shared index answers from whichever tenant is nearest. |
| LLM10 Improper Bearing Handling | ● REF-99 Unvalidated Output | The planner acts on a bearing it never validated. |

## Invented Control Set (XYZ) -v0.1

Each row maps a relay entry to the invented XYZ control set.

| Risk | Element | Relevance |
|---|---|---|
| LLM01 Beacon Spoofing | ○ XYZ Transmitter Assurance | Signing at the transmitter is the control this entry needs. |

## Framework Sources & Versions

| Framework | Version | Source |
|---|---|---|
| Synthetic Reference Taxonomy (REF) | v9.9 | invented |
| Invented Control Set (XYZ) | v0.1 | invented |

## Appendix B: Relay Architecture

## References

## LLM01: Beacon Spoofing

Invented reference list. This heading repeats the entry label without the 2026
tag, and it sits below Appendix A, so a parser that does not stop at the
appendix boundary would swallow it.

## Acknowledgements

Nobody. This file is invented.
