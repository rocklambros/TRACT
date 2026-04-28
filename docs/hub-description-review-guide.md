# Hub Description Review Guide

## What You're Reviewing

We generated AI-written descriptions for 400 security control categories ("hubs") in the CRE (Common Requirements Enumeration) taxonomy. These descriptions will be used as features in a machine learning model that maps security framework controls to the correct CRE hub. **Description quality directly affects model accuracy.**

Each hub sits in a tree like:
```
Technical application security controls
  > Session management
    > Session token generation
      > Generate a new session token after authentication   <-- this is a leaf hub
```

There are 400 leaf hubs to review. Each description is 2-3 sentences (~600-1200 characters).

## The File

```
data/processed/hub_descriptions.json
```

Open it in any JSON-capable editor (VS Code with JSON folding works well). The file has a top-level `descriptions` object keyed by hub ID. Each entry looks like:

```json
"002-630": {
  "hub_id": "002-630",
  "hub_name": "Generate a new session token after authentication",
  "hierarchy_path": "Technical application security controls > Session management > Session token generation > ...",
  "description": "This hub covers the requirement to generate fresh session tokens...",
  "review_status": "pending",
  "reviewed_description": null,
  "reviewer_notes": null
}
```

## What to Evaluate

Each description was asked to do three things. Check all three:

### 1. Does it define what this hub covers in concrete terms?

**Good:** "This hub covers techniques for hiding or distorting the confidence scores, probability distributions, and certainty indicators that AI models produce alongside their primary outputs."

**Bad:** "This hub is about obscuring AI confidence." (too vague to distinguish from siblings)

### 2. Does it distinguish this hub from its sibling hubs?

Siblings are hubs that share the same parent. The description should make clear why a control belongs in *this* hub rather than one of its neighbors.

**Good:** "Unlike its siblings that focus on protecting inputs (Prompt input segregation, AI Input distortion) or model architecture (Ensemble AI models), this hub specifically addresses output-side information leakage."

**Bad:** "This hub is related to AI security controls." (could apply to any sibling)

### 3. Does it state what is NOT in scope?

**Good:** "This hub does not cover the cryptographic strength of tokens, the secure storage methods for tokens in browsers, or session token generation for other lifecycle events like privilege escalation."

**Bad:** No boundary statement at all, or a boundary that's so broad it's useless.

## How to Mark Your Review

For each hub, change three fields:

### `review_status` (required)

| Value | When to use |
|-------|-------------|
| `"accepted"` | Description is accurate, specific, and well-bounded. No changes needed. |
| `"edited"` | Description is mostly right but you improved it. Put your version in `reviewed_description`. |
| `"rejected"` | Description is wrong, misleading, or too vague to be useful. Explain why in `reviewer_notes`. |

### `reviewed_description` (required if `"edited"`)

Your improved version. Keep the same style: 2-3 sentences, concrete, technical. The original stays in `description` for comparison.

### `reviewer_notes` (optional but encouraged for edits/rejects)

Brief note on what was wrong. Examples:
- "Confused this hub with its parent — description covers the entire branch, not just this leaf"
- "Scope boundary contradicts the linked standards"
- "Factually wrong: CAPEC is an attack pattern catalog, not a vulnerability list"

## Common Problems to Watch For

1. **Parent/child confusion** — description covers the entire branch instead of just this specific leaf hub
2. **Sibling confusion** — description includes scope that belongs to a neighboring hub
3. **Vague scope boundaries** — "does not cover other topics" tells the model nothing
4. **Factual errors** — wrong definitions of standards, protocols, or attack types
5. **Circular definitions** — "This hub covers X" where X is just the hub name restated
6. **Missing scope boundary** — no statement of what is excluded

## Workflow Suggestions

- **Batch in sessions of 50** (takes roughly 2-3 hours per batch)
- **Work top-down by hierarchy path** — reviewing siblings together makes it easier to check for overlap and boundary clarity
- **Don't agonize over perfect wording** — if the description is accurate and distinguishes the hub, accept it. Only edit when something is meaningfully wrong or missing.
- **Save frequently** — the file is valid JSON at all times, so saving mid-batch is fine

## Checking Your Progress

After each session, run:

```bash
python -m scripts.phase1a.validate_descriptions
```

Output:
```
Total:    400
Pending:  350
Accepted: 40
Edited:   8
Rejected: 2
All validations passed.
```

## Example Reviews

### Accept (no changes needed)

```json
"010-108": {
  "description": "This hub covers techniques for hiding or distorting the confidence scores...",
  "review_status": "accepted",
  "reviewed_description": null,
  "reviewer_notes": null
}
```

### Edit (improved the description)

```json
"088-316": {
  "description": "This hub covers controls that ensure the integrity and trustworthiness of data used to augment AI training datasets...",
  "review_status": "edited",
  "reviewed_description": "This hub covers validation and integrity controls for synthetic and augmented data used in AI training pipelines, including detection of poisoned samples, verification of data transformation fidelity, and quality assurance before model ingestion. It focuses on pre-training data quality, not runtime input validation or model architecture integrity. Excludes original training data collection and post-training model evaluation.",
  "reviewer_notes": "Original was accurate but buried the key distinction (pre-training vs runtime) in the middle. Restructured to lead with it."
}
```

### Reject (fundamentally wrong)

```json
"XXX-YYY": {
  "description": "...",
  "review_status": "rejected",
  "reviewed_description": null,
  "reviewer_notes": "Description confuses this leaf hub with its parent branch. Covers all of 'Secure AI inference' rather than just the specific technique of obscuring confidence scores."
}
```

## Questions?

If you're unsure whether a description is correct, check what standards are linked to that hub. The linked standard section names were used as input when generating the description — they tell you what real-world controls map to this hub.

You can look up linked sections in:
```
data/training/hub_links.jsonl
```

Search for the hub's `cre_id` to see which framework sections are mapped to it.
