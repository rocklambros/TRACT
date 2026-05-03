# TRACT Crosswalk Review Guide

## Role

You are a cybersecurity domain expert reviewing AI-generated mappings between
security framework controls and the Common Requirement Enumeration (CRE) hub
taxonomy. Your review determines which mappings are published as a
peer-reviewed research dataset.

## Background

**CRE (Common Requirement Enumeration)** is a universal taxonomy that links
security requirements across standards. It organizes security topics into
**522 hubs** arranged in a hierarchy — each hub represents a distinct security
concept (e.g., "Cryptography > Key Management" or "Authentication > Multi-Factor").

**TRACT** (Transitive Reconciliation and Assignment of CRE Taxonomies) maps
controls from **31 security frameworks** to CRE hubs using a fine-tuned
bi-encoder model. Each mapping is called an **assignment** — it says "this
control belongs under this hub."

Your job is to review these assignments and decide whether each one is correct.

## Step-by-Step Process

1. Open `review_predictions.json` in your editor.
2. For each prediction, read `control_text` to understand the control's security intent.
3. Read `assigned_hub_name` and `assigned_hub_path` to understand the model's suggestion.
4. Check `confidence` — above 0.70 is fairly certain, below 0.30 needs careful attention.
5. Check `review_priority` — "critical" items need the most care.
6. **Accept:** Control belongs under this hub → set `"status": "accepted"`.
7. **Reassign:** Wrong hub, but a better one exists → set `"status": "reassigned"`, set `"reviewer_hub_id"` to the correct hub ID (find IDs in `hub_reference.json`).
8. **Reject:** No hub fits → set `"status": "rejected"`, explain in `"reviewer_notes"`.
9. Add `reviewer_notes` for any non-obvious decision.

## Decision Criteria

The control's security **PURPOSE** should align with the hub's security
**DOMAIN**. This is not keyword matching — a control about "encrypting AI
training data" maps to encryption, not AI training.

## Common Pitfalls

- **MITRE ATLAS hubs are the hardest** — many sound similar. Read the full path.
- **"Rejected" means no hub fits at all.** Check `alternative_hubs` and `hub_reference.json` before rejecting.
- **High `is_ood: true`** means the model thinks this control is outside its training distribution — review more carefully.
- **NIST AI RMF predictions** are based on short control descriptions and may be less reliable.
- **Items flagged `text_quality: "low"`** had sparse input text — predictions may be unreliable

## redictions

This file contains **898 predictions** to review:
- **739 routine** items 
- **119 careful** items 
- **20 critical** item

## Deliverable

TThe deliverable is a final, reviewed review_export.json  that is reviewed and completed

