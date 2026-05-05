# TRACT Glossary

Cross-domain reference for terms used throughout TRACT documentation. Security practitioners will find ML concepts explained; ML researchers will find security framework terminology clarified.

---

**Assignment paradigm** — TRACT's core design principle: map each control independently to a CRE hub position via `g(control_text) → CRE_hub`. Never compare two controls pairwise. This scales linearly with the number of controls, unlike pairwise comparison which scales quadratically.

**Bi-encoder** — A neural network architecture that encodes two inputs (here: a control text and a CRE hub representation) independently into fixed-size vectors, then compares them via cosine similarity. TRACT uses BGE-large-v1.5 as its bi-encoder backbone. Contrast with *cross-encoder*, which processes both inputs jointly (more accurate but much slower).

**Bootstrap CI** — A confidence interval computed by resampling the evaluation data with replacement (typically 10,000 times) and computing the metric on each resample. TRACT reports 95% bootstrap confidence intervals for all evaluation metrics. Example: "hit@1 = 0.537 [0.463, 0.612]" means the true hit@1 is between 0.463 and 0.612 with 95% confidence.

**Bridge** — A discovered connection between an AI-specific CRE hub and a traditional security CRE hub, established when their embeddings are highly similar. Bridges reveal that an AI security concern (e.g., a MITRE ATLAS technique) maps to the same underlying security concept as a traditional control (e.g., a NIST 800-53 control). TRACT discovered 46 bridges from 63 candidates.

**Calibration** — The process of converting raw model outputs (cosine similarities) into meaningful probability estimates. TRACT uses temperature scaling (a form of Platt scaling) so that a reported confidence of 0.8 means the model is correct ~80% of the time. See also: *ECE*.

**Conformal prediction** — A statistical method that produces prediction *sets* with guaranteed coverage. If configured for 90% coverage, the prediction set will contain the correct CRE hub at least 90% of the time, regardless of the underlying model's accuracy. The trade-off is that prediction sets may contain multiple candidates.

**Contrastive fine-tuning** — A training approach where the model learns to place related items closer together and unrelated items farther apart in embedding space. TRACT trains with (control, correct_hub) positive pairs and hard negative hubs — hubs that are similar but incorrect — to sharpen the model's discrimination.

**Control** — A single security requirement, technique, weakness, or practice within a framework. Examples: NIST 800-53 control "AC-1 Access Control Policy", MITRE ATLAS technique "AML.T0043 Adversarial ML Attack", CWE weakness "CWE-79 Cross-site Scripting." TRACT's atomic unit of analysis.

**Cosine similarity** — A measure of similarity between two vectors, ranging from -1 (opposite) to +1 (identical direction). TRACT uses cosine similarity between control embeddings and hub embeddings to rank assignment candidates. Raw cosine scores are not probabilities — see *calibration*.

**CRE hub** — A node in the OpenCRE hierarchy that represents a specific security concept. Each hub has an ID (e.g., "646-285"), a name (e.g., "Input validation"), and links to controls in various frameworks. TRACT's label space consists of 400 leaf hubs. See also: *OpenCRE*.

**Crosswalk** — A mapping between controls in different security frameworks that address the same underlying concept. Traditional crosswalks are manually curated; TRACT generates crosswalks automatically by assigning controls from different frameworks to the same CRE hubs.

**ECE (Expected Calibration Error)** — A metric measuring how well a model's confidence scores match its actual accuracy. An ECE of 0.05 means the model's stated confidences are off by 5 percentage points on average. Lower is better. TRACT targets ECE < 0.10.

**Embedding** — A fixed-size numerical vector (1,024 dimensions for TRACT's BGE-large-v1.5) that captures the semantic meaning of a text. Similar texts produce similar embeddings. TRACT embeds both control texts and hub representations, then matches them by cosine similarity.

**Framework** — A published collection of security controls, requirements, or practices. Examples: NIST 800-53, MITRE ATLAS, OWASP Top 10 for LLM Applications. TRACT processes 31 frameworks spanning AI-specific and traditional security domains.

**Hard negative** — During training, a CRE hub that is semantically similar to the correct hub but is actually incorrect for a given control. Training with hard negatives (rather than random negatives) forces the model to make fine-grained distinctions. TRACT samples 3 hard negatives per positive pair using temperature-scaled similarity.

**hit@k** — The fraction of controls for which the correct CRE hub appears in the model's top-k predictions. hit@1 measures exact accuracy; hit@5 measures whether the correct hub is among the top 5 candidates. TRACT's trained model achieves hit@1 = 0.537 and the zero-shot baseline achieves hit@1 = 0.399.

**Hub description** — An LLM-generated natural language description of what a CRE hub covers, used to enrich the hub's embedding. Generated by Claude Opus with zero temperature for determinism. Combined with the hub's hierarchy path to form the hub representation.

**Hub firewall** — TRACT's evaluation integrity mechanism. When evaluating on framework X, all of X's linked sections are removed from CRE hub representations before computing embeddings. This prevents information leakage — the model cannot "cheat" by recognizing text it was trained on. Non-negotiable for honest evaluation.

**Hub hierarchy path** — The path from the root of the OpenCRE hierarchy to a specific hub, expressed as a text string. Example: "Technical controls > Input validation > SQL injection prevention". Concatenated with hub descriptions to form hub representations. Adding hierarchy paths improved zero-shot hit@1 by +7.6%.

**Hub proposal** — A suggested new CRE hub for controls that don't map well to any existing hub (out-of-distribution controls). Generated by clustering OOD control embeddings using HDBSCAN and naming clusters via LLM.

**LOFO (Leave-One-Framework-Out)** — TRACT's cross-validation strategy. For each framework with known CRE links, the model is retrained and hub representations are rebuilt *without* that framework, then evaluated on it. This is stricter than random holdout because it tests generalization to entirely unseen frameworks. See also: *hub firewall*.

**LoRA (Low-Rank Adaptation)** — A parameter-efficient fine-tuning technique that adds small trainable matrices to a frozen pretrained model. TRACT uses LoRA with rank 16 and alpha 32 on the query, key, and value attention layers of BGE-large-v1.5, training ~0.3% of total parameters.

**Mapping unit** — The atomic element within a framework that TRACT processes. Usually a "control" but may be a "technique" (MITRE ATLAS), "weakness" (CWE), "attack pattern" (CAPEC), or "article" (EU AI Act), depending on the framework's structure.

**MRR (Mean Reciprocal Rank)** — The average of 1/rank for each control's correct CRE hub in the ranked prediction list. If the correct hub is ranked 1st, the reciprocal rank is 1.0; if ranked 3rd, it's 0.333. Higher is better. Sensitive to the position of the first correct answer.

**NDCG@10 (Normalized Discounted Cumulative Gain)** — A ranking quality metric that rewards correct answers appearing higher in the list, with logarithmic discounting for lower positions. Ranges from 0 to 1. TRACT reports NDCG@10 as a complement to hit@k and MRR.

**OOD (Out-of-Distribution) detection** — Identifying controls whose embeddings are far from any known CRE hub, suggesting the control covers a concept not yet represented in the OpenCRE hierarchy. TRACT uses the 5th percentile of in-distribution similarity scores as the OOD threshold.

**OpenCRE (Open Common Requirement Enumeration)** — A community-maintained taxonomy that organizes security requirements into a hierarchy of groups and hubs, with links to controls in major frameworks. TRACT uses OpenCRE as its universal coordinate system — every control is positioned by its assignment to a CRE hub. See [opencre.org](https://opencre.org).

**Provenance** — The origin of a control-to-hub assignment in the crosswalk dataset. TRACT tracks four types: `opencre_ground_truth` (existing OpenCRE links), `ground_truth_T1-AI` (AI framework ground truth), `active_learning_round_2` (model predictions reviewed by experts), and `model_prediction` (accepted model output).

**Temperature scaling** — A post-hoc calibration method that divides model logits by a learned temperature parameter T before applying softmax. T > 1 makes the model less confident (spreading probability mass); T < 1 makes it more confident. TRACT learns T on a held-out calibration set to minimize ECE.

**Tier** — TRACT classifies frameworks into three tiers based on their relationship to OpenCRE:
- **Tier 1:** Frameworks already linked to OpenCRE (19 frameworks, 4,405 curated links) — used as training signal.
- **Tier 2:** AI frameworks with primary-source parsers but no CRE links (12 frameworks) — inference targets.
- **Tier 3:** New frameworks added via `tract prepare` — processed on demand.
