# Catalog Use Case: Product Attribute Extraction & Content Generation

Referral tip: flagship use case #3. 🔵 This is the **best-grounded** topic in the whole prep set — Example Company's own engineering blog publicly describes exactly this system, built by the Catalog Data Science org. Treat the architecture below as close to "what the interviewer already has in mind as a good answer," not generic textbook GenAI. 🟢 (company engineering blog)

> "Design a system that extracts structured product attributes (brand, size, material, color, etc.) from unstructured seller-provided text and images, and/or generates missing product content."

---

### 1. The problem, framed correctly

Sellers provide messy, inconsistent, incomplete text descriptions and images. The catalog needs **structured, normalized attributes** (brand, size, color, material, ingredients, etc.) for search, filtering, recommendations, and compliance — and content generation (e.g., filling in a missing description) needs to be *accurate*, not just fluent.

The core tension to name up front: **LLMs are great at extraction/generation but not inherently trustworthy at scale** — you cannot pipe raw LLM output straight into a live catalog serving hundreds of millions of customers. The system design problem is really "how do you get LLM-quality extraction with non-LLM reliability guarantees."

### 2. Example Company's actual published architecture — two-agent design 🟢

- **Extraction Agent**: an LLM that pulls attributes from listing text *and* images (multimodal).
- **Quality-Check Agent**: a **separate** LLM that validates the Extraction Agent's output against human-validated ground truth / policy rules — this second agent is the trust layer, not an afterthought.
- **Conditional ingestion**: attributes scoring above a **90–95% accuracy threshold** are auto-ingested directly into the catalog; anything below threshold routes to deeper QC or human review.

This maps directly onto the same "auto-approve / gray-zone / block" pattern from the fraud and compliance systems ([04](04_ML_System_Design_Fraud_Detection.md), [07](07_Content_Compliance_Moderation.md)) — worth explicitly noting the pattern reuse across all three use cases if asked to compare them; it signals systems-level thinking rather than three unrelated point solutions.

### 3. Fine-tuning stack (also from Example Company's published architecture) 🟢

| Technique | Why |
|---|---|
| LoRA (parameter-efficient fine-tuning) | Fine-tune without updating all base-model weights — dramatically cheaper, and multiple task-specific LoRA adapters can share one base model |
| Quantization | Reduces memory footprint for training and serving |
| Gradient checkpointing + Flash Attention | Lets fine-tuning fit on constrained hardware — Example Company's blog explicitly frames this as **single-GPU training "to democratize development"** rather than requiring large GPU clusters for every task |
| Multi-task learning | One model handles multiple related attribute-extraction tasks rather than a model-per-attribute-type, sharing representations |
| Knowledge distillation | Distill a larger teacher model into a smaller, cheaper-to-serve student model for production inference |
| Single-epoch training | Deliberately avoid catastrophic forgetting of the base model's pretrained general knowledge |

If asked "how would you fine-tune an LLM for this," reproducing this stack (and explaining *why* each piece — cost, catastrophic forgetting, serving latency) is a strong, team-specific answer.

### 4. Evaluation

- **Dual benchmark sets**: one LLM-specific, one LLM-agnostic — lets you compare a new LLM-based approach against a non-LLM baseline on the same ground truth, not just against previous LLM versions.
- **Human-annotated ground truth**, notably including **provenance labeling** — annotators mark whether an attribute's evidence came from the text or the image, which is unusually rigorous and worth mentioning if asked how you'd build eval data for a multimodal extraction task.

### 5. System design for content generation (the generative half)

For generating missing content (e.g., a product description when the seller provided none):
- **Grounding is mandatory** — generation must be conditioned on verified structured attributes (from the extraction pipeline above) and existing verified content, not generated freely, to avoid hallucinated claims about a real physical product (a hallucinated "waterproof" claim on a non-waterproof product is a real liability, not a cosmetic error).
- Same conditional-ingestion pattern applies: generated content above a confidence/quality threshold auto-publishes; below threshold, routes to human copy review.
- RAG-style retrieval of the seller's other verified listings / category templates as generation context is a natural design choice — reduces hallucination by grounding generation in real, verified product data rather than the model's parametric knowledge alone.

### 6. Edge cases to raise proactively

- **Text/image conflict**: title says one material, image shows another (or a seller changes the image without updating text) — the Quality-Check Agent needs a defined tie-breaking policy, not silent failure.
- **Long-tail/rare categories**: attribute schemas differ wildly by category (a "sleeve length" attribute is meaningless for electronics) — the extraction schema itself needs to be category-conditioned, which argues for retrieving the right attribute schema per category before extraction runs.
- **Model drift as the base LLM is upgraded**: swapping the underlying LLM can silently shift extraction behavior across millions of live listings — argue for the dual-benchmark eval (point 4) being run as a **gate before any model swap**, not just at initial launch.

### What This Question Tests

- Whether you understand that production LLM systems need a **verification/trust layer**, not just a good prompt — the two-agent pattern is the single most important idea to demonstrate here
- Practical, cost-aware fine-tuning knowledge (LoRA, quantization, single-GPU constraints) rather than "just fine-tune it" hand-waving
- Grounding/hallucination awareness specific to *generating* content about real physical products, where a fluent-but-wrong answer is a genuine business/legal risk
