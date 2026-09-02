# LLM Fine-Tuning, RAG & GenAI

> Deeper reference: [`../ML_Fundamentals/Transformers_Cheatsheet.md`](../ML_Fundamentals/Transformers_Cheatsheet.md) covers attention math, training stages (SFT → RLHF/DPO/GRPO), LoRA/PEFT, decoding strategies, KV cache, and serving optimizations in full.

Referral tip: "LLM fine-tuning or RAG system" questions. 🔵 Job-description language for Catalog/GenAI-adjacent roles explicitly lists "extensive experience with Generative AI, LLMs (Transformers, BERT, Llama, GPTs, Gemini), RAG, and agentic AI systems, including designing, developing, deploying, and fine-tuning these models for production" — strong signal this is core to the role, not a nice-to-have. 🟢 Also see [08](08_Attribute_Extraction_Content_Generation.md) for the team's actual published GenAI architecture, which is the strongest prep material for this whole topic.

---

### Q1. When do you fine-tune vs. use RAG vs. prompt engineering alone?

| Approach | Best for | Not good for |
|---|---|---|
| Prompt engineering / few-shot | Fast iteration, no training infra, tasks the base model is already decent at | Consistent formatting/behavior at scale, deeply domain-specific tasks |
| RAG | Injecting up-to-date or proprietary factual knowledge, reducing hallucination via grounding, when knowledge changes often | Teaching the model a new *skill/behavior/style* — RAG adds facts, not capability |
| Fine-tuning (full or LoRA) | Teaching consistent task-specific behavior/format, domain adaptation, when you need it baked into weights (lower latency than a long RAG prompt) | Frequently-changing facts (would require constant retraining) |

For catalog attribute extraction specifically, both are used together: fine-tuning teaches the model *how* to extract/format attributes consistently, RAG-style retrieval of category schemas or the seller's other listings grounds *what* it should extract for that specific product ([08](08_Attribute_Extraction_Content_Generation.md)).

### Q2. Parameter-efficient fine-tuning: LoRA

- Freezes the base model weights; injects small trainable low-rank matrices (`A`, `B`) into attention/FFN layers, so the effective weight update is `W + BA` where `B`,`A` are low-rank.
- Why it matters practically: full fine-tuning of a multi-billion parameter model needs the optimizer state for *every* parameter (2-8x memory for Adam); LoRA only needs gradients/optimizer state for the small adapter matrices — enables fine-tuning on a single GPU, and multiple task-specific adapters can share one frozen base model (swap adapters per task instead of hosting N full model copies).
- Quantization (e.g., QLoRA — 4-bit base model + LoRA adapters) stacks with this for further memory reduction.
- This is exactly the stack Example Company's own catalog team has published using — see [08](08_Attribute_Extraction_Content_Generation.md) for the full list (LoRA, quantization, gradient checkpointing, Flash Attention, single-epoch training to avoid catastrophic forgetting).

### Q3. Catastrophic forgetting — what it is and how to avoid it

Fine-tuning on a narrow task can degrade the base model's general pretrained capabilities. Mitigations: fewer epochs (single-epoch is a deliberate, published choice at Example Company for exactly this reason), lower learning rate, LoRA/adapter-based tuning (leaves most weights frozen, inherently less destructive than full fine-tuning), mixing in a small amount of general-domain data during fine-tuning, and knowledge distillation from a strong general teacher model to retain broad capability in a smaller student.

### Q4. RAG quality debugging (retrieval looks good, answers are still bad)

See the full framework in [AI_Engineer Q4](../AI_Engineer/Q4_RAG_Quality.md) — directly relevant here. Key point to hit: good retrieval metrics (recall@k) don't guarantee good generation — the gap usually lives in context assembly (chunk boundaries splitting an answer across chunks), the model ignoring retrieved context in favor of parametric knowledge, or synthesis failure when the answer requires combining multiple retrieved documents. Diagnose by manually reviewing 20-30 bad answers and categorizing *where* the pipeline broke before proposing a fix.

### Q5. Evaluating a fine-tuned/RAG system in production

- **Dual benchmarks**: an LLM-specific benchmark and an LLM-agnostic one, so you can compare against non-LLM baselines, not just prior LLM versions (this is literally how Example Company's catalog team evaluates its extraction system — [08](08_Attribute_Extraction_Content_Generation.md)).
- Faithfulness/groundedness (is the output actually supported by retrieved/verified source data) separate from fluency — a fluent, ungrounded answer about a real product is a liability, not a minor quality issue, in an ecommerce catalog context.
- Human-annotated ground truth with provenance (did the correct answer come from text or image, in a multimodal setting) — worth mentioning as a more rigorous eval design than plain accuracy.

### Q6. Hallucination mitigation in a production GenAI system

- Grounding: condition generation strictly on retrieved/verified source data (product's own verified attributes, not free generation).
- Confidence thresholding + human review for low-confidence output (the same conditional-ingestion pattern used across [04](04_ML_System_Design_Fraud_Detection.md), [07](07_Content_Compliance_Moderation.md), [08](08_Attribute_Extraction_Content_Generation.md) — worth naming as a recurring architectural pattern if asked to synthesize across the interview).
- A separate verifier/QC model checking generated output against ground truth/policy, rather than trusting the generation model to self-police.
- Lower temperature / constrained decoding for factual extraction tasks vs. higher temperature for genuinely creative generation tasks — know when each is appropriate.

### Q7. Architecture basics: attention mechanism

Be ready for a from-scratch explanation, not just "it's how transformers work": queries/keys/values are learned projections of the input; attention score = scaled dot product of Q and K (`softmax(QKᵀ/√d_k)`), which weights V. Multi-head attention runs several of these in parallel with different learned projections, letting different heads attend to different types of relationships (syntactic, positional, semantic), then concatenates and projects back down. Know why the scaling factor `√d_k` exists (prevents dot products from growing large in magnitude and pushing softmax into a saturated, low-gradient regime as dimensionality increases).

---

### What This Round Tests

- Whether you understand fine-tuning vs. RAG as complementary, not competing, tools — and can justify which to reach for
- Practical, cost-aware knowledge of modern fine-tuning (LoRA/quantization) rather than "just fine-tune the whole model"
- Production-mindedness: hallucination, groundedness, and eval design specific to a system whose output is customer-facing product information
