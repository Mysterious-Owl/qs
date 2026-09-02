# GenAI / LLM Cheat Sheet

Applied GenAI for interviews: tokenization, inference parameters, prompting, RAG (including ~18 named variants), evaluation, fine-tuning, agents, and model selection.

Companion docs — deliberately no overlap:
- [`Transformers_Cheatsheet.md`](Transformers_Cheatsheet.md) — attention math, architecture internals, KV cache, serving optimizations, PEFT mechanics
- [`DL_Concepts_Cheatsheet.md`](DL_Concepts_Cheatsheet.md) — DL fundamentals and non-transformer architectures
- [`../Example_Company/10_LLM_Finetuning_RAG_GenAI.md`](../Example_Company/10_LLM_Finetuning_RAG_GenAI.md) — the same topics framed as company-loop interview questions

---

# 1. Tokenization

**What it is**: Text → integer IDs the model consumes. Not words — *subword* units.

| Algorithm | How it works | Used by |
|---|---|---|
| **BPE** (Byte-Pair Encoding) | Start from characters, iteratively merge the most frequent adjacent pair | GPT-2/3 |
| **Byte-level BPE** | BPE over raw bytes, so **any** Unicode is representable, no `<UNK>` ever | GPT-3.5/4+, Llama |
| **WordPiece** | Like BPE, but merges the pair that most increases training-corpus likelihood | BERT |
| **SentencePiece / Unigram** | Trains a probabilistic vocab directly on raw text; no pre-tokenization, language-agnostic | T5, Llama, mT5 |

**Why subword at all**: word-level vocabularies explode in size and can't handle unseen words; character-level makes sequences far too long. Subword is the compromise — frequent words stay single tokens, rare words decompose.

**Practical facts worth knowing:**
- English averages roughly **~4 characters ≈ 1 token** (~0.75 words/token). Useful for cost and context-budget estimates.
- Code, JSON, non-Latin scripts, and long numbers tokenize *much* less efficiently — the same "amount" of content costs more tokens.
- Numbers often split oddly (`1234` → `12|34`), which is part of why LLMs struggle with arithmetic.
- Whitespace usually attaches to the *following* token (`" cat"` ≠ `"cat"`) — a common cause of subtle prompt bugs.

**Gotcha**: Tokenizer and model are a matched pair. Swapping tokenizers without retraining produces garbage, since embedding-table row *i* means whatever token *i* meant during training.

---

# 2. SLM vs LLM

| | SLM (Small Language Model) | LLM |
|---|---|---|
| Scale | ~1–15B params | ~70B–trillions |
| Hosting | Single GPU, sometimes CPU/edge | Multi-GPU or API |
| Cost/latency | Low | High |
| Strength | Narrow, well-defined tasks after fine-tuning | Broad reasoning, generalization, few-shot |
| Weakness | Weak zero-shot generalization, poor complex reasoning | Cost, latency, overkill for simple tasks |

**When an SLM is the right answer** (a genuinely strong interview point, since candidates over-reach for the biggest model): classification, routing, extraction with a fixed schema, PII detection, or any high-volume task where a fine-tuned 3–7B model matches a large model's accuracy at a fraction of the cost. Distillation from a large teacher into a small student is the standard path — see [`Transformers_Cheatsheet.md`](Transformers_Cheatsheet.md) §16.

**Common production pattern — cascade/router**: a cheap SLM handles the easy majority of traffic and escalates only low-confidence or complex cases to a large model. Dramatically cuts cost at near-equal quality.

---

# 3. Inference Parameters

These control *sampling*, not model capability. Interviewers ask because misusing them is the most common applied-GenAI mistake.

| Parameter | What it does | Typical range |
|---|---|---|
| **temperature** | Divides logits before softmax. `→0` = deterministic/greedy; `>1` flattens the distribution → more random | 0 for extraction, 0.7–1.0 for creative |
| **top_p** (nucleus) | Sample from the smallest set of tokens whose cumulative probability ≥ p | 0.9–1.0 |
| **top_k** | Sample only from the k highest-probability tokens | 20–100 |
| **max_tokens** | Hard cap on generated length | task-dependent |
| **stop sequences** | Strings that halt generation when produced | e.g. `"\n\n"`, `"</answer>"` |
| **frequency_penalty** | Penalizes a token **proportionally to how many times it has already appeared**. Reduces verbatim repetition | 0–2 |
| **presence_penalty** | Penalizes a token **once, if it has appeared at all** (count-independent). Pushes toward new topics/vocabulary | 0–2 |
| **seed** | Makes sampling reproducible (best-effort) | — |
| **logit_bias** | Manually up/down-weight specific tokens | — |

**frequency vs presence penalty** — the distinction is the question:
> Both discourage reuse, but **frequency scales with the repeat count** (each additional occurrence is penalized more, so it targets loops and stuttering), while **presence is a flat one-time penalty** the moment a token appears at all (so it encourages topical diversity rather than just suppressing repetition).

**temperature vs top_p**: temperature reshapes the *whole* distribution; top_p *truncates* its tail. Convention is to tune one and leave the other at default — moving both together makes behavior hard to reason about.

**Gotcha**: For structured output (JSON, attribute extraction, classification labels) use **temperature 0** and, where available, constrained/structured decoding. Nonzero temperature on an extraction task is a self-inflicted reliability bug.

---

# 4. Prompt Engineering

| Technique | What it is | When |
|---|---|---|
| **Zero-shot** | Instruction only | Simple, well-known tasks |
| **Few-shot** | Include worked examples | Need a specific format or edge-case handling |
| **Chain-of-Thought (CoT)** | "Think step by step" — elicit intermediate reasoning | Multi-step arithmetic/logic |
| **Self-consistency** | Sample multiple CoT paths, majority-vote the answer | High-stakes reasoning; costs N× |
| **ReAct** | Interleave Reasoning + Acting (tool calls) | Agents needing external tools |
| **Role/persona** | "You are a compliance analyst…" | Steers tone and domain framing |
| **Structured output** | Demand JSON/schema; better, enforce it with constrained decoding | Anything machine-consumed |
| **Delimiters** | Fence untrusted input in `<data>…</data>` | Injection resistance, clarity |
| **Decomposition** | Split one hard prompt into chained simpler calls | Complex pipelines |
| **Rubric/criteria in prompt** | State explicit evaluation criteria | LLM-as-judge |

**Principles that actually matter:**
- Be specific about the *output contract* (format, length, what to do when unsure) — most "model failures" are underspecified prompts.
- Give an explicit escape hatch: "If the document doesn't contain the answer, reply exactly `NOT_FOUND`." Without one, the model invents something.
- Put instructions **before** long context, and critical content at the **start or end** — not buried in the middle (see "lost in the middle", §6).
- Few-shot examples should cover edge cases, not just the happy path.

**Prompt injection** — the security question. Retrieved documents, user input, and tool outputs are all **untrusted**. Mitigations: fence untrusted content in delimiters and say so, never let retrieved text alter system instructions, validate/parse output rather than executing it, least-privilege tools with human approval for destructive actions, and a separate guardrail classifier on input and output. Note that prompt-level defenses are mitigations, not guarantees — the real boundary is the permission model around the tools.

---

# 5. Embeddings & Vector Databases

**Embedding**: text → dense vector where semantic similarity ≈ geometric proximity. Retrieval compares vectors, usually by **cosine similarity** (magnitude-invariant, the default for text).

**ANN indexes** (exact nearest-neighbor search is O(n) — too slow at scale):

| Index | Idea | Trade-off |
|---|---|---|
| **HNSW** | Navigable small-world graph, greedy descent through layers | Fast + high recall; memory-hungry. Most common default |
| **IVF** | Cluster vectors, search only the nearest few clusters | Tunable speed/recall via `nprobe` |
| **IVF-PQ** | IVF + Product Quantization (compress vectors) | Huge memory savings, some accuracy loss |
| **ScaNN / DiskANN** | Anisotropic quantization / SSD-resident graphs | Very large corpora |

**Vector stores**: FAISS (library, in-process), Chroma (local/dev), Qdrant / Weaviate / Milvus (self-hosted services), pgvector (Postgres extension — attractive when you already run Postgres), Pinecone (managed).

**Things interviewers probe:**
- **Metadata filtering** — pre-filter (filter then search: exact but can under-fill results) vs post-filter (search then filter: fast but may return too few). Real systems need filtered ANN support.
- **Hybrid search** — combine dense (semantic) with sparse/BM25 (exact keyword). Fuse with **Reciprocal Rank Fusion (RRF)**. Essential when exact terms matter (product codes, legal citations, SKUs).
- **Chunking strategy** — fixed-size with overlap is the baseline; semantic/recursive splitting on document structure is better. Chunk too small and you lose context; too large and you dilute the embedding and waste context window.
- **Re-embedding cost** — changing embedding model means re-indexing the entire corpus. Version your index alongside the model.

---

# 6. RAG — Retrieval-Augmented Generation

## The basic pipeline

```
Query → [embed] → retrieve top-k → assemble context → LLM → answer (+ citations)
                        ↑
                  vector index (chunked, embedded corpus)
```

**Why RAG at all**: injects knowledge the model doesn't have (proprietary, post-cutoff, or too niche) without retraining; keeps facts updatable; enables citations and auditability. RAG adds **knowledge**, fine-tuning adds **behavior** — that framing answers most "RAG vs fine-tune" questions.

## Types of RAG

Grouped by what each one actually changes.

**Baseline**
1. **Naive / Standard RAG** — embed query, retrieve top-k, stuff into prompt. The reference point everything else improves on.
2. **Modular / Advanced RAG** — the pipeline decomposed into swappable stages (pre-retrieval, retrieval, post-retrieval, generation) rather than one fixed chain.

**Better retrieval**
3. **Hybrid RAG** — dense + sparse (BM25) retrieval fused, typically via RRF. Fixes semantic search's blindness to exact identifiers.
4. **Multi-Query RAG** — LLM rewrites the query into several paraphrases, retrieves for each, unions results. Fixes a single phrasing missing relevant docs.
5. **RAG-Fusion** — multi-query plus reciprocal-rank fusion to produce one properly ranked list.
6. **HyDE** (Hypothetical Document Embeddings) — LLM writes a *hypothetical answer* first, embeds **that**, and retrieves with it. Closes the asymmetry between a short question and long documents.
7. **Query decomposition RAG** — break a compound question into sub-questions, retrieve per sub-question, then synthesize. Handles multi-hop.
8. **Step-back RAG** — ask a more general question first to retrieve broader grounding, then answer the specific one.

**Better context assembly**
9. **Re-ranking RAG** — retrieve a wide candidate set cheaply (ANN), then re-score with a **cross-encoder** which reads query and document *together*. The single highest-ROI upgrade over naive RAG.
10. **Parent-document / small-to-big** — embed and match on small precise chunks, but pass the **larger parent** chunk to the LLM. Precision in retrieval, context in generation.
11. **Sentence-window** — retrieve on single sentences, expand to surrounding sentences for context.
12. **Contextual compression** — filter or summarize retrieved chunks down to just the query-relevant spans before generation. Saves context and reduces distraction.

**Control flow / agency**
13. **Self-RAG** — model decides *whether* to retrieve, then critiques its own retrieved evidence and output with reflection tokens.
14. **Corrective RAG (CRAG)** — a lightweight evaluator grades retrieval quality; on poor retrieval it falls back (query rewrite, or web search) instead of answering from bad context.
15. **Adaptive RAG** — a router classifies query complexity and picks the strategy: answer directly / single retrieval / multi-step.
16. **Agentic RAG** — retrieval exposed as one tool among several to an agent that plans, retrieves iteratively, and verifies. Most flexible, most expensive, hardest to evaluate.

**Different data structures**
17. **Graph RAG** — build a knowledge graph of entities/relations; traverse it for retrieval. Strong for multi-hop and "how do X and Y relate" questions that flat chunk retrieval fundamentally can't answer.
18. **RAPTOR / hierarchical RAG** — recursively cluster and summarize the corpus into a tree; retrieve at whatever abstraction level fits the query. Good for whole-document questions.
19. **Multimodal RAG** — retrieve over images/tables/audio alongside text via a shared or per-modality embedding space.
20. **Structured / Text-to-SQL RAG** — translate the question into a query against a database rather than retrieving prose.

**How to answer "which type would you use?"**: start naive, measure, then add the *specific* fix for the *observed* failure. Exact-term misses → hybrid. Question/document phrasing mismatch → HyDE or multi-query. Right topic, wrong chunk ranking → re-ranker. Multi-hop → decomposition or Graph RAG. Sometimes-unnecessary retrieval → adaptive/self-RAG. Naming a diagnostic path beats reciting the list.

## Problems with RAG

**Retrieval-side**
- Chunking artifacts — the answer straddles a chunk boundary, so no single chunk is sufficient.
- Query/document asymmetry — short question embeddings don't match long document embeddings well.
- Semantic search misses exact identifiers (SKUs, error codes, dates) → needs hybrid.
- Wrong granularity — retrieves overviews when specifics were asked for (or vice versa).
- Stale index — source updated, embeddings not re-indexed.
- Domain gap — a general-purpose embedding model on specialized jargon.

**Generation-side**
- **Retrieved but unused** — the answer *was* in context and the model ignored it, falling back on parametric knowledge. Diagnose by comparing answers with and without retrieval.
- **Lost in the middle** — content buried mid-context is attended to far less reliably than content at the edges. Order retrieved chunks accordingly.
- **Hallucination despite context** — fluent output not supported by sources.
- **Synthesis failure** — retrieves docs about A and B but summarizes each separately instead of comparing.
- **Conflicting sources** — no policy for which of two contradictory documents wins.

**System-side**
- Latency and cost stack up (embedding + retrieval + re-ranking + long-context generation).
- **Security**: retrieved documents are untrusted input — a poisoned document is a prompt-injection vector.
- Access control — retrieval must respect per-user permissions, or RAG becomes a data-leak channel.
- Evaluation is genuinely hard: you must measure retrieval and generation *separately* to know which half is broken.

---

# 7. LLM Evaluation

**The core principle**: evaluate the stages separately. A single end-to-end score tells you *that* something is wrong, never *what*.

**Reference-based (you have a gold answer)**
- Exact match / F1 — extraction, classification, closed QA.
- **ROUGE** (recall-oriented n-gram overlap) — summarization. **BLEU** (precision-oriented) — translation. Both are weak proxies for meaning.
- **BERTScore** — embedding similarity instead of n-gram overlap; better semantic sensitivity.

**Reference-free (no gold answer)**
- **LLM-as-judge** — a model scores output against a rubric. Scales well; known biases: position bias (favors first option), verbosity bias (favors longer answers), self-preference (favors its own family's style). Mitigate by randomizing order, using a rubric with explicit criteria, requiring reasoning before the score, and calibrating against a human-labeled subset.
- **Pairwise preference / win rate** — more reliable than absolute 1–5 scoring, which suffers severe scale compression.

**RAG-specific (the RAGAS-style quadrant)**
| Metric | Question it answers | Which half it blames |
|---|---|---|
| **Context precision** | Are retrieved chunks relevant (and ranked well)? | Retrieval |
| **Context recall** | Was all needed evidence retrieved? | Retrieval |
| **Faithfulness / groundedness** | Is every claim supported by the retrieved context? | Generation |
| **Answer relevance** | Does the answer address the actual question? | Generation |

High context recall + low faithfulness = the retriever is fine, the generator is hallucinating. Low context recall = fix retrieval; nothing downstream can compensate.

**Operational metrics** (interviewers like that you track these too): TTFT (time to first token), tokens/sec, p50/p95/p99 latency, cost per request, cache hit rate, error/refusal rate.

**Other**
- Perplexity — pretraining/LM quality only; near-useless for judging instruction-following.
- Safety: toxicity, bias, jailbreak resistance, PII leakage.
- Public benchmarks: MMLU (knowledge), GSM8K (math), HumanEval (code), MT-Bench / Chatbot Arena Elo (chat). Treat with suspicion — contamination is rampant and they rarely predict your task.

**What to actually build**: a **golden set** stratified across real task types, retrieval and generation graded separately, LLM-as-judge calibrated against human review, and **regression gates in CI** that block a deploy when a metric drops. Offline benchmarks alone don't predict production; pair them with online A/B and staged rollout.

---

# 8. Fine-Tuning LLMs

## Decide *whether* to fine-tune first

```
Prompt engineering  →  RAG  →  Fine-tuning
   (hours)            (days)      (weeks)
```
Work down this list, not up. Fine-tune when: you need consistent format/style/behavior at scale; the domain has its own vocabulary or reasoning conventions; you want a smaller/cheaper model to match a big one's task quality; or prompts have grown so long that latency and cost are unacceptable.

Do **not** fine-tune to add facts that change — that's RAG's job, and retraining on every change is untenable.

## Approaches

| Approach | Trains | Notes |
|---|---|---|
| **Full fine-tuning** | All weights | Highest ceiling; needs optimizer state for every parameter (Adam ≈ 2–8× params in memory) |
| **LoRA** | Low-rank adapters (`W + BA`) | ~0.1–1% of params; orders of magnitude cheaper; adapters hot-swappable at serving time |
| **QLoRA** | LoRA over a 4-bit quantized frozen base | Fine-tune large models on a single GPU |
| **Adapters** | Small inserted bottleneck layers | Similar spirit to LoRA |
| **Prefix / prompt tuning** | Continuous "virtual tokens" only | Smallest footprint, least expressive |

Mechanics and hyperparameters (`r`, `alpha`, target modules) are in [`Transformers_Cheatsheet.md`](Transformers_Cheatsheet.md) §12.

## Data — where fine-tuning actually succeeds or fails

Quality and consistency beat volume. A few thousand clean, consistent examples usually beat a hundred thousand noisy ones. Requirements:
- Consistent formatting — the model learns your inconsistencies faithfully.
- Coverage of edge cases and refusals, not just the happy path.
- A held-out set drawn from the **same distribution as production**, split *before* any preprocessing.
- Deduplication — near-duplicates across train/test silently inflate your scores.

## Training-side pitfalls

- **Catastrophic forgetting** — narrow fine-tuning degrades general capability. Mitigate with fewer epochs (single-epoch is a common deliberate choice), lower LR, LoRA over full FT, and mixing in some general-domain data.
- **Learning rate**: use far lower than pretraining (often 1e-5 to 1e-4 for LoRA); too high destroys the pretrained representations you're paying for.
- **Overfitting**: watch validation loss, not training loss; stop early.
- Evaluate against the *pre-fine-tuning* model on a general benchmark too, so you can see what you broke.

## Preference alignment (after SFT)

**RLHF/PPO** (reward model + policy optimization, needs a critic), **DPO** (directly optimizes a classification loss on preference pairs, no reward model — simpler and stable), **GRPO** (advantage from comparing a *group* of sampled outputs, so no separate critic model). Details in [`Transformers_Cheatsheet.md`](Transformers_Cheatsheet.md) §11.

---

# 9. Agentic AI

**Agent = LLM + tools + a loop + memory.** The LLM decides which tool to call, observes the result, and repeats until the goal is met or a limit is hit.

```
Goal → [plan] → select tool → execute → observe → reflect → (loop) → answer
                     ↑                                          │
                     └──────────────── memory ──────────────────┘
```

**Core components**
- **Tools** — typed function schemas (name, description, parameters). Description quality drives selection accuracy more than anything else.
- **Planning** — ReAct (interleaved reason/act), plan-and-execute (full plan upfront, then execute), or tree/graph search for harder problems.
- **Memory** — short-term (conversation/scratchpad), long-term (vector store of past episodes), and structured state.
- **Reflection** — self-critique of intermediate results before proceeding.

**Patterns**
- Single agent with tools — the default; start here.
- **Router** — classify the request, dispatch to a specialist.
- **Supervisor / orchestrator-worker** — a coordinator delegates subtasks to focused sub-agents.
- **Multi-agent debate / critic** — independent agents critique each other to catch errors.
- **Reflexion** — attempt, self-critique, retry with the critique in context.

**Failure modes interviewers want you to name**
- **Error compounding** — 90% per-step reliability over 10 steps ≈ 35% end-to-end. This is *the* fundamental agent problem.
- Infinite loops / repeated identical tool calls → need step budgets and loop detection.
- Wrong tool selection from vague tool descriptions.
- Context growth — the trace fills the window; needs summarization or pruning.
- Cost/latency unpredictability — a single request can fan out to dozens of LLM calls.
- Unsafe actions — an agent with write access can do real damage; hence least privilege and human-in-the-loop for irreversible operations.

**Design principles**: constrain the action space (fewer, well-described tools), make tools idempotent and validate their arguments, cap steps and spend, log every step for auditability, and require human approval for destructive/outward-facing actions. Prefer a deterministic pipeline over an agent whenever the workflow is actually known in advance — "an agent" is not automatically the better design, and saying so is a maturity signal.

---

# 10. Model Selection

**Caution on specifics**: model lineups, context windows, and pricing change fast — verify current numbers against provider docs rather than trusting a memorized table (including this one). What interviewers actually assess is your *selection framework*, so lead with that.

**Generational shape** (GPT family as the example the question usually names): GPT-3 established that scale plus few-shot prompting works but was not instruction-tuned. GPT-3.5 added instruction tuning/RLHF, which is what made chat usable. GPT-4 brought a large jump in reasoning, reliability, and instruction-following, plus multimodality and much longer context. Subsequent generations (4o, then the 5 family) have pushed further on reasoning quality, multimodality, latency, and cost-efficiency, with explicit reasoning-effort controls becoming a common feature. The consistent pattern across generations: **better reasoning, longer context, lower cost for equivalent capability, and more modalities** — with each generation's flagship eventually matched by a later cheaper tier.

**The framework that actually answers "which would you choose?"**

1. **Start from task difficulty.** Classification, routing, extraction, and short summarization rarely need a flagship model — a small/cheap tier or a fine-tuned SLM usually matches it. Multi-step reasoning, ambiguous instructions, code generation, and agentic planning justify the frontier tier.
2. **Then apply constraints**: latency budget (interactive vs batch), cost at your actual volume, context length needed, multimodality, data residency / self-hosting requirements, and whether you need fine-tuning control.
3. **Route, don't standardize.** Production systems mix tiers: cheap model for the bulk, escalate on low confidence. Same principle as the SLM cascade in §2.
4. **Decide open-weight vs API** on: data residency and privacy, cost at scale (self-hosting wins at high sustained volume, loses at low/spiky volume), fine-tuning and quantization control, latency floor, and ops burden.
5. **Benchmark on *your* task.** Public benchmarks are contaminated and rarely correlate with a specific production task. Build the golden set (§7) and measure candidates on it — this is the answer that actually lands.

---

# 11. Rapid-Fire

**Q: RAG or fine-tuning?**
RAG for knowledge (changing facts, need citations, auditability). Fine-tuning for behavior (format, style, domain conventions, distilling into a smaller model). They compose — fine-tune *how* to answer, retrieve *what* to answer with.

**Q: Why is my RAG retrieving well but answering badly?**
The gap is downstream of retrieval: chunk boundaries splitting the answer, lost-in-the-middle ordering, the model preferring parametric knowledge, or synthesis failure across multiple documents. Diagnose by manually categorizing 20–30 failures before changing anything.

**Q: frequency_penalty vs presence_penalty?**
Frequency scales with how many times a token already appeared (kills loops/stuttering); presence is a flat one-time penalty on any token already present (pushes topical novelty).

**Q: How do you stop hallucination?**
You reduce it, you don't eliminate it: ground generation in retrieved/verified data, require citations to specific spans, temperature 0 for factual tasks, a separate verifier model checking claims against sources, and a confidence threshold routing low-confidence output to human review.

**Q: How would you evaluate a RAG system?**
Separately. Retrieval: context precision/recall. Generation: faithfulness and answer relevance. Plus a stratified golden set, LLM-as-judge calibrated against human labels, and CI regression gates.

**Q: Cheapest big win on a slow/expensive LLM feature?**
Usually: cache aggressively (including prompt-prefix caching), then use a smaller model for the easy majority with escalation, then batch, then quantize. Structural routing beats micro-optimizing prompts.

**Q: Biggest risk of an agent in production?**
Compounding per-step error, and unconstrained actions. Fix with step/spend budgets, a minimal well-described tool set, idempotent validated tools, full trace logging, and human approval gates on anything irreversible.

**Q: When is a small model the right call?**
High-volume narrow tasks with a fixed output schema, tight latency budgets, on-prem/edge constraints, or when a fine-tuned SLM has been measured to match the large model on *your* golden set.
