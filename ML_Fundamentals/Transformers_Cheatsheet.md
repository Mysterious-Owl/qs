# Transformers — Deep Dive Cheat Sheet

The architecture behind every modern LLM. Companion to [`DL_Concepts_Cheatsheet.md`](DL_Concepts_Cheatsheet.md). Given the Example Company Catalog team's GenAI focus ([`../Example_Company/08_Attribute_Extraction_Content_Generation.md`](../Example_Company/08_Attribute_Extraction_Content_Generation.md), [`../Example_Company/10_LLM_Finetuning_RAG_GenAI.md`](../Example_Company/10_LLM_Finetuning_RAG_GenAI.md)), expect this to be probed in depth — be able to draw it and explain *why* each component exists.

---

# Part 1 — The Core Mechanism

## 1. Why Transformers Won

The problem with RNNs/LSTMs:
1. **Sequential computation** — timestep `t` needs `t-1` finished, so training can't be parallelized across the sequence
2. **Long-range dependencies** — information degrades over many steps even with gating
3. **Fixed-size bottleneck** in vanilla seq2seq

Transformers fix all three: self-attention connects every position to every other position in **one operation** (constant path length between any two tokens, vs. O(n) for an RNN), and every position is computed in parallel.

**The trade-off**: attention is O(n²) in sequence length, whereas an RNN is O(n). You traded a sequential bottleneck for a quadratic memory/compute one — which is why long-context efficiency is such an active research area.

## 2. Self-Attention — The Central Equation

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**What each piece is** — Q, K, V are all linear projections of the same input (`Q = XW_Q`, `K = XW_K`, `V = XW_V`):
- **Query**: "what am I looking for?"
- **Key**: "what do I contain?" — matched against queries
- **Value**: "what do I actually contribute if attended to?"

**Step by step:**
1. `QKᵀ` — dot product every query against every key → an `n×n` matrix of raw relevance scores
2. `/√d_k` — scale down (see below)
3. `softmax` — normalize each row into attention weights summing to 1
4. `· V` — weighted sum of value vectors → the output for each position

**Why divide by √d_k** (near-guaranteed question): as dimensionality `d_k` grows, dot products of random vectors grow in magnitude with variance proportional to `d_k`. Large inputs push softmax into a saturated regime where one weight ≈ 1 and the rest ≈ 0, making gradients vanish. Dividing by `√d_k` normalizes the variance back to ~1, keeping softmax in a well-conditioned range.

**Complexity**: O(n²·d) time and O(n²) memory for the attention matrix — the fundamental scaling constraint.

## 3. Multi-Head Attention

```
MultiHead(Q,K,V) = Concat(head₁, ..., head_h)·W_O
where headᵢ = Attention(QW_Qⁱ, KW_Kⁱ, VW_Vⁱ)
```

**Why multiple heads instead of one big one**: a single softmax attention distribution can really only emphasize one kind of relationship per layer. Splitting `d_model` into `h` heads of dimension `d_k = d_model/h` lets different heads specialize — some attend to syntactic dependencies, some to positional adjacency, some to semantic relatedness — then results are concatenated and projected back to `d_model`.

**Key point**: total parameters stay roughly the same as one full-width head, because each head is `d_model/h` wide. You get representational diversity essentially for free.

## 4. Attention Variants

| Variant | What's different | Where used |
|---|---|---|
| **Self-attention** | Q, K, V all from the same sequence | Encoder and decoder blocks |
| **Cross-attention** | Q from the decoder, K/V from the encoder | Encoder-decoder models (translation) |
| **Causal / masked self-attention** | Future positions masked to `-∞` before softmax so they get zero weight | Decoder-only LMs (GPT-style) — this is what makes autoregressive generation valid |
| **Multi-Query Attention (MQA)** | All heads share one K/V head | Cuts KV-cache memory dramatically for inference |
| **Grouped-Query Attention (GQA)** | Heads share K/V in groups — a middle ground between MHA and MQA | Llama 2/3 and most modern LLMs |
| **Flash Attention** | Mathematically identical output, but an IO-aware kernel that tiles the computation to avoid materializing the full n×n matrix in HBM | Standard for efficient training/inference |

**Gotcha**: Flash Attention is **not an approximation** — it computes exact attention, just with a far better memory-access pattern. It changes speed/memory, not results. (Referenced in Example Company's own published fine-tuning stack — see [`../Example_Company/08_Attribute_Extraction_Content_Generation.md`](../Example_Company/08_Attribute_Extraction_Content_Generation.md).)

---

# Part 2 — The Full Block

## 5. Anatomy of a Transformer Block

```
        Input (embeddings + positional info)
                    │
        ┌───────────▼────────────┐
        │  Multi-Head Attention  │
        └───────────┬────────────┘
                    │
              Add & Norm  ◄──── residual from input
                    │
        ┌───────────▼────────────┐
        │  Feed-Forward Network  │   FFN(x) = W₂·GELU(W₁x + b₁) + b₂
        └───────────┬────────────┘
                    │
              Add & Norm  ◄──── residual
                    │
                 Output          (repeat N× — 12 to 100+ layers)
```

**The FFN**: two linear layers with a non-linearity, applied **independently to each position**. Inner dimension is typically `4×d_model`. Attention mixes information *across* positions; the FFN processes each position's representation *in isolation*. Roughly two-thirds of a transformer's parameters live in the FFN layers.

**Residual connections**: essential — give gradients a direct path through dozens of layers (same principle as ResNet, see [`DL_Concepts_Cheatsheet.md`](DL_Concepts_Cheatsheet.md) §15).

**Layer Norm, not Batch Norm**: LayerNorm normalizes across the feature dimension per sample, so it's independent of batch size and sequence length — critical for variable-length sequences and small batches.

## 6. Pre-LN vs. Post-LN

- **Post-LN** (original 2017 paper): `x + Sublayer(x)` → then LayerNorm. Needs careful warmup; unstable when very deep.
- **Pre-LN** (modern standard): LayerNorm *first*, then the sublayer, then add. `x + Sublayer(LN(x))`. Much more stable training at depth because the residual path stays clean and unnormalized.

**Gotcha**: Nearly all modern LLMs use Pre-LN. Knowing this distinction (and that it's about training stability at depth) is a strong depth signal.

## 7. Positional Encoding

Self-attention is **permutation-invariant** — shuffle the input tokens and you get correspondingly shuffled outputs with identical values. Without positional information, "dog bites man" and "man bites dog" would be indistinguishable. Hence:

| Method | How | Notes |
|---|---|---|
| **Sinusoidal** (original) | Fixed `sin`/`cos` of varying frequencies added to embeddings | No parameters; some extrapolation to unseen lengths |
| **Learned absolute** | A trainable embedding per position | Used by BERT/GPT-2; can't extrapolate past trained max length |
| **RoPE** (Rotary) | *Rotates* Q and K by a position-dependent angle, so the dot product naturally encodes *relative* position | **Standard in modern LLMs** (Llama, Mistral, Qwen); extrapolates better and can be extended via scaling |
| **ALiBi** | Adds a linear distance-based penalty directly to attention scores | Strong length extrapolation |

**Gotcha**: RoPE's key property is that it makes attention scores depend on *relative* position (`m-n`) rather than absolute positions — which is both more natural for language and better for length generalization.

---

# Part 3 — Model Families

## 8. Encoder-only (BERT-style)
- **Attention**: Bidirectional — every token sees every other token
- **Pretraining**: Masked Language Modeling (mask ~15% of tokens, predict them) + (originally) Next Sentence Prediction
- **Good at**: Classification, NER, extraction, embeddings/retrieval — anything where you need to *understand* a complete input
- **Can't do**: Autoregressive generation (bidirectional attention means it has already "seen" the future)
- **Examples**: BERT, RoBERTa, DeBERTa, ModernBERT
- **Relevance**: Product attribute *extraction* and compliance *classification* are understanding tasks — an encoder model is often the right, far cheaper choice over an LLM. Worth raising in a system-design answer ([`../Example_Company/07_Content_Compliance_Moderation.md`](../Example_Company/07_Content_Compliance_Moderation.md), [`../Example_Company/08_Attribute_Extraction_Content_Generation.md`](../Example_Company/08_Attribute_Extraction_Content_Generation.md)).

## 9. Decoder-only (GPT-style)
- **Attention**: Causal/masked — each token attends only to itself and prior tokens
- **Pretraining**: Next-token prediction
- **Good at**: Generation, few-shot/in-context learning, instruction following — and, at scale, essentially everything
- **Examples**: GPT family, Llama, Claude, Mistral, Gemini (broadly)
- **Why decoder-only dominates**: Simpler objective, scales cleanly, and next-token prediction on enough data turns out to subsume most understanding tasks

## 10. Encoder-Decoder (T5/BART-style)
- **Structure**: Encoder reads input bidirectionally; decoder generates while cross-attending to the encoder output
- **Good at**: Sequence-to-sequence with a clean input/output split — translation, summarization
- **Examples**: T5, BART, FLAN-T5
- **Trade-off**: More parameters and complexity than decoder-only, and decoder-only models have largely caught up by simply conditioning on the input as a prompt

---

# Part 4 — Training & Adapting

## 11. The Training Stages

```
Pretraining  →  Supervised Fine-Tuning  →  Preference Alignment
(next-token      (instruction/response      (RLHF / DPO / GRPO)
 on trillions     pairs, teaches format
 of tokens)       and task behavior)
```

- **Pretraining**: self-supervised next-token prediction; where nearly all capability comes from. Enormously expensive.
- **SFT**: supervised on curated (prompt, ideal response) pairs. Teaches format, style, and task-following.
- **Preference alignment**: optimizes toward human preferences rather than a fixed target string.

| Method | Core idea |
|---|---|
| **RLHF (PPO)** | Train a reward model on human preference pairs, then optimize the policy with PPO against it — needs a separate learned value/critic model |
| **DPO** | Skips the explicit reward model — directly optimizes a classification-style loss on preference pairs. Simpler, stable, popular |
| **GRPO** | Estimates advantage by comparing a *group* of sampled outputs against each other's reward, eliminating the need for a separate critic model |


## 12. Parameter-Efficient Fine-Tuning (PEFT)

| Method | How |
|---|---|
| **LoRA** | Freeze base weights; learn low-rank matrices `A` (r×d) and `B` (d×r) so the effective update is `W + BA`. Trainable params drop by orders of magnitude |
| **QLoRA** | LoRA on top of a 4-bit quantized frozen base model — fine-tune large models on a single GPU |
| Adapters | Insert small trainable bottleneck layers between frozen blocks |
| Prefix/Prompt tuning | Learn continuous "virtual token" embeddings prepended to the input; base model fully frozen |

**Why LoRA works**: the weight *update* needed to adapt a model to a task is empirically low-rank — you don't need full-rank freedom to specialize a pretrained model.

**Key hyperparameters**: `r` (rank, typically 8-64), `alpha` (scaling), and which modules to target (attention projections at minimum; adding FFN layers helps for bigger behavioral shifts).

**Serving advantage**: many task-specific adapters can share one base model in GPU memory and be hot-swapped per request — see [`../Example_Company/12_MLOps_CICD_LLM_Hosting.md`](../Example_Company/12_MLOps_CICD_LLM_Hosting.md).

## 13. Catastrophic Forgetting

Fine-tuning narrowly degrades general pretrained capability. Mitigations: fewer epochs (single-epoch training is a deliberate choice in Example Company's own published stack), lower learning rate, LoRA (most weights stay frozen), mixing in general-domain data, and distillation from a strong general teacher.

---

# Part 5 — Inference & Serving

## 14. Decoding Strategies

| Strategy | How | When |
|---|---|---|
| Greedy | Always take the argmax token | Deterministic extraction tasks |
| Beam search | Keep top-k partial sequences | Translation; tends toward bland output for open-ended text |
| **Temperature** | Divide logits by `T` before softmax — `T<1` sharpens, `T>1` flattens | The main creativity/determinism dial |
| **Top-k** | Sample only from the k highest-probability tokens | Caps the tail |
| **Top-p (nucleus)** | Sample from the smallest set whose cumulative probability ≥ p | Adapts to how peaked the distribution is — usually preferred over top-k |

**Gotcha**: For structured extraction (product attributes, JSON output), use low/zero temperature — you want determinism, not creativity. For copy generation, higher. Knowing when each applies is the practical point ([`../Example_Company/08_Attribute_Extraction_Content_Generation.md`](../Example_Company/08_Attribute_Extraction_Content_Generation.md)).

## 15. The KV Cache

During autoregressive generation, the K and V projections of all previous tokens are recomputed at every step unless cached. The **KV cache** stores them, turning per-token cost from O(n²) into O(n).

**The cost**: cache size = `2 · n_layers · n_heads · d_head · seq_len · batch · bytes` — this is often the dominant memory consumer in LLM serving, and it's exactly what MQA/GQA and PagedAttention (vLLM) exist to reduce.

**Two inference phases**:
- **Prefill**: process the whole prompt in parallel — compute-bound
- **Decode**: generate one token at a time — memory-bandwidth-bound

**Gotcha**: The fact that decode is memory-bandwidth-bound (not compute-bound) is *why* batching helps throughput so much — you're reusing the same weight reads across more sequences.

## 16. Serving Optimizations

| Technique | What it buys |
|---|---|
| **Continuous batching** | New requests join the batch as others finish, instead of waiting for a whole batch to complete — the single largest throughput win |
| **PagedAttention** (vLLM) | Manages KV cache in non-contiguous pages like OS virtual memory — near-eliminates fragmentation waste |
| **Quantization** | int8/int4 weights → less memory, higher throughput, small quality cost |
| **Speculative decoding** | A small draft model proposes several tokens; the large model verifies them in one pass |
| **Distillation** | Train a smaller student on the teacher's outputs — cheapest long-run serving win |
| Prefix/prompt caching | Reuse the KV cache for a shared prompt prefix across requests |

If you have hands-on serving experience, this table is the vocabulary to make it concrete.

## 17. Long Context

The O(n²) attention cost is the constraint. Approaches: sparse/local-window attention (Longformer, Mistral's sliding window), RoPE scaling (extend a trained context window post-hoc), memory/retrieval augmentation (RAG instead of brute-force long context), and Flash Attention (doesn't change complexity, but massively reduces the memory constant).

**Gotcha**: **"Lost in the middle"** — models reliably attend to the beginning and end of long contexts but degrade on information buried in the middle. A direct practical implication: in RAG, put the most relevant retrieved chunks at the *edges*, not the middle ([`../Example_Company/10_LLM_Finetuning_RAG_GenAI.md`](../Example_Company/10_LLM_Finetuning_RAG_GenAI.md)).

---

# Part 6 — Rapid-Fire Questions

**Q: Why does self-attention need positional encoding?**
It's permutation-invariant — without it, token order carries no information.

**Q: Why scale by √d_k?**
Dot-product variance grows with `d_k`; unscaled, softmax saturates and gradients vanish.

**Q: Why LayerNorm instead of BatchNorm?**
LayerNorm is independent of batch size and sequence length, and behaves identically at train and inference — BatchNorm needs batch statistics and is unstable with small/variable batches.

**Q: Where are most parameters?**
The FFN layers (~2/3), with inner dimension typically 4×d_model.

**Q: Encoder-only vs. decoder-only — how do I choose?**
Understanding/classification/embedding → encoder (cheaper, bidirectional). Generation or broad general capability → decoder-only.

**Q: What's the actual bottleneck in LLM inference?**
Memory bandwidth during decode, plus KV-cache size — not raw FLOPs.

**Q: How does attention differ from an RNN for long-range dependencies?**
Constant path length between any two positions (one attention hop) vs. O(n) sequential steps, so no signal degradation over distance — at the cost of O(n²) compute.

**Q: What does multi-head attention actually add?**
Multiple representation subspaces attending to different relationship types simultaneously, at roughly the same total parameter cost as one full-width head.

**Q: Why is Flash Attention faster?**
IO-aware tiling avoids materializing the n×n attention matrix in slow HBM. Exact, not approximate.

**Q: LoRA vs. full fine-tuning?**
LoRA trains orders of magnitude fewer parameters (low-rank update), fits on far less hardware, resists catastrophic forgetting, and allows adapter hot-swapping at serving time. Full fine-tuning has a higher capability ceiling when you have the data and compute to justify it.
