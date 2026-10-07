# Generative AI / LLM

The candidate's professional background included GenAI, so they *"prepared heavily around LLM systems"* 📄 — while also giving the sharpest warning in the whole post about doing exactly that.

---

## The topics named 📄

| Group | Topics |
|---|---|
| Architecture | Transformers, Attention, Context windows |
| Retrieval | Embeddings, Vector databases, RAG, Chunking, Retrieval, Reranking |
| Steering the model | Prompt engineering, Fine-tuning, LoRA/PEFT |
| Judging it | Evaluation, Hallucination |
| Running it | Inference optimization, LLM observability, Production LLM architecture |
| Agents | Agentic systems, Multi-agent architectures, Tool calling |

## The warning worth reading twice 📄

Listed under *what I'd do differently*:

> **"Don't over-index on GenAI.** LLM knowledge is valuable, but strong fundamentals still matter."

And in the takeaways:

> "Strong ML fundamentals still matter even if you work in GenAI."

This is a candidate **whose own job is GenAI** saying they'd shift time *away* from it. Two of the five technical areas they prepared — ML fundamentals and CS fundamentals — are things a GenAI specialist is most likely to have let decay, and those are the ones that bit. If your recent work is all LLMs, the highest-return prep is probably [02](02_ML_Fundamentals.md) and [05](05_Distributed_Systems_CS.md), not this file.

## Same question shape as everywhere else 📄

> "The important part wasn't simply knowing terminology. You need to be able to answer:
> - *Why this architecture?*
> - *What happens when the system doesn't work?*
> - *How would you measure whether it's actually better?*"

Those three map cleanly onto the weak spots in most GenAI answers 🧩:

**"Why this architecture"** punishes cargo-culting. If you reach for RAG, you should be able to say why not fine-tuning, and vice versa — RAG injects *knowledge* that changes often and needs citations; fine-tuning installs *behaviour*, format and domain convention. If you propose an agent, you should be able to say why a deterministic pipeline wouldn't do, because usually it would.

**"What happens when it doesn't work"** is the production question. For RAG the honest answer is that failure is layered: retrieval can miss, retrieval can succeed and the model ignore it, chunk boundaries can split the answer, or the model can hallucinate over good context. You diagnose by grading retrieval and generation *separately* — which is also the answer to the third question.

**"How would you measure whether it's better"** is where most candidates have nothing. "It seemed better" is not an answer. The credible version: a stratified golden set, retrieval and generation scored independently, an LLM-as-judge calibrated against human labels, and regression gates in CI.

## Where the depth lives 🔗

Everything on that topic list is covered:

| Topics | Where |
|---|---|
| Transformers, attention, context windows, RoPE, KV cache | [`Transformers_Cheatsheet`](../ML_Fundamentals/Transformers_Cheatsheet.md) |
| Embeddings, vector DBs, ANN indexes, chunking | [`GenAI_Cheatsheet`](../ML_Fundamentals/GenAI_Cheatsheet.md) §5 |
| **RAG + reranking** | [`GenAI_Cheatsheet`](../ML_Fundamentals/GenAI_Cheatsheet.md) §6 — ~20 named variants and the failure modes |
| Prompt engineering (+ injection) | [`GenAI_Cheatsheet`](../ML_Fundamentals/GenAI_Cheatsheet.md) §4 |
| Fine-tuning, LoRA/PEFT | [`GenAI_Cheatsheet`](../ML_Fundamentals/GenAI_Cheatsheet.md) §8 · [`Transformers_Cheatsheet`](../ML_Fundamentals/Transformers_Cheatsheet.md) §11–12 |
| **Evaluation + hallucination** | [`GenAI_Cheatsheet`](../ML_Fundamentals/GenAI_Cheatsheet.md) §7 — the RAGAS quadrant, LLM-as-judge biases |
| Inference optimization | [`Transformers_Cheatsheet`](../ML_Fundamentals/Transformers_Cheatsheet.md) §15–16 · [`../Example_Company/12_MLOps_CICD_LLM_Hosting.md`](../Example_Company/12_MLOps_CICD_LLM_Hosting.md) |
| **Agentic + multi-agent + tool calling** | [`GenAI_Cheatsheet`](../ML_Fundamentals/GenAI_Cheatsheet.md) §9 |
| LLM observability | [`../AI_Engineer/Q8_Inference_Observability_and_Fallbacks.md`](../AI_Engineer/Q8_Inference_Observability_and_Fallbacks.md) |
| Production LLM architecture | [`../AI_Architect/Q1_Multi_Agent_Consistency.md`](../AI_Architect/Q1_Multi_Agent_Consistency.md) · [`../AI_Engineer/Q2_Latency_Optimization.md`](../AI_Engineer/Q2_Latency_Optimization.md) |

## The follow-up ladder, applied here 🧩

- *"You'd use RAG — why not fine-tune?"* → knowledge vs behaviour → *"what if the knowledge is stable and the format isn't?"* → then fine-tune, or both → *"how would you know which one helped?"* → ablate, measured on the golden set
- *"Retrieval looks good but answers are wrong. What now?"* → separate retrieval and generation metrics → *"context precision is high, faithfulness is low — what does that tell you?"* → retriever is fine, generator is hallucinating → *"so what do you change?"* → grounding, citations, a verifier pass, lower temperature
- *"Your agent works in a demo. What breaks at scale?"* → compounding per-step error (90% over 10 steps ≈ 35%) → *"how do you contain it?"* → step budgets, idempotent validated tools, a narrow tool surface, human approval on irreversible actions
