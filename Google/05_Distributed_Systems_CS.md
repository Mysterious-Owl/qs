# Distributed Systems / CS Fundamentals

The blind spot the candidate flags most sharply 📄:

> "One thing I underestimated initially was how useful fundamental CS knowledge is for ML interviews. […] Some of these topics hadn't been part of my day-to-day work for a while, so I had to rebuild some fundamentals. **This was actually one of the most useful parts of my preparation.**"

And it reappears in *what I'd do differently*: **"Don't neglect CS fundamentals. Especially if your recent work has been heavily focused on ML/LLMs."**

When a commenter pushed back that the topic list was impossibly broad — *"backend SWE + distributed systems + ML + GenAI + DSA"* — the author's reply was the useful part 📄:

> "I know the list is huge, but believe me these topics would make sense in ML design rounds when it comes to scaling things."

So this isn't a separate systems round. It's the substrate the **ML system design** round stands on. You're not asked "what is sharding" — you're asked how your feature store handles 50k QPS, and sharding is the vocabulary you need to answer.

---

## The topics named 📄

| Group | Topics |
|---|---|
| Hardware | CPU vs GPU, Memory |
| Concurrency | Processes vs threads, Concurrency |
| Scale | Distributed computing, Sharding, Replication, Fault tolerance |
| Plumbing | Networking basics, Caching, Queues |
| Persistence | Databases, Storage |

## Why each one shows up in an ML design round 🧩

The mapping is what makes this list tractable — each item has a specific ML moment where it becomes the answer:

| Topic | Where it surfaces in an ML round |
|---|---|
| **CPU vs GPU** | Why training is GPU-bound but serving often isn't; batch size vs utilisation; when a GPU is wasted money |
| **Memory** | Why a model doesn't fit; optimizer state being 2–8× parameters for Adam; KV-cache growth dominating LLM serving |
| **Processes vs threads** | Python's GIL — why threads help I/O-bound serving but not CPU-bound training; why data loaders use processes |
| **Concurrency** | Request batching, async feature fetches, the race between a model reload and in-flight requests |
| **Distributed computing** | Data vs model vs pipeline parallelism; all-reduce; why a 10-node job isn't 10× faster |
| **Networking** | Why cross-region feature fetches blow a latency budget; gradient sync as the bottleneck in distributed training |
| **Caching** | Precomputing embeddings; caching predictions for repeat inputs; and the staleness this buys you |
| **Databases / storage** | Feature store design; columnar formats for training reads vs row/key-value for serving reads |
| **Queues** | Decoupling ingestion from scoring; absorbing traffic spikes; async inference for non-interactive workloads |
| **Sharding** | Partitioning a feature store or embedding index; partitioning by entity so writes are single-writer |
| **Replication** | Read scaling for serving; and the consistency cost — a replica may serve a stale feature |
| **Fault tolerance** | What serves when the model is down (cached prediction? rules fallback? fail open or closed?); checkpoint/resume for long training |

That last one is the most commonly missed 🧩. "What happens when the model service is unavailable?" has a real answer — a heuristic fallback, a cached last-known prediction, or failing open vs closed depending on whether a miss or a false positive is worse — and most candidates have never thought about it.

## Where the depth lives 🔗

- [`../Example_Company/03_Distributed_Systems_Multithreading.md`](../Example_Company/03_Distributed_Systems_Multithreading.md) — processes vs threads, the GIL, deadlock prevention, and the mapping from single-process primitives to distributed equivalents (locks → leases, queues → Kafka, thread pools → worker fleets)
- [`../ML_Engineer/Q3_Distributed_Training.md`](../ML_Engineer/Q3_Distributed_Training.md) — the memory breakdown and data/tensor/pipeline/FSDP trade-offs
- [`Transformers_Cheatsheet`](../ML_Fundamentals/Transformers_Cheatsheet.md) §15–16 — KV cache, batching, quantization, distillation
- [`../AI_Architect/Q2_Scaling_Strategy.md`](../AI_Architect/Q2_Scaling_Strategy.md) · [`../AI_Architect/Q4_Reliability_Design.md`](../AI_Architect/Q4_Reliability_Design.md)
- [`../ML_Fundamentals/Glossary.md`](../ML_Fundamentals/Glossary.md) §8 — MLOps vocabulary

## Honest gap in this repo 🧩

The post's list includes plain **networking basics**, **databases/storage** and **queues** as general CS topics. This repo covers them only where they touch ML (feature stores, streaming ingestion, serving). If your CS fundamentals are genuinely rusty rather than just ML-adjacent-rusty, that's a gap to fill from a systems resource — it isn't here, and pretending otherwise would waste your time.
