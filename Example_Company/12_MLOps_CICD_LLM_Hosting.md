# MLOps, Deployment, CI/CD & LLM Hosting

Referral tip: "jyadatar MLE hi hire kiye jate h, to deployment CICD ke related v puchh h, LLM hosting etc" — most hires lean MLE, so deployment/CI-CD and LLM hosting questions are likely. 🔵 Public research found a real gap here — no corroborated leaked question with specifics (e.g., "how would you serve a 7B model at low latency"). This is genuinely under-documented publicly; treat the referral tip as your primary signal and prep from general MLOps depth rather than expecting to match a known question. 🔴

One recurring, moderately-corroborated framing across several search summaries: Round 1 often blends "basic questions on ML, SQL, MLOps, and Cloud Deployment" with a live ~20-minute Python preprocessing + model-fitting exercise. 🟡

---

### Q1. What does a production ML CI/CD pipeline look like, end to end?

```
Code commit ──► CI: unit tests + data/schema validation tests ──► Model training job (triggered/scheduled)
                                                                          │
                                                                          ▼
                                                        Offline eval vs. current production model
                                                                          │
                                                          ┌───────────────┴───────────────┐
                                                          ▼                                ▼
                                                  Fails eval gate                   Passes eval gate
                                                  → block promotion                 → shadow/canary deploy
                                                                                            │
                                                                                            ▼
                                                                          Online metrics match/exceed baseline
                                                                                            │
                                                                                            ▼
                                                                                Full rollout + rollback plan
```

Key ideas to state explicitly: **model CI/CD differs from software CI/CD** because correctness isn't just "tests pass" — it also needs data validation (schema/distribution checks on training data), an offline eval gate against the current production model (not just against a static test set — comparing to the *incumbent* is what prevents silent regressions), and staged rollout (shadow traffic → canary → full) because offline metrics don't always predict online behavior.

### Q2. Model monitoring in production

- **Input/feature drift**: distribution of incoming features shifts from training distribution (e.g., PSI/KL-divergence monitoring per feature) — an early warning sign independent of having fresh labels.
- **Prediction drift**: the distribution of model outputs shifts even if you can't yet measure accuracy (labels lag, as in fraud — see [04](04_ML_System_Design_Fraud_Detection.md)).
- **Performance monitoring**: online accuracy/precision/recall once labels arrive, compared against an SLA and against the offline eval numbers at launch time (catches "worked in offline eval, degraded in production" gaps).
- **Latency/throughput/error-rate monitoring**: standard service-level observability, not ML-specific, but a DS candidate should know it's still part of "owning a model in production."
- Alerting thresholds should trigger a defined action (auto-rollback, retrain trigger, page a human) — a monitoring answer that stops at "we'd track metrics" without saying what happens when they breach is incomplete.

### Q3. Serving an LLM in production — latency/cost trade-offs

Even without a confirmed leaked question, this is squarely implied by "LLM hosting" and the team's published GenAI system ([08](08_Attribute_Extraction_Content_Generation.md)) — worth having a real answer:

- **Model size vs. latency/cost**: knowledge distillation to a smaller student model (as Example Company's own team does) trades a little quality for materially lower serving cost/latency at catalog scale.
- **Quantization** (int8/int4) for inference reduces memory footprint and can increase throughput, at a small, usually-acceptable accuracy cost.
- **Batching**: dynamic/continuous batching of concurrent requests improves GPU utilization dramatically for LLM serving vs. one-request-at-a-time — worth naming as the single highest-leverage serving optimization for throughput.
- **Caching**: for extraction/generation tasks with repeated or similar inputs (e.g., re-processing a similar product across categories), caching intermediate or final outputs avoids redundant, expensive LLM calls.
- **Async/batch vs. real-time serving**: catalog attribute extraction is largely a batch/async workload (new listing arrives, gets processed within some SLA — not sub-100ms like fraud scoring), which relaxes latency pressure compared to something like the fraud-scoring path in [04](04_ML_System_Design_Fraud_Detection.md) — explicitly contrasting the two shows you understand that "how you host a model" depends on the use case's actual latency budget, not a one-size-fits-all answer.

### Q4. LoRA adapters in production serving

Ties back to [10](10_LLM_Finetuning_RAG_GenAI.md): serving multiple task-specific LoRA adapters on top of one shared frozen base model (rather than hosting N full fine-tuned model copies) is a real, published cost-saving pattern — some serving frameworks (e.g., vLLM's multi-LoRA serving) support hot-swapping adapters per request against a shared base model in GPU memory. Mentioning this shows awareness of current (2025-era) LLM-serving practice, not just training-time fine-tuning theory.

### Q5. Rollback strategy

If a newly deployed model regresses in production: canary/staged rollout limits blast radius before full rollout; keep the previous model version's artifacts and serving config versioned and instantly deployable (not "retrain from scratch to fix it" — that's far too slow for an incident); feature-flag-style routing lets you instantly shift traffic back to the previous model version without a full redeploy cycle.

### Q6. Cloud/infra basics

Job postings for the team most commonly reference GCP/Azure and internal cloud-native infra, plus Kubeflow/Airflow-style orchestration for MLOps. 🟡 Know at a conceptual level (not necessarily company-specific tooling): container-based model serving (Docker/Kubernetes), workflow orchestration for training pipelines (Airflow/Kubeflow), and feature stores for online/offline parity (already covered in [04](04_ML_System_Design_Fraud_Detection.md)) — if you have hands-on experience with any of these, lead with that concretely rather than reciting definitions.

---

### What This Round Tests

- Whether you've actually owned a model past the notebook stage — monitoring, rollback, staged rollout are the tells
- Cost/latency-aware LLM serving knowledge specific to 2025-era practice (quantization, distillation, batching, multi-LoRA serving) rather than only training-time fine-tuning knowledge
- Judgment about matching serving architecture to the actual latency budget of the use case (real-time fraud scoring vs. async catalog processing are not the same problem)
