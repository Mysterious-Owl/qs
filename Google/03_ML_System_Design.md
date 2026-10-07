# ML System Design

The second ML round 📄, and the area the candidate calls *"probably one of the most important areas of preparation for me."*

---

## The core claim 📄

The sentence worth internalising:

> "ML system design is not just system design with a model inserted somewhere. You need to understand the interaction between the ML lifecycle and the distributed system."

That's the whole round in one line. A candidate who designs a solid distributed service and then says "and a model scores the request" has answered a different question.

## The lifecycle they practised 📄

```
Problem → data → features → training → evaluation → deployment → serving → monitoring → retraining
```

And the specific concerns named:

| Stage | Concerns named in the post |
|---|---|
| Data | Data pipelines, offline vs online features, feature freshness |
| Training | Training infrastructure |
| Serving | Model serving, latency, throughput, scalability, batch vs real-time inference |
| Release | Model versioning, A/B testing |
| Operate | Monitoring, data drift, model drift, retraining, failure scenarios |

## Where the ML lifecycle and the distributed system actually collide 🧩

The post asserts the interaction matters but doesn't enumerate it. These are the seams where the two halves genuinely couple — and they're what the round is probing:

**Offline/online feature skew.** The single most common production ML failure. Training reads features from a warehouse with the whole history available; serving computes them from a live request with milliseconds and partial state. If those two paths are written by different people in different languages, they *will* diverge, and the model silently degrades. This is why feature stores exist — not for storage, but to make one definition serve both paths.

**Point-in-time correctness.** A training row must only contain what was knowable at its decision timestamp. Joining a feature table naively pulls in values computed *after* the label event — which is leakage that looks like brilliant offline performance and collapses in production. This is a *data-engineering* property enforced by *ML* correctness requirements.

**Label latency.** Ground truth often arrives days or weeks after the prediction (a chargeback, a churn event, a conversion). So your monitoring cannot be "accuracy" — for the window where labels don't exist yet, you monitor *input drift* and *prediction drift* as leading indicators, and accept accuracy as a lagging one.

**The feedback loop.** If the model's output determines which examples get labelled — you only investigate what you flagged — your next training set is shaped by the current model's blind spots. Fixing it costs something: a randomized holdout that you deliberately *don't* act on, purely to keep the label distribution honest.

**Retraining cadence as a systems decision.** How often is not an ML question; it's a function of drift rate, label latency, and the cost and risk of a deploy. And every retrain needs an offline gate against the *incumbent* model, not against a static benchmark.

**Latency budget shapes the model.** A 10ms p99 budget rules out architectures regardless of their accuracy. The budget is set by the product, and it propagates backwards into feature choice (no expensive joins), model size (distillation, quantization), and inference mode (precompute-and-cache vs real-time).

## A structure to answer with 🧩

```
1. Clarify     — what's the prediction, for whom, at what latency, at what volume?
                 What's the cost of a wrong answer in each direction?
2. Metrics     — offline metric, online metric, and the business metric. Say how they differ.
3. Data        — sources, labels (and their latency), volume, point-in-time correctness
4. Features    — offline/online split, freshness requirements, the skew risk
5. Model       — start simple, justify complexity; state the baseline you'd beat
6. Training    — cadence, infrastructure, validation split (time-based if temporal)
7. Serving     — batch vs real-time, latency budget, caching, fallback when the model is down
8. Release     — shadow → canary → rollout, with the rollback path
9. Monitor     — input drift, prediction drift, delayed-label performance, and what each alert *does*
10. Failure    — what breaks, what degrades gracefully, what pages a human
```

Step 1 and step 10 are where candidates most often lose points 🧩 — jumping to architecture without pinning down the requirement, and never saying what happens when it breaks.

## Where the depth lives 🔗

- [`../Example_Company/04_ML_System_Design_Fraud_Detection.md`](../Example_Company/04_ML_System_Design_Fraud_Detection.md) — a full worked end-to-end design with the latency tiers, label-latency problem and feedback-loop bias spelled out
- [`../Example_Company/05_Ecommerce_Catalog_System_Design.md`](../Example_Company/05_Ecommerce_Catalog_System_Design.md) — multi-stage pipeline with humans in the loop
- [`../Example_Company/12_MLOps_CICD_LLM_Hosting.md`](../Example_Company/12_MLOps_CICD_LLM_Hosting.md) — CI/CD gates, monitoring, rollback
- [`../ML_Fundamentals/Feature_Engineering_Selection.md`](../ML_Fundamentals/Feature_Engineering_Selection.md) §1 — leakage, including the temporal and group varieties that bite in system design
- [`../ML_Engineer/Q8_Feature_Stores_and_Serving_Consistency.md`](../ML_Engineer/Q8_Feature_Stores_and_Serving_Consistency.md) — the offline/online seam directly
- [`../AI_Architect/Q2_Scaling_Strategy.md`](../AI_Architect/Q2_Scaling_Strategy.md) — scaling reasoning

## The candidate's own warning 📄

> "Reading system-design solutions is very different from actually designing a system yourself."

Listed under *what I'd do differently*: **practise ML system design out loud.** Reading this file is not the same as having said it; talk through one design end to end, on a whiteboard, against a timer.
