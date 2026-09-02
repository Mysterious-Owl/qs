# ML System Design: Fraud Detection

Referral tip: **"fraud system design"** was previously asked. 🔵 This is the single most corroborated system-design prompt found in public research — it appears near-identically across three independent prep sources. 🟢

> "Design a comprehensive fraud detection system for Example Company that processes millions of transactions daily across online, mobile, and in-store channels."

---

### Answer Framework

#### 1. Clarify scope and constraints first

- Channels: online checkout, mobile app, in-store POS — do they share a model or need channel-specific signals?
- Latency requirement: **sub-100ms** for online/in-store (can't hold up checkout); can be looser for post-transaction batch review.
- Class imbalance: fraud is typically **<0.1%** of transactions — this drives almost every downstream modeling choice.
- Cost asymmetry: a missed fraud (false negative) costs money directly; a false positive blocks a legitimate customer — both have a cost, and the org has to pick an operating point on that trade-off, not the model.
- Explainability: needed for compliance/dispute resolution — "the model said so" isn't an acceptable answer to a customer or regulator.

#### 2. High-level architecture

```
Transaction Event
      │
      ▼
Streaming ingestion (Kafka)
      │
      ▼
Feature computation ──► Feature Store (online + offline parity)
      │
      ▼
Real-time scoring service (sub-100ms)
      │
      ├─► Score < low threshold  → Approve
      ├─► Score in gray zone     → Step-up auth / manual review queue
      └─► Score > high threshold → Block + alert
      │
      ▼
Feedback loop: confirmed fraud/chargeback labels → retraining pipeline
```

Name the components explicitly: streaming platform (Kafka/Kinesis) for event ingestion, a low-latency feature store (online store like Redis/DynamoDB backing the same features as the offline training store — **online/offline parity is the thing that breaks in practice**), a real-time scoring service, a rules engine sitting alongside the ML model (hard blocks for known-bad patterns don't need a model), and a human-review queue for the gray zone.

#### 3. Features

| Category | Examples |
|---|---|
| Transaction-level | amount, time of day, item category, payment method |
| Velocity | # transactions in last 1min/1hr/24hr for this user/card/device |
| Behavioral/historical | deviation from user's typical basket size, typical purchase location |
| Network/graph | shared device/IP/shipping address across many accounts (fraud rings) |
| Device/session | new device flag, IP reputation, mismatched billing/shipping geolocation |

#### 4. Modeling approach

- **Extreme class imbalance** → don't naively train on raw ratio. Use class weighting, focal loss, or resampling (SMOTE on minority class) combined with a metric that isn't accuracy: **PR-AUC**, precision@k, or recall at a fixed false-positive budget.
- Model family: gradient-boosted trees (XGBoost/LightGBM) are the industry default here — good with tabular/velocity features, fast inference, and offer feature importance for explainability. A graph model (for fraud rings/shared-identity signals) can be a second-stage enrichment rather than the primary model.
- **Two-stage design** is common in practice: a fast, cheap model (rules + lightweight model) for the sub-100ms path, and a heavier ensemble/graph model running asynchronously for post-transaction review and label generation.
- Explainability: SHAP values per prediction, surfaced to the manual-review team and usable in dispute resolution.

#### 5. Evaluation & feedback loop

- Offline: precision/recall at the operating threshold, PR-AUC, cost-weighted metric (dollar loss from false negatives + false positives at candidate thresholds — this is the number the business actually cares about).
- **Label latency problem**: confirmed fraud (chargebacks) can take weeks to materialize — so your "training label" is delayed and your model can silently drift in the meantime. Mitigate with: shorter-latency proxy labels (manual review verdicts) blended with the slower ground-truth chargeback labels, and active monitoring for score-distribution drift as an early-warning signal independent of labels.
- Human-in-the-loop review queue doubles as your labeling pipeline — that data feeds retraining.

#### 6. Threats to correctness / edge cases to raise proactively

- **Feedback loop bias**: if you only ever review transactions the model already flagged, you never learn about fraud the model is confidently missing — need some randomized/sampled review of low-score transactions too.
- **Adversarial adaptation**: fraudsters change behavior once they learn what gets blocked — plan for a retraining cadence and monitoring for sudden shifts in feature distributions, not a "train once" model.
- **Cold start**: new users/cards have no history — velocity/behavioral features are unavailable; fall back to device/network signals and stricter rules-based checks for genuinely new identities.

---

### What This Question Tests

- Whether you frame the problem around business/cost trade-offs (precision vs. recall, latency budget) before jumping to a model choice
- Understanding of extreme class imbalance and why accuracy is the wrong metric
- Awareness of production realities specific to fraud: label latency, feedback loop bias, adversarial drift — these separate a "textbook ML" answer from a "built one of these" answer
- System-level thinking: streaming ingestion, online/offline feature parity, tiered latency paths
