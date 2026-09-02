# Ecommerce Compliance & Catalog System Design

Referral tip: **"ecommerce compliance system design"** was previously asked. 🔵 Public research surfaced a closely related, independently-sourced system design prompt: designing a read-and-write-heavy **product catalog service** — the infra backbone that any catalog-compliance system sits on top of. 🟡

---

## Part A — Ecommerce Compliance System Design

> Design a system that enforces marketplace compliance policies (prohibited items, counterfeit detection, seller policy violations) across a large, multi-seller ecommerce catalog.

### Answer framework

**1. Where compliance has to act — multiple checkpoints, not one**

```
Seller onboarding ──► Pre-listing check ──► Post-listing monitoring ──► Investigation/enforcement
  (identity/KYC)      (category/content      (ongoing scan of live       (takedown, seller
                        rules gate)            listings + signals)        suspension, appeals)
```

This multi-layer framing matches how Example Company's own Trust & Safety org (part of Catalog Data Science) has publicly described its approach: seller vetting pre-onboarding, pre-listing category restrictions, proactive listing removal, and rapid investigation response — not a single "compliance model" but a pipeline of gates. 🟢 (Example Company corporate/engineering content)

**2. Signals to combine (this should not be one model)**

| Signal type | Examples |
|---|---|
| Content-based | product title/description text classifier, image classifier (nudity/weapons/counterfeit logos) |
| Metadata-based | category mismatch (e.g., "toy" category but description mentions a controlled substance), price anomalies (near-zero price often signals counterfeit) |
| Seller-based | new seller + high-risk category, seller's historical violation rate, sudden listing-volume spike |
| Network-based | shared bank account/address/device across "different" seller accounts (ring detection) |

**3. Pipeline design**

- **Pre-listing (synchronous, blocking)**: cheap, fast checks — banned keyword/category rules, a lightweight classifier — must run in the seconds-scale listing-creation path.
- **Post-listing (async, batch/streaming)**: heavier ensemble models (text + image + seller-graph) scanning the live catalog continuously, since violations can appear after initial approval (edited listings, review-driven signals).
- **Escalation**: model output isn't auto-enforcement past a certain risk tier — route to human review for anything with real seller/customer impact (false positive = wrongly delisting a legitimate seller's product, which is a real business cost).

**4. Metrics that matter**

- Precision is often prioritized over recall at the auto-enforcement tier (wrongly banning a compliant seller is expensive and erodes marketplace trust) — recall matters more for the human-review queue, where a lower bar is fine because a human filters it.
- Time-to-detection (how fast does a violating listing get caught after going live) is a first-class metric, not just accuracy.

**5. Edge cases to raise**

- Adversarial sellers reword listings to dodge keyword/classifier detection — plan for a retraining cadence and text-obfuscation-robust features (e.g., embeddings over exact keyword match).
- Cross-locale complexity — "prohibited" varies by country/region regulation; the rule engine needs to be locale-aware, not global-only.
- Appeals workflow — any automated enforcement needs a human appeals path; design it in from the start, not bolted on later.

---

## Part B — Product Catalog Service Design (infra backbone)

> How would you design a read-heavy, write-heavy product catalog service for millions of SKUs across many sellers?

This is the systems-design layer the compliance/ML system above has to sit on top of. 🟡 (single corroborated source, but standard for this class of problem)

### Key design tensions to address explicitly

| Tension | Design response |
|---|---|
| Read-heavy (browse/search) vs. write-heavy (price/inventory updates, new listings) | Separate read and write paths: CQRS-style — writes go to a source-of-truth store, reads served from a denormalized, heavily-cached, search-optimized index (Elasticsearch/Solr) |
| Staleness vs. freshness | Async propagation from source-of-truth → search index, with a bounded staleness SLA (e.g., "listing changes reflected in search within N seconds") — don't try to make search perfectly consistent, it doesn't scale |
| Concurrent writes to the same SKU | Partition by `product_id`/`sku_id` so a single shard/writer owns a given item — avoids distributed-lock contention (ties back to [03](03_Distributed_Systems_Multithreading.md)) |
| Pagination at scale | Cursor-based (keyset) pagination, not offset-based — offset pagination degrades badly past a few thousand pages |
| Listing withdrawal/failure recovery | Soft-delete + audit log rather than hard delete — needed for compliance investigations and to recover from bad automated takedowns |

### What This Question Tests

- Whether you separate "the ML problem" from "the systems problem" — compliance detection is a model, but *catalog service design* is a distributed-systems/data-modeling problem, and a good answer treats them as related but distinct layers
- Precision/recall trade-off reasoning specific to enforcement actions (not just generic classification metrics)
- Awareness that a real system is a multi-stage pipeline with humans in the loop, not a single model behind an API
