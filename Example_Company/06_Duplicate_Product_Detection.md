# Catalog Use Case: Duplicate Product Detection

Referral tip: this is one of the team's three flagship use cases. 🔵 No corroborated public leak of this exact interview prompt was found — Trust & Safety/catalog-quality system design questions for this specific team don't appear to leak to public forums. Treat the framework below as your primary prep material rather than a "known question with a known answer."

> "Design a system to detect duplicate products in a catalog with hundreds of millions of listings from thousands of sellers."

---

### 1. Define "duplicate" precisely first

This is the single most important clarifying move — "duplicate" is ambiguous and the answer changes a lot depending on which of these you mean:

| Type | Example | Detection approach differs because... |
|---|---|---|
| Exact duplicate | Same seller lists identical item twice | Simple hash/fingerprint match suffices |
| Same product, different seller | Two sellers list the same physical product (common, legitimate — Example Company wants to *merge* these into one catalog entry with multiple offers, not "detect and remove" them as if they were spam) | This is really an **entity resolution / product matching** problem |
| Near-duplicate / variant | Same base product, different size/color, listed as if separate | Needs attribute-aware matching — must NOT merge legitimately distinct variants |
| Spam/repost duplicate | Same seller reposts a delisted item slightly reworded to evade detection | Adversarial — needs robustness to intentional obfuscation |

Say explicitly: for a marketplace like Example Company's, most "duplicates" are actually **case 2** — the goal isn't to delete one, it's to *canonicalize* multiple seller listings into a single product entry ("buy box" model) — this reframes the problem from classification to entity resolution, which is a meaningfully different system.

### 2. High-level architecture (entity resolution pipeline)

```
New/updated listing
      │
      ▼
Blocking / candidate generation  (cheap, high-recall — avoid O(n²) comparison over hundreds of millions of items)
      │
      ▼
Pairwise similarity scoring      (expensive, high-precision — only run on candidate pairs)
      │
      ▼
Clustering / canonicalization    (group matched pairs into a single canonical product entity)
      │
      ▼
Human review for low-confidence matches
```

**Blocking (candidate generation)** — you cannot compare every pair among hundreds of millions of items. Standard approaches: locality-sensitive hashing (LSH) or approximate nearest-neighbor search (FAISS/HNSW) over product embeddings, blocking keys (brand + category + first N chars of normalized title), or UPC/GTIN exact match as a free, high-precision first pass when available.

**Pairwise/similarity scoring** — once you have a manageable candidate set per item:
- Text similarity: title/description embeddings (sentence-transformer style), fuzzy string matching (Jaccard/Levenshtein) as cheap features feeding a learned model rather than a hard threshold.
- Image similarity: perceptual hashing for exact/near-exact image matches, CNN/CLIP embeddings + cosine similarity for genuinely different photos of the same product.
- Structured attribute similarity: brand, UPC/GTIN if present, size, color, model number — these are the highest-signal, lowest-noise features when available.
- Combine as features into a learned pairwise classifier (gradient-boosted trees are typical) rather than hand-tuned thresholds — lets you weight UPC match vs. fuzzy-title match vs. image similarity appropriately, and gives you a tunable precision/recall knob.

**Clustering** — pairwise match scores define a graph (edge = "these two are the same product"); connected components / graph clustering group listings into canonical entities. Watch for **transitivity errors**: A~B and B~C strongly matched doesn't guarantee A~C — needs cluster-level consistency checks, not just pairwise thresholds.

### 3. Scale considerations

- Blocking is the load-bearing scalability decision — get it wrong (too narrow) and you miss true duplicates; too broad and pairwise scoring becomes the bottleneck again. This tension is worth stating explicitly.
- New listings need near-real-time dedup at ingestion (don't let an obvious duplicate go live and get discovered days later) — architecturally this pushes toward an online candidate-lookup (ANN index queryable at listing time) plus an offline batch re-clustering job that catches slower-forming clusters and corrects drift.

### 4. Metrics

- Standard entity-resolution metrics: pairwise precision/recall, and cluster-level metrics (e.g., B-cubed F1) — pairwise precision alone can look good while cluster quality is bad if transitivity errors are common.
- Business-facing metric: "buy box accuracy" or seller/catalog-health complaints — ties the ML metric back to what the business actually experiences.

### 5. Edge cases to raise proactively

- **Legitimately similar-but-distinct products**: a 3-pack vs. single unit of the same item, or different flavors/scents of the same product line — false-positive merges here directly hurt customers (wrong item ships) and sellers (wrong offer attached).
- **Adversarial reposting**: a delisted item reappears with a slightly reworded title/cropped image specifically to evade the matcher — argue for embedding-based (not exact-string) matching precisely because it's harder to trivially evade.
- **Missing/inconsistent identifiers**: not every seller provides UPC/GTIN — the system has to degrade gracefully to content-based matching rather than assuming a clean identifier always exists.

### What This Question Tests

- Whether you recognize this is fundamentally an **entity resolution** problem, not a binary classifier — that reframing is the single highest-signal thing you can say early
- Systems thinking about scale (blocking/candidate generation before expensive scoring)
- Precision/recall trade-off awareness specific to *merging* decisions, where a false positive has direct customer/seller impact (wrong item, wrong offer) — different failure mode than most classification problems
