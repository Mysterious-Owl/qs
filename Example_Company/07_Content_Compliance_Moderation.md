# Catalog Use Case: Content Compliance Moderation

Referral tip: flagship use case #2 — detecting nudity, weapons/arms, and other prohibited content in catalog listings. 🔵 No corroborated public leak of this exact prompt found. Best grounding is Example Company's own published content: its Trust & Safety team (part of the Catalog Data Science org) "employs GenAI and ML models to identify products that violate marketplace compliance policies, with end-to-end ML pipelines designed to scale and detect policy violations across millions of items." 🟢 (company corporate site, 2025)

> "Design a system to detect prohibited content (nudity, weapons/arms, other banned items) in product listings across text, images, and video."

---

### 1. Scope the content types — each needs a different model family

| Content type | Modality | Model approach |
|---|---|---|
| Nudity/adult content | Image (+ sometimes video) | Vision classifier fine-tuned for NSFW detection; can often start from an off-the-shelf/pretrained safety classifier and fine-tune on in-domain catalog imagery |
| Weapons/arms | Image, sometimes title/description text | Object detection (not just classification) — need to *localize* the weapon in the image for both accuracy and explainability to a human reviewer |
| Other prohibited items (drugs, counterfeit, hazardous materials) | Text (title/description) primarily, sometimes image | Text classifier over title/description/category, often combined with category-based rules (a rule engine handles clearly-defined categories cheaply; ML handles the ambiguous long tail) |

Say this early: **treat it as a multi-model system, not one classifier** — nudity, weapons, and "other prohibited" have different base rates, different failure costs, and different available training data, so bundling them into one multi-class model is usually the wrong call. A per-category model (or per-category head on a shared backbone) lets you tune thresholds independently per policy category, which matters because false-positive tolerance genuinely differs by category (a false positive on "weapons" that blocks a kitchen knife listing is very different in cost/frequency from one on hard drug paraphernalia).

### 2. Pipeline architecture

```
New/edited listing (text + images)
      │
      ▼
Cheap rule/keyword pre-filter   (category blocklist, known-bad keyword list — catches the obvious cases for free)
      │
      ▼
ML classification (per policy category, text + image models)
      │
      ├─ High confidence violation  → Auto-block + notify seller
      ├─ Gray zone                  → Route to human moderation queue
      └─ High confidence clean      → Auto-approve
      │
      ▼
Human moderator decisions ──► feedback into retraining data
```

This mirrors the same multi-checkpoint framing as [05](05_Ecommerce_Catalog_System_Design.md) — pre-listing (blocking, fast) and post-listing (continuous re-scan, since a listing can be edited after approval to introduce violating content, or a model can improve and re-catch things it missed initially).

### 3. Multimodal fusion

For weapons/nudity specifically, both text and image signals matter and often disagree (a compliant image with a violating title mention, or vice versa) — fuse rather than pick one modality:
- Late fusion (score each modality independently, combine scores) is simpler to build, debug, and independently improve/retrain per modality.
- Early/joint fusion (shared multimodal embedding, e.g., CLIP-style) can catch cases where neither modality alone is confidently violating but the combination is (e.g., ambiguous product photo + a title that disambiguates it as a weapon) — worth mentioning as the more sophisticated option if the interviewer pushes on it, while being honest that it's harder to debug per-modality.

### 4. Human-in-the-loop design

- This must never be a fully automated take-it-or-leave-it system for high-stakes categories — false positives (wrongly blocking a compliant seller) are a real trust/business cost, false negatives (letting truly prohibited content through) are a legal/brand-risk cost. Both costs argue for a **calibrated gray zone routed to humans**, not a single hard threshold.
- The human review queue is also your ground-truth label source — design the review UI/data capture so verdicts flow directly back into the training pipeline (this is the same feedback-loop point as the fraud system in [04](04_ML_System_Design_Fraud_Detection.md)).

### 5. Metrics

- Precision at the auto-block tier is usually prioritized (a wrongful auto-block is a direct seller/customer-facing failure); recall matters more at the "route to human" tier, since a human filters the false positives there.
- Time-to-detection for post-listing violations (how long does a violating listing stay live before being caught) is a first-class metric independent of classifier accuracy.
- Category-specific metrics — don't report one blended number; nudity, weapons, and other-prohibited likely have very different base rates and should be tracked/tuned separately.

### 6. Edge cases to raise proactively

- **Adversarial evasion**: sellers crop/blur images or use euphemistic text to dodge detection — argue for periodic retraining and image-robustness techniques (the model shouldn't rely on a narrow visual pattern that's trivially evaded by a slight crop).
- **Cultural/legal variation by locale**: what counts as "prohibited" varies by country and even by state/region — the policy layer (not just the model) needs to be locale-parameterized.
- **Ambiguous legitimate items**: kitchen knives, replica/toy weapons, medical/anatomical products — these are exactly why a hard threshold fails and a human-review gray zone is necessary; a good answer names this tension unprompted.
- **GenAI-generated content**: with sellers increasingly using AI-generated product images/descriptions, watch for AI-generated content that subtly misrepresents a product's actual content — a growing edge case worth naming if the conversation goes toward 2025-era GenAI risks.

### What This Question Tests

- Whether you decompose "content moderation" into per-category models rather than one blended classifier, and can justify why
- Multimodal reasoning (text + image) and honest trade-offs between fusion strategies
- Precision/recall trade-offs specific to *enforcement* actions, and the judgment to keep humans in the loop for high-stakes, ambiguous categories rather than proposing full automation
