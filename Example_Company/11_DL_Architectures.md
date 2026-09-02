# Deep Learning Architectures

Referral tip: "kuchh DL model ka architecture v puchh skate h" — some DL architecture questions are likely. 🔵 One single-source account (single-author Medium series, unverified) describes a round probing "experience with transformer architectures" specifically. 🔴

> Deeper reference: [`../ML_Fundamentals/DL_Concepts_Cheatsheet.md`](../ML_Fundamentals/DL_Concepts_Cheatsheet.md) (DL fundamentals + 14 architectures) and [`../ML_Fundamentals/Transformers_Cheatsheet.md`](../ML_Fundamentals/Transformers_Cheatsheet.md) (transformers end to end, incl. a rapid-fire Q&A section).

---

### Q1. Transformer architecture — draw it, don't just describe it

Be ready to sketch and narrate the full encoder or decoder block:

```
Input embeddings + positional encoding
        │
        ▼
┌─────────────────────────┐
│ Multi-Head Self-Attention│──► Add & Norm (residual connection)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Feed-Forward Network    │──► Add & Norm (residual connection)
└─────────────────────────┘
        │
        ▼
    (repeat N layers)
```

Know: why positional encodings exist (self-attention has no inherent notion of sequence order — it's permutation-invariant without them), why residual connections + layer norm matter (gradient flow through deep stacks), and the encoder-decoder vs. decoder-only distinction (BERT-style encoder for understanding/embedding tasks, GPT-style decoder-only with causal masking for generation — relevant since catalog attribute extraction ([08](08_Attribute_Extraction_Content_Generation.md)) is fundamentally an understanding/extraction task, which is worth noting if asked which architecture family fits it best).

### Q2. CNN basics

Corroborated as asked. 🟡 Know: convolution as a learned, spatially-shared filter (parameter efficiency vs. a fully-connected layer over the same input), why this induces translation invariance, pooling for spatial downsampling/invariance, and the receptive-field-growth intuition across layers. Directly relevant to the image side of content-moderation and attribute-extraction systems ([07](07_Content_Compliance_Moderation.md), [08](08_Attribute_Extraction_Content_Generation.md)) — vision backbones for those tasks are typically CNN-based (ResNet/EfficientNet family) or ViT-based, and being able to compare the two families (CNN's built-in inductive bias for locality vs. ViT needing more data to learn spatial structure from scratch) is a strong follow-up answer.

### Q3. How regularization changes the training objective

Corroborated as asked (Prepfully). 🟡 L2 regularization adds `λ‖w‖²` to the loss — equivalent to placing a Gaussian prior on weights in a Bayesian view, pulling them toward zero and reducing variance. Dropout randomly zeroes activations during training, which can be viewed as training an implicit ensemble of subnetworks and prevents co-adaptation of neurons. Know the practical distinction: dropout is disabled at inference (or scaled, depending on implementation — inverted dropout scales during training so inference needs no change), while L2 stays in effect only via its already-baked-in effect on the learned weights.

### Q4. Multi-head attention — why multiple heads instead of one larger head

Single attention head with the full dimensionality can only learn one type of relationship pattern per layer. Splitting into multiple smaller heads lets different heads specialize (e.g., one head attending to syntactic dependencies, another to positional proximity) and the outputs are concatenated and linearly projected back to the model dimension — empirically improves representational capacity without a compute blow-up, since total parameters across heads roughly match a single full-size head.

### Q5. Vision-language / multimodal architectures (CLIP-style)

Relevant given the team's multimodal (text+image) problems ([06](06_Duplicate_Product_Detection.md), [07](07_Content_Compliance_Moderation.md), [08](08_Attribute_Extraction_Content_Generation.md)). Know the contrastive pretraining idea: a text encoder and image encoder are trained jointly so that matching text-image pairs have high cosine similarity in a shared embedding space and non-matching pairs have low similarity (contrastive loss, e.g., InfoNCE). This shared embedding space is exactly what powers image-based duplicate/similarity search and multimodal attribute extraction — a good answer connects the architecture concept back to the specific team use case unprompted.

### Q6. Knowledge distillation

Corroborated as part of the team's published fine-tuning stack ([08](08_Attribute_Extraction_Content_Generation.md)). 🟢 A smaller "student" model is trained to match a larger "teacher" model's output distribution (soft labels, often via KL divergence between student and teacher softmax outputs at a raised temperature) rather than just hard ground-truth labels — soft labels carry more information (relative confidence across classes) than one-hot labels. Used to get most of a large model's quality into a cheaper-to-serve model, which is the practical reason it matters for a production catalog system serving hundreds of millions of items.

---

### What This Round Tests

- Whether you can go beyond naming an architecture to explaining *why* each component exists (positional encoding, residual connections, multi-head attention) — the "why" is what separates memorized trivia from real understanding
- Ability to connect architecture choices to this team's actual multimodal, production-scale problems rather than answering in the abstract
