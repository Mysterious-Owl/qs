# NLP Tasks & Evaluation

The canonical task shapes, the technique behind each, and — the part candidates most often fumble — how each one is actually scored.

Companion: [`01_Text_Representation.md`](01_Text_Representation.md).

---

# 1. The Task Shapes

Most NLP problems reduce to one of five shapes. Recognising the shape tells you the output layer, the loss and the metric.

| Shape | Output | Examples |
|---|---|---|
| **Sequence classification** | One label per document | Sentiment, topic, spam, intent, NLI |
| **Token classification** | One label per token | NER, POS tagging, chunking |
| **Sequence-to-sequence** | A generated sequence | Translation, summarization, paraphrase |
| **Span extraction** | Start and end indices | Extractive QA |
| **Pairwise scoring** | A score for a text pair | Retrieval, re-ranking, similarity, entailment |

Naming the shape before proposing an architecture is the move — it's what turns "I'd use BERT" into an actual design.

---

# 2. Text Classification

The default task. Covered end to end in [`03_Notebook_Text_Classification.ipynb`](03_Notebook_Text_Classification.ipynb).

**Approaches, cheapest first:** rules/keywords → TF-IDF + linear model (logistic regression or linear SVM) → gradient-boosted trees on TF-IDF (rarely better, and slower) → fine-tuned transformer → LLM zero/few-shot.

**Why linear models specifically** for high-dimensional sparse text: with |V| features and far fewer documents you're in the `p ≫ n` regime, where a strongly regularized linear model is the correct choice and a flexible model overfits ([`../ML_Fundamentals/Feature_Engineering_Selection.md`](../ML_Fundamentals/Feature_Engineering_Selection.md) §5). Linear SVM and logistic regression are the two standards — SVM often marginally better on accuracy, logistic regression gives you calibrated-ish probabilities.

**Multi-class vs multi-label** — a genuine distinction that gets tested. Multi-class: exactly one label, softmax, cross-entropy. Multi-label: any number of labels, **sigmoid per label**, binary cross-entropy, and a per-label threshold. Using softmax for a multi-label problem is a real and common bug, because softmax forces the probabilities to compete and sum to 1.

**Class imbalance** in text behaves as it does elsewhere — see [`../Question_Bank/README.md`](../Question_Bank/README.md) R1Q6 — with one text-specific note: oversampling duplicated documents is particularly prone to memorization, so class weighting is usually the safer first lever.

---

# 3. Sequence Labeling (NER, POS)

One label per token. The mechanics here are specific enough to be worth knowing properly, and this is the task most often gotten wrong in interviews.

## BIO tagging

Entities span multiple tokens, so a flat label per token is ambiguous — you can't tell one two-token entity from two adjacent one-token entities. **BIO** (also IOB) fixes this:

```
Jane    Smith   visited  New     York    City    in   April
B-PER   I-PER   O        B-LOC   I-LOC   I-LOC   O    B-DATE
```

- **B-** begins an entity, **I-** continues it, **O** is outside any entity.
- **BIOES/BILOU** adds **E-** (end) and **S-** (single-token) — slightly more supervision, often marginally better, more classes to learn.

**The constraint that matters**: `I-PER` may not follow `O` or `B-LOC`. A plain per-token classifier doesn't know this and will emit invalid sequences. Two fixes: a **CRF** layer, or post-hoc repair of illegal transitions.

## Linear-chain CRF

The classical answer to "the labels are not independent". A per-token softmax picks each label in isolation; a CRF scores the **whole sequence** and lets neighbouring labels constrain each other.

```
score(x, y) = Σ  [  emission(x_i, y_i)  +  transition(y_{i-1}, y_i)  ]
              i
```

- **Emission** — how well does this token support this label (from features, or a neural encoder's output)
- **Transition** — a learned |L|×|L| matrix: how likely is this label to follow that one. This is what learns that `I-PER` cannot follow `B-LOC`

Trained by maximizing the log-likelihood of the gold sequence, which needs the partition function over all label sequences — computed with the **forward algorithm** in `O(n·|L|²)`. At inference, the best sequence comes from **Viterbi** decoding, same complexity.

The architecture that dominated pre-transformers was **BiLSTM-CRF**, and **BERT + CRF** is still used where valid structure matters. With a strong enough encoder, the CRF's marginal gain shrinks — many modern systems drop it and repair illegal transitions instead.

## Evaluation — the part that gets fumbled

**Token-level accuracy is a trap.** Most tokens are `O`, so predicting `O` everywhere can score above 90% while finding zero entities.

The correct metric is **entity-level** (span-level) precision/recall/F1, where a prediction counts as correct only if **both boundaries and the type match exactly**. This is what `seqeval` and the CoNLL scorer compute.

Consider gold `B-PER I-PER` for "Jane Smith":
- Predicting `B-PER I-PER` → one true positive
- Predicting `B-PER O` → **zero** true positives. One false positive (the span "Jane"), one false negative (the span "Jane Smith"). Partial credit is not given, even though token accuracy is 50%.

Being able to explain why exact-match span F1 is the standard — and that a partial-overlap scheme (MUC, or relaxed matching) exists for cases where boundary precision matters less — is a strong signal.

Worked in [`05_Notebook_Sequence_Labeling.ipynb`](05_Notebook_Sequence_Labeling.ipynb).

---

# 4. Topic Modeling

Unsupervised discovery of themes across a corpus.

| Method | Mechanism |
|---|---|
| **LSA / LSI** | Truncated SVD of the term-document matrix. Fast, but components can be negative and are hard to read |
| **pLSA** | Probabilistic, no prior over documents, overfits |
| **LDA** | Generative Bayesian model with Dirichlet priors |
| **NMF** | Non-negative factorization — parts-based, so components are additive and often more readable than LSA |
| **BERTopic** etc. | Embed → cluster (UMAP + HDBSCAN) → label with class-based TF-IDF |

> ⚠️ **LDA is overloaded.** *Latent Dirichlet Allocation* (topic modeling, unsupervised) and *Linear Discriminant Analysis* (supervised classification + dimensionality reduction, in [`../ML_Fundamentals/ML_Algorithms_Cheatsheet.md`](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md) §10) are unrelated. Ask which one is meant.

**Latent Dirichlet Allocation's generative story** — worth being able to tell, because it explains every hyperparameter:

> For each document, draw a topic mixture θ ~ Dirichlet(α). For each word position: draw a topic z ~ Multinomial(θ), then draw a word from that topic's word distribution φ_z ~ Dirichlet(β).

So `α` controls how many topics a document tends to mix (low α → documents are about few topics) and `β` how concentrated each topic's vocabulary is. Inference by variational Bayes or collapsed Gibbs sampling.

**Evaluation**: held-out perplexity is the classic choice and is known to correlate *poorly* with human judgement of topic quality. **Topic coherence** (UMass, or NPMI-based C_v) is the better automatic proxy, and eyeballing the top-N words per topic remains standard practice. Choosing *k* has the same character as choosing clusters — see [`../ML_Fundamentals/Plots_Visual_Diagnostics.md`](../ML_Fundamentals/Plots_Visual_Diagnostics.md) §16.

---

# 5. Generation Tasks

## Summarization

**Extractive** selects existing sentences (TextRank, or a sentence classifier). Guaranteed faithful to the source, but choppy. **Abstractive** generates new text (seq2seq, LLM). Fluent, and can hallucinate — which makes faithfulness, not fluency, the metric that matters.

## Machine translation

Rule-based → statistical (phrase tables, alignment) → neural seq2seq with attention → transformers. The attention mechanism was *invented* for MT, to solve the bottleneck of compressing a whole source sentence into one fixed vector.

## Question answering

**Extractive** predicts a span in a provided passage — two classifiers over token positions, for start and end. **Abstractive/generative** writes an answer. **Open-domain** adds a retrieval step first, which is exactly RAG ([`../ML_Fundamentals/GenAI_Cheatsheet.md`](../ML_Fundamentals/GenAI_Cheatsheet.md) §6).

## Natural language inference

Given a premise and hypothesis, predict entailment / contradiction / neutral. A pair-classification task, and the standard probe for whether a model does sentence-level reasoning.

---

# 6. Evaluation Metrics

The table interviewers probe, because using the wrong one is a common real-world error.

| Metric | Measures | Used for | Weakness |
|---|---|---|---|
| **Accuracy / F1** | Label correctness | Classification | Accuracy misleads under imbalance |
| **Entity F1** | Exact span + type match | NER | Harsh — no partial credit |
| **Perplexity** | `exp(mean NLL)` | Language modelling | Only comparable within the same tokenizer and corpus |
| **BLEU** | n-gram **precision**, with a brevity penalty | Translation | Precision-oriented; ignores meaning; corpus-level |
| **ROUGE** | n-gram **recall** (R-1, R-2, R-L) | Summarization | Rewards copying; blind to fluency and factuality |
| **METEOR** | Unigram match with stems/synonyms + alignment | Translation | Needs language resources |
| **chrF** | Character n-gram F-score | Translation | Better for morphologically rich languages |
| **BERTScore** | Embedding similarity of token pairs | Generation | Model-dependent; not interpretable |
| **Exact Match** | String equality | Extractive QA | Brutally strict |
| **MRR / NDCG / Recall@k** | Ranking quality | Retrieval | Need graded or binary relevance labels |

**BLEU vs ROUGE in one line** — BLEU is precision-flavoured (*of what I generated, how much appears in the reference*), ROUGE is recall-flavoured (*of the reference, how much did I generate*). That maps onto their tasks: translation shouldn't add content, a summary shouldn't miss content.

**Perplexity caveat**: it's `exp` of the mean negative log-likelihood per token, so it depends on *how text is tokenized*. Comparing perplexity across models with different tokenizers is meaningless — a frequently-made error.

**The honest modern position**: all n-gram overlap metrics correlate weakly with human judgement, because a good paraphrase scores badly and a fluent falsehood scores well. They survive on cheapness and comparability. Serious evaluation of generation pairs them with human review or a calibrated LLM-as-judge — see [`../ML_Fundamentals/GenAI_Cheatsheet.md`](../ML_Fundamentals/GenAI_Cheatsheet.md) §7, including the judge's own biases.

---

# 7. Rapid-Fire

**Q: Stemming or lemmatization?**
Lemmatization if you need real words and have the POS; stemming if you need speed and don't care about readability. With subword tokenization feeding a neural model, usually neither.

**Q: Why is token accuracy wrong for NER?**
Most tokens are `O`. Predicting all-`O` scores >90% and finds nothing. Use entity-level F1 with exact boundary and type match.

**Q: Why TF-IDF over raw counts?**
Raw counts reward words that are frequent *everywhere*. IDF down-weights them, so a term scores high only when it's frequent in this document *and* rare across the corpus.

**Q: Why can't word2vec handle polysemy?**
One vector per word *type*. `bank` gets a single vector averaging both senses. Contextual embeddings compute the vector from the sentence, so the two senses separate.

**Q: What does negative sampling solve?**
The full softmax over |V| per training step. It replaces it with binary classification against `k` sampled negatives, drawn from the unigram distribution raised to the 3/4 power.

**Q: BLEU or ROUGE?**
BLEU (precision) for translation; ROUGE (recall) for summarization. Both are weak proxies for quality.

**Q: Why might a CRF beat a per-token softmax?**
It scores the whole sequence with a learned transition matrix, so it can't emit structurally invalid tag sequences like `I-PER` after `B-LOC`.

**Q: Your text classifier is 98% accurate. What do you check first?**
Leakage and class balance. Duplicated documents across the split, a label-correlated artifact (boilerplate, a header, a source marker), or a dominant class. See [`../ML_Fundamentals/Feature_Engineering_Selection.md`](../ML_Fundamentals/Feature_Engineering_Selection.md) §1.
