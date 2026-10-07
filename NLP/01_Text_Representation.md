# Text Representation — From Characters to Vectors

How text becomes numbers, and the mechanics behind each step. This is the half of NLP that pre-dates transformers and still gets asked, because it's where the reasoning about *representation* lives.

Companions: [`02_Tasks_and_Evaluation.md`](02_Tasks_and_Evaluation.md) · [`../ML_Fundamentals/Transformers_Cheatsheet.md`](../ML_Fundamentals/Transformers_Cheatsheet.md) for contextual embeddings and attention · [`../ML_Fundamentals/GenAI_Cheatsheet.md`](../ML_Fundamentals/GenAI_Cheatsheet.md) for tokenization in the LLM era.

---

# 1. Preprocessing

The pipeline, and what each step costs you.

| Step | What it does | What it destroys |
|---|---|---|
| **Unicode normalization** (NFC/NFKC) | Canonicalizes equivalent code-point sequences — `é` as one char vs `e` + combining accent | NFKC also folds `ﬁ`→`fi`, `²`→`2`, full-width→ASCII. Usually wanted, occasionally not |
| **Lowercasing** | Collapses `Apple`/`apple` | The company/fruit distinction, and all-caps emphasis. Bad for NER |
| **Punctuation / special-char stripping** | Reduces vocabulary | Negation cues, emoticons, sentence boundaries, URLs |
| **Stopword removal** | Drops high-frequency function words | Negation (`not`), and anything where word order or function words carry meaning |
| **Stemming** | Crude suffix chopping — `studies`→`studi` | Readability; produces non-words. Fast, aggressive |
| **Lemmatization** | Dictionary + POS-aware reduction — `studies`→`study`, `better`→`good` | Nothing much, but needs a POS tag and a lexicon. Slower |

## Stemming vs lemmatization — the interview distinction

**Stemming** applies hand-written suffix rules (Porter, Snowball) with no knowledge of the word. It's fast, language-specific, and frequently produces non-words. Two errors to name:
- **Over-stemming** — `universal`, `university`, `universe` all → `univers`. Distinct meanings collapsed.
- **Under-stemming** — `data` and `datum` stay separate. Same meaning, not collapsed.

**Lemmatization** maps to a real dictionary form, needs the part of speech to disambiguate (`saw` → `see` as a verb, `saw` as a noun), and is slower.

**When it matters**: both exist to shrink vocabulary and merge sparse variants — which mattered enormously for bag-of-words models. With **subword tokenization** (BPE/WordPiece) feeding a neural model, the model learns morphological relationships itself, so stemming and lemmatization are largely obsolete in modern pipelines. Saying *that* is the better answer than reciting the definitions.

## Tokenization

| Level | Trade-off |
|---|---|
| **Character** | Tiny vocabulary, no OOV ever, but very long sequences and the model must learn words from scratch |
| **Word** | Intuitive, short sequences, but huge vocabulary, a hard OOV problem, and fails on agglutinative languages |
| **Subword** (BPE, WordPiece, Unigram) | The compromise that won — frequent words stay whole, rare words decompose. No OOV with byte-level fallback |

Subword algorithm details are in [`../ML_Fundamentals/GenAI_Cheatsheet.md`](../ML_Fundamentals/GenAI_Cheatsheet.md) §1. One classical point worth keeping here: tokenizing on whitespace is wrong for most of the world — Chinese and Japanese have no spaces, and German compounds and Turkish agglutination make word-level vocabularies explode.

---

# 2. Count-Based Representations

## One-hot and bag-of-words

A document becomes a vector over the vocabulary. **One-hot** per token; **bag-of-words (BoW)** sums them into counts per document.

```
vocab = [the, cat, sat, on, mat]
"the cat sat on the mat"  ->  [2, 1, 1, 1, 1]
```

Properties: sparse, high-dimensional (|V| can be 10⁵–10⁶), and **order-free** — "dog bites man" and "man bites dog" are identical vectors. That's the defining limitation.

## N-grams

Partially recover order by treating contiguous runs as units. Bigrams of the sentence above: `the cat`, `cat sat`, `sat on`, `on the`, `the mat`.

The trade: unigram+bigram typically helps text classification measurably (it captures negation — `not good` becomes its own feature), but vocabulary grows roughly quadratically and sparsity worsens. Trigrams rarely pay for themselves outside large corpora. In practice `ngram_range=(1,2)` with a `min_df` floor is the standard starting point.

## TF-IDF

Raw counts over-weight words that are frequent everywhere. TF-IDF down-weights them:

```
tfidf(t, d) = tf(t, d) × idf(t)

idf(t) = log( N / df(t) )          N = number of documents
                                   df(t) = documents containing t
```

**Smoothed** (scikit-learn's default) adds 1 to numerator, denominator and result, which prevents division by zero and keeps the weight of a universal term non-zero:

```
idf(t) = log( (1 + N) / (1 + df(t)) ) + 1
```

Then each document vector is **L2-normalized**, so document length doesn't dominate the similarity.

**Why it works:** `tf` says "important in this document", `idf` says "discriminative across the corpus". A word scores highly only when both hold. It's a hand-designed feature weighting that remains a genuinely strong baseline — fast, interpretable, and competitive with neural models on small, topic-driven classification tasks.

**The hashing trick** — when vocabulary is unbounded or streaming, hash tokens into a fixed `k` buckets instead of maintaining a vocabulary. Fixed memory, no fitted state, handles unseen words; the cost is collisions and the loss of the ability to map a feature back to its word.

---

# 3. Static Word Embeddings

The shift: from a sparse |V|-dimensional vector per word to a **dense ~100–300 dimensional learned** vector, where geometry encodes meaning.

The underlying principle is the **distributional hypothesis** — *"you shall know a word by the company it keeps"* (Firth). Words appearing in similar contexts get similar vectors.

## Co-occurrence, PPMI and LSA

The oldest version, and worth knowing because it explains why the neural versions work.

Build a word × context co-occurrence matrix, then reweight. Raw counts are dominated by frequent words, so use **Pointwise Mutual Information**:

```
PMI(w, c) = log  P(w, c)
                 ─────────
                 P(w) P(c)
```

PMI is high when a word and context co-occur more than chance predicts. It's unstable for unseen pairs (log 0 = −∞), so use **PPMI** = `max(0, PMI)`.

Then factor it with **truncated SVD** to get dense vectors — this is **LSA/LSI**. It is literally PCA on a reweighted co-occurrence matrix.

The result that ties the era together: **word2vec with negative sampling is implicitly factorizing a shifted PPMI matrix** (Levy & Goldberg, 2014). The neural method and the count method are two routes to the same place.

## word2vec

Two architectures, both shallow:

| | Predicts | Better for |
|---|---|---|
| **CBOW** | The centre word *from* its context | Frequent words; faster to train |
| **Skip-gram** | Each context word *from* the centre | Rare words; small corpora |

**Skip-gram with negative sampling (SGNS)** is the one to be able to write. The naive objective needs a softmax over the entire vocabulary per step — `O(|V|)`, impossible. Negative sampling replaces it with binary classification: push real (word, context) pairs together, push `k` sampled fake ones apart.

```
J = log σ( v'_c · v_w )  +  Σ   E          [ log σ( −v'_n · v_w ) ]
                          i=1..k  n ~ P_n(w)
```

- `v_w` — the vector of the centre word (input embedding)
- `v'_c` — the vector of the true context word (output embedding)
- `σ` — sigmoid; `k` — number of negatives, typically 5–20 for small corpora, 2–5 for large

Two details that get asked:
- **Negatives are sampled from a unigram distribution raised to the 3/4 power** — `P(w) ∝ count(w)^0.75`. This flattens the distribution, sampling rare words more often than their frequency alone would.
- **Frequent-word subsampling** — tokens are randomly discarded with probability rising in frequency, which both speeds training and improves rare-word vectors.

Each word ends up with **two** vectors (input and output); the input matrix is normally what you keep.

**Analogies.** The famous `king − man + woman ≈ queen` works because consistent semantic relations show up as roughly consistent offset vectors. Worth knowing the caveats: the standard evaluation excludes the input words from the answer candidates, which flatters the result, and many analogy categories work far less well than the cherry-picked examples suggest.

## GloVe

Where word2vec is predictive and local, GloVe is explicitly a **global matrix factorization** with a weighted least-squares objective on log co-occurrence counts:

```
J = Σ   f(X_ij) ( w_i·w̃_j + b_i + b̃_j − log X_ij )²
   i,j
```

`X_ij` is how often word *j* appears in word *i*'s context. The weighting `f` caps the influence of very frequent pairs:

```
f(x) = (x/x_max)^α  if x < x_max   else  1        (α ≈ 0.75, x_max ≈ 100)
```

The motivating insight is that **ratios** of co-occurrence probabilities carry meaning better than the probabilities themselves — `P(solid|ice)/P(solid|steam)` is large, `P(water|ice)/P(water|steam)` is near 1.

## fastText

word2vec, but a word's vector is the **sum of its character n-gram vectors** (plus the whole word). `where` with n = 3 contributes `<wh`, `whe`, `her`, `ere`, `re>`.

Two consequences, and both are the reason to pick it:
- **No OOV.** An unseen word still has n-grams, so it still gets a vector.
- **Morphology for free.** `run`/`running`/`runner` share substrings, so they land near each other without any stemming.

Strongest on morphologically rich languages (Turkish, Finnish, German) and on noisy text with typos.

## The shared limitation

All three produce **one vector per word type**. `bank` gets a single vector averaging the riverbank and the financial sense. Polysemy is unrepresentable — and that is precisely the gap **contextual** embeddings (ELMo, BERT and everything after) exist to close: the vector is computed from the sentence, so `bank` in two sentences gets two different vectors.

For that half of the story see [`../ML_Fundamentals/Transformers_Cheatsheet.md`](../ML_Fundamentals/Transformers_Cheatsheet.md).

---

# 4. Choosing a Representation

| Situation | Reach for |
|---|---|
| Small labeled set, topic-driven classes, need interpretability | **TF-IDF + linear model** — still the right baseline, and often wins |
| Need word similarity / analogies, limited compute | Static embeddings (GloVe or fastText pretrained) |
| Morphologically rich language, typos, unseen words | **fastText** |
| Meaning depends on context; polysemy matters | Contextual embeddings |
| Semantic search / RAG retrieval | Sentence embeddings (bi-encoder), then a cross-encoder re-ranker |
| Streaming or unbounded vocabulary | Hashing vectorizer |

**The point worth making out loud in an interview:** reach for TF-IDF + logistic regression first and measure it. It trains in seconds, is fully interpretable, has no serving cost, and on small topic-classification problems the gap to a fine-tuned transformer is often small enough that the transformer isn't worth its latency and ops burden. Establishing that baseline *before* proposing a neural model is the reasoning being scored.

---

# 5. Document-Level Vectors

Given word vectors, how do you get one vector for a sentence or document?

| Method | Note |
|---|---|
| **Mean pooling** | Average the word vectors. Crude, surprisingly hard to beat |
| **TF-IDF weighted mean** | Weight each word vector by its IDF. Usually better than plain mean |
| **SIF** (smooth inverse frequency) | Weighted average, then remove the first principal component. Strong classical baseline |
| **Doc2Vec / paragraph vectors** | Learns a document vector jointly with words |
| **[CLS] token** | BERT's pooled output — needs fine-tuning to be good; raw [CLS] from a pretrained model is a poor sentence vector |
| **Mean-pooled transformer states** | Better than raw [CLS] off the shelf |
| **Sentence-transformers / bi-encoders** | Trained with a contrastive objective explicitly *for* sentence similarity. The right default now |

**The gotcha**: a pretrained BERT's `[CLS]` vector is widely assumed to be a good sentence embedding and generally isn't — it was trained for next-sentence-prediction, not similarity. Models trained with a sentence-level contrastive objective beat it substantially. Knowing that distinction is a reliable signal of having actually built retrieval.

---

## Runnable companions

- [`03_Notebook_Text_Classification.ipynb`](03_Notebook_Text_Classification.ipynb) — BoW vs TF-IDF vs n-grams, with feature inspection and error analysis
- [`04_Notebook_Embeddings_From_Scratch.ipynb`](04_Notebook_Embeddings_From_Scratch.ipynb) — co-occurrence → PPMI → SVD, then skip-gram with negative sampling implemented in numpy
