# NLP

Natural language processing from the ground up — how text becomes numbers, what the canonical
tasks are, and how each one is actually scored. Plus three runnable notebooks that implement
the core mechanics rather than calling a library.

The emphasis is deliberately on the **pre-transformer foundations**, because that's what this
repo was missing and what interviews still probe. Attention, BERT and the LLM stack are
covered in [`../ML_Fundamentals/Transformers_Cheatsheet.md`](../ML_Fundamentals/Transformers_Cheatsheet.md)
and [`../ML_Fundamentals/GenAI_Cheatsheet.md`](../ML_Fundamentals/GenAI_Cheatsheet.md).

---

## Contents

| # | File | What's in it |
|---|---|---|
| 1 | [01_Text_Representation.md](01_Text_Representation.md) | Preprocessing (stemming vs lemmatization, tokenization levels) · BoW, n-grams, TF-IDF with the formulas · PPMI and LSA · **word2vec/SGNS, GloVe and fastText objectives written out** · document-level vectors · how to choose |
| 2 | [02_Tasks_and_Evaluation.md](02_Tasks_and_Evaluation.md) | The five task shapes · text classification · **BIO tagging and linear-chain CRFs** · topic modeling · generation tasks · the full metric table (BLEU vs ROUGE, perplexity, entity F1) · rapid-fire Q&A |
| 3 | [03_Notebook_Text_Classification.ipynb](03_Notebook_Text_Classification.ipynb) | TF-IDF baseline end to end: representation comparison, classifier comparison, regularization sweep, feature inspection, error analysis |
| 4 | [04_Notebook_Embeddings_From_Scratch.ipynb](04_Notebook_Embeddings_From_Scratch.ipynb) | Co-occurrence → PPMI → SVD, **and** skip-gram with negative sampling, both in numpy. Then: do they agree? |
| 5 | [05_Notebook_Sequence_Labeling.ipynb](05_Notebook_Sequence_Labeling.ipynb) | NER with BIO tags, span decoding, entity-level F1 from scratch, and why token accuracy lies |

## The notebooks

All three **run offline in seconds to a minute** — numpy, scikit-learn and matplotlib only.
No downloads, no model weights, no API keys.

```bash
pip install numpy scikit-learn
jupyter notebook           # or open them in VS Code
```

They use **synthetic corpora on purpose**. A real dataset with disjoint class vocabularies
scores 100% and demonstrates nothing; these are tuned so the lesson is visible:

| Notebook | What it's tuned to show |
|---|---|
| Text classification | TF-IDF ≈ 0.89 vs raw counts ≈ 0.84 — the IDF effect is measurable, not asserted. Plus visible overfitting as `C` rises, and real errors to analyse |
| Embeddings | Same-theme purity reaches 1.0 by both routes, with ~0.6 neighbour overlap between them — the Levy & Goldberg result, observed |
| Sequence labeling | ~87% of tokens are `O`, so token accuracy hits **0.98** while entity F1 is **0.69** — the gap *is* the lesson |

Each ends with a "try next" suggestion that changes a parameter and breaks the result in an
instructive way.

## Where this connects

| If you're after | Go to |
|---|---|
| Attention, transformer internals, BERT, RoPE, KV cache | [`Transformers_Cheatsheet`](../ML_Fundamentals/Transformers_Cheatsheet.md) |
| Modern tokenization, RAG, LLM evaluation, agents | [`GenAI_Cheatsheet`](../ML_Fundamentals/GenAI_Cheatsheet.md) |
| RNN / LSTM / GRU / seq2seq mechanics | [`DL_Concepts_Cheatsheet`](../ML_Fundamentals/DL_Concepts_Cheatsheet.md) |
| Why linear models suit sparse text (`p ≫ n`) | [`Feature_Engineering_Selection`](../ML_Fundamentals/Feature_Engineering_Selection.md) §5 |
| Precision/recall, PR vs ROC, calibration | [`Plots_Visual_Diagnostics`](../ML_Fundamentals/Plots_Visual_Diagnostics.md) §8–11 |
| Handling class imbalance | [`../Question_Bank/README.md`](../Question_Bank/README.md) R1Q6 |

## Five things worth being able to say

1. **TF-IDF + logistic regression is still the right first move** on a small topic-classification
   problem — seconds to train, fully interpretable, and often within a point or two of a
   fine-tuned transformer that costs far more to serve.
2. **Stemming and lemmatization are largely obsolete** when subword tokenization feeds a neural
   model; the model learns morphology itself.
3. **Negative sampling exists to avoid the `O(|V|)` softmax**, and negatives are drawn from the
   unigram distribution raised to the 3/4 power.
4. **Static embeddings give one vector per word type**, so polysemy is unrepresentable — which
   is precisely the gap contextual embeddings close.
5. **Never report token accuracy for NER.** Entity-level F1 with exact span and type match is
   the standard, and the gap between the two is usually dramatic.
