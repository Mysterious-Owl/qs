# CatBoost — Interview Deep Dive

The third boosting library, and the one candidates usually can't explain past "it handles categoricals." This file goes to the mechanism level.

Companions — deliberately no overlap:
- [`ML_Algorithms_Cheatsheet.md` §19](ML_Algorithms_Cheatsheet.md#19-catboost) — the one-page reference card
- [`XGBoost_Trees_Deep_Dive.md`](XGBoost_Trees_Deep_Dive.md) — boosting fundamentals, the gain formula, regularization theory. **Read that first**; everything here assumes it.
- [`Feature_Engineering_Selection.md` §3](Feature_Engineering_Selection.md#3-high-cardinality-categorical-features) — manual categorical encoding, which is what CatBoost automates

---

## 1. Why CatBoost exists — the two problems

CatBoost (Yandex, 2017) is gradient boosting with the same additive objective as any GBM. It exists to fix two specific leakage-shaped problems that plain GBM implementations have:

**Problem 1 — target leakage in categorical encoding.** Target (mean) encoding replaces a category with the average target for that category. Done naively, the row's *own* label is inside its own feature value. The model learns to read the label off the feature, training error collapses, and validation performance is garbage. This is severe for high-cardinality features: a category appearing once gets encoded as exactly its own target.

**Problem 2 — prediction shift in boosting itself.** Standard GBM computes the gradient for row *i* using a model that was *trained on row i*. So the gradient distribution on training data is systematically different from the distribution on unseen data. The model is slightly, consistently biased toward its own training set — a subtle overfitting source present in every plain GBM.

CatBoost's answer to both is the same idea, applied twice: **ordering**. Pretend the data arrived sequentially, and only ever use "the past" to compute anything about "the present."

Everything distinctive about CatBoost follows from this — plus a third, unrelated choice (oblivious trees) made for speed and regularization.

---

## 2. Ordered Target Statistics — leakage-safe categorical encoding

For a categorical feature, CatBoost computes a **target statistic** (a "CTR" — Counter / Click-Through-Rate, from Yandex's ad-ranking origins) rather than one-hot encoding it.

**The naive version** (what causes Problem 1):

```
encoding(category c) = mean(y) over all rows with category c
```

**CatBoost's ordered version.** Fix a random permutation σ of the rows. For row *k*, encode its category using **only the rows that precede k in σ**:

```
                Σ_{j < k} [ x_j = x_k ] · y_j   +   a · p
  x̂_k  =  ─────────────────────────────────────────────────
                Σ_{j < k} [ x_j = x_k ]        +   a
```

- `[x_j = x_k]` — 1 when row *j* has the same category as row *k*, else 0
- `p` — a **prior**, typically the global target mean
- `a` — the prior's weight (smoothing strength, `ctr_target_border_count` / prior settings)

Two things this buys you:

1. **No leakage.** Row *k*'s own label `y_k` never appears in its own encoding, because the sum runs over `j < k` only.
2. **Smoothing.** The `a·p` terms shrink rare categories toward the global mean, exactly like the manual smoothing formula in [`Feature_Engineering_Selection.md` §3](Feature_Engineering_Selection.md#3-high-cardinality-categorical-features). A category seen once gets an encoding dominated by the prior, not by its single noisy observation.

**The variance problem, and why permutations are plural.** Early rows in the permutation have almost no history, so their encodings are high-variance (row 2 is encoded from row 1 alone). CatBoost mitigates this by using **several random permutations** and averaging/rotating between them across boosting iterations, so no single unlucky ordering dominates. This is why CatBoost's categorical handling costs more compute than LightGBM's.

**At inference time** the ordering is irrelevant — the statistic is computed once over the whole training set and applied to new rows.

**Why this beats doing it yourself**: manual out-of-fold target encoding is the standard workaround, and it works, but it's fiddly, easy to get subtly wrong, and has to be re-implemented consistently in training *and* serving. CatBoost makes it a library guarantee rather than a pipeline you maintain. That's the honest answer to "why not just target-encode manually?"

---

## 3. Ordered Boosting — fixing prediction shift

Same idea, applied to the gradients.

**Plain boosting**: at iteration *t*, compute residuals/gradients for every row using model `F_{t-1}` — which was trained on all of those rows. Gradients are optimistically biased.

**Ordered boosting**: conceptually, maintain a set of models `M_1 … M_n`, where `M_i` is trained on only the first *i* rows of a permutation. The gradient for row `i+1` is computed with `M_i` — a model that has never seen row `i+1`.

Done literally, that's O(n²) models and completely impractical. CatBoost approximates it with a manageable number of permutations and shared structure, which is the engineering contribution of the paper.

**Controlled by `boosting_type`:**

| Value | Behaviour |
|---|---|
| `Ordered` | The unbiased scheme above. Slower, better on **small datasets** where prediction shift matters most |
| `Plain` | Classic GBM gradient computation. Faster, the sensible default once you have plenty of rows |

CatBoost picks between them automatically based on dataset size (Ordered for small data, Plain for large). Setting it explicitly is a legitimate tuning knob: if you have a few thousand rows and are overfitting, try `boosting_type='Ordered'`.

**The interview point**: ordered boosting addresses a bias that exists in *every* GBM, XGBoost included — it just usually goes unnoticed because it's small relative to other error sources. CatBoost is the implementation that treats it as a first-class problem.

---

## 4. Oblivious (symmetric) trees

CatBoost's default tree is **oblivious**: at a given depth, **every node uses the same (feature, threshold) split**.

```
Standard tree (XGBoost)            Oblivious tree (CatBoost)
        [f3 < 5]                        [f3 < 5]          <- depth 0: one rule
       /        \                      /        \
  [f7 < 2]    [f1 < 9]            [f7 < 2]    [f7 < 2]    <- depth 1: SAME rule
   /    \      /    \              /    \      /    \
  L1    L2    L3    L4            L1    L2    L3    L4
```

A depth-*d* oblivious tree is therefore just a **decision table** with `2^d` leaves, indexed by *d* binary answers.

**What this buys:**

- **Extremely fast inference.** The leaf index is a *d*-bit number built by concatenating *d* comparisons — branchless, vectorizable, cache-friendly. CatBoost's prediction latency is its standout practical advantage, often several times faster than an equivalent XGBoost model. This matters a lot for real-time scoring (fraud decisioning, ad ranking).
- **Regularization.** Forcing one rule per level is a hard constraint on tree expressiveness, which resists overfitting — part of why CatBoost's defaults work well untuned.
- **Balanced trees by construction**, so no leaf-wise runaway growth of the kind LightGBM's `num_leaves` guards against.

**What it costs**: each individual tree is weaker than a free-form tree, because it can't spend depth where the signal is. CatBoost compensates with more iterations. Net accuracy is usually comparable; training is often slower.

**You can turn it off** with `grow_policy`:

| `grow_policy` | Behaviour |
|---|---|
| `SymmetricTree` | Oblivious. Default. Keeps the fast-inference property |
| `Depthwise` | Level-wise like XGBoost |
| `Lossguide` | Leaf-wise like LightGBM (pair with `max_leaves`) |

Switching away from `SymmetricTree` **forfeits the fast-inference advantage** — the model is no longer a decision table. If someone chose CatBoost *for* latency and then set `Lossguide`, they've traded away the reason they picked it.

---

## 5. Automatic feature combinations

CatBoost greedily builds **combinations of categorical features** during tree construction — e.g. `country × device × ad_slot` becomes a single new categorical whose levels are the observed tuples, then target-encoded like any other.

It's greedy and incremental: at each split, the current tree's categorical splits are combined with the remaining categorical features, and only combinations that actually help survive.

**Controlled by `max_ctr_complexity`** (how many features may be combined; commonly 4 by default, 1 disables combinations entirely).

**Why it matters**: interaction effects between categoricals are exactly what one-hot encoding destroys and what trees struggle to rebuild — a tree needs many splits to express "this country *and* this device." Getting them constructed automatically is a genuine advantage on the kind of data CatBoost targets.

**The cost**: combination search is a major driver of CatBoost's training time and memory. Lowering `max_ctr_complexity` to 1–2 is the first thing to try when CatBoost is too slow.

---

## 6. Hyperparameters

**Core**
| Param | Does | Typical |
|---|---|---|
| `iterations` | Number of trees (= `n_estimators`) | 500–5000, set by early stopping |
| `learning_rate` | Shrinkage per tree | 0.01–0.3; CatBoost auto-selects a sensible value if unset |
| `depth` | Tree depth. **The main capacity dial** | 4–10 (default 6; max 16) |
| `l2_leaf_reg` | L2 on leaf values | 1–10 (default 3) |
| `loss_function` | `RMSE`, `MAE`, `Quantile`, `Logloss`, `MultiClass`, `YetiRank`… | — |
| `eval_metric` | What early stopping watches | `AUC`, `PRAUC`, `RMSE`, `F1`… |

**Overfitting control**
| Param | Does |
|---|---|
| `od_type` | Overfitting detector: `Iter` (patience, like `early_stopping_rounds`) or `IncToDec` |
| `od_wait` | Patience in iterations |
| `random_strength` | Noise added to split scores — randomizes structure, fights overfitting |
| `bagging_temperature` | Bayesian bootstrap intensity. 0 = no randomness; higher = more aggressive resampling |
| `subsample` | Row sampling (for Bernoulli / MVS bootstrap types) |
| `rsm` | Feature sampling per split (`colsample`). **CPU-only for symmetric trees** — silently unavailable in some GPU configs |

**Categorical handling**
| Param | Does |
|---|---|
| `cat_features` | Indices or names of categorical columns. **The whole point — set this** |
| `one_hot_max_size` | Categoricals with ≤ this many levels are **one-hot encoded** instead of target-encoded |
| `max_ctr_complexity` | Max number of features in an automatic combination |

**Structure / performance**
| Param | Does |
|---|---|
| `grow_policy` | `SymmetricTree` / `Depthwise` / `Lossguide` (§4) |
| `boosting_type` | `Ordered` / `Plain` (§3) |
| `bootstrap_type` | `Bayesian`, `Bernoulli`, `MVS`, `Poisson` (GPU), `No`. MVS = Minimal Variance Sampling, a variance-reduced row sampler |
| `border_count` | Numeric binning granularity (≈ `max_bin`); commonly 254 on CPU, lower on GPU |
| `task_type` | `CPU` / `GPU` |
| `nan_mode` | `Min` / `Max` / `Forbidden` — which side NaNs go (§8) |

**Imbalance**
| Param | Does |
|---|---|
| `auto_class_weights` | `Balanced` or `SqrtBalanced` — computes weights for you |
| `scale_pos_weight` | Manual positive-class weight, same idea as XGBoost |
| `class_weights` | Explicit per-class weights |

> Defaults shift between CatBoost versions (particularly `bootstrap_type` and the Ordered/Plain threshold). Check `model.get_all_params()` for what your version actually used rather than quoting a remembered default — and say that in an interview instead of asserting a number you're unsure of.

**Tuning order** (same staged logic as [`XGBoost_Trees_Deep_Dive.md` §7](XGBoost_Trees_Deep_Dive.md#7-systematic-xgboost-tuning)):

1. Baseline with defaults + early stopping. **CatBoost's defaults are genuinely strong** — often within a couple of points of a tuned model, which is a real selling point.
2. `depth` (4–10) — biggest effect.
3. `l2_leaf_reg` (log scale, 1–30).
4. `learning_rate` down + more iterations, with early stopping.
5. `random_strength` / `bagging_temperature` if still overfitting.
6. `one_hot_max_size` and `max_ctr_complexity` if categorical handling looks like the bottleneck (accuracy or speed).

---

## 7. Categorical features in practice

**Pass them, don't encode them:**

```python
from catboost import CatBoostClassifier, Pool

cat_cols = ["merchant_id", "city", "device_type", "product_category"]

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    eval_metric="PRAUC",       # imbalanced problem
    auto_class_weights="Balanced",
    od_type="Iter", od_wait=100,
    random_seed=42,
    verbose=200,
)
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
```

**Requirements and traps:**

- `cat_features` values must be **`int` or `str`, never `float`.** A float categorical column raises an error; NaN in a float column silently makes it a float column. Cast to string first — `df[c] = df[c].fillna("__missing__").astype(str)` — which also gives missing values their own honest category.
- **`one_hot_max_size` decides the encoding path.** Below it, one-hot; above it, ordered target statistics. Low-cardinality features often do better one-hot, so raising this (to ~10) is a cheap experiment.
- **Don't pre-encode.** Passing already-target-encoded columns as numeric defeats the entire mechanism *and* reintroduces the leakage CatBoost exists to prevent. Feed raw categories.
- **`Pool` is worth using** for anything non-trivial — it binds features, labels, categorical indices, weights and baselines together, and avoids re-quantizing the data on every call.

---

## 8. Missing values

**Numeric**: controlled by `nan_mode`.

| Mode | Behaviour |
|---|---|
| `Min` | NaN treated as smaller than every observed value (default) |
| `Max` | NaN treated as larger than every observed value |
| `Forbidden` | NaN raises an error — useful when missingness should never happen |

Note the contrast with XGBoost, which **learns** a per-node default direction from the gain ([`XGBoost_Trees_Deep_Dive.md` §10](XGBoost_Trees_Deep_Dive.md#10-how-does-xgboost-handle-missing-values)). CatBoost's rule is fixed and global rather than learned per split. Either way missingness stays informative, but XGBoost's is the more adaptive mechanism — a fair point to concede if asked to compare.

**Categorical**: `None`/NaN becomes its own category. Since you're casting to string anyway (§7), make it explicit with a `"__missing__"` sentinel so you can see it in feature analysis.

---

## 9. Text and embedding features

Less known, occasionally a differentiator:

- **`text_features`** — pass raw text columns and CatBoost tokenizes them, builds dictionaries, and derives features (bag-of-words and Naive-Bayes/BM25-style estimators) internally. Useful for short fields — product titles, merchant descriptors, free-text reason codes — where standing up a separate TF-IDF pipeline isn't worth it.
- **`embedding_features`** — pass precomputed dense vectors (e.g. sentence-transformer output) as a first-class feature type rather than 384 loose numeric columns.

Neither replaces a proper NLP model for real language work. They exist so a *mostly tabular* model can absorb a text column without a second pipeline. That framing — "it's for tabular data with a text column, not for text data" — is the right way to describe it.

---

## 10. CatBoost vs XGBoost vs LightGBM

| | **CatBoost** | **XGBoost** | **LightGBM** |
|---|---|---|---|
| Tree growth | Oblivious / symmetric | Level-wise (depth-wise) | Leaf-wise (best-first) |
| Categoricals | **Native, ordered target statistics** | Must pre-encode | Native (Fisher-style optimal split) |
| Feature combinations | **Automatic** | Manual | Manual |
| Gradient bias | **Ordered boosting** corrects it | Uncorrected | Uncorrected |
| Default quality | **Strongest out-of-the-box** | Needs tuning | Needs tuning (esp. `num_leaves`) |
| Training speed | Slowest, especially with many categoricals | Middle | **Fastest** |
| **Inference speed** | **Fastest** (decision table) | Middle | Middle |
| Overfitting risk | Lowest (symmetric trees + ordered boosting) | Middle | **Highest** (leaf-wise needs capping) |
| Missing values | Fixed rule (`nan_mode`) | **Learned per node** | Native |
| Ecosystem maturity | Good | **Largest** | Large |
| Text features | Built-in | No | No |

**How to answer "which would you pick?"** — the framework, not a favourite:

- **Many high-cardinality categoricals** (merchant, ZIP, SKU, diagnosis code) → **CatBoost**. This is its home turf, and the leakage-safety argument is the strongest one you can make.
- **Large mostly-numeric data, training speed matters** → **LightGBM**.
- **Need the widest ecosystem** — SHAP tooling, deployment targets, team familiarity, every tutorial ever written → **XGBoost**.
- **Tight inference latency budget** → **CatBoost** (oblivious trees), assuming you keep `SymmetricTree`.
- **Small dataset, overfitting easily** → **CatBoost** with `boosting_type='Ordered'`.
- **Honest closer**: on most tabular problems, all three land within a point or two of each other once tuned. Feature quality and label quality dominate the library choice, and saying so is a maturity signal rather than a dodge.

---

## 11. When CatBoost is the wrong choice

Everything in [`XGBoost_Trees_Deep_Dive.md` §5](XGBoost_Trees_Deep_Dive.md#5-when-does-xgboost-fail-and-what-beats-it) applies — trees are trees, so CatBoost equally cannot extrapolate, struggles on wide-short (`p ≫ n`) data, and has no business on raw images or sequences. On top of that:

| Situation | Why | Prefer |
|---|---|---|
| **Purely numeric features** | The categorical machinery is its entire edge; without it you pay training cost for nothing | LightGBM / XGBoost |
| **Very large data, tight training budget** | Slowest of the three, and CTR + combination search scales badly | LightGBM |
| **Ecosystem-heavy workflows** | Fewer integrations, fewer Stack Overflow answers | XGBoost |
| **You need learned missing-value routing** | `nan_mode` is a fixed rule, not learned per split | XGBoost |
| **Highly irregular signal needing deep asymmetric trees** | Oblivious trees are deliberately constrained | XGBoost / LightGBM, or CatBoost with `Lossguide` |

---

## 12. Common pitfalls

1. **Forgetting `cat_features`.** Silent and expensive: string columns error out, but integer-coded categories (`city_id = 4821`) are treated as *numeric* and split on ordinally. Model still trains, quality quietly drops. Always verify with `model.get_cat_feature_indices()`.
2. **Pre-target-encoding before passing to CatBoost.** Re-introduces exactly the leakage ordered target statistics exist to prevent.
3. **Float categorical columns** → error. Cast to `str`.
4. **Expecting LightGBM training speed.** It's slower by design; the payoff is inference speed and defaults.
5. **`Lossguide`/`Depthwise` chosen for accuracy, sacrificing the latency win** without noticing the trade.
6. **Train/serve category mismatch.** Categories unseen in training fall back to the prior at inference — fine, but a *large* share of unseen categories means drift, and needs monitoring rather than silent tolerance.
7. **Trusting a remembered default.** Defaults move between versions — read `model.get_all_params()`.
8. **Reading probabilities after `auto_class_weights`.** Class weighting distorts calibration exactly as it does elsewhere; recalibrate if the probability itself is consumed downstream ([`Plots_Visual_Diagnostics.md` §11](Plots_Visual_Diagnostics.md#11-calibration-curve-reliability-diagram)).

---

## 13. Rapid-fire

**Q: What makes CatBoost different in one sentence?**
Ordered target statistics for leakage-free native categorical encoding, ordered boosting to remove the gradient bias every plain GBM carries, and oblivious trees for very fast inference.

**Q: What's a CTR here?**
A target statistic — the smoothed mean target for a category, computed only from rows preceding the current one in a random permutation, so a row's own label can't enter its own feature.

**Q: What is prediction shift?**
Standard boosting computes row *i*'s gradient with a model trained on row *i*, so training-set gradients are biased relative to unseen data. Ordered boosting computes each row's gradient from a model fitted only on rows before it.

**Q: What's an oblivious tree, and what does it buy?**
Every node at a given depth shares one split rule, so the tree is a decision table with `2^d` leaves. Inference becomes a branchless bit-index lookup — very fast — and the constraint regularizes.

**Q: Why not just target-encode manually with out-of-fold means?**
You can, and it works. CatBoost makes it a library guarantee instead of pipeline code you must keep consistent between training and serving, adds principled smoothing, and builds categorical *combinations* automatically.

**Q: CatBoost or LightGBM?**
Many high-cardinality categoricals or a tight inference budget → CatBoost. Large mostly-numeric data with training-speed pressure → LightGBM. Benchmark both; the gap after tuning is usually small.

**Q: Biggest CatBoost mistake you see?**
Not passing `cat_features`, so integer-coded categories get treated as ordered numerics. It trains fine and quietly underperforms.

**Q: Does it handle missing values?**
Yes — `nan_mode` sends NaN to the min or max side (a fixed global rule), and categorical NaN becomes its own level. Unlike XGBoost, the direction is not learned per node.
