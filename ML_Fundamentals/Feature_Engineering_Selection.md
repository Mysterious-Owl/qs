# Feature Engineering, Selection & Data Leakage

The pre-modelling half of the pipeline — where most real accuracy comes from, and where the subtle failures live.

Companions: [`XGBoost_Trees_Deep_Dive.md`](XGBoost_Trees_Deep_Dive.md), [`ML_Algorithms_Cheatsheet.md`](ML_Algorithms_Cheatsheet.md), and [`../Question_Bank/`](../Question_Bank/README.md) for the source questions.

---

# 1. Data Leakage

**Definition**: information available to the model at training time that will **not** be legitimately available at prediction time — or that encodes the target itself. It inflates validation scores and the model then collapses in production.

**The tell**: performance that's too good. A 0.99 AUC on a hard business problem is a leakage investigation, not a success.

## Types

**Target leakage** — a feature is a consequence (or restatement) of the target:
- `loan_written_off_flag` when predicting default.
- `discharge_date` when predicting length of stay.
- `num_late_payment_notices` when predicting delinquency — the notices only exist *because* of the outcome.
- **Test**: for each suspiciously strong feature, ask "at the exact moment I need to score, does this value exist yet?"

**Train-test contamination** — preprocessing fitted on the full dataset before splitting:
- `StandardScaler`, `SimpleImputer`, PCA, or feature selection fitted on train+test. The test statistics bleed in.
- **Fix**: fit every transformer on train only, inside a `Pipeline`, and let cross-validation refit per fold.
  ```python
  pipe = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("m", XGBClassifier())])
  cross_val_score(pipe, X, y, cv=5)   # transformers refit inside each fold
  ```

**Temporal leakage** — training on future data to predict the past:
- Random `train_test_split` on time-series data. Use a **time-based split** (`TimeSeriesSplit`) always.
- Features built from windows that extend past the prediction timestamp (e.g. a 30-day average that includes days after the decision point).

**Group leakage** — the same entity appears in both train and test:
- Multiple loans from one customer, multiple images of one patient, several sessions from one user. The model memorizes the entity, not the pattern.
- **Fix**: `GroupKFold` / `StratifiedGroupKFold` on the entity ID.

**Duplicate leakage** — exact or near-duplicate rows straddling the split. Deduplicate *before* splitting.

**Target-encoding leakage** — encoding a category by its mean target using the row's own target. This is why CatBoost uses ordered target statistics, and why manual target encoding must be done **out-of-fold** with smoothing (see §4).

**Metadata / artifact leakage** — an incidental artifact correlates with the label: row ID order, file naming, a scanner ID that differs by hospital, a timestamp that differs by data source.

## Prevention checklist

1. Split first — before any exploration or fitting.
2. All preprocessing inside a `Pipeline`.
3. Time-based splits for temporal data; group-aware splits for repeated entities.
4. Audit top-importance features individually against "would I know this at scoring time?"
5. Build a **point-in-time correct** feature store: every feature timestamped, joined as-of the decision time.
6. Verify offline/online feature parity — the classic production leak is a training feature computed differently (or later) than the serving one.
7. Hold out a final test set touched exactly once.

---

# 2. Feature Selection

## Why bother
Reduce overfitting and variance, cut training/inference cost, improve interpretability, remove multicollinearity, and drop features that are unstable or unavailable in production. Note the tension: tree ensembles tolerate irrelevant features fairly well, so aggressive selection matters more for linear/distance-based models and for wide-short data (§5).

## Filter methods — score features against the target, model-free

Fast, run once, no model needed. Used for a first cut on wide data.

| Method | Use for |
|---|---|
| Variance threshold | Drop near-constant features |
| Pearson / Spearman correlation | Numeric feature ↔ numeric target |
| ANOVA F-test | Numeric feature ↔ categorical target |
| Chi-squared | Categorical feature ↔ categorical target |
| **Mutual information** | Any type; captures **non-linear** dependence (correlation won't) |
| **Information Value / WOE** | Credit-risk standard for binary targets |
| Correlation matrix pruning / VIF | Drop redundancy *among features* |

- ✅ Cheap, scalable, no overfitting risk from the selection itself
- ❌ **Univariate** — ignores feature interactions, so it discards features that are only useful in combination, and keeps redundant correlated ones

## Wrapper methods — search subsets by training a model

| Method | How |
|---|---|
| Forward selection | Start empty, greedily add the feature that most improves CV score |
| Backward elimination | Start full, greedily remove the least useful |
| **RFE / RFECV** | Repeatedly fit, drop the weakest feature(s), recurse; RFECV picks the count by CV |
| Exhaustive search | All subsets — combinatorial, rarely feasible |
| Genetic / stochastic search | Heuristic search over subset space |

- ✅ Accounts for interactions and is tailored to the actual model
- ❌ Expensive (many model fits), and **prone to overfitting the selection to your validation set** — nest it inside cross-validation, or you're selecting on noise

## Embedded methods — selection happens during training

| Method | How |
|---|---|
| **Lasso / Elastic Net** | L1 penalty drives coefficients to exactly 0 |
| Tree/ensemble importance | Gain, split count, or permutation importance |
| **SHAP-based selection** | Rank by mean absolute SHAP value — more reliable than impurity importance |
| `reg_alpha` in XGBoost | L1 on leaf weights |
| Attention/gating in NNs | Learned feature weighting |

- ✅ One training run, interaction-aware, cheap relative to wrappers
- ❌ Model-specific (Lasso's picks aren't XGBoost's), and unstable under correlated features — L1 picks one of a correlated group arbitrarily

## Which to use

```
Very wide data → filter to a manageable set → embedded (L1 / SHAP) to rank
                                            → wrapper (RFECV) only if compute allows
```
Practical recipe: variance + correlation pruning → mutual information or model importance → validate that removal doesn't hurt CV. And add a **stability check**: run selection across CV folds and keep features chosen consistently — features selected in only one fold are usually noise.

**Caveat on importance**: impurity-based importance is biased toward high-cardinality and continuous features. Prefer **permutation importance** or **SHAP**. With correlated features, importance splits arbitrarily across the group — so a low score doesn't prove a feature is useless.

---

# 3. High-Cardinality Categorical Features

**The problem**: a feature with thousands-to-millions of levels (merchant ID, ZIP, product SKU, diagnosis code, user ID). One-hot encoding explodes dimensionality and creates extreme sparsity; label encoding invents a false ordinal ordering; rare levels have too few observations to estimate anything reliably; and unseen levels appear at inference.

## Encoding strategies

| Strategy | How | Notes |
|---|---|---|
| **One-hot** | Binary column per level | Only viable under ~10–50 levels. Explodes beyond that |
| **Frequency / count encoding** | Replace level with its occurrence count | Trivial, leak-free, surprisingly strong when frequency is informative |
| **Target / mean encoding** | Replace level with the mean target for that level | Powerful and compact — **but the top leakage risk**; see below |
| **WOE encoding** | `ln(%good / %bad)` per level | Credit-risk standard; monotonic with the target and pairs with IV |
| **Ordinal / label** | Integer per level | Fine for **tree** models (they can split anywhere), wrong for linear/distance models |
| **Hashing trick** | Hash level → one of `k` buckets | Fixed memory, handles unseen levels, no fitting needed; collisions are the cost |
| **Embeddings** | Learn a dense vector per level | Best for very high cardinality with lots of data (neural nets, or as features from a trained model) |
| **Native handling** | Let the library do it | **CatBoost** (ordered target statistics), **LightGBM** (native categorical splits) |
| **Binary / BaseN** | Encode the level index in base-2/N | Log-scale column count; poor interpretability |
| **Leave-one-out** | Target mean excluding the current row | Reduces (doesn't remove) leakage |

**Doing target encoding safely** — this is the part interviewers probe:
1. Compute encodings **out-of-fold** (K-fold: encode each fold using the other folds' statistics only).
2. **Smooth** toward the global mean so rare levels aren't trusted:
   ```
   encoding = (n_level · mean_level + m · global_mean) / (n_level + m)
   ```
   where `m` controls shrinkage strength.
3. Add small noise during training to reduce over-reliance.
4. Define an explicit fallback (global mean) for unseen levels at inference.

## Grouping / reducing levels

When you can consolidate classes, the options are:

**Frequency-based**
- **Top-N + "Other"**: keep the N most frequent levels, bucket the rest. The default first move — simple, robust, and usually captures most of the signal.
- **Minimum-count threshold**: any level with `< k` occurrences → `Other` (or `Rare`).
- Cumulative-coverage cut: keep levels covering the top 90–95% of rows.

**Target-based**
- **Supervised binning / decision-tree grouping**: fit a shallow tree on that single feature and use its leaves as groups — merges levels with similar target rates, and is exactly how credit-risk coarse classing is often automated.
- **Monotonic binning by event rate**: sort levels by target rate, merge adjacent levels until each bin has adequate volume and the rate is monotonic (WOE/IV workflow).
- Merge levels whose target rates aren't statistically distinguishable (chi-square / CHAID-style merging).

**Domain / hierarchy-based** — usually the best when available
- Use the natural hierarchy: ZIP → city → state → region; SKU → subcategory → category; ICD code → chapter; merchant → merchant category code (MCC).
- Build **multiple** features at different granularities and let the model pick (SKU-level target encoding *and* category-level).

**Similarity-based**
- Cluster levels by their feature profiles or embeddings.
- For text-like levels, group by string similarity or a shared prefix.

**Practical guidance**: prefer domain hierarchy → supervised binning → Top-N + Other. Always validate that grouping doesn't destroy signal (compare CV before/after), keep a `Rare`/`Unknown` bucket for inference-time surprises, and derive groupings from **training data only**.

---

# 4. Feature Engineering Patterns

**Numeric**: ratios and differences (often more informative than raw values — `debt/income` beats both), log/Box-Cox transforms for skew, binning (equal-width, quantile, or supervised), polynomial/interaction terms, clipping/winsorizing outliers, scaling (needed for linear/distance/NN models, not for trees).

**Datetime**: never feed a raw timestamp. Extract hour/day-of-week/month/quarter, is_weekend, is_holiday, **cyclical encoding** (`sin`/`cos` of hour or month, so 23:00 and 00:00 are adjacent), time since/until a reference event, and recency/tenure.

**Aggregations (the workhorse for transactional data)**: group-by statistics per entity — count, sum, mean, std, min/max, nunique; **rolling and expanding windows** (7/30/90-day); lag features; velocity and acceleration (count in last hour ÷ count in last 30 days); deviation from the entity's own history (`current / rolling_mean`, or a z-score). These are what actually power fraud and risk models.

**Categorical**: see §3.

**Text**: length/word counts, TF-IDF, embeddings, keyword flags.

**Domain/interaction**: cross-features (`state × product`), flags from business rules, distance/geo features, graph-derived features (shared device/address counts across accounts).

**The rules that keep it honest**: every aggregation must be **point-in-time correct** (computed only from data available before the decision moment); compute training and serving features with the same code path (a feature store exists for this reason); and prefer few strong features over many weak ones — each added feature costs variance and a maintenance obligation.

---

# 5. Tall-and-Narrow vs Short-and-Wide Data

The contrast from the question — **1M rows × 2 features** vs **500 rows × 200 features** — is a bias-variance question in disguise.

| | 1M × 2 (tall, narrow) | 500 × 200 (short, wide) |
|---|---|---|
| Regime | `n ≫ p` | `p ≫ n` (high-dimensional) |
| Binding constraint | **Bias** — too little signal in 2 features | **Variance** — too many ways to fit 500 points |
| Overfitting risk | Very low | Severe; can fit training data perfectly and generalize at chance |
| Right model | Flexible/non-parametric: GBM, deep trees, splines, NN | Strongly regularized and simple: Ridge/Lasso/Elastic Net, linear SVM, Naive Bayes |
| Regularization | Light | Heavy, and non-negotiable |
| Feature work | **Create** features (interactions, aggregations, transforms) — the ceiling is feature-limited | **Remove** features (filter → embedded selection), or project (PCA/PLS) |
| Validation | A single hold-out is fine and stable | Repeated / nested k-fold or LOOCV; a single split is far too noisy |
| Curse of dimensionality | Not a factor | Real — distances concentrate, so KNN/RBF-SVM degrade |
| Compute concern | Data volume: sampling, `hist` tree method, out-of-core, distributed | Trivial to train; the cost is search/validation cycles |
| Multiple-testing risk | Low | High — with 200 features some will look significant by pure chance |
| Statistical footing | Tight confidence intervals, small effects detectable | Wide intervals; findings are fragile and need replication |

**How to answer this well**: name the regime (`n ≫ p` vs `p ≫ n`), say which of bias or variance dominates, and let the model/regularization/validation choices follow from that. Then add the practical asymmetry: on the tall dataset your leverage is **feature engineering** (and you can afford a flexible model); on the wide dataset your leverage is **regularization and honest validation** (and a flexible model is a trap). Worth noting explicitly: XGBoost is a poor default on 500×200 — the greedy split search over 200 features on 500 rows finds spurious splits readily. See [`XGBoost_Trees_Deep_Dive.md`](XGBoost_Trees_Deep_Dive.md) §5.
