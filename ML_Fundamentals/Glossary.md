# Glossary — ML / DL / GenAI Terminology

Fast lookup for terms that come up in interviews. Alphabetical within sections. Where a term has a longer treatment elsewhere, the entry links to it rather than repeating it.

**Jump to:** [Statistics & Probability](#1-statistics-probability) · [Data & Features](#2-data-features) · [Modelling & Training](#3-modelling-training) · [Evaluation](#4-evaluation) · [Algorithms](#5-algorithms) · [Deep Learning](#6-deep-learning) · [GenAI & LLMs](#7-genai-llms) · [MLOps](#8-mlops-production) · [Confusables](#9-commonly-confused-pairs)

---

# 1. Statistics & Probability

**Bayes' theorem** — `P(A|B) = P(B|A)·P(A) / P(B)`. Updates a prior belief with evidence to get a posterior.

**Bias (statistical)** — systematic error; the gap between a model's expected prediction and the truth. Distinct from *bias* as unfairness, and from the *bias term* (intercept) in a layer.

**Central Limit Theorem** — the distribution of sample **means** approaches normal as sample size grows, regardless of the population's own shape. Why normal-based inference works on non-normal data.

**Confidence interval** — a range that would contain the true parameter in X% of repeated samples. *Not* "95% probability the parameter is in this interval" — that's the Bayesian credible interval.

**Correlation** — strength of linear co-movement. Pearson (linear, on values), Spearman (monotonic, on ranks), Kendall's tau (concordance). See [`Plots_Visual_Diagnostics.md` §15](Plots_Visual_Diagnostics.md#15-correlation-traps).

**Covariate shift** — input distribution `P(X)` changes while `P(y|X)` stays fixed. One of several distribution shifts; see *drift*.

**Cumulative Distribution Function (CDF)** — `P(X ≤ x)`. The integral of the PDF.

**Degrees of freedom** — number of independent values free to vary after constraints.

**Effect size** — the *magnitude* of a difference (Cohen's d, lift, odds ratio). Statistical significance without effect size is nearly useless — with enough data, trivial effects become significant.

**Expected value** — probability-weighted mean, `E[X] = Σ x·P(x)`.

**Heteroscedasticity** — non-constant variance of errors across the range of fitted values. Leaves coefficients unbiased but invalidates standard errors, p-values, and confidence intervals.

**Hypothesis testing** — H₀ (null) vs H₁ (alternative). **Type I error** = false positive, rejecting a true H₀ (rate = α). **Type II error** = false negative, failing to reject a false H₀ (rate = β). **Power** = 1 − β.

**IQR** — interquartile range, Q3 − Q1. Basis of the 1.5×IQR outlier rule.

**Kurtosis** — tailedness. Leptokurtic = heavy tails (more extremes than normal); platykurtic = light tails.

**Law of Large Numbers** — sample mean converges to the population mean as n grows.

**Maximum Likelihood Estimation (MLE)** — choose parameters maximising the probability of the observed data. **MAP** adds a prior — and L2 regularization is exactly MAP with a Gaussian prior.

**Multicollinearity** — predictors highly correlated with each other. Inflates coefficient variance and makes individual coefficients uninterpretable, without necessarily hurting predictions. Detect with **VIF**.

**p-value** — probability of seeing data at least this extreme if H₀ were true. **Not** the probability that H₀ is true, and not a measure of effect size.

**PDF / PMF** — probability density (continuous) / mass (discrete) function.

**Percentile / quantile** — value below which a given proportion falls.

**Prior / likelihood / posterior** — belief before evidence / probability of the evidence under a hypothesis / belief after evidence.

**p-hacking / multiple comparisons** — testing many hypotheses inflates the chance of a spurious "significant" result. Correct with Bonferroni or Benjamini-Hochberg (FDR).

**Skewness** — asymmetry. Right/positive skew: tail to the right, mean > median. See [`Plots_Visual_Diagnostics.md` §1](Plots_Visual_Diagnostics.md#1-skewness-right-skewed-vs-symmetric-vs-left-skewed).

**Standard deviation vs standard error** — SD measures spread of the *data*; SE measures spread of a *statistic* (`SE = SD/√n`) and shrinks with sample size.

**Simpson's paradox** — a trend present in every subgroup reverses when the groups are pooled. Symptom of a confounder correlated with both variables.

**Statistical power** — probability of detecting a real effect. Driven by effect size, sample size, and α.

**Variance** — spread around the mean, `E[(X − μ)²]`. In the bias-variance sense: sensitivity of the fitted model to the particular training sample.

**Z-score** — `(x − μ)/σ`, i.e. how many standard deviations from the mean.

---

# 2. Data & Features

**Cardinality** — number of distinct values in a categorical feature. High cardinality needs special encoding; see [`Feature_Engineering_Selection.md` §3](Feature_Engineering_Selection.md#3-high-cardinality-categorical-features).

**Curse of dimensionality** — as dimensions grow, data becomes sparse and all pairwise distances converge, which breaks distance-based methods (KNN, RBF kernels, K-Means).

**Data leakage** — information available at training that won't exist at prediction time, or that encodes the target. See [`Feature_Engineering_Selection.md` §1](Feature_Engineering_Selection.md#1-data-leakage).

**Drift** — **data/covariate drift**: `P(X)` changes. **Concept drift**: `P(y|X)` changes — the relationship itself moved. **Label drift**: `P(y)` changes. Concept drift is the dangerous one; retraining on new features won't fix a changed relationship.

**Embedding** — a learned dense vector representation where geometric proximity encodes semantic similarity.

**Feature store** — infrastructure serving the *same* feature definitions to training (offline) and serving (online), enforcing point-in-time correctness. Exists to prevent train/serve skew.

**Imputation** — filling missing values (mean/median/mode, KNN, iterative/MICE, or a learned model). Note that XGBoost learns a default direction instead; see [`XGBoost_Trees_Deep_Dive.md` §10](XGBoost_Trees_Deep_Dive.md#10-how-does-xgboost-handle-missing-values).

**MCAR / MAR / MNAR** — missing Completely At Random (missingness unrelated to anything) / At Random (explained by observed variables) / Not At Random (related to the unobserved value itself, e.g. high earners hiding income). MNAR is the one that biases everything and can't be imputed away.

**Normalization vs standardization** — min-max to [0,1] vs zero-mean-unit-variance (`z = (x−μ)/σ`). Tree models need neither; linear, distance-based, and neural models need one.

**One-hot / label / target / ordinal encoding** — see the encoding table in [`Feature_Engineering_Selection.md` §3](Feature_Engineering_Selection.md#3-high-cardinality-categorical-features).

**Point-in-time correctness** — every feature value reflects only what was knowable at the decision timestamp. The discipline that prevents temporal leakage.

**SMOTE** — Synthetic Minority Over-sampling: generates minority points by interpolating between neighbours. Must be applied *inside* the CV fold.

**Stratified sampling** — sampling that preserves class proportions in each split. Standard for imbalanced classification.

**Train/serve skew** — features computed differently in training vs production. The most common cause of "great offline, bad online".

**WOE / IV** — Weight of Evidence `ln(%good/%bad)` per bin, and Information Value, its aggregate predictive-power score. Credit-risk standards.

---

# 3. Modelling & Training

**Batch / mini-batch / epoch / iteration** — a batch is the samples per gradient update; an epoch is one full pass over the data; an iteration is one update step.

**Boosting vs bagging** — sequential error-correction (reduces bias) vs parallel averaging (reduces variance). See [`XGBoost_Trees_Deep_Dive.md` §6](XGBoost_Trees_Deep_Dive.md#6-bagging-vs-boosting).

**Cross-validation** — k-fold; **stratified** k-fold preserves class balance; **group** k-fold keeps an entity in one fold; **TimeSeriesSplit** respects time order; **nested** CV separates hyperparameter selection from performance estimation.

**Early stopping** — halt when validation loss stops improving. Both a regularizer and the standard way to choose the number of boosting rounds.

**Ensemble** — combining models. **Bagging** (parallel), **boosting** (sequential), **stacking** (a meta-model learns to combine base-model predictions), **voting/blending** (simple aggregate).

**Gradient descent** — `θ ← θ − η·∇L`. Batch / stochastic / mini-batch by how much data each step uses.

**Hyperparameter vs parameter** — set before training (learning rate, depth) vs learned during it (weights, split thresholds).

**Hyperparameter search** — grid (exhaustive), random (usually better per unit compute), Bayesian/TPE (Optuna, Hyperopt — models the objective), Hyperband/ASHA (early-kills bad trials).

**Learning rate** — step size. Usually the highest-impact single hyperparameter.

**Loss vs metric** — loss is what the optimizer minimizes (must be differentiable for gradient methods); metric is what you evaluate and report. They often differ: train on log-loss, report PR-AUC.

**Objective function** — loss + regularization terms; what training actually minimizes.

**Overfitting / underfitting** — fitting noise (low train error, high validation) vs too simple to fit signal (both high). See [`Plots_Visual_Diagnostics.md` §6](Plots_Visual_Diagnostics.md#6-learning-curves-diagnosing-which-problem-you-have).

**Regularization** — a penalty on complexity that trades training fit for generalization. **L1/Lasso** can zero coefficients (feature selection); **L2/Ridge** shrinks smoothly; **Elastic Net** blends them. See [`XGBoost_Trees_Deep_Dive.md` §11-12](XGBoost_Trees_Deep_Dive.md#11-what-is-regularization-actually-doing).

**Shrinkage** — scaling each boosting stage's contribution by the learning rate; a regularizer.

**Class weighting** — reweighting the loss by class to counter imbalance (`class_weight='balanced'`, `scale_pos_weight`). Distorts predicted probabilities, so recalibrate if you need them.

---

# 4. Evaluation

**Accuracy** — (TP+TN)/all. Misleading under imbalance.

**AUC / ROC-AUC** — area under the ROC curve; equals `P(random positive ranked above random negative)`. Threshold- and calibration-independent, and invariant to class balance.

**Calibration** — do predicted probabilities match observed frequencies. Fix with Platt scaling or isotonic regression on held-out data. See [`Plots_Visual_Diagnostics.md` §11](Plots_Visual_Diagnostics.md#11-calibration-curve-reliability-diagram).

**Confusion matrix** — TP/FP/FN/TN; every classification metric derives from it. See [`Plots_Visual_Diagnostics.md` §8](Plots_Visual_Diagnostics.md#8-the-confusion-matrix-and-everything-derived-from-it).

**F1 / F-beta** — harmonic mean of precision and recall; β > 1 favours recall.

**Gini coefficient** — `2·AUC − 1`. Common in credit risk. Unrelated to Gini *impurity* in trees.

**KS statistic** — max vertical gap between the score CDFs of the two classes.

**Lift / gains** — how much better than random the top x% of a ranking performs. See [`Plots_Visual_Diagnostics.md` §12](Plots_Visual_Diagnostics.md#12-cumulative-gains-and-lift).

**MAE / MSE / RMSE / MAPE** — mean absolute error (robust, in target units) / mean squared (penalises large errors, differentiable everywhere) / root MSE (target units) / mean absolute *percentage* error (scale-free but explodes near zero and asymmetrically penalises over- vs under-prediction).

**PR-AUC / average precision** — area under the precision-recall curve. Preferred over ROC-AUC under heavy imbalance; baseline equals prevalence.

**Precision / recall** — TP/(TP+FP) and TP/(TP+FN). Precision depends on prevalence; recall does not.

**R² / adjusted R²** — variance explained; R² never decreases when you add features, so use adjusted R² for comparison.

**Specificity / sensitivity** — TN/(TN+FP) and TP/(TP+FN) (sensitivity = recall).

**SHAP / LIME** — per-prediction attribution. SHAP is grounded in Shapley values and is additive/consistent; LIME fits a local surrogate model. SHAP is the practical default for tabular explainability.

**Permutation importance** — drop in performance when a feature is randomly shuffled. More trustworthy than impurity-based importance, which is biased toward high-cardinality features.

---

# 5. Algorithms

Full reference cards in [`ML_Algorithms_Cheatsheet.md`](ML_Algorithms_Cheatsheet.md).

**AdaBoost** — boosting by reweighting misclassified samples. Outlier-sensitive.

**CatBoost** — gradient boosting with leakage-safe ordered target encoding for categoricals and oblivious trees. See [`CatBoost_Deep_Dive.md`](CatBoost_Deep_Dive.md).

**CTR (CatBoost)** — "counter"/target statistic: the smoothed mean target for a category, computed only from rows preceding the current one in a random permutation. Nothing to do with click-through rate as a metric, despite the name's origin.

**Oblivious (symmetric) tree** — a tree where every node at a given depth shares one split rule, making it a `2^d`-leaf decision table. Gives branchless, very fast inference and acts as regularization. CatBoost's default.

**Ordered boosting** — computing each row's gradient from a model fitted only on rows preceding it, removing the prediction shift present in plain GBM.

**Prediction shift** — the bias from computing a row's gradient with a model that was trained on that same row; training-set gradients end up systematically unlike unseen-data gradients.

**DBSCAN** — density-based clustering; needs `eps`/`min_samples` instead of k, finds arbitrary shapes, labels noise explicitly.

**Decision tree** — recursive axis-aligned splits. Classification splits on Gini/entropy; **regression splits on variance reduction (MSE)**, not Gini.

**Gini impurity vs entropy** — `1 − Σpᵢ²` vs `−Σpᵢlog₂pᵢ`. Both measure class impurity; Gini is cheaper, results are near-identical.

**GMM** — Gaussian Mixture Model; soft/probabilistic clustering fit by EM. K-Means is its hard-assignment, spherical special case.

**HDBSCAN** — hierarchical DBSCAN; removes the single global density threshold.

**Isolation Forest** — anomaly detection by how easily a point is isolated by random splits.

**K-Means** — centroid clustering minimising inertia. Needs k, feature scaling, and roughly spherical clusters.

**KNN** — lazy, instance-based; prediction from the k nearest neighbours. Scale-sensitive, degrades in high dimensions.

**LightGBM** — gradient boosting with leaf-wise growth, histogram binning, native categoricals. Fast on large data; cap `num_leaves` to avoid overfitting.

**Naive Bayes** — generative classifier assuming conditional independence of features. Strong on high-dimensional sparse text.

**PCA** — orthogonal projection maximising variance (unsupervised). **LDA** maximises class separability (supervised). Both need standardized features.

**Random Forest** — bagged trees plus per-split feature subsampling. Low-tuning, noise-robust, gives OOB error for free.

**SVM** — maximum-margin classifier; kernels give non-linear boundaries. Scale-sensitive; training scales poorly with n.

**t-SNE / UMAP** — non-linear embeddings for visualization. Preserve local neighbourhoods; **inter-cluster distances are not meaningful**. UMAP is faster and can transform new points.

**XGBoost** — regularized second-order gradient boosting; the tabular default. See [`XGBoost_Trees_Deep_Dive.md`](XGBoost_Trees_Deep_Dive.md).

---

# 6. Deep Learning

Full treatment in [`DL_Concepts_Cheatsheet.md`](DL_Concepts_Cheatsheet.md).

**Activation function** — the non-linearity (ReLU, GELU, sigmoid, tanh, softmax). Without one, stacked linear layers collapse to a single linear map.

**Attention** — `softmax(QKᵀ/√d_k)·V`. See [`Transformers_Cheatsheet.md`](Transformers_Cheatsheet.md).

**Backpropagation** — reverse-mode automatic differentiation; computes all gradients in one backward pass. The *optimizer* then applies them.

**Batch norm vs layer norm** — normalize across the batch (per feature) vs across features (per sample). Batch norm behaves differently at train vs inference and degrades with small batches; layer norm doesn't, which is why transformers use it.

**Dead ReLU** — a unit whose pre-activation is always negative outputs 0 forever with zero gradient. Leaky ReLU/GELU mitigate.

**Dropout** — randomly zero activations during training; an implicit ensemble. Disabled at inference.

**Embedding layer** — learned lookup table mapping discrete tokens to dense vectors.

**Exploding / vanishing gradients** — gradient products growing or shrinking through depth. Fixed by residual connections, normalization, ReLU-family activations, gradient clipping, and LSTM gating.

**Gradient clipping** — cap gradient norm to prevent exploding updates.

**LSTM / GRU** — gated recurrent units. LSTM's additive cell-state path is what lets gradients survive long sequences.

**Optimizer** — SGD, momentum, RMSProp, Adam, **AdamW** (decoupled weight decay — the transformer standard).

**Residual / skip connection** — `output = F(x) + x`. Gives gradients a direct path and makes the identity easy to learn.

**Transfer learning / fine-tuning** — reuse a pretrained model, adapt to a new task. Use a much lower learning rate than training from scratch.

**Weight initialization** — Xavier/Glorot for tanh/sigmoid, He/Kaiming for ReLU. All-zeros never breaks symmetry.

---

# 7. GenAI & LLMs

Full treatment in [`GenAI_Cheatsheet.md`](GenAI_Cheatsheet.md).

**Agent** — LLM + tools + a loop + memory. Main risk is compounding per-step error.

**BPE / WordPiece / SentencePiece** — subword tokenization algorithms.

**Context window** — max tokens the model can attend to at once, prompt plus generation.

**Chain-of-Thought (CoT)** — eliciting intermediate reasoning steps.

**Distillation** — train a small student on a large teacher's outputs.

**DPO / GRPO / RLHF** — preference-alignment methods. RLHF/PPO needs a reward model *and* a critic; DPO removes the reward model; GRPO removes the critic by comparing a group of sampled outputs.

**Flash Attention** — IO-aware exact attention kernel. Changes speed and memory, not results.

**Frequency vs presence penalty** — frequency scales with how many times a token already appeared (kills repetition loops); presence is a flat one-time penalty on any already-present token (encourages new topics).

**Grounding** — conditioning generation on retrieved or verified source data rather than parametric memory.

**Hallucination** — fluent output unsupported by any source.

**HyDE** — generate a hypothetical answer, embed *that*, retrieve with it. Closes the short-query/long-document gap.

**KV cache** — cached key/value projections of prior tokens; turns per-token generation cost from O(n²) into O(n). Usually the dominant memory cost in serving.

**LoRA / QLoRA** — parameter-efficient fine-tuning via low-rank adapters (`W + BA`); QLoRA adds a 4-bit quantized frozen base.

**LLM-as-judge** — using a model to score outputs. Biases: position, verbosity, self-preference. Calibrate against human labels.

**Lost in the middle** — models attend reliably to the start and end of long contexts and degrade in the middle. Order retrieved chunks accordingly.

**Perplexity** — exponentiated average negative log-likelihood; a language-modelling quality measure, near-useless for judging instruction-following.

**Prompt injection** — untrusted input (user text, retrieved docs, tool output) hijacking instructions. Mitigate with delimiters, least-privilege tools, and output validation — the real boundary is the permission model.

**Quantization** — lower-precision weights (int8/int4) for smaller memory and higher throughput.

**RAG** — Retrieval-Augmented Generation. Adds *knowledge*; fine-tuning adds *behaviour*. ~20 named variants in [`GenAI_Cheatsheet.md` §6](GenAI_Cheatsheet.md#6-rag-retrieval-augmented-generation).

**RAGAS quadrant** — context precision/recall (blame retrieval) vs faithfulness/answer relevance (blame generation).

**Re-ranker (cross-encoder)** — scores query and document *together* after cheap ANN retrieval. Highest-ROI upgrade over naive RAG.

**RoPE** — rotary positional embedding; encodes *relative* position by rotating Q/K. Modern LLM standard.

**SLM** — Small Language Model (~1-15B). Often the right answer for narrow, high-volume tasks.

**Temperature / top_p / top_k** — sampling controls. Use temperature 0 for extraction and structured output.

**Tokenization** — text → integer IDs. Roughly 4 characters ≈ 1 token in English.

**Vector database** — ANN index over embeddings (HNSW, IVF, IVF-PQ). FAISS, Chroma, Qdrant, Weaviate, Milvus, pgvector, Pinecone.

---

# 8. MLOps & Production

**A/B test** — randomized online comparison. The only reliable way to attribute a business metric change to a model change.

**Canary / shadow deployment** — route a small traffic slice to the new model (canary), or run it silently alongside without serving its output (shadow).

**CI/CD for ML** — adds data validation, an offline eval gate against the *incumbent* model, and staged rollout to normal software CI/CD. See [`../Example_Company/12_MLOps_CICD_LLM_Hosting.md`](../Example_Company/12_MLOps_CICD_LLM_Hosting.md).

**Champion / challenger** — the production model vs a candidate evaluated against it.

**Feature parity** — training and serving compute features identically. The classic production failure when they don't.

**Model registry / versioning** — tracked artifacts, metrics, and lineage so a rollback is a config change rather than a retrain.

**Monitoring** — feature drift (PSI, KL divergence), prediction drift, delayed-label performance, plus latency/throughput/error rates. Each alert should have a defined action.

**PSI (Population Stability Index)** — drift measure between two distributions; rough convention: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant shift.

**Rollback** — reverting to a previous model version. Requires versioned artifacts and traffic routing.

**Shadow mode** — new model scores real traffic without acting on it, so you can compare before committing.

**SLA / SLO** — the latency or availability commitment your serving path must meet; drives whether batch or real-time inference is viable.

---

# 9. Commonly Confused Pairs

| Pair | The distinction |
|---|---|
| **Accuracy vs precision** | Closeness to truth (bias) vs closeness to each other (variance) — *or* the classification metric TP/(TP+FP). Ask which sense is meant |
| **Bias (statistical) vs bias (fairness) vs bias (intercept)** | Systematic error / unfair disparate impact / the additive constant in a layer |
| **Correlation vs causation** | Co-movement vs intervention effect. Confounders and Simpson's paradox are why |
| **Gini impurity vs Gini coefficient** | Tree split criterion vs `2·AUC − 1` in credit risk. Unrelated |
| **Loss vs metric** | Optimized (differentiable) vs reported (business-meaningful) |
| **Normalization vs standardization** | Min-max to [0,1] vs zero-mean-unit-variance |
| **Parameter vs hyperparameter** | Learned during training vs set before it |
| **PCA vs LDA** | Max variance (unsupervised) vs max class separation (supervised) |
| **Precision vs recall** | Of what I flagged, how much was real vs of what was real, how much I flagged |
| **Recall vs specificity** | Positives caught vs negatives cleared — *different denominators*, not complements |
| **ROC-AUC vs PR-AUC** | Balance-invariant ranking vs imbalance-sensitive positive-class quality |
| **Sensitivity vs specificity** | Sensitivity = recall = TPR; specificity = TNR |
| **SMOTE vs class weighting** | Synthesize minority rows vs reweight the loss. Both distort probabilities |
| **Standard deviation vs standard error** | Spread of data vs spread of a statistic (`SD/√n`) |
| **t-SNE vs PCA** | Non-linear local-structure visualization vs linear global-variance projection |
| **Type I vs Type II error** | False positive (α) vs false negative (β) |
| **Validation vs test set** | Used to *choose* (models, hyperparameters, thresholds) vs used *once* to report |
| **RAG vs fine-tuning** | Adds knowledge vs adds behaviour |
| **Data drift vs concept drift** | `P(X)` moved vs `P(y|X)` moved — the second is the dangerous one |
| **Batch vs epoch vs iteration** | Samples per update / full pass over data / one update step |
| **Bagging vs boosting** | Parallel averaging (cuts variance) vs sequential correction (cuts bias) |
