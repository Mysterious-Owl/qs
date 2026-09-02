# Question Bank

Real questions from two actual DS/ML interview rounds, kept in the order asked, plus a
topic checklist for tracking coverage.

Heavily XGBoost / classification / risk-modelling flavoured, with a GenAI thread. Each
entry links to its deep-dive answer where one exists, and is answered in place where the
question doesn't belong in a topic doc (project-specific, judgement, or stakeholder
questions).

Unlike the [`../Example_Company/`](../Example_Company/) track, these aren't attributed to a
specific company, so they carry no confidence tags — they're recorded as asked.

**Contents** — [the questions](#round-1) · [patterns across both rounds](#patterns-across-both-rounds) · [topic checklist](#topic-checklist) · [gaps to fill yourself](#notable-gaps-to-fill-yourself)

---

**Answer-location key**
- 🌲 [`XGBoost_Trees_Deep_Dive.md`](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md)
- 🔧 [`Feature_Engineering_Selection.md`](../ML_Fundamentals/Feature_Engineering_Selection.md)
- 🤖 [`GenAI_Cheatsheet.md`](../ML_Fundamentals/GenAI_Cheatsheet.md)
- 📊 [`ML_Algorithms_Cheatsheet.md`](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md)
- 👤 **Your own projects** — no doc for these; they need your specifics (numbers, decisions, what didn't work)
- ✅ Answered inline below

---

# Round 1

### 1. Introduction 👤
Not a warm-up — it sets the frame for everything after. Have a 60–90 second version: current role and scope → the one system you most want them to ask about → why this role. Land on a hook that invites the question you're best prepared for.

### 2. Have you worked on classification problems? 👤
Lead with the fraud/claims risk-scoring engine (supervised classifiers + DBSCAN + LLM entity analysis) — it's a classification system with real business impact and it sets up Q6, Q8 and Q9 naturally.

### 3. Working principle of XGBoost (boosting technique) 🌲
→ [§1](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#1-working-principle-of-xgboost-boosting). Sequential trees fitted to pseudo-residuals = gradient descent in function space; XGBoost adds second-order (gradient + Hessian) split gain and explicit `γT + ½λ‖w‖²` complexity regularization.

### 4. Major hyperparameters of XGBoost (regression and classification) 🌲
→ [§2](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#2-major-hyperparameters). Grouped by structure / boosting / randomization / regularization / task. Only the objective, eval metric and imbalance handling differ between regression and classification — `scale_pos_weight` is classification-only.

### 5. Difference between `predict` and `predict_proba` 🌲
→ [§3](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#3-predict-vs-predict_proba). `predict` applies a hard 0.5 cutoff; `predict_proba` returns the probabilities (post-sigmoid over the summed log-odds margin) and doesn't exist for regression.

### 6. How do you deal with imbalanced data in modelling ✅
Four levers, and the mature answer is that you usually combine them rather than reaching only for SMOTE:

**Metric first** — before touching the data, stop using accuracy. Use **PR-AUC / average precision** (more informative than ROC-AUC under heavy imbalance, since ROC-AUC is dominated by the abundant negatives), recall at a fixed precision, or an explicit expected-cost metric.

**Algorithm level** (preferred starting point — no data distortion)
- `scale_pos_weight ≈ neg/pos` in XGBoost, or `class_weight='balanced'` in sklearn.
- Focal loss — down-weights easy examples so gradient goes to the hard minority cases.

**Data level**
- Random **under**sampling of the majority — fast, works well when you have millions of rows and can afford to throw some away.
- **SMOTE** / ADASYN — synthesize minority points by interpolation. Caveats worth naming: apply **inside** the CV fold (never before splitting, or you leak), it degrades in high dimensions, and it can manufacture points in regions that don't actually contain the minority class.
- Combined (SMOTE + Tomek/ENN cleaning).

**Decision level**
- Tune the **threshold** on validation instead of accepting 0.5 → [§4](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#4-where-do-you-set-the-classification-threshold-in-xgboost).
- Calibrate probabilities (Platt/isotonic) if downstream logic consumes the probability, since reweighting and resampling both distort it.

**Reframe when it's extreme** — below ~0.1% positives, treat it as **anomaly detection** (isolation forest, autoencoder reconstruction error) or a rules+model hybrid rather than pure supervised classification.

Also: stratify your splits, and if the imbalance comes from label *scarcity* rather than genuine rarity, the highest-leverage fix is often better labelling, not clever resampling.

### 7. 1M rows × 2 features vs 500 rows × 200 features — how do these behave differently? 🔧
→ [§5](../ML_Fundamentals/Feature_Engineering_Selection.md#5-tall-and-narrow-vs-short-and-wide-data). Name the regime (`n ≫ p` vs `p ≫ n`), say which of bias/variance dominates, and let model choice, regularization and validation strategy follow.

### 8. Explain precision and recall — for fraud detection, which would you suggest? ✅

```
Precision = TP / (TP + FP)   "of everything I flagged, how much was really fraud"
Recall    = TP / (TP + FN)   "of all real fraud, how much did I catch"
```

**The answer interviewers want is not "recall" — it's "it depends on what happens after the flag."** That distinction is the whole question:

- If a flag **auto-blocks** a transaction, a false positive is a blocked legitimate customer — direct revenue loss and churn. **Precision matters more.**
- If a flag **routes to a human review queue**, a false positive costs only analyst time while a false negative is realized fraud loss. **Recall matters more**, bounded by review capacity.

So the real design is **tiered**, and that's the strong answer: high-confidence scores auto-block (tuned for precision), a mid band goes to human review (tuned for recall within the queue's throughput), low scores pass. Then you're not choosing one metric — you're choosing an operating point per tier.

**Supporting points to have ready:**
- Quantify it: minimize `FN·(avg fraud loss) + FP·(cost of a blocked good customer)`. This turns a metric debate into a business decision.
- Report **PR-AUC** rather than ROC-AUC given the imbalance, plus **recall at a fixed false-positive rate** — the number ops actually care about.
- Use **F-beta** if you must have one number: β > 1 weights recall higher.
- Mention **precision@k** when review capacity is fixed ("we can review 500 alerts/day") — that's the honest metric for a capacity-constrained queue, and it's exactly the framing Q9 needs.

### 9. With ~2M loan records, how would you find the top 1% highest risk in loan recovery? What features? ✅

**Reframe it as ranking, not classification.** "Top 1%" is a capacity constraint, so the objective is to order accounts by expected loss and cut at the 1% mark — optimize **precision@1%** / lift in the top decile, not global accuracy. 1% of 2M = 20,000 accounts, so also ask what the recovery team can actually action; the useful cut may be smaller than the requested one.

**Target definition first** (the most important design decision, and the one candidates skip): what is "high risk in recovery"? Probability of no recovery within N days? Expected **loss given default** (amount-weighted)? Roll-rate to a worse delinquency bucket? Each implies a different model. Note that ranking by *probability* alone is wrong if exposures differ wildly — rank by **expected loss = P(non-recovery) × outstanding exposure**, so a 40%-risk ₹50L account outranks a 90%-risk ₹20k one.

**Approach**
1. Establish a **point-in-time** snapshot: features as of the decision date, outcome observed in a forward window. Time-based train/validation/test split — no random splits.
2. Baseline: gradient-boosted trees (XGBoost/LightGBM) for the ranking score. Add monotonic constraints on features where the direction is known (higher DPD ⇒ higher risk) — this matters for regulatory defensibility.
3. Calibrate probabilities, then compute expected loss, then rank.
4. Evaluate with precision@1%, lift/gains curves, KS statistic, and a **segment-level** check (does the top 1% collapse onto one geography or product? that's a bias/leakage smell).
5. Add reason codes via SHAP so the recovery team knows *why* an account surfaced — an unexplained list doesn't get used.

**Features** (grouped, and worth naming the aggregation logic, not just the raw fields):
- **Delinquency dynamics** — current DPD, max DPD over 3/6/12m, number of times delinquent, **roll-rate direction** (improving vs deteriorating — the trend is more predictive than the level), time in current bucket.
- **Payment behaviour** — payment-to-EMI ratio, count of partial/missed/bounced payments, cheque/mandate failures, days between due date and payment, recency of last payment, whether payments are shrinking over time.
- **Exposure & product** — outstanding principal, EMI amount, original vs remaining tenure, LTV, interest rate, secured vs unsecured, collateral value and liquidity.
- **Affordability** — DTI / FOIR, income (and whether it's verified), income stability, EMI-to-income drift since origination.
- **Bureau** — current bureau score and **change** since origination, total external exposure, new credit enquiries (a spike is a strong distress signal), delinquencies on *other* lenders' accounts, write-offs/settlements elsewhere.
- **Origination / vintage** — months on book, sourcing channel, whether it was an exception/deviation approval, origination-era policy regime.
- **Contactability & servicing** — valid phone/email, contact-attempt success rate, promise-to-pay kept vs broken, dispute/complaint flags, prior restructuring or moratorium.
- **Borrower profile** — employment type (salaried/self-employed), industry, tenure, geography, urban/rural.
- **Macro / external** — regional unemployment or delinquency trend, sector stress for self-employed borrowers, seasonality.
- **Network features** — shared address/phone/employer across accounts (catches organized default rings), which is where a graph layer helps.

**Watch for**: leakage from post-decision fields (recovery-agent notes, legal-action flags, settlement records — see 🔧 §1), survivorship bias (closed accounts excluded), and the fact that any past collections *intervention* affected the outcome you're now modelling — so the label is partly a record of your own prior policy, not pure borrower behaviour.

### 10. What evaluation metrics were used in your project? 👤
Answer from your own work. Be ready to justify *why* that metric — and if you reported accuracy anywhere on an imbalanced problem, pre-empt the pushback: know either the justification for it or the precision/recall numbers alongside.

### 11. A feature has very high importance and the client wants to drop it. How do you handle that? ✅

**Do not lead with "but the model needs it."** This is a stakeholder question wearing a technical costume — the first move is to find out *why* they want it gone, because each reason has a different correct response:

| Their real reason | Right response |
|---|---|
| **Legal / regulatory** (protected attribute, or a proxy for one) | They're right — drop it. Also audit for proxies, and test disparate impact. Non-negotiable, and agreeing fast builds credibility |
| **Privacy / consent** — data isn't licensed for this use | Drop it. Look for a compliant substitute |
| **Availability** — won't exist at scoring time, or is being deprecated | They've caught a **leakage/production-parity bug**. Thank them; this is a real save |
| **Data quality distrust** — "that field is unreliable" | Investigate. If it's unreliable *now* but was clean historically, the model will silently degrade — a strong argument to drop |
| **Interpretability / business intuition** — "this shouldn't matter" | The most interesting case. High importance on an implausible feature is often leakage or a confound. Investigate before defending |
| **Fairness optics** — defensible but uncomfortable | Discuss trade-offs and alternatives openly |

**Then quantify the cost, don't assert it.** Retrain without the feature and present the actual delta in business terms: "removing it costs 4 points of recall at fixed precision, which is roughly ₹X in undetected fraud per month — here's the number, it's your call." That converts an argument into a decision with a price tag, which is what a senior candidate does.

**Then offer middle paths**: a proxy or coarsened version (bucketed instead of raw), a version at lower granularity, dropping it only from the auto-action tier while keeping it for human-review ranking, or a monotonic/constrained form that behaves more defensibly.

**Close the loop**: document the decision and the measured cost, ship the model both ways if feasible, and monitor. And genuinely: if the reason is legal, fairness, or availability, the client is right and the model is wrong — recognizing that is the point of the question.

### 12. What is data leakage? 🔧
→ [§1](../ML_Fundamentals/Feature_Engineering_Selection.md#1-data-leakage). Target leakage, train-test contamination, temporal, group, duplicate, target-encoding and metadata leakage, plus the prevention checklist.

### 13. Have you worked on any GenAI projects? 👤 🤖
Strong ground for you — the dispute-resolution RAG/agent platform, LLM email triage, and GRPO-post-trained explanations. Pick one and go deep rather than listing several; see 🤖 for the underlying concepts.

### 14. Which XGBoost parameter sets the classification probability threshold? 🌲
→ [§4](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#4-where-do-you-set-the-classification-threshold-in-xgboost). **Trick question — there isn't one.** Thresholding is post-hoc on `predict_proba`. `scale_pos_weight` reweights training gradients and `base_score` sets the initial prediction; neither is a threshold. Being able to say that cleanly *and* explain how you'd actually pick the threshold is the win.

---

# Round 2

### 1. How big was the data you handled? 👤
Give concrete numbers (rows, features, storage, refresh cadence) and what the scale actually forced you to change — sampling, PySpark/Databricks, out-of-core training, incremental retraining. Scale claims invite "and what broke at that size?"

### 2. In which cases is XGBoost prone to fail, where another model does better? 🌲
→ [§5](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#5-when-does-xgboost-fail-and-what-beats-it). Wide-short data, genuinely linear relationships, **extrapolation** (trees are piecewise-constant, so predictions go flat outside the training range), raw images/text/audio, strong temporal structure, very high-cardinality categoricals, heavy label noise, and hard interpretability requirements.

### 3. How do you tune XGBoost systematically? 🌲
→ [§7](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#7-systematic-xgboost-tuning). Staged: baseline with early stopping → tree capacity (`max_depth` + `min_child_weight` together) → `gamma` → subsampling → L1/L2 → imbalance → finally lower the learning rate and re-run.

### 4. What happens if you reduce the learning rate further? 🌲
→ [§8](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#8-what-happens-if-you-keep-lowering-the-learning-rate). Needs proportionally more trees; generalization improves slightly then plateaus; with a *fixed* tree budget it underfits. `learning_rate` and `n_estimators` are coupled and are never tuned independently.

### 5. XGBoost is overfitting — what do you do at the modelling level? Anything with the learning rate? 🌲
→ [§9](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#9-xgboost-is-overfitting-what-do-you-do). Reduce depth, raise `min_child_weight`/`gamma`, raise `reg_lambda`, lower `subsample`/`colsample`, and above all **early stopping**. On the learning-rate part specifically: lowering LR alone with a fixed tree count mostly *underfits* — it regularizes only in combination with early stopping or a raised tree budget. That caveat is the point of the question.

### 6. Difference between bagging and boosting 🌲
→ [§6](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#6-bagging-vs-boosting). Parallel-independent vs sequential-dependent; variance reduction vs bias reduction; strong vs weak base learners; noise-robust vs noise-sensitive.

### 7. How does XGBoost handle missing values? 🌲
→ [§10](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#10-how-does-xgboost-handle-missing-values). Sparsity-aware split finding: it tries missing-rows-left vs missing-rows-right, keeps the higher-gain option as that node's **learned default direction**. Missingness is treated as informative, decided per node. `NaN` ≠ 0.

### 8. What is regularization doing behind the scenes? (generic view) 🌲
→ [§11](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#11-what-is-regularization-actually-doing). `Loss + λ·Complexity`. Four framings: bias-variance trade, Bayesian prior, constrained optimization (which explains *why* L1 zeroes coefficients and L2 doesn't), and capacity control. Precise claim: it trades training fit for generalization, and only helps if the model can overfit in the first place.

### 9. Regularization techniques beyond L1/L2 — what do RF and XGBoost do? 🌲
→ [§12](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#12-regularization-beyond-l1l2). Structural limits, pruning, bagging/feature subsampling (Random Forest *is* a regularization scheme), shrinkage, early stopping, more data, and domain constraints (monotonicity/interaction). Includes the side-by-side RF vs XGBoost mechanism table.

### 10. Major flaws in Decision Trees that Random Forest overcame 🌲
→ [§13](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#13-decision-tree-flaws-that-random-forest-fixes). High variance, overfitting, instability, greedy splits, dominant-feature bias, axis-aligned boundaries, outlier sensitivity — plus what RF *doesn't* fix (extrapolation, wide-short data, importance bias) and what it costs (interpretability, compute).

### 11. Splitting criteria for a Decision Tree **Regressor** 🌲
→ [§14](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md#14-splitting-criteria-for-a-decision-tree-regressor). MSE/variance reduction (default), MAE, Friedman-MSE, Poisson deviance — **not** Gini/entropy, which are class-distribution impurity measures. Bonus depth: XGBoost's gradient/Hessian gain formula reduces to regularized variance reduction for squared-error loss.

### 12. High-cardinality feature — preprocessing, and grouping strategies? 🔧
→ [§3](../ML_Fundamentals/Feature_Engineering_Selection.md#3-high-cardinality-categorical-features). Encoding table (frequency, target/WOE with out-of-fold + smoothing, hashing, embeddings, CatBoost/LightGBM native) and the four grouping families: frequency-based (Top-N + Other), target-based (supervised/monotonic binning), domain-hierarchy, and similarity-based.

### 13. Coding: sort & highest sequence in a string
→ [`../Example_Company/01_DSA_Coding.md`](../Example_Company/01_DSA_Coding.md#question-4-sort-a-string-and-find-its-longest-run) — added there as Question 4, with all three plausible readings of the prompt, since "highest sequence" is ambiguous and the right first move is to ask which one they mean.

---

## Patterns across both rounds

1. **XGBoost dominates** — 9 of 27 questions. Know it mechanically, not just as an API: the gain formula, why second-order, what each hyperparameter physically does.
2. **They probe the caveat, not the fact.** "Anything with the learning rate?", "which parameter sets the threshold?" — several questions are traps where the correct answer is a correction. Confident, specific correction scores higher than a plausible guess.
3. **Fraud/credit-risk framing throughout** — imbalance, precision/recall, top-1% ranking. Bring cost-based reasoning and operating points, not just metrics.
4. **Two soft questions carry real weight** — the client-wants-to-drop-a-feature question and "what metrics did you use" test judgement and honesty more than knowledge.
5. **GenAI is asked but not deeply, at least here** — one question in round 1. Have a crisp project narrative ready; depth lives in 🤖 if they push.

---

# Topic checklist

Where each reference topic is covered. Use it to find gaps fast.

## Core ML

| Topic | Covered in |
|---|---|
| ML overview / model families | [ML_Algorithms_Cheatsheet](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md) — 31 algorithms |
| Linear & logistic regression | [ML_Algorithms_Cheatsheet](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md) §1, §6 · [notebook](../ML_Fundamentals/ML_Training_Notebook.ipynb) |
| Decision tree | [ML_Algorithms_Cheatsheet](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md) §12 · [XGBoost deep dive](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md) §13, §14 |
| XGBoost / LightGBM / CatBoost | [XGBoost deep dive](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md) (all) · [CatBoost deep dive](../ML_Fundamentals/CatBoost_Deep_Dive.md) · [cheatsheet](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md) §17–19 |
| Bagging vs boosting | [XGBoost deep dive](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md) §6 |
| Clustering / DBSCAN | [ML_Algorithms_Cheatsheet](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md) §20–25 |
| Overfitting / underfitting | [XGBoost deep dive](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md) §9, §11 · [Example Company 09](../Example_Company/09_Classical_ML_Statistics.md) Q1–Q2 |
| Imbalanced datasets | [above](#6-how-do-you-deal-with-imbalanced-data-in-modelling) R1Q6 · [Example Company 09](../Example_Company/09_Classical_ML_Statistics.md) Q4 |
| ROC / AUC (and PR-AUC) | [above](#8-explain-precision-and-recall-for-fraud-detection-which-would-you-suggest) R1Q8 · [notebook](../ML_Fundamentals/ML_Training_Notebook.ipynb) |
| Feature engineering | [Feature_Engineering_Selection](../ML_Fundamentals/Feature_Engineering_Selection.md) §4 |
| Feature selection — filter / wrapper / embedded | [Feature_Engineering_Selection](../ML_Fundamentals/Feature_Engineering_Selection.md) §2 |
| Data leakage | [Feature_Engineering_Selection](../ML_Fundamentals/Feature_Engineering_Selection.md) §1 |
| High-cardinality features | [Feature_Engineering_Selection](../ML_Fundamentals/Feature_Engineering_Selection.md) §3 · [CatBoost deep dive](../ML_Fundamentals/CatBoost_Deep_Dive.md) §2, §7 |
| Model development lifecycle | [Example Company 12](../Example_Company/12_MLOps_CICD_LLM_Hosting.md) |
| Multithreading vs multiprocessing | [Example Company 03](../Example_Company/03_Distributed_Systems_Multithreading.md) |

## Deep learning

| Topic | Covered in |
|---|---|
| DL fundamentals (backprop, optimizers, normalization, regularization) | [DL_Concepts_Cheatsheet](../ML_Fundamentals/DL_Concepts_Cheatsheet.md) Part 1 |
| Architectures (CNN, RNN, LSTM, GAN, VAE, diffusion, GNN, ViT, CLIP) | [DL_Concepts_Cheatsheet](../ML_Fundamentals/DL_Concepts_Cheatsheet.md) Part 2 |
| Transformer architecture | [Transformers_Cheatsheet](../ML_Fundamentals/Transformers_Cheatsheet.md) Parts 1–2 |
| BERT / encoder-only models | [Transformers_Cheatsheet](../ML_Fundamentals/Transformers_Cheatsheet.md) §8 |
| Decoder layer / causal attention | [Transformers_Cheatsheet](../ML_Fundamentals/Transformers_Cheatsheet.md) §4, §9 |

## GenAI / LLM

| Topic | Covered in |
|---|---|
| Tokenization | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §1 |
| SLM vs LLM | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §2 |
| LLM params — temperature, top_p, **frequency & presence penalty** | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §3 |
| Prompt engineering (+ injection) | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §4 |
| Vector DBs & embeddings | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §5 |
| **Types of RAG (~20 named variants)** | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §6 |
| Problems of RAG | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §6 · [Example Company 10](../Example_Company/10_LLM_Finetuning_RAG_GenAI.md) Q4 |
| LLM evaluation metrics (incl. RAGAS quadrant) | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §7 |
| Fine-tuning LLMs (LoRA/QLoRA/PEFT, RLHF/DPO/GRPO) | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §8 · [Transformers_Cheatsheet](../ML_Fundamentals/Transformers_Cheatsheet.md) §11–12 |
| Agentic AI | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §9 |
| Model selection across GPT generations | [GenAI_Cheatsheet](../ML_Fundamentals/GenAI_Cheatsheet.md) §10 |
| LLM hosting & serving | [Example Company 12](../Example_Company/12_MLOps_CICD_LLM_Hosting.md) · [Transformers_Cheatsheet](../ML_Fundamentals/Transformers_Cheatsheet.md) §15–16 |

## Your own projects

| Topic | Covered in |
|---|---|
| Intro / classification work / data scale / metrics used / GenAI projects | Not documented — see the gaps list below; these need your own specifics |

---

## Notable gaps to fill yourself

These need *your* specifics and can't be pre-written:

- [ ] 60–90 second introduction, ending on a hook toward your strongest system
- [ ] Exact data scale per project (rows, features, refresh cadence) and what scale forced you to change
- [ ] The real measurement methodology behind every impact number on your CV — what the baseline/counterfactual was, and what confound could explain it away
- [ ] Which evaluation metrics you actually used per project, and why that metric
- [ ] One story for the "client wants to drop a high-importance feature" pattern — a time you pushed back, or conceded, on a data decision
