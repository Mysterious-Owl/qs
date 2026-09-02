# Classical ML & Statistics

Referral tip: "classical ML questions." 🔵 This entire topic cluster is the **highest-confidence** in the whole prep set — corroborated across 4–5 independent source types (Glassdoor summaries, direct-fetched SQLPad/Exponent/Prepfully content, Medium reference). Near-certain to be tested. 🟢

> For fuller theory/math on every algorithm below (plus ~25 more), see [`../ML_Fundamentals/ML_Algorithms_Cheatsheet.md`](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md), and [`../ML_Fundamentals/ML_Training_Notebook.ipynb`](../ML_Fundamentals/ML_Training_Notebook.ipynb) for hands-on reps.

---

### Q1. Bias-variance trade-off

**Answer**: Total generalization error decomposes as `Bias² + Variance + Irreducible error`.
- **Bias**: error from overly simplistic assumptions (underfitting) — e.g., fitting a linear model to a nonlinear relationship.
- **Variance**: error from sensitivity to the specific training sample (overfitting) — small changes in training data cause large changes in the fitted model.
- Increasing model complexity typically *decreases* bias but *increases* variance — the trade-off is the reason validation curves are U-shaped in test error vs. complexity.
- Practical levers: regularization (L1/L2), ensembling (bagging reduces variance, boosting reduces bias), more training data (reduces variance without increasing bias), simpler features (reduces variance at cost of bias).

### Q2. Preventing/detecting overfitting

**Detect**: large gap between train and validation/test metric; validation loss increases while train loss keeps decreasing; performance degrades sharply on a slightly different data slice.

**Prevent**: regularization (L1/L2/dropout), early stopping, cross-validation for model selection (not just a single train/val split), reducing model capacity, more/better training data, data augmentation, ensembling.

### Q3. Random Forest vs. Gradient Boosting

| | Random Forest | Gradient Boosting (XGBoost/LightGBM) |
|---|---|---|
| Tree building | Independent, parallel (bagging) | Sequential — each tree corrects prior residuals |
| Bias/variance | Reduces variance, bias ≈ single tree's bias | Reduces bias, can overfit (increase variance) if unregularized |
| Speed | Trains in parallel, generally faster to train | Sequential — slower to train, faster inference typically |
| Sensitivity to hyperparameters | Fairly robust out of the box | More sensitive — needs tuning (learning rate, depth, early stopping) |
| When to use | Strong baseline, less tuning needed, robust to noise | Usually higher accuracy ceiling with proper tuning, standard choice for tabular Kaggle/production problems (fraud, ranking) |

### Q4. Handling imbalanced datasets

- Don't use accuracy — use PR-AUC, F1, recall@fixed-precision, or a cost-weighted metric.
- Resampling: SMOTE (synthetic minority oversampling), random undersampling of majority class, or a hybrid.
- Algorithmic: class weighting in the loss function, focal loss (down-weights easy/majority examples).
- Threshold tuning post-training rather than assuming 0.5 is the right cutoff — pick the threshold that matches the actual cost trade-off (directly relevant to fraud/compliance system design in files [04](04_ML_System_Design_Fraud_Detection.md)/[07](07_Content_Compliance_Moderation.md)).

### Q5. Linear regression assumptions

1. Linearity (relationship between X and y is linear)
2. Independence of errors (no autocorrelation — often violated in time series)
3. Homoscedasticity (constant variance of residuals across fitted values)
4. Normality of residuals (mainly matters for inference/confidence intervals, less for pure prediction)
5. No (or low) multicollinearity among predictors

**If violated**: heteroscedasticity → use robust standard errors or transform the target (log); multicollinearity → VIF check, drop/combine correlated features, or use ridge regression; non-linearity → polynomial/spline features or switch model family; autocorrelated errors → this is a time-series signal, likely need a time-series-aware model instead.

### Q6. p-value and R² — why they're not sufficient alone

- p-value tells you whether an effect is *statistically distinguishable from zero* given the sample size — it says nothing about *effect size* or practical significance. With enough data, trivially small effects become "significant."
- R² measures variance explained but always weakly increases as you add predictors (even useless ones) — use adjusted R² for model comparison, and R² alone says nothing about whether individual coefficients are meaningful, whether assumptions hold, or whether the model generalizes.

### Q7. Ridge vs. Lasso

- Both add a penalty to reduce overfitting/multicollinearity.
- **Ridge (L2)**: shrinks coefficients smoothly toward zero, never exactly zero — keeps all features, good when many features are weakly informative.
- **Lasso (L1)**: can shrink coefficients exactly to zero — performs implicit feature selection, good when you believe only a subset of features truly matters.
- Elastic Net combines both when you want feature selection but also want to handle groups of correlated features better than Lasso alone does.

### Q8. Time-series forecasting

Reported as asked "extensively" of at least one 2025 candidate. 🟡 Know: stationarity and why it matters (ADF test, differencing to achieve it), classical methods (ARIMA/SARIMA, exponential smoothing) vs. ML approaches (gradient-boosted trees with lag/rolling features, or sequence models), and the critical practical point — **time-based train/test splits, never random splits**, to avoid leaking future information into training.

### Q9. Supervised vs. unsupervised vs. reinforcement learning

Standard definitions, but be ready to give a catalog-relevant example of each: supervised (attribute classification with labeled data), unsupervised (clustering similar products for the duplicate-detection candidate-generation step in [06](06_Duplicate_Product_Detection.md)), reinforcement learning (less common here, but could frame ranking/recommendation policy optimization as RL if pushed).

### Q10. K-means reasoning on a sample dataset

Be ready to reason through, not just define: how you'd choose k (elbow method, silhouette score), sensitivity to feature scaling (must standardize features first — Euclidean distance is scale-dependent), sensitivity to initialization (k-means++ vs. random init), and its core limitation (assumes roughly spherical, similarly-sized clusters — fails on elongated or highly imbalanced-density clusters, where DBSCAN or GMM would be more appropriate).

---

### What This Round Tests

- Depth, not breadth — interviewers reportedly push past the textbook definition into "what happens if X is violated" or "why not just use accuracy" follow-ups
- Whether you can connect fundamentals to *this team's actual problems* unprompted (imbalanced-class reasoning → fraud/compliance; clustering → duplicate detection) — this is the single easiest way to stand out in this round
