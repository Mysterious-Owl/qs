# ML Algorithms Cheat Sheet

~30 major, commonly-asked ML algorithms across regression, classification, ensembles, clustering, dimensionality reduction, and a couple of adjacent essentials. Pairs with [`ML_Training_Notebook.ipynb`](ML_Training_Notebook.ipynb) (hands-on code for Linear Regression, Logistic Regression, K-Means, XGBoost) and [`../Example_Company/09_Classical_ML_Statistics.md`](../Example_Company/09_Classical_ML_Statistics.md) (how these get interrogated in an actual interview).

## How to read each entry

| Field | What it answers |
|---|---|
| Type | Supervised/unsupervised, regression/classification/clustering/etc. |
| What it does | One-line summary |
| Core equation | The math you should be able to write on a whiteboard |
| Key theory | The intuition behind why it works |
| Differs from | The nearest-neighbor algorithms and the actual distinction |
| Key hyperparameters | What you'd tune and roughly why |
| Preconditions / assumptions | What to check before trusting the algorithm's output |
| Preprocessing needed | Scaling, encoding, missing-value handling required or not |
| Evaluation / output checks | Metrics and diagnostic checks on the result |
| When to choose | The scenario that makes this the right tool |
| Works best with | Data characteristics / companion techniques |
| Key terms | Vocabulary you should be fluent in |
| Gotcha | The thing interviewers commonly probe or candidates commonly get wrong |

---

# A. Regression

### 1. Linear Regression
- **Type**: Supervised, regression
- **What it does**: Fits a linear relationship between features and a continuous target
- **Core equation**: `y = β₀ + β₁x₁ + ... + βₙxₙ + ε`, minimizing `Σ(y - ŷ)²` (Ordinary Least Squares); closed form `β = (XᵀX)⁻¹Xᵀy`
- **Key theory**: Minimizes squared error → the OLS solution is the Best Linear Unbiased Estimator (BLUE) under the Gauss-Markov assumptions
- **Differs from**: Ridge/Lasso add a penalty term to the same objective; Polynomial Regression is still linear *in parameters* but adds nonlinear feature terms
- **Key hyperparameters**: None for closed-form OLS; `fit_intercept`
- **Preconditions/assumptions**: Linearity, independence of errors, homoscedasticity, normally distributed residuals (for inference), low multicollinearity
- **Preprocessing needed**: Not strictly required for OLS itself, but scaling helps numerical stability; one-hot/encode categoricals
- **Evaluation/output checks**: R², adjusted R², RMSE/MAE, residual-vs-fitted plot (should be random scatter), Q-Q plot (normality), VIF (multicollinearity)
- **When to choose**: Need an interpretable baseline, relationship is roughly linear, inference on coefficients matters
- **Works best with**: Low-dimensional, mostly linear, low-multicollinearity numeric data
- **Key terms**: OLS, residual, homoscedasticity, multicollinearity, VIF, R² vs. adjusted R²
- **Gotcha**: R² never decreases as you add features (even useless ones) — always compare models with adjusted R², not raw R²

### 2. Ridge Regression (L2)
- **Type**: Supervised, regression (regularized)
- **What it does**: Linear regression with an L2 penalty that shrinks coefficients toward zero
- **Core equation**: Minimize `Σ(y - ŷ)² + λΣβᵢ²`
- **Key theory**: Equivalent to placing a Gaussian prior on the coefficients (MAP estimation) — trades a little bias for a large reduction in variance when features are correlated
- **Differs from**: Lasso (L1) can zero out coefficients entirely; Ridge shrinks but never exactly to zero
- **Key hyperparameters**: `alpha`/`λ` (regularization strength) — tune via cross-validation
- **Preconditions/assumptions**: Same as linear regression, but specifically helps when multicollinearity is present
- **Preprocessing needed**: **Feature scaling is mandatory** — the penalty is scale-dependent
- **Evaluation/output checks**: Cross-validated RMSE/R² across a range of `λ`; coefficient shrinkage path plot
- **When to choose**: Many correlated features, want to keep all of them but reduce overfitting
- **Works best with**: High-dimensional or multicollinear numeric data
- **Key terms**: L2 penalty, shrinkage, bias-variance trade-off, regularization path
- **Gotcha**: Forgetting to scale features before fitting Ridge/Lasso silently biases which coefficients get penalized most

### 3. Lasso Regression (L1)
- **Type**: Supervised, regression (regularized)
- **What it does**: Linear regression with an L1 penalty that can shrink coefficients exactly to zero — implicit feature selection
- **Core equation**: Minimize `Σ(y - ŷ)² + λΣ|βᵢ|`
- **Key theory**: The L1 penalty's geometry (a diamond-shaped constraint region) makes it likely that the optimum lands exactly on an axis (coefficient = 0), unlike L2's circular region
- **Differs from**: Ridge shrinks smoothly, never to exactly zero; Elastic Net blends L1+L2 to get selection + stability
- **Key hyperparameters**: `alpha`/`λ`
- **Preconditions/assumptions**: Same as linear regression
- **Preprocessing needed**: Feature scaling mandatory
- **Evaluation/output checks**: Cross-validated RMSE/R²; count of non-zero coefficients at chosen `λ`
- **When to choose**: Believe only a subset of features truly matters and want automatic selection
- **Works best with**: High-dimensional data where sparsity is plausible
- **Key terms**: L1 penalty, sparsity, feature selection
- **Gotcha**: With groups of highly correlated features, Lasso arbitrarily picks one and zeroes the rest — unstable selection; Elastic Net handles this better

### 4. Elastic Net
- **Type**: Supervised, regression (regularized)
- **What it does**: Combines L1 and L2 penalties
- **Core equation**: Minimize `Σ(y - ŷ)² + λ₁Σ|βᵢ| + λ₂Σβᵢ²`
- **Key theory**: Gets Lasso's sparsity while handling correlated-feature groups more stably (via the L2 term)
- **Differs from**: A strict blend of Ridge and Lasso — has two tunable penalty weights (or one strength + one mixing ratio) instead of one
- **Key hyperparameters**: `alpha` (overall strength), `l1_ratio` (mix between L1 and L2)
- **Preconditions/assumptions**: Same as linear regression
- **Preprocessing needed**: Feature scaling mandatory
- **Evaluation/output checks**: Cross-validated RMSE/R² over a grid of `(alpha, l1_ratio)`
- **When to choose**: Many features, some correlated groups, want both selection and stability
- **Works best with**: High-dimensional, correlated feature sets (genomics-style data is the classic example)
- **Key terms**: L1 ratio, penalty mixing
- **Gotcha**: Two hyperparameters to tune instead of one — more expensive cross-validation grid

### 5. Polynomial Regression
- **Type**: Supervised, regression
- **What it does**: Fits linear regression on polynomial-expanded features to capture nonlinear relationships
- **Core equation**: `y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε` — still linear in the βs
- **Key theory**: Feature engineering trick — the model is still a linear model, just over a nonlinear basis
- **Differs from**: True nonlinear models (trees, kernel SVM) learn nonlinearity directly; this hand-engineers it via explicit polynomial terms
- **Key hyperparameters**: Polynomial `degree`
- **Preconditions/assumptions**: Same as linear regression on the transformed feature space
- **Preprocessing needed**: Scaling strongly recommended — high-degree terms blow up in magnitude
- **Evaluation/output checks**: Train vs. validation error curve vs. degree — high degree overfits badly (classic bias-variance demo)
- **When to choose**: Clear, moderate nonlinearity, still want interpretability of a linear-in-basis model
- **Works best with**: Low-dimensional data (curse of dimensionality hits polynomial expansion hard)
- **Key terms**: Basis expansion, overfitting with degree
- **Gotcha**: Degree is a variance dial — too high and it memorizes noise; always validate against held-out data, never pick degree by train error alone

---

# B. Classification — Linear, Probabilistic, Instance-Based, Margin-Based

### 6. Logistic Regression
- **Type**: Supervised, classification (binary or multinomial via softmax)
- **What it does**: Models the log-odds of the positive class as a linear function of features
- **Core equation**: `p = 1 / (1 + e^-(β₀+β₁x₁+...))`; trained by maximizing log-likelihood (equivalently minimizing log loss/cross-entropy)
- **Key theory**: The sigmoid squashes a linear score into a valid probability; decision boundary is linear in feature space
- **Differs from**: Linear regression predicts a continuous value directly; Naive Bayes models the joint distribution generatively rather than the decision boundary discriminatively; SVM maximizes margin rather than likelihood
- **Key hyperparameters**: `C` (inverse regularization strength), penalty type (L1/L2/elastic net), `class_weight`
- **Preconditions/assumptions**: Roughly linear decision boundary in (possibly transformed) feature space; features not perfectly collinear
- **Preprocessing needed**: Scaling recommended (gradient-based optimization converges faster and regularization is scale-dependent); encode categoricals
- **Evaluation/output checks**: Accuracy, precision/recall/F1, ROC-AUC, PR-AUC (for imbalance), calibration curve, confusion matrix
- **When to choose**: Need interpretable, probabilistic, fast baseline classifier
- **Works best with**: Roughly linearly separable classes, moderate feature count
- **Key terms**: Sigmoid, log-odds, log loss, decision boundary, regularization (`C`)
- **Gotcha**: Default 0.5 threshold is arbitrary — always justify the operating threshold against the actual cost trade-off (see [Example_Company/04](../Example_Company/04_ML_System_Design_Fraud_Detection.md))

### 7. K-Nearest Neighbors (KNN)
- **Type**: Supervised, classification or regression (instance-based / lazy learning)
- **What it does**: Predicts based on the majority class (or average value) of the k nearest training points
- **Core equation**: Distance metric, typically Euclidean: `d(x, xᵢ) = √Σ(xⱼ - xᵢⱼ)²`; predict via majority vote (classification) or mean (regression) of k nearest
- **Key theory**: No explicit training/model — "the model is the data"; assumes similar points (in feature space) have similar labels
- **Differs from**: Parametric models (logistic regression) learn a fixed set of parameters; KNN stores the whole training set and computes at inference time
- **Key hyperparameters**: `k` (number of neighbors), distance metric, weighting (uniform vs. distance-weighted)
- **Preconditions/assumptions**: Meaningful distance metric in the feature space; features on comparable scales
- **Preprocessing needed**: **Scaling is mandatory** — distance metrics are scale-dependent; curse of dimensionality means it degrades in very high dimensions
- **Evaluation/output checks**: Accuracy/F1 vs. k (elbow-style curve); cross-validation to pick k
- **When to choose**: Simple, non-linear decision boundary, small-to-medium dataset, low latency not critical
- **Works best with**: Low-to-moderate dimensional, well-scaled, reasonably dense data
- **Key terms**: Lazy learning, curse of dimensionality, distance-weighted voting
- **Gotcha**: Inference is O(n) per query without an index (KD-tree/ball-tree/ANN) — doesn't scale naively to large n; very sensitive to irrelevant/noisy features since all features contribute equally to distance unless weighted

### 8. Naive Bayes
- **Type**: Supervised, classification (generative, probabilistic)
- **What it does**: Applies Bayes' theorem assuming features are conditionally independent given the class
- **Core equation**: `P(y|x) ∝ P(y)·Π P(xᵢ|y)` — pick the class maximizing this
- **Key theory**: The "naive" conditional-independence assumption is almost always false in practice, yet the classifier is often still competitive because it only needs the *ranking* of class probabilities to be right, not their exact calibration
- **Differs from**: Logistic regression is discriminative (models P(y|x) directly); Naive Bayes is generative (models P(x|y) and P(y), then applies Bayes' rule)
- **Key hyperparameters**: Smoothing parameter (Laplace/additive smoothing, e.g. `alpha`); choice of variant (Gaussian for continuous, Multinomial for counts, Bernoulli for binary features)
- **Preconditions/assumptions**: Conditional independence of features given the class (rarely true, model is still often useful)
- **Preprocessing needed**: Minimal — doesn't require scaling; needs the right variant matched to feature type (counts vs. continuous vs. binary)
- **Evaluation/output checks**: Accuracy/F1; check calibration separately if probability estimates (not just class ranking) matter, since NB probabilities are often poorly calibrated
- **When to choose**: Text classification (spam, sentiment), very high-dimensional sparse features, need a fast, strong baseline
- **Works best with**: High-dimensional, sparse, roughly-independent features (bag-of-words text data is the classic case)
- **Key terms**: Bayes' theorem, conditional independence, Laplace smoothing, generative vs. discriminative
- **Gotcha**: Zero-frequency problem — an unseen feature-class combination gives probability 0 and wipes out the whole product unless smoothing is applied

### 9. Support Vector Machine (SVM)
- **Type**: Supervised, classification (or regression via SVR) — margin-based
- **What it does**: Finds the hyperplane that maximizes the margin between classes; kernels allow non-linear boundaries
- **Core equation**: Maximize margin `2/‖w‖` subject to `yᵢ(w·xᵢ+b) ≥ 1` (hard margin); soft-margin adds slack variables + penalty `C`; kernel trick replaces dot products `x·x'` with `K(x,x')` to implicitly work in higher-dimensional space
- **Key theory**: Only the points closest to the boundary (support vectors) determine it — robust to points far from the boundary; the kernel trick avoids ever explicitly computing the high-dimensional mapping
- **Differs from**: Logistic regression optimizes likelihood everywhere; SVM only cares about the margin/support vectors, making it more robust to well-separated outliers but less naturally probabilistic
- **Key hyperparameters**: `C` (margin softness — low C = wider margin, more tolerance for misclassification), kernel choice (linear/RBF/polynomial), kernel-specific params (`gamma` for RBF)
- **Preconditions/assumptions**: Works best with a clear margin between classes (with the right kernel); doesn't assume a specific data distribution
- **Preprocessing needed**: **Scaling is mandatory** — margin/distance computation is scale-dependent
- **Evaluation/output checks**: Accuracy/F1/ROC-AUC; grid search over `C`/`gamma` via cross-validation
- **When to choose**: Medium-sized datasets, need a strong non-linear classifier, high-dimensional data (e.g. text) where linear SVM often shines
- **Works best with**: Clean, well-scaled, not-too-massive datasets (training scales poorly, roughly O(n²)-O(n³))
- **Key terms**: Support vectors, margin, kernel trick, `C`, `gamma`, soft margin
- **Gotcha**: Doesn't scale well to very large n (training cost); RBF kernel with a poorly tuned `gamma` can wildly overfit (too high) or underfit (too low)

### 10. Linear Discriminant Analysis (LDA)
- **Type**: Supervised, classification (also used for dimensionality reduction)
- **What it does**: Finds a linear combination of features that best separates classes, assuming each class is Gaussian with a shared covariance matrix
- **Core equation**: Projects data to maximize between-class variance relative to within-class variance (Fisher's criterion): maximize `(w·(μ₁-μ₂))² / (wᵀΣw)`
- **Key theory**: Generative model — assumes class-conditional Gaussian distributions with equal covariance, which gives a linear (not quadratic) decision boundary
- **Differs from**: Logistic regression is discriminative and makes no distributional assumption on features; QDA (Quadratic Discriminant Analysis) relaxes the equal-covariance assumption and gets a curved boundary; PCA finds directions of maximum variance (unsupervised) vs. LDA finding directions of maximum class separability (supervised)
- **Key hyperparameters**: Solver choice; shrinkage (for regularized covariance estimation in high dimensions)
- **Preconditions/assumptions**: Class-conditional normality, equal covariance across classes
- **Preprocessing needed**: Scaling generally recommended
- **Evaluation/output checks**: Accuracy/F1; check the equal-covariance assumption roughly holds (else prefer QDA)
- **When to choose**: Assumptions plausible, want a fast linear classifier or a supervised dimensionality-reduction step before another model
- **Works best with**: Roughly Gaussian, similarly-scattered classes
- **Key terms**: Fisher's criterion, between-class/within-class variance, QDA
- **Gotcha**: Often confused with PCA — LDA is supervised and optimizes for class separation, not for variance

### 11. Perceptron / Basic Neural Network (MLP)
- **Type**: Supervised, classification or regression
- **What it does**: Perceptron — a single linear threshold unit; MLP stacks multiple layers of weighted sums + nonlinear activations to learn non-linear functions
- **Core equation**: Perceptron: `ŷ = sign(w·x + b)`, updated via `w ← w + η(y-ŷ)x` on misclassification; MLP: `h = σ(Wx+b)` stacked across layers, trained via backpropagation + gradient descent
- **Key theory**: A single perceptron can only learn linearly separable functions (famously fails on XOR); stacking layers with nonlinear activations gives a universal function approximator
- **Differs from**: Logistic regression is essentially a single-layer network with a sigmoid output and a likelihood loss; deep learning architectures (CNNs, Transformers — see [Example_Company/11](../Example_Company/11_DL_Architectures.md)) are MLPs with specialized structure for images/sequences
- **Key hyperparameters**: Number of layers/units, activation function, learning rate, regularization (dropout, weight decay), epochs, batch size
- **Preconditions/assumptions**: Needs enough data to avoid overfitting a high-capacity model; non-convex optimization — no global-optimum guarantee
- **Preprocessing needed**: Scaling essential (gradient-based training)
- **Evaluation/output checks**: Train/validation loss curves, early stopping on validation loss, standard classification/regression metrics
- **When to choose**: Complex non-linear relationships, enough data/compute available, don't need full interpretability
- **Works best with**: Larger datasets; tabular use cases often still favor gradient-boosted trees over plain MLPs
- **Key terms**: Backpropagation, activation function, epoch, learning rate, overfitting/dropout
- **Gotcha**: On plain tabular data, a well-tuned gradient-boosted tree model frequently beats a plain MLP — reach for deep nets when structure (image/sequence/graph) justifies it, not by default

---

# C. Tree-Based & Ensemble Methods

### 12. Decision Tree
- **Type**: Supervised, classification or regression
- **What it does**: Recursively splits the feature space into regions using single-feature thresholds, forming a tree of if/else rules
- **Core equation**: Split selection maximizes impurity reduction — Gini impurity `1 - Σpᵢ²` or entropy `-Σpᵢlog₂pᵢ` for classification; variance reduction for regression
- **Key theory**: Greedy, recursive partitioning — each split is locally optimal, not globally optimal (NP-hard to find the truly optimal tree)
- **Differs from**: Random Forest/boosting are ensembles *of* decision trees; a single tree is the base learner these ensembles build on
- **Key hyperparameters**: `max_depth`, `min_samples_split`/`min_samples_leaf`, `max_features`, impurity criterion
- **Preconditions/assumptions**: None on data distribution — non-parametric; handles non-linear relationships and interactions natively
- **Preprocessing needed**: **No scaling needed** (splits are threshold-based, scale-invariant); handles mixed feature types; can handle missing values with some implementations
- **Evaluation/output checks**: Train vs. validation accuracy/error at varying depth (classic overfitting curve); feature importance; visualize the tree for sanity-checking splits
- **When to choose**: Need interpretability (can literally read the rules), non-linear relationships, mixed feature types, minimal preprocessing
- **Works best with**: Tabular data with non-linear feature interactions
- **Key terms**: Gini impurity, entropy, information gain, pruning, overfitting
- **Gotcha**: A single unpruned tree overfits almost by default (near-zero train error, poor generalization) — this is *the* motivating reason ensembles (bagging/boosting) exist

### 13. Bagging (Bootstrap Aggregating)
- **Type**: Ensemble meta-technique (usually applied to decision trees)
- **What it does**: Trains many base models on bootstrap-resampled (random-sample-with-replacement) subsets of the data and averages/votes their predictions
- **Core equation**: Prediction `= (1/M)Σfₘ(x)` (regression) or majority vote (classification), where each `fₘ` is trained on a bootstrap sample
- **Key theory**: Averaging independent, high-variance models reduces variance without increasing bias — works because errors of independently-trained models partially cancel out
- **Differs from**: Boosting trains models sequentially, each correcting the previous ones' errors (reduces bias); bagging trains models independently/in parallel (reduces variance)
- **Key hyperparameters**: Number of base estimators, base estimator's own hyperparameters, sample size per bootstrap
- **Preconditions/assumptions**: Base learner should be high-variance/low-bias (e.g. deep, unpruned trees) for bagging to help most
- **Preprocessing needed**: Whatever the base learner needs (none extra for trees)
- **Evaluation/output checks**: Out-of-bag (OOB) error — a free validation estimate from the ~1/3 of data each tree didn't see
- **When to choose**: Base model overfits (high variance) and you want to stabilize it without changing its bias
- **Works best with**: Unstable, high-variance base learners (deep decision trees)
- **Key terms**: Bootstrap sample, OOB error, variance reduction
- **Gotcha**: Bagging a low-variance, high-bias model (like linear regression) barely helps — it targets variance, not bias

### 14. Random Forest
- **Type**: Supervised, classification or regression (ensemble of decision trees)
- **What it does**: Bagging of decision trees, with an added twist — each split only considers a random subset of features, decorrelating the trees further
- **Core equation**: Same aggregation as bagging; feature subsampling at each split is `max_features` (commonly `√p` for classification, `p/3` for regression, where p = total features)
- **Key theory**: Decorrelating the trees (via feature subsampling, not just row subsampling) improves the variance-reduction benefit of averaging beyond plain bagging
- **Differs from**: Plain bagging of trees only resamples rows; Random Forest also resamples features per split; Gradient Boosting builds trees sequentially to reduce bias, not in parallel to reduce variance
- **Key hyperparameters**: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`
- **Preconditions/assumptions**: None on data distribution
- **Preprocessing needed**: No scaling needed; handles mixed types
- **Evaluation/output checks**: OOB error, feature importance (mean decrease in impurity, or permutation importance — the latter is more reliable, especially with correlated features)
- **When to choose**: Strong, low-maintenance baseline that's robust to noise and needs minimal tuning
- **Works best with**: Tabular data of almost any size; a good default before reaching for gradient boosting
- **Key terms**: OOB error, feature importance, decorrelation
- **Gotcha**: Default impurity-based feature importance is biased toward high-cardinality features — prefer permutation importance for a trustworthy ranking

### 15. AdaBoost
- **Type**: Supervised, classification (or regression) — boosting ensemble
- **What it does**: Sequentially trains weak learners (often shallow "decision stumps"), up-weighting misclassified examples each round, then combines learners via a weighted vote
- **Core equation**: Each learner `mₜ` gets weight `αₜ = ½ln((1-εₜ)/εₜ)` based on its weighted error `εₜ`; sample weights update as `wᵢ ← wᵢ·e^(±αₜ)` (increase for misclassified)
- **Key theory**: Focuses subsequent learners on the examples the ensemble is currently getting wrong — an explicit, interpretable boosting mechanism
- **Differs from**: Gradient Boosting generalizes this idea to fit new learners to the *gradient of an arbitrary loss function* rather than reweighting misclassified samples explicitly — AdaBoost is a special case (exponential loss)
- **Key hyperparameters**: Number of estimators, learning rate, base estimator (default: depth-1 trees/"stumps")
- **Preconditions/assumptions**: Sensitive to noisy data/outliers (they get heavily up-weighted, potentially derailing later learners)
- **Preprocessing needed**: Whatever the base learner needs
- **Evaluation/output checks**: Train/validation error vs. number of estimators (watch for overfitting with too many rounds on noisy data)
- **When to choose**: Clean-ish data, want a simple, historically foundational boosting method
- **Works best with**: Low-noise datasets with weak-but-better-than-random base learners
- **Key terms**: Weak learner, sample reweighting, exponential loss
- **Gotcha**: Highly sensitive to outliers/mislabeled data since they keep getting up-weighted round after round — Gradient Boosting/XGBoost are generally preferred in practice today

### 16. Gradient Boosting Machine (GBM)
- **Type**: Supervised, classification or regression — boosting ensemble
- **What it does**: Sequentially fits new trees to the *residual errors* (more precisely, the negative gradient of the loss) of the current ensemble
- **Core equation**: `Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x)`, where `hₘ` is fit to approximate `-∂L/∂F` (the negative gradient / pseudo-residual) of the current model, `η` = learning rate
- **Key theory**: Reframes boosting as gradient descent *in function space* — each new tree is a step in the direction that most reduces the loss
- **Differs from**: AdaBoost is a specific instance (exponential loss, reweighting); GBM generalizes to any differentiable loss function (log loss, squared error, quantile loss, etc.); XGBoost/LightGBM/CatBoost are optimized, regularized, and engineered implementations of this same core idea
- **Key hyperparameters**: `n_estimators`, `learning_rate`, `max_depth` (usually shallow, 3-8), `subsample` (stochastic gradient boosting)
- **Preconditions/assumptions**: None on data distribution; sequential nature means it can overfit if run too long without regularization
- **Preprocessing needed**: No scaling needed (tree-based); handles missing values variably by implementation
- **Evaluation/output checks**: Validation loss curve with early stopping; learning-rate/n_estimators trade-off (lower LR + more trees generally generalizes better but costs more compute)
- **When to choose**: Need the best achievable accuracy on tabular data and can afford tuning
- **Works best with**: Tabular data, moderate-to-large datasets
- **Key terms**: Pseudo-residual, learning rate (shrinkage), additive model, early stopping
- **Gotcha**: Learning rate and number of trees trade off directly — lowering the learning rate without increasing tree count just underfits; always tune them together (or use early stopping to pick tree count automatically)

### 17. XGBoost
- **Type**: Supervised, classification or regression — optimized gradient boosting
- **What it does**: An engineered, regularized, and heavily optimized implementation of gradient boosting
- **Core equation**: Adds an explicit regularization term to the GBM objective: `Obj = Σl(yᵢ,ŷᵢ) + ΣΩ(fₖ)` where `Ω(f) = γT + ½λΣwⱼ²` (T = number of leaves, w = leaf weights) — regularizes tree complexity directly, not just via depth/learning rate
- **Key theory**: Uses a second-order (Newton's method) approximation of the loss for split-finding, plus built-in L1/L2 regularization on leaf weights, and a specific method for automatically learning the best direction for missing values at each split
- **Differs from**: Plain GBM only regularizes indirectly (depth, learning rate, subsampling); XGBoost adds explicit leaf-weight and leaf-count regularization plus much faster, more efficient training (histogram-based split-finding, parallelized column block structure)
- **Key hyperparameters**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`/`reg_lambda` (L1/L2), `min_child_weight`, `early_stopping_rounds`
- **Preconditions/assumptions**: None on data distribution
- **Preprocessing needed**: No scaling needed; handles missing values natively (learns default split direction)
- **Evaluation/output checks**: Validation-set loss with early stopping; feature importance (gain-based preferred over frequency-based); SHAP values for per-prediction explainability
- **When to choose**: The default, near-universal choice for tabular ML problems — fraud scoring, ranking, churn, attribute-quality models
- **Works best with**: Structured/tabular data of essentially any size that fits in memory (or via distributed mode)
- **Key terms**: Second-order approximation, leaf weight regularization, gain-based importance, SHAP
- **Gotcha**: Extreme class imbalance (e.g. <1% fraud) still needs explicit handling — `scale_pos_weight`, resampling, or threshold tuning — XGBoost's optimization doesn't automatically fix a badly imbalanced objective

### 18. LightGBM
- **Type**: Supervised, classification or regression — optimized gradient boosting
- **What it does**: A gradient boosting implementation optimized for speed and memory on very large datasets
- **Core equation**: Same GBM/leaf-weight-regularized objective as XGBoost, but grows trees **leaf-wise** (best-first, split the leaf with the highest loss reduction) rather than level-wise, and uses histogram-based binning of continuous features for faster split-finding
- **Key theory**: Leaf-wise growth converges faster per tree but can overfit more easily on small datasets if depth isn't constrained; Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) further speed up training on large, sparse, high-dimensional data
- **Differs from**: XGBoost grows trees level-wise (breadth-first) by default; LightGBM's leaf-wise growth is faster but needs `max_depth`/`num_leaves` capped to avoid overfitting
- **Key hyperparameters**: `num_leaves` (the primary complexity control, not `max_depth`), `learning_rate`, `n_estimators`, `min_data_in_leaf`
- **Preconditions/assumptions**: None on data distribution; handles categorical features natively without one-hot encoding
- **Preprocessing needed**: No scaling needed; can pass categorical columns directly (native categorical handling)
- **Evaluation/output checks**: Same as GBM/XGBoost — validation loss, early stopping, feature importance
- **When to choose**: Very large datasets or many categorical features where training speed/memory matters
- **Works best with**: Large-scale tabular data, high-cardinality categorical features
- **Key terms**: Leaf-wise growth, GOSS, EFB, `num_leaves`
- **Gotcha**: `num_leaves` should be set relative to `max_depth` (roughly `2^max_depth`) — leaving it too high on a small dataset is a very easy way to badly overfit

### 19. CatBoost
> Mechanism-level detail: [`CatBoost_Deep_Dive.md`](CatBoost_Deep_Dive.md)

- **Type**: Supervised, classification or regression — optimized gradient boosting
- **What it does**: A gradient boosting implementation specifically engineered for datasets with many categorical features, with built-in target-encoding-style handling
- **Core equation**: Same core GBM objective, but uses **ordered target statistics** for encoding categorical features (a permutation-based scheme that avoids the target leakage a naive mean-target-encoding would cause) and **symmetric (oblivious) trees**, where the same split condition is used across an entire tree level
- **Key theory**: Naive target encoding (replacing a category with the mean target for that category) leaks label information into the training features; CatBoost's ordered boosting computes each row's encoding using only rows that "came before" it in a random permutation, avoiding this leakage
- **Differs from**: XGBoost/LightGBM require categorical features to be pre-encoded (one-hot, label, or manually target-encoded); CatBoost handles them internally with leakage-safe encoding, and its symmetric trees act as a form of built-in regularization
- **Key hyperparameters**: `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `cat_features` (indices of categorical columns)
- **Preconditions/assumptions**: None on data distribution
- **Preprocessing needed**: No manual categorical encoding needed — pass raw categorical columns and their indices
- **Evaluation/output checks**: Same as other GBM variants; CatBoost also ships strong default hyperparameters, often competitive out-of-the-box
- **When to choose**: Datasets dominated by categorical features, want to avoid manual target-encoding pipelines and their leakage risk
- **Works best with**: Tabular data with many/high-cardinality categorical columns
- **Key terms**: Ordered target encoding, symmetric/oblivious trees, target leakage
- **Gotcha**: Symmetric trees are a form of regularization but can be less expressive per tree than asymmetric trees — CatBoost tends to need more trees to match XGBoost's per-tree flexibility, though its defaults usually compensate well

---

# D. Clustering

### 20. K-Means
- **Type**: Unsupervised, clustering
- **What it does**: Partitions data into k clusters by iteratively assigning points to the nearest centroid and recomputing centroids
- **Core equation**: Minimizes within-cluster sum of squares (inertia): `Σₖ Σ_{x∈Cₖ} ‖x - μₖ‖²`
- **Key theory**: Lloyd's algorithm — alternates assignment step (nearest centroid) and update step (recompute centroid as cluster mean); converges to a local optimum, not guaranteed global
- **Differs from**: DBSCAN doesn't need k specified and finds arbitrary-shaped clusters; GMM is a soft/probabilistic generalization allowing elliptical clusters and cluster-membership probabilities instead of hard assignment
- **Key hyperparameters**: `k` (number of clusters), `n_init` (number of random restarts — mitigates sensitivity to initialization), init method (`k-means++` is the standard, smarter-than-random default)
- **Preconditions/assumptions**: Assumes roughly spherical, similarly-sized, similarly-dense clusters (Euclidean-distance-based)
- **Preprocessing needed**: **Scaling is mandatory** — Euclidean distance is scale-dependent
- **Evaluation/output checks**: Elbow method (inertia vs. k), silhouette score (more objective — quantifies separation/cohesion), Davies-Bouldin index
- **When to choose**: Roughly spherical, similarly-sized clusters expected, need a fast, simple, scalable method
- **Works best with**: Numeric, scaled, low-to-moderate dimensional data without extreme outliers
- **Key terms**: Centroid, inertia, k-means++, silhouette score
- **Gotcha**: Fails badly on elongated, unequal-density, or non-convex clusters (e.g. two concentric rings) — DBSCAN or spectral clustering handle those shapes; also sensitive to outliers, which can drag a centroid noticeably

### 21. K-Medoids (PAM — Partitioning Around Medoids)
- **Type**: Unsupervised, clustering
- **What it does**: Like K-Means, but cluster centers are actual data points (medoids) rather than computed means, and any distance metric can be used
- **Core equation**: Minimizes total dissimilarity of points to their assigned medoid, using any distance metric `d(x, medoid)` (not restricted to Euclidean)
- **Key theory**: Using an actual data point as the center makes the method robust to outliers (a mean can be dragged arbitrarily far by one outlier; a medoid cannot) and allows non-Euclidean distance metrics
- **Differs from**: K-Means uses the mean (sensitive to outliers, Euclidean-only); K-Medoids uses a real point as center and works with arbitrary distance metrics
- **Key hyperparameters**: `k`, distance metric
- **Preconditions/assumptions**: Same rough shape assumptions as K-Means but more outlier-tolerant
- **Preprocessing needed**: Scaling still generally recommended
- **Evaluation/output checks**: Same as K-Means (silhouette score, etc.)
- **When to choose**: Need outlier robustness or a non-Euclidean distance metric (e.g. categorical/mixed-type data with a custom dissimilarity measure)
- **Works best with**: Data with outliers, or non-numeric/mixed feature types with a defined dissimilarity function
- **Key terms**: Medoid, PAM algorithm
- **Gotcha**: Computationally more expensive than K-Means (medoid search is more costly than computing a mean) — doesn't scale to as large n without approximations (e.g. CLARA)

### 22. Hierarchical Clustering (Agglomerative)
- **Type**: Unsupervised, clustering
- **What it does**: Builds a tree (dendrogram) of nested clusters by iteratively merging the closest pair of clusters (agglomerative/bottom-up) or splitting (divisive/top-down)
- **Core equation**: Linkage criteria define "distance between clusters": single (min pairwise distance), complete (max pairwise distance), average, or Ward's (minimizes increase in within-cluster variance)
- **Key theory**: Doesn't require specifying the number of clusters upfront — you cut the dendrogram at whatever height gives the desired number of clusters, and the full hierarchy is visualizable
- **Differs from**: K-Means requires k upfront and produces a flat partition; hierarchical clustering produces a full nested hierarchy you can cut at any level
- **Key hyperparameters**: Linkage method, distance metric, number of clusters (or dendrogram cut height) chosen after the fact
- **Preconditions/assumptions**: None as strict as K-Means's spherical assumption, but linkage choice strongly shapes what cluster shapes are found (e.g. single linkage can produce elongated "chained" clusters)
- **Preprocessing needed**: Scaling generally recommended
- **Evaluation/output checks**: Dendrogram inspection, cophenetic correlation coefficient, silhouette score at a chosen cut
- **When to choose**: Don't know k in advance, want to explore cluster structure at multiple granularities, or need a visual hierarchy (e.g. taxonomy-like structure)
- **Works best with**: Small-to-medium datasets (naively O(n²) or O(n³) depending on implementation — doesn't scale as well as K-Means)
- **Key terms**: Dendrogram, linkage (single/complete/average/Ward), cophenetic correlation
- **Gotcha**: Once two points are merged into a cluster, that decision is never undone (greedy) — a bad early merge can propagate; doesn't scale to very large n without approximations

### 23. DBSCAN (Density-Based Spatial Clustering)
- **Type**: Unsupervised, clustering
- **What it does**: Groups together points that are closely packed (many nearby neighbors), marking points in low-density regions as noise/outliers
- **Core equation**: A point is a "core point" if at least `min_samples` other points lie within distance `eps`; clusters are formed by chaining together core points and their reachable neighbors
- **Key theory**: Density-based rather than centroid-based — doesn't need k specified, naturally identifies outliers as noise, and can find arbitrarily-shaped (non-convex) clusters
- **Differs from**: K-Means needs k specified and assumes spherical clusters; DBSCAN needs `eps`/`min_samples` instead, finds arbitrary shapes, and explicitly labels outliers rather than forcing every point into a cluster — this is exactly why it's well-suited to fraud/anomaly-adjacent clustering — coordinated fraud rings form dense pockets while most legitimate activity stays sparse
- **Key hyperparameters**: `eps` (neighborhood radius), `min_samples` (density threshold)
- **Preconditions/assumptions**: Assumes clusters have relatively uniform density (struggles when clusters have very different densities)
- **Preprocessing needed**: **Scaling is mandatory** — `eps` is a distance threshold
- **Evaluation/output checks**: Silhouette score (excluding noise points), fraction of points labeled noise (sanity check against domain expectations), k-distance plot to help choose `eps`
- **When to choose**: Unknown number of clusters, expect noise/outliers in the data, clusters may be non-spherical
- **Works best with**: Spatial or density-varying data where outlier identification itself is valuable (fraud rings, geographic clustering)
- **Key terms**: Core point, border point, noise point, `eps`, `min_samples`, density-reachability
- **Gotcha**: A single global `eps` struggles when clusters have very different densities — HDBSCAN (hierarchical DBSCAN) relaxes this by not requiring one fixed density threshold

### 24. Gaussian Mixture Model (GMM)
- **Type**: Unsupervised, clustering (probabilistic/soft) — also a generative density model
- **What it does**: Models the data as a mixture of k Gaussian distributions, assigning each point a probability of belonging to each cluster (soft assignment) rather than a hard label
- **Core equation**: `P(x) = Σₖ πₖ·N(x; μₖ, Σₖ)`, fit via Expectation-Maximization (EM): E-step computes cluster-membership probabilities given current parameters, M-step updates `(π, μ, Σ)` given those probabilities
- **Key theory**: Generalizes K-Means — K-Means is essentially GMM with hard assignment and equal, spherical, fixed-size covariance; allowing full covariance matrices lets GMM fit elliptical, differently-oriented clusters
- **Differs from**: K-Means gives hard cluster assignment and assumes spherical clusters of similar size; GMM gives soft/probabilistic assignment and can fit elliptical clusters of varying size/orientation
- **Key hyperparameters**: Number of components (k), covariance type (`full`, `diagonal`, `tied`, `spherical`)
- **Preconditions/assumptions**: Data is plausibly generated by a mixture of Gaussians
- **Preprocessing needed**: Scaling recommended
- **Evaluation/output checks**: Log-likelihood on held-out data, BIC/AIC for choosing number of components (penalize model complexity, unlike raw likelihood which always improves with more components)
- **When to choose**: Need soft cluster assignments, expect elliptical/differently-shaped clusters, or want a proper generative/density model (e.g. for anomaly detection via low-likelihood points)
- **Works best with**: Continuous data plausibly multi-modal Gaussian
- **Key terms**: EM algorithm, soft assignment, BIC/AIC, covariance type
- **Gotcha**: EM converges to a local optimum — sensitive to initialization; can also collapse a component onto a single point (singular covariance) if not regularized

### 25. Mean Shift
- **Type**: Unsupervised, clustering
- **What it does**: Iteratively shifts each point toward the mode (peak) of the density in its neighborhood, and points converging to the same mode form a cluster
- **Core equation**: Repeatedly recompute a kernel-weighted mean of points within a bandwidth `h` around the current estimate, and shift toward it, until convergence to a local density maximum
- **Key theory**: A mode-seeking algorithm — clusters are defined by the local maxima ("peaks") of the estimated data density, found via kernel density estimation
- **Differs from**: K-Means requires k specified; Mean Shift automatically determines the number of clusters from the data's density structure, controlled instead by the bandwidth parameter
- **Key hyperparameters**: `bandwidth` (kernel window size — the single most important tuning knob, analogous to `eps` in DBSCAN)
- **Preconditions/assumptions**: Assumes clusters correspond to density modes
- **Preprocessing needed**: Scaling recommended
- **Evaluation/output checks**: Silhouette score; sensitivity check across a range of bandwidths
- **When to choose**: Don't know k, expect arbitrarily-shaped clusters defined by density peaks (common in image segmentation/computer vision)
- **Works best with**: Lower-dimensional data (computationally expensive in high dimensions) — a classic use case is image segmentation
- **Key terms**: Kernel density estimation, bandwidth, mode-seeking
- **Gotcha**: Computationally expensive (each point requires iterative shifting) — doesn't scale well to large, high-dimensional datasets; bandwidth choice is as sensitive as `k` in K-Means or `eps` in DBSCAN, just relabeled

---

# E. Dimensionality Reduction

### 26. Principal Component Analysis (PCA)
- **Type**: Unsupervised, dimensionality reduction / feature extraction
- **What it does**: Finds orthogonal directions (principal components) that capture maximum variance in the data, and projects data onto the top few
- **Core equation**: Eigen-decomposition (or SVD) of the covariance matrix `Σ = (1/n)XᵀX`; principal components are the eigenvectors, ranked by eigenvalue (variance explained)
- **Key theory**: The first principal component is the direction of maximum variance; each subsequent one is the direction of maximum remaining variance, orthogonal to all previous ones
- **Differs from**: LDA finds directions that maximize *class separability* (supervised); PCA finds directions of maximum *variance* (unsupervised) — the two can point in very different directions if the direction of max variance isn't the direction that separates classes
- **Key hyperparameters**: Number of components to keep (or variance-explained threshold, e.g. keep components explaining 95% of variance)
- **Preconditions/assumptions**: Assumes the directions of largest variance are the most informative (not always true — can be violated if the signal of interest is low-variance)
- **Preprocessing needed**: **Standardization is mandatory** — PCA is scale-sensitive; a feature with a larger raw scale would otherwise dominate the variance calculation regardless of actual importance
- **Evaluation/output checks**: Scree plot / cumulative explained-variance-ratio plot to choose number of components; reconstruction error if used for compression
- **When to choose**: Need to reduce dimensionality for visualization, speed up downstream models, or remove multicollinearity, and interpretability of the reduced axes isn't critical
- **Works best with**: Numeric, correlated, roughly-linear-structured high-dimensional data
- **Key terms**: Eigenvector/eigenvalue, explained variance ratio, scree plot, whitening
- **Gotcha**: Principal components are linear combinations of *all* original features and generally aren't individually interpretable — don't confuse "top component" with "top original feature"

### 27. Singular Value Decomposition (SVD) / Truncated SVD
- **Type**: Unsupervised, dimensionality reduction / matrix factorization
- **What it does**: Decomposes any matrix into `X = UΣVᵀ`; truncated SVD keeps only the top singular values/vectors for a low-rank approximation
- **Core equation**: `X ≈ UₖΣₖVₖᵀ` using the top k singular values
- **Key theory**: The same underlying math as PCA when applied to a mean-centered matrix — PCA is essentially SVD on centered data — but SVD works directly on non-centered and even sparse matrices without needing to compute a covariance matrix explicitly
- **Differs from**: PCA is typically framed via the covariance matrix and requires centering; Truncated SVD (as implemented in most ML libraries) works directly on the data matrix, including sparse matrices, which makes it the standard choice for text (TF-IDF matrices) where centering would destroy sparsity
- **Key hyperparameters**: Number of components (k)
- **Preconditions/assumptions**: None strict; works on sparse matrices unlike standard PCA
- **Preprocessing needed**: Doesn't require centering (unlike PCA) — this is precisely why it's preferred for sparse data
- **Evaluation/output checks**: Explained variance ratio, reconstruction error
- **When to choose**: Sparse, high-dimensional data (text/TF-IDF, recommender-system matrices) where you want PCA-like compression without densifying the matrix
- **Works best with**: Sparse matrices — text data, user-item interaction matrices (also the basis of classic matrix-factorization recommender systems)
- **Key terms**: Singular values, low-rank approximation, latent semantic analysis (LSA = SVD applied to text)
- **Gotcha**: Often used interchangeably with PCA in casual conversation, but knowing the sparse-matrix distinction is the detail that signals real understanding in an interview

### 28. t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Type**: Unsupervised, non-linear dimensionality reduction (primarily for visualization)
- **What it does**: Maps high-dimensional points to 2D/3D such that similar points (in the original space) stay close together in the embedding, focusing on preserving *local* neighborhood structure
- **Core equation**: Converts pairwise distances to similarity probabilities in both high-dimensional space (Gaussian) and low-dimensional space (Student's t-distribution, which has heavier tails — this is the "t" in t-SNE), then minimizes the KL-divergence between the two probability distributions via gradient descent
- **Key theory**: The heavy-tailed t-distribution in the low-dimensional space specifically counteracts the "crowding problem" (in low dimensions, there isn't enough room to represent all the moderate distances faithfully) — this is why t-SNE plots often show well-separated, visually distinct blobs
- **Differs from**: PCA is linear and preserves global variance structure; t-SNE is non-linear and preserves local neighborhood structure, at the cost of *not* preserving global distances or cluster sizes/densities meaningfully
- **Key hyperparameters**: `perplexity` (roughly, the effective number of neighbors considered — typically 5-50), learning rate, number of iterations
- **Preconditions/assumptions**: Intended for visualization, not as a general-purpose feature-reduction step feeding another model
- **Preprocessing needed**: Scaling recommended; often applied *after* an initial PCA reduction (e.g. to 50 dims) for speed on very high-dimensional input
- **Evaluation/output checks**: Run with multiple `perplexity` values and random seeds — t-SNE is stochastic and sensitive to both
- **When to choose**: Visualizing high-dimensional embeddings/clusters (e.g. inspecting learned representations) — not for feeding into a downstream predictive model
- **Works best with**: Exploratory visualization of embeddings, image features, or any high-dimensional representation you want to eyeball
- **Key terms**: Perplexity, KL-divergence, crowding problem, stochastic embedding
- **Gotcha**: **Inter-cluster distances and cluster sizes in a t-SNE plot are not meaningful** — a common and serious misinterpretation is reading relative distances between visually separated blobs as if they reflected real similarity; t-SNE only reliably preserves *local* neighbor relationships, not global geometry

### 29. UMAP (Uniform Manifold Approximation and Projection)
- **Type**: Unsupervised, non-linear dimensionality reduction
- **What it does**: Similar goal to t-SNE (non-linear embedding preserving local structure), but built on manifold-learning/topological theory, and generally faster and better at preserving more global structure
- **Core equation**: Constructs a weighted graph representing the data's topology (via fuzzy simplicial sets) in high-dimensional space, then optimizes a low-dimensional layout to have as similar a graph structure as possible (via cross-entropy minimization between the two graphs' edge-weight distributions)
- **Key theory**: Grounded in Riemannian geometry and algebraic topology (approximating the data as sampled from a manifold) — in practice, the main practical differences from t-SNE are speed (scales better to large n), better preservation of some global structure, and the ability to transform *new* unseen points after fitting (t-SNE has no natural out-of-sample extension)
- **Differs from**: t-SNE focuses purely on local structure and doesn't support transforming new points after fitting; UMAP is faster, scales better, and can embed new points via a learned transform
- **Key hyperparameters**: `n_neighbors` (local vs. global structure trade-off, similar role to t-SNE's perplexity), `min_dist` (how tightly points are packed in the embedding)
- **Preconditions/assumptions**: Same manifold-based assumptions as most non-linear embedding methods
- **Preprocessing needed**: Scaling recommended
- **Evaluation/output checks**: Same caution as t-SNE — visually inspect at multiple `n_neighbors`/`min_dist` settings; still shouldn't over-trust inter-cluster distances, though UMAP preserves more global structure than t-SNE
- **When to choose**: Similar use case to t-SNE but with a larger dataset, need for speed, or need to embed new/unseen points later
- **Works best with**: Large-scale embedding visualization, and increasingly as a general-purpose non-linear dimensionality-reduction preprocessing step (unlike t-SNE, which is visualization-only in practice)
- **Key terms**: Fuzzy simplicial set, manifold learning, `n_neighbors`, `min_dist`
- **Gotcha**: Widely considered the modern default over t-SNE for most practical purposes, but the same "don't over-interpret exact distances/densities" caution still applies, just somewhat less severely

---

# F. Other Important Algorithms

### 30. Apriori (Association Rule Mining)
- **Type**: Unsupervised, association rule learning
- **What it does**: Finds frequent itemsets in transactional data and derives "if X then Y" rules (e.g. market-basket analysis — "customers who buy bread and butter also buy milk")
- **Core equation**: Support `= P(X∩Y)`, Confidence `= P(Y|X) = P(X∩Y)/P(X)`, Lift `= P(X∩Y)/(P(X)P(Y))` — lift > 1 means X and Y co-occur more than chance
- **Key theory**: The "apriori principle" — if an itemset is frequent, all its subsets must also be frequent (and vice versa: if a subset is infrequent, no superset can be frequent) — this prunes the search space dramatically instead of checking every possible itemset combination
- **Differs from**: Not a predictive model at all — it's a rule-discovery/pattern-mining method; FP-Growth is a faster alternative that avoids Apriori's repeated database scans by building a compressed tree structure
- **Key hyperparameters**: Minimum support threshold, minimum confidence threshold (sometimes minimum lift)
- **Preconditions/assumptions**: Transactional/basket-style binary "item present or not" data
- **Preprocessing needed**: Data needs to be in transaction/basket format (one-hot encoded item presence per transaction)
- **Evaluation/output checks**: Support, confidence, and lift of the generated rules — filter to rules with high lift, not just high confidence (high confidence alone can be misleading if Y is just generally popular)
- **When to choose**: Market-basket analysis, recommendation rule mining, any "what tends to co-occur" business question
- **Works best with**: Large transactional datasets (purchase baskets, clickstreams, co-purchase data)
- **Key terms**: Support, confidence, lift, frequent itemset, apriori principle
- **Gotcha**: High confidence doesn't imply a meaningful rule if the consequent item is just popular overall — always check lift, not confidence alone

### 31. ARIMA (AutoRegressive Integrated Moving Average)
- **Type**: Supervised (self-referential), time-series forecasting
- **What it does**: Models a time series as a combination of its own past values (AR), a differencing step to achieve stationarity (I), and past forecast errors (MA)
- **Core equation**: `ARIMA(p,d,q)`: AR(p) term `= Σφᵢyₜ₋ᵢ`, differencing of order d to remove trend, MA(q) term `= Σθⱼεₜ₋ⱼ`; combined: `y'ₜ = c + Σφᵢy'ₜ₋ᵢ + Σθⱼεₜ₋ⱼ + εₜ` where `y'` is the differenced series
- **Key theory**: Requires the series to be (or be transformed into, via differencing) stationary — a stationary series has constant mean/variance over time, which is what makes the AR/MA structure valid and stable to fit
- **Differs from**: Exponential smoothing (Holt-Winters) models trend/seasonality directly via smoothing weights rather than AR/MA terms; modern ML approaches (gradient-boosted trees with lag/rolling features, or sequence models) can incorporate exogenous features far more easily than classical ARIMA
- **Key hyperparameters**: `p` (AR order), `d` (differencing order), `q` (MA order) — often chosen via ACF/PACF plot inspection or automated search (`auto_arima`); `P,D,Q,s` extend to SARIMA for seasonality
- **Preconditions/assumptions**: Stationarity (after differencing) — check via the Augmented Dickey-Fuller (ADF) test; no structural breaks
- **Preprocessing needed**: Differencing to achieve stationarity; log-transform if variance grows with the level of the series
- **Evaluation/output checks**: ADF test p-value (stationarity), ACF/PACF plots (for choosing p, q), residual diagnostics (residuals should look like white noise — no remaining autocorrelation), out-of-sample forecast error (MAE/RMSE) using a **time-based** train/test split, never a random split
- **When to choose**: Univariate time series with a clear autoregressive structure and no need for extra exogenous features
- **Works best with**: Single time series (or with a handful of features via ARIMAX/SARIMAX) with stable underlying dynamics
- **Key terms**: Stationarity, differencing, ACF/PACF, white-noise residuals, SARIMA/SARIMAX
- **Gotcha**: Using a random train/test split (instead of a time-based split) on a time series leaks future information into training — this is one of the most common and most-checked mistakes in a time-series interview question (see [Example_Company/09](../Example_Company/09_Classical_ML_Statistics.md) Q8)

---

## Quick cross-reference: "which algorithm when"

| If the data/problem looks like... | Reach for |
|---|---|
| Continuous target, roughly linear, need interpretability | Linear Regression (+ Ridge/Lasso if many/correlated features) |
| Binary/multiclass target, need probabilities + interpretability | Logistic Regression |
| Tabular data, want the best raw accuracy | XGBoost / LightGBM / CatBoost |
| Need a fast, low-maintenance baseline classifier | Random Forest |
| High-dimensional sparse features (text) | Naive Bayes, linear SVM, or Truncated SVD + a classifier |
| Small dataset, non-linear boundary | KNN or kernel SVM |
| Unknown number of clusters, expect outliers | DBSCAN |
| Roughly spherical, known number of clusters | K-Means |
| Elliptical clusters, want soft/probabilistic membership | GMM |
| Need to visualize high-dimensional structure | UMAP (or t-SNE) |
| Need to compress/decorrelate numeric features for a downstream model | PCA |
| Market-basket / co-occurrence patterns | Apriori |
| Single time series, classical statistical approach | ARIMA/SARIMA |
