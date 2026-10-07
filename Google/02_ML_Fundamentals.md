# ML Fundamentals

One of the two ML rounds in the reported loop 📄. The candidate says they went *"significantly deeper than I initially expected."*

---

## The topics named 📄

Verbatim from the post, grouped here by theme:

| Group | Topics |
|---|---|
| Supervised | Classification, Regression, Logistic regression, SVM |
| Trees & ensembles | Decision trees, Random forests, Gradient boosting |
| Unsupervised | Clustering, Dimensionality reduction |
| Data & features | Feature engineering, Imbalanced datasets |
| Fitting | Regularization, Bias/variance, Overfitting |
| Judging a model | Evaluation metrics, Model selection, Calibration |
| Applied | Ranking / recommendation concepts, Experimentation |

## The shape of the questions 📄

This is the part of the post worth quoting directly, because it changes how you prepare:

> The important part wasn't memorizing definitions. The questions often become:
> - *"Why would you choose X instead of Y?"*
> - *"What happens if this assumption doesn't hold?"*
> - *"How would you debug this?"*

Three question types, and each rewards a different kind of preparation. 🧩

**"Why X instead of Y"** is a comparison question, and comparisons are only crisp when you know each method's *failure mode*, not its feature list. "Random Forest reduces variance, gradient boosting reduces bias" is a real answer; "Random Forest is an ensemble of trees" is not.

**"What if the assumption doesn't hold"** is the one that separates people who learned methods from people who learned *conditions*. Every method has preconditions — linear regression's homoscedasticity, Naive Bayes' conditional independence, K-Means' spherical equal-density clusters — and the good answer names the assumption, says how you'd detect the violation, and says what breaks (often: the point predictions survive, the inference doesn't).

**"How would you debug this"** is the most production-flavoured and the hardest to fake. It wants a *procedure*: reproduce → isolate (data vs features vs label vs model vs serving) → check the simplest explanation first (leakage, a broken join, train/serve skew) → only then reach for model changes.

## Where the depth lives 🔗

Every topic on that list is already covered, mostly in more depth than the post itself offers:

| Topic from the post | Where |
|---|---|
| Classification, regression, logistic regression, SVM, clustering, dimensionality reduction | [`ML_Algorithms_Cheatsheet`](../ML_Fundamentals/ML_Algorithms_Cheatsheet.md) — 31 algorithms, each with assumptions, when-to-choose and its gotcha |
| Decision trees, random forests, gradient boosting | [`XGBoost_Trees_Deep_Dive`](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md) · [`CatBoost_Deep_Dive`](../ML_Fundamentals/CatBoost_Deep_Dive.md) |
| Bias/variance, overfitting | [`Plots_Visual_Diagnostics`](../ML_Fundamentals/Plots_Visual_Diagnostics.md) §5–6 — the decomposition and learning curves |
| Regularization | [`XGBoost_Trees_Deep_Dive`](../ML_Fundamentals/XGBoost_Trees_Deep_Dive.md) §11–12 — what it does behind the scenes, and beyond L1/L2 |
| Feature engineering | [`Feature_Engineering_Selection`](../ML_Fundamentals/Feature_Engineering_Selection.md) §4 |
| Imbalanced datasets | [`Question_Bank`](../Question_Bank/README.md) R1Q6 — the four levers |
| Evaluation metrics | [`Plots_Visual_Diagnostics`](../ML_Fundamentals/Plots_Visual_Diagnostics.md) §8–10 · [`Glossary`](../ML_Fundamentals/Glossary.md) §4 |
| **Calibration** | [`Plots_Visual_Diagnostics`](../ML_Fundamentals/Plots_Visual_Diagnostics.md) §11 — reliability diagrams, and why good AUC ≠ good calibration |
| Model selection | [`Glossary`](../ML_Fundamentals/Glossary.md) §3 — CV variants, nested CV |
| **Ranking / recommendation** | Partially: [`Plots_Visual_Diagnostics`](../ML_Fundamentals/Plots_Visual_Diagnostics.md) §12–13 (gains, lift, KS) covers ranking *evaluation* |
| **Experimentation** | [`../Data_Scientist/Q1_Experimentation.md`](../Data_Scientist/Q1_Experimentation.md) |

**Two gaps worth noting** 🧩 — the post names *ranking/recommendation concepts* and *experimentation*, and this repo covers ranking evaluation but not ranking *models* (learning-to-rank, pointwise vs pairwise vs listwise, NDCG/MAP, two-tower retrieval, candidate generation vs re-ranking). If you're targeting an ML role at a company whose core product is ranking, that's the gap to fill yourself.

## Calibration deserves a callout 🧩

It's the one item on the list that candidates most often can't discuss, and it's specifically named in the post. The short version: a model can rank perfectly and still output probabilities that mean nothing. AUC only cares about *order*; calibration cares about *level*. It matters the moment a probability is consumed by something other than a sort — expected-value decisions, thresholds tied to a cost ratio, or a number shown to a human. Fixed with Platt scaling or isotonic regression, fit on held-out data. And note that class weighting or resampling to fix imbalance systematically *decalibrates* the model — so those two topics from the list interact.

## The follow-up ladder, applied here 🧩

- *"Why gradient boosting over random forest here?"* → bias vs variance → *"what if the labels are noisy?"* → boosting chases noise, RF averages it away → *"so how would you tell if that's happening?"* → train/validation gap, learning curves, label audit
- *"How do you handle the imbalance?"* → class weights → *"what does that do to your probabilities?"* → decalibrates them → *"so how do you fix that?"* → recalibrate on held-out data
- *"Which metric?"* → PR-AUC → *"why not ROC-AUC?"* → the abundant negatives inflate it → *"what's your operating point and why?"* → the cost function
