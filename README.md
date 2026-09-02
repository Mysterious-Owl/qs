# Interview Prep

Notes for Data Science / ML Engineer interviews — a company-specific prep track, ML/GenAI reference material, a bank of real asked questions, and a broader role-based question set.

Company names, contacts and personal details have been genericized; the target company appears throughout as **Example Company**.

---

## What's in here

### 📂 [`ML_Fundamentals/`](ML_Fundamentals/README.md) — theory, math and practice

| File | What's in it |
|---|---|
| [ML_Algorithms_Cheatsheet](ML_Fundamentals/ML_Algorithms_Cheatsheet.md) | 31 algorithms: equation, theory, assumptions, hyperparameters, preprocessing, when to choose, gotcha |
| [XGBoost_Trees_Deep_Dive](ML_Fundamentals/XGBoost_Trees_Deep_Dive.md) | Gain formula, hyperparameter effects, `predict` vs `predict_proba`, when XGBoost fails, tuning order, overfitting, missing values, regularization |
| [CatBoost_Deep_Dive](ML_Fundamentals/CatBoost_Deep_Dive.md) | Ordered target statistics, ordered boosting, oblivious trees, feature combinations, three-way library comparison |
| [Feature_Engineering_Selection](ML_Fundamentals/Feature_Engineering_Selection.md) | Data leakage (7 types), filter/wrapper/embedded selection, high-cardinality encoding, `n ≫ p` vs `p ≫ n` |
| [Plots_Visual_Diagnostics](ML_Fundamentals/Plots_Visual_Diagnostics.md) | 16 generated figures with how-to-read guidance: skew, QQ, box plots, bias-variance, learning curves, ROC vs PR, calibration, gains/lift, KS, residuals |
| [GenAI_Cheatsheet](ML_Fundamentals/GenAI_Cheatsheet.md) | Tokenization, inference params, prompting, vector DBs, ~20 RAG variants, LLM eval (RAGAS), fine-tuning, agents, model selection |
| [Transformers_Cheatsheet](ML_Fundamentals/Transformers_Cheatsheet.md) | Attention math and every "why", block anatomy, RoPE, training stages, LoRA/PEFT, KV cache, serving |
| [DL_Concepts_Cheatsheet](ML_Fundamentals/DL_Concepts_Cheatsheet.md) | DL fundamentals plus 14 architectures (CNN, ResNet, LSTM, VAE, GAN, diffusion, GNN, ViT, CLIP) |
| [Glossary](ML_Fundamentals/Glossary.md) | ~200 terms, plus a commonly-confused-pairs table |
| [ML_Training_Notebook](ML_Fundamentals/ML_Training_Notebook.ipynb) | Runnable: linear/logistic regression, K-Means, XGBoost with real diagnostics |

### 📂 [`Question_Bank/`](Question_Bank/README.md) — questions actually asked

[27 questions from two real DS/ML rounds](Question_Bank/README.md), verbatim and in order, each linked to its answer or answered inline. Heavily XGBoost, imbalance and credit-risk flavoured. The folder README maps every topic to where it's covered and lists the gaps only you can fill.

### 📂 Role-based question sets

Eight Q&A documents per role, from a public reference set — useful for breadth, and for seeing how the same underlying answer shifts depending on which role is asking. Each answer ends with a "what this question tests" section.

| Role | Focus | Start at |
|---|---|---|
| `AI_Engineer/` | Production systems, inference, deployment | [Context and Memory](AI_Engineer/Q1_Context_and_Memory.md) |
| `ML_Engineer/` | Training, optimization, pipelines | [Training Pipeline](ML_Engineer/Q1_Training_Pipeline.md) |
| `Data_Scientist/` | Analysis, experimentation, business impact | [Experimentation](Data_Scientist/Q1_Experimentation.md) |
| `AI_Researcher/` | Novel methods, theoretical foundations | [Evaluation Methodology](AI_Researcher/Q1_Evaluation_Methodology.md) |
| `AI_Architect/` | System design, multi-agent, infrastructure | [Multi-Agent Consistency](AI_Architect/Q1_Multi_Agent_Consistency.md) |

### 📂 [`Example_Company/`](Example_Company/README.md) — the company-specific track

Prep for a Data Scientist role on a **Catalog Data Science** team: duplicate product detection, compliance/content moderation, and product attribute extraction with content generation.

Fed by two sources: topic and format tips from a contact who interviewed on the team, and public interview-experience research (Glassdoor, AmbitionBox, Naukri, Indeed, LinkedIn, Reddit, LeetCode Discuss, 1point3acres, Blind, GeeksforGeeks and prep aggregators), plus the company's own published engineering material for team-specific grounding.

**Every claim carries a confidence tag** — 🟢 corroborated across sources, 🟡 echoed in a couple, 🔴 single unverified mention, 🔵 from the referral. Treat 🔴 as directional, not gospel. 13 topic files: DSA, SQL, distributed systems, fraud system design, catalog/compliance design, duplicate detection, content moderation, attribute extraction, classical ML, LLM fine-tuning/RAG, DL architectures, MLOps, and behavioral.

---

## Suggested order

1. **[`XGBoost_Trees_Deep_Dive`](ML_Fundamentals/XGBoost_Trees_Deep_Dive.md)** — 9 of the 27 real asked questions were on this one topic.
2. **[`Feature_Engineering_Selection`](ML_Fundamentals/Feature_Engineering_Selection.md)** and **[`Plots_Visual_Diagnostics`](ML_Fundamentals/Plots_Visual_Diagnostics.md)** — leakage, high-cardinality handling, and reading a plot out loud all came up directly.
3. Run the **[notebook](ML_Fundamentals/ML_Training_Notebook.ipynb)** — muscle memory for a live coding round.
4. **[`GenAI_Cheatsheet`](ML_Fundamentals/GenAI_Cheatsheet.md)** before any RAG/LLM/agents conversation, then **[`Transformers_Cheatsheet`](ML_Fundamentals/Transformers_Cheatsheet.md)** for depth.
5. **[`Example_Company/README.md`](Example_Company/README.md)** if you're prepping that specific loop — its confidence tags tell you where to spend time.

Keep [`Glossary.md`](ML_Fundamentals/Glossary.md) open as a lookup throughout.

---

## How to run it

Everything is plain markdown, readable straight from GitHub. For a better reading experience, run the local reader:

```bash
pip install markdown
python reader.py
```

`--port 9000` to change port, `--no-browser` to skip auto-opening, `/reload` to re-scan after adding files.

The local reader and the published site share one renderer, so they behave identically:

- **Sidebar** grouped by folder, each group collapsible with its document count. State is remembered per group, the group holding the current page always opens, and **Collapse all / Expand all** sits in the footer.
- **Per-page table of contents** with scroll-spy highlighting.
- **Full-text search** across every document — **Ctrl+K** (or `/`), arrow keys to move, Enter to open.
- **Selection toolbar** — select any text and a small toolbar appears with **Copy**, **ChatGPT** and **Claude**. The assistant buttons open a new tab pre-filled with "Explain this:" and your selection (truncated to keep the URL valid).
- **Light/dark theme**, remembered between visits.
- Working cross-links, inline notebooks, and wide tables/figures that scroll rather than shrink.

To build the same thing as static HTML:

```bash
python build_site.py --clean            # -> ./_site
python -m http.server -d _site 8080     # preview at localhost:8080
```

### Regenerating the figures

The 16 diagnostic plots are generated, not hand-drawn:

```bash
pip install matplotlib numpy scipy scikit-learn
python ML_Fundamentals/figures/generate_figures.py
```

Transparent backgrounds and a mid-grey ink that clears contrast requirements on both light and dark surfaces, so one SVG serves both themes.

---

## Publishing to GitHub Pages

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) builds and deploys on every push to `main`, so **there is no output folder to commit or point at**. The build output is gitignored; the workflow regenerates it in CI.

One-time setup: **Settings → Pages → Source: _GitHub Actions_**. (The workflow's `configure-pages` step tries to enable this itself on first run, so it may already be done.) The site then lives at `https://<user>.github.io/<repo>/`.

Every URL the builder writes is relative, so the site works at a domain root and under a project subpath equally.

<details>
<summary>Alternative: publish without CI</summary>

Build into a committed folder and serve it from the branch instead:

```bash
python build_site.py --clean --out docs
```

Remove `docs/` from [`.gitignore`](.gitignore), commit the folder, then set **Settings → Pages → Source: _Deploy from a branch_ → `main` → `/docs`**. Simpler, but the built HTML lives in your git history and you must remember to rebuild before every push.
</details>

### Leaving pages out of the published site

```bash
python build_site.py --clean --exclude "Example_Company/*" --exclude "*Glossary*"
```

`--exclude` takes any glob and is repeatable. Links to an excluded page are struck through rather than left pointing at a 404. Only assets a document actually links to are copied, so unreferenced files are never published.

> A Pages site on a public repo is public and search-indexable. Nothing here identifies a person or employer, but review anything you add before pushing.
