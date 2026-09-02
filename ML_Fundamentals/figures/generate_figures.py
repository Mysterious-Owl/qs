#!/usr/bin/env python3
"""Generate the diagnostic-plot figures for Plots_Visual_Diagnostics.md.

    python generate_figures.py

Writes theme-neutral SVGs next to this script. Every figure uses a transparent
background and a mid-grey ink that clears 3:1 on both the light (#fbfbfa) and
dark (#17171a) reader surfaces, so one file serves both themes.

Palette: categorical slots 1-3 of the validated reference palette. This trio
passes lightness-band, chroma, all-pairs CVD (worst deutan dE 9.3) and
normal-vision (dE 24.0) checks in both modes; aqua sits below 3:1 on the light
surface, so anything drawn in aqua carries a direct label (the relief rule).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)

# --- palette -----------------------------------------------------------------
BLUE = "#2a78d6"   # series 1
ORANGE = "#d95926"  # series 2
AQUA = "#1baf7a"   # series 3 - always direct-labelled
INK = "#7f7e78"    # 3.93:1 light / 4.39:1 dark
GRID = "#8a8a85"

plt.rcParams.update({
    "svg.fonttype": "none",          # keep text as text, not paths
    "font.family": "sans-serif",
    "font.sans-serif": ["system-ui", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 9,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  wrote {name}")


def panel_title(ax, text):
    ax.set_title(text, pad=8, loc="left")


# =============================================================== distributions
def distribution_shapes():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
    right = RNG.gamma(2.2, 1.0, 6000)
    data = [
        ("Right-skewed (positive skew)", right,
         "mean > median > mode   ·   tail points right"),
        ("Symmetric (normal)", RNG.normal(0, 1, 6000),
         "mean = median = mode"),
        ("Left-skewed (negative skew)", -RNG.gamma(2.2, 1.0, 6000),
         "mean < median < mode   ·   tail points left"),
    ]
    for ax, (title, x, note) in zip(axes, data):
        lo, hi = np.percentile(x, [0.2, 99.5])
        counts, bins, _ = ax.hist(x, bins=60, range=(lo, hi), color=BLUE,
                                  alpha=0.5, edgecolor="none")
        mode = bins[np.argmax(counts)] + (bins[1] - bins[0]) / 2
        # Reference lines get a legend, not inline labels - at this scale the three
        # values sit too close together for direct labels not to collide.
        ax.axvline(np.mean(x), color=ORANGE, linestyle="-", linewidth=2.0, label="mean")
        ax.axvline(np.median(x), color=ORANGE, linestyle="--", linewidth=1.8, label="median")
        ax.axvline(mode, color=INK, linestyle=":", linewidth=1.8, label="mode")
        panel_title(ax, title)
        ax.set_yticks([])
        ax.set_xlabel("value")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, counts.max() * 1.22)
        ax.legend(loc="upper left" if "Left-skewed" in title else "upper right",
                  fontsize=7.5)
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center",
                fontsize=7.5, color=INK)
        grid(ax, axis="x")
    save(fig, "distribution-shapes.svg")


def skew_transform():
    x = RNG.lognormal(2.2, 0.8, 4000)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axes[0].hist(x, bins=50, color=ORANGE, alpha=0.6, edgecolor="none")
    panel_title(axes[0], "Raw: right-skewed, long tail")
    axes[0].set_xlabel("income")
    axes[1].hist(np.log1p(x), bins=50, color=BLUE, alpha=0.6, edgecolor="none")
    panel_title(axes[1], "After log1p: near-symmetric")
    axes[1].set_xlabel("log(1 + income)")
    for ax in axes:
        ax.set_yticks([])
        grid(ax, axis="x")
    save(fig, "skew-transform.svg")


def qq_plots():
    from scipy import stats
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))
    samples = {
        "Normal\n(points on the line)": RNG.normal(0, 1, 260),
        "Right-skewed\n(upward-curving)": RNG.lognormal(0, 0.7, 260),
        "Left-skewed\n(downward-curving)": -RNG.lognormal(0, 0.7, 260),
        "Heavy-tailed\n(S-shape, ends flare)": RNG.standard_t(2.5, 260),
    }
    for ax, (title, s) in zip(axes, samples.items()):
        (osm, osr), (slope, inter, _) = stats.probplot(s, dist="norm")
        ax.plot(osm, osr, "o", color=BLUE, markersize=3.2, alpha=0.6,
                markeredgewidth=0)
        ax.plot(osm, slope * osm + inter, color=ORANGE, linewidth=1.8,
                label="normal reference")
        panel_title(ax, title)
        ax.set_xlabel("theoretical quantiles")
        if ax is axes[0]:
            ax.set_ylabel("sample quantiles")
            ax.legend(loc="upper left", fontsize=7)
        grid(ax)
    save(fig, "qq-plots.svg")


def boxplot_anatomy():
    x = np.concatenate([RNG.normal(50, 9, 400), [88, 92, 95, 12, 8]])
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.0),
                             gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.45})
    ax = axes[0]
    bp = ax.boxplot(x, vert=False, widths=0.5, patch_artist=True,
                    flierprops=dict(marker="o", markersize=5, markerfacecolor=ORANGE,
                                    markeredgecolor="none", alpha=0.9))
    bp["boxes"][0].set(facecolor=BLUE, alpha=0.4, edgecolor=INK, linewidth=1.0)
    for part in ("whiskers", "caps", "medians"):
        for a in bp[part]:
            a.set(color=INK, linewidth=1.4)
    bp["medians"][0].set(color=ORANGE, linewidth=2.2)
    # The fences are NOT the whisker ends: whiskers stop at the last real data
    # point inside the fence. Draw both so the distinction is visible.
    for f in (lo, hi):
        ax.axvline(f, color=ORANGE, linestyle=":", linewidth=1.3, alpha=0.9)
    for val, lab, y in [(lo, "lower fence\nQ1 - 1.5xIQR", 1.46),
                        (q1, "Q1", 1.30), (med, "median", 1.30), (q3, "Q3", 1.30),
                        (hi, "upper fence\nQ3 + 1.5xIQR", 1.46)]:
        ax.annotate(lab, xy=(val, 1.20), xytext=(val, y), fontsize=7.2,
                    ha="center", va="center", color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.7))
    ax.annotate("outliers (beyond the fence)", xy=(x.max(), 1.0),
                xytext=(x.max() - 2, 0.60), fontsize=7.2, color=ORANGE, ha="right",
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=0.8))
    ax.annotate("whisker ends at the last\npoint inside the fence", xy=(x[x >= lo].min(), 0.98),
                xytext=(q1 - 1, 0.56), fontsize=7.2, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.8))
    ax.set_yticks([])
    ax.set_ylim(0.40, 1.68)
    panel_title(ax, "Box plot anatomy - the 1.5xIQR outlier rule")
    grid(ax, axis="x")

    ax2 = axes[1]
    counts, _, _ = ax2.hist(x, bins=45, color=BLUE, alpha=0.5, edgecolor="none")
    ax2.axvspan(q1, q3, color=BLUE, alpha=0.16)
    for val, style in [(q1, "--"), (med, "-"), (q3, "--")]:
        ax2.axvline(val, color=ORANGE, linestyle=style, linewidth=1.6)
    ax2.set_ylim(0, counts.max() * 1.42)
    ax2.annotate("IQR = middle 50%", xy=((q1 + q3) / 2, counts.max() * 1.02),
                 xytext=((q1 + q3) / 2, counts.max() * 1.32), ha="center",
                 fontsize=7.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.7))
    panel_title(ax2, "Same data as a histogram - the box plot hides the shape")
    ax2.set_yticks([])
    ax2.set_xlabel("value")
    grid(ax2, axis="x")
    ax2.set_xlim(ax.get_xlim())
    save(fig, "boxplot-anatomy.svg")


# ================================================== accuracy vs precision
def accuracy_vs_precision():
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.2))
    cases = [
        ("High accuracy\nHigh precision", (0.00, 0.00), 0.10),
        ("Low accuracy\nHigh precision", (0.52, 0.34), 0.10),
        ("High accuracy\nLow precision", (0.00, 0.00), 0.42),
        ("Low accuracy\nLow precision", (0.44, 0.30), 0.42),
    ]
    for ax, (title, (bx, by), spread) in zip(axes, cases):
        for r, a in [(1.0, 0.10), (0.66, 0.13), (0.33, 0.18)]:
            ax.add_patch(Circle((0, 0), r, facecolor=INK, alpha=a, edgecolor="none"))
            ax.add_patch(Circle((0, 0), r, facecolor="none", edgecolor=INK,
                                linewidth=0.7, alpha=0.6))
        ang = np.linspace(0, 2 * np.pi, 9, endpoint=False) + RNG.uniform(0, 1, 9)
        rad = np.abs(RNG.normal(0, spread, 9))
        px, py = bx + rad * np.cos(ang), by + rad * np.sin(ang)
        # Shot group first, bullseye cross on top so neither hides the other.
        ax.plot(px, py, "o", color=BLUE, markersize=6.5, markeredgewidth=1.1,
                markeredgecolor="#fbfbfa", alpha=0.95, zorder=3)
        ax.plot(0, 0, "+", color=ORANGE, markersize=10, markeredgewidth=1.6, zorder=4)
        panel_title(ax, title)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, -0.02,
             "accuracy = closeness to the target (bias)   ·   "
             "precision = closeness to each other (variance)",
             ha="center", fontsize=8, color=INK)
    save(fig, "accuracy-vs-precision.svg")


# ====================================================== bias-variance / curves
def bias_variance():
    c = np.linspace(1, 12, 400)
    bias2 = 4.2 * np.exp(-0.42 * c) + 0.12
    var = 0.030 * c ** 1.75
    noise = 0.55
    test = bias2 + var + noise
    best = c[np.argmin(test)]
    top = 4.6

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(c, bias2, color=AQUA, linewidth=1.8, linestyle="--")
    ax.plot(c, var, color=ORANGE, linewidth=1.8, linestyle="-.")
    ax.plot(c, test, color=BLUE, linewidth=2.5)
    ax.axhline(noise, color=INK, linewidth=1.0, linestyle=":", alpha=0.8)

    # Direct labels placed off the curves. Aqua carries one by the relief rule.
    ax.text(2.9, bias2[np.argmin(np.abs(c - 2.9))] + 0.30, "bias$^2$",
            color=AQUA, fontsize=9, weight="bold")
    ax.text(9.4, var[np.argmin(np.abs(c - 9.4))] + 0.26, "variance",
            color=ORANGE, fontsize=9, weight="bold")
    ax.text(4.6, test[np.argmin(np.abs(c - 4.6))] + 0.34,
            "total test error", color=BLUE, fontsize=9, weight="bold")
    ax.text(11.85, noise + 0.14, "irreducible noise", color=INK, fontsize=7.5, ha="right")

    ax.axvline(best, color=INK, linewidth=0.9, alpha=0.65)
    ax.annotate("optimal complexity", xy=(best, test.min()),
                xytext=(best + 1.1, test.min() + 1.15), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    ax.text(best - 0.3, top * 0.93, "underfitting", ha="right", fontsize=8.5,
            color=INK, alpha=0.9)
    ax.text(best + 0.3, top * 0.93, "overfitting", ha="left", fontsize=8.5,
            color=INK, alpha=0.9)

    ax.set_xlabel("model complexity  (depth, parameters, features)")
    ax.set_ylabel("expected error")
    ax.set_ylim(0, top)
    ax.set_xlim(1, 12)
    ax.set_xticks([])
    ax.set_yticks([])
    grid(ax)
    save(fig, "bias-variance.svg")


def learning_curves():
    n = np.linspace(50, 1000, 120)
    d = np.exp(-(n - 50) / 260)          # decays 1 -> ~0 as n grows
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    # Training error RISES toward the asymptote (harder to fit more data) while
    # validation error FALLS toward it - drawing training as monotonically
    # falling is the classic way these plots get taught wrong.
    setups = [
        ("Underfitting (high bias)",
         0.30 - 0.05 * d, 0.34 + 0.12 * d,
         "both errors high and already converged\n-> more data will not help"),
        ("Good fit",
         0.09 - 0.05 * d, 0.13 + 0.26 * d,
         "small final gap, both low\n-> healthy"),
        ("Overfitting (high variance)",
         0.015 + 0.005 * d, 0.27 + 0.24 * d,
         "wide gap that persists\n-> regularize, or add data"),
    ]
    for ax, (title, train, val, note) in zip(axes, setups):
        ax.plot(n, train, color=BLUE, linewidth=2.2, label="training error")
        ax.plot(n, val, color=ORANGE, linewidth=2.2, label="validation error")
        ax.fill_between(n, train, val, color=ORANGE, alpha=0.10)
        ax.annotate("", xy=(1000, train[-1]), xytext=(1000, val[-1]),
                    arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.8))
        ax.text(975, (train[-1] + val[-1]) / 2, "gap ", ha="right", va="center",
                fontsize=7.2, color=INK)
        panel_title(ax, title)
        ax.set_xlabel("training set size")
        ax.set_ylim(0, 0.50)
        ax.set_xlim(50, 1080)
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=7.2, color=INK)
        if ax is axes[0]:
            ax.set_ylabel("error")
            ax.legend(loc="upper right", fontsize=7.5)
        grid(ax)
    save(fig, "learning-curves.svg")


# ============================================================ classification
def confusion_matrix_fig():
    tp, fp, fn, tn = 80, 30, 20, 870
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                  gridspec_kw={"width_ratios": [1, 1.05]})
    # Layout: columns = PREDICTED (+ left, - right), rows = ACTUAL (+ top, - bottom).
    # cx/cy are cell origins in data coords, y increasing upward.
    cells = [
        (0, 1, "TP", tp, "actual +, predicted +", BLUE),
        (1, 1, "FN", fn, "actual +, predicted -", ORANGE),
        (0, 0, "FP", fp, "actual -, predicted +", ORANGE),
        (1, 0, "TN", tn, "actual -, predicted -", BLUE),
    ]
    for cx, cy, lab, val, desc, col in cells:
        ax.add_patch(Rectangle((cx + 0.02, cy + 0.02), 0.96, 0.96,
                               facecolor=col, alpha=0.16, edgecolor=col, linewidth=1.1))
        ax.text(cx + 0.5, cy + 0.62, f"{lab} = {val}", ha="center", fontsize=11.5,
                weight="bold", color=INK)
        ax.text(cx + 0.5, cy + 0.32, desc, ha="center", fontsize=6.8, color=INK)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["predicted  +", "predicted  -"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["actual  -", "actual  +"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    panel_title(ax, "Confusion matrix  (1000 cases, 100 positives)")

    ax2.axis("off")
    total = tp + fp + fn + tn
    rows = [
        ("Accuracy", "(TP+TN) / all", (tp + tn) / total,
         "misleading here - 90% by predicting all-negative"),
        ("Precision", "TP / (TP+FP)", tp / (tp + fp),
         "of what I flagged, how much was real"),
        ("Recall (TPR)", "TP / (TP+FN)", tp / (tp + fn),
         "of all real positives, how many caught"),
        ("F1", "2PR / (P+R)", 2 * (tp / (tp + fp)) * (tp / (tp + fn)) /
         ((tp / (tp + fp)) + (tp / (tp + fn))), "harmonic mean of P and R"),
        ("Specificity (TNR)", "TN / (TN+FP)", tn / (tn + fp), "of all real negatives, how many cleared"),
        ("FPR", "FP / (FP+TN)", fp / (fp + tn), "x-axis of the ROC curve"),
    ]
    y = 0.94
    for name, formula, val, note in rows:
        ax2.text(0.00, y, name, fontsize=9, weight="bold", color=INK)
        ax2.text(0.34, y, formula, fontsize=8, color=INK, family="monospace")
        ax2.text(0.66, y, f"{val:.2f}", fontsize=9, weight="bold",
                 color=ORANGE if name == "Accuracy" else BLUE)
        ax2.text(0.00, y - 0.072, note, fontsize=6.9, color=INK, alpha=0.9)
        y -= 0.163
    panel_title(ax2, "Metrics from the same four cells")
    save(fig, "confusion-matrix.svg")


def _imbalanced_scores(n=4000, pos_rate=0.02):
    n_pos = int(n * pos_rate)
    y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)])
    s = np.concatenate([RNG.normal(1.6, 1.0, n_pos), RNG.normal(0.0, 1.0, n - n_pos)])
    return y, s


def roc_vs_pr():
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    y, s = _imbalanced_scores()
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    k = max(1, len(fpr) // 400); fpr, tpr = fpr[::k], tpr[::k]
    j = max(1, len(prec) // 400); prec, rec = prec[::j], rec[::j]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.3)
    ax.fill_between(fpr, tpr, color=BLUE, alpha=0.10)
    ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(0.55, 0.42, "random", color=INK, fontsize=7.5, rotation=34, alpha=0.85)
    ax.text(0.42, 0.72, f"ROC-AUC = {auc(fpr, tpr):.2f}", color=BLUE,
            fontsize=9, weight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (recall)")
    panel_title(ax, "ROC - looks great")
    ax = axes[1]
    ax.plot(rec, prec, color=ORANGE, linewidth=2.3)
    ax.fill_between(rec, prec, color=ORANGE, alpha=0.10)
    base = y.mean()
    ax.axhline(base, color=INK, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(0.52, base + 0.03, f"baseline = prevalence = {base:.2f}",
            color=INK, fontsize=7.5)
    ax.text(0.34, 0.62, f"PR-AUC = {average_precision_score(y, s):.2f}",
            color=ORANGE, fontsize=9, weight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    panel_title(ax, "Precision-Recall - tells the truth")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        grid(ax)
    fig.text(0.5, -0.04, "Same model, same 2%-positive data. ROC-AUC is inflated by the "
             "870 easy negatives; PR-AUC is not.", ha="center", fontsize=7.8, color=INK)
    save(fig, "roc-vs-pr.svg")


def threshold_tradeoff():
    from sklearn.metrics import precision_recall_curve
    y, s = _imbalanced_scores(6000, 0.05)
    p = 1 / (1 + np.exp(-s))          # map margins to probabilities so the
    prec, rec, thr = precision_recall_curve(y, p)   # x-axis reads 0..1
    prec, rec = prec[:-1], rec[:-1]
    k = max(1, len(thr) // 500)
    prec, rec, thr = prec[::k], rec[::k], thr[::k]
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    best = thr[np.argmax(f1)]

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.plot(thr, prec, color=BLUE, linewidth=2.2, label="precision")
    ax.plot(thr, rec, color=ORANGE, linewidth=2.2, label="recall")
    ax.plot(thr, f1, color=AQUA, linewidth=2.0, linestyle="--")
    # Aqua needs a direct label (relief rule); anchor it away from the crossings.
    ax.annotate("F1", xy=(best, f1.max()), xytext=(best - 0.16, f1.max() + 0.17),
                color=AQUA, fontsize=9.5, weight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=AQUA, linewidth=0.9))
    ax.axvline(best, color=INK, linewidth=0.9, alpha=0.75)
    ax.annotate(f"max-F1 threshold = {best:.2f}", xy=(best, 0.04),
                xytext=(min(best + 0.30, 0.95), 0.20), fontsize=7.8, color=INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.8))
    ax.axvline(0.5, color=INK, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.annotate("default 0.5 cut\n(arbitrary here)", xy=(0.5, 0.86),
                xytext=(0.66, 0.94), fontsize=7.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.8))
    ax.set_xlabel("decision threshold on predicted probability")
    ax.set_ylabel("metric value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="center left", fontsize=8)
    grid(ax)
    save(fig, "threshold-tradeoff.svg")


def calibration_curve_fig():
    from sklearn.calibration import calibration_curve
    n = 6000
    p_true = RNG.uniform(0, 1, n)
    y = RNG.binomial(1, p_true)
    over = np.clip(p_true ** 0.55, 0, 1)          # over-confident
    under = np.clip(p_true ** 1.9, 0, 1)          # under-confident
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1.3, alpha=0.8)
    ax.text(0.62, 0.56, "perfectly calibrated", color=INK, fontsize=7.5, rotation=38)
    for probs, col, lab in [(p_true, BLUE, "well calibrated"),
                            (over, ORANGE, "over-confident"),
                            (under, AQUA, "under-confident")]:
        frac, mean_pred = calibration_curve(y, probs, n_bins=12, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=col, linewidth=2.0, markersize=5,
                markeredgewidth=0, label=lab)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction of positives")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=7.8)
    grid(ax)
    save(fig, "calibration-curve.svg")


# ================================================================== regression
def residual_plots():
    n = 260
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
    fitted = np.linspace(5, 60, n)
    cases = [
        ("Healthy: random scatter", RNG.normal(0, 3.2, n)),
        ("Heteroscedastic: funnel", RNG.normal(0, 1, n) * (0.7 + fitted * 0.11)),
        ("Non-linearity missed: curve", 0.021 * (fitted - 32) ** 2 - 6 + RNG.normal(0, 1.5, n)),
    ]
    for ax, (title, resid) in zip(axes, cases):
        ax.plot(fitted, resid, "o", color=BLUE, markersize=4.2, alpha=0.5,
                markeredgewidth=0)
        ax.axhline(0, color=ORANGE, linewidth=1.8)
        panel_title(ax, title)
        ax.set_xlabel("fitted value")
        if ax is axes[0]:
            ax.set_ylabel("residual")
        grid(ax)
    notes = ["assumptions hold",
             "variance grows with the fit\n-> transform y, or use weights",
             "systematic pattern left\n-> add terms or change model"]
    for ax, note in zip(axes, notes):
        ax.text(0.03, 0.05, note, transform=ax.transAxes, fontsize=7.2,
                color=INK, va="bottom")
    save(fig, "residual-plots.svg")


def correlation_traps():
    n = 120
    x1 = RNG.uniform(0, 10, n); y1 = 0.8 * x1 + RNG.normal(0, 2.2, n)
    x2 = np.linspace(0, 10, n); y2 = 0.35 * (x2 - 5) ** 2 + RNG.normal(0, 0.7, n)
    x3 = np.concatenate([RNG.uniform(0, 4, n - 6), [17, 17.5, 18, 18.5, 19, 19.5]])
    y3 = np.concatenate([RNG.uniform(0, 4, n - 6), [16, 17, 17.5, 18, 18.5, 19]])
    # A real Simpson's paradox needs group membership CORRELATED with x, so the
    # within-group slope and the pooled slope carry OPPOSITE signs.
    xa = RNG.uniform(0.5, 5.0, n // 2); ya = -0.62 * xa + 3.0 + RNG.normal(0, 0.45, n // 2)
    xb = RNG.uniform(5.5, 10.0, n // 2); yb = -0.62 * xb + 11.0 + RNG.normal(0, 0.45, n // 2)
    x4 = np.concatenate([xa, xb]); y4 = np.concatenate([ya, yb])

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3))
    for ax, (x, y, title) in zip(axes[:3], [
        (x1, y1, "Genuine linear trend"),
        (x2, y2, "Strong but non-linear"),
        (x3, y3, "Outlier-driven r"),
    ]):
        ax.plot(x, y, "o", color=BLUE, markersize=4.4, alpha=0.6, markeredgewidth=0)
        m, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, m * xs + b, color=ORANGE, linewidth=1.9)
        panel_title(ax, f"{title}\nPearson r = {np.corrcoef(x, y)[0, 1]:.2f}")
        ax.set_xticks([]); ax.set_yticks([])
        grid(ax)
    axes[1].text(0.5, -0.12, "r ~ 0, yet y is fully determined by x",
                 transform=axes[1].transAxes, ha="center", fontsize=7.2, color=INK)
    axes[2].text(0.5, -0.12, "6 points create the whole correlation",
                 transform=axes[2].transAxes, ha="center", fontsize=7.2, color=INK)

    ax = axes[3]
    rs = {}
    for xg, yg, col, lab in [(xa, ya, BLUE, "group A"), (xb, yb, AQUA, "group B")]:
        ax.plot(xg, yg, "o", color=col, markersize=4.4, alpha=0.75, markeredgewidth=0)
        m, b = np.polyfit(xg, yg, 1)
        xs = np.array([xg.min(), xg.max()])
        ax.plot(xs, m * xs + b, color=col, linewidth=1.8)
        rs[lab] = (np.corrcoef(xg, yg)[0, 1], col)
    m, b = np.polyfit(x4, y4, 1)
    xs = np.array([x4.min(), x4.max()])
    ax.plot(xs, m * xs + b, color=ORANGE, linewidth=2.2, linestyle="--")
    # One corner block instead of labels on the marks - the two groups sit on the
    # diagonal, so there is no collision-free spot beside either of them.
    ax.set_ylim(y4.min() - 0.4, y4.max() + 3.2)
    y0 = 0.96
    for lab, (r, col) in rs.items():
        ax.text(0.04, y0, f"{lab}:  r = {r:+.2f}", transform=ax.transAxes,
                color=col, fontsize=7.4, weight="bold", va="top")
        y0 -= 0.085
    ax.text(0.04, y0, "pooled (dashed):  r = "
            f"{np.corrcoef(x4, y4)[0, 1]:+.2f}", transform=ax.transAxes,
            color=ORANGE, fontsize=7.4, weight="bold", va="top")
    ax.text(0.5, -0.12, "each group trends down, the pooled fit trends up",
            transform=ax.transAxes, ha="center", fontsize=7.2, color=INK)
    panel_title(ax, f"Simpson's paradox\nPooled r = {np.corrcoef(x4, y4)[0, 1]:+.2f}")
    ax.set_xticks([]); ax.set_yticks([])
    grid(ax)

    fig.text(0.5, -0.06, "Pearson r summarizes only linear co-movement - always plot before trusting it.",
             ha="center", fontsize=7.8, color=INK)
    save(fig, "correlation-traps.svg")


# =================================================================== clustering
def elbow_silhouette():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
    X, _ = make_blobs(n_samples=700, centers=4, cluster_std=1.05, random_state=7)
    ks = range(2, 10)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax = axes[0]
    ax.plot(list(ks), inertia, "o-", color=BLUE, linewidth=2.1, markeredgewidth=0)
    ax.annotate("elbow", xy=(4, inertia[2]), xytext=(5.6, inertia[2] + 0.30 * (max(inertia) - min(inertia))),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    ax.set_xlabel("k"); ax.set_ylabel("inertia (within-cluster SS)")
    panel_title(ax, "Elbow method - judgement call")
    ax = axes[1]
    ax.plot(list(ks), sil, "o-", color=ORANGE, linewidth=2.1, markeredgewidth=0)
    kbest = list(ks)[int(np.argmax(sil))]
    ax.plot([kbest], [max(sil)], "o", color=ORANGE, markersize=11,
            markerfacecolor="none", markeredgewidth=1.8)
    ax.text(kbest + 0.25, max(sil), f"max at k={kbest}", fontsize=8, color=INK, va="center")
    ax.set_xlabel("k"); ax.set_ylabel("silhouette score")
    panel_title(ax, "Silhouette - an actual optimum")
    for ax in axes:
        grid(ax)
    save(fig, "elbow-silhouette.svg")


# ================================================================ ranking / risk
def gains_lift():
    y, s = _imbalanced_scores(5000, 0.06)
    order = np.argsort(-s)
    y_sorted = y[order]
    pct = np.arange(1, len(y) + 1) / len(y)
    cum_capture = np.cumsum(y_sorted) / y.sum()
    lift = cum_capture / pct

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    ax = axes[0]
    ax.plot(pct * 100, cum_capture * 100, color=BLUE, linewidth=2.3, label="model")
    ax.plot([0, 100], [0, 100], color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
    ax.text(58, 47, "random", color=INK, fontsize=7.5, rotation=34, alpha=0.85)
    for cut in (1, 10):
        cap = cum_capture[int(len(y) * cut / 100) - 1] * 100
        ax.plot([cut, cut], [0, cap], color=ORANGE, linewidth=1.2, linestyle=":")
        ax.plot([0, cut], [cap, cap], color=ORANGE, linewidth=1.2, linestyle=":")
        ax.plot([cut], [cap], "o", color=ORANGE, markersize=6, markeredgewidth=0)
        ax.text(cut + 2.5, cap, f"top {cut}% captures {cap:.0f}%", fontsize=7.4,
                color=ORANGE, va="center")
    ax.set_xlabel("% of population contacted (ranked by score)")
    ax.set_ylabel("% of all positives captured")
    panel_title(ax, "Cumulative gains")
    ax = axes[1]
    ax.plot(pct * 100, lift, color=ORANGE, linewidth=2.3)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
    top_lift = max(lift[5:]) * 1.18
    ax.annotate("lift = 1 (no better than random)", xy=(72, 1.0),
                xytext=(62, top_lift * 0.30), fontsize=7.4, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.8))
    ax.set_xlabel("% of population contacted")
    ax.set_ylabel("lift")
    ax.set_ylim(0, top_lift)
    panel_title(ax, "Lift curve")
    for ax in axes:
        grid(ax)
    fig.text(0.5, -0.05, "The right chart for a capacity-constrained problem: "
             "\"find the top 1% highest-risk accounts\".", ha="center",
             fontsize=7.8, color=INK)
    save(fig, "gains-lift.svg")


def ks_statistic():
    y, s = _imbalanced_scores(6000, 0.25)
    thr = np.linspace(s.min(), s.max(), 400)
    cdf_pos = [(s[y == 1] <= t).mean() for t in thr]
    cdf_neg = [(s[y == 0] <= t).mean() for t in thr]
    gap = np.abs(np.array(cdf_neg) - np.array(cdf_pos))
    i = int(np.argmax(gap))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax = axes[0]
    ax.hist(s[y == 0], bins=40, color=BLUE, alpha=0.5, edgecolor="none", label="negatives")
    ax.hist(s[y == 1], bins=40, color=ORANGE, alpha=0.55, edgecolor="none", label="positives")
    ax.set_yticks([]); ax.set_xlabel("model score")
    ax.legend(loc="upper right", fontsize=7.8)
    panel_title(ax, "Score distributions by class")
    grid(ax, axis="x")

    ax = axes[1]
    ax.plot(thr, cdf_neg, color=BLUE, linewidth=2.2, label="CDF negatives")
    ax.plot(thr, cdf_pos, color=ORANGE, linewidth=2.2, label="CDF positives")
    ax.plot([thr[i], thr[i]], [cdf_pos[i], cdf_neg[i]], color=INK, linewidth=1.8)
    ax.annotate(f"KS = {gap[i]:.2f}", xy=(thr[i], (cdf_pos[i] + cdf_neg[i]) / 2),
                xytext=(thr[i] + 0.55, (cdf_pos[i] + cdf_neg[i]) / 2 - 0.14),
                fontsize=8.5, color=INK, weight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    ax.set_xlabel("score threshold"); ax.set_ylabel("cumulative proportion")
    ax.legend(loc="upper left", fontsize=7.8)
    panel_title(ax, "KS = max vertical gap between CDFs")
    grid(ax)
    save(fig, "ks-statistic.svg")


def main():
    print("Generating figures ->", OUT)
    for fn in (distribution_shapes, skew_transform, qq_plots, boxplot_anatomy,
               accuracy_vs_precision, bias_variance, learning_curves,
               confusion_matrix_fig, roc_vs_pr, threshold_tradeoff,
               calibration_curve_fig, residual_plots, correlation_traps,
               elbow_silhouette, gains_lift, ks_statistic):
        fn()
    print("done.")


if __name__ == "__main__":
    main()
