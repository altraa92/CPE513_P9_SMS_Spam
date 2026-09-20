"""
Aggregation, statistical testing and figures for the SMS spam comparison.

Reads results/fold_level_results.csv (written by run_experiments.py) and
produces every table and figure that appears in the report.

Usage:  python src/analyse_results.py
"""

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

PRIMARY = "macro_f1"
METRICS = ["macro_f1", "pr_auc", "mcc", "recall_at_99_precision",
           "balanced_accuracy", "roc_auc", "accuracy"]
ALGO_ORDER = ["MultinomialNB", "LogisticRegression", "LinearSVM", "kNN",
              "RandomForest"]
EXP_LABEL = {"E1": "E1 word 1-2 grams, full corpus",
             "E2": "E2 char_wb 2-5 grams, full corpus",
             "E3": "E3 word 1-2 grams, de-duplicated corpus"}


# --------------------------------------------------------------------------
def majority_baseline_rows(df):
    """Per-fold scores of the trivial always-ham predictor, computed from the
    same test folds, so it can be placed in the same table as the models."""
    ref = df[df.algorithm == ALGO_ORDER[0]].copy()
    out = []
    for _, r in ref.iterrows():
        pos, neg = r.tp + r.fn, r.tn + r.fp
        n = pos + neg
        prec_ham, rec_ham = neg / n, 1.0
        f1_ham = 2 * prec_ham * rec_ham / (prec_ham + rec_ham)
        out.append(dict(
            experiment=r.experiment, representation=r.representation,
            corpus=r.corpus, algorithm="MajorityBaseline(all-ham)",
            family="Baseline", repeat=r.repeat, fold=r.fold,
            macro_f1=f1_ham / 2, spam_f1=0.0, pr_auc=pos / n, roc_auc=0.5,
            mcc=0.0, balanced_accuracy=0.5, accuracy=neg / n,
            recall_at_99_precision=0.0, tn=neg, fp=0, fn=pos, tp=0,
            train_time_s=0.0, predict_ms_per_1000=0.0, n_configs=0,
            n_train=r.n_train, n_test=r.n_test))
    return pd.DataFrame(out)


def summarise(df):
    rows = []
    for (exp, algo), g in df.groupby(["experiment", "algorithm"], sort=False):
        row = dict(experiment=exp, algorithm=algo, family=g.family.iloc[0],
                   n_fold_scores=len(g))
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_sd"] = g[m].std(ddof=1)
        row["train_time_s_mean"] = g.train_time_s.mean()
        row["train_time_s_sd"] = g.train_time_s.std(ddof=1)
        row["predict_ms_per_1000_mean"] = g.predict_ms_per_1000.mean()
        row["n_configs"] = int(g.n_configs.iloc[0])
        rows.append(row)
    s = pd.DataFrame(rows)
    s["rank"] = s.groupby("experiment")[f"{PRIMARY}_mean"].rank(
        ascending=False, method="min").astype(int)
    return s.sort_values(["experiment", "rank"])


# --------------------------------------------------------------------------
def wilcoxon_pair(a, b):
    """Paired Wilcoxon signed-rank with a normal-approximation effect size
    r = |Z| / sqrt(N), N = number of non-zero differences."""
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return dict(W=np.nan, p=1.0, z=0.0, r=0.0, n_nonzero=0)
    stat, p = stats.wilcoxon(nz, zero_method="wilcox")
    ranks = stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sigma
    return dict(W=float(stat), p=float(p), z=float(z),
                r=float(abs(z) / np.sqrt(n)), n_nonzero=n)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def interpret(p, r):
    if p >= 0.05:
        return "within fold-to-fold noise"
    size = "trivial" if r < 0.1 else "small" if r < 0.3 else \
        "medium" if r < 0.5 else "large"
    return f"reliable, {size} effect"


def statistical_tests(df):
    rows = []
    for exp in ["E1", "E2", "E3"]:
        sub = df[(df.experiment == exp) & (df.algorithm.isin(ALGO_ORDER))]
        wide = sub.pivot_table(index=["repeat", "fold"], columns="algorithm",
                               values=PRIMARY)[ALGO_ORDER]
        fr_stat, fr_p = stats.friedmanchisquare(*[wide[a].values
                                                 for a in ALGO_ORDER])
        mean_ranks = wide.rank(axis=1, ascending=False).mean()
        rows.append(dict(
            experiment=exp, comparison="Friedman (all 5 algorithms)",
            test="Friedman", statistic=round(fr_stat, 3), p_raw=fr_p,
            correction="none", p_adjusted=fr_p, effect_size=np.nan,
            effect_type="mean ranks",
            interpretation="; ".join(f"{a}={mean_ranks[a]:.2f}"
                                     for a in ALGO_ORDER)))
        pairs = list(combinations(ALGO_ORDER, 2))
        res = [wilcoxon_pair(wide[a].values, wide[b].values) for a, b in pairs]
        adj = holm([r["p"] for r in res])
        for (a, b), r_, pa in zip(pairs, res, adj):
            rows.append(dict(
                experiment=exp, comparison=f"{a} vs {b}",
                test="Wilcoxon signed-rank", statistic=r_["W"], p_raw=r_["p"],
                correction="Holm (10 pairs)", p_adjusted=pa,
                effect_size=round(r_["r"], 3), effect_type="r = |Z|/sqrt(N)",
                interpretation=interpret(pa, r_["r"])))
    # word vs char, matched on identical folds
    for algo in ALGO_ORDER:
        w = df[(df.experiment == "E1") & (df.algorithm == algo)].sort_values(
            ["repeat", "fold"])[PRIMARY].values
        c = df[(df.experiment == "E2") & (df.algorithm == algo)].sort_values(
            ["repeat", "fold"])[PRIMARY].values
        r_ = wilcoxon_pair(w, c)
        rows.append(dict(
            experiment="E1 vs E2", comparison=f"{algo}: word vs char",
            test="Wilcoxon signed-rank", statistic=r_["W"], p_raw=r_["p"],
            correction="none (pre-specified, 5 tests)", p_adjusted=r_["p"],
            effect_size=round(r_["r"], 3), effect_type="r = |Z|/sqrt(N)",
            interpretation=f"mean diff (word - char) = {np.mean(w - c):+.4f}; "
                           + interpret(r_["p"], r_["r"])))
    out = pd.DataFrame(rows)
    out["p_raw"] = out.p_raw.map(lambda v: float(f"{v:.3g}"))
    out["p_adjusted"] = out.p_adjusted.map(lambda v: float(f"{v:.3g}"))
    return out


def dedup_comparison(summary):
    rows = []
    for algo in ALGO_ORDER + ["MajorityBaseline(all-ham)"]:
        e1 = summary[(summary.experiment == "E1") & (summary.algorithm == algo)]
        e3 = summary[(summary.experiment == "E3") & (summary.algorithm == algo)]
        if e1.empty or e3.empty:
            continue
        rows.append(dict(
            algorithm=algo,
            full_macro_f1=e1[f"{PRIMARY}_mean"].iloc[0],
            full_sd=e1[f"{PRIMARY}_sd"].iloc[0],
            dedup_macro_f1=e3[f"{PRIMARY}_mean"].iloc[0],
            dedup_sd=e3[f"{PRIMARY}_sd"].iloc[0],
            optimism=e1[f"{PRIMARY}_mean"].iloc[0] - e3[f"{PRIMARY}_mean"].iloc[0],
            full_pr_auc=e1["pr_auc_mean"].iloc[0],
            dedup_pr_auc=e3["pr_auc_mean"].iloc[0]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def figure_dispersion(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, exp in zip(axes, ["E1", "E2"]):
        sub = df[(df.experiment == exp) & (df.algorithm.isin(ALGO_ORDER))]
        data = [sub[sub.algorithm == a][PRIMARY].values for a in ALGO_ORDER]
        bp = ax.boxplot(data, labels=[a.replace("Regression", "Reg")
                                      for a in ALGO_ORDER],
                        widths=0.55, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe3f7")
            patch.set_edgecolor("#33648f")
        for i, d in enumerate(data, start=1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.055, len(d)), d,
                       s=13, color="#1b3a57", alpha=0.65, zorder=3)
        ax.set_title(EXP_LABEL[exp], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    axes[0].set_ylabel("Macro-F1 per fold")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_fold_dispersion.png", dpi=200)
    plt.close(fig)


def figure_confusion(df):
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1))
    sub = df[df.experiment == "E1"]
    for ax, algo in zip(axes, ALGO_ORDER):
        g = sub[sub.algorithm == algo]
        cm = np.array([[g.tn.sum(), g.fp.sum()], [g.fn.sum(), g.tp.sum()]],
                      dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}\n({norm[i, j]:.3f})",
                        ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred ham", "pred spam"], fontsize=7)
        ax.set_yticks([0, 1], ["ham", "spam"], fontsize=7)
        ax.set_title(algo, fontsize=8)
    fig.suptitle("Counts summed over 15 outer test folds (E1, word n-grams)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_confusion_matrices.png", dpi=200)
    plt.close(fig)


def figure_cost(summary):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    markers = {"E1": "o", "E2": "s"}
    colours = dict(zip(ALGO_ORDER,
                       ["#d1495b", "#00798c", "#edae49", "#66a182", "#2e4057"]))
    for exp in ["E1", "E2"]:
        s = summary[(summary.experiment == exp) &
                    (summary.algorithm.isin(ALGO_ORDER))]
        for _, r in s.iterrows():
            ax.errorbar(r.train_time_s_mean, r[f"{PRIMARY}_mean"],
                        yerr=r[f"{PRIMARY}_sd"], fmt=markers[exp],
                        color=colours[r.algorithm], markersize=8, capsize=3,
                        alpha=0.9)
            ax.annotate(r.algorithm, (r.train_time_s_mean, r[f"{PRIMARY}_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean tuning + fit time per outer fold (s, log scale, 1 CPU core)")
    ax.set_ylabel("Macro-F1 (mean +/- sd over 15 folds)")
    ax.set_title("Accuracy against cost: circles = word, squares = char",
                 fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_cost.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    df = pd.read_csv(RESULTS / "fold_level_results.csv")
    df = pd.concat([df, majority_baseline_rows(df)], ignore_index=True)
    df.to_csv(RESULTS / "fold_level_results_with_baseline.csv", index=False)

    summary = summarise(df)
    summary.to_csv(RESULTS / "summary_results.csv", index=False)
    tests = statistical_tests(df)
    tests.to_csv(RESULTS / "statistical_tests.csv", index=False)
    dedup = dedup_comparison(summary)
    dedup.to_csv(RESULTS / "duplicate_ablation.csv", index=False)

    figure_dispersion(df)
    figure_confusion(df)
    figure_cost(summary)

    print(summary[["experiment", "algorithm", "rank", "macro_f1_mean",
                   "macro_f1_sd", "pr_auc_mean", "mcc_mean",
                   "recall_at_99_precision_mean", "accuracy_mean",
                   "train_time_s_mean"]].round(4).to_string(index=False))
    print()
    print(tests.to_string(index=False))
    print()
    print(dedup.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
