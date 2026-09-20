"""
Dumps every number, table row, and derived quantity used in the report into
a single JSON file, read directly from the archived CSVs. The docx builder
(build_docx.js) does no computation of its own - it only lays out what is
here. This keeps the Word version numerically identical to report.pdf.

Usage:  python src/prepare_docx_data.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

summary = pd.read_csv(RESULTS / "summary_results.csv")
tests = pd.read_csv(RESULTS / "statistical_tests.csv")
dedup = pd.read_csv(RESULTS / "duplicate_ablation.csv")
audit = pd.read_csv(RESULTS / "data_audit.csv").set_index("quantity")["value"]
env = json.loads((RESULTS / "environment.json").read_text())
params = pd.read_csv(RESULTS / "selected_hyperparameters.csv")

ALGO_ORDER = ["MultinomialNB", "LogisticRegression", "LinearSVM", "kNN",
              "RandomForest"]
ALGO_ALL = ALGO_ORDER + ["MajorityBaseline(all-ham)"]
PRETTY = {"MultinomialNB": "Multinomial naive Bayes",
          "LogisticRegression": "Logistic regression",
          "LinearSVM": "Linear SVM",
          "kNN": "k-NN (cosine)",
          "RandomForest": "Random forest",
          "MajorityBaseline(all-ham)": "Majority baseline (all-ham)"}
PRIMARY = "macro_f1"


def val(exp, algo, metric):
    r = summary[(summary.experiment == exp) & (summary.algorithm == algo)]
    return float(r[metric].iloc[0])


def ranked(exp):
    s = summary[(summary.experiment == exp) &
                (summary.algorithm != "MajorityBaseline(all-ham)")]
    return s.sort_values("rank").algorithm.tolist()


def test_row(exp, comparison):
    r = tests[(tests.experiment == exp) & (tests.comparison == comparison)]
    if r.empty:
        a, b = comparison.split(" vs ")
        r = tests[(tests.experiment == exp) &
                  (tests.comparison == f"{b} vs {a}")]
    return r.iloc[0]


def main_table(exp):
    cols = ["macro_f1", "pr_auc", "mcc", "recall_at_99_precision",
            "balanced_accuracy", "accuracy"]
    best = {m: summary[(summary.experiment == exp) &
                       (summary.algorithm != "MajorityBaseline(all-ham)")
                       ][m + "_mean"].max() for m in cols}
    rows = []
    for algo in ALGO_ALL:
        r = summary[(summary.experiment == exp) & (summary.algorithm == algo)]
        if r.empty:
            continue
        r = r.iloc[0]
        cells = {}
        for m in cols:
            cells[m] = dict(
                text=f"{r[m + '_mean']:.4f} +/- {r[m + '_sd']:.4f}",
                bold=bool(algo != "MajorityBaseline(all-ham)" and
                         abs(r[m + "_mean"] - best[m]) < 1e-12))
        rows.append(dict(
            algorithm=PRETTY[algo], family=r["family"], metrics=cells,
            train_s=f"{r['train_time_s_mean']:.1f}",
            pred_ms=f"{r['predict_ms_per_1000_mean']:.1f}",
            rank=("-" if algo == "MajorityBaseline(all-ham)"
                  else str(int(r["rank"])))))
    return rows


def tests_table(exp_list):
    rows = []
    for _, r in tests[tests.experiment.isin(exp_list)].iterrows():
        es = "-" if pd.isna(r.effect_size) else f"r = {r.effect_size:.2f}"
        stat = "-" if pd.isna(r.statistic) else f"{r.statistic:.2f}"
        rows.append(dict(
            experiment=r.experiment, comparison=r.comparison, test=r.test,
            statistic=stat, p_raw=f"{r.p_raw:.3g}",
            p_adjusted=f"{r.p_adjusted:.3g}", effect_size=es,
            interpretation=r.interpretation))
    return rows


def tuning_table():
    spaces = {
        "MultinomialNB": "alpha in 0.005, 0.01, 0.03, 0.1, 0.3, 1, 3, 10",
        "LogisticRegression": ("C in 0.1, 1, 10, 100 x class_weight in "
                               "None, balanced"),
        "LinearSVM": "C in 0.01, 0.1, 1, 10 x class_weight in None, balanced",
        "kNN": ("n_neighbors in 1, 3, 5, 9 x weights in uniform, distance; "
                "cosine metric"),
        "RandomForest": ("n_estimators in 100, 200 x min_samples_leaf in "
                         "1, 3 x class_weight in None, balanced_subsample"),
    }
    rows = []
    for algo, space in spaces.items():
        modal = params[(params.experiment == "E1") &
                       (params.algorithm == algo)].best_params.mode().iloc[0]
        modal = modal.replace('"', "").replace("{", "").replace("}", "")
        rows.append(dict(algorithm=PRETTY[algo], space=space, configs="8",
                         modal=modal))
    return rows


def dedup_table():
    rows = []
    for _, r in dedup.iterrows():
        rows.append(dict(
            algorithm=PRETTY.get(r.algorithm, r.algorithm),
            full=f"{r.full_macro_f1:.4f} +/- {r.full_sd:.4f}",
            dedup=f"{r.dedup_macro_f1:.4f} +/- {r.dedup_sd:.4f}",
            optimism=f"{r.optimism:+.4f}",
            pr_full=f"{r.full_pr_auc:.4f}", pr_dedup=f"{r.dedup_pr_auc:.4f}"))
    return rows


def repr_table():
    rows = []
    for algo in ALGO_ORDER:
        r = tests[(tests.experiment == "E1 vs E2") &
                  (tests.comparison == f"{algo}: word vs char")].iloc[0]
        d = val("E1", algo, "macro_f1_mean") - val("E2", algo, "macro_f1_mean")
        rows.append(dict(
            algorithm=PRETTY[algo],
            word=f"{val('E1', algo, 'macro_f1_mean'):.4f} +/- "
                 f"{val('E1', algo, 'macro_f1_sd'):.4f}",
            char=f"{val('E2', algo, 'macro_f1_mean'):.4f} +/- "
                 f"{val('E2', algo, 'macro_f1_sd'):.4f}",
            diff=f"{d:+.4f}", p=f"{r.p_adjusted:.3g}",
            effect=f"{r.effect_size:.2f}",
            train=f"{val('E1', algo, 'train_time_s_mean'):.1f} / "
                  f"{val('E2', algo, 'train_time_s_mean'):.1f}"))
    return rows


def main():
    e1_rank = ranked("E1")
    e2_rank = ranked("E2")
    top, second, last = e1_rank[0], e1_rank[1], e1_rank[-1]
    fried_e1 = test_row("E1", "Friedman (all 5 algorithms)")
    fried_e2 = test_row("E2", "Friedman (all 5 algorithms)")
    svm_v_lr = test_row("E1", f"{top} vs {second}")
    nb_v_svm = test_row("E1", "MultinomialNB vs LinearSVM")
    gap = val("E1", top, "macro_f1_mean") - val("E1", second, "macro_f1_mean")
    spread_top = val("E1", top, "macro_f1_sd")
    dd_top = dedup[dedup.algorithm == top].iloc[0]
    char_gains = {a: val("E2", a, "macro_f1_mean") - val("E1", a, "macro_f1_mean")
                  for a in ALGO_ORDER}
    n_char_better = sum(1 for v in char_gains.values() if v > 0)
    rf_ratio = (val("E1", "RandomForest", "train_time_s_mean")
               / val("E1", "LinearSVM", "train_time_s_mean"))
    base_acc = val("E1", "MajorityBaseline(all-ham)", "accuracy_mean")
    base_f1 = val("E1", "MajorityBaseline(all-ham)", "macro_f1_mean")
    svm_r99 = val("E1", "LinearSVM", "recall_at_99_precision_mean")
    max_repr_p = tests[tests.experiment == "E1 vs E2"].p_adjusted.max()
    dedup_range = dedup[dedup.algorithm != "MajorityBaseline(all-ham)"]

    data = dict(
        env=env,
        audit={k: (int(v) if isinstance(v, (int,)) or
                   (isinstance(v, str) and v.lstrip("-").isdigit())
                   else v) for k, v in audit.to_dict().items()},
        top=PRETTY[top], second=PRETTY[second], last=PRETTY[last],
        top_key=top,
        e1_rank=[PRETTY[a] for a in e1_rank],
        e2_rank=[PRETTY[a] for a in e2_rank],
        gap=round(gap, 4), spread_top=round(spread_top, 4),
        base_acc=round(base_acc, 4), base_f1=round(base_f1, 4),
        svm_r99=round(svm_r99, 4),
        rf_ratio=round(rf_ratio, 1),
        dedup_optimism_top=round(float(dd_top.optimism), 4),
        char_gain_min=round(min(char_gains.values()), 4),
        char_gain_max=round(max(char_gains.values()), 4),
        char_gain_max_abs=round(max(abs(v) for v in char_gains.values()), 4),
        n_char_better=n_char_better,
        max_repr_p=round(float(max_repr_p), 4),
        dedup_min=round(float(dedup_range.optimism.min()), 4),
        dedup_max=round(float(dedup_range.optimism.max()), 4),
        fried_e1=dict(stat=round(float(fried_e1.statistic), 2),
                      p=f"{fried_e1.p_adjusted:.3g}",
                      ranks=fried_e1.interpretation),
        fried_e2=dict(stat=round(float(fried_e2.statistic), 2),
                      p=f"{fried_e2.p_adjusted:.3g}"),
        svm_v_lr=dict(W=f"{svm_v_lr.statistic:.0f}",
                     p=f"{svm_v_lr.p_adjusted:.3g}",
                     r=f"{svm_v_lr.effect_size:.2f}"),
        nb_v_svm=dict(W=f"{nb_v_svm.statistic:.0f}",
                     p=f"{nb_v_svm.p_adjusted:.3g}",
                     r=f"{nb_v_svm.effect_size:.2f}"),
        table3=tuning_table(),
        table4_e1=main_table("E1"),
        table5=tests_table(["E1"]),
        table6=repr_table(),
        table7=dedup_table(),
        rf_recall99_sd=round(val("E1", "RandomForest",
                                 "recall_at_99_precision_sd"), 4),
        knn_recall99_sd=round(val("E1", "kNN",
                                  "recall_at_99_precision_sd"), 4),
    )

    out = ROOT / "report" / "docx_data.json"
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
