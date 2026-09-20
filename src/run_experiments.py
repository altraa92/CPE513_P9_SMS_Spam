"""
Empirical comparison of learning algorithms for SMS spam filtering (Problem P9).

Runs a protocol-controlled comparison of five classifiers under an identical
feature-extraction pipeline, with nested cross-validation:

  outer: stratified 5-fold, 3 repeats (15 paired fold scores per algorithm)
  inner: stratified 3-fold grid search, 8 configurations per algorithm
         (identical budget for every algorithm), selected on macro-F1

Three experiment matrices are produced:
  E1  word 1-2 grams, full corpus          (main results)
  E2  character_wb 2-5 grams, full corpus  (required representation experiment)
  E3  word 1-2 grams, de-duplicated corpus (duplicate-handling ablation)

Every transformation, including the TF-IDF vectoriser, is fitted on the
training folds only. Nothing is fitted on, or selected using, held-out data.

Usage:  python src/run_experiments.py
Output: results/fold_level_results.csv, results/selected_hyperparameters.csv,
        results/data_audit.csv
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# Global protocol constants. Changing any of these changes every table.
# --------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
N_REPEATS = 3
INNER_SPLITS = 3
SELECTION_METRIC = "f1_macro"
TARGET_PRECISION = 0.99          # operating point a real filter would need

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SMSSpamCollection"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Data loading and audit
# --------------------------------------------------------------------------
def load_data():
    """Load the raw UCI file. quoting=3 (QUOTE_NONE) is required: several
    messages contain unbalanced double quotes that would otherwise swallow
    following lines."""
    df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8")
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def normalise(s):
    """Aggressive normalisation used only for near-duplicate detection."""
    return (s.str.lower()
             .str.replace(r"[^a-z0-9 ]", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def audit(df):
    norm = normalise(df["text"])
    rows = {
        "n_rows": len(df),
        "n_ham": int((df.y == 0).sum()),
        "n_spam": int((df.y == 1).sum()),
        "spam_share": round(float(df.y.mean()), 4),
        "majority_class_accuracy": round(float(1 - df.y.mean()), 4),
        "n_missing": int(df.isna().sum().sum()),
        "n_exact_duplicate_rows": int(df.duplicated(subset=["label", "text"]).sum()),
        "n_duplicate_texts": int(df.duplicated(subset=["text"]).sum()),
        "n_texts_with_conflicting_labels": int(
            (df.groupby("text")["label"].nunique() > 1).sum()),
        "n_near_duplicates_normalised": int(norm.duplicated().sum()),
        "mean_chars_ham": round(float(df.loc[df.y == 0, "text"].str.len().mean()), 1),
        "mean_chars_spam": round(float(df.loc[df.y == 1, "text"].str.len().mean()), 1),
        "pct_ham_containing_digit": round(
            float(df.loc[df.y == 0, "text"].str.contains(r"\d").mean()), 3),
        "pct_spam_containing_digit": round(
            float(df.loc[df.y == 1, "text"].str.contains(r"\d").mean()), 3),
    }
    pd.DataFrame([rows]).T.reset_index().rename(
        columns={"index": "quantity", 0: "value"}).to_csv(
        RESULTS / "data_audit.csv", index=False)
    return rows


def deduplicate(df):
    """Drop rows whose normalised text has already been seen (keep first)."""
    norm = normalise(df["text"])
    return df.loc[~norm.duplicated()].reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature extraction: identical for every classifier within an experiment
# --------------------------------------------------------------------------
VECTORISERS = {
    "word": dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                 sublinear_tf=True, strip_accents="unicode", lowercase=True),
    "char": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                 sublinear_tf=True, strip_accents="unicode", lowercase=True),
}


# --------------------------------------------------------------------------
# Algorithms. Eight configurations each: the tuning budget is held constant.
# --------------------------------------------------------------------------
def algorithms(seed):
    return {
        "MultinomialNB": (
            MultinomialNB(),
            {"clf__alpha": [0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
            "Linear / probabilistic"),
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, solver="liblinear",
                               random_state=seed),
            {"clf__C": [0.1, 1.0, 10.0, 100.0],
             "clf__class_weight": [None, "balanced"]},
            "Linear / probabilistic"),
        "LinearSVM": (
            LinearSVC(dual=True, max_iter=5000, random_state=seed),
            {"clf__C": [0.01, 0.1, 1.0, 10.0],
             "clf__class_weight": [None, "balanced"]},
            "Kernel (linear)"),
        "kNN": (
            KNeighborsClassifier(metric="cosine", algorithm="brute"),
            {"clf__n_neighbors": [1, 3, 5, 9],
             "clf__weights": ["uniform", "distance"]},
            "Instance-based"),
        "RandomForest": (
            RandomForestClassifier(max_features="sqrt", n_jobs=1,
                                   random_state=seed),
            {"clf__n_estimators": [100, 200],
             "clf__min_samples_leaf": [1, 3],
             "clf__class_weight": [None, "balanced_subsample"]},
            "Ensemble"),
    }


def decision_scores(model, X):
    """Continuous score for the positive class, whatever the estimator is."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def recall_at_precision(y_true, scores, target):
    """Highest recall attainable at precision >= target. 0.0 if unattainable."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision >= target
    return float(recall[ok].max()) if ok.any() else 0.0


# --------------------------------------------------------------------------
# One experiment matrix
# --------------------------------------------------------------------------
FOLD_CSV = RESULTS / "fold_level_results.csv"
PARAM_CSV = RESULTS / "selected_hyperparameters.csv"


def already_done():
    """(experiment, repeat, fold, algorithm) tuples already written to disk."""
    if not FOLD_CSV.exists():
        return set()
    d = pd.read_csv(FOLD_CSV)
    return set(zip(d.experiment, d.repeat, d.fold, d.algorithm))


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(),
                               index=False)


def run_matrix(exp_id, rep_name, df, done):
    X_text = df["text"].values
    y = df["y"].values
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                    random_state=SEED)
    algos = algorithms(SEED)

    for fold_idx, (tr, te) in enumerate(outer.split(X_text, y)):
        repeat = fold_idx // N_SPLITS + 1
        fold = fold_idx % N_SPLITS + 1
        for name, (estimator, grid, family) in algos.items():
            if (exp_id, repeat, fold, name) in done:
                continue
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**VECTORISERS[rep_name])),
                ("clf", estimator),
            ])
            inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                                    random_state=SEED + fold_idx)
            search = GridSearchCV(pipe, grid, scoring=SELECTION_METRIC,
                                  cv=inner, n_jobs=1, refit=True)

            t0 = time.perf_counter()
            search.fit(X_text[tr], y[tr])      # tuning + refit, training data only
            fit_s = time.perf_counter() - t0

            best = search.best_estimator_
            t0 = time.perf_counter()
            y_pred = best.predict(X_text[te])
            pred_s = time.perf_counter() - t0
            scores = decision_scores(best, X_text[te])

            tn, fp, fn, tp = confusion_matrix(y[te], y_pred, labels=[0, 1]).ravel()
            append(FOLD_CSV, dict(
                experiment=exp_id, representation=rep_name, corpus=(
                    "deduplicated" if exp_id == "E3" else "full"),
                algorithm=name, family=family, repeat=repeat, fold=fold,
                n_train=len(tr), n_test=len(te),
                macro_f1=f1_score(y[te], y_pred, average="macro"),
                spam_f1=f1_score(y[te], y_pred, pos_label=1),
                pr_auc=average_precision_score(y[te], scores),
                roc_auc=roc_auc_score(y[te], scores),
                mcc=matthews_corrcoef(y[te], y_pred),
                balanced_accuracy=balanced_accuracy_score(y[te], y_pred),
                accuracy=(tp + tn) / (tp + tn + fp + fn),
                recall_at_99_precision=recall_at_precision(
                    y[te], scores, TARGET_PRECISION),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                train_time_s=fit_s,
                predict_ms_per_1000=1000 * pred_s / len(te) * 1000,
                n_configs=len(list(search.cv_results_["params"])),
            ))
            append(PARAM_CSV, dict(
                experiment=exp_id, algorithm=name, repeat=repeat, fold=fold,
                best_params=json.dumps(
                    {k.replace("clf__", ""): v
                     for k, v in search.best_params_.items()}),
                inner_best_macro_f1=search.best_score_))
        print(f"[{exp_id}] repeat {repeat} fold {fold} done "
              f"({fold_idx + 1}/{N_SPLITS * N_REPEATS})", flush=True)


def main():
    df = load_data()
    info = audit(df)
    print("AUDIT:", info, flush=True)
    df_dedup = deduplicate(df)
    print(f"de-duplicated corpus: {len(df_dedup)} rows "
          f"({df_dedup.y.mean():.4f} spam)", flush=True)

    done = already_done()
    if done:
        print(f"resuming: {len(done)} algorithm-fold results already on disk",
              flush=True)
    run_matrix("E1", "word", df, done)
    run_matrix("E2", "char", df, done)
    run_matrix("E3", "word", df_dedup, done)

    env = {
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "x86_64",
        "seed": SEED,
        "dedup_rows": int(len(df_dedup)),
        "dedup_spam_share": round(float(df_dedup.y.mean()), 4),
    }
    env.update({f"audit_{k}": v for k, v in info.items()})
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
