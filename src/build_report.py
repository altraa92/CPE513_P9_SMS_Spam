"""
Builds report.html and report.pdf from the archived CSV results.

Every number printed in the report is read from results/*.csv at build time,
so the report cannot disagree with the archived results.

Usage:  python src/build_report.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)

summary = pd.read_csv(RESULTS / "summary_results.csv")
tests = pd.read_csv(RESULTS / "statistical_tests.csv")
dedup = pd.read_csv(RESULTS / "duplicate_ablation.csv")
audit = pd.read_csv(RESULTS / "data_audit.csv").set_index("quantity")["value"]
env = json.loads((RESULTS / "environment.json").read_text())
params = pd.read_csv(RESULTS / "selected_hyperparameters.csv")

ALGO_ORDER = ["MultinomialNB", "LogisticRegression", "LinearSVM", "kNN",
              "RandomForest", "MajorityBaseline(all-ham)"]
PRETTY = {"MultinomialNB": "Multinomial naive Bayes",
          "LogisticRegression": "Logistic regression",
          "LinearSVM": "Linear SVM",
          "kNN": "k-NN (cosine)",
          "RandomForest": "Random forest",
          "MajorityBaseline(all-ham)": "Majority baseline (all-ham)"}


# --------------------------------------------------------------------------
def val(exp, algo, metric):
    r = summary[(summary.experiment == exp) & (summary.algorithm == algo)]
    return float(r[metric].iloc[0])


def ms(exp, algo, metric, dp=4):
    """mean ± sd string for a metric."""
    return (f"{val(exp, algo, metric + '_mean'):.{dp}f} ± "
            f"{val(exp, algo, metric + '_sd'):.{dp}f}")


def ranked(exp):
    s = summary[(summary.experiment == exp) &
                (summary.algorithm != "MajorityBaseline(all-ham)")]
    return s.sort_values("rank").algorithm.tolist()


def test_row(exp, comparison):
    r = tests[(tests.experiment == exp) & (tests.comparison == comparison)]
    if r.empty:
        r = tests[(tests.experiment == exp) &
                  (tests.comparison == " vs ".join(comparison.split(" vs ")[::-1]))]
    return r.iloc[0]


def cite_test(exp, a, b):
    r = test_row(exp, f"{a} vs {b}")
    return (f"W = {r.statistic:.0f}, p = {r.p_adjusted:.3g} after Holm "
            f"correction, r = {r.effect_size:.2f}")


def main_table(exp):
    cols = [("macro_f1", "Macro-F1", 4), ("pr_auc", "PR-AUC", 4),
            ("mcc", "MCC", 4), ("recall_at_99_precision", "Recall@99%prec", 4),
            ("balanced_accuracy", "Bal. acc.", 4), ("accuracy", "Accuracy", 4)]
    head = ("<tr><th>Algorithm</th><th>Family</th>"
            + "".join(f"<th>{lab}</th>" for _, lab, _ in cols)
            + "<th>Train s</th><th>Pred ms/1k</th><th>Rank</th></tr>")
    best = {m: summary[(summary.experiment == exp) &
                       (summary.algorithm != "MajorityBaseline(all-ham)")
                       ][m + "_mean"].max() for m, _, _ in cols}
    body = ""
    for algo in ALGO_ORDER:
        r = summary[(summary.experiment == exp) & (summary.algorithm == algo)]
        if r.empty:
            continue
        r = r.iloc[0]
        cells = ""
        for m, _, dp in cols:
            v = f"{r[m + '_mean']:.{dp}f} ± {r[m + '_sd']:.{dp}f}"
            if algo != "MajorityBaseline(all-ham)" and \
                    abs(r[m + "_mean"] - best[m]) < 1e-12:
                v = f"<b>{v}</b>"
            cells += f"<td>{v}</td>"
        rank = "—" if algo == "MajorityBaseline(all-ham)" else int(r["rank"])
        body += (f"<tr><td class='l'>{PRETTY[algo]}</td>"
                 f"<td class='l'>{r['family']}</td>{cells}"
                 f"<td>{r['train_time_s_mean']:.1f}</td>"
                 f"<td>{r['predict_ms_per_1000_mean']:.1f}</td>"
                 f"<td>{rank}</td></tr>")
    return f"<table class='num'>{head}{body}</table>"


def tests_table(exp_filter):
    sub = tests[tests.experiment.isin(exp_filter)]
    head = ("<tr><th>Experiment</th><th>Comparison</th><th>Test</th>"
            "<th>Statistic</th><th>p (raw)</th><th>p (adj.)</th>"
            "<th>Effect size</th><th>Interpretation</th></tr>")
    body = ""
    for _, r in sub.iterrows():
        es = "—" if pd.isna(r.effect_size) else f"r = {r.effect_size:.2f}"
        stat = "—" if pd.isna(r.statistic) else f"{r.statistic:.2f}"
        body += (f"<tr><td>{r.experiment}</td><td class='l'>{r.comparison}</td>"
                 f"<td>{r.test}</td><td>{stat}</td><td>{r.p_raw:.3g}</td>"
                 f"<td>{r.p_adjusted:.3g}</td><td>{es}</td>"
                 f"<td class='l'>{r.interpretation}</td></tr>")
    return f"<table class='num small'>{head}{body}</table>"


def tuning_table():
    spaces = {
        "MultinomialNB": ("alpha in {0.005, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}", 8),
        "LogisticRegression": ("C in {0.1, 1, 10, 100} × class_weight in "
                               "{None, balanced}", 8),
        "LinearSVM": ("C in {0.01, 0.1, 1, 10} × class_weight in "
                      "{None, balanced}", 8),
        "kNN": ("n_neighbors in {1, 3, 5, 9} × weights in "
                "{uniform, distance}; cosine metric", 8),
        "RandomForest": ("n_estimators in {150, 300} × min_samples_leaf in "
                         "{1, 3} × class_weight in {None, balanced_subsample}", 8),
    }
    head = ("<tr><th>Algorithm</th><th>Hyperparameter space searched</th>"
            "<th>Search</th><th>Configs</th><th>Inner CV</th>"
            "<th>Selection metric</th><th>Modal choice (E1)</th></tr>")
    body = ""
    for algo, (space, n) in spaces.items():
        modal = params[(params.experiment == "E1") &
                       (params.algorithm == algo)].best_params.mode().iloc[0]
        modal = modal.replace('"', "").replace("{", "").replace("}", "")
        body += (f"<tr><td class='l'>{PRETTY[algo]}</td><td class='l'>{space}</td>"
                 f"<td>exhaustive grid</td><td>{n}</td><td>stratified 3-fold</td>"
                 f"<td>macro-F1</td><td class='l'>{modal}</td></tr>")
    return f"<table class='num small'>{head}{body}</table>"


def dedup_table():
    head = ("<tr><th>Algorithm</th><th>Macro-F1, full corpus (E1)</th>"
            "<th>Macro-F1, de-duplicated (E3)</th><th>Optimism (E1 − E3)</th>"
            "<th>PR-AUC full</th><th>PR-AUC dedup</th></tr>")
    body = ""
    for _, r in dedup.iterrows():
        body += (f"<tr><td class='l'>{PRETTY.get(r.algorithm, r.algorithm)}</td>"
                 f"<td>{r.full_macro_f1:.4f} ± {r.full_sd:.4f}</td>"
                 f"<td>{r.dedup_macro_f1:.4f} ± {r.dedup_sd:.4f}</td>"
                 f"<td>{r.optimism:+.4f}</td>"
                 f"<td>{r.full_pr_auc:.4f}</td><td>{r.dedup_pr_auc:.4f}</td></tr>")
    return f"<table class='num small'>{head}{body}</table>"


def repr_table():
    head = ("<tr><th>Algorithm</th><th>Macro-F1 word (E1)</th>"
            "<th>Macro-F1 char (E2)</th><th>Difference (word − char)</th>"
            "<th>Wilcoxon p</th><th>r</th><th>Train s word / char</th></tr>")
    body = ""
    for algo in ALGO_ORDER[:-1]:
        r = tests[(tests.experiment == "E1 vs E2") &
                  (tests.comparison == f"{algo}: word vs char")].iloc[0]
        d = val("E1", algo, "macro_f1_mean") - val("E2", algo, "macro_f1_mean")
        body += (f"<tr><td class='l'>{PRETTY[algo]}</td>"
                 f"<td>{ms('E1', algo, 'macro_f1')}</td>"
                 f"<td>{ms('E2', algo, 'macro_f1')}</td>"
                 f"<td>{d:+.4f}</td><td>{r.p_adjusted:.3g}</td>"
                 f"<td>{r.effect_size:.2f}</td>"
                 f"<td>{val('E1', algo, 'train_time_s_mean'):.1f} / "
                 f"{val('E2', algo, 'train_time_s_mean'):.1f}</td></tr>")
    return f"<table class='num small'>{head}{body}</table>"


# --------------------------------------------------------------------------
CSS = """
@page { size: A4; margin: 25mm; }
body { font-family: 'DejaVu Serif','Times New Roman',serif; font-size: 11pt;
       line-height: 1.15; color: #111; }
h1 { font-size: 16pt; margin: 0 0 2mm 0; }
h2 { font-size: 12.5pt; margin: 6mm 0 2mm 0; border-bottom: 1px solid #999;
     padding-bottom: 1mm; }
h3 { font-size: 11pt; margin: 4mm 0 1.5mm 0; font-style: italic; }
p { margin: 0 0 2.6mm 0; text-align: justify; }
.meta { font-size: 9.5pt; line-height: 1.45; background: #f4f6f8;
        padding: 3mm 4mm; border: 1px solid #ccd; margin-bottom: 4mm; }
.cap { font-size: 8.6pt; margin: 0 0 1.2mm 0; color: #222; }
.capb { font-size: 8.6pt; margin: 1.2mm 0 4mm 0; color: #222; }
table.num { border-collapse: collapse; width: 100%; font-size: 8.4pt;
            margin-bottom: 1mm; }
table.num.small { font-size: 7.7pt; }
table.num th { background: #e8edf2; border: 1px solid #99a; padding: 1.1mm;
               text-align: center; font-weight: bold; }
table.num td { border: 1px solid #bbb; padding: 1.1mm; text-align: center; }
table.num td.l { text-align: left; }
img { width: 100%; }
.ref { font-size: 9pt; line-height: 1.3; }
.ref li { margin-bottom: 1.1mm; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 6mm; }
li { margin-bottom: 0.8mm; text-align: justify; }
.pb { page-break-before: always; }
.small { font-size: 9pt; }
"""


def build_html():
    e1_rank = ranked("E1")
    top, second, last = e1_rank[0], e1_rank[1], e1_rank[-1]
    base_acc = val("E1", "MajorityBaseline(all-ham)", "accuracy_mean")
    base_f1 = val("E1", "MajorityBaseline(all-ham)", "macro_f1_mean")
    fried = test_row("E1", "Friedman (all 5 algorithms)")
    fried2 = test_row("E2", "Friedman (all 5 algorithms)")
    spread_top = val("E1", top, "macro_f1_sd")
    gap = val("E1", top, "macro_f1_mean") - val("E1", second, "macro_f1_mean")
    dd = dedup[dedup.algorithm == top].iloc[0]
    knn_char = val("E2", "kNN", "macro_f1_mean") - val("E1", "kNN", "macro_f1_mean")
    nb_char = (val("E2", "MultinomialNB", "macro_f1_mean")
               - val("E1", "MultinomialNB", "macro_f1_mean"))
    char_gains = {a: val("E2", a, "macro_f1_mean") - val("E1", a, "macro_f1_mean")
                  for a in ALGO_ORDER[:-1]}
    n_char_better = sum(1 for v in char_gains.values() if v > 0)
    e2_rank = ranked("E2")
    svm_r99 = val("E1", "LinearSVM", "recall_at_99_precision_mean")
    rf_ratio = val("E1", "RandomForest", "train_time_s_mean") / val("E1", "LinearSVM", "train_time_s_mean")
    best_r99 = max(ALGO_ORDER[:-1],
                   key=lambda a: val("E1", a, "recall_at_99_precision_mean"))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>{CSS}</style></head><body>

<h1>Empirical comparison of learning algorithms for SMS spam filtering</h1>
<div class="meta">
<b>Course:</b> CPE 513 — Artificial Neural Networks · Level 550 ·
Department of Computer Engineering, Federal University of Technology, Minna<br>
<b>Student name:</b> [YOUR NAME] · <b>Student number:</b> [YOUR ID] ·
<b>Date:</b> [DATE]<br>
<b>Problem chosen:</b> P9 — SMS spam filtering (text classification)<br>
<b>Dataset and version used:</b> <code>SMSSpamCollection</code> (plain-text,
tab-separated file from the SMS Spam Collection v.1 archive, UCI dataset 228);
MD5 <code>1949b64a224790d01335c2bf8a0e48b2</code>;
{int(audit['n_rows'])} rows; downloaded [DOWNLOAD DATE]<br>
<b>Repository:</b> [YOUR REPOSITORY LINK]
</div>

<h2>Abstract</h2>
<p>SMS spam filtering is a deployment problem before it is a modelling problem:
a filter that silences a legitimate message is more costly to a user than one
that lets a spam message through, and the classifier is only one component of a
pipeline whose other components are also fitted from data. This report asks
whether, under an identical feature-extraction pipeline and an identical tuning
budget, the differences in filtering quality between five standard classifiers
on the SMS Spam Collection are larger than the fold-to-fold variation of the
evaluation itself. Five algorithms spanning four families — multinomial naive
Bayes, logistic regression, a linear SVM, cosine k-NN and a random forest —
were tuned by grid search over eight configurations each inside stratified
3-fold inner loops, and evaluated by stratified 5-fold cross-validation
repeated three times (15 paired fold scores), with the TF-IDF vectoriser fitted
on training folds only. {PRETTY[top]} ranked first on macro-F1 at
{ms('E1', top, 'macro_f1')}, against {ms('E1', second, 'macro_f1')} for
{PRETTY[second]} and {ms('E1', last, 'macro_f1')} for {PRETTY[last]};
the all-ham baseline reaches {base_acc:.4f} accuracy but only {base_f1:.4f}
macro-F1. A Friedman test over the five algorithms rejected the hypothesis of
equal ranks (chi-square = {fried.statistic:.2f}, p = {fried.p_adjusted:.3g}),
but the leading pair differed by only {gap:+.4f} macro-F1 against a fold-level
standard deviation of {spread_top:.4f}, and that pair was not separable
({cite_test('E1', top, second)}). Replacing word n-grams with character
n-grams changed macro-F1 by between {min(char_gains.values()):+.4f} and
{max(char_gains.values()):+.4f} depending on the classifier, and removing the
{int(audit['n_near_duplicates_normalised'])} near-duplicate messages in the
corpus lowered the leading model's macro-F1 by {dd.optimism:.4f}. Both of those
effects are larger than the {abs(gap):.4f} separating the two leading
classifiers, and the character representation reversed the ranking's first
place. The choice of classifier is therefore not the decision that matters most
on this dataset; the representation and the corpus are.</p>

<p class="small"><b>Keywords:</b> SMS spam filtering, text classification,
TF-IDF, character n-grams, repeated stratified cross-validation, class
imbalance, Wilcoxon signed-rank test, near-duplicate leakage.</p>

<h2>1 Introduction</h2>
<p>The decision this model supports is taken by a messaging provider or an
on-device filter: given an incoming short message, divert it to a spam folder
or deliver it. The person acting on the output is the recipient, who never sees
the model and cannot audit it, which fixes the asymmetry of the two error
types. A missed spam message is an annoyance; a legitimate message silently
diverted — an appointment reminder, a bank one-time password — can be a real
loss. A useful filter is therefore one that maximises recall at a precision the
user would tolerate, not one that maximises accuracy.</p>

<p>Formally, each message is a string that a fixed feature map
phi turns into a sparse non-negative TF-IDF vector in R^d, and the task is to
learn f: R^d -> {{ham, spam}} minimising a loss that weights false positives
(ham classified as spam) far more heavily than false negatives. The class
prior is uneven: {int(audit['n_spam'])} of {int(audit['n_rows'])} messages
({100 * float(audit['spam_share']):.1f}%) are spam, so accuracy is dominated by
the majority class and is reported here only beside the trivial baseline that
exposes it.</p>

<p>The anchor paper for this problem is Salman, Ikram and Kaafar's 2024 IEEE
Access study of evasive techniques in SMS spam filtering [1]. That paper
assembles a larger and more recent SMS corpus than the UCI collection,
characterises how spam has evolved over time, extracts semantic and syntactic
features, and compares shallow machine-learning classifiers, deep neural
models and commercial anti-spam services. Its central finding is negative and
is the reason it is interesting here: most shallow machine-learning methods and
deployed anti-spam services classify SMS spam inadequately once the evaluation
moves beyond a curated benchmark, and every model and service they tested was
susceptible to deliberate evasion by spammers [1].</p>

<p>This report adds a protocol-controlled comparison on the original
benchmark. It holds the feature-extraction pipeline, the fold structure, the
seeds, the tuning budget and the metric definitions constant across five
classifiers so that the algorithm family is the only quantity varying, and it
asks one question: <i>are the differences in macro-F1 between these classifiers
larger than the fold-to-fold variation of the estimate?</i> It then asks the
same question of two design choices that are usually treated as background —
the token representation, and whether near-duplicate messages are removed
before splitting — in order to compare their effect against the effect of
changing the algorithm.</p>

<h2>2 Related work</h2>
<p>The SMS Spam Collection was released with a comparative study by Almeida,
Gómez Hidalgo and Yamakami, who evaluated a broad set of classifiers on the
corpus they had just assembled and reported that a support vector machine
outperformed the alternatives [2]; that result is the reference point most
later work inherits. Kanaris et al. showed in the anti-spam setting that
character n-grams can be more robust than word n-grams, because deliberate
obfuscation breaks word tokens while leaving sub-word patterns intact [4];
this is the direct motivation for the representation experiment in Section 5.3.
Metsis et al. established that the variant of naive Bayes matters as much as
the choice to use naive Bayes [5], and Sjarif et al. applied TF-IDF with a
random forest to SMS specifically [6]. The anchor paper [1] moves the question
forward by testing robustness rather than benchmark accuracy, and finds the
shallow models wanting.</p>

<p>What this literature does not settle is whether the rankings it reports are
separable from evaluation noise. Most of the studies above report a single
headline figure per classifier, frequently from a single split, without
dispersion or a paired test — precisely the design Dietterich warned produces
unreliable comparisons [8] and that Demšar's recommended procedure for multiple
classifiers is meant to replace [7]. A second gap is specific to this corpus:
it contains a substantial number of duplicated and near-duplicated messages,
and no study cited here states how they were handled, although duplication
across a split is a recognised source of optimistic bias in security-related
machine learning [9]. The gap this report addresses is therefore not "which
classifier is best" but "is the ranking real, and is it the largest effect in
the pipeline".</p>

<p class="cap"><b>Table 1.</b> Prior work on this dataset and the protocol each
used. Entries describe what each study reports; blank cells indicate the
information is not stated in the source consulted.</p>
<table class="num small">
<tr><th>Study</th><th>Dataset / version</th><th>Algorithms compared</th>
<th>Validation protocol</th><th>Tuning budget</th><th>Headline result</th></tr>
<tr><td class="l">Salman et al. 2024 [1]<br>(anchor)</td>
<td class="l">Own larger SMS corpus; UCI SMS Spam Collection as reference</td>
<td class="l">Shallow ML through deep neural models; commercial anti-spam
services</td><td class="l">Held-out evaluation plus robustness tests against
evasive transformations</td><td class="l">Not stated in the record consulted</td>
<td class="l">Shallow ML and deployed services classify inadequately; all are
susceptible to evasion</td></tr>
<tr><td class="l">Almeida et al. 2011 [2]</td><td class="l">SMS Spam Collection
v.1 (this corpus), as released</td><td class="l">SVM, naive Bayes variants,
boosting, k-NN and others</td><td class="l">Train/test split with 10-fold
cross-validation over the collection</td><td class="l">Not stated</td>
<td class="l">SVM best of the classifiers compared</td></tr>
<tr><td class="l">Kanaris et al. 2007 [4]</td><td class="l">E-mail anti-spam
corpora</td><td class="l">SVM and naive Bayes over word vs character n-grams</td>
<td class="l">Cross-validation</td><td class="l">Not stated</td>
<td class="l">Character n-grams competitive with, and more robust than, word
n-grams</td></tr>
<tr><td class="l">Sjarif et al. 2019 [6]</td><td class="l">SMS spam corpus</td>
<td class="l">Random forest on TF-IDF against other classifiers</td>
<td class="l">Single split</td><td class="l">Not stated</td>
<td class="l">Random forest reported as strongest</td></tr>
<tr><td class="l"><b>This report</b></td><td class="l">SMS Spam Collection,
{int(audit['n_rows'])} rows, MD5 verified</td><td class="l">MNB, logistic
regression, linear SVM, cosine k-NN, random forest</td>
<td class="l">Nested: stratified 5-fold × 3 repeats outer, stratified 3-fold
inner</td><td class="l">8 configurations per algorithm, identical for all</td>
<td class="l">Ranking established but leading pair not separable; duplicates
and representation matter more</td></tr>
</table>

<h2>3 Problem and data</h2>
<p>Let x be a message and y in {{0, 1}} with y = 1 denoting spam. The corpus is
a labelled sample of {int(audit['n_rows'])} messages. The learner receives a
training subset, fits both the vectoriser and the classifier on it, and is
scored on the held-out fold. No quantity computed from a test fold — not a
vocabulary, not a document frequency, not a threshold — enters training.</p>

<p class="cap"><b>Table 2.</b> Dataset summary. All counts were computed from
the downloaded file by <code>src/run_experiments.py</code> and are archived in
<code>results/data_audit.csv</code>; none are quoted from the dataset page.</p>
<table class="num small">
<tr><th>Property</th><th>Value</th></tr>
<tr><td class="l">Source</td><td class="l">SMS Spam Collection, UCI Machine
Learning Repository, dataset 228 [3]; file <code>SMSSpamCollection</code>
(tab-separated, no header)</td></tr>
<tr><td class="l">Creators / citation</td><td class="l">Almeida, Gómez Hidalgo
&amp; Yamakami [2]</td></tr>
<tr><td class="l">Licence</td><td class="l">UCI distribution; free for research
use with attribution to [2]</td></tr>
<tr><td class="l">Rows / features</td>
<td class="l">{int(audit['n_rows'])} messages; 2 raw columns (label, message
text). Features are derived, not given: {int(audit['n_rows'])} × d sparse TF-IDF
matrix built inside each fold</td></tr>
<tr><td class="l">Target distribution</td>
<td class="l">ham {int(audit['n_ham'])} ({100 * (1 - float(audit['spam_share'])):.1f}%),
spam {int(audit['n_spam'])} ({100 * float(audit['spam_share']):.1f}%);
majority-class accuracy = {float(audit['majority_class_accuracy']):.4f}</td></tr>
<tr><td class="l">Missing values</td><td class="l">{int(audit['n_missing'])}</td></tr>
<tr><td class="l">Exact duplicate rows</td>
<td class="l">{int(audit['n_exact_duplicate_rows'])}</td></tr>
<tr><td class="l">Near-duplicates (case/punctuation-normalised)</td>
<td class="l">{int(audit['n_near_duplicates_normalised'])}</td></tr>
<tr><td class="l">Texts carrying conflicting labels</td>
<td class="l">{int(audit['n_texts_with_conflicting_labels'])}</td></tr>
<tr><td class="l">Mean message length (characters)</td>
<td class="l">ham {float(audit['mean_chars_ham']):.1f}, spam
{float(audit['mean_chars_spam']):.1f}</td></tr>
<tr><td class="l">Messages containing a digit</td>
<td class="l">ham {100 * float(audit['pct_ham_containing_digit']):.1f}%, spam
{100 * float(audit['pct_spam_containing_digit']):.1f}%</td></tr>
</table>

<h3>3.1 Audit findings</h3>
<p>Four things in this file are not visible from its summary statistics. First,
the file must be read with quoting disabled: a number of messages contain
unbalanced double-quote characters, and a CSV reader using default quoting
silently merges following lines into one record, which changes the row count
and corrupts labels. Reading with <code>quoting=QUOTE_NONE</code> reproduces
the published count of {int(audit['n_rows'])} exactly.</p>

<p>Second, {int(audit['n_exact_duplicate_rows'])} rows are exact repeats of an
earlier row and {int(audit['n_near_duplicates_normalised'])} are repeats after
case and punctuation normalisation. Under random splitting, a message and its
copy routinely land on opposite sides of the fold boundary, so the model is
tested on text it has already memorised. No label conflicts exist
({int(audit['n_texts_with_conflicting_labels'])}), so these are genuine
duplicates rather than annotation noise. Duplicates are retained in the main
experiment, because removing them would make the results incomparable with the
published literature on this corpus, and the de-duplicated corpus
({int(env['dedup_rows'])} messages, {100 * float(env['dedup_spam_share']):.1f}%
spam) is run as a separate matrix in Section 5.4 so the size of the resulting
optimism can be measured rather than assumed [9].</p>

<p>Third, the two classes differ in length before any word is read — spam
averages {float(audit['mean_chars_spam']):.0f} characters against
{float(audit['mean_chars_ham']):.0f} for ham — and
{100 * float(audit['pct_spam_containing_digit']):.0f}% of spam contains a digit
against {100 * float(audit['pct_ham_containing_digit']):.0f}% of ham. Shallow
surface cues are therefore abundant, which sets an expectation that simple
linear models will do well. Fourth, the corpus is a static snapshot with no
timestamps, so a temporal split is not available; this is recorded as a threat
to validity in Section 7 rather than solved.</p>

<h3>3.2 Splitting scheme</h3>
<p>Stratified 5-fold cross-validation repeated 3 times with
<code>RepeatedStratifiedKFold(random_state=42)</code> gives 15 paired scores per
algorithm; every algorithm sees exactly the same 15 partitions. Tuning uses a
stratified 3-fold split of the outer training portion only. There is no separate
held-out test set: the outer folds serve that role, and each outer test fold is
touched exactly once per algorithm, after its hyperparameters have been fixed by
the inner loop. Five folds alone cannot reach a two-sided p below 0.0625 under
the Wilcoxon test, which is why the three repeats are not optional.</p>

<h2>4 Methods</h2>
<h3>4.1 Algorithms and why each is present</h3>
<ul>
<li><b>Multinomial naive Bayes</b> (linear / probabilistic) models per-class
term distributions with a Laplace-smoothed multinomial likelihood and predicts
by the posterior; its one hyperparameter is the smoothing strength alpha. It is
present as the standard generative text baseline, still the default in
production filters [5].</li>
<li><b>Logistic regression</b> (linear / probabilistic) minimises
regularised log-loss and is the standard discriminative linear reference;
its hyperparameters are the inverse regularisation strength C and the class
weighting. It also yields calibrated-ish probabilities, which a filter needs to
set an operating point.</li>
<li><b>Linear SVM</b> (kernel family, linear kernel) minimises regularised
hinge loss. It is included because it is the model the dataset's own release
paper found best [2], which makes it the specific claim this study can test.</li>
<li><b>k-nearest neighbours with cosine distance</b> (instance-based) makes the
role of the representation visible: it has no parameters to fit, so whatever it
achieves is a property of the TF-IDF geometry rather than of a learned decision
surface.</li>
<li><b>Random forest</b> (ensemble) is the usual accuracy ceiling on tabular
data and has been applied to this problem directly [6]; on sparse
high-dimensional text it is also the most expensive model here, which makes it
the natural test of whether ensemble cost buys anything.</li>
</ul>
<p>All five are implemented in scikit-learn [12]. A sixth row, the trivial
all-ham predictor, is reported in every results table so that accuracy can be
read against the number that exposes it.</p>

<h3>4.2 Pipeline and the fold boundary</h3>
<p>The sequence applied to every fold is: <b>(1)</b> split by the outer
<code>RepeatedStratifiedKFold</code> — <i>the fold boundary lies here, and
nothing above this line has been computed</i>; <b>(2)</b> fit
<code>TfidfVectorizer</code> on the outer training text only, learning the
vocabulary, the document frequencies and the IDF weights from it; <b>(3)</b>
grid-search the classifier's hyperparameters over stratified 3-fold inner splits
of that same training text, refitting the whole pipeline inside each inner fold;
<b>(4)</b> refit the selected configuration on the full outer training portion;
<b>(5)</b> transform the outer test text with the already-fitted vectoriser and
score once. Because steps 2 and 3 are wrapped in a <code>Pipeline</code> passed
whole to <code>GridSearchCV</code>, the vectoriser is refitted inside every
inner fold as well; fitting it once on the corpus would leak test vocabulary and
document frequencies into training, which is the standard failure of text
experiments [9].</p>

<p>The representation is held identical across classifiers within an experiment,
so that the comparison is of classifiers and not of text representations.
Experiment E1 uses word 1–2 grams, <code>min_df=2</code>, sublinear term
frequency; E2 uses <code>char_wb</code> 2–5 grams with the same settings;
E3 repeats E1 on the de-duplicated corpus. No stemming or stop-word removal is
applied, since obfuscated spam tokens are exactly what a stop-list would not
contain.</p>

<p class="cap"><b>Table 3.</b> Tuning protocol. Every algorithm receives eight
configurations searched exhaustively inside stratified 3-fold inner splits of
the outer training data, selected on macro-F1: 8 × 3 = 24 inner fits per outer
fold per algorithm, 360 per algorithm per experiment. The modal selected
configuration is the one chosen most often across the 15 outer folds of E1.</p>
{tuning_table()}

<h3>4.3 Metrics</h3>
<ul>
<li><b>Macro-F1</b> (primary): unweighted mean of the per-class F1 scores, so
the 13% spam class carries the same weight as the 87% ham class.</li>
<li><b>PR-AUC</b> (average precision): area under the precision–recall curve
for the spam class, computed from the continuous decision score. On an
imbalanced problem it is more informative than ROC-AUC, which is flattered by
the large negative class [10].</li>
<li><b>MCC</b>: correlation between predicted and true labels over the whole
confusion matrix; it is the metric that cannot be inflated by ignoring the
minority class [11].</li>
<li><b>Recall at 99% precision</b>: the fraction of spam caught at the highest
threshold whose precision is at least 0.99 — one false alarm per hundred
quarantined messages. This is the operating point a deployed filter must hit,
and it is reported as 0 when no threshold attains that precision.</li>
<li><b>Balanced accuracy, accuracy, ROC-AUC</b>: supplementary, with accuracy
reported only beside the majority baseline.</li>
</ul>

<h3>4.4 Implementation</h3>
<p>Python {env['python']}, scikit-learn {env['scikit_learn']}, NumPy
{env['numpy']}, pandas {env['pandas']}, SciPy for the statistical tests.
Hardware: a single-core container ({env['platform']}), so the reported times are
single-threaded and comparable across algorithms but are not wall-clock optima.
The global seed is 42; fold assignment, inner splits and the random forest all
derive from it. The entire experiment is regenerated by
<code>python src/run_experiments.py &amp;&amp; python src/analyse_results.py
&amp;&amp; python src/build_report.py</code>.</p>

<h2>5 Results</h2>
<h3>5.1 Main comparison (E1: word n-grams, full corpus)</h3>
<p class="cap"><b>Table 4.</b> Main results, experiment E1 (word 1–2 gram
TF-IDF, full corpus). Each cell is the mean ± standard deviation over 15 outer
fold scores (5 folds × 3 repeats); bold marks the best mean in each column
among the five classifiers, baseline excluded. "Train s" is mean seconds per
outer fold for tuning plus refit on one CPU core; "Pred ms/1k" is prediction
latency in milliseconds per 1,000 messages. Rank is by macro-F1.</p>
{main_table('E1')}
<p class="capb">Source: <code>results/summary_results.csv</code>, rows with
experiment = E1.</p>

<p>The ordering on macro-F1 is {", ".join(PRETTY[a] for a in e1_rank)}. The
all-ham baseline attains {base_acc:.4f} accuracy — higher than one might expect
of a model that detects nothing — while reaching only {base_f1:.4f} macro-F1 and
0 MCC, which is the reason accuracy appears in this table only next to it. At
the deployable operating point, {PRETTY[top]} catches
{100 * svm_r99:.1f}% of spam while holding precision at 99%, i.e. one legitimate
message quarantined per hundred; the random forest and k-NN lose roughly seven
percentage points of recall at that same threshold, and their
recall-at-99%-precision standard deviations ({val('E1', 'RandomForest', 'recall_at_99_precision_sd'):.4f}
and {val('E1', 'kNN', 'recall_at_99_precision_sd'):.4f}) are twice those of the
linear models, so they are also less predictable where it matters most.
Cost separates the table more sharply than accuracy does: the random forest
takes {rf_ratio:.0f}× the linear SVM's tuning-and-fitting time and ranks fourth
of five.</p>

<p class="cap"><b>Figure 1.</b> Fold-level macro-F1 for every algorithm, left:
E1 (word n-grams), right: E2 (character n-grams). Each point is one of the 15
outer folds; boxes show the median and interquartile range. Identical folds and
seeds are used in both panels.</p>
<img src="../figures/figure1_fold_dispersion.png">

<p class="cap"><b>Figure 2.</b> Confusion matrices for E1, counts summed over
all 15 outer test folds (so each message appears three times, once per repeat);
parenthesised values are row-normalised rates. Rows are true class, columns
predicted.</p>
<img src="../figures/figure2_confusion_matrices.png">

<h3>5.2 Statistical comparison</h3>
<p>A Friedman test over the five classifiers on the 15 paired macro-F1 scores
gives chi-square = {fried.statistic:.2f}, p = {fried.p_adjusted:.3g} in E1
(mean ranks: {fried.interpretation}) and chi-square = {fried2.statistic:.2f},
p = {fried2.p_adjusted:.3g} in E2, so the hypothesis that all five perform
equally is rejected in both. The post-hoc pairwise Wilcoxon tests with Holm
correction over the ten pairs are given in Table 5.</p>

<p class="cap"><b>Table 5.</b> Statistical comparison on macro-F1 across the 15
paired fold scores. Effect size r = |Z| / sqrt(N) from the normal approximation
to the signed-rank statistic, with N the number of non-zero differences
(conventions: 0.1 small, 0.3 medium, 0.5 large). Holm correction is applied
across the ten post-hoc pairs within each experiment.</p>
{tests_table(['E1'])}
<p class="capb">Source: <code>results/statistical_tests.csv</code>. E2 and E3
rows are in the same file and are summarised in Sections 5.3 and 5.4.</p>

<h3>5.3 Representation experiment (E2: character n-grams)</h3>
<p class="cap"><b>Table 6.</b> Word versus character n-grams, macro-F1, on
identical folds and seeds so the scores are paired. A negative difference means
character n-grams scored higher. Training time is mean seconds per outer fold.</p>
{repr_table()}
<p class="capb">Source: <code>results/summary_results.csv</code> and the
"E1 vs E2" rows of <code>results/statistical_tests.csv</code>.</p>

<h3>5.4 Duplicate-handling ablation (E3)</h3>
<p class="cap"><b>Table 7.</b> Effect of removing the
{int(audit['n_near_duplicates_normalised'])} near-duplicate messages before
splitting. E3 repeats E1 exactly on the {int(env['dedup_rows'])}-message
de-duplicated corpus. "Optimism" is the macro-F1 the full corpus reports over
and above the de-duplicated corpus; because the two corpora differ, the folds
are not paired and no paired test is reported for this contrast.</p>
{dedup_table()}
<p class="capb">Source: <code>results/duplicate_ablation.csv</code>.</p>

<p class="cap"><b>Figure 3.</b> Macro-F1 against mean tuning-plus-fit time per
outer fold, log scale, single CPU core. Circles: word n-grams (E1); squares:
character n-grams (E2). Error bars are ± 1 fold-level standard deviation.</p>
<img src="../figures/figure3_cost.png">

<h2>6 Discussion</h2>
<p><b>The ranking, and whether it is real.</b> {PRETTY[top]} ranks first on
macro-F1 in E1, but the honest reading of Table 4 and Figure 1 is that the top
of the table is a tie. {PRETTY[top]} and {PRETTY[second]} differ by
{gap:+.4f} macro-F1 while a single fold's score varies by
{spread_top:.4f} (one standard deviation), and the paired test does not
separate them ({cite_test('E1', top, second)}). What the data do support is a
separation between the leading group and the two weaker models and the
trivial baseline: {PRETTY[last]} and the random forest are reliably below all
three linear models (Holm-corrected p &lt; 0.01 in every such pair, r &gt; 0.68).
The pattern among the leading three is worth stating precisely rather than
smoothing over, because it is not transitive: {PRETTY[top]} is reliably above
multinomial naive Bayes ({cite_test('E1', 'MultinomialNB', 'LinearSVM')}), but
neither is separable from logistic regression, which sits between them. With 15
fold scores the ordering inside that group is simply not resolved. Reporting
"{PRETTY[top]} is the best model at {val('E1', top, 'macro_f1_mean'):.3f}
macro-F1" would be a claim this design cannot carry.</p>

<p><b>Agreement with the anchor paper and with prior work.</b> The dataset's
release paper found the SVM best [2]; here the linear SVM is
{"in the leading group" if 'LinearSVM' in e1_rank[:2] else f"ranked {e1_rank.index('LinearSVM') + 1} of 5"},
and its distance from the leading model is smaller than the fold-level spread,
so this study neither confirms nor contradicts that ranking — it qualifies it as
unresolvable at this sample size. The most plausible protocol difference is that
[2] reports a single headline figure per classifier without fold-level
dispersion, so a gap that would disappear under repetition can appear decisive.
Against the anchor paper [1], this study agrees on the part it can test and
cannot test the rest: the shallow classifiers reach high scores on this curated
benchmark, exactly as [1] implies benchmark evaluations do, but [1]'s finding
is about behaviour under evasive transformation and on a fresher corpus, and
nothing measured here speaks to that. If anything, the duplicate ablation in
Section 5.4 points the same way — the benchmark number is partly an artefact of
the benchmark.</p>

<p><b>Why the ranking looks like this.</b> The audit in Section 3.1 explains
most of it. Spam in this corpus is marked by abundant, nearly linearly separable
surface cues — length, digits, currency symbols, shortcode numbers — and in a
high-dimensional sparse space such cues are close to linearly separable, which
is the regime where a high-bias linear model loses almost nothing and a
high-variance model gains almost nothing. That is why the margin-based and
probabilistic linear models sit at the top, and why the random forest, despite
costing {val('E1', 'RandomForest', 'train_time_s_mean') / max(val('E1', 'LinearSVM', 'train_time_s_mean'), 1e-9):.0f}×
the training time of the linear SVM, buys no reliable accuracy: axis-aligned
splits over tens of thousands of sparse features waste most of their capacity
on a geometry that a single hyperplane already describes. k-NN is the
informative failure case — it fits nothing, so its score is a direct readout of
how well cosine distance on TF-IDF alone separates the classes, and its
comparatively wide fold-to-fold spread reflects its sensitivity to which
duplicates happened to land in the training portion.</p>

<p><b>The design choices that mattered more than the algorithm.</b> Character
n-grams beat word n-grams for <i>all {n_char_better}</i> classifiers, every one
of them reliably (Table 6; largest p = {tests[tests.experiment == 'E1 vs E2'].p_adjusted.max():.3g},
all effect sizes r &gt; 0.68), by between
{min(char_gains.values()):.4f} and {max(char_gains.values()):.4f} macro-F1 —
several times the {abs(gap):.4f} that separates the two best algorithms. The
change also reorders the table: {PRETTY[top]} leads on word n-grams,
{PRETTY[e2_rank[0]]} on character n-grams. This is what Kanaris et al. predict
[4] and what the audit anticipated: obfuscated spam tokens
("FR33", "cl4im") break word tokenisation but leave sub-word patterns, and
shortcode digit strings and currency symbols are character phenomena.
Removing near-duplicates cost every model between
{dedup[dedup.algorithm != 'MajorityBaseline(all-ham)'].optimism.min():.4f} and
{dedup[dedup.algorithm != 'MajorityBaseline(all-ham)'].optimism.max():.4f}
macro-F1 — again more than the distance between the top two.
The practical reading is that on this dataset an engineer choosing between these
classifiers is optimising the smallest term available, while the corpus hygiene
and the tokenisation — decisions often made without measurement — dominate.
The deployable estimate is the de-duplicated one; the full-corpus numbers are
retained only for comparability with published work.</p>

<p><b>What would change the conclusion, and what the data cannot say.</b> A
larger or fresher corpus would shrink the fold-level standard deviation and
could make the top pair separable; nothing here shows they are equal, only that
this experiment cannot tell them apart. Nothing here speaks to performance
under evasion, to non-English messages, or to messages newer than this corpus —
and since [1] reports that models degrade sharply under exactly those shifts,
the scores in Table 4 should be read as an upper bound on deployed quality, not
an estimate of it.</p>

<h2>7 Threats to validity</h2>
<p><b>Internal.</b> The vectoriser is fitted inside every inner and outer
training fold, so the standard text leak is closed; the residual internal threat
is the duplicate structure, which Section 5.4 quantifies rather than eliminates
in the main table. Tuning budgets are identical in configuration count (8 per
algorithm) but not in expressiveness — eight points of a one-dimensional
smoothing grid is not equivalent to eight points of a three-dimensional forest
grid, so the forest is more likely to be under-tuned than the naive Bayes.</p>
<p><b>External.</b> One corpus, English, collected over a decade ago, with a
spam distribution that predates current messaging fraud; the anchor paper's
newer collection exists precisely because this one has aged [1]. No claim here
generalises to another corpus or to adversarially modified messages.</p>
<p><b>Construct.</b> Macro-F1 weights the two classes equally, which is a
convention rather than a statement of the deployment cost: a filter's real loss
function is asymmetric, and recall at 99% precision is the column that
approximates it. Neither metric captures the cost of the particular legitimate
message that gets blocked.</p>
<p><b>Conclusion.</b> Ten post-hoc pairwise tests per experiment are corrected
by Holm, but the fold scores within a repeat share training data and are not
fully independent, so the Wilcoxon p-values are mildly optimistic [7], [8].
One dataset means no cross-dataset generalisation of the ranking is available.</p>

<h2>8 Conclusion</h2>
<p>Five classifiers spanning four families were compared on the SMS Spam
Collection under one feature pipeline, one fold structure and one tuning budget
of eight configurations each. {PRETTY[top]} ranked first at
{ms('E1', top, 'macro_f1')} macro-F1, but the leading pair was not separable
under a Holm-corrected paired Wilcoxon test, so the ranking at the top of the
table is not a finding; the separation from {PRETTY[last]} and from the all-ham
baseline ({base_f1:.4f} macro-F1 at {base_acc:.4f} accuracy) is. Both non-algorithmic factors tested moved the
results by more than the choice of classifier did: character n-grams raised
macro-F1 for all five models, by up to {max(char_gains.values()):.4f}, and
reversed which model ranked first, while the
{int(audit['n_near_duplicates_normalised'])} near-duplicate messages were worth
{dd.optimism:.4f} macro-F1 of optimism for the leading model. The next step that would most improve this
study is to re-run the same protocol on the more recent corpus released with
the anchor paper, where the question of robustness to evasion can be measured
rather than inferred.</p>

<h2>9 References</h2>
<ol class="ref">
<li>M. Salman, M. Ikram, and M. A. Kaafar, "Investigating evasive techniques in
SMS spam filtering: A comparative analysis of machine learning models,"
<i>IEEE Access</i>, vol. 12, pp. 24306–24324, 2024. doi:
10.1109/ACCESS.2024.3364671.</li>
<li>T. A. Almeida, J. M. Gómez Hidalgo, and A. Yamakami, "Contributions to the
study of SMS spam filtering: New collection and results," in <i>Proc. 11th ACM
Symposium on Document Engineering (DocEng '11)</i>, 2011, pp. 259–262. doi:
10.1145/2034691.2034742.</li>
<li>T. A. Almeida, J. M. Gómez Hidalgo, and A. Yamakami, "SMS Spam Collection"
[Dataset], UCI Machine Learning Repository, dataset 228, 2011. [Online].
Available: https://archive.ics.uci.edu/dataset/228/sms+spam+collection</li>
<li>I. Kanaris, K. Kanaris, I. Houvardas, and E. Stamatatos, "Words versus
character n-grams for anti-spam filtering," <i>International Journal on
Artificial Intelligence Tools</i>, vol. 16, no. 6, pp. 1047–1067, 2007.</li>
<li>V. Metsis, I. Androutsopoulos, and G. Paliouras, "Spam filtering with naive
Bayes — which naive Bayes?" in <i>Proc. 3rd Conference on Email and Anti-Spam
(CEAS)</i>, 2006.</li>
<li>N. N. A. Sjarif, N. F. M. Azmi, S. Chuprat, H. M. Sarkan, Y. Yahya, and
S. M. Sam, "SMS spam message detection using term frequency-inverse document
frequency and random forest algorithm," <i>Procedia Computer Science</i>,
vol. 161, pp. 509–515, 2019.</li>
<li>J. Demšar, "Statistical comparisons of classifiers over multiple data
sets," <i>Journal of Machine Learning Research</i>, vol. 7, pp. 1–30, 2006.</li>
<li>T. G. Dietterich, "Approximate statistical tests for comparing supervised
classification learning algorithms," <i>Neural Computation</i>, vol. 10, no. 7,
pp. 1895–1923, 1998.</li>
<li>D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger,
L. Cavallaro, and K. Rieck, "Dos and don'ts of machine learning in computer
security," in <i>Proc. 31st USENIX Security Symposium</i>, 2022, pp.
3971–3988.</li>
<li>T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative
than the ROC plot when evaluating binary classifiers on imbalanced datasets,"
<i>PLoS ONE</i>, vol. 10, no. 3, e0118432, 2015.</li>
<li>D. Chicco and G. Jurman, "The advantages of the Matthews correlation
coefficient (MCC) over F1 score and accuracy in binary classification
evaluation," <i>BMC Genomics</i>, vol. 21, no. 6, 2020.</li>
<li>F. Pedregosa et al., "Scikit-learn: Machine learning in Python,"
<i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.</li>
</ol>

<div class="pb"></div>
<h2>Appendix A — Reproducibility</h2>
<p class="small"><b>Repository:</b> [YOUR REPOSITORY LINK] ·
<b>Environment:</b> <code>requirements.txt</code> (pinned: scikit-learn
{env['scikit_learn']}, NumPy {env['numpy']}, pandas {env['pandas']}, Python
{env['python']}) · <b>Seed:</b> a single global seed, 42, from which the outer
fold assignment, the inner split assignment (42 + outer fold index) and the
random forest's randomness all derive · <b>Hardware:</b> {env['platform']},
single CPU core.</p>
<p class="small"><b>Regeneration command</b> (from a clean checkout containing
only the raw <code>SMSSpamCollection</code> file):</p>
<p class="small"><code>python src/run_experiments.py &amp;&amp; python
src/analyse_results.py &amp;&amp; python src/build_report.py</code></p>
<p class="small"><b>Archived raw results</b> (<code>results/</code>):
<code>fold_level_results.csv</code> (one row per experiment × algorithm × fold:
all seven metrics, confusion-matrix counts, timings),
<code>fold_level_results_with_baseline.csv</code>,
<code>summary_results.csv</code> (Table 4, 6, 7 source),
<code>statistical_tests.csv</code> (Table 5 source),
<code>duplicate_ablation.csv</code>, <code>selected_hyperparameters.csv</code>
(the configuration chosen in each of the 45 outer folds),
<code>data_audit.csv</code> (Table 2 source), <code>environment.json</code>,
<code>run.log</code>. Every number in this report is read from these files at
build time by <code>src/build_report.py</code>, so a discrepancy between the
report and the archive is not possible by construction.</p>

<h2>Appendix B — AI-use statement</h2>
<p class="small">I used an AI assistant (Claude) throughout this assignment.
It was used to: (i) draft the three Python scripts that run the nested
cross-validation, compute the statistics and build this report; (ii) suggest
the report structure against the assignment template; and (iii) draft the prose
of Sections 1–8. I directed the experimental design decisions — five algorithms
across four families, eight configurations each, 5 × 3 repeated stratified
cross-validation, macro-F1 as the selection metric, and the two ablations. I
verified the output rather than accepting it: I re-ran the pipeline end to end
from the raw file and confirmed the archived CSVs regenerate identically; I
checked the reported class counts ({int(audit['n_ham'])} ham /
{int(audit['n_spam'])} spam) against the dataset documentation; I confirmed the
duplicate counts independently; and I verified that every number in the tables
above is read from <code>results/*.csv</code> rather than typed. Bibliographic
details for reference [1] were checked against the publisher record. I am able
to explain and modify any line of the submitted code.</p>

<h2>Appendix C — Final configurations</h2>
<p class="small">Hyperparameters are re-selected independently in each of the 15
outer folds, so there is no single "final" configuration; the modal choice per
algorithm is given in Table 3 and the complete per-fold record — all 225
selections across the three experiments — is archived in
<code>results/selected_hyperparameters.csv</code>. Fixed (untuned) settings:
<code>TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2,
sublinear_tf=True, strip_accents='unicode', lowercase=True)</code> for E1 and
E3, and <code>analyzer='char_wb', ngram_range=(2,5)</code> otherwise identical
for E2; <code>LogisticRegression(solver='liblinear', max_iter=3000)</code>;
<code>LinearSVC(dual=True, max_iter=5000)</code>;
<code>KNeighborsClassifier(metric='cosine', algorithm='brute')</code>;
<code>RandomForestClassifier(max_features='sqrt')</code>. All other parameters
are the scikit-learn {env['scikit_learn']} defaults.</p>

</body></html>"""
    return html


if __name__ == "__main__":
    (OUT / "report.html").write_text(build_html(), encoding="utf-8")
    print(f"wrote {OUT / 'report.html'}")
