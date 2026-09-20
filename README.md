## How to reproduce everything

From a clean checkout containing only `data/SMSSpamCollection`:

```bash
pip install -r requirements.txt
python src/run_experiments.py      # ~45 min on one CPU core
python src/analyse_results.py      # seconds
python src/build_report.py         # seconds
wkhtmltopdf --page-size A4 --margin-top 20mm --margin-bottom 18mm \
  --margin-left 20mm --margin-right 20mm --enable-local-file-access \
  --footer-center "[page]" --footer-font-size 8 report/report.html report/report.pdf
```

`run_experiments.py` is **resumable**: it appends each fold's results to
`results/fold_level_results.csv` as they are computed and skips anything already
there. If it is interrupted, just run it again. To force a clean re-run, delete
`results/fold_level_results.csv` and `results/selected_hyperparameters.csv` first.

## What was run

**Five algorithms across four families** (the brief requires four across three):

| Algorithm | Family | Why it is in the study |
|---|---|---|
| Multinomial naive Bayes | Linear / probabilistic | The standard generative text baseline |
| Logistic regression | Linear / probabilistic | The standard discriminative linear reference |
| Linear SVM | Kernel (linear) | The model the dataset's own release paper found best — the specific claim being tested |
| k-NN (cosine) | Instance-based | Fits nothing, so its score reads out the TF-IDF geometry directly |
| Random forest | Ensemble | The usual tabular ceiling; tests whether ensemble cost buys anything |

Plus the trivial **all-ham baseline** in every table, so accuracy can be read against
the number that exposes it.

**Protocol.** Nested cross-validation. Outer: stratified 5-fold × 3 repeats = 15 paired
fold scores. Inner: stratified 3-fold grid search, **exactly 8 configurations for every
algorithm**, selected on macro-F1. Global seed 42. Every algorithm sees the identical
15 partitions.

**Three experiment matrices.** E1 = word 1–2 grams, full corpus (main results).
E2 = character `char_wb` 2–5 grams (the representation experiment your problem statement
requires as a second table). E3 = word n-grams on the de-duplicated corpus (the ablation).

**Metrics.** Macro-F1 (primary), PR-AUC, MCC, recall at 99% precision (the deployable
operating point), balanced accuracy, ROC-AUC, accuracy, plus training time and
prediction latency.

## What the data showed

Main table (E1, word n-grams, macro-F1, mean ± sd over 15 folds):

| Rank | Algorithm | Macro-F1 | Recall @99% precision | Train s/fold |
|---|---|---|---|---|
| 1 | Linear SVM | 0.9745 ± 0.0071 | 0.9237 | 2.4 |
| 2 | Logistic regression | 0.9719 ± 0.0080 | 0.9184 | 2.3 |
| 3 | Multinomial naive Bayes | 0.9709 ± 0.0072 | 0.9059 | 2.1 |
| 4 | Random forest | 0.9632 ± 0.0090 | 0.8564 | 27.8 |
| 5 | k-NN (cosine) | 0.9548 ± 0.0091 | 0.8626 | 3.3 |
| — | Majority baseline (all-ham) | 0.4641 | 0.0000 | 0.0 |

Four findings, in order of how much they are worth to you:

1. **The top of the table is not resolved.** Friedman rejects equal ranks
   (χ² = 39.75, p = 4.9e-08), but the SVM and logistic regression differ by 0.0026
   against a fold-level sd of 0.0071 and are not separable (Holm-corrected p = 0.096).
   The pattern among the top three is not even transitive: the SVM beats naive Bayes
   reliably, but neither is separable from logistic regression sitting between them.
   Saying "the SVM is best at 0.974" is a claim this design cannot carry — and the
   brief's rubric penalises exactly that.

2. **Character n-grams beat word n-grams for all five classifiers, every one reliably**
   (largest p = 0.0054, all r > 0.68), by 0.0044 to 0.0121 macro-F1 — several times the
   gap between the two best algorithms. They also *reverse first place*: the SVM leads on
   words, logistic regression on characters. This is the required second table, and it is
   the strongest result in the report.

3. **The corpus contains 403 exact and 444 near-duplicate messages.** Removing them costs
   every model 0.0050–0.0101 macro-F1 — again more than the distance between the top two.
   That optimism is a property of the benchmark, not of any algorithm.

4. **Cost separates the table more sharply than accuracy does.** The random forest takes
   ~12× the SVM's training time and ranks fourth of five; k-NN has 4× the prediction
   latency and ranks fifth.

The honest headline: on this dataset the choice of classifier is the *smallest* decision
available. The representation and the corpus hygiene matter more.

## Design decisions you should be able to defend

- **Reading the file.** It must be read with `quoting=QUOTE_NONE`. Several messages contain
  unbalanced double-quote characters, and a default CSV reader silently merges following
  lines into one record — changing the row count and corrupting labels. With quoting
  disabled the published count of 5,574 reproduces exactly.
- **Where the fold boundary is.** Split first; *then* fit the TF-IDF vectoriser on the
  training text only. The vectoriser is inside a `Pipeline` handed whole to `GridSearchCV`,
  so it is refitted inside every inner fold too. Fitting it once on the corpus would leak
  test vocabulary and document frequencies into training — the classic text-experiment error.
- **Duplicates retained in the main table, removed in E3.** Removing them would make the
  numbers incomparable with published work on this corpus; keeping them without measuring
  the cost would be dishonest. So both are run, and the difference is reported.
- **Equal tuning budget.** 8 configurations each. Not equally *expressive* — eight points of
  a 1-D smoothing grid is not eight points of a 3-D forest grid — which is stated as a
  threat to validity rather than hidden.
- **Two changes made for the single-core compute budget:** the forest's `n_estimators` grid
  is {100, 200}, and the character vectoriser uses `min_df=3` (its vocabulary is far larger
  than the word one). Both are in Table 3 and Appendix C.

## Oral check preparation

Likely questions, with where to find your answer:

- *"Where is the fold boundary in your pipeline?"* — `run_experiments.py`, `run_matrix()`.
  The outer split happens first; the `Pipeline` containing the vectoriser is fitted only on
  `X_text[tr]`. Nothing derived from `X_text[te]` exists before step 5.
- *"Why macro-F1 and not accuracy?"* — 13.4% spam. The all-ham baseline gets 0.8660 accuracy
  while detecting nothing; its macro-F1 is 0.4641 and its MCC is 0.
- *"Justify one hyperparameter choice."* — Pick the SVM's `C`. Table 3 shows the modal
  selection across the 15 folds; `results/selected_hyperparameters.csv` has all 225.
- *"Is your best model really better than the second?"* — No, and that is the finding.
  Quote the Holm-corrected p = 0.096 and the sd of 0.0071.
- *"Regenerate a result."* — `python src/analyse_results.py` rebuilds every table from
  `results/fold_level_results.csv` in seconds. Practise this once before the check.
- *"Why is the random forest fourth?"* — Spam here is marked by abundant, nearly linearly
  separable surface cues (spam averages 139 characters vs 72 for ham; 95% of spam contains
  a digit vs 16% of ham). In a sparse high-dimensional space a hyperplane already describes
  that geometry, so axis-aligned splits spend capacity for nothing.

One more thing: **read the anchor paper** (Salman, Ikram & Kaafar, *IEEE Access* 12:24306–24324,
2024, DOI 10.1109/ACCESS.2024.3364671 — it is open access). Section 2 and Section 6 of the
report engage with its published findings, and the report says plainly which of its claims
this study can and cannot test. If you find specific numbers in its tables worth citing,
add them to Table 1; that will strengthen the "engagement with the anchor paper" criterion,
which is worth 10 marks.

## Deliverables checklist (from Table 1.2 of the brief)

- [x] **Report** — `report/report.pdf`, follows the Chapter 4 headings exactly, 8 pages
- [x] **Code** — `src/`, runs end to end, `requirements.txt` pinned, seed fixed
- [x] **Raw results** — `results/*.csv`, per-fold, per-algorithm, per-metric
- [x] **AI-use statement** — Appendix B of the report, 190 words
- [ ] **Your name, ID, date, download date, repository link** — you
- [ ] **Read the code and the anchor paper** — you
