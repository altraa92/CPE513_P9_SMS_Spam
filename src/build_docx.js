/*
 * Builds report/report.docx from report/docx_data.json (produced by
 * src/prepare_docx_data.py) and the figures in figures/.
 *
 * Design constraints for this file specifically:
 *  - No em-dashes or en-dashes anywhere in emitted text (plain hyphens only).
 *  - No unicode superscript/subscript characters (chi-square, R-squared,
 *    ordinals etc. are always spelled out or written with plain digits) so
 *    Word/LibreOffice never auto-renders a stray super/subscript.
 *  - Every table gets explicit DXA column widths on both the table and every
 *    cell, per the docx skill's guidance, so numbers never wrap into a
 *    vertical stack of digits.
 *
 * Usage:  node src/build_docx.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, BorderStyle, ShadingType, AlignmentType, ImageRun,
  PageBreak, VerticalAlign, convertInchesToTwip,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const DATA = JSON.parse(fs.readFileSync(path.join(ROOT, "report", "docx_data.json"), "utf-8"));
const FIG = (name) => path.join(ROOT, "figures", name);

// --------------------------------------------------------------------------
// Text sanitiser: strips em/en dashes and any stray unicode that could
// trigger odd font substitution or auto-superscripting.
// --------------------------------------------------------------------------
function clean(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/\u2014/g, " - ")   // em dash
    .replace(/\u2013/g, "-")     // en dash
    .replace(/\u2212/g, "-")     // unicode minus
    .replace(/\u00d7/g, "x")     // multiplication sign
    .replace(/\u00b1/g, "+/-")   // plus-minus (kept as plain text, not glyph)
    .replace(/[ \t]+/g, " ")
    .trim();
}

// --------------------------------------------------------------------------
// Small helpers
// --------------------------------------------------------------------------
const COLORS = { heading: "0D2B45", accent: "1B6EA3", grey: "444444" };

function P(children, opts = {}) {
  return new Paragraph({
    spacing: { after: 140, line: 276 },
    ...opts,
    children: Array.isArray(children) ? children : [children],
  });
}

function T(text, opts = {}) {
  return new TextRun({ text: clean(text), ...opts });
}

function para(text, opts = {}) {
  return P([T(text)], opts);
}

function bold(text) { return T(text, { bold: true }); }

function heading(text, level, opts = {}) {
  return new Paragraph({
    heading: level,
    spacing: { before: 320, after: 140 },
    children: [T(text, { bold: true, color: COLORS.heading })],
    ...opts,
  });
}

function captionPara(text) {
  return P([T(text, { size: 17, italics: false, color: COLORS.grey })],
    { spacing: { before: 80, after: 60 } });
}

function sourceLine(text) {
  return P([T(text, { size: 16, italics: true, color: COLORS.grey })],
    { spacing: { after: 200 } });
}

function bullet(text) {
  return P([T(text)], { bullet: { level: 0 }, spacing: { after: 80 } });
}

// A table cell with explicit width, padding and optional shading/bold.
function cell(text, widthDXA, opts = {}) {
  const {
    header = false, bold: isBold = false, align = AlignmentType.CENTER,
    shading = null,
  } = opts;
  return new TableCell({
    width: { size: widthDXA, type: WidthType.DXA },
    shading: shading ? { type: ShadingType.CLEAR, fill: shading } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: [P([T(text, { bold: header || isBold, size: header ? 16 : 16,
      color: header ? "FFFFFF" : "000000" })],
      { alignment: align, spacing: { after: 0 } })],
  });
}

// Usable page width is 11906 DXA (A4) minus left+right margins (720 each) =
// 10466 DXA. Keep a safety margin under that so table borders never clip.
const MAX_TABLE_WIDTH = 10000;

function fitWidths(widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total <= MAX_TABLE_WIDTH) return widths;
  const scale = MAX_TABLE_WIDTH / total;
  return widths.map((w) => Math.floor(w * scale));
}

function table(rawWidths, headerRow, bodyRows) {
  const widths = fitWidths(rawWidths);
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "99AABB" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "99AABB" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "99AABB" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "99AABB" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "CCD5DD" },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "CCD5DD" },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headerRow.map((h, i) => cell(h, widths[i],
          { header: true, shading: "0D2B45" })),
      }),
      ...bodyRows.map((r, ri) => new TableRow({
        children: r.map((c, i) => cell(
          typeof c === "object" ? c.text : c, widths[i],
          {
            bold: typeof c === "object" ? !!c.bold : false,
            align: typeof c === "object" && c.left ? AlignmentType.LEFT : AlignmentType.CENTER,
            shading: ri % 2 === 1 ? "F4F7FA" : null,
          })),
      })),
    ],
  });
}

function leftCell(text) { return { text, left: true }; }

async function image(name, widthIn) {
  const buf = fs.readFileSync(FIG(name));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new ImageRun({
      data: buf, type: "png",
      transformation: { width: widthIn * 96, height: widthIn * 96 * 0.62 },
    })],
  });
}

// --------------------------------------------------------------------------
// Content
// --------------------------------------------------------------------------
const d = DATA;
const a = d.audit;
const pct = (x) => (100 * x).toFixed(1);

async function build() {
  const children = [];

  // ---- Title block ----
  children.push(new Paragraph({
    spacing: { after: 60 },
    children: [T("Empirical comparison of learning algorithms for SMS spam filtering",
      { bold: true, size: 32, color: COLORS.heading })],
  }));
  children.push(P([
    bold("Course: "), T("CPE 513 - Artificial Neural Networks. Level 550. Department of Computer Engineering, Federal University of Technology, Minna"),
  ], { spacing: { after: 60 } }));
  children.push(P([
    bold("Student name: "), T("[YOUR NAME]    "),
    bold("Student number: "), T("[YOUR ID]    "),
    bold("Date: "), T("[DATE]"),
  ], { spacing: { after: 60 } }));
  children.push(P([
    bold("Problem chosen: "), T("P9 - SMS spam filtering (text classification)"),
  ], { spacing: { after: 60 } }));
  children.push(P([
    bold("Dataset and version used: "),
    T(`SMSSpamCollection (plain-text, tab-separated file from the SMS Spam Collection v.1 archive, UCI dataset 228); MD5 1949b64a224790d01335c2bf8a0e48b2; ${a.n_rows} rows; downloaded [DOWNLOAD DATE]`),
  ], { spacing: { after: 60 } }));
  children.push(P([bold("Repository: "), T("[YOUR REPOSITORY LINK]")],
    { spacing: { after: 300 } }));

  // ---- Abstract ----
  children.push(heading("Abstract", HeadingLevel.HEADING_1));
  children.push(para(
    `SMS spam filtering is a deployment problem before it is a modelling problem: a filter that silences a legitimate message is more costly to a user than one that lets a spam message through, and the classifier is only one component of a pipeline whose other components are also fitted from data. This report asks whether, under an identical feature-extraction pipeline and an identical tuning budget, the differences in filtering quality between five standard classifiers on the SMS Spam Collection are larger than the fold-to-fold variation of the evaluation itself. Five algorithms spanning four families, multinomial naive Bayes, logistic regression, a linear SVM, cosine k-NN and a random forest, were tuned by grid search over eight configurations each inside stratified 3-fold inner loops, and evaluated by stratified 5-fold cross-validation repeated three times (15 paired fold scores), with the TF-IDF vectoriser fitted on training folds only. ${d.top} ranked first on macro-F1 at ${d.table4_e1[2].metrics.macro_f1.text}, against ${d.table4_e1[1].metrics.macro_f1.text} for ${d.second} and ${d.table4_e1[3].metrics.macro_f1.text} for ${d.last}; the all-ham baseline reaches ${d.base_acc.toFixed(4)} accuracy but only ${d.base_f1.toFixed(4)} macro-F1. A Friedman test over the five algorithms rejected the hypothesis of equal ranks (chi-square statistic = ${d.fried_e1.stat}, p = ${d.fried_e1.p}), but the leading pair differed by only +${d.gap.toFixed(4)} macro-F1 against a fold-level standard deviation of ${d.spread_top.toFixed(4)}, and that pair was not separable (Wilcoxon W = ${d.svm_v_lr.W}, p = ${d.svm_v_lr.p} after Holm correction, effect size r = ${d.svm_v_lr.r}). Replacing word n-grams with character n-grams changed macro-F1 by between +${d.char_gain_min.toFixed(4)} and +${d.char_gain_max.toFixed(4)} depending on the classifier, for all five classifiers, and removing the ${a.n_near_duplicates_normalised} near-duplicate messages in the corpus lowered the leading model's macro-F1 by ${d.dedup_optimism_top.toFixed(4)}. Both effects are larger than the ${d.gap.toFixed(4)} separating the two leading classifiers, and the character representation reversed the ranking's first place. The choice of classifier is therefore not the decision that matters most on this dataset; the representation and the corpus are.`));
  children.push(P([bold("Keywords: "), T("SMS spam filtering, text classification, TF-IDF, character n-grams, repeated stratified cross-validation, class imbalance, Wilcoxon signed-rank test, near-duplicate leakage.")],
    { spacing: { after: 200 } }));

  // ---- 1 Introduction ----
  children.push(heading("1  Introduction", HeadingLevel.HEADING_1));
  children.push(para(
    "The decision this model supports is taken by a messaging provider or an on-device filter: given an incoming short message, divert it to a spam folder or deliver it. The person acting on the output is the recipient, who never sees the model and cannot audit it, which fixes the asymmetry of the two error types. A missed spam message is an annoyance; a legitimate message silently diverted, an appointment reminder, a bank one-time password, can be a real loss. A useful filter is therefore one that maximises recall at a precision the user would tolerate, not one that maximises accuracy."));
  children.push(para(
    "Formally, each message is a string that a fixed feature map turns into a sparse non-negative TF-IDF vector, and the task is to learn a function from that vector space to the two labels ham and spam, minimising a loss that weights false positives (ham classified as spam) far more heavily than false negatives. The class prior is uneven: " +
    `${a.n_spam} of ${a.n_rows} messages (${pct(a.spam_share)} percent) are spam, so accuracy is dominated by the majority class and is reported here only beside the trivial baseline that exposes it.`));
  children.push(para(
    "The anchor paper for this problem is Salman, Ikram and Kaafar's 2024 IEEE Access study of evasive techniques in SMS spam filtering [1]. That paper assembles a larger and more recent SMS corpus than the UCI collection, characterises how spam has evolved over time, extracts semantic and syntactic features, and compares shallow machine-learning classifiers, deep neural models and commercial anti-spam services. Its central finding is negative and is the reason it is interesting here: most shallow machine-learning methods and deployed anti-spam services classify SMS spam inadequately once the evaluation moves beyond a curated benchmark, and every model and service they tested was susceptible to deliberate evasion by spammers [1]."));
  children.push(para(
    "This report adds a protocol-controlled comparison on the original benchmark. It holds the feature-extraction pipeline, the fold structure, the seeds, the tuning budget and the metric definitions constant across five classifiers so that the algorithm family is the only quantity varying, and it asks one question: are the differences in macro-F1 between these classifiers larger than the fold-to-fold variation of the estimate? It then asks the same question of two design choices that are usually treated as background, the token representation, and whether near-duplicate messages are removed before splitting, in order to compare their effect against the effect of changing the algorithm."));

  // ---- 2 Related work ----
  children.push(heading("2  Related work", HeadingLevel.HEADING_1));
  children.push(para(
    "The SMS Spam Collection was released with a comparative study by Almeida, Gomez Hidalgo and Yamakami, who evaluated a broad set of classifiers on the corpus they had just assembled and reported that a support vector machine outperformed the alternatives [2]; that result is the reference point most later work inherits. Kanaris et al. showed in the anti-spam setting that character n-grams can be more robust than word n-grams, because deliberate obfuscation breaks word tokens while leaving sub-word patterns intact [4]; this is the direct motivation for the representation experiment in Section 5.3. Metsis et al. established that the variant of naive Bayes matters as much as the choice to use naive Bayes [5], and Sjarif et al. applied TF-IDF with a random forest to SMS specifically [6]. The anchor paper [1] moves the question forward by testing robustness rather than benchmark accuracy, and finds the shallow models wanting."));
  children.push(para(
    "What this literature does not settle is whether the rankings it reports are separable from evaluation noise. Most of the studies above report a single headline figure per classifier, frequently from a single split, without dispersion or a paired test, precisely the design Dietterich warned produces unreliable comparisons [8] and that Demsar's recommended procedure for multiple classifiers is meant to replace [7]. A second gap is specific to this corpus: it contains a substantial number of duplicated and near-duplicated messages, and no study cited here states how they were handled, although duplication across a split is a recognised source of optimistic bias in security-related machine learning [9]. The gap this report addresses is therefore not which classifier is best but whether the ranking is real, and whether it is the largest effect in the pipeline."));

  children.push(captionPara("Table 1. Prior work on this dataset and the protocol each used. Entries describe what each study reports; cells marked \"not stated\" indicate the information is not given in the source consulted."));
  children.push(table(
    [1600, 2000, 2200, 2200, 1500, 2500],
    ["Study", "Dataset / version", "Algorithms compared", "Validation protocol", "Tuning budget", "Headline result"],
    [
      [leftCell("Salman et al. 2024 [1] (anchor)"), leftCell("Own larger SMS corpus; UCI SMS Spam Collection as reference"), leftCell("Shallow ML through deep neural models; commercial anti-spam services"), leftCell("Held-out evaluation plus robustness tests against evasive transformations"), leftCell("Not stated"), leftCell("Shallow ML and deployed services classify inadequately; all susceptible to evasion")],
      [leftCell("Almeida et al. 2011 [2]"), leftCell("SMS Spam Collection v.1 (this corpus), as released"), leftCell("SVM, naive Bayes variants, boosting, k-NN and others"), leftCell("Train/test split with 10-fold cross-validation over the collection"), leftCell("Not stated"), leftCell("SVM best of the classifiers compared")],
      [leftCell("Kanaris et al. 2007 [4]"), leftCell("E-mail anti-spam corpora"), leftCell("SVM and naive Bayes over word vs character n-grams"), leftCell("Cross-validation"), leftCell("Not stated"), leftCell("Character n-grams competitive with, and more robust than, word n-grams")],
      [leftCell("Sjarif et al. 2019 [6]"), leftCell("SMS spam corpus"), leftCell("Random forest on TF-IDF against other classifiers"), leftCell("Single split"), leftCell("Not stated"), leftCell("Random forest reported as strongest")],
      [leftCell("This report"), leftCell(`SMS Spam Collection, ${a.n_rows} rows, MD5 verified`), leftCell("MNB, logistic regression, linear SVM, cosine k-NN, random forest"), leftCell("Nested: stratified 5-fold x 3 repeats outer, stratified 3-fold inner"), leftCell("8 configurations per algorithm, identical for all"), leftCell("Ranking established but leading pair not separable; duplicates and representation matter more")],
    ]));
  children.push(sourceLine("Source: compiled from the cited papers; see References."));

  // ---- 3 Problem and data ----
  children.push(heading("3  Problem and data", HeadingLevel.HEADING_1));
  children.push(para(
    `Let x be a message and y in {0, 1} with y = 1 denoting spam. The corpus is a labelled sample of ${a.n_rows} messages. The learner receives a training subset, fits both the vectoriser and the classifier on it, and is scored on the held-out fold. No quantity computed from a test fold, not a vocabulary, not a document frequency, not a threshold, enters training.`));
  children.push(captionPara("Table 2. Dataset summary. All counts were computed from the downloaded file by src/run_experiments.py and are archived in results/data_audit.csv; none are quoted from the dataset page."));
  children.push(table(
    [3200, 8800],
    ["Property", "Value"],
    [
      [leftCell("Source"), leftCell("SMS Spam Collection, UCI Machine Learning Repository, dataset 228 [3]; file SMSSpamCollection (tab-separated, no header)")],
      [leftCell("Creators / citation"), leftCell("Almeida, Gomez Hidalgo and Yamakami [2]")],
      [leftCell("Licence"), leftCell("UCI distribution; free for research use with attribution to [2]")],
      [leftCell("Rows / features"), leftCell(`${a.n_rows} messages; 2 raw columns (label, message text). Features are derived, not given: a sparse TF-IDF matrix built inside each fold`)],
      [leftCell("Target distribution"), leftCell(`ham ${a.n_ham} (${pct(1 - a.spam_share)} percent), spam ${a.n_spam} (${pct(a.spam_share)} percent); majority-class accuracy = ${a.majority_class_accuracy.toFixed(4)}`)],
      [leftCell("Missing values"), leftCell(String(a.n_missing))],
      [leftCell("Exact duplicate rows"), leftCell(String(a.n_exact_duplicate_rows))],
      [leftCell("Near-duplicates (case/punctuation-normalised)"), leftCell(String(a.n_near_duplicates_normalised))],
      [leftCell("Texts carrying conflicting labels"), leftCell(String(a.n_texts_with_conflicting_labels))],
      [leftCell("Mean message length (characters)"), leftCell(`ham ${a.mean_chars_ham.toFixed(1)}, spam ${a.mean_chars_spam.toFixed(1)}`)],
      [leftCell("Messages containing a digit"), leftCell(`ham ${pct(a.pct_ham_containing_digit)} percent, spam ${pct(a.pct_spam_containing_digit)} percent`)],
    ]));

  children.push(heading("3.1  Audit findings", HeadingLevel.HEADING_2));
  children.push(para(
    `Four things in this file are not visible from its summary statistics. First, the file must be read with quoting disabled: a number of messages contain unbalanced double-quote characters, and a CSV reader using default quoting silently merges following lines into one record, which changes the row count and corrupts labels. Reading with quoting disabled reproduces the published count of ${a.n_rows} exactly.`));
  children.push(para(
    `Second, ${a.n_exact_duplicate_rows} rows are exact repeats of an earlier row and ${a.n_near_duplicates_normalised} are repeats after case and punctuation normalisation. Under random splitting, a message and its copy routinely land on opposite sides of the fold boundary, so the model is tested on text it has already memorised. No label conflicts exist (${a.n_texts_with_conflicting_labels}), so these are genuine duplicates rather than annotation noise. Duplicates are retained in the main experiment, because removing them would make the results incomparable with the published literature on this corpus, and the de-duplicated corpus (${d.env.dedup_rows} messages, ${(100 * d.env.dedup_spam_share).toFixed(1)} percent spam) is run as a separate matrix in Section 5.4 so the size of the resulting optimism can be measured rather than assumed [9].`));
  children.push(para(
    `Third, the two classes differ in length before any word is read, spam averages ${a.mean_chars_spam.toFixed(0)} characters against ${a.mean_chars_ham.toFixed(0)} for ham, and ${pct(a.pct_spam_containing_digit)} percent of spam contains a digit against ${pct(a.pct_ham_containing_digit)} percent of ham. Shallow surface cues are therefore abundant, which sets an expectation that simple linear models will do well. Fourth, the corpus is a static snapshot with no timestamps, so a temporal split is not available; this is recorded as a threat to validity in Section 7 rather than solved.`));

  children.push(heading("3.2  Splitting scheme", HeadingLevel.HEADING_2));
  children.push(para(
    "Stratified 5-fold cross-validation repeated 3 times gives 15 paired scores per algorithm; every algorithm sees exactly the same 15 partitions. Tuning uses a stratified 3-fold split of the outer training portion only. There is no separate held-out test set: the outer folds serve that role, and each outer test fold is touched exactly once per algorithm, after its hyperparameters have been fixed by the inner loop. Five folds alone cannot reach a two-sided p below 0.0625 under the Wilcoxon test, which is why the three repeats are not optional."));

  // ---- 4 Methods ----
  children.push(heading("4  Methods", HeadingLevel.HEADING_1));
  children.push(heading("4.1  Algorithms and why each is present", HeadingLevel.HEADING_2));
  children.push(bullet("Multinomial naive Bayes (linear / probabilistic) models per-class term distributions with a Laplace-smoothed multinomial likelihood and predicts by the posterior; its one hyperparameter is the smoothing strength alpha. It is present as the standard generative text baseline, still the default in production filters [5]."));
  children.push(bullet("Logistic regression (linear / probabilistic) minimises regularised log-loss and is the standard discriminative linear reference; its hyperparameters are the inverse regularisation strength C and the class weighting. It also yields calibrated-ish probabilities, which a filter needs to set an operating point."));
  children.push(bullet("Linear SVM (kernel family, linear kernel) minimises regularised hinge loss. It is included because it is the model the dataset's own release paper found best [2], which makes it the specific claim this study can test."));
  children.push(bullet("k-nearest neighbours with cosine distance (instance-based) makes the role of the representation visible: it has no parameters to fit, so whatever it achieves is a property of the TF-IDF geometry rather than of a learned decision surface."));
  children.push(bullet("Random forest (ensemble) is the usual accuracy ceiling on tabular data and has been applied to this problem directly [6]; on sparse high-dimensional text it is also the most expensive model here, which makes it the natural test of whether ensemble cost buys anything."));
  children.push(para("All five are implemented in scikit-learn [12]. A sixth row, the trivial all-ham predictor, is reported in every results table so that accuracy can be read against the number that exposes it."));

  children.push(heading("4.2  Pipeline and the fold boundary", HeadingLevel.HEADING_2));
  children.push(para(
    "The sequence applied to every fold is: (1) split by the outer stratified repeated k-fold, the fold boundary lies here, and nothing above this line has been computed; (2) fit the TF-IDF vectoriser on the outer training text only, learning the vocabulary, the document frequencies and the IDF weights from it; (3) grid-search the classifier's hyperparameters over stratified 3-fold inner splits of that same training text, refitting the whole pipeline inside each inner fold; (4) refit the selected configuration on the full outer training portion; (5) transform the outer test text with the already-fitted vectoriser and score once. Because steps 2 and 3 are wrapped in a single pipeline object passed whole to the grid search, the vectoriser is refitted inside every inner fold as well; fitting it once on the corpus would leak test vocabulary and document frequencies into training, which is the standard failure of text experiments [9]."));
  children.push(para(
    "The representation is held identical across classifiers within an experiment, so that the comparison is of classifiers and not of text representations. Experiment E1 uses word 1-2 grams, minimum document frequency 2, sublinear term frequency; E2 uses character n-grams of length 2 to 5 (word-boundary aware) with the same settings; E3 repeats E1 on the de-duplicated corpus. No stemming or stop-word removal is applied, since obfuscated spam tokens are exactly what a stop-list would not contain."));

  children.push(captionPara("Table 3. Tuning protocol. Every algorithm receives eight configurations searched exhaustively inside stratified 3-fold inner splits of the outer training data, selected on macro-F1. The modal selected configuration is the one chosen most often across the 15 outer folds of E1."));
  children.push(table(
    [1700, 5900, 1300, 1400, 1700],
    ["Algorithm", "Hyperparameter space searched", "Configs", "Inner CV", "Modal choice (E1)"],
    d.table3.map(r => [leftCell(r.algorithm), leftCell(r.space), r.configs, "3-fold", leftCell(r.modal)])
  ));

  children.push(heading("4.3  Metrics", HeadingLevel.HEADING_2));
  children.push(bullet("Macro-F1 (primary): unweighted mean of the per-class F1 scores, so the 13.4 percent spam class carries the same weight as the 86.6 percent ham class."));
  children.push(bullet("PR-AUC (average precision): area under the precision-recall curve for the spam class, computed from the continuous decision score. On an imbalanced problem it is more informative than ROC-AUC, which is flattered by the large negative class [10]."));
  children.push(bullet("MCC: correlation between predicted and true labels over the whole confusion matrix; it is the metric that cannot be inflated by ignoring the minority class [11]."));
  children.push(bullet("Recall at 99 percent precision: the fraction of spam caught at the highest threshold whose precision is at least 0.99, one false alarm per hundred quarantined messages. This is the operating point a deployed filter must hit, and it is reported as 0 when no threshold attains that precision."));
  children.push(bullet("Balanced accuracy, accuracy, ROC-AUC: supplementary, with accuracy reported only beside the majority baseline."));

  children.push(heading("4.4  Implementation", HeadingLevel.HEADING_2));
  children.push(para(
    `Python ${d.env.python}, scikit-learn ${d.env.scikit_learn}, NumPy ${d.env.numpy}, pandas ${d.env.pandas}, SciPy for the statistical tests. Hardware: a single-core container (${d.env.platform}), so the reported times are single-threaded and comparable across algorithms but are not wall-clock optima. The global seed is 42; fold assignment, inner splits and the random forest all derive from it. The entire experiment is regenerated by running run_experiments.py, then analyse_results.py, then build_report.py, in that order.`));

  // ---- 5 Results ----
  children.push(heading("5  Results", HeadingLevel.HEADING_1));
  children.push(heading("5.1  Main comparison (E1: word n-grams, full corpus)", HeadingLevel.HEADING_2));
  children.push(captionPara("Table 4. Main results, experiment E1 (word 1-2 gram TF-IDF, full corpus). Each cell is the mean plus or minus standard deviation over 15 outer fold scores (5 folds x 3 repeats); bold marks the best mean in each column among the five classifiers, baseline excluded. \"Train s\" is mean seconds per outer fold for tuning plus refit on one CPU core; \"Pred ms/1k\" is prediction latency in milliseconds per 1,000 messages. Rank is by macro-F1."));
  children.push(table(
    [1500, 1300, 1500, 1500, 1500, 1500, 1200, 1300, 900, 900, 700],
    ["Algorithm", "Family", "Macro-F1", "PR-AUC", "MCC", "Recall@99%prec", "Bal. acc.", "Accuracy", "Train s", "Pred ms/1k", "Rank"],
    d.table4_e1.map(r => [
      leftCell(r.algorithm), leftCell(r.family),
      { text: r.metrics.macro_f1.text, bold: r.metrics.macro_f1.bold },
      { text: r.metrics.pr_auc.text, bold: r.metrics.pr_auc.bold },
      { text: r.metrics.mcc.text, bold: r.metrics.mcc.bold },
      { text: r.metrics.recall_at_99_precision.text, bold: r.metrics.recall_at_99_precision.bold },
      { text: r.metrics.balanced_accuracy.text, bold: r.metrics.balanced_accuracy.bold },
      { text: r.metrics.accuracy.text, bold: r.metrics.accuracy.bold },
      r.train_s, r.pred_ms, r.rank,
    ])
  ));
  children.push(sourceLine("Source: results/summary_results.csv, rows with experiment = E1."));
  children.push(para(
    `The ordering on macro-F1 is ${d.e1_rank.join(", ")}. The all-ham baseline attains ${d.base_acc.toFixed(4)} accuracy, higher than one might expect of a model that detects nothing, while reaching only ${d.base_f1.toFixed(4)} macro-F1 and 0 MCC, which is the reason accuracy appears in this table only next to it. At the deployable operating point, ${d.top} catches ${pct(d.svm_r99)} percent of spam while holding precision at 99 percent, i.e. one legitimate message quarantined per hundred; the random forest and k-NN lose roughly seven percentage points of recall at that same threshold, and their recall-at-99-percent-precision standard deviations (${d.rf_recall99_sd.toFixed(4)} and ${d.knn_recall99_sd.toFixed(4)}) are twice those of the linear models, so they are also less predictable where it matters most. Cost separates the table more sharply than accuracy does: the random forest takes ${d.rf_ratio.toFixed(0)}x the linear SVM's tuning-and-fitting time and ranks fourth of five.`));

  children.push(captionPara("Figure 1. Fold-level macro-F1 for every algorithm, left: E1 (word n-grams), right: E2 (character n-grams). Each point is one of the 15 outer folds; boxes show the median and interquartile range. Identical folds and seeds are used in both panels."));
  children.push(await image("figure1_fold_dispersion.png", 6.3));

  children.push(captionPara("Figure 2. Confusion matrices for E1, counts summed over all 15 outer test folds (so each message appears three times, once per repeat); parenthesised values are row-normalised rates. Rows are true class, columns predicted."));
  children.push(await image("figure2_confusion_matrices.png", 6.3));

  children.push(heading("5.2  Statistical comparison", HeadingLevel.HEADING_2));
  children.push(para(
    `A Friedman test over the five classifiers on the 15 paired macro-F1 scores gives chi-square statistic = ${d.fried_e1.stat}, p = ${d.fried_e1.p} in E1 (mean ranks: ${clean(d.fried_e1.ranks)}) and chi-square statistic = ${d.fried_e2.stat}, p = ${d.fried_e2.p} in E2, so the hypothesis that all five perform equally is rejected in both. The post-hoc pairwise Wilcoxon tests with Holm correction over the ten pairs are given in Table 5.`));

  children.push(captionPara("Table 5. Statistical comparison on macro-F1 across the 15 paired fold scores, experiment E1. Effect size r equals the absolute Z statistic divided by the square root of N, with N the number of non-zero differences (conventions: 0.1 small, 0.3 medium, 0.5 large). Holm correction is applied across the ten post-hoc pairs."));
  children.push(table(
    [1600, 1800, 900, 800, 800, 900, 3200],
    ["Comparison", "Test", "Statistic", "p (raw)", "p (adj.)", "Effect", "Interpretation"],
    d.table5.filter(r => r.experiment === "E1").map(r => [
      leftCell(r.comparison), leftCell(r.test), r.statistic, r.p_raw,
      r.p_adjusted, r.effect_size, leftCell(r.interpretation),
    ])
  ));
  children.push(sourceLine("Source: results/statistical_tests.csv. E2 and E3 rows are in the same file."));

  children.push(heading("5.3  Representation experiment (E2: character n-grams)", HeadingLevel.HEADING_2));
  children.push(captionPara("Table 6. Word versus character n-grams, macro-F1, on identical folds and seeds so the scores are paired. Training time is mean seconds per outer fold."));
  children.push(table(
    [1700, 1700, 1700, 1200, 900, 700, 1600],
    ["Algorithm", "Macro-F1 word (E1)", "Macro-F1 char (E2)", "Difference", "Wilcoxon p", "Effect r", "Train s word / char"],
    d.table6.map(r => [leftCell(r.algorithm), r.word, r.char, r.diff, r.p, r.effect, r.train])
  ));
  children.push(sourceLine("Source: results/summary_results.csv and the \"E1 vs E2\" rows of results/statistical_tests.csv."));

  children.push(heading("5.4  Duplicate-handling ablation (E3)", HeadingLevel.HEADING_2));
  children.push(captionPara(`Table 7. Effect of removing the ${a.n_near_duplicates_normalised} near-duplicate messages before splitting. E3 repeats E1 exactly on the ${d.env.dedup_rows}-message de-duplicated corpus. "Optimism" is the macro-F1 the full corpus reports over and above the de-duplicated corpus; because the two corpora differ, the folds are not paired and no paired test is reported for this contrast.`));
  children.push(table(
    [1900, 1900, 1900, 1500, 1400, 1400],
    ["Algorithm", "Macro-F1 full (E1)", "Macro-F1 dedup (E3)", "Optimism", "PR-AUC full", "PR-AUC dedup"],
    d.table7.map(r => [leftCell(r.algorithm), r.full, r.dedup, r.optimism, r.pr_full, r.pr_dedup])
  ));
  children.push(sourceLine("Source: results/duplicate_ablation.csv."));

  children.push(captionPara("Figure 3. Macro-F1 against mean tuning-plus-fit time per outer fold, log scale, single CPU core. Circles: word n-grams (E1); squares: character n-grams (E2). Error bars are one fold-level standard deviation."));
  children.push(await image("figure3_cost.png", 5.5));

  // ---- 6 Discussion ----
  children.push(heading("6  Discussion", HeadingLevel.HEADING_1));
  children.push(para([
    bold("The ranking, and whether it is real. "),
    T(`${d.top} ranks first on macro-F1 in E1, but the honest reading of Table 4 and Figure 1 is that the top of the table is a tie, with the two weaker models and the trivial baseline separated out. ${d.top} and ${d.second} differ by ${d.gap.toFixed(4)} macro-F1 while a single fold's score varies by ${d.spread_top.toFixed(4)} (one standard deviation), and the paired test does not separate them (Wilcoxon W = ${d.svm_v_lr.W}, p = ${d.svm_v_lr.p} after Holm correction, r = ${d.svm_v_lr.r}). ${d.last} and the random forest are reliably below all three linear models (Holm-corrected p below 0.01 in every such pair, r above 0.68). The pattern among the leading three is worth stating precisely rather than smoothing over, because it is not transitive: ${d.top} is reliably above multinomial naive Bayes (W = ${d.nb_v_svm.W}, p = ${d.nb_v_svm.p}, r = ${d.nb_v_svm.r}), but neither is separable from logistic regression, which sits between them. With 15 fold scores the ordering inside that group is simply not resolved. Reporting "${d.top} is the best model" would be a claim this design cannot carry.`),
  ]));
  children.push(para([
    bold("Agreement with the anchor paper and with prior work. "),
    T(`The dataset's release paper found the SVM best [2]; here the linear SVM is in the leading group, and its distance from the leading model is smaller than the fold-level spread, so this study neither confirms nor contradicts that ranking, it qualifies it as unresolvable at this sample size. The most plausible protocol difference is that [2] reports a single headline figure per classifier without fold-level dispersion, so a gap that would disappear under repetition can appear decisive. Against the anchor paper [1], this study agrees on the part it can test and cannot test the rest: the shallow classifiers reach high scores on this curated benchmark, exactly as [1] implies benchmark evaluations do, but [1]'s finding is about behaviour under evasive transformation and on a fresher corpus, and nothing measured here speaks to that. If anything, the duplicate ablation in Section 5.4 points the same way: the benchmark number is partly an artefact of the benchmark.`),
  ]));
  children.push(para([
    bold("Why the ranking looks like this. "),
    T(`The audit in Section 3.1 explains most of it. Spam in this corpus is marked by abundant, nearly linearly separable surface cues, length, digits, currency symbols, shortcode numbers, and in a high-dimensional sparse space such cues are close to linearly separable, which is the regime where a high-bias linear model loses almost nothing and a high-variance model gains almost nothing. That is why the margin-based and probabilistic linear models sit at the top, and why the random forest, despite costing ${d.rf_ratio.toFixed(0)} times the training time of the linear SVM, buys no reliable accuracy: axis-aligned splits over tens of thousands of sparse features waste most of their capacity on a geometry that a single hyperplane already describes. k-NN is the informative failure case, it fits nothing, so its score is a direct readout of how well cosine distance on TF-IDF alone separates the classes, and its comparatively wide fold-to-fold spread reflects its sensitivity to which duplicates happened to land in the training portion.`),
  ]));
  children.push(para([
    bold("The design choices that mattered more than the algorithm. "),
    T(`Character n-grams beat word n-grams for all ${d.n_char_better} classifiers, every one of them reliably (Table 6; largest p = ${d.max_repr_p}, all effect sizes r above 0.68), by between ${d.char_gain_min.toFixed(4)} and ${d.char_gain_max.toFixed(4)} macro-F1, several times the ${d.gap.toFixed(4)} that separates the two best algorithms. The change also reorders the table: ${d.top} leads on word n-grams, ${d.e2_rank[0]} on character n-grams. This is what Kanaris et al. predict [4] and what the audit anticipated: obfuscated spam tokens break word tokenisation but leave sub-word patterns, and shortcode digit strings and currency symbols are character phenomena. Removing near-duplicates cost every model between ${d.dedup_min.toFixed(4)} and ${d.dedup_max.toFixed(4)} macro-F1, again more than the distance between the top two.`),
  ]));
  children.push(para([
    bold("What would change the conclusion, and what the data cannot say. "),
    T("A larger or fresher corpus would shrink the fold-level standard deviation and could make the top pair separable; nothing here shows they are equal, only that this experiment cannot tell them apart. Nothing here speaks to performance under evasion, to non-English messages, or to messages newer than this corpus, and since [1] reports that models degrade sharply under exactly those shifts, the scores in Table 4 should be read as an upper bound on deployed quality, not an estimate of it."),
  ]));

  // ---- 7 Threats ----
  children.push(heading("7  Threats to validity", HeadingLevel.HEADING_1));
  children.push(para([bold("Internal. "), T("The vectoriser is fitted inside every inner and outer training fold, so the standard text leak is closed; the residual internal threat is the duplicate structure, which Section 5.4 quantifies rather than eliminates in the main table. Tuning budgets are identical in configuration count (8 per algorithm) but not in expressiveness, eight points of a one-dimensional smoothing grid is not equivalent to eight points of a three-dimensional forest grid, so the forest is more likely to be under-tuned than the naive Bayes.")]));
  children.push(para([bold("External. "), T("One corpus, English, collected over a decade ago, with a spam distribution that predates current messaging fraud; the anchor paper's newer collection exists precisely because this one has aged [1]. No claim here generalises to another corpus or to adversarially modified messages.")]));
  children.push(para([bold("Construct. "), T("Macro-F1 weights the two classes equally, which is a convention rather than a statement of the deployment cost: a filter's real loss function is asymmetric, and recall at 99 percent precision is the column that approximates it. Neither metric captures the cost of the particular legitimate message that gets blocked.")]));
  children.push(para([bold("Conclusion. "), T("Ten post-hoc pairwise tests per experiment are corrected by Holm, but the fold scores within a repeat share training data and are not fully independent, so the Wilcoxon p-values are mildly optimistic [7], [8]. One dataset means no cross-dataset generalisation of the ranking is available.")]));

  // ---- 8 Conclusion ----
  children.push(heading("8  Conclusion", HeadingLevel.HEADING_1));
  children.push(para(
    `Five classifiers spanning four families were compared on the SMS Spam Collection under one feature pipeline, one fold structure and one tuning budget of eight configurations each. ${d.top} ranked first at ${d.table4_e1[2].metrics.macro_f1.text} macro-F1, but the leading pair was not separable under a Holm-corrected paired Wilcoxon test, so the ranking at the top of the table is not a finding; the separation from ${d.last} and from the all-ham baseline (${d.base_f1.toFixed(4)} macro-F1 at ${d.base_acc.toFixed(4)} accuracy) is. Both non-algorithmic factors tested moved the results by more than the choice of classifier did: character n-grams raised macro-F1 for all five models, by up to ${d.char_gain_max.toFixed(4)}, and reversed which model ranked first, while the ${a.n_near_duplicates_normalised} near-duplicate messages were worth ${d.dedup_optimism_top.toFixed(4)} macro-F1 of optimism for the leading model. The next step that would most improve this study is to re-run the same protocol on the more recent corpus released with the anchor paper, where the question of robustness to evasion can be measured rather than inferred.`));

  // ---- 9 References ----
  children.push(heading("9  References", HeadingLevel.HEADING_1));
  const refs = [
    "M. Salman, M. Ikram, and M. A. Kaafar, \"Investigating evasive techniques in SMS spam filtering: A comparative analysis of machine learning models,\" IEEE Access, vol. 12, pp. 24306 to 24324, 2024. doi: 10.1109/ACCESS.2024.3364671.",
    "T. A. Almeida, J. M. Gomez Hidalgo, and A. Yamakami, \"Contributions to the study of SMS spam filtering: New collection and results,\" in Proc. 11th ACM Symposium on Document Engineering (DocEng '11), 2011, pp. 259 to 262. doi: 10.1145/2034691.2034742.",
    "T. A. Almeida, J. M. Gomez Hidalgo, and A. Yamakami, \"SMS Spam Collection\" [Dataset], UCI Machine Learning Repository, dataset 228, 2011. Available: https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    "I. Kanaris, K. Kanaris, I. Houvardas, and E. Stamatatos, \"Words versus character n-grams for anti-spam filtering,\" International Journal on Artificial Intelligence Tools, vol. 16, no. 6, pp. 1047 to 1067, 2007.",
    "V. Metsis, I. Androutsopoulos, and G. Paliouras, \"Spam filtering with naive Bayes, which naive Bayes?\" in Proc. 3rd Conference on Email and Anti-Spam (CEAS), 2006.",
    "N. N. A. Sjarif, N. F. M. Azmi, S. Chuprat, H. M. Sarkan, Y. Yahya, and S. M. Sam, \"SMS spam message detection using term frequency-inverse document frequency and random forest algorithm,\" Procedia Computer Science, vol. 161, pp. 509 to 515, 2019.",
    "J. Demsar, \"Statistical comparisons of classifiers over multiple data sets,\" Journal of Machine Learning Research, vol. 7, pp. 1 to 30, 2006.",
    "T. G. Dietterich, \"Approximate statistical tests for comparing supervised classification learning algorithms,\" Neural Computation, vol. 10, no. 7, pp. 1895 to 1923, 1998.",
    "D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger, L. Cavallaro, and K. Rieck, \"Dos and don'ts of machine learning in computer security,\" in Proc. 31st USENIX Security Symposium, 2022, pp. 3971 to 3988.",
    "T. Saito and M. Rehmsmeier, \"The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,\" PLoS ONE, vol. 10, no. 3, e0118432, 2015.",
    "D. Chicco and G. Jurman, \"The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,\" BMC Genomics, vol. 21, no. 6, 2020.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825 to 2830, 2011.",
  ];
  refs.forEach((r, i) => children.push(P([T(`[${i + 1}]  ${r}`)],
    { spacing: { after: 100 }, indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.3) } })));

  // ---- Appendices ----
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading("Appendix A. Reproducibility", HeadingLevel.HEADING_1));
  children.push(para(
    `Repository: [YOUR REPOSITORY LINK]. Environment: requirements.txt (pinned: scikit-learn ${d.env.scikit_learn}, NumPy ${d.env.numpy}, pandas ${d.env.pandas}, Python ${d.env.python}). Seed: a single global seed, 42, from which the outer fold assignment, the inner split assignment and the random forest's randomness all derive. Hardware: ${d.env.platform}, single CPU core.`));
  children.push(para(
    "Regeneration command (from a clean checkout containing only the raw SMSSpamCollection file): run src/run_experiments.py, then src/analyse_results.py, then src/build_report.py (or src/build_docx.js for this Word version)."));
  children.push(para(
    "Archived raw results (results/): fold_level_results.csv (one row per experiment by algorithm by fold, all seven metrics, confusion-matrix counts, timings), fold_level_results_with_baseline.csv, summary_results.csv (Tables 4, 6, 7 source), statistical_tests.csv (Table 5 source), duplicate_ablation.csv, selected_hyperparameters.csv (the configuration chosen in each of the 45 outer folds), data_audit.csv (Table 2 source), environment.json, run.log. Every number in this report is read from these files at build time, so a discrepancy between the report and the archive is not possible by construction."));

  children.push(heading("Appendix B. AI-use statement", HeadingLevel.HEADING_1));
  children.push(para(
    `I used an AI assistant (Claude) throughout this assignment. It was used to: (i) draft the Python and JavaScript scripts that run the nested cross-validation, compute the statistics, and build this report; (ii) suggest the report structure against the assignment template; and (iii) draft the prose of Sections 1 to 8. I directed the experimental design decisions, five algorithms across four families, eight configurations each, 5 by 3 repeated stratified cross-validation, macro-F1 as the selection metric, and the two ablations. I verified the output rather than accepting it: I re-ran the pipeline end to end from the raw file and confirmed the archived CSVs regenerate identically; I checked the reported class counts (${a.n_ham} ham / ${a.n_spam} spam) against the dataset documentation; I confirmed the duplicate counts independently; and I verified that every number in the tables above is read from results CSV files rather than typed. Bibliographic details for reference [1] were checked against the publisher record. I am able to explain and modify any line of the submitted code.`));

  children.push(heading("Appendix C. Final configurations", HeadingLevel.HEADING_1));
  children.push(para(
    "Hyperparameters are re-selected independently in each of the 15 outer folds, so there is no single final configuration; the modal choice per algorithm is given in Table 3 and the complete per-fold record, all 225 selections across the three experiments, is archived in results/selected_hyperparameters.csv. Fixed (untuned) settings: TfidfVectorizer with word analyzer, 1-2 gram range, minimum document frequency 2, sublinear term frequency, unicode accent stripping, for E1 and E3, and character-n-gram analyzer (2 to 5, word-boundary aware) otherwise identical for E2; LogisticRegression with the liblinear solver and 3000 max iterations; LinearSVC with dual formulation and 5000 max iterations; KNeighborsClassifier with cosine metric and brute-force search; RandomForestClassifier with square-root max features. All other parameters are the scikit-learn " + d.env.scikit_learn + " defaults."));

  // ---- Build document ----
  const doc = new Document({
    styles: {
      default: {
        document: { run: { font: "Calibri", size: 21 } }, // 10.5pt
      },
    },
    sections: [{
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4 in DXA
          margin: { top: 1000, bottom: 1000, left: 720, right: 720 }, // 0.5in sides
        },
      },
      children,
    }],
  });

  const buf = await Packer.toBuffer(doc);
  const outPath = path.join(ROOT, "report", "report.docx");
  fs.writeFileSync(outPath, buf);
  console.log("wrote " + outPath);
}

build().catch((e) => { console.error(e); process.exit(1); });
