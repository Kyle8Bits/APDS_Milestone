# Task 3 — Implementation Plan

> **Scope:** Cosmetics/Beauty review classification — predicting `is_a_buyer`
> from review text + auxiliary product information.
> **File to edit:** [`data/task2_3.ipynb`](../data/task2_3.ipynb), Section "Task 3"
> (currently a single mega-cell at `cell-22` that needs to be broken up).
> **Marks at stake:** 9% (7% implementation + 2% notebook presentation).
> **Target grade:** HD per [`docs/RUBRIC.md`](RUBRIC.md). HD requires
> *"tasks fully investigated **before** modeling"*, *"strong experiments and
> evaluations"*, and *"strong justification and analysis"* — these phrases
> drive the design choices below (esp. §3.0.5, §3.1.5, §3.5, §4.5).

---

## 1. How Task 3 fits into the milestone pipeline

The three tasks form a **strict producer → consumer chain**. Each task's
outputs are inputs to the next task, and Task 3's correctness depends on
Tasks 1 and 2 having been done properly.

```
┌─────────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
│   TASK 1 — Cleaning     │───▶│  TASK 2 — Vectorisation  │───▶│  TASK 3 — Classification│
│  task1.ipynb            │    │  task2_3.ipynb (top)     │    │  task2_3.ipynb (bottom) │
└─────────────────────────┘    └──────────────────────────┘    └─────────────────────────┘

Produces:                       Produces:                        Produces:
- output/processed.csv          - output/count_vectors.txt       - 5-fold CV result tables
  (cleaned review_text +          (sparse BoW, vocab.txt indices)  (Q1 and Q2)
  all original metadata)        - output/unweighted_vectors.txt  - Discussion / conclusion
- output/vocab.txt                (FastText 300d, sum)             markdown
  (8,054 unigrams,              - output/weighted_vectors.txt
  alphabetical, indexed 0..N)     (FastText 300d, TF-IDF weighted)
```

### What each upstream task gives Task 3

| Artefact | Built in | What Task 3 uses it for |
|---|---|---|
| `output/processed.csv` | Task 1 | Source of `is_a_buyer` labels, `review_id` (for alignment), `review_title`, `brand_name`, `price`, `avg_product_rating`, `product_rating_count`, `product_title`, `review_rating`. The cleaned `review_text` column is also used when Q2 needs to re-vectorise text+title together. |
| `output/vocab.txt` | Task 1 | Decoder for the sparse BoW vectors — needed only if we want to map indices back to words. The classifier itself doesn't need it directly. |
| `output/count_vectors.txt` | Task 2 | **Bag-of-Words feature matrix** for Q1 (and Q2-text-only baseline). Reconstructed into a `scipy.sparse.csr_matrix` of shape `(61284, 8054)`. |
| `output/unweighted_vectors.txt` | Task 2 | **Unweighted FastText embedding matrix** for Q1. Loaded as a dense `np.float32` array of shape `(61284, 300)`. |
| `output/weighted_vectors.txt` | Task 2 | **TF-IDF-weighted FastText embedding matrix** for Q1. Same shape `(61284, 300)`. |

### Why this dependency matters

The spec is explicit:
> *"build machine learning models based on the feature representations of the
> documents you generated in **Task 2**"* (page 4)

This means Task 3 is **not allowed to invent its own feature pipeline**.
Concretely:
- The BoW classifier must be trained on the count vectors that use the
  **Task-1 vocabulary** — not a fresh `sklearn.CountVectorizer` fit.
- The embedding classifiers must use the **same FastText embeddings + the
  same TF-IDF model** that Task 2 produced.
- The reviews used must be the **cleaned, stop-word-removed, top-20-removed**
  text from `processed.csv` — not the raw CSV.

---

## 2. What's wrong with the current Task 3 (`cell-22`)

These are the issues that must be fixed in the rewrite:

| # | Problem | Fix |
|---|---|---|
| 1 | Loads raw `cosmetics_beauty_products_reviews.csv`, bypassing all Task 1 cleaning | Load `output/processed.csv` |
| 2 | Trains a **fresh Word2Vec(dim=50)** instead of using Task 2's FastText 300d outputs | Read the three `*_vectors.txt` files produced by Task 2 |
| 3 | Uses `sklearn.CountVectorizer(min_df=2)` — different vocabulary than `vocab.txt` | Reconstruct the sparse matrix from `count_vectors.txt` |
| 4 | **`df.sample(n=2000)`** — discards 97% of the data with no spec justification | Run on all 61,284 rows; subsample only specific slow steps if measurably needed |
| 5 | Q2 only uses `LogReg` and only on BoW; spec requires "at least three different [feature representation] models … bag-of-words, weighted, unweighted" *for both questions* | Run all 3 representations × all 3 Q2 configurations (3×3 grid) |
| 6 | Q2 "extra info" only includes `price` + `brand_name` — leaves `avg_product_rating`, `product_rating_count`, `product_title`, `review_rating` on the table | Add the rating-related numeric columns; decide on `review_rating` (see §6 — it likely leaks signal) |
| 7 | Single 100-line cell mixes data load + W2V training + Q1 + Q2 — hard to grade for the 2% presentation marks | Split into ~8 cells with markdown headers (3.1–3.4) |
| 8 | No interpretation of results — just prints two tables | Add a discussion paragraph after each results table answering the question the spec asked |
| 9 | `warnings.filterwarnings('ignore')` hides important sklearn convergence/scaling warnings | Remove; address warnings if any surface |
| 10 | `LogisticRegression(class_weight='balanced')` chosen without checking whether the label is actually imbalanced | Print class distribution first, then justify the choice |

---

## 3. Detailed plan — Q1: Language Model Comparison [3 marks]

**The question:** Which of the three Task-2 feature representations
(BoW / unweighted FastText / TF-IDF-weighted FastText) gives the best
classifier for predicting `is_a_buyer`?

### 3.0.5 Implementation-decisions log (HD criterion 4)

A single dedicated markdown cell at the top of Task 3 that justifies every
non-obvious choice up-front. This directly addresses RUBRIC Task-3 criterion 4
("Justification/explanation of implementation decisions") and the Notebook-
Presentation "strong justification and analysis" line. Required entries:

| Decision | Justification |
|---|---|
| Use Task-2 outputs (`count_vectors.txt`, `unweighted_vectors.txt`, `weighted_vectors.txt`) verbatim | Spec page 4 mandates "feature representations of the documents you generated in Task 2" |
| Drop empty `review_text` rows (1,911 rows from Task 1's filtering) | They have no signal for any text-based representation; keeping them inflates baseline accuracy by predicting majority class |
| `LogisticRegression` as primary classifier | Spec page 4 explicitly suggests it; linear baseline appropriate for high-dim sparse and dense features alike |
| Add `LinearSVC` and `RandomForest` as secondary classifiers | RUBRIC HD requires "strong experiments"; tests whether ranking is classifier-dependent |
| `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | Spec requires 5-fold; stratified preserves class ratio per fold; fixed seed for reproducibility |
| Report **macro-F1** as headline metric, accuracy as secondary | Macro-F1 is robust to class imbalance (verified in §3.1.5) |
| Report **mean ± std** across folds, not just mean | Std signals stability; small mean delta inside the std band ≠ meaningful difference |
| Add confusion matrix + per-class P/R/F1 for the best model | RUBRIC HD "strong evaluations" — accuracy alone hides class-specific failures |
| `StandardScaler` for dense embeddings only (not sparse BoW) | LogReg/SVM are scale-sensitive on dense input; scaling sparse BoW would densify and OOM |
| Exclude `review_rating` from Q2(c) extras | Same user provided rating *and* buy decision; near-perfect correlation ⇒ label leakage |
| `OneHotEncoder(min_frequency=20)` for `brand_name`, `product_title` | Cardinality control: rare brands/titles collapsed into "infrequent" bucket to avoid sparse explosion |
| Re-fit `TfidfModel` for Q2(b) | IDF weights are corpus-statistic-dependent; adding titles changes doc frequencies |

### 3.1 Setup
1. Load `output/processed.csv` → `df` (61,284 rows). Drop rows with empty
   `review_text` (Task 1 produced 1,911 of them — they have no signal for
   any text-based representation; documented in the notebook).
2. Extract `y = df['is_a_buyer'].astype(int).values`.
3. Print class distribution → decide whether to use `class_weight='balanced'`.
4. Build a `review_id → row index` map so we can align the three vector
   files (which are written in the same row order, but we verify).

### 3.2 Loaders (one helper per file)
| File | Loader | Output shape |
|---|---|---|
| `count_vectors.txt` | Parse each line into `(review_id, [(idx, freq), ...])`, build a `scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, V))` | `(61284, 8054)` sparse |
| `unweighted_vectors.txt` | `np.loadtxt` skipping the `#id,` header per line; faster: stream-parse with `np.fromstring` | `(61284, 300)` dense float32 |
| `weighted_vectors.txt` | Same loader as above, parameterised by path | `(61284, 300)` dense float32 |

After loading, **assert** that the row-order `review_id`s in all four sources
(processed.csv + 3 vector files) match — this guards against silent
mis-alignment of features and labels.

### 3.1.5 Pre-modeling investigation / EDA (HD criterion: "tasks fully investigated *before* modeling")

A dedicated EDA section that runs *before* any classifier is fit. This is the
single biggest gap between a CR/DI plan and an HD plan according to
[`docs/RUBRIC.md`](RUBRIC.md). Required artefacts:

1. **Class balance.** `df['is_a_buyer'].value_counts(normalize=True)` + a
   bar chart. Conclusion drives the metric choice (macro-F1) and whether to
   set `class_weight='balanced'`.
2. **Missingness audit** across the columns Q2 will consume — `review_title`,
   `brand_name`, `product_title`, `price`, `avg_product_rating`,
   `product_rating_count`. A 1-row table of % missing per column. This
   pre-justifies the imputers in §4.3(c).
3. **Review-length distribution per class.** Box plot or histogram of token
   counts (`review_text.str.split().str.len()`) split by `is_a_buyer`. Tells
   us whether buyers vs non-buyers write systematically longer/shorter
   reviews — a feature any model could exploit.
4. **Top tokens per class.** From the BoW matrix, top-15 tokens by mean
   count for each class side-by-side. Sanity check that the cleaned text
   contains class-discriminative signal (otherwise no model will work).
5. **Feature-matrix shapes summary.** A 1-row table:
   `(N rows, BoW dim, dense dim, label balance)` — confirms alignment and
   gives the reader the dimensions they'll see referenced later.
6. **Embedding sparsity check.** Count rows that are all-zero in the
   embedding matrices (all-OOV reviews). Documents an edge case the
   classifiers will see.

Each EDA artefact gets a **1–2 sentence interpretation** in the markdown
beneath it (not just plots-without-text). The EDA findings are then
explicitly referenced in the §3.0.5 decisions table where applicable.

### 3.3 Cross-validation protocol
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (spec requires 5-fold).
- Metrics: `accuracy` + `f1_macro` (macro-F1 is the right metric if the
  classes are imbalanced).
- Use `cross_validate` with `n_jobs=-1` to parallelise folds.
- For dense embedding features fed into linear models, wrap in a `Pipeline`
  with `StandardScaler` (LogReg/SVM are scale-sensitive).
- For the sparse BoW matrix, do **not** standard-scale (would densify it
  and blow up memory).

### 3.4 Classifier choice — 3 classifiers × 3 representations

Promoted from "optional" to **mandatory** to satisfy:
- The strictest reading of the spec note ("at least three different models")
  under either interpretation (3 reps OR 3 ML algorithms).
- RUBRIC HD requirement of "strong experiments and evaluations".

Three classifiers, all run on all three representations (9 cells per
question):

| Classifier | Hyperparameters | Why this one |
|---|---|---|
| `LogisticRegression` | `max_iter=1000, random_state=42` | Spec-suggested; strong linear baseline; works on sparse and dense |
| `LinearSVC` | `max_iter=2000, random_state=42` | Different linear loss (hinge); robustness check on whether ranking is loss-dependent |
| `RandomForestClassifier` | `n_estimators=200, n_jobs=-1, random_state=42` | Non-linear baseline; tests whether tree-based models exploit dense embeddings differently |

`LogReg` is designated the **primary** classifier whose results are carried
into Q2 (otherwise Q2 would be a 9-classifier × 3-config × 3-rep = 81-cell
explosion). The other two are reported in Q1 only, with a 1-sentence
discussion of whether they change the ranking.

### 3.5 Results table (Q1) — strong evaluation

**Headline table** (3 reps × 3 classifiers = 9 rows, mean ± std across folds):

| Representation | Classifier | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|---|---|---|---|
| BoW (count_vectors)         | LogReg        | …±… | …±… |
| BoW (count_vectors)         | LinearSVC     | …±… | …±… |
| BoW (count_vectors)         | RandomForest  | …±… | …±… |
| Unweighted FastText         | LogReg        | …±… | …±… |
| Unweighted FastText         | LinearSVC     | …±… | …±… |
| Unweighted FastText         | RandomForest  | …±… | …±… |
| TF-IDF-weighted FastText    | LogReg        | …±… | …±… |
| TF-IDF-weighted FastText    | LinearSVC     | …±… | …±… |
| TF-IDF-weighted FastText    | RandomForest  | …±… | …±… |

**Plus, for the best (representation, classifier) combination:**
- Confusion matrix (aggregated across folds via `cross_val_predict`)
- Per-class precision / recall / F1 (`classification_report`)
- ROC-AUC (binary task — single number, not per-fold)

**Plus a bar chart** with error bars showing macro-F1 per representation,
grouped by classifier.

### 3.6 Discussion (Q1)
A markdown cell answering: **which won, by how much, and why is that
plausible?** Things to mention:
- BoW often wins on short reviews because key sentiment words ("love",
  "hate", "broke") dominate; averaging dilutes them.
- Weighted vs unweighted embeddings — TF-IDF weighting should down-weight
  generic words, so weighted ≥ unweighted is the expected ordering.
- Note 1.91% OOV rate from Task 2 — embeddings already drop these tokens.

---

## 4. Detailed plan — Q2: Does more information help? [6 marks]

**The question:** Does adding the review *title* and product *metadata*
improve classification over text alone?

### 4.1 The three configurations
| Config | Inputs | Marks |
|---|---|---|
| **(a) Text only** | Cleaned `review_text` only | (baseline; reuses Q1) |
| **(b) Text + Title** | `review_title` concatenated to `review_text` before vectorising | 3 marks |
| **(c) Text + Title + Extras** | (b) + `brand_name`, `product_title`, `price`, `avg_product_rating`, `product_rating_count` | 3 marks |

### 4.2 Each configuration must be run for all 3 representations
The spec's note on page 4 says:
> *"For both questions above, consider demonstrating the comparison using
> at least three different models: one based on the bag-of-words model,
> and the other two focusing on weighted and unweighted models."*

So Q2 produces a **3×3 grid** (3 representations × 3 input configs), 9 cells.

### 4.3 How to construct features for each config

**(a) Text only** — already computed in Q1. Reuse the three matrices.

**(b) Text + Title** — title is *not* in Task 2's output (Task 2 explicitly
ignored title). So we re-vectorise here:
1. Apply the **same Task-1 cleaning** to `review_title` (the 9-step pipeline:
   regex tokenise → lowercase → drop len<2 → drop stopwords). We do **not**
   re-apply the rare-word and top-20 filters because those were corpus-wide
   decisions; instead we just keep tokens that survive the per-doc filters.
   Tokens not in `vocab.txt` will simply be skipped by the BoW step.
2. Concatenate cleaned title tokens to the cleaned `review_text` tokens.
3. Re-build all three representations on the combined token stream using
   the **exact same classes/models** as Task 2:
   - BoW: `Counter` against `vocab.txt`
   - Unweighted: sum of FastText vectors (skip OOV)
   - Weighted: re-fit a fresh `TfidfModel` on the combined corpus, then weighted sum

   *(We re-fit TF-IDF on the combined corpus because IDF weights depend on
   the doc-frequency distribution, which changes once titles are added.)*

**(c) Text + Title + Extras** — combine the (b) text features with structured
features via `sklearn.compose.ColumnTransformer`:

| Block | Feature | Transformer |
|---|---|---|
| Text | (b) BoW *or* (b) weighted *or* (b) unweighted | passthrough (already vectorised numpy/sparse) |
| Numeric | `price`, `avg_product_rating`, `product_rating_count` | `SimpleImputer(strategy='median')` → `StandardScaler` |
| Categorical | `brand_name`, `product_title` | `OneHotEncoder(handle_unknown='ignore', min_frequency=20)` to cap cardinality (`product_title` has many unique values; `min_frequency` collapses rare ones into an "infrequent" bucket) |

Stack with `scipy.sparse.hstack` (sparse-friendly) when the text block is
BoW; with `np.hstack` when it's a dense embedding (after densifying the
encoded structured block).

### 4.4 What about `review_rating`?
**Decision needed (see §6).** `review_rating` is a per-review star rating
given by the same user who decided whether to buy. It's almost certainly
*highly* correlated with `is_a_buyer` and could leak the label — including
it would inflate Q2(c) artificially and miss the point of the exercise.
**Default plan: exclude it.** Document the decision in the notebook.

### 4.5 Results table (Q2) — strong evaluation

Q2 uses **LogReg only** as the classifier (the primary chosen in §3.4) so
the comparison cleanly isolates the effect of *added information*, not
classifier × feature interactions.

**Headline table — macro-F1, mean ± std across 5 folds:**

| Config \ Representation        | BoW       | Unweighted | TF-IDF Weighted |
|---|---|---|---|
| (a) Text only                  | …±…       | …±…        | …±…             |
| (b) Text + Title               | …±…       | …±…        | …±…             |
| (c) Text + Title + Extras      | …±…       | …±…        | …±…             |

**Secondary table** with the same shape but reporting **Accuracy ± std**.

**Delta table** — gain from adding info, shows whether deltas exceed std bands:

| Δ                          | BoW | Unweighted | TF-IDF Weighted |
|---|---|---|---|
| (b) − (a)                  | … | … | … |
| (c) − (b)                  | … | … | … |
| (c) − (a)                  | … | … | … |

**Plus, for the best (config, representation) combination:**
- Confusion matrix (`cross_val_predict`-based)
- Per-class precision / recall / F1
- A short note on whether the best Q2 cell beats the best Q1 cell — this is
  the literal answer to the assignment question "does more information help?"

**Plus a heatmap** of the 3×3 macro-F1 grid for the presentation marks.

### 4.6 Discussion (Q2)
A markdown cell answering: **does more information help?**
- Compare row deltas: (b) vs (a), (c) vs (b), (c) vs (a).
- Identify *which kind* of extra info moved the needle (text-side vs
  structured-side) by looking at gains across columns.
- Comment on whether structured features helped more for sparse BoW than
  for dense embeddings (or vice versa) and offer a hypothesis.

---

## 5. Final notebook structure (Task 3 part of `task2_3.ipynb`)

Replace `cell-22` with this sequence:

```
## Task 3 — Cosmetics/Beauty Review Classification
### 3.0   Imports + CV protocol                          (code cell)
### 3.0.5 Implementation-decisions log                   (markdown — §3.0.5 table)
### 3.1   Load Task-1 + Task-2 artefacts                 (markdown + 3 code cells)
          - load processed.csv
          - load count_vectors.txt → sparse CSR
          - load unweighted/weighted vectors → dense arrays
          - alignment assertions
### 3.1.5 Pre-modeling investigation / EDA               (markdown + ~5 code cells)
          - class balance + bar chart
          - missingness audit
          - review-length distribution per class
          - top tokens per class (BoW)
          - feature-matrix shapes summary
          - embedding sparsity check
### 3.2   Q1 — Language model comparison                 (markdown + 1 code cell)
          - 3 reps × 3 classifiers, 5-fold CV
### 3.3   Q1 results — table (mean±std) + bar chart      (code cell)
### 3.3.5 Q1 best-model evaluation — confusion matrix +  (code cell)
          per-class P/R/F1 + ROC-AUC
### 3.4   Q1 discussion                                  (markdown — answers "which won and why")
### 3.5   Q2 — config (b) Text + Title vectorisation     (markdown + 1 code cell)
### 3.6   Q2 — config (c) Text + Title + Extras pipeline (markdown + 1 code cell)
### 3.7   Q2 — run all 9 (config × rep) cells            (code cell)
### 3.8   Q2 results — F1 table + Acc table + Δ table +  (code cell)
          heatmap
### 3.8.5 Q2 best-model evaluation — confusion matrix +  (code cell)
          per-class P/R/F1
### 3.9   Q2 discussion                                  (markdown — answers "does more info help")
### 3.10  Conclusion                                     (markdown — ties Q1+Q2 to spec questions)
```

This keeps the existing Section 2.x untouched; only the Task 3 region
changes.

### 5.1 Code commenting standard (Notebook Presentation rubric — Full Marks)

The rubric awards Full Marks (Notebook Presentation) only when there are
*"clear, thorough comments explaining code logic and purpose"*. To meet
this consistently, every code cell in Task 3 follows these rules:

1. **Markdown header above each code cell** explaining *what* the cell does
   and *why* it's needed in the pipeline (1–3 sentences).
2. **Inline comments on any non-obvious step** — never comment trivia like
   `# import pandas`, but always comment:
   - parsing logic (e.g. how `count_vectors.txt` is reconstructed into CSR)
   - choice of hyperparameter that isn't sklearn default
   - any assertion (state what invariant it's protecting)
   - any branch (`if feat == 'BoW': ...` — say *why* sparse path differs)
3. **One-line summary print** at the end of long-running cells
   (`print(f"Q1 done — best macro-F1 = {best:.3f} ({best_rep}/{best_clf})")`)
   so the notebook stays readable without re-running.
4. **No `warnings.filterwarnings('ignore')`** at module level. If a specific
   warning is genuinely noisy and known-benign, suppress it locally with
   `warnings.catch_warnings()` and a 1-line comment explaining why.
5. **Random seed fixed** in every model and CV split (`random_state=42`)
   for reproducibility.
6. **Paths and constants at the top** of the section, not scattered through
   cells (e.g. `OUTPUT_DIR`, `CV`, `RANDOM_STATE` defined once in §3.0).

---

## 6. Open decisions (please confirm before I implement)

Resolved (folded into the plan above):
- ~~Single vs multiple classifiers for Q1~~ → §3.4 mandates 3 classifiers × 3 reps.
- ~~Single classifier for Q2~~ → §4.5 fixes Q2 to LogReg-only (carried from Q1's primary).

Still open:

1. **Run Q1 on the full 61,284 rows?** — recommended yes; LogReg + BoW takes
   seconds, and the dense 300d × 61k × 5-fold CV is also fast (<1 min on a
   modern Mac). The 2,000-row subsample in the current code has no
   justification in the spec. RandomForest will be the slowest leg —
   acceptable trade-off, but if it turns out to be >5 min on your machine
   we can either (a) reduce `n_estimators` to 100, or (b) subsample only
   the RF cells with documented justification.

2. **For Q2(c), include `review_rating` as a feature?** — recommended **no**
   (label leakage risk; see §4.4). If you want it included for completeness,
   we report results both with and without it and discuss the gap.

3. **`brand_name` and `product_title` cardinality** — both have many unique
   values. Plan uses `OneHotEncoder(min_frequency=20)`. If you'd prefer
   target encoding or hashing, say so.

4. **Include `product_tags` (text column) in Q2(c) extras?** — not in the
   current plan. Could add as a separate TF-IDF text block in the
   `ColumnTransformer`. Adds one more "extra info" feature but increases
   model complexity. Default plan: skip; mention in discussion.

Once these are confirmed, the rewrite of the Task 3 cells is mechanical
and should take one editing pass.

---

## 7. Plan ↔ RUBRIC.md traceability

A final cross-check that every rubric criterion has a plan section
addressing it:

| RUBRIC item | Plan section |
|---|---|
| **Task 3 (1)** Comparison of description document vectors (training, evaluation, analysis) | §3.2 + §3.3 + §3.3.5 + §3.4 |
| **Task 3 (2)** Comparison of description + title vectors | §4.1(b) + §4.3(b) + §4.5 |
| **Task 3 (3)** Comparison of title + description + extra info vectors | §4.1(c) + §4.3(c) + §4.5 |
| **Task 3 (4)** Justification/explanation of implementation decisions | §3.0.5 (decisions log) + §3.4 (discussion) + §4.6 (discussion) |
| **Task 3 HD** "tasks fully investigated *before* modeling" | §3.1.5 (EDA section) |
| **Task 3 HD** "strong experiments and evaluations" | §3.4 (3×3 design) + §3.5 (mean±std + confusion + per-class + ROC-AUC) + §4.5 (delta table + heatmap) |
| **Notebook Presentation** "clear, thorough comments" | §5.1 commenting standard |
| **Notebook Presentation** "strong justification and analysis" | §3.0.5 + §3.4 + §4.6 + §3.10 conclusion |
| **Notebook Presentation** "excellent markdown usage / well-organized" | §5 cell structure (~16 cells with explicit headers) |
