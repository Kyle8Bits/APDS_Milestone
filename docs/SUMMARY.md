# Assignment 3 — Milestone I: Natural Language Processing

**Course:** COSC3081/3082/3015 Advanced Programming for Data Science
**Type:** Group assignment (submit via Canvas)
**Marks:** 20
**Due:** End of Week 10 (check Canvas for exact date/time)

---

## What is the task?

Build a **text analytics pipeline** on a dataset of ~161,200 cosmetics & beauty product reviews. The pipeline must:

1. Pre-process raw review text.
2. Generate multiple feature representations (bag-of-words + word embeddings).
3. Train and evaluate classification models that predict **`is_a_buyer`** (whether the reviewer actually bought the product — `True` / `False`).

**Dataset features used:** `review_title`, `review_text`, `is_a_buyer`
**Data file:** [data/cosmetics_beauty_products_reviews.csv](data/cosmetics_beauty_products_reviews.csv)
**Stopwords:** [data/stopwords_en.txt](data/stopwords_en.txt)

> Milestone II (later) builds a web app on top of the model produced here.

---

## What I have to do

### Task 1 — Basic Text Pre-processing _(4 marks)_ — [data/task1.ipynb](data/task1.ipynb)

Pre-process **`review_text` only** with these exact steps, in order:

1. Extract the review text.
2. Tokenize using the regex: `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`
3. Lowercase all words.
4. Remove words with length < 2.
5. Remove stopwords using `stopwords_en.txt`.
6. Remove words that appear **only once** in the whole collection (term frequency).
7. Remove the **top 20 most frequent** words by **document** frequency.
8. Save processed data as `processed.csv`.
9. Build a unigram vocabulary from the cleaned reviews and save as `vocab.txt`.

**`vocab.txt` format:** `word_string:word_integer_index`, sorted alphabetically, index starts at 0. Removed words must NOT appear in the vocabulary.

---

### Task 2 — Feature Representations _(7 marks)_ — [data/task2_3.ipynb](data/task2_3.ipynb)

Use **only `review_text`** (ignore the title here). Build **3 representations**:

1. **Count vectors** (bag-of-words, based on `vocab.txt` from Task 1) → `count_vectors.txt`
2. **Unweighted embedding vectors** — pick **one** pretrained embedding model (FastText / GoogleNews300 / GloVe / etc.) → `unweighted_vectors.txt`
3. **TF-IDF weighted embedding vectors** using the same model → `weighted_vectors.txt`

**Output line format** (each line = one review):

- `count_vectors.txt`: `#<review_index>,word_idx:freq,word_idx:freq,...`
- `unweighted_vectors.txt` / `weighted_vectors.txt`: `#<review_index>,v1,v2,v3,...` (length = embedding dim)

---

### Task 3 — Review Classification _(9 marks)_ — [data/task2_3.ipynb](data/task2_3.ipynb)

Build classifiers for `is_a_buyer`. Logistic regression is fine; other models are allowed. Run **two experiments** with **5-fold cross-validation**.

**Q1 — Language model comparison (3 marks):**
Compare classifiers built on each of the 3 feature representations from Task 2 (count, unweighted, weighted). Which performs best?

**Q2 — Does more information help? (6 marks):**
Compare model performance when using:

- Only `review_text` (already done in Q1)
- **(3 marks)** `review_text` + `review_title`
- **(3 marks)** `review_text` + title + extra product info (brand, product title, avg rating, price, …) — feature representation is your choice.

**For both Q1 and Q2:** demonstrate with at least 3 models (one bag-of-words based, plus weighted and unweighted embedding based). Use 5-fold CV.

---

## What to submit

Put everything in a folder named with your **student ID**, then zip it as `s1234567.zip`.

The folder must contain:

| File                     | Source                          |
| ------------------------ | ------------------------------- |
| `vocab.txt`              | Task 1                          |
| `processed.csv`          | Task 1                          |
| `count_vectors.txt`      | Task 2                          |
| `unweighted_vectors.txt` | Task 2                          |
| `weighted_vectors.txt`   | Task 2                          |
| `task1.ipynb`            | Task 1 notebook                 |
| `task1.py`               | `.py` export of `task1.ipynb`   |
| `task2_3.ipynb`          | Tasks 2 & 3 notebook            |
| `task2_3.py`             | `.py` export of `task2_3.ipynb` |

> The `.py` files are used for plagiarism detection. **Submission without matching `.py` files will NOT be marked.** Export from Jupyter via _File → Download as → Python (.py)_.

---

## Marking breakdown

| Task   | Implementation | Notebook presentation | Total  |
| ------ | -------------- | --------------------- | ------ |
| Task 1 | 3%             | 1%                    | **4%** |
| Task 2 | 5%             | 2%                    | **7%** |
| Task 3 | 7%             | 2%                    | **9%** |

- **Mechanical pass:** outputs are diffed against expected outputs — file names, formatting, sort order, and indexing all matter.
- **Expert pass:** notebook is reviewed for logic, library use, comments, structure, and presentation. Markers will NOT fix typos or import errors for you.

**Late penalty:** −10% per day; ≥5 days late = 0.
