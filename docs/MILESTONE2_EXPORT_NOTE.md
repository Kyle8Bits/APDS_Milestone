# Milestone II — Model Export Note

> **This is a reference note, not a Milestone I deliverable.**
> Use it when you start Milestone II to re-generate the model artefacts the
> web app needs.

The Q2(c) winning model from Task 3 (BoW + LogisticRegression on text + title
+ structured metadata, macro-F1 = 0.7126) is the recommended model to serve
in the web app. To use it from outside the Jupyter kernel, persist the
trained pipeline to disk with `joblib`.

---

## When to run this

After you have:
1. Run **all** cells of [`../data/task2_3.ipynb`](../data/task2_3.ipynb) so
   the variables `X_bow_c`, `y`, `preproc_struct`, `OUTPUT_DIR`, and
   `RANDOM_STATE` exist in the kernel.
2. Decided to start Milestone II development and want the model files to
   load from a Python API server.

---

## Where to put the cell

Add a new code cell at the bottom of [`../data/task2_3.ipynb`](../data/task2_3.ipynb)
(after §3.10 Conclusion). Make sure the imports at the top of the notebook
include:

```python
import joblib
import shutil
```

(They already are in the consolidated imports cell — see cell 2.)

---

## The export code

```python
WEBAPP_MODEL_DIR = os.path.join("..", "web", "model")
os.makedirs(WEBAPP_MODEL_DIR, exist_ok=True)

# Re-fit the winning classifier on the full dataset (no CV split — CV already
# validated it; for serving we want the strongest possible model).
final_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
final_clf.fit(X_bow_c, y)

joblib.dump(final_clf,     os.path.join(WEBAPP_MODEL_DIR, "clf.joblib"))
joblib.dump(preproc_struct, os.path.join(WEBAPP_MODEL_DIR, "preproc_struct.joblib"))
shutil.copy(os.path.join(OUTPUT_DIR, "vocab.txt"),
            os.path.join(WEBAPP_MODEL_DIR, "vocab.txt"))
shutil.copy("stopwords_en.txt",
            os.path.join(WEBAPP_MODEL_DIR, "stopwords_en.txt"))

print(f"Exported model artefacts to {WEBAPP_MODEL_DIR}/")
for fname in sorted(os.listdir(WEBAPP_MODEL_DIR)):
    size_kb = os.path.getsize(os.path.join(WEBAPP_MODEL_DIR, fname)) / 1024
    print(f"  {fname:30s}  {size_kb:8.1f} KB")
```

---

## What this produces

```
web/model/
├── clf.joblib              ← fitted LogisticRegression
├── preproc_struct.joblib   ← fitted ColumnTransformer for product metadata
├── vocab.txt               ← Task-1 vocabulary (copied from output/)
└── stopwords_en.txt        ← Task-1 stopword list (copied from data/)
```

For the role of each file and the runtime inference flow, see
[`../web/MODEL_FILES.md`](../web/MODEL_FILES.md).

---

## Why this is *not* in Milestone I

The Milestone I spec ([`AP4DS_2026A_A3 - Milestone 1.pdf`](AP4DS_2026A_A3%20-%20Milestone%201.pdf),
page 5) lists the required submission files:

* `vocab.txt`, `count_vectors.txt`, `unweighted_vectors.txt`, `weighted_vectors.txt`
* `task1.ipynb` + `task1.py`
* `task2_3.ipynb` + `task2_3.py`

No model export files are required for Milestone I. Including the export cell
in `task2_3.ipynb` adds extra runtime, slows down the marker's
restart-and-run-all check, and writes files outside the submission folder
(into `web/`). Cleaner to keep the export step here as a note and run it
once when you actually need the artefacts.
