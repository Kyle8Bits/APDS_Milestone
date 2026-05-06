# Model Files — `web/model/`

This folder contains the four artefacts the web app needs at runtime to
predict the *likelihood of purchase* for a given review + product. The files
are produced by the export cell at the bottom of
[`../data/task2_3.ipynb`](../data/task2_3.ipynb) (§3.11) after running the
full Task 3 pipeline.

```
web/model/
├── clf.joblib              ← trained LogisticRegression (the "brain")
├── preproc_struct.joblib   ← fitted ColumnTransformer for product metadata
├── vocab.txt               ← Task-1 vocabulary (8,054 words, 0-indexed)
└── stopwords_en.txt        ← Task-1 stopword list (570 words)
```

---

## 1. What each file is

| File | What it is | Why it's needed |
|---|---|---|
| `clf.joblib` | The trained `LogisticRegression` model — converts a 1×8215 feature vector into a buy probability | The actual prediction step |
| `preproc_struct.joblib` | The fitted `ColumnTransformer` from Task 3 §3.6 — turns raw `(price, brand, product_title, ...)` fields into a 1×161 numeric vector | Translates user-facing fields into model-ready numbers |
| `vocab.txt` | The Task-1 vocabulary — 8,054 cleaned words, each mapped to a column index (`works:7891`, `claims:1140`, etc.) | Tells the BoW step *which column* each token belongs in |
| `stopwords_en.txt` | The 570-word stopword list from Task 1 | Used to clean *new* review text the user submits, the same way Task 1 cleaned the training data |

## 2. Why `vocab.txt` and `stopwords_en.txt` are included

Because **the web app must clean new text exactly the same way Task 1 cleaned
the training data** — otherwise the input vector means something different
than what the model was trained on, and predictions are garbage.

When a user types a review on the site, the app has to apply the same
transformation pipeline that produced the training BoW vectors. That pipeline
depends on **two reference files** from Task 1:

1. **`stopwords_en.txt`** — to drop words like "the", "and", "is" during
   cleaning. Without it, the user's review keeps stopwords that the training
   data doesn't have, and the resulting vector is shifted relative to the
   model's expectations.

2. **`vocab.txt`** — to know which column index each surviving token maps to.
   The classifier expects a 1×8054 vector where, e.g., column 1140 always
   means the word `claims`. Without `vocab.txt` the app can't put the right
   counts in the right columns.

These aren't model files — they're **lookup tables** the app needs to
reproduce the Task-1 + Task-2 transformations at runtime.

---

## 3. How a single prediction flows

```
                  USER INPUT (review + product info)
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Step 1 — Clean text using stopwords_en.txt    │
        │   regex tokenise → lowercase → drop len<2     │
        │   → drop stopwords                            │
        └────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Step 2 — Count tokens against vocab.txt        │
        │   sparse 1×8054 BoW row                        │
        └────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Step 3 — preproc_struct.transform(metadata)   │
        │   sparse 1×161 structured row                  │
        │   (price, avg_rating, brand one-hot, ...)     │
        └────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Step 4 — sparse.hstack([BoW, struct])          │
        │   sparse 1×8215 final feature row              │
        └────────────────────────────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ Step 5 — clf.predict_proba(row)[0, 1]         │
        │   → 0.83  (likelihood of purchase)             │
        └────────────────────────────────────────────────┘
```

Removing **any one** of the four files breaks a step in this chain.

---

## 4. How to use them — Python prediction function

```python
import re
import joblib
import pandas as pd
from collections import Counter
from scipy import sparse
from scipy.sparse import csr_matrix

# ── Load once at server startup (NOT per request) ──────────────
MODEL_DIR = "web/model"

CLF = joblib.load(f"{MODEL_DIR}/clf.joblib")
PREPROC = joblib.load(f"{MODEL_DIR}/preproc_struct.joblib")

VOCAB = {}
with open(f"{MODEL_DIR}/vocab.txt") as f:
    for line in f:
        word, idx = line.rstrip().rsplit(":", 1)
        VOCAB[word] = int(idx)

with open(f"{MODEL_DIR}/stopwords_en.txt") as f:
    STOPWORDS = {w.strip() for w in f if w.strip()}

TOKENISER = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")


def clean(text):
    """Same Task-1 cleaning pipeline applied to new user input."""
    if not text:
        return []
    return [t for t in TOKENISER.findall(text.lower())
            if len(t) >= 2 and t not in STOPWORDS]


def predict_buy_probability(review_text, review_title, brand_name,
                             product_title, price, avg_product_rating,
                             product_rating_count):
    # Step 1+2 — text → BoW row
    tokens = clean(review_text) + clean(review_title)
    counts = Counter(t for t in tokens if t in VOCAB)
    rows, cols, data = [], [], []
    for w, c in counts.items():
        rows.append(0); cols.append(VOCAB[w]); data.append(c)
    bow = csr_matrix((data, (rows, cols)), shape=(1, len(VOCAB)))

    # Step 3 — metadata → structured row
    struct = PREPROC.transform(pd.DataFrame([{
        "price": price,
        "avg_product_rating": avg_product_rating,
        "product_rating_count": product_rating_count,
        "brand_name": brand_name,
        "product_title": product_title,
    }]))

    # Step 4 — combine
    X = sparse.hstack([bow, struct])

    # Step 5 — predict
    return float(CLF.predict_proba(X)[0, 1])
```

### Calling it

```python
p = predict_buy_probability(
    review_text="Worked great, smells lovely",
    review_title="Pretty good moisturiser",
    brand_name="Olay",
    product_title="Olay Whip Day Cream",
    price=1599,
    avg_product_rating=4.1,
    product_rating_count=43,
)
# p ≈ 0.83  → display as "83% likelihood of purchase"
```

---

## 5. React + Python split

The frontend in this `web/` folder is React (Vite). React can't run sklearn
directly (it's Python-only), so the typical setup is:

```
React (frontend)  ──  HTTP  ──▶  Python API (Flask/FastAPI)  ──▶  sklearn models
   web/                                                              web/model/
```

The Python API exposes one endpoint (e.g. `POST /api/predict`), wraps
`predict_buy_probability` from §4, and returns JSON. React fetches from it
on each product page and renders the probability as a badge.

A minimal Flask wrapper would look like:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    p = predict_buy_probability(**request.json)
    return jsonify(probability=p, label="likely buyer" if p >= 0.5 else "unlikely")

if __name__ == "__main__":
    app.run(port=5000)
```

CORS may need to be enabled (`pip install flask-cors`) so React's dev server
on port 5173 can call the API on port 5000.

---

## 6. Re-generating these files

If the model changes (re-trained on new data, hyperparameters tuned, etc.),
re-export by uncommenting and running the export cell at the end of Task 3
in [`../data/task2_3.ipynb`](../data/task2_3.ipynb) (§3.11). All four files
in this folder will be overwritten with the new versions.

The export does **not** require a network connection or FastText download —
the winning model (Q2(c) BoW + LogReg) does not use embeddings, so it ships
with no large model dependencies.
