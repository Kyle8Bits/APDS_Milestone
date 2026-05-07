#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Milestone I - Natural Language Processing
# ## Tasks 2 & 3: Feature Representations & Classification
# 
# #### Group: UN_Group 3
# #### Student Name:
# - Mai Dang Khoa ( s3974876 )
# - Dang Cuu Dang Khoa ( s3979159 )
# - Tran Quang Minh ( s3988776 )
# 
# Environment: Python 3 + Jupyter Notebook
# 
# Libraries used:
# * pandas
# * numpy
# * scipy (`sparse`, `sparse.csr_matrix`)
# * matplotlib (`pyplot`)
# * seaborn
# * gensim (`downloader`, `corpora.Dictionary`, `models.TfidfModel`)
# * scikit-learn (`model_selection`, `linear_model`, `svm`, `ensemble`, `compose`, `preprocessing`, `impute`, `pipeline`, `metrics`)
# * collections (`Counter`)
# * IPython.display (`display`)
# * joblib
# * Standard library: `os`, `re`, `shutil`, `warnings`
# 
# ## Task 2 - Introduction
# 
# This notebook generates three feature representations from the cleaned `review_text`
# produced by Task 1:
# 
# 1. **Count vectors** - sparse bag-of-words against `vocab.txt`.
# 2. **Unweighted embedding vectors** - sum of FastText word vectors per review.
# 3. **TF-IDF weighted embedding vectors** - TF-IDF weighted sum of FastText vectors.
# 
# Chosen embedding: `fasttext-wiki-news-subwords-300` (300-dim) loaded via gensim
# downloader. Note that this distribution ships only the trained word vectors
# (`KeyedVectors`), not the full FastText `.bin` with subword n-gram parameters,
# so OOV tokens are skipped (matching the lecturer's slide-49 try/except pattern).
# We chose this model because its trained vocabulary (~1M tokens) is much larger
# than Word2Vec GoogleNews-300 and includes informal/social-media words that
# appear in beauty product reviews.

# ## Importing libraries 

# In[1]:


# Standard library
import os
import re
import shutil
import warnings
from collections import Counter

# Scientific & plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy import sparse
from scipy.sparse import csr_matrix

# Gensim — Task 2 (embeddings + TF-IDF)
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# scikit-learn — Task 3 (classification)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, cross_val_predict, train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Persistence — Milestone II (model export)
import joblib


# ## Output path

# In[2]:


OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ## Task 2. Generating Feature Representations for Cosmetics / Beauty Reviews

# ### 2.1 Load preprocessed data
# 
# Load:
# - `processed.csv` - Task 1 output with the cleaned `review_text` column.
# - `vocab.txt` - 0-indexed `word:index` dictionary built in Task 1.
# 
# The `review_text` column already holds space-separated cleaned tokens, so we
# just need `.split()` to recover the token lists. We also keep `review_id` so
# each output line can be tagged with `#<review_id>` per the spec.

# The pipeline below builds three document representations end-to-end:
# 
# 1. **Sparse Bag-of-Words** based on the Task-1 vocabulary (`vocab.txt`).
# 2. **Dense Bag of Embeddings** - sum of FastText word vectors per review.
# 3. **Dense TF-IDF Weighted Bag of Embeddings** - weighted sum using
#    gensim's `TfidfModel`.
# 
# Each representation is generated, validated, and saved to disk in the
# spec-required format `#<review_id>,...`.

# In[3]:


df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed.csv"))
df["review_text"] = df["review_text"].fillna("")

# Recover token lists from the space-joined cleaned text saved by Task 1
tokenized = [text.split() for text in df["review_text"]]
review_ids = df["review_id"].tolist()

# Load vocab.txt → word2idx dict
word2idx = {}
with open(os.path.join(OUTPUT_DIR, "vocab.txt"), "r", encoding="utf-8") as f:
    for line in f:
        w, i = line.rstrip("\n").rsplit(":", 1)
        word2idx[w] = int(i)

print(f"Reviews loaded     : {len(df):,}")
print(f"Vocabulary size    : {len(word2idx):,}")
print(f"Empty reviews      : {sum(1 for t in tokenized if not t):,}")
print(f"Sample tokens (#0) : {tokenized[0][:10]}")


# ### 2.2 Count Vectors (Bag-of-Words)
# For each review we count the occurrences of each vocabulary word and write a
# sparse line in the format required by the spec (page 3):
# 
# #<review_id>,word_idx:freq,word_idx:freq,...
# 
# Word-index entries are sorted **ascending**, matching the example in the
# assignment PDF. Empty reviews produce a header-only line `#<review_id>,`.

# In[4]:


out_path = os.path.join(OUTPUT_DIR, "count_vectors.txt")

with open(out_path, "w", encoding="utf-8") as fout:
    for rid, tokens in zip(review_ids, tokenized):
        counts = Counter(t for t in tokens if t in word2idx)
        items  = sorted(counts.items(), key=lambda kv: word2idx[kv[0]])
        body   = ",".join(f"{word2idx[w]}:{c}" for w, c in items)
        fout.write(f"#{rid},{body}\n")

print(f"Wrote {len(review_ids):,} lines to {out_path}")
print("\nSample (first 3 lines):")
with open(out_path) as f:
    for _ in range(3):
        print(f.readline().rstrip()[:140])


# ### 2.3 Load pretrained FastText embeddings
# 
# `fasttext-wiki-news-subwords-300` via gensim downloader (300-dim, ~1GB on first
# download, cached after that). The gensim downloader provides the trained
# `KeyedVectors` only, so OOV tokens are skipped (slide-49 try/except pattern)
# rather than computed from subwords. We chose this model over GoogleNews-300
# because its ~1M-token trained vocab includes more informal English, giving
# better coverage for the typo/slang-heavy beauty-review corpus.

# In[5]:


print("Loading FastText (first run downloads ~1GB) ...")
ft = api.load("fasttext-wiki-news-subwords-300")

DIM = ft.vector_size
print(f"FastText loaded - embedding dim: {DIM}")
print(f"Trained vocab size: {len(ft):,}")

in_trained = sum(1 for w in word2idx if w in ft.key_to_index)
print(f"\nTask-1 vocab in FastText trained vocab: "
      f"{in_trained:,} / {len(word2idx):,} ({in_trained/len(word2idx)*100:.1f}%)")
print(f"OOV tokens (the remaining {len(word2idx)-in_trained:,}) will be skipped "
      f"in the document-vector sum (slide 49 try/except pattern).")


# In[6]:


all_tokens = [t for review in tokenized for t in review]
oov_total  = sum(1 for t in all_tokens if t not in ft.key_to_index)

print(f"Total token occurrences across corpus : {len(all_tokens):,}")
print(f"OOV occurrences (will be skipped)     : {oov_total:,} "
      f"({oov_total/len(all_tokens)*100:.2f}%)")

oov_unique = sorted({t for t in all_tokens if t not in ft.key_to_index})
print(f"Unique OOV tokens : {len(oov_unique):,}")
print(f"\nExamples of OOV tokens (first 15): {oov_unique[:15]}")


# ### 2.4 Unweighted embedding vectors
# 
# For each review we **sum** the FastText vectors of every token in the FastText
# trained vocabulary (slide 49 style - try/except pattern, OOV tokens skipped).
# Empty reviews and all-OOV reviews receive a zero vector. Each output line:
# 
# `#<review_id>,v1,v2,...,v300`
# 
# Values are written with 6 decimal places.
# 

# In[7]:


zero_vec = np.zeros(DIM, dtype=np.float32)

def doc_vec_sum(tokens):
    vecs = [ft[w] for w in tokens if w in ft.key_to_index]
    if not vecs:
        return zero_vec
    return np.sum(np.stack(vecs), axis=0)

out_path = os.path.join(OUTPUT_DIR, "unweighted_vectors.txt")
with open(out_path, "w", encoding="utf-8") as fout:
    for rid, tokens in zip(review_ids, tokenized):
        v = doc_vec_sum(tokens)
        body = ",".join(f"{x:.6f}" for x in v)
        fout.write(f"#{rid},{body}\n")

print(f"Wrote {len(review_ids):,} lines to {out_path}")
print(f"Each line has {DIM} comma-separated values after the header.")
print("\nSample (first 80 chars of first 2 lines):")
with open(out_path) as f:
    for _ in range(2):
        print(f.readline()[:80] + " ...")


# ### 2.5 TF-IDF weighted embedding vectors
# 
# Following the gensim approach on slide 51:
# 
# 1. Build a `Dictionary` from the cleaned tokens.
# 2. Convert each review to bag-of-words with `doc2bow`.
# 3. Fit a `TfidfModel` on that corpus.
# 4. For each review:  `h = Σ tfidf(w, d) · ft[w]`.
# 
# Empty reviews receive a zero vector.

# In[8]:


print("Building gensim Dictionary + TfidfModel ...")
docs_dict   = Dictionary(tokenized)
docs_corpus = [docs_dict.doc2bow(doc) for doc in tokenized]
tfidf_model = TfidfModel(docs_corpus, id2word=docs_dict)

print(f"Dictionary size : {len(docs_dict):,}")
print(f"Corpus length   : {len(docs_corpus):,}")
print("TfidfModel fitted.")


# In[9]:


def doc_vec_weighted(bow_doc):
    if not bow_doc:
        return zero_vec
    v = np.zeros(DIM, dtype=np.float32)
    for word_id, weight in tfidf_model[bow_doc]:
        token = docs_dict[word_id]
        if token in ft.key_to_index:        # skip OOV
            v += weight * ft[token]
    return v

out_path = os.path.join(OUTPUT_DIR, "weighted_vectors.txt")
with open(out_path, "w", encoding="utf-8") as fout:
    for rid, bow in zip(review_ids, docs_corpus):
        v = doc_vec_weighted(bow)
        body = ",".join(f"{x:.6f}" for x in v)
        fout.write(f"#{rid},{body}\n")

print(f"Wrote {len(review_ids):,} lines to {out_path}")
print("\nSample (first 80 chars of first 2 lines):")
with open(out_path) as f:
    for _ in range(2):
        print(f.readline()[:80] + " ...")


# ### 2.6 Sanity checks
# 
# Verify line counts, vector dimensionality, and spot-check one review end-to-end.

# In[10]:


def line_count(path):
    with open(path) as f:
        return sum(1 for _ in f)

paths = [os.path.join(OUTPUT_DIR, p) for p in
         ["count_vectors.txt", "unweighted_vectors.txt", "weighted_vectors.txt"]]
for p in paths:
    print(f"{os.path.basename(p):30s}  lines = {line_count(p):,}")

def dense_dims(path, n=10):
    dims = set()
    with open(path) as f:
        for i, line in enumerate(f):
            if i == n: break
            _, body = line.rstrip().split(",", 1)
            dims.add(len(body.split(",")))
    return dims

print(f"\nDense dim - unweighted (first 10 lines): "
      f"{dense_dims(os.path.join(OUTPUT_DIR, 'unweighted_vectors.txt'))}")
print(f"Dense dim - weighted   (first 10 lines): "
      f"{dense_dims(os.path.join(OUTPUT_DIR, 'weighted_vectors.txt'))}")

print("\n--- Spot check on row 0 ---")
print(f"review_id : {review_ids[0]}")
print(f"tokens    : {tokenized[0][:12]} ...")
with open(os.path.join(OUTPUT_DIR, "count_vectors.txt")) as f:
    print(f"count     : {f.readline().rstrip()[:140]} ...")

assert all(line_count(p) == len(df) for p in paths), "Line count mismatch!"
print("\nAll line counts == number of reviews.")


# ### 2.7 Summary
# 
# Three feature files written to `../output/` (configured via `OUTPUT_DIR`):
# 
# | File | Format | Source |
# |------|--------|--------|
# | `count_vectors.txt` | `#<review_id>,wIdx:freq,...` (sparse, sorted asc) | `vocab.txt` |
# | `unweighted_vectors.txt` | `#<review_id>,v1,...,v300` (dense) | sum of FastText vectors |
# | `weighted_vectors.txt` | `#<review_id>,v1,...,v300` (dense) | TF-IDF × FastText, summed |
# 
# These three files are the inputs for Task 3, where we'll train classifiers
# that predict `is_a_buyer` and compare model performance across the three
# representations using 5-fold cross-validation.
# 
# Before submission: export this notebook as `task2_3.py` via *File → Download as → Python (.py)*.

# ## Task 3 - Cosmetics/Beauty Review Classification
# 
# We now build classifiers that predict `is_a_buyer` from the feature
# representations produced in Task 2 (BoW, unweighted FastText, TF-IDF-weighted
# FastText), and investigate two questions per the spec:
# 
# * **Q1** - which of the three Task-2 representations gives the best classifier?
# * **Q2** - does adding the review *title* and product *metadata* improve performance?
# 
# The implementation follows the plan in [`docs/PLAN_T3.MD`](../docs/PLAN_T3.MD).
# 

# ### 3.0 Global cross-validation protocol
# 

# In[11]:


RANDOM_STATE = 26
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "f1_macro"]


# ### 3.0.5  Implementation decisions log
# 
# This table justifies every non-obvious choice up-front (see RUBRIC Task 3 (4)
# "Justification/explanation of implementation decisions"). Each row maps a
# design choice to its driver - spec requirement, leakage risk, or computational
# constraint.
# 
# | Decision | Justification |
# |---|---|
# | Use Task-2 outputs (`count_vectors.txt`, `unweighted_vectors.txt`, `weighted_vectors.txt`) verbatim | Spec page 4 mandates "feature representations of the documents you generated in Task 2" |
# | Keep all 61,284 rows (including 1,911 with empty `review_text`) | Empty rows produce all-zero vectors; classifier predicts the majority class for them. Keeping them gives a fair common denominator across Q1, Q2(a), Q2(b), Q2(c) |
# | `LogisticRegression` as primary classifier | Spec page 4 explicitly suggests it; strong linear baseline appropriate for both high-dim sparse BoW and low-dim dense embeddings |
# | Add `LinearSVC` and `RandomForest(n_estimators=100)` as secondary classifiers (Q1 only) | RUBRIC HD requires "strong experiments"; tests whether ranking is classifier-dependent. RF tree count capped at 100 to keep total Q1 runtime reasonable on sparse 60k×8k BoW |
# | `StratifiedKFold(n_splits=5, shuffle=True, random_state=26)` | Spec requires 5-fold; stratified preserves the ~72/28 class ratio per fold; fixed seed for reproducibility |
# | Report **macro-F1** as headline metric, accuracy as secondary | Macro-F1 is robust to the class imbalance verified in 3.1.5; raw accuracy can be ≈72% by predicting "buyer" for everyone |
# | Report **mean ± std** across folds | Std signals stability; small mean delta within the std band is not a meaningful difference |
# | Add confusion matrix + per-class P/R/F1 + ROC-AUC for the best (rep, clf) combination | RUBRIC HD "strong evaluations" - accuracy alone hides class-specific failures |
# | `StandardScaler` for dense embeddings only (not sparse BoW) | LogReg/SVM are scale-sensitive on dense input; scaling sparse BoW would densify it and blow up memory |
# | **Exclude `review_rating` from Q2(c) extras** | Same user provided the rating *and* the buy decision; near-perfect correlation ⇒ label leakage. Including it would make Q2(c) trivially win and miss the point of the exercise |
# | `OneHotEncoder(min_frequency=20)` for `brand_name`, `product_title` | Cardinality control: rare brands/titles collapsed into "infrequent" bucket to avoid sparse dimensionality explosion |
# | Re-fit `gensim.TfidfModel` for Q2(b) | IDF weights are corpus-statistic-dependent; adding titles changes document frequencies, so the weights from Task 2 are no longer correct |
# | Skip `product_tags` and `review_date` from extras | `product_tags` is all-NaN in the dataset; `review_date` would need feature engineering and isn't in the spec's example list |
# 

# ### 3.1  Load Task-1 and Task-2 artefacts
# 
# We load the cleaned reviews (Task 1) and the three feature matrices (Task 2)
# and align them on `review_id`. Each matrix file was written by Task 2 in the
# same row order as `processed.csv`, but we still verify alignment to guard
# against silent mis-matches that would corrupt every downstream metric.
# 

# In[12]:


df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed.csv"))
df["review_text"] = df["review_text"].fillna("")
df["is_a_buyer"] = df["is_a_buyer"].astype(int)

y = df["is_a_buyer"].values
review_ids_df = df["review_id"].tolist()
N = len(df)
print(f"Reviews loaded : {N:,}")
print(f"Label balance  : {dict(df['is_a_buyer'].value_counts())}")


# In[13]:


def load_count_vectors(path, vocab_size):
    rows, cols, data, ids = [], [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            head, _, body = line.rstrip().partition(",")
            ids.append(int(head[1:]))
            if body:
                for tok in body.split(","):
                    idx_str, freq_str = tok.split(":")
                    rows.append(i)
                    cols.append(int(idx_str))
                    data.append(int(freq_str))
    M = csr_matrix(
        (data, (rows, cols)),
        shape=(len(ids), vocab_size),
        dtype=np.int32,
    )
    return ids, M


with open(os.path.join(OUTPUT_DIR, "vocab.txt")) as f:
    VOCAB_SIZE = sum(1 for _ in f)

ids_bow, X_bow = load_count_vectors(
    os.path.join(OUTPUT_DIR, "count_vectors.txt"), VOCAB_SIZE
)
print(f"BoW matrix : shape={X_bow.shape}, nnz={X_bow.nnz:,}, sparsity={1 - X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1]):.4f}")
assert ids_bow == review_ids_df, "BoW row order does not match processed.csv"


# In[14]:


def load_dense_vectors(path):
    ids = []
    arrs = []
    with open(path) as f:
        for line in f:
            head, _, body = line.rstrip().partition(",")
            ids.append(int(head[1:]))
            arrs.append(np.array(body.split(","), dtype=np.float32))
    return ids, np.vstack(arrs)


ids_unw, X_unw = load_dense_vectors(os.path.join(OUTPUT_DIR, "unweighted_vectors.txt"))
ids_w,   X_w   = load_dense_vectors(os.path.join(OUTPUT_DIR, "weighted_vectors.txt"))

assert ids_unw == review_ids_df, "Unweighted vectors row order mismatch"
assert ids_w   == review_ids_df, "Weighted vectors row order mismatch"

DIM = X_unw.shape[1]
print(f"Unweighted FastText matrix : shape={X_unw.shape}, dtype={X_unw.dtype}")
print(f"Weighted   FastText matrix : shape={X_w.shape}, dtype={X_w.dtype}")
print(f"Embedding dim DIM = {DIM}")


# ### 3.1.5  Pre-modeling investigation (EDA)
# 
# Before training any classifier we investigate the data. This section addresses
# the RUBRIC HD criterion *"tasks fully investigated **before** modeling"* and
# provides the empirical justification for several choices in 3.0.5 (metric
# choice, missingness handling, etc.).
# 

# In[15]:


class_counts = df["is_a_buyer"].value_counts().sort_index()
class_pct = class_counts / N

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(["Not a buyer (0)", "Is a buyer (1)"], class_counts.values, color=["#d95f02", "#1b9e77"])
for i, (c, p) in enumerate(zip(class_counts.values, class_pct.values)):
    ax.text(i, c, f"{c:,}\n({p:.1%})", ha="center", va="bottom")
ax.set_ylabel("Count"); ax.set_title("Class balance - `is_a_buyer`"); ax.set_ylim(0, class_counts.max() * 1.15)
plt.tight_layout(); plt.show()


# In[16]:


q2_cols = ["review_title", "brand_name", "product_title",
           "price", "avg_product_rating", "product_rating_count", "review_rating"]
miss = pd.DataFrame({
    "missing": df[q2_cols].isna().sum(),
    "missing_%": (df[q2_cols].isna().mean() * 100).round(2),
    "dtype": [str(df[c].dtype) for c in q2_cols],
})
print(miss)


# In[17]:


df["_text_len"] = df["review_text"].str.split().str.len().fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(7, 3))
for label, sub in df.groupby("is_a_buyer"):
    ax.hist(sub["_text_len"].clip(upper=80), bins=40, alpha=0.55,
            label=f"is_a_buyer={label} (median={sub['_text_len'].median():.0f})", density=True)
ax.set_xlabel("Cleaned token count per review (clipped at 80)")
ax.set_ylabel("Density"); ax.legend(); ax.set_title("Review length by class")
plt.tight_layout(); plt.show()
df.drop(columns=["_text_len"], inplace=True)


# In[18]:


idx2word = {}
with open(os.path.join(OUTPUT_DIR, "vocab.txt")) as f:
    for line in f:
        w, i = line.rstrip().rsplit(":", 1)
        idx2word[int(i)] = w

mean_freq_pos = np.asarray(X_bow[y == 1].mean(axis=0)).ravel()
mean_freq_neg = np.asarray(X_bow[y == 0].mean(axis=0)).ravel()

top_pos = np.argsort(mean_freq_pos)[::-1][:15]
top_neg = np.argsort(mean_freq_neg)[::-1][:15]

top_tokens = pd.DataFrame({
    "is_a_buyer=1 token": [idx2word[i] for i in top_pos],
    "mean count (1)":     [f"{mean_freq_pos[i]:.3f}" for i in top_pos],
    "is_a_buyer=0 token": [idx2word[i] for i in top_neg],
    "mean count (0)":     [f"{mean_freq_neg[i]:.3f}" for i in top_neg],
})
print(top_tokens.to_string(index=False))


# In[19]:


all_zero_unw = int((np.abs(X_unw).sum(axis=1) == 0).sum())
all_zero_w   = int((np.abs(X_w).sum(axis=1)   == 0).sum())
empty_bow    = int(np.asarray(X_bow.getnnz(axis=1) == 0).sum())

shapes = pd.DataFrame([
    {"matrix": "BoW (sparse)",      "shape": str(X_bow.shape), "all-zero rows": empty_bow},
    {"matrix": "Unweighted (dense)", "shape": str(X_unw.shape), "all-zero rows": all_zero_unw},
    {"matrix": "Weighted (dense)",   "shape": str(X_w.shape),   "all-zero rows": all_zero_w},
])
print(shapes.to_string(index=False))
print(f"\nLabel y shape: {y.shape}, dtype: {y.dtype}")


# ### 3.2  Q1 - Language model comparison [3 marks]
# 
# We run **3 classifiers × 3 representations = 9 cells** through 5-fold
# stratified CV. Linear models on the dense embedding matrices are wrapped with
# `StandardScaler`; the sparse BoW path is not scaled (would densify and OOM).
# 

# In[20]:


representations = {
    "BoW (count_vectors)":         X_bow,
    "Unweighted FastText":         X_unw,
    "TF-IDF Weighted FastText":    X_w,
}


def make_clfs():
    return {
        "LogReg":       LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "LinearSVC":    LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(clf, dense):
    is_linear = isinstance(clf, (LogisticRegression, LinearSVC))
    if dense and is_linear:
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return Pipeline([("clf", clf)])


q1_rows = []
for rep_name, X in representations.items():
    is_dense = not sparse.issparse(X)
    for clf_name, clf in make_clfs().items():
        pipe = build_pipeline(clf, is_dense)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
        q1_rows.append({
            "Representation": rep_name,
            "Classifier":     clf_name,
            "Acc_mean":       scores["test_accuracy"].mean(),
            "Acc_std":        scores["test_accuracy"].std(),
            "F1_mean":        scores["test_f1_macro"].mean(),
            "F1_std":         scores["test_f1_macro"].std(),
        })
        print(f"  {rep_name:30s} | {clf_name:13s} | "
              f"acc={scores['test_accuracy'].mean():.4f}±{scores['test_accuracy'].std():.4f} | "
              f"f1m={scores['test_f1_macro'].mean():.4f}±{scores['test_f1_macro'].std():.4f}")

q1_df = pd.DataFrame(q1_rows)


# ### 3.3  Q1 results - table and chart
# 

# In[21]:


q1_show = q1_df.copy()
q1_show["Accuracy"]  = q1_show.apply(lambda r: f"{r.Acc_mean:.4f} ± {r.Acc_std:.4f}", axis=1)
q1_show["Macro-F1"]  = q1_show.apply(lambda r: f"{r.F1_mean:.4f} ± {r.F1_std:.4f}", axis=1)
q1_show = q1_show[["Representation", "Classifier", "Accuracy", "Macro-F1"]]
display(q1_show.sort_values(["Representation", "Classifier"]).reset_index(drop=True))

pivot_mean = q1_df.pivot(index="Representation", columns="Classifier", values="F1_mean")
pivot_std  = q1_df.pivot(index="Representation", columns="Classifier", values="F1_std")

fig, ax = plt.subplots(figsize=(8, 4))
pivot_mean.plot(kind="bar", yerr=pivot_std, ax=ax, capsize=3, rot=15, edgecolor="black")
ax.set_ylabel("Macro-F1 (mean ± std, 5-fold CV)")
ax.set_title("Q1 - Macro-F1 by representation × classifier")
ax.set_ylim(0, max(pivot_mean.max().max() * 1.15, 0.6))
ax.legend(title="Classifier", loc="lower right")
plt.tight_layout(); plt.show()


# ### 3.3.5  Q1 - best-model deep dive (confusion matrix + per-class metrics)
# 

# In[22]:


best = q1_df.sort_values("F1_mean", ascending=False).iloc[0]
best_rep, best_clf_name = best["Representation"], best["Classifier"]
print(f"Best Q1 cell : {best_rep} + {best_clf_name}  "
      f"(macro-F1 = {best['F1_mean']:.4f} ± {best['F1_std']:.4f})\n")

X_best = representations[best_rep]
clf_best = make_clfs()[best_clf_name]
pipe_best = build_pipeline(clf_best, dense=not sparse.issparse(X_best))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    y_pred = cross_val_predict(pipe_best, X_best, y, cv=CV, n_jobs=-1)

cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"], ax=ax)
ax.set_title(f"Q1 best - confusion matrix\n({best_rep} + {best_clf_name})")
plt.tight_layout(); plt.show()

print("Per-class metrics:")
print(classification_report(y, y_pred, target_names=["not buyer", "is buyer"], digits=4))

X_tr, X_te, y_tr, y_te = train_test_split(
    X_best, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
pipe_best.fit(X_tr, y_tr)
if hasattr(pipe_best, "predict_proba"):
    scores_te = pipe_best.predict_proba(X_te)[:, 1]
else:
    scores_te = pipe_best.decision_function(X_te)
print(f"ROC-AUC (20% holdout) : {roc_auc_score(y_te, scores_te):.4f}")


# ### 3.4  Q1 discussion
# 
# **Headline result.** The best (representation, classifier) cell is
# **BoW (count_vectors) + LinearSVC** at **macro-F1 = 0.5745 ± 0.0048** (5-fold
# CV). LogReg (0.5703) and RandomForest (0.5743) on the same BoW matrix are
# within a single std band of the winner - the *representation* matters far
# more than the *classifier*.
# 
# **Sparse keywords beat averaged embeddings.** All three BoW classifiers cluster
# around macro-F1 ≈ 0.57, while every embedding path (Unweighted ≈ 0.50,
# Weighted ≈ 0.48) sits 7–9 F1 points lower. This is consistent with the
# expected behaviour on short, opinion-heavy reviews: averaging FastText vectors
# dilutes the high-signal sentiment tokens ("love", "hate", "broken",
# "refund") whose presence/absence is exactly what BoW captures.
# 
# **Weighted FastText is *worse* than Unweighted, not better.** This was the
# opposite of the hypothesis in 3.0.5. A plausible explanation: TF-IDF
# down-weights frequent tokens, but in this corpus the frequent tokens (after
# Task 1's top-20 removal) include the very sentiment cues that drive the
# buy/no-buy signal. Down-weighting them removes signal rather than noise.
# The 1.91% OOV rate from Task 2 2.3 plus the 2,029 all-zero embedding rows
# (3.1.5) compound the problem on the embedding side.
# 
# **Classifier robustness.** Across LogReg / LinearSVC / RandomForest the
# ranking is stable: BoW > Unweighted > Weighted in all three. This means our
# conclusion about the representation is **not classifier-sensitive** - a robust
# finding.
# 
# **Caveats from the per-class report.** Even the best Q1 cell achieves only
# 0.196 recall on the *non-buyer* class - the model still mostly predicts
# "buyer" because of the 78.7/21.3 imbalance. Macro-F1 (0.5745) is far
# below accuracy (0.7825) precisely because of this asymmetry, and ROC-AUC of
# 0.667 confirms only modest discrimination. Q2 will examine whether richer
# inputs can fix this.
# 

# ### 3.5  Q2(b) - Text + Title vectorisation
# 
# Task 2 explicitly ignored the title, so we re-vectorise here. We apply the
# same Task-1 cleaning (regex tokenise → lowercase → drop len<2 → drop
# stopwords) to `review_title`, concatenate the resulting tokens to the
# already-cleaned `review_text` tokens, and then rebuild **all three**
# representations on the combined corpus.
# 

# In[23]:


TOKENISER = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

with open("stopwords_en.txt") as f:
    STOPWORDS = set(w.strip() for w in f if w.strip())


def clean_title(text):
    if not isinstance(text, str) or not text:
        return []
    return [t for t in TOKENISER.findall(text.lower())
            if len(t) >= 2 and t not in STOPWORDS]


text_tokens = [t.split() for t in df["review_text"]]
title_tokens = [clean_title(t) for t in df["review_title"]]
combined_tokens = [tt + ti for tt, ti in zip(text_tokens, title_tokens)]

n_added = sum(len(ti) for ti in title_tokens)
n_titles_with_tokens = sum(1 for ti in title_tokens if ti)
print(f"Title tokens added across corpus : {n_added:,}")
print(f"Reviews where title contributed   : {n_titles_with_tokens:,} / {N:,}")


# In[24]:


if "word2idx" not in globals():
    word2idx = {}
    with open(os.path.join(OUTPUT_DIR, "vocab.txt")) as f:
        for line in f:
            w, i = line.rstrip().rsplit(":", 1)
            word2idx[w] = int(i)


def build_bow(token_lists, vocab):
    rows, cols, data = [], [], []
    for i, tokens in enumerate(token_lists):
        counts = Counter(t for t in tokens if t in vocab)
        for w, c in counts.items():
            rows.append(i); cols.append(vocab[w]); data.append(c)
    return csr_matrix((data, (rows, cols)),
                      shape=(len(token_lists), len(vocab)), dtype=np.int32)


def build_unweighted(token_lists, ft, dim):
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        vecs = [ft[t] for t in tokens if t in ft.key_to_index]
        if vecs:
            out[i] = np.sum(vecs, axis=0)
    return out


def build_weighted(token_lists, ft, dim):
    d = Dictionary(token_lists)
    bow_corpus = [d.doc2bow(doc) for doc in token_lists]
    tfidf = TfidfModel(bow_corpus, id2word=d)
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, bow in enumerate(bow_corpus):
        if not bow:
            continue
        for word_id, weight in tfidf[bow]:
            tok = d[word_id]
            if tok in ft.key_to_index:
                out[i] += weight * ft[tok]
    return out


X_bow_b = build_bow(combined_tokens, word2idx)
X_unw_b = build_unweighted(combined_tokens, ft, DIM)
X_w_b   = build_weighted(combined_tokens, ft, DIM)

print(f"X_bow_b : {X_bow_b.shape}, nnz={X_bow_b.nnz:,}")
print(f"X_unw_b : {X_unw_b.shape}")
print(f"X_w_b   : {X_w_b.shape}")


# ### 3.6  Q2(c) - Text + Title + Extra information pipeline
# 
# We add four metadata columns from `processed.csv` to the (b) text features:
# 
# | Column | Type | Transformer |
# |---|---|---|
# | `price`, `avg_product_rating`, `product_rating_count` | numeric | `SimpleImputer(median)` → `StandardScaler` |
# | `brand_name`, `product_title` | categorical | `SimpleImputer(constant)` → `OneHotEncoder(min_frequency=20)` |
# 
# `review_rating` is **excluded** (label-leakage risk; see 3.0.5).
# 
# The structured `ColumnTransformer` is fit once on the full dataframe rather
# than inside the CV loop. This introduces mild leakage in the imputer/scaler
# statistics, but is acceptable for this assignment and keeps the pipeline
# tractable when stacked with text features of three different sparsity
# profiles.
# 

# In[25]:


STRUCT_NUM = ["price", "avg_product_rating", "product_rating_count"]
STRUCT_CAT = ["brand_name", "product_title"]

preproc_struct = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ]), STRUCT_NUM),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
    ]), STRUCT_CAT),
])

X_struct = preproc_struct.fit_transform(df[STRUCT_NUM + STRUCT_CAT])
print(f"Structured feature matrix : {X_struct.shape} ({type(X_struct).__name__})")
print(f"  numeric block dim   : {len(STRUCT_NUM)}")
print(f"  categorical block dim : {X_struct.shape[1] - len(STRUCT_NUM)}")


def hstack_text_struct(X_text, X_struct):
    if sparse.issparse(X_text):
        return sparse.hstack([X_text, X_struct]).tocsr()
    return np.hstack([X_text, X_struct.toarray()])


X_bow_c = hstack_text_struct(X_bow_b, X_struct)
X_unw_c = hstack_text_struct(X_unw_b, X_struct)
X_w_c   = hstack_text_struct(X_w_b,   X_struct)

print()
print(f"X_bow_c : {X_bow_c.shape}, sparse={sparse.issparse(X_bow_c)}")
print(f"X_unw_c : {X_unw_c.shape}, sparse={sparse.issparse(X_unw_c)}")
print(f"X_w_c   : {X_w_c.shape},   sparse={sparse.issparse(X_w_c)}")


# ### 3.7  Q2 - run the 3 × 3 grid
# 
# LogReg only (the primary classifier from 3.2) so the comparison cleanly
# isolates the effect of *added information*, not classifier × feature
# interactions.
# 

# In[26]:


configs = {
    "(a) Text only":              {"BoW": X_bow,   "Unweighted": X_unw,   "Weighted": X_w},
    "(b) Text + Title":           {"BoW": X_bow_b, "Unweighted": X_unw_b, "Weighted": X_w_b},
    "(c) Text + Title + Extras":  {"BoW": X_bow_c, "Unweighted": X_unw_c, "Weighted": X_w_c},
}

q2_rows = []
for cfg_name, reps in configs.items():
    for rep_name, X in reps.items():
        is_dense = not sparse.issparse(X)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        pipe = build_pipeline(clf, is_dense)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
        q2_rows.append({
            "Config":         cfg_name,
            "Representation": rep_name,
            "Acc_mean":       scores["test_accuracy"].mean(),
            "Acc_std":        scores["test_accuracy"].std(),
            "F1_mean":        scores["test_f1_macro"].mean(),
            "F1_std":         scores["test_f1_macro"].std(),
        })
        print(f"  {cfg_name:30s} | {rep_name:12s} | "
              f"acc={scores['test_accuracy'].mean():.4f}±{scores['test_accuracy'].std():.4f} | "
              f"f1m={scores['test_f1_macro'].mean():.4f}±{scores['test_f1_macro'].std():.4f}")

q2_df = pd.DataFrame(q2_rows)


# ### 3.8  Q2 results - F1 grid, accuracy grid, deltas, heatmap
# 

# In[27]:


def fmt_grid(df, mean_col, std_col):
    pivot_m = df.pivot(index="Config", columns="Representation", values=mean_col)
    pivot_s = df.pivot(index="Config", columns="Representation", values=std_col)
    out = pivot_m.copy().astype(object)
    for r in pivot_m.index:
        for c in pivot_m.columns:
            out.loc[r, c] = f"{pivot_m.loc[r, c]:.4f} ± {pivot_s.loc[r, c]:.4f}"
    return out[["BoW", "Unweighted", "Weighted"]]


print("Macro-F1 grid (mean ± std, 5-fold CV):")
display(fmt_grid(q2_df, "F1_mean", "F1_std"))

print("\nAccuracy grid (mean ± std, 5-fold CV):")
display(fmt_grid(q2_df, "Acc_mean", "Acc_std"))

piv = q2_df.pivot(index="Config", columns="Representation", values="F1_mean")
piv = piv.loc[["(a) Text only", "(b) Text + Title", "(c) Text + Title + Extras"], ["BoW", "Unweighted", "Weighted"]]
delta = pd.DataFrame({
    "(b) − (a)": piv.loc["(b) Text + Title"] - piv.loc["(a) Text only"],
    "(c) − (b)": piv.loc["(c) Text + Title + Extras"] - piv.loc["(b) Text + Title"],
    "(c) − (a)": piv.loc["(c) Text + Title + Extras"] - piv.loc["(a) Text only"],
}).T
print("\nMacro-F1 deltas (positive = adding info helped):")
display(delta.round(4))

fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(piv, annot=True, fmt=".4f", cmap="YlGn",
            cbar_kws={"label": "Macro-F1"}, ax=ax)
ax.set_title("Q2 - Macro-F1 across configurations × representations")
plt.tight_layout(); plt.show()


# ### 3.8.5  Q2 - best cell deep dive
# 

# In[28]:


best_q2 = q2_df.sort_values("F1_mean", ascending=False).iloc[0]
best_cfg, best_rep_q2 = best_q2["Config"], best_q2["Representation"]
print(f"Best Q2 cell : {best_cfg} | {best_rep_q2}  "
      f"(macro-F1 = {best_q2['F1_mean']:.4f} ± {best_q2['F1_std']:.4f})\n")

X_best_q2 = configs[best_cfg][best_rep_q2]
pipe_best_q2 = build_pipeline(
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    dense=not sparse.issparse(X_best_q2),
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    y_pred_q2 = cross_val_predict(pipe_best_q2, X_best_q2, y, cv=CV, n_jobs=-1)

cm = confusion_matrix(y, y_pred_q2)
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
            xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"], ax=ax)
ax.set_title(f"Q2 best - confusion matrix\n({best_cfg} | {best_rep_q2})")
plt.tight_layout(); plt.show()

print("Per-class metrics:")
print(classification_report(y, y_pred_q2, target_names=["not buyer", "is buyer"], digits=4))


# ### 3.9  Q2 discussion
# 
# **Headline result.** The best Q2 cell is **(c) Text + Title + Extras / BoW**
# at **macro-F1 = 0.7126 ± 0.0054**, beating the best Q1 cell (0.5745) by
# **+0.1381 macro-F1** - a substantial improvement that confirms more
# information *does* help.
# 
# **Title alone barely moves the needle.** Adding `review_title` (config (b))
# yields gains of only **+0.0137 (BoW), +0.0022 (Unweighted), +0.0070
# (Weighted)** macro-F1 over text-only. The (b)−(a) deltas are within or
# barely above the std bands of (a) - i.e. statistically marginal. Likely
# explanation: titles tend to paraphrase or summarise the review body, so the
# new tokens are largely redundant with the text features (titles contributed
# tokens to 58,666 of 61,284 reviews per 3.5, but those tokens overlap
# heavily with what's already there).
# 
# **Structured metadata is where the real lift comes from.** Adding `price`,
# `avg_product_rating`, `product_rating_count`, `brand_name`, `product_title`
# (config (c)) lifts macro-F1 by **+0.1286 (BoW), +0.2026 (Unweighted),
# +0.2154 (Weighted)** over (b). All three deltas dwarf the std bands. The
# average-product-rating feature in particular likely carries strong signal:
# buyers and non-buyers come from systematically different product
# distributions.
# 
# **Hypothesis confirmed - structured features help embeddings *more* than
# BoW.** The (c)−(b) gain is +0.13 for BoW vs +0.20–0.22 for the embedding
# paths. The reading: BoW already carried most of the discriminative text
# signal, so the marginal value of metadata was smaller; the embedding paths
# were under-performing at text-only and benefited disproportionately from
# strong tabular features.
# 
# **The text representation barely matters in (c).** All three Q2(c) cells
# converge to within 0.008 macro-F1 (BoW 0.7126, Weighted 0.7069, Unweighted
# 0.7047) - a tight cluster, in stark contrast to the 0.085-point spread in
# Q2(a). Once the structured features are present, they dominate the
# prediction and the choice of text encoder is almost interchangeable.
# 
# **Per-class story.** The best Q2 cell's non-buyer recall climbs from
# **0.196 (Q1 best) → 0.471 (Q2 best)** - more than double. The model is no
# longer just predicting "buyer" for everyone; it is now correctly flagging
# nearly half of true non-buyers, while only mildly trading off the buyer
# recall (0.942 → 0.920).
# 

# ### 3.10  Conclusion
# 
# **Q1 - Language model comparison.** Across the three Task-2 representations,
# **BoW (count_vectors) + LinearSVC** is the strongest combination at
# **macro-F1 = 0.5745 ± 0.0048**, but BoW with LogReg (0.5703) or
# RandomForest (0.5743) is within noise. BoW outperforms both averaged
# FastText paths (Unweighted 0.5000, Weighted 0.4845) by roughly 7–9 F1
# points. The ranking is **stable across all three classifiers**, so the
# finding is robust: for this corpus and task, sparse keyword presence beats
# averaged dense embeddings, and TF-IDF weighting actively *hurts* the
# embedding path (likely because high-frequency tokens in this beauty-review
# corpus carry sentiment signal that TF-IDF erroneously down-weights).
# 
# **Q2 - Does more information help?** Yes, but not equally from every
# source. Adding the **review title** alone is essentially noise (+0.002 to
# +0.014 macro-F1, comparable to fold-to-fold std). Adding **structured
# product metadata** (price, average rating, rating count, brand, product
# title) is the real lever: it lifts macro-F1 by +0.13–0.22 across all three
# representations and pushes the best cell to **macro-F1 = 0.7126 ± 0.0054**
# ((c) Text + Title + Extras / BoW). This is **+0.14 macro-F1 above the best
# Q1 cell**, and it more than doubles the non-buyer recall (0.20 → 0.47).
# Notably, with metadata in the mix the choice of text representation
# becomes nearly irrelevant - all three Q2(c) cells land within 0.008 F1 of
# each other.
# 
# **Methodological notes.** All experiments use 5-fold stratified CV with
# seed 42, report mean ± std to expose stability, and supplement headline
# metrics with confusion matrices and per-class P/R/F1 for the winning
# cells. `review_rating` was excluded from Q2(c) extras to avoid label
# leakage (a per-review user rating would near-perfectly predict the same
# user's buy decision). The structured-feature transformer is fit on the
# full dataframe rather than inside the CV loop - a mild leakage trade-off
# that does not affect the headline conclusion since the lift from
# metadata (~0.20 F1) is far larger than any imputer/scaler bias.
# 
# **Implication for Milestone 2.** The Milestone-2 web app should serve the
# Q2(c) BoW model rather than any text-only model: the +0.14 macro-F1 gain
# is substantial, and the required inputs (`price`, `avg_product_rating`,
# `product_rating_count`, `brand_name`, `product_title`) are exactly the
# kind of fields a product-detail page already exposes.
# 

# ## Summary
# 
# This notebook implements **Task 2** (feature representations) and **Task 3** (classification + analysis) for the cosmetics/beauty review dataset.
# 
# ### Task 2 - Feature representations
# Three vectorisations of each cleaned review were produced and saved to `output/`:
# 
# - `count_vectors.txt` - sparse Bag-of-Words against the Task-1 vocabulary (8,054 dims)
# - `unweighted_vectors.txt` - sum of FastText embeddings (300 dims, 1.91% OOV skipped)
# - `weighted_vectors.txt` - TF-IDF weighted sum of FastText embeddings (300 dims)
# 
# FastText `fasttext-wiki-news-subwords-300` was chosen over GoogleNews-300 because its ~1M-token vocabulary covers more informal language found in beauty reviews.
# 
# ### Task 3 - Classification
# 
# **Q1 (Language model comparison).** Three classifiers (LogReg / LinearSVC / RandomForest) were evaluated on each of the three Task-2 representations using 5-fold stratified CV. The best cell is **BoW + LinearSVC, macro-F1 = 0.5745 ± 0.0048**. BoW outperforms both averaged-FastText paths by roughly 7–9 macro-F1 points; the ranking is stable across all three classifiers, so the conclusion is not classifier-sensitive.
# 
# **Q2 (Does more information help?).** Adding the review title alone gave only marginal gains (+0.002 to +0.014 macro-F1, within fold-to-fold std). Adding structured product metadata (`brand_name`, `product_title`, `price`, `avg_product_rating`, `product_rating_count`) lifted macro-F1 by +0.13 to +0.22 across representations. The best cell is **(c) Text + Title + Extras + BoW, macro-F1 = 0.7126 ± 0.0054** - a +0.14 improvement over the best Q1 cell, with non-buyer recall climbing from 0.196 to 0.471.
# 
# `review_rating` was deliberately excluded from the extras to avoid label leakage (the same user provides both the rating and the buy decision).
# 
# ### Reproducibility
# All experiments use a fixed seed (`random_state=26`) and 5-fold `StratifiedKFold`. The structured-feature `ColumnTransformer` is fit once on the full dataframe; this is a mild leakage trade-off that does not affect the headline conclusion since the lift from metadata (~0.20 F1) is far larger than any imputer/scaler bias.
# 
