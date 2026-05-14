import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

MODEL_DIR = Path(__file__).resolve().parent / ".." / "model"

THRESHOLD = 0.5
threshold_path = MODEL_DIR / "threshold.json"
if threshold_path.exists():
    with open(threshold_path) as f:
        THRESHOLD = json.load(f)["threshold"]
    print(f"Loaded optimal threshold: {THRESHOLD}")
else:
    print("No threshold.json found, using default 0.5")

STOPWORDS = set()
with open(MODEL_DIR / "stopwords_en.txt", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if word:
            STOPWORDS.add(word)
print(f"Loaded {len(STOPWORDS)} stopwords")

VOCAB = {}
with open(MODEL_DIR / "vocab.txt", encoding="utf-8") as f:
    for line in f:
        word, idx = line.rstrip("\n").rsplit(":", 1)
        VOCAB[word] = int(idx)
print(f"Loaded vocab: {len(VOCAB)} words")

clf_bow = joblib.load(MODEL_DIR / "clf_bow.joblib")
pipe_unw = joblib.load(MODEL_DIR / "pipe_unw.joblib")
pipe_w = joblib.load(MODEL_DIR / "pipe_w.joblib")
preproc_struct = joblib.load(MODEL_DIR / "preproc_struct.joblib")
ft_vectors = np.load(MODEL_DIR / "ft_vectors.npy")
tfidf_dict = Dictionary.load(str(MODEL_DIR / "tfidf.dict"))
tfidf_model = TfidfModel.load(str(MODEL_DIR / "tfidf.model"))

print(f"All models loaded (ft_vectors shape: {ft_vectors.shape})")

TOKENISER = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")


def clean(text):
    if not text or not isinstance(text, str):
        return []
    return [t for t in TOKENISER.findall(text.lower())
            if len(t) >= 2 and t not in STOPWORDS]


def predict(review_text, review_title, brand_name, product_title,
            price, avg_product_rating, product_rating_count):
    tokens = clean(review_text) + clean(review_title)

    # Model 1: Bag-of-Words (text only — no metadata)
    bow_counts = np.zeros(len(VOCAB))
    for token in tokens:
        idx = VOCAB.get(token)
        if idx is not None:
            bow_counts[idx] += 1
    X_bow = csr_matrix(bow_counts.reshape(1, -1))
    p_bow = float(clf_bow.predict_proba(X_bow)[0, 1])

    # Model 2: Unweighted embeddings (text only — no metadata)
    embedding = np.zeros(ft_vectors.shape[1])
    for token in tokens:
        idx = VOCAB.get(token)
        if idx is not None:
            embedding += ft_vectors[idx]
    X_unw = embedding.reshape(1, -1)
    p_unw = float(pipe_unw.predict_proba(X_unw)[0, 1])

    # Model 3: TF-IDF weighted embeddings + metadata (text + structured)
    struct_input = pd.DataFrame([{
        "brand_name": brand_name,
        "product_title": product_title,
        "price": price,
        "avg_product_rating": avg_product_rating,
        "product_rating_count": product_rating_count,
    }])
    struct_sparse = preproc_struct.transform(struct_input)
    struct_dense = struct_sparse.toarray() if hasattr(struct_sparse, "toarray") else np.array(struct_sparse)

    bow_gensim = tfidf_dict.doc2bow(tokens)
    tfidf_weights = tfidf_model[bow_gensim]
    id2word = {v: k for k, v in tfidf_dict.token2id.items()}
    weighted_embedding = np.zeros(ft_vectors.shape[1])
    for word_id, weight in tfidf_weights:
        word = id2word.get(word_id, "")
        idx = VOCAB.get(word)
        if idx is not None:
            weighted_embedding += weight * ft_vectors[idx]
    weighted_dense = weighted_embedding.reshape(1, -1)
    X_w = np.hstack([weighted_dense, struct_dense])
    p_w = float(pipe_w.predict_proba(X_w)[0, 1])

    p_fused = (p_bow + p_unw + p_w) / 3.0

    return {
        "probability": round(p_fused, 4),
        "label": "Likely Buyer" if p_fused >= THRESHOLD else "Unlikely Buyer",
        "models": {
            "bow": {"probability": round(p_bow, 4), "name": "Bag-of-Words"},
            "unweighted": {"probability": round(p_unw, 4), "name": "Unweighted Embeddings"},
            "weighted": {"probability": round(p_w, 4), "name": "TF-IDF Weighted Embeddings"},
        },
    }
