import math
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sentiment_analyzer = SentimentIntensityAnalyzer()

BASE_DIR = Path(__file__).resolve().parent

_primary_path = BASE_DIR / ".." / "data" / "processed_with_images.csv"
_fallback_path = BASE_DIR / ".." / ".." / "milestone1" / "output" / "processed.csv"
_csv_path = _primary_path if _primary_path.exists() else _fallback_path

_df = pd.read_csv(_csv_path)
print(f"Loaded {len(_df)} reviews from {_csv_path.resolve()}")

_review_counts = _df.groupby("product_id").size().reset_index(name="review_count")
_product_cols = ["product_id", "brand_name", "product_title", "price",
                 "avg_product_rating", "product_rating_count", "product_url"]
if "image" in _df.columns:
    _product_cols.append("image")

_products_df = _df.drop_duplicates(subset="product_id", keep="first")[
    _product_cols
].merge(_review_counts, on="product_id")


def _to_native(val):
    if hasattr(val, "item"):
        return val.item()
    return val


def _product_to_dict(row):
    return {k: _to_native(v) for k, v in row.items()}


PRODUCTS = [_product_to_dict(row) for _, row in _products_df.iterrows()]
_products_by_id = {p["product_id"]: p for p in PRODUCTS}

print(f"Built product table: {len(PRODUCTS)} products, "
      f"{_df['brand_name'].nunique()} brands")

def _analyze_sentiment(text):
    if not text or not isinstance(text, str) or not text.strip():
        return {"compound": 0.0, "label": "neutral"}
    scores = _sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": round(compound, 4), "label": label}


_reviews_by_product = {}
for pid, grp in _df.groupby("product_id"):
    reviews = []
    for _, r in grp.iterrows():
        text = str(r["review_text"]) if pd.notna(r["review_text"]) else ""
        sentiment = _analyze_sentiment(text)
        reviews.append({
            "review_id": int(r["review_id"]),
            "product_id": int(r["product_id"]),
            "review_title": str(r["review_title"]) if pd.notna(r["review_title"]) else "",
            "review_text": text,
            "author": str(r["author"]) if pd.notna(r["author"]) else "Unknown",
            "review_date": str(r["review_date"]) if pd.notna(r["review_date"]) else "",
            "review_rating": float(r["review_rating"]) if pd.notna(r["review_rating"]) else 0.0,
            "is_a_buyer": bool(r["is_a_buyer"]) if not isinstance(r["is_a_buyer"], str) else r["is_a_buyer"].lower() == "true",
            "sentiment": sentiment,
        })
    _reviews_by_product[int(pid)] = reviews

print("Sentiment analysis complete for all reviews")

_user_reviews = []
_next_review_id = int(_df["review_id"].max()) + 1

# --- Similarity matrix ---
_agg_text = _df.groupby("product_id")["review_text"].apply(
    lambda texts: " ".join(str(t) for t in texts if pd.notna(t))
).reset_index()
_agg_text = _agg_text.sort_values("product_id").reset_index(drop=True)

_vectorizer = TfidfVectorizer()
_tfidf_matrix = _vectorizer.fit_transform(_agg_text["review_text"])
_sim_matrix = cosine_similarity(_tfidf_matrix)

_sim_product_ids = _agg_text["product_id"].tolist()
_pid_to_idx = {pid: i for i, pid in enumerate(_sim_product_ids)}

_similarity = {}
for i, pid in enumerate(_sim_product_ids):
    scores = []
    for j, other_pid in enumerate(_sim_product_ids):
        if i != j:
            scores.append((int(other_pid), float(_sim_matrix[i, j])))
    scores.sort(key=lambda x: x[1], reverse=True)
    _similarity[int(pid)] = scores

print(f"Precomputed {len(_similarity)}x{len(_similarity)} similarity matrix")


def get_products(page=1, per_page=20, sort_by="name", brand=None,
                  min_price=None, max_price=None, min_rating=None):
    items = PRODUCTS
    if brand:
        items = [p for p in items if p["brand_name"].lower() == brand.lower()]
    if min_price is not None:
        items = [p for p in items if p.get("price", 0) >= min_price]
    if max_price is not None:
        items = [p for p in items if p.get("price", 0) <= max_price]
    if min_rating is not None:
        items = [p for p in items if p.get("avg_product_rating", 0) >= min_rating]

    sort_keys = {
        "rating": lambda p: (-p.get("avg_product_rating", 0),),
        "price_asc": lambda p: (p.get("price", 0),),
        "price_desc": lambda p: (-p.get("price", 0),),
        "reviews": lambda p: (-p.get("review_count", 0),),
        "name": lambda p: (p.get("product_title", "").lower(),),
    }
    key_fn = sort_keys.get(sort_by, sort_keys["name"])
    items = sorted(items, key=key_fn)

    total = len(items)
    pages = math.ceil(total / per_page) if total else 1
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "products": items[start:end],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }


def get_product(product_id):
    """Return single product dict with reviews, or None."""
    product = _products_by_id.get(product_id)
    if product is None:
        return None
    result = dict(product)
    existing = _reviews_by_product.get(product_id, [])
    user = [r for r in _user_reviews if r["product_id"] == product_id]
    result["reviews"] = existing + user
    return result


def search_products(query):
    if not query or not query.strip():
        return {"products": [], "count": 0, "query": query or ""}
    q_words = query.lower().split()
    scored = []
    for p in PRODUCTS:
        haystack = (p["brand_name"] + " " + p["product_title"]).lower()
        if all(w in haystack for w in q_words):
            scored.append((1.0, p))
            continue

        h_words = haystack.split()
        all_matched = True
        total_ratio = 0.0
        for qw in q_words:
            best = max(
                (SequenceMatcher(None, qw, hw).ratio() for hw in h_words),
                default=0.0,
            )
            if best < 0.6:
                all_matched = False
                break
            total_ratio += best
        if all_matched:
            scored.append((total_ratio / len(q_words), p))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [p for _, p in scored]
    return {"products": results, "count": len(results), "query": query}


def get_similar(product_id, n=6):
    """Return top-N similar products."""
    sim_list = _similarity.get(product_id)
    if sim_list is None:
        return []
    results = []
    for other_pid, score in sim_list[:n]:
        p = _products_by_id.get(other_pid)
        if p:
            entry = dict(p)
            entry["similarity"] = round(score, 3)
            results.append(entry)
    return results


def get_brands():
    return sorted({p["brand_name"] for p in PRODUCTS})


def get_review(review_id):
    for reviews in _reviews_by_product.values():
        for r in reviews:
            if r["review_id"] == review_id:
                return r
    for r in _user_reviews:
        if r["review_id"] == review_id:
            return r
    return None


def get_top_rated(n=10):
    rated = [p for p in PRODUCTS if p.get("avg_product_rating", 0) >= 4.0]
    rated.sort(key=lambda p: (-p.get("avg_product_rating", 0), -p.get("review_count", 0)))
    return rated[:n]


def _collect_all_reviews():
    all_reviews = []
    for reviews in _reviews_by_product.values():
        all_reviews.extend(reviews)
    all_reviews.extend(_user_reviews)
    return all_reviews


def get_sentiment_stats():
    all_reviews = _collect_all_reviews()
    positive = sum(1 for r in all_reviews if r.get("sentiment", {}).get("label") == "positive")
    negative = sum(1 for r in all_reviews if r.get("sentiment", {}).get("label") == "negative")
    neutral = sum(1 for r in all_reviews if r.get("sentiment", {}).get("label") == "neutral")
    total = len(all_reviews)
    avg_compound = sum(r.get("sentiment", {}).get("compound", 0) for r in all_reviews) / total if total else 0

    brand_sentiment = {}
    for r in all_reviews:
        product = _products_by_id.get(r["product_id"])
        if not product:
            continue
        brand = product["brand_name"]
        if brand not in brand_sentiment:
            brand_sentiment[brand] = {"total_compound": 0.0, "count": 0,
                                      "positive": 0, "negative": 0, "neutral": 0}
        s = r.get("sentiment", {})
        brand_sentiment[brand]["total_compound"] += s.get("compound", 0)
        brand_sentiment[brand]["count"] += 1
        label = s.get("label", "neutral")
        brand_sentiment[brand][label] += 1

    brands = []
    for brand, info in brand_sentiment.items():
        c = info["count"]
        brands.append({
            "brand": brand,
            "avg_sentiment": round(info["total_compound"] / c, 4) if c else 0,
            "positive": info["positive"],
            "negative": info["negative"],
            "neutral": info["neutral"],
            "total": c,
        })
    brands.sort(key=lambda x: x["avg_sentiment"], reverse=True)

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "avg_compound": round(avg_compound, 4),
        "by_brand": brands,
    }


def get_stats():
    total_products = len(PRODUCTS)
    total_brands = len({p["brand_name"] for p in PRODUCTS})

    all_reviews = _collect_all_reviews()
    total_reviews = len(all_reviews)

    buyers = sum(1 for r in all_reviews if r["is_a_buyer"])
    non_buyers = total_reviews - buyers

    rating_distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for r in all_reviews:
        star = str(int(round(r["review_rating"])))
        if star in rating_distribution:
            rating_distribution[star] += 1

    brand_data = {}
    for r in all_reviews:
        product = _products_by_id.get(r["product_id"])
        if product is None:
            continue
        brand = product["brand_name"]
        if brand not in brand_data:
            brand_data[brand] = {"reviews": 0, "buyers": 0, "total_rating": 0.0}
        brand_data[brand]["reviews"] += 1
        if r["is_a_buyer"]:
            brand_data[brand]["buyers"] += 1
        brand_data[brand]["total_rating"] += r["review_rating"]

    reviews_by_brand = []
    for brand, info in brand_data.items():
        count = info["reviews"]
        reviews_by_brand.append({
            "brand": brand,
            "reviews": count,
            "buyer_pct": round(info["buyers"] / count * 100, 1) if count else 0.0,
            "avg_rating": round(info["total_rating"] / count, 2) if count else 0.0,
        })
    reviews_by_brand.sort(key=lambda x: x["reviews"], reverse=True)

    product_review_counts = {}
    for r in all_reviews:
        pid = r["product_id"]
        if pid not in product_review_counts:
            product_review_counts[pid] = {"count": 0, "buyers": 0, "total_rating": 0.0}
        product_review_counts[pid]["count"] += 1
        if r["is_a_buyer"]:
            product_review_counts[pid]["buyers"] += 1
        product_review_counts[pid]["total_rating"] += r["review_rating"]

    sorted_products = sorted(product_review_counts.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
    top_products = []
    for pid, info in sorted_products:
        product = _products_by_id.get(pid)
        if product is None:
            continue
        count = info["count"]
        top_products.append({
            "product_id": pid,
            "product_title": product["product_title"],
            "brand_name": product["brand_name"],
            "review_count": count,
            "buyer_pct": round(info["buyers"] / count * 100, 1) if count else 0.0,
            "avg_rating": round(info["total_rating"] / count, 2) if count else 0.0,
        })

    sentiment = get_sentiment_stats()

    return {
        "total_products": total_products,
        "total_reviews": total_reviews,
        "total_brands": total_brands,
        "buyer_vs_non_buyer": {"buyers": buyers, "non_buyers": non_buyers},
        "rating_distribution": rating_distribution,
        "reviews_by_brand": reviews_by_brand,
        "top_products": top_products,
        "sentiment": sentiment,
    }


def add_review(product_id, review_title, review_text, review_rating,
               is_a_buyer, author="Anonymous"):
    """Add a user-created review and return it."""
    global _next_review_id
    if product_id not in _products_by_id:
        return None
    sentiment = _analyze_sentiment(review_text)
    review = {
        "review_id": _next_review_id,
        "product_id": product_id,
        "review_title": review_title,
        "review_text": review_text,
        "author": author,
        "review_date": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
        "review_rating": float(review_rating),
        "is_a_buyer": bool(is_a_buyer),
        "sentiment": sentiment,
    }
    _next_review_id += 1
    _user_reviews.append(review)
    return review
