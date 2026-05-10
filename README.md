# GlowCart — Cosmetics & Beauty Product Review Platform

> COSC3801/3015 Advanced Programming for Data Science — Assignment 3  
> Mai Dang Khoa (s3974876) · Dang Cuu Dang Khoa (s3979159) · Tran Quang Minh (s3988776)

An ML-powered online shopping website for cosmetics and beauty products. Shoppers can browse products, search by keyword, read reviews, and submit new reviews — each of which receives a real-time buyer prediction from a 3-model ensemble trained on 61K+ product reviews.

---

## Features

| Feature | Description |
|---|---|
| **Product Browsing** | Filter by brand, price range, rating. Sort by name, price, rating, review count. Paginated grid with product images. |
| **Keyword Search** | Fuzzy search across brand and product names. Handles typos (e.g. "Maybeline" → Maybelline) via SequenceMatcher. |
| **Review + Buyer Prediction** | Submit a review → 3 ML models predict buyer likelihood → user sees prediction with per-model confidence → can override before confirming. |
| **Similar Products** | TF-IDF cosine similarity on aggregated review text. "You May Also Like" section on every product page. |
| **Analytics Dashboard** | Total products/reviews/brands, rating distribution, per-brand buyer %, top reviewed products. |
| **Dark Mode** | Toggle in navbar, persisted in localStorage. |
| **Responsive Design** | Mobile, tablet, and desktop layouts via Tailwind CSS. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.13, Flask, scikit-learn, gensim, pandas, numpy |
| Frontend | React 19, React Router 7, Vite, Tailwind CSS 4 |
| ML Models | LogisticRegression (BoW), FastText embeddings (weighted + unweighted) |
| Deployment | Docker (multi-stage), Gunicorn |

---

## Project Structure

```
├── milestone1/
│   ├── data/
│   │   ├── task1.ipynb                 # Task 1: Text preprocessing
│   │   ├── task2_3.ipynb               # Task 2 & 3: Feature representations + classification
│   │   ├── task1.py                    # .py export of task1.ipynb
│   │   ├── task2_3.py                  # .py export of task2_3.ipynb
│   │   ├── cosmetics_beauty_products_reviews.csv
│   │   └── stopwords_en.txt
│   └── output/
│       ├── processed.csv
│       ├── vocab.txt
│       ├── count_vectors.txt
│       ├── unweighted_vectors.txt
│       └── weighted_vectors.txt
│
├── milestone2/
│   ├── core/                           # Flask backend
│   │   ├── app.py                      # API routes + SPA serving
│   │   ├── data.py                     # Data loading, search, similarity
│   │   └── util.py                     # ML prediction pipeline
│   ├── web/                            # React frontend
│   │   └── src/
│   │       ├── pages/                  # Home, Search, ProductDetail, Dashboard
│   │       └── components/             # ProductCard, Navbar
│   ├── model/                          # Trained ML artifacts (~10 MB)
│   ├── data/
│   │   └── processed_with_images.csv   # Reviews + image URLs
│   ├── Dockerfile
│   └── README.txt                      # Submission README with full details
```

---

## ML Pipeline

### Preprocessing
61,284 cosmetics reviews → tokenised → lowercased → removed short words, stopwords, hapaxes, and top-20 frequent terms → **8,054-word vocabulary**.

### Feature Representations
| Representation | Dimensions | Method |
|---|---|---|
| Bag-of-Words | 8,054 | Sparse count vectors over vocabulary |
| Unweighted Embeddings | 300 | Sum of FastText wiki-news-subwords-300 vectors |
| TF-IDF Weighted Embeddings | 300 | TF-IDF weighted sum of FastText vectors |

### Classification Results (5-fold stratified CV)

| Configuration | Best Macro-F1 |
|---|---|
| Q1: Review text only (BoW + LinearSVC) | 0.5745 |
| Q2(b): + Review title | 0.5882 |
| **Q2(c): + Structural features (deployed)** | **0.7126** |

Structural features: `brand_name`, `product_title`, `price`, `avg_product_rating`, `product_rating_count`.  
`review_rating` excluded to avoid label leakage.

### Why Include `product_title`?

In real e-commerce, the same review text produces different buying outcomes on different products. A glowing review on a $5 drugstore mascara converts differently than on a $60 prestige serum. Including `product_title` (one-hot encoded, `min_frequency=20`) lets the model learn product-level purchasing patterns — boosting macro-F1 by +0.14 and more than doubling non-buyer recall (0.196 → 0.471).

### Ensemble Prediction

Three classifiers run in parallel at inference, each combining text features with structural features:

1. **BoW** — sparse count vector + metadata → LogisticRegression
2. **Unweighted** — FastText embedding sum + metadata → LogisticRegression
3. **Weighted** — TF-IDF weighted FastText + metadata → LogisticRegression

Final probability = average of all three. Threshold = **0.3** (tuned for macro-F1).

---

## Getting Started

### Local Development

```bash
# Backend
cd milestone2/core
pip install -r requirements.txt
python app.py                    # → http://localhost:8000

# Frontend (separate terminal)
cd milestone2/web
npm install
npm run dev                      # → http://localhost:5173
```

### Docker

```bash
cd milestone2
docker build -t glowcart .
docker run -p 8000:8000 glowcart  # → http://localhost:8000
```

The Dockerfile uses a two-stage build:
- **Stage 1** (`node:22-alpine`): builds the React frontend
- **Stage 2** (`python:3.13-slim`): installs Python deps, copies model/data/frontend, runs Gunicorn (2 workers, 120s timeout)

Flask serves both the API (`/api/*`) and the built SPA (catch-all → `index.html`).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/brands` | List all brand names |
| GET | `/api/products` | Paginated product listing (supports `page`, `per_page`, `sort_by`, `brand`, `min_price`, `max_price`, `min_rating`) |
| GET | `/api/products/<id>` | Single product with all reviews |
| GET | `/api/products/<id>/similar` | Top-6 similar products |
| GET | `/api/search?q=<query>` | Fuzzy keyword search |
| GET | `/api/reviews/<id>` | Single review by ID |
| POST | `/api/reviews` | Create review + ML prediction |
| GET | `/api/stats` | Dashboard analytics |
