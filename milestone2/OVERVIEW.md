# GlowCart — Project Overview

## 1. Project Overview

GlowCart is a cosmetics/beauty review website we built for this assignment. Users
can browse products, search by brand or keyword, read reviews, and write their own.
When someone submits a review, the app runs it through a 3-model ML ensemble (from
Milestone I) to predict if the reviewer would buy the product. The prediction shows
up on screen and the user can accept or override it before confirming.

For our Task 4, we picked **VADER sentiment analysis** — it scores every review as
positive/neutral/negative and we show the breakdown on the dashboard. More on why
we chose VADER below.

---

## 2. How the Model Was Trained (Milestone I)

### 2.1 Text Preprocessing (Task 1)

- Tokenised 61,284 reviews using regex: `[a-zA-Z]+(?:[-'][a-zA-Z]+)?`
- Lowercased all tokens
- Removed tokens shorter than 2 characters
- Removed English stopwords (`stopwords_en.txt`, 570 words)
- Removed hapax legomena (term frequency = 1 across the corpus)
- Removed the top-20 most frequent words by document frequency
- **Final vocabulary:** 8,054 unique terms

### 2.2 Feature Representations (Task 2)

We built three document representations from the cleaned text:

| Representation | Description |
|----------------|-------------|
| **Bag-of-Words (BoW)** | Sparse count vectors over the 8,054-word vocabulary. |
| **Unweighted FastText Embeddings** | Sum of 300-dim vectors from `fasttext-wiki-news-subwords-300` for each token. |
| **TF-IDF Weighted FastText Embeddings** | Same FastText vectors, but each token's vector is scaled by its TF-IDF score (gensim `TfidfModel`). |

### 2.3 Classification Experiments (Task 3)

**Q1 — Language Model Comparison (review text only):**

We tried 3 classifiers (`LogisticRegression`, `LinearSVC`, `RandomForest`)
on all 3 representations with 5-fold stratified cross-validation.

- Best result: **BoW + LinearSVC**, macro-F1 = **0.5745**

**Q2(b) — Adding Review Title:**

We concatenated cleaned `review_title` tokens with `review_text` and rebuilt
all representations. It only gave a small bump (~+0.01 F1), so not a big deal.

**Q2(c) — Adding Structural Features:**

We added product metadata on top of the text features:

- **Numeric:** `price`, `avg_product_rating`, `product_rating_count`
- **Categorical:** `brand_name`, `product_title`
- Best result: **BoW + structural features**, macro-F1 = **0.7126**

For the web app we export **3 models that use different data types**: Models 1
and 2 are text-only (review_text + review_title), while Model 3 adds the
structural metadata. This satisfies the "different data types" requirement
and lets the ensemble combine a pure-text signal with a text+structured signal.

### 2.4 Why We Included `product_title` as a Feature

Just looking at review text wasn't enough. Whether someone buys a product
depends a lot on *what* the product is — a great review on a cheap drugstore
mascara leads to a different buying outcome than the same review on a $60
prestige serum. People factor in price, brand, and product type when deciding
to buy, not just what the reviewer wrote.

One-hot encoding `product_title` (with `min_frequency=20` so it doesn't
explode) lets the model pick up product-level patterns. For example, some
products get great reviews but people don't actually buy them (luxury items
that get window-shopped), while others sell well no matter what (everyday
essentials people keep repurchasing).

Adding it alongside `brand_name`, `price`, and ratings pushed macro-F1 from
0.5745 to **0.7126** (+0.14), so it clearly makes a difference.

> We left out `review_rating` on purpose — it's too correlated with
> `is_a_buyer` and would cause label leakage.

### 2.5 How the 3 Models Compare

The three models deliberately use **different data types**:

- **Model 1 (BoW)** — text only. Sparse word counts from review_text +
  review_title. Word counts work well here because beauty reviews use very
  predictable vocabulary — words like "love", "repurchase", "broke out"
  carry strong signal.
- **Model 2 (Unweighted FastText)** — text only (different representation).
  Sum of 300-dim FastText vectors from review_text + review_title. Captures
  semantic meaning but loses some word-count signal compared to BoW.
- **Model 3 (TF-IDF Weighted FastText + metadata)** — text + structured.
  TF-IDF weighted FastText vectors plus product metadata (price, brand,
  avg_product_rating, product_rating_count, product_title). The weakest on
  text alone, but the metadata gives it a different angle — it can factor in
  *what* the product is, not just what the reviewer wrote.

The **ensemble of all three** beats any single model. Models 1 and 2 provide
two independent text-only views, while Model 3 adds structured context. Each
model sees the data from a different angle, and averaging their probabilities
smooths out individual weaknesses.

---

## 3. Application Architecture & Flow

### 3.1 Tech Stack

| Layer    | Technologies |
|----------|-------------|
| **Backend**  | Python 3.13, Flask, scikit-learn, gensim, pandas, numpy, vaderSentiment |
| **Frontend** | React 19, React Router 7, Vite, Tailwind CSS 4, Framer Motion |
| **Deploy**   | Docker (multi-stage), Gunicorn |

### 3.2 Project Structure

```
milestone2/
├── core/                        # Flask backend
│   ├── app.py                   # API routes + static file serving
│   ├── data.py                  # Data loading, search, similarity, sentiment
│   └── util.py                  # ML prediction pipeline
├── web/                         # React frontend
│   └── src/
│       ├── pages/               # Landing, Home, Search, ProductDetail, Dashboard
│       └── components/          # ProductCard, Navbar
├── model/                       # Trained ML artifacts (~10 MB)
│   ├── clf_bow.joblib           # BoW LogReg (text only)
│   ├── pipe_unw.joblib          # Unweighted embeddings pipeline (text only)
│   ├── pipe_w.joblib            # TF-IDF weighted embeddings + metadata pipeline
│   ├── preproc_struct.joblib    # ColumnTransformer for metadata (Model 3 only)
│   ├── ft_vectors.npy           # FastText vectors (vocab subset)
│   ├── tfidf.dict               # Gensim dictionary
│   ├── tfidf.model              # Gensim TF-IDF model
│   ├── vocab.txt                # 8,054-word vocabulary
│   ├── stopwords_en.txt         # English stopwords
│   └── threshold.json           # Tuned decision threshold
├── data/
│   └── processed_with_images.csv  # Product + review data with images
└── Dockerfile                   # Multi-stage production build
```

### 3.3 Application Flow

#### Landing Page (`/`)

This is the first thing users see — a hero banner, some quick stats (total
products/reviews/brands), a carousel of top-rated products, and a grid of
brands they can click to jump straight to that brand's products. There's also
a "Why GlowCart" section highlighting the ML features.

#### Browsing (`/products`)

The main product page. Shows a grid of product cards with filters on top —
users can narrow down by brand, price range (slider), and minimum star rating.
There's also sorting (name, rating, price, review count) and pagination
(20 per page). Clicking a brand on the landing page pre-fills the brand filter.

#### Search (Task 1)

There's a search bar in the navbar that works from any page. It hits
`GET /api/search?q=...` and the backend does two passes:

1. **Exact match:** checks if all query words appear in `brand_name + product_title`
2. **Fuzzy match:** uses `SequenceMatcher` with a 0.6 threshold, so typos
   like "Maybeline" still find "Maybelline"

Results are ranked by match score and shown with a count.

#### Review Creation + Prediction (Task 2)

On a product page, there's a review form at the bottom (title, text, rating,
name). When submitted, the backend:

1. Cleans the text (tokenise, lowercase, remove stopwords + short tokens)
2. Runs Models 1 & 2 on the text features only (BoW and unweighted embeddings)
3. Grabs the product metadata and runs Model 3 (TF-IDF weighted embeddings + metadata)
4. Averages the 3 probabilities and applies the threshold — above it means "Likely Buyer"

The frontend then shows the prediction with per-model confidence bars. If
the user disagrees with the label, they can override it. After confirming,
the review shows up on the page immediately.

#### Similar Items (Task 3)

Every product page has a "You May Also Like" carousel with the 6 most similar
products. We precompute this at startup:

- Join all reviews per product into one big text blob
- Run `TfidfVectorizer` on all product documents
- Compute pairwise cosine similarity

So similarity is based on what reviewers talk about, not just product metadata.

#### Sentiment Analysis Dashboard (Task 4 — Additional Functionality)

We picked **VADER** for sentiment analysis because it's designed for short,
informal text like product reviews — it handles slang, emoticons, and
intensifiers (e.g., "VERY good", "good!!!" score higher than just "good")
out of the box. We considered TextBlob but VADER is more suited to
review-style writing and doesn't need any training data.

At startup, every review gets a VADER compound score and a label:
- **Positive:** compound >= 0.05
- **Negative:** compound <= -0.05
- **Neutral:** everything in between

**Why this is useful for a shopping site:** store owners can quickly see which
brands or products have negative sentiment trends, spot recurring complaints,
and identify which products customers feel strongly about (positive or
negative). It turns raw review text into actionable data at a glance.

The dashboard shows:
- A sentiment gauge with the overall average score (-100 to +100 scale)
- Breakdown cards (positive / neutral / negative counts) with a stacked bar
- Sentiment per brand — avg score, counts, and mini bar charts per row
- Top-rated products carousel
- Rating distribution bars, brand breakdown table, top reviewed products

Each review on product pages also gets a small sentiment badge next to the
buyer status badge, so users can see at a glance whether a review is positive
or negative.

#### Other Features

- Landing page with hero, brand grid, and feature cards
- Page transitions and hover animations (Framer Motion)
- Filtering by brand, price range, and minimum rating
- Sorting by name, rating, price, review count
- Pagination (20 products/page, 5 reviews/page)
- Dark mode (saved in `localStorage`)
- Responsive layout (works on mobile, tablet, desktop)
- Product images with brand-coloured gradient overlays

---

## 4. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/brands` | List all brand names |
| `GET`  | `/api/products` | Paginated product listing (`?page=`, `&per_page=`, `&sort_by=`, `&brand=`, `&min_price=`, `&max_price=`, `&min_rating=`) |
| `GET`  | `/api/products/<id>` | Single product with reviews and sentiment |
| `GET`  | `/api/products/<id>/similar` | Top-6 similar products |
| `GET`  | `/api/search?q=<query>` | Search products by keyword |
| `GET`  | `/api/reviews/<id>` | Single review by ID |
| `POST` | `/api/reviews` | Create review + get ML prediction (body: `product_id`, `review_title`, `review_text`, `review_rating`, `author`) |
| `GET`  | `/api/stats` | Dashboard stats and sentiment breakdown |
| `GET`  | `/api/top-rated?n=10` | Top N highest-rated products |

---

## 5. Dependencies

### Backend (Python)

`flask`, `flask-cors`, `numpy`, `scikit-learn==1.7.2`, `pandas`, `scipy`,
`gensim`, `joblib`, `gunicorn`, `vaderSentiment`

### Frontend (Node)

`react`, `react-dom`, `react-router-dom`, `rc-slider`, `framer-motion`,
`tailwindcss`, `vite`, `eslint`

---

## 6. Notes

- User-submitted reviews are stored in memory and reset on server restart.
  That's fine for a demo.
- Model files are about 10 MB total, in the `model/` directory.
- The dataset (`processed_with_images.csv`, ~23 MB) is in `data/`.
- Everything needed to run the app is included in the submission.
