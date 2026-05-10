================================================================================
  GlowCart - Cosmetics & Beauty Product Review Website
  COSC3801/3015 Advanced Programming for Data Science
  Assignment 3 - Milestone II: Web-based Data Application
================================================================================

TEAM MEMBERS
------------
  Name: Mai Dang Khoa          Student ID: s3974876
  Name: Dang Cuu Dang Khoa     Student ID: s3979159
  Name: Tran Quang Minh        Student ID: s3988776


================================================================================
1. PROJECT OVERVIEW
================================================================================

GlowCart is a cosmetics/beauty product review website that lets shoppers browse
products, read reviews, and submit new reviews. When a review is submitted, the
system uses a machine-learning ensemble (trained in Milestone I) to predict
whether the reviewer is likely to purchase the product. The prediction is shown
to the reviewer, who can accept or override it before confirming.


================================================================================
2. HOW THE MODEL WAS TRAINED (Milestone I)
================================================================================

2.1  Text Preprocessing (Task 1)
    - Tokenised 61,284 reviews using regex: [a-zA-Z]+(?:[-'][a-zA-Z]+)?
    - Lowercased all tokens
    - Removed tokens shorter than 2 characters
    - Removed English stopwords (stopwords_en.txt, 570 words)
    - Removed hapax legomena (term frequency = 1 across corpus)
    - Removed top-20 most frequent words by document frequency
    - Final vocabulary: 8,054 unique terms

2.2  Feature Representations (Task 2)
    Three document representations were generated from the cleaned review text:

    (a) Bag-of-Words (BoW): Sparse count vectors over the 8,054-word vocabulary.
    (b) Unweighted FastText Embeddings: Sum of 300-dim vectors from the
        pretrained fasttext-wiki-news-subwords-300 model for each token.
    (c) TF-IDF Weighted FastText Embeddings: Same FastText vectors but weighted
        by each token's TF-IDF score (computed via gensim TfidfModel).

2.3  Classification Experiments (Task 3)

    Q1 - Language Model Comparison (review text only):
      Evaluated 3 classifiers (LogisticRegression, LinearSVC, RandomForest)
      across all 3 representations using 5-fold stratified cross-validation.
      Best result: BoW + LinearSVC, macro-F1 = 0.5745

    Q2(b) - Adding Review Title:
      Concatenated cleaned review_title tokens with review_text tokens and
      rebuilt all representations. Marginal improvement (~+0.01 F1).

    Q2(c) - Adding Structural Features:
      Added product metadata alongside text features:
        - Numeric: price, avg_product_rating, product_rating_count
        - Categorical: brand_name, product_title
      Best result: BoW + structural features, macro-F1 = 0.7126

    The Q2(c) configuration was selected for deployment because it achieved the
    highest macro-F1 and more than doubled non-buyer recall (0.196 -> 0.471).

2.4  Why product_title Is Included as a Feature

    In real-world e-commerce, a customer's purchasing decision is not based on
    the review text alone. The specific product matters: the same positive review
    written about a $5 drugstore mascara and a $60 prestige serum will produce
    different buying outcomes because customers weigh price, brand reputation,
    and product category when deciding to purchase.

    Including product_title lets the model learn product-level buying patterns.
    For example, some products have high satisfaction but low conversion (luxury
    items where customers browse but rarely buy), while others convert at high
    rates regardless of review sentiment (essentials, repurchases). One-hot
    encoding with min_frequency=20 collapses rare products into an "infrequent"
    bucket, keeping dimensionality manageable while preserving signal for
    popular products with enough data to learn from.

    Empirically, adding product_title (along with brand_name, price, and
    ratings) boosted macro-F1 from 0.5745 to 0.7126 — a +0.14 improvement.
    This confirms that the model benefits from knowing which product is being
    reviewed, not just what the reviewer wrote.

    Note: review_rating was deliberately excluded to avoid label leakage, since
    rating is strongly correlated with the is_a_buyer target variable.


================================================================================
3. APPLICATION ARCHITECTURE & FLOW
================================================================================

3.1  Tech Stack
    Backend:  Python 3.13, Flask, scikit-learn, gensim, pandas, numpy
    Frontend: React 19, React Router 7, Vite, Tailwind CSS 4
    Deploy:   Docker (multi-stage), Gunicorn

3.2  Project Structure

    milestone2/
    ├── core/                   # Flask backend
    │   ├── app.py              # API routes + static file serving
    │   ├── data.py             # Data loading, search, similarity
    │   └── util.py             # ML prediction pipeline
    ├── web/                    # React frontend
    │   └── src/
    │       ├── pages/          # Home, Search, ProductDetail, Dashboard
    │       └── components/     # ProductCard, Navbar
    ├── model/                  # Trained ML artifacts (~10 MB)
    │   ├── clf_bow.joblib      # BoW LogisticRegression classifier
    │   ├── pipe_unw.joblib     # Unweighted embeddings pipeline
    │   ├── pipe_w.joblib       # TF-IDF weighted embeddings pipeline
    │   ├── preproc_struct.joblib  # ColumnTransformer for metadata
    │   ├── ft_vectors.npy      # FastText vectors (vocab subset)
    │   ├── tfidf.dict          # Gensim dictionary
    │   ├── tfidf.model         # Gensim TF-IDF model
    │   ├── vocab.txt           # 8,054-word vocabulary
    │   ├── stopwords_en.txt    # English stopwords
    │   └── threshold.json      # Decision threshold (0.3)
    ├── data/
    │   └── processed_with_images.csv   # Product + review data with images
    └── Dockerfile              # Multi-stage production build

3.3  Application Flow

    BROWSING:
      User opens the home page -> Flask serves the React SPA -> React calls
      GET /api/products with filters/pagination -> Products displayed as cards
      -> User clicks a card -> GET /api/products/<id> loads full product detail
      with all reviews.

    SEARCH (Task 1):
      User types a keyword in the navbar search bar -> GET /api/search?q=...
      -> Backend performs two-pass matching:
        Pass 1: Exact substring match on brand_name + product_title (score 1.0)
        Pass 2: Fuzzy matching via SequenceMatcher (threshold 0.6) for typo
                tolerance (e.g. "Maybeline" matches "Maybelline")
      -> Results ranked by match score and displayed with result count.

    REVIEW CREATION + PREDICTION (Task 2):
      User fills in review form (title, text, rating, author) on a product
      page -> POST /api/reviews -> Backend runs the prediction pipeline:

        1. Clean review_text + review_title (tokenise, lowercase, remove
           stopwords, remove short tokens)
        2. Look up product metadata (brand, title, price, rating, review count)
        3. Run 3 models in parallel:
           - BoW classifier: count vector + structural features
           - Unweighted embedding: FastText sum + structural features
           - Weighted embedding: TF-IDF weighted FastText + structural features
        4. Average the 3 probabilities -> fused prediction
        5. Apply threshold (0.3): >= threshold = "Likely Buyer"

      -> Frontend shows the prediction panel with fused probability, individual
         model confidence bars, and the predicted label.
      -> User can override the label if they disagree.
      -> On confirm, the review is saved and immediately visible on the product
         page.

    SIMILAR ITEMS (Task 3):
      On every product detail page, GET /api/products/<id>/similar returns
      the top 6 most similar products. Similarity is precomputed at startup:
        - Aggregate all review_text per product into a single document
        - Fit sklearn TfidfVectorizer on all product documents
        - Compute pairwise cosine similarity matrix
      -> "You May Also Like" section displays similar product cards.

    DASHBOARD (Task 4 - Additional Functionality):
      GET /api/stats returns aggregate analytics:
        - Total products, reviews, brands, buyer rate
        - Rating distribution (1-5 stars)
        - Reviews by brand with buyer % and average rating
        - Top 5 most-reviewed products
      Displayed as summary cards, bar charts, and data tables.

    ADDITIONAL FEATURES:
      - Product filtering by brand, price range, minimum rating
      - Sorting by name, rating, price, review count
      - Pagination (20 products/page, 5 reviews/page)
      - Dark mode (persisted in localStorage)
      - Responsive layout (mobile/tablet/desktop)
      - Product images with brand-coloured gradient overlay


================================================================================
4. HOW TO RUN LOCALLY
================================================================================

Prerequisites:
  - Python 3.11+ (tested on 3.13)
  - Node.js 18+ (tested on 22)

Step 1: Install backend dependencies
  cd milestone2/core
  pip install -r requirements.txt

Step 2: Start the Flask backend
  python app.py
  # Runs on http://localhost:8000

Step 3: Install frontend dependencies (separate terminal)
  cd milestone2/web
  npm install

Step 4: Start the React dev server
  npm run dev
  # Runs on http://localhost:5173, proxies API calls to :8000

Open http://localhost:5173 in your browser.


================================================================================
5. HOW TO RUN WITH DOCKER
================================================================================

The Dockerfile uses a two-stage build:

  Stage 1 (node:22-alpine):
    - Installs npm dependencies (npm ci for reproducible builds)
    - Builds the React frontend (npm run build -> dist/)

  Stage 2 (python:3.13-slim):
    - Installs system build tools (gcc, g++) for native Python packages
    - Installs Python dependencies from requirements.txt + gunicorn
    - Copies Flask source code, model artifacts, and data CSV
    - Copies the built frontend from Stage 1 into core/static/
    - Flask serves both the API (/api/*) and the SPA (all other routes)
    - Gunicorn runs with 2 workers and 120s timeout on port 8000

Build and run:
  cd milestone2
  docker build -t glowcart .
  docker run -p 8000:8000 glowcart

Open http://localhost:8000 in your browser.

To deploy on Railway or similar platforms:
  - Set root directory to milestone2/
  - The PORT environment variable is respected automatically
  - No additional configuration needed


================================================================================
6. API ENDPOINTS
================================================================================

  GET  /api/brands                     List all brand names
  GET  /api/products                   Paginated product listing
       ?page=1&per_page=20&sort_by=name&brand=...&min_price=...&max_price=...&min_rating=...
  GET  /api/products/<id>              Single product with reviews
  GET  /api/products/<id>/similar      Top-6 similar products
  GET  /api/search?q=<query>           Search products by keyword
  GET  /api/reviews/<id>               Single review by ID
  POST /api/reviews                    Create review + get ML prediction
       Body: {product_id, review_title, review_text, review_rating, author}
  GET  /api/stats                      Dashboard statistics


================================================================================
7. DEPENDENCIES
================================================================================

Backend (Python):
  flask, flask-cors, numpy, scikit-learn==1.7.2, pandas, scipy,
  gensim, joblib, gunicorn

Frontend (Node):
  react, react-dom, react-router-dom, rc-slider,
  tailwindcss, vite, eslint


================================================================================
8. NOTES
================================================================================

  - User-submitted reviews are stored in memory and will be lost on server
    restart. This is by design for a demo/assignment application.
  - Model files total ~10 MB and are included in the model/ directory.
  - The dataset (processed_with_images.csv, ~23 MB) is in data/.
  - All files needed to run the application are included in this submission.
