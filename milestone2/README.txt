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

GlowCart is a cosmetics/beauty product review website built for this assignment.
Users can browse products, search by brand or keyword, read reviews, and write
their own. When someone submits a review, the app runs it through a 3-model
ML ensemble (from Milestone I) to predict if the reviewer would buy the product.
The user sees this prediction and can accept or change it before confirming.

We also added a sentiment analysis feature (VADER) as our Task 4, which scores
every review as positive/neutral/negative and shows the results on the dashboard.


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
    We built three different document representations from the cleaned text:

    (a) Bag-of-Words (BoW): Sparse count vectors over the 8,054-word vocabulary.
    (b) Unweighted FastText Embeddings: Sum of 300-dim vectors from the
        pretrained fasttext-wiki-news-subwords-300 model for each token.
    (c) TF-IDF Weighted FastText Embeddings: Same FastText vectors but each
        token's vector is scaled by its TF-IDF score (gensim TfidfModel).

2.3  Classification Experiments (Task 3)

    Q1 - Language Model Comparison (review text only):
      We tested 3 classifiers (LogisticRegression, LinearSVC, RandomForest)
      on all 3 representations using 5-fold stratified cross-validation.
      Best result: BoW + LinearSVC, macro-F1 = 0.5745

    Q2(b) - Adding Review Title:
      Concatenated cleaned review_title tokens with review_text tokens and
      rebuilt all representations. Only gave a small bump (~+0.01 F1).

    Q2(c) - Adding Structural Features:
      We added product metadata on top of the text features:
        - Numeric: price, avg_product_rating, product_rating_count
        - Categorical: brand_name, product_title
      Best result: BoW + structural features, macro-F1 = 0.7126

    We went with the Q2(c) setup for the web app since it gave the best F1
    and more than doubled non-buyer recall (0.196 -> 0.471).

2.4  Why We Included product_title as a Feature

    We noticed that just looking at the review text wasn't enough. Whether
    someone actually buys a product depends a lot on what the product is -
    a glowing review on a cheap drugstore mascara converts differently than
    the same review on an expensive prestige serum. People weigh price, brand
    reputation, and product type when they decide to purchase, not just what
    the review says.

    By one-hot encoding product_title (with min_frequency=20 to keep things
    manageable), the model can pick up on product-level patterns. Some products
    get great reviews but low conversion (luxury items people window-shop),
    while others sell well no matter what (everyday essentials, repurchases).

    Adding product_title along with brand_name, price, and ratings pushed
    macro-F1 from 0.5745 to 0.7126 (+0.14), so it clearly helps.

    We left out review_rating on purpose to avoid label leakage - rating is
    too correlated with is_a_buyer.


================================================================================
3. APPLICATION ARCHITECTURE & FLOW
================================================================================

3.1  Tech Stack
    Backend:  Python 3.13, Flask, scikit-learn, gensim, pandas, numpy,
              vaderSentiment (for sentiment analysis)
    Frontend: React 19, React Router 7, Vite, Tailwind CSS 4,
              Framer Motion (animations)
    Deploy:   Docker (multi-stage), Gunicorn

3.2  Project Structure

    milestone2/
    ├── core/                   # Flask backend
    │   ├── app.py              # API routes + static file serving
    │   ├── data.py             # Data loading, search, similarity, sentiment
    │   └── util.py             # ML prediction pipeline
    ├── web/                    # React frontend
    │   └── src/
    │       ├── pages/          # Landing, Home, Search, ProductDetail, Dashboard
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

    LANDING PAGE (/):
      The home page is a landing page that shows the site branding, stats
      (total products/reviews/brands), a carousel of top-rated products,
      a "Shop by Brand" section, and feature highlights. Clicking "Shop Now"
      or a brand takes users to the product browse page.

    BROWSING (/products):
      The product page shows a paginated grid of all products. Users can
      filter by brand, price range, and minimum rating, and sort by name,
      rating, price, or review count. Clicking a product card opens the
      detail page.

    SEARCH (Task 1):
      There's a search bar in the navbar. When users type a keyword, it hits
      GET /api/search?q=... and the backend does two-pass matching:
        Pass 1: exact substring check on brand_name + product_title (score 1.0)
        Pass 2: fuzzy match via SequenceMatcher (threshold 0.6) so typos
                like "Maybeline" still find "Maybelline"
      Results are ranked by score and shown with a count.

    REVIEW CREATION + PREDICTION (Task 2):
      On a product page, users fill in a review form (title, text, rating,
      author name) and submit it. The backend then:

        1. Cleans the review text + title (tokenise, lowercase, remove
           stopwords and short tokens)
        2. Looks up the product metadata (brand, title, price, etc.)
        3. Runs 3 models:
           - BoW: count vector + structural features
           - Unweighted: FastText vector sum + structural features
           - Weighted: TF-IDF weighted FastText + structural features
        4. Averages the 3 probabilities for a fused prediction
        5. Applies threshold (0.3): above = "Likely Buyer"

      The frontend shows the prediction with confidence bars for each model.
      Users can override the label if they disagree, then confirm. The review
      then shows up on the product page right away.

    SIMILAR ITEMS (Task 3):
      Every product page has a "You May Also Like" section with the 6 most
      similar products. We precompute this at startup by:
        - Joining all reviews per product into one document
        - Running TfidfVectorizer on all product documents
        - Computing pairwise cosine similarity

    SENTIMENT ANALYSIS DASHBOARD (Task 4 - Additional Functionality):
      We added VADER sentiment analysis as our extra feature. At startup, every
      review gets scored with VADER's compound score and labelled as positive
      (>= 0.05), negative (<= -0.05), or neutral.

      The dashboard (GET /api/stats) shows:
        - A sentiment gauge with the overall average score
        - Distribution breakdown (positive/neutral/negative counts + bar)
        - Sentiment per brand with avg scores and mini bar charts
        - Top-rated products carousel
        - Rating distribution, brand breakdown, top reviewed products

      Each review on product pages also shows a sentiment badge next to the
      buyer status badge.

    OTHER FEATURES:
      - Landing page with hero section, brand grid, and feature cards
      - Animated page transitions and UI interactions (Framer Motion)
      - Product filtering by brand, price range, minimum rating
      - Sorting by name, rating, price, review count
      - Pagination (20 products/page, 5 reviews/page)
      - Dark mode toggle (saved in localStorage)
      - Responsive layout for mobile/tablet/desktop
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
    - Installs npm dependencies (npm ci)
    - Builds the React frontend (npm run build -> dist/)

  Stage 2 (python:3.13-slim):
    - Installs gcc/g++ for native Python packages
    - Installs Python deps from requirements.txt + gunicorn
    - Copies Flask code, model files, and data CSV
    - Copies built frontend from Stage 1 into core/static/
    - Gunicorn runs with 2 workers on port 8000

Build and run:
  cd milestone2
  docker build -t glowcart .
  docker run -p 8000:8000 glowcart

Open http://localhost:8000 in your browser.

For Railway or similar platforms:
  - Set root directory to milestone2/
  - PORT env var is picked up automatically


================================================================================
6. API ENDPOINTS
================================================================================

  GET  /api/brands                     List all brand names
  GET  /api/products                   Paginated product listing
       ?page=1&per_page=20&sort_by=name&brand=...&min_price=...&max_price=...&min_rating=...
  GET  /api/products/<id>              Single product with reviews + sentiment
  GET  /api/products/<id>/similar      Top-6 similar products
  GET  /api/search?q=<query>           Search products by keyword
  GET  /api/reviews/<id>               Single review by ID
  POST /api/reviews                    Create review + get ML prediction
       Body: {product_id, review_title, review_text, review_rating, author}
  GET  /api/stats                      Dashboard stats + sentiment breakdown
  GET  /api/top-rated?n=10             Top N highest-rated products


================================================================================
7. DEPENDENCIES
================================================================================

Backend (Python):
  flask, flask-cors, numpy, scikit-learn==1.7.2, pandas, scipy,
  gensim, joblib, gunicorn, vaderSentiment

Frontend (Node):
  react, react-dom, react-router-dom, rc-slider, framer-motion,
  tailwindcss, vite, eslint


================================================================================
8. NOTES
================================================================================

  - User-submitted reviews are stored in memory and reset on server restart.
    This is fine for a demo app.
  - Model files total ~10 MB and are in the model/ directory.
  - The dataset (processed_with_images.csv, ~23 MB) is in data/.
  - All files needed to run the app are included in this submission.
