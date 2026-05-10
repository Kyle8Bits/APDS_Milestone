import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import data
import util

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)


@app.route("/api/brands", methods=["GET"])
def list_brands():
    return jsonify(data.get_brands())


@app.route("/api/products", methods=["GET"])
def list_products():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    sort_by = request.args.get("sort_by", "name")
    brand = request.args.get("brand", None)
    min_price = request.args.get("min_price", None, type=int)
    max_price = request.args.get("max_price", None, type=int)
    min_rating = request.args.get("min_rating", None, type=float)
    return jsonify(data.get_products(
        page, per_page, sort_by=sort_by, brand=brand,
        min_price=min_price, max_price=max_price, min_rating=min_rating,
    ))


@app.route("/api/products/<int:product_id>", methods=["GET"])
def get_product(product_id):
    product = data.get_product(product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404
    return jsonify(product)


@app.route("/api/products/<int:product_id>/similar", methods=["GET"])
def get_similar(product_id):
    if data.get_product(product_id) is None:
        return jsonify({"error": "Product not found"}), 404
    return jsonify(data.get_similar(product_id))


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    return jsonify(data.search_products(query))


@app.route("/api/reviews/<int:review_id>", methods=["GET"])
def get_review(review_id):
    review = data.get_review(review_id)
    if review is None:
        return jsonify({"error": "Review not found"}), 404
    return jsonify(review)


@app.route("/api/reviews", methods=["POST"])
def create_review():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    product_id = body.get("product_id")
    review_title = body.get("review_title", "")
    review_text = body.get("review_text", "")
    review_rating = body.get("review_rating")
    author = body.get("author", "Anonymous")

    if product_id is None or review_rating is None:
        return jsonify({"error": "product_id and review_rating are required"}), 400

    product = data.get_product(product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404

    prediction = util.predict(
        review_text=review_text,
        review_title=review_title,
        brand_name=product["brand_name"],
        product_title=product["product_title"],
        price=product["price"],
        avg_product_rating=product["avg_product_rating"],
        product_rating_count=product["product_rating_count"],
    )

    is_a_buyer = prediction["label"] == "Likely Buyer"

    review = data.add_review(
        product_id=product_id,
        review_title=review_title,
        review_text=review_text,
        review_rating=review_rating,
        is_a_buyer=is_a_buyer,
        author=author,
    )

    return jsonify({"review": review, "prediction": prediction}), 201


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(data.get_stats())


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path and os.path.isfile(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
