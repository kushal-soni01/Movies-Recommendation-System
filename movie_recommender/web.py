from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from movie_recommender.recommender import get_featured_movies, recommend_movies


GENRES = ["All", "Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["SECRET_KEY"] = "movie-recommendation-dev"

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            genres=GENRES,
            recommendations=get_featured_movies(),
            query="",
            selected_genre="All",
            is_featured_view=True,
        )

    @app.post("/recommend")
    def recommend():
        query = request.form.get("query", "").strip()
        genre = request.form.get("genre", "All")

        if not query:
            return render_template(
                "index.html",
                genres=GENRES,
                recommendations=get_featured_movies(),
                query=query,
                selected_genre=genre,
                is_featured_view=True,
                error="Please enter a movie, actor, genre, or mood.",
            )

        try:
            recommendations = recommend_movies(query=query, genre=genre)
        except Exception as exc:  # pragma: no cover - defensive Flask boundary
            return render_template(
                "index.html",
                genres=GENRES,
                recommendations=[],
                query=query,
                selected_genre=genre,
                is_featured_view=False,
                error=f"Unable to generate recommendations right now: {exc}",
            )

        return render_template(
            "index.html",
            genres=GENRES,
            recommendations=recommendations,
            query=query,
            selected_genre=genre,
            is_featured_view=False,
        )

    @app.post("/api/recommend")
    def recommend_api():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", "")).strip()
        genre = str(payload.get("genre", "All")).strip() or "All"

        if not query:
            return jsonify({"error": "Query is required."}), 400

        try:
            recommendations = recommend_movies(query=query, genre=genre)
        except Exception as exc:  # pragma: no cover - defensive Flask boundary
            return jsonify({"error": str(exc)}), 500

        return jsonify({"query": query, "genre": genre, "results": recommendations})

    return app
