from __future__ import annotations

from dataclasses import dataclass

import requests

from movie_recommender.config import Settings


@dataclass(frozen=True)
class MovieDetails:
    title: str
    poster_url: str | None
    rating: float | None


class TMDBClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = requests.Session()

    def fetch_movie_details(self, movie_name: str) -> MovieDetails:
        if not self._settings.tmdb_api_key:
            return MovieDetails(movie_name, None, None)

        url = f"{self._settings.tmdb_base_url}/search/movie"
        params = {
            "api_key": self._settings.tmdb_api_key,
            "query": movie_name,
        }

        for _ in range(2):
            try:
                response = self._session.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException:
                data = {}
        else:
            data = {}

        results = data.get("results") or []
        if not results:
            return MovieDetails(movie_name, None, None)

        movie = next((item for item in results if item.get("poster_path")), results[0])
        poster_path = movie.get("poster_path")
        poster_url = (
            f"{self._settings.tmdb_image_base_url}/{poster_path.lstrip('/')}"
            if poster_path
            else None
        )

        return MovieDetails(
            title=movie.get("title") or movie_name,
            poster_url=poster_url,
            rating=round(float(movie.get("vote_average", 0.0)), 1) if movie.get("vote_average") is not None else None,
        )
