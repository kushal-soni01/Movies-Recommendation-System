from __future__ import annotations

from dataclasses import dataclass

import requests

from movie_recommender.config import Settings


@dataclass(frozen=True)
class MovieDetails:
    title: str
    poster_url: str | None
    rating: float | None
    trailer_url: str | None


class TMDBClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = requests.Session()

    def _request_json(self, path: str, params: dict[str, object]) -> dict:
        if not self._settings.tmdb_api_key:
            return {}

        url = f"{self._settings.tmdb_base_url}{path}"
        request_params = {
            "api_key": self._settings.tmdb_api_key,
            **params,
        }

        for _ in range(2):
            try:
                response = self._session.get(url, params=request_params, timeout=5)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                continue

        return {}

    def _pick_best_result(self, results: list[dict]) -> dict | None:
        if not results:
            return None

        def score(item: dict) -> tuple[int, float, float]:
            has_poster = 1 if item.get("poster_path") else 0
            popularity = float(item.get("popularity") or 0.0)
            votes = float(item.get("vote_count") or 0.0)
            return (has_poster, popularity, votes)

        return max(results, key=score)

    def _video_priority(self, video: dict) -> tuple[int, int]:
        type_order = {
            "Trailer": 4,
            "Teaser": 3,
            "Clip": 2,
            "Featurette": 1,
        }
        site_bonus = 1 if video.get("site") == "YouTube" else 0
        return (site_bonus, type_order.get(video.get("type"), 0))

    def _fetch_trailer_url(self, media_type: str, media_id: int | str | None) -> str | None:
        if not media_id or media_type not in {"movie", "tv"}:
            return None

        data = self._request_json(f"/{media_type}/{media_id}/videos", {})
        videos = data.get("results") or []
        youtube_videos = [video for video in videos if video.get("site") == "YouTube" and video.get("key")]
        if not youtube_videos:
            return None

        best_video = max(youtube_videos, key=self._video_priority)
        return f"https://www.youtube.com/watch?v={best_video['key']}"

    def fetch_movie_details(self, movie_name: str) -> MovieDetails:
        if not self._settings.tmdb_api_key:
            return MovieDetails(movie_name, None, None, None)

        data = self._request_json("/search/multi", {"query": movie_name})
        results = [
            item
            for item in (data.get("results") or [])
            if item.get("media_type") in {"movie", "tv"}
        ]
        match = self._pick_best_result(results)
        if not match:
            return MovieDetails(movie_name, None, None, None)

        poster_path = match.get("poster_path")
        poster_url = (
            f"{self._settings.tmdb_image_base_url}/{poster_path.lstrip('/')}"
            if poster_path
            else None
        )
        title = match.get("title") or match.get("name") or movie_name
        trailer_url = self._fetch_trailer_url(match.get("media_type"), match.get("id"))

        return MovieDetails(
            title=title,
            poster_url=poster_url,
            rating=round(float(match.get("vote_average", 0.0)), 1) if match.get("vote_average") is not None else None,
            trailer_url=trailer_url,
        )
