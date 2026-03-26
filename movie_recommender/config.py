from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    tmdb_api_key: str | None
    tmdb_base_url: str
    tmdb_image_base_url: str
    retriever_k: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        tmdb_api_key=os.getenv("TMDB_API_KEY"),
        tmdb_base_url=os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3"),
        tmdb_image_base_url=os.getenv("TMDB_IMAGE_BASE_URL", "https://image.tmdb.org/t/p/w500"),
        retriever_k=int(os.getenv("RETRIEVER_K", "5")),
    )
