from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


DROP_COLUMNS = [
    "tagline",
    "homepage",
    "status",
    "production_companies",
    "spoken_languages",
    "budget",
    "id",
    "crew",
]


def _extract_names(raw_value: str) -> list[str]:
    try:
        return [item["name"] for item in ast.literal_eval(raw_value)]
    except (ValueError, SyntaxError, TypeError, KeyError):
        return []


def _get_top_cast(raw_value: str, limit: int = 3) -> list[str]:
    try:
        cast_list = ast.literal_eval(raw_value)
        return [item["name"] for item in cast_list[:limit]]
    except (ValueError, SyntaxError, TypeError, KeyError):
        return []


def _get_director(raw_value: str) -> str | None:
    try:
        for item in ast.literal_eval(raw_value):
            if item.get("job") == "Director":
                return item.get("name")
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return None
    return None


def load_movies_frame(data_dir: Path) -> pd.DataFrame:
    movies_path = data_dir / "tmdb_5000_movies.csv"
    credits_path = data_dir / "tmdb_5000_credits.csv"

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    movies = movies.merge(credits, on="title")

    movies["genres"] = movies["genres"].apply(_extract_names)
    movies["keywords"] = movies["keywords"].apply(_extract_names)
    movies["countries"] = movies["production_countries"].apply(_extract_names)
    movies["cast"] = movies["cast"].apply(_get_top_cast)
    movies["director"] = movies["crew"].apply(_get_director)
    movies["overview"] = movies["overview"].fillna("")
    movies = movies.drop(columns=DROP_COLUMNS, errors="ignore")

    movies["tags"] = movies.apply(build_tags, axis=1)
    return movies


def build_tags(row: pd.Series) -> str:
    director = [row["director"]] if row.get("director") else []
    parts = row["genres"] + row["keywords"] + row["cast"] + director + [row["overview"]]
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def build_documents(movies: pd.DataFrame) -> list[Document]:
    return [
        Document(
            page_content=row["tags"],
            metadata={
                "title": row["title"],
                "genres": row["genres"],
                "director": row["director"],
            },
        )
        for _, row in movies.iterrows()
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def build_vector_store(data_dir: Path, persist_directory: Path) -> Chroma:
    movies = load_movies_frame(data_dir)
    documents = build_documents(movies)
    split_docs = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    vectorstore.persist()
    return vectorstore
