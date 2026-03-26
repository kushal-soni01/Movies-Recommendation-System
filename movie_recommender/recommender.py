from __future__ import annotations

from functools import lru_cache

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from movie_recommender.config import DB_DIR, load_settings
from movie_recommender.parsers import parse_recommendations
from movie_recommender.tmdb import TMDBClient


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a movie recommendation system.

Use the context below to recommend exactly 5 movies.

{context}

Return output in this format:
1. Movie Name - short reason
2. Movie Name - short reason
3. Movie Name - short reason
4. Movie Name - short reason
5. Movie Name - short reason

Do not include any introduction, outro, headings, or extra commentary.
""",
        ),
        ("human", "{input}"),
    ]
)


def _build_chain():
    settings = load_settings()
    if not settings.groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})
    llm = ChatGroq(model_name=settings.groq_model, api_key=settings.groq_api_key)
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever, document_chain)


@lru_cache(maxsize=1)
def get_chain():
    return _build_chain()


@lru_cache(maxsize=1)
def get_tmdb_client() -> TMDBClient:
    return TMDBClient(load_settings())


def _fallback_candidates(response: dict[str, object]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for document in response.get("context", []):
        metadata = getattr(document, "metadata", {}) or {}
        title = str(metadata.get("title", "")).strip()
        if not title:
            continue
        candidates.append(
            {
                "title": title,
                "reason": "Recommended from similar genre, director, cast, or storyline context.",
                "display_text": title,
            }
        )
    return candidates


def recommend_movies(query: str, genre: str = "All") -> list[dict[str, object]]:
    final_query = query.strip()
    if genre and genre != "All":
        final_query = f"{final_query} in {genre} genre"

    response = get_chain().invoke({"input": final_query})
    recommendations = parse_recommendations(response["answer"])
    recommendations.extend(_fallback_candidates(response))

    tmdb_client = get_tmdb_client()
    enriched_results: list[dict[str, object]] = []
    seen_titles: set[str] = set()
    for item in recommendations:
        normalized_title = item["title"].strip().lower()
        if not normalized_title or normalized_title in seen_titles:
            continue

        details = tmdb_client.fetch_movie_details(item["title"])
        if not details.poster_url:
            continue

        resolved_title = details.title.strip()
        resolved_key = resolved_title.lower()
        if resolved_key in seen_titles:
            continue

        seen_titles.add(normalized_title)
        seen_titles.add(resolved_key)
        enriched_results.append(
            {
                **item,
                "title": resolved_title,
                "rating": details.rating,
                "poster_url": details.poster_url,
            }
        )
        if len(enriched_results) == 5:
            break

    return enriched_results
