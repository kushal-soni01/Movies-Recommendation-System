# Movies Recommendation System

An AI-powered movie and TV recommendation web app built with Flask, LangChain, Groq, ChromaDB, and TMDB.

The app lets users search with natural language prompts like:

- `Suggest me more movies like Interstellar`
- `Romantic comedy movies`
- `One-Sided love story movies`
- `Suggest me some good movies by Robert Downey Jr.`
- `Movies with great sound tracks`

It combines:

- a retrieval pipeline built from the TMDB 5000 dataset
- Groq LLM recommendations using LangChain
- TMDB poster, rating, and trailer enrichment
- a polished Flask frontend with featured movies on the landing page

## Features

- Natural language movie recommendations
- Genre-aware search support
- Poster-backed movie cards only
- Clickable posters that open trailers when available
- Rotating typewriter placeholder suggestions
- Featured high-rated movies on the landing page
- Clean Flask app structure instead of a single notebook-driven UI

## Tech Stack

- Flask
- LangChain
- Groq
- ChromaDB
- Sentence Transformers
- Pandas
- TMDB API

## Project Structure

```text
Movies_Recommendation/
├── app.py
├── requirements.txt
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── db/
├── movie_recommender/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── parsers.py
│   ├── recommender.py
│   ├── tmdb.py
│   └── web.py
├── scripts/
│   └── build_vector_store.py
├── static/
│   └── styles.css
└── templates/
    └── index.html
```

## How It Works

1. The TMDB CSV files are cleaned and transformed into tags using genres, keywords, cast, director, and overview text.
2. These tags are converted into LangChain documents.
3. The documents are embedded using `all-MiniLM-L6-v2`.
4. Chunks are stored in a persistent ChromaDB vector store.
5. User prompts are sent through a retrieval chain powered by Groq.
6. Recommended titles are enriched with poster, rating, and trailer data from TMDB.
7. The Flask frontend renders the final cards.

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
TMDB_API_KEY=your_tmdb_api_key
```

Optional:

```env
TMDB_BASE_URL=https://api.themoviedb.org/3
TMDB_IMAGE_BASE_URL=https://image.tmdb.org/t/p/w500
RETRIEVER_K=5
```

## Installation

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Build the Vector Store

If you update the dataset or preprocessing logic, rebuild the Chroma database:

```powershell
python -m scripts.build_vector_store
```

## Run the App

Run the Flask app locally:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Example Queries

- `Suggest me more movies like Interstellar`
- `One-sided Love Story movies`
- `Suggest me shows by duffer brothers`
- `Movies with great sound tracks`
- `Romantic comedy movies`

## Notes

- Only titles with valid posters are displayed in recommendation cards.
- Posters open the trailer in a new tab when TMDB provides one.
- The notebook used during experimentation is no longer required for the app runtime.

## Author

Made with 💖 by [Kushal Soni](https://github.com/kushal-soni01/Movies-Recommendation-System)
