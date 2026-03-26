from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from movie_recommender.config import DATA_DIR, DB_DIR
from movie_recommender.data_pipeline import build_vector_store


if __name__ == "__main__":
    build_vector_store(DATA_DIR, DB_DIR)
    print(f"Vector store built at: {DB_DIR}")
