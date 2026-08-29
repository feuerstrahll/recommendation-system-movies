from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "database" / "recommender.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"


def initialize_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_movielens(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    movies_path = raw_dir / "movies.csv"
    ratings_path = raw_dir / "ratings.csv"

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError("Place movies.csv and ratings.csv in data/raw first.")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def save_processed(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    ratings.to_csv(PROCESSED_DIR / "ratings.csv", index=False)


def main() -> None:
    initialize_database()
    movies, ratings = load_movielens()
    save_processed(movies, ratings)


if __name__ == "__main__":
    main()

