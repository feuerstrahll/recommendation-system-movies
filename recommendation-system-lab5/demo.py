from pathlib import Path

import pandas as pd

from recommender.content_based import ContentBasedRecommender


PROJECT_ROOT = Path(__file__).resolve().parent
MOVIES_PATH = PROJECT_ROOT / "data" / "processed" / "movies.csv"


def main() -> None:
    if not MOVIES_PATH.exists():
        print("Run etl/etl_pipeline.py after placing MovieLens files in data/raw.")
        return

    movies = pd.read_csv(MOVIES_PATH)
    recommender = ContentBasedRecommender().fit(movies)
    sample_movie_id = int(movies.iloc[0]["movieId"])
    recommendations = recommender.recommend(sample_movie_id, top_n=5)
    print(recommendations[["movieId", "title", "genres"]])


if __name__ == "__main__":
    main()

