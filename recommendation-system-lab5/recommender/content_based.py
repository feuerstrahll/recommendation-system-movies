import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self) -> None:
        self.movies: pd.DataFrame | None = None
        self.similarity = None

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        self.movies = movies.reset_index(drop=True).copy()
        genres = self.movies["genres"].fillna("").str.replace("|", " ", regex=False)
        features = TfidfVectorizer().fit_transform(genres)
        self.similarity = cosine_similarity(features)
        return self

    def recommend(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if self.movies is None or self.similarity is None:
            raise RuntimeError("Call fit before recommend.")

        matches = self.movies.index[self.movies["movieId"] == movie_id].tolist()
        if not matches:
            raise ValueError(f"Unknown movieId: {movie_id}")

        movie_index = matches[0]
        scores = list(enumerate(self.similarity[movie_index]))
        scores = sorted(scores, key=lambda item: item[1], reverse=True)
        selected = [idx for idx, _ in scores if idx != movie_index][:top_n]
        return self.movies.iloc[selected].copy()

