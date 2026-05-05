import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    def __init__(self) -> None:
        self.user_item_matrix: pd.DataFrame | None = None
        self.item_similarity: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame) -> "CollaborativeRecommender":
        self.user_item_matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0,
        )
        similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns,
        )
        return self

    def recommend(self, user_id: int, top_n: int = 10) -> pd.Series:
        if self.user_item_matrix is None or self.item_similarity is None:
            raise RuntimeError("Call fit before recommend.")
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"Unknown userId: {user_id}")

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        scores = self.item_similarity[rated_movies].dot(user_ratings[rated_movies])
        scores = scores.drop(index=rated_movies, errors="ignore")
        return scores.sort_values(ascending=False).head(top_n)

