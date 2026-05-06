from pathlib import Path
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]

MOVIES_PATH = BASE_DIR / "data" / "processed" / "movies.csv"
RATINGS_PATH = BASE_DIR / "data" / "processed" / "ratings.csv"

# Путь к сохранённой SVD-модели.
# При необходимости поменяйте название файла под ваш проект.
SVD_MODEL_PATH = BASE_DIR / "models" / "collab_model_svd.pkl"

# Если у вас модель сохранена без расширения, код ниже тоже попробует её найти.
SVD_MODEL_PATH_ALT = BASE_DIR / "models" / "collab_model_svd"


def load_movies():
    movies = pd.read_csv(MOVIES_PATH, low_memory=False)

    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
    movies = movies.dropna(subset=["movie_id"])
    movies["movie_id"] = movies["movie_id"].astype(int)

    movies["title"] = movies["title"].fillna("").astype(str)

    if "genre_names" not in movies.columns:
        movies["genre_names"] = ""

    movies["genre_names"] = movies["genre_names"].fillna("").astype(str)

    if "overview" not in movies.columns:
        movies["overview"] = ""

    movies["overview"] = movies["overview"].fillna("").astype(str)

    if "release_year" not in movies.columns:
        if "release_date" in movies.columns:
            movies["release_year"] = pd.to_datetime(
                movies["release_date"],
                errors="coerce"
            ).dt.year
        else:
            movies["release_year"] = np.nan

    movies["release_year"] = pd.to_numeric(
        movies["release_year"],
        errors="coerce"
    ).fillna(1900)

    movies = movies.drop_duplicates(subset=["movie_id"])
    movies = movies.drop_duplicates(subset=["title"]).reset_index(drop=True)

    return movies


def load_ratings():
    ratings = pd.read_csv(RATINGS_PATH)

    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["movie_id"] = pd.to_numeric(ratings["movie_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])

    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)

    return ratings


def load_svd_model():
    if SVD_MODEL_PATH.exists():
        with open(SVD_MODEL_PATH, "rb") as file:
            return pickle.load(file)

    if SVD_MODEL_PATH_ALT.exists():
        with open(SVD_MODEL_PATH_ALT, "rb") as file:
            return pickle.load(file)

    raise FileNotFoundError(
        "SVD model was not found. Check the path:\n"
        f"{SVD_MODEL_PATH}\n"
        f"{SVD_MODEL_PATH_ALT}"
    )


def normalize_scores(values):
    values = np.array(values, dtype=float)

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return np.ones_like(values) * 0.5

    return (values - min_value) / (max_value - min_value)


def build_faiss_index(movies):
    """
    Создаёт content-based FAISS index на основе:
    title + genre_names + overview + release_year.
    """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    full_texts = (
        movies["title"].fillna("") + " " +
        movies["genre_names"].fillna("") + " " +
        movies["overview"].fillna("")
    ).str.lower().tolist()

    text_embeddings = embedding_model.encode(
        full_texts,
        show_progress_bar=True
    )

    text_embeddings = np.array(text_embeddings, dtype="float32")

    years = movies["release_year"].values.reshape(-1, 1).astype("float32")
    years_norm = normalize_scores(years.flatten()).reshape(-1, 1).astype("float32")

    combined_features = np.hstack([text_embeddings, years_norm])

    norms = np.linalg.norm(combined_features, axis=1, keepdims=True)
    norms[norms == 0] = 1

    normalized_features = combined_features / norms
    normalized_features = normalized_features.astype("float32")

    faiss_index = faiss.IndexFlatIP(normalized_features.shape[1])
    faiss_index.add(normalized_features)

    return faiss_index, normalized_features


def get_user_liked_movie_ids(user_id, ratings, min_rating=4.0):
    """
    Берём фильмы, которые пользователь высоко оценил.
    Они используются как content-based профиль пользователя.
    """
    user_likes = ratings[
        (ratings["user_id"] == user_id) &
        (ratings["rating"] >= min_rating)
    ]

    return user_likes["movie_id"].unique().tolist()


def get_liked_movie_ids_from_titles(movies, liked_movie_titles):
    """
    Если пользователь явно передал названия любимых фильмов,
    находим их movie_id.
    """
    if liked_movie_titles is None:
        return []

    title_to_movie_id = pd.Series(
        movies["movie_id"].values,
        index=movies["title"].str.lower()
    ).drop_duplicates()

    liked_movie_ids = []

    for title in liked_movie_titles:
        title_lower = title.strip().lower()

        if title_lower in title_to_movie_id:
            liked_movie_ids.append(int(title_to_movie_id[title_lower]))

    return liked_movie_ids


def get_content_scores_for_user(
    movies,
    faiss_index,
    normalized_features,
    liked_movie_ids,
    top_k=500
):
    """
    Возвращает content-based score для фильмов,
    похожих на фильмы, которые понравились пользователю.
    """
    if not liked_movie_ids:
        raise ValueError(
            "No liked movies were found for content-based part of hybrid model."
        )

    movie_id_to_index = pd.Series(
        movies.index.values,
        index=movies["movie_id"]
    ).to_dict()

    liked_indices = [
        movie_id_to_index[movie_id]
        for movie_id in liked_movie_ids
        if movie_id in movie_id_to_index
    ]

    if not liked_indices:
        raise ValueError(
            "Liked movie IDs do not exist in movies.csv."
        )

    liked_vectors = normalized_features[np.array(liked_indices)]

    user_vector = np.mean(liked_vectors, axis=0)
    user_norm = np.linalg.norm(user_vector)

    if user_norm > 0:
        user_vector = user_vector / user_norm

    user_vector = user_vector.reshape(1, -1).astype("float32")

    search_k = min(top_k + len(liked_indices), len(movies))

    D, I = faiss_index.search(user_vector, search_k)

    liked_indices_set = set(liked_indices)

    content_results = []

    for idx, score in zip(I[0], D[0]):
        if idx in liked_indices_set:
            continue

        row = movies.iloc[idx]

        content_results.append({
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "content_score": float(score)
        })

    content_df = pd.DataFrame(content_results)

    if content_df.empty:
        return content_df

    content_df["content_score_norm"] = normalize_scores(
        content_df["content_score"].values
    )

    return content_df


def get_svd_scores_for_user(svd_model, user_id, candidate_movie_ids):
    """
    Получает predicted rating от SVD для candidate_movie_ids.
    """
    svd_results = []

    for movie_id in candidate_movie_ids:
        prediction = svd_model.predict(
            uid=int(user_id),
            iid=int(movie_id)
        )

        svd_results.append({
            "movie_id": int(movie_id),
            "svd_score": float(prediction.est)
        })

    svd_df = pd.DataFrame(svd_results)

    # SVD was trained on rating_scale=(1, 5).
    svd_df["svd_score_norm"] = (svd_df["svd_score"] - 1.0) / 4.0
    svd_df["svd_score_norm"] = svd_df["svd_score_norm"].clip(0, 1)

    return svd_df


def get_hybrid_recommendations(
    user_id,
    liked_movie_titles=None,
    alpha=0.6,
    top_n=10,
    content_top_k=5000
):
    """
    Hybrid recommender.

    Formula:
        hybrid_score = alpha * svd_score_norm
                     + (1 - alpha) * content_score_norm

    alpha = 0.6 means:
        60% collaborative filtering SVD
        40% content-based FAISS
    """
    movies = load_movies()
    ratings = load_ratings()
    svd_model = load_svd_model()

    faiss_index, normalized_features = build_faiss_index(movies)

    # 1. Берём liked movies из истории пользователя
    liked_movie_ids_from_ratings = get_user_liked_movie_ids(
        user_id=user_id,
        ratings=ratings,
        min_rating=4.0
    )

    # 2. Плюс можно передать фильмы вручную
    liked_movie_ids_from_titles = get_liked_movie_ids_from_titles(
        movies=movies,
        liked_movie_titles=liked_movie_titles
    )

    liked_movie_ids = sorted(
        set(liked_movie_ids_from_ratings + liked_movie_ids_from_titles)
    )

    if not liked_movie_ids:
        raise ValueError(
            f"User {user_id} has no liked movies. "
            "For hybrid recommendations, provide liked_movie_titles "
            "or use cold-start questionnaire first."
        )

    # 3. Content-based кандидаты через FAISS
    content_df = get_content_scores_for_user(
        movies=movies,
        faiss_index=faiss_index,
        normalized_features=normalized_features,
        liked_movie_ids=liked_movie_ids,
        top_k=content_top_k
    )

    if content_df.empty:
        return pd.DataFrame()

    candidate_movie_ids = content_df["movie_id"].unique()

    # 4. SVD score для этих же кандидатов
    svd_df = get_svd_scores_for_user(
        svd_model=svd_model,
        user_id=user_id,
        candidate_movie_ids=candidate_movie_ids
    )

    # 5. Объединяем по movie_id
    hybrid_df = content_df.merge(
        svd_df,
        on="movie_id",
        how="inner"
    )

    # 6. Remove all movies the user has already rated.
    rated_movie_ids = ratings.loc[
        ratings["user_id"] == user_id,
        "movie_id"
    ].unique()

    hybrid_df = hybrid_df[
        ~hybrid_df["movie_id"].isin(rated_movie_ids)
    ].copy()

    # 7. Финальный hybrid score
    hybrid_df["hybrid_score"] = (
        alpha * hybrid_df["svd_score_norm"]
        + (1 - alpha) * hybrid_df["content_score_norm"]
    )

    hybrid_df = hybrid_df.sort_values(
        by="hybrid_score",
        ascending=False
    )

    return hybrid_df[
        [
            "movie_id",
            "title",
            "svd_score",
            "content_score",
            "svd_score_norm",
            "content_score_norm",
            "hybrid_score"
        ]
    ].head(top_n)


if __name__ == "__main__":
    recommendations = get_hybrid_recommendations(
        user_id=1,
        liked_movie_titles=["The Matrix", "Inception"],
        alpha=0.6,
        top_n=10
    )

    print(recommendations.round(3).to_string(index=False))
