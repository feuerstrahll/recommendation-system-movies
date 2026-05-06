import sqlite3
from pathlib import Path

import pandas as pd

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    cwd = Path.cwd()
    BASE_DIR = cwd if (cwd / "database").exists() else cwd.parent

DB_PATH = BASE_DIR / "database" / "recommender.db"


RECOMMENDATION_COLUMNS = [
    "movie_id",
    "title",
    "release_year",
    "genres",
    "vote_average",
    "vote_count",
    "popularity",
    "genre_match_score",
    "cold_start_score",
]


GENRE_ALIASES = {
    "sci-fi": "Science Fiction",
    "scifi": "Science Fiction",
    "science fiction": "Science Fiction",
    "romance": "Romance",
    "action": "Action",
    "comedy": "Comedy",
    "drama": "Drama",
    "horror": "Horror",
    "adventure": "Adventure",
    "animation": "Animation",
    "thriller": "Thriller",
    "crime": "Crime",
    "fantasy": "Fantasy",
    "mystery": "Mystery",
    "family": "Family",
    "documentary": "Documentary",
}


def connect_db():
    """
    Connects to SQLite database.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Make sure recommender.db exists in the database/ folder."
        )

    return sqlite3.connect(DB_PATH)


def normalize_genre_name(genre_name):
    """
    Converts user input genre to the format used in the database.

    Example:
    'sci-fi' -> 'Science Fiction'
    'action' -> 'Action'
    """
    genre_name = genre_name.strip().lower()
    return GENRE_ALIASES.get(genre_name, genre_name.title())


def normalize_column(series):
    """
    Min-max normalization for numeric columns.
    Converts values to range from 0 to 1.
    """
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


def empty_recommendations():
    return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)


def load_movies_with_genres():
    """
    Loads movies and their genres from the database.

    Returns a DataFrame where each movie has a list of genres.
    """
    conn = connect_db()

    movies = pd.read_sql_query(
        """
        SELECT
            movie_id,
            title,
            overview,
            release_year,
            vote_average,
            vote_count,
            popularity
        FROM movies
        """,
        conn
    )

    movie_genres = pd.read_sql_query(
        """
        SELECT
            mg.movie_id,
            g.name AS genre_name
        FROM movie_genres mg
        JOIN genres g ON mg.genre_id = g.genre_id
        """,
        conn
    )

    conn.close()

    # Combine several genres of one movie into one text field
    genres_grouped = (
        movie_genres
        .groupby("movie_id")["genre_name"]
        .apply(lambda genres: list(sorted(set(genres))))
        .reset_index()
    )

    movies = movies.merge(genres_grouped, on="movie_id", how="left")
    movies["genre_name"] = movies["genre_name"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    return movies


def find_liked_movie_ids(movies, liked_movie_titles):
    """
    Finds movie IDs based on titles selected by the user.

    Example:
    user writes ['Matrix', 'Titanic']
    system finds matching movies in the database.
    """
    if not liked_movie_titles:
        return []

    liked_movie_ids = []

    for title in liked_movie_titles:
        title_lower = title.strip().lower()

        matches = movies[
            movies["title"]
            .fillna("")
            .str.lower()
            .str.contains(title_lower, regex=False)
        ]

        if not matches.empty:
            # Take the most popular matching movie
            best_match = matches.sort_values(
                by="popularity",
                ascending=False
            ).iloc[0]

            liked_movie_ids.append(int(best_match["movie_id"]))

    return liked_movie_ids


def extract_genres_from_liked_movies(movies, liked_movie_ids):
    """
    If the user selected movies they already like,
    we can extract genres from these movies.

    Example:
    The Matrix -> Action, Science Fiction
    Titanic -> Drama, Romance
    """
    if not liked_movie_ids:
        return []

    liked_movies = movies[movies["movie_id"].isin(liked_movie_ids)]

    extracted_genres = set()

    for genres in liked_movies["genre_name"]:
        for genre in genres:
            extracted_genres.add(genre)

    return list(extracted_genres)


def apply_release_preference(movies, release_preference):
    """
    Filters movies according to old/new/both preference.

    old  -> movies before 2000
    new  -> movies from 2000 and later
    both -> no filtering
    """
    movies = movies.copy()
    movies["release_year"] = pd.to_numeric(
        movies["release_year"],
        errors="coerce"
    )

    release_preference = release_preference.strip().lower()

    if release_preference == "old":
        return movies[movies["release_year"] < 2000]

    if release_preference == "new":
        return movies[movies["release_year"] >= 2000]

    return movies


def recommend_for_new_user(
    selected_genres,
    liked_movie_titles=None,
    release_preference="both",
    top_n=10
):
    """
    Main cold start function for a new user.

    It uses:
    1. selected genres;
    2. movies the user already likes;
    3. old/new/both preference.

    Then it recommends popular movies from suitable genres.
    """
    movies = load_movies_with_genres()

    selected_genres = selected_genres or []
    liked_movie_titles = liked_movie_titles or []

    # Normalize selected genres from questionnaire
    selected_genres = [
        normalize_genre_name(genre)
        for genre in selected_genres
        if genre.strip()
    ]

    # Find movies user already likes
    liked_movie_ids = find_liked_movie_ids(
        movies=movies,
        liked_movie_titles=liked_movie_titles
    )

    # Extract additional genres from liked movies
    genres_from_liked_movies = extract_genres_from_liked_movies(
        movies=movies,
        liked_movie_ids=liked_movie_ids
    )

    # Final user preference profile
    final_genres = sorted(set(selected_genres + genres_from_liked_movies))
    final_genres_set = set(final_genres)

    candidates = movies.copy()

    # Filter by genre
    if final_genres:
        candidates = candidates[
            candidates["genre_name"].apply(
                lambda movie_genres: bool(set(movie_genres) & final_genres_set)
            )
        ]

    # Apply old/new/both preference
    candidates = apply_release_preference(
        movies=candidates,
        release_preference=release_preference
    )

    # Exclude movies the user already likes
    if liked_movie_ids:
        candidates = candidates[
            ~candidates["movie_id"].isin(liked_movie_ids)
        ]

    if candidates.empty:
        return empty_recommendations(), final_genres, liked_movie_ids

    # Remove movies with too little information
    candidates["vote_average"] = pd.to_numeric(
        candidates["vote_average"],
        errors="coerce"
    ).fillna(0)

    candidates["vote_count"] = pd.to_numeric(
        candidates["vote_count"],
        errors="coerce"
    ).fillna(0)

    candidates["popularity"] = pd.to_numeric(
        candidates["popularity"],
        errors="coerce"
    ).fillna(0)

    # If possible, keep only movies with at least some votes
    if len(candidates[candidates["vote_count"] >= 50]) >= top_n:
        candidates = candidates[candidates["vote_count"] >= 50]

    # Normalize values
    candidates["vote_average_norm"] = candidates["vote_average"] / 10
    candidates["vote_count_norm"] = normalize_column(candidates["vote_count"])
    candidates["popularity_norm"] = normalize_column(candidates["popularity"])
    candidates["genre_match_score"] = candidates["genre_name"].apply(
        lambda movie_genres: (
            len(set(movie_genres) & final_genres_set) / len(final_genres_set)
            if final_genres_set else 0
        )
    )

    # Final cold start score
    candidates["cold_start_score"] = (
        candidates["genre_match_score"] * 0.35
        + candidates["vote_average_norm"] * 0.30
        + candidates["popularity_norm"] * 0.25
        + candidates["vote_count_norm"] * 0.10
    )

    recommendations = candidates.sort_values(
        by="cold_start_score",
        ascending=False
    ).head(top_n)

    result = recommendations[
        [column for column in RECOMMENDATION_COLUMNS if column != "genres"] +
        ["genre_name"]
    ].copy()

    result = result.rename(columns={
        "genre_name": "genres"
    })

    result = result[RECOMMENDATION_COLUMNS]

    return result, final_genres, liked_movie_ids


def run_questionnaire():
    """
    Simple console questionnaire for a new user.
    """
    print("\nNew User Registration Questionnaire")
    print("-----------------------------------")

    print("\nQuestion 1: Choose your favorite genres.")
    print("Example: Action, Comedy, Drama, Horror, Romance, Sci-Fi")
    genres_input = input("Your genres: ")

    selected_genres = [
        genre.strip()
        for genre in genres_input.split(",")
        if genre.strip()
    ]

    print("\nQuestion 2: Choose movies you already like.")
    print("Example: The Matrix, Toy Story, Titanic, Avatar")
    liked_movies_input = input("Movies you like: ")

    liked_movie_titles = [
        title.strip()
        for title in liked_movies_input.split(",")
        if title.strip()
    ]

    print("\nQuestion 3: Do you prefer old or new movies?")
    print("Options: old / new / both")
    release_preference = input("Your preference: ").strip().lower()

    if release_preference not in ["old", "new", "both"]:
        release_preference = "both"

    recommendations, final_genres, liked_movie_ids = recommend_for_new_user(
        selected_genres=selected_genres,
        liked_movie_titles=liked_movie_titles,
        release_preference=release_preference,
        top_n=10
    )

    print("\nUser preference profile")
    print("-----------------------")
    print("Selected and inferred genres:", final_genres)
    print("Liked movie IDs found:", liked_movie_ids)
    print("Release preference:", release_preference)

    print("\nCold Start Recommendations")
    print("--------------------------")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    run_questionnaire()
