from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "processed"

MOVIES_PATH = DATA_DIR / "movies.csv"
KEYWORDS_CLEAN_PATH = DATA_DIR / "keywords_clean.csv"

# Если у вас уже есть отдельные таблицы:
KEYWORDS_PATH = DATA_DIR / "keywords.csv"
MOVIE_KEYWORDS_PATH = DATA_DIR / "movie_keywords.csv"


def main():
    movies = pd.read_csv(MOVIES_PATH, low_memory=False)

    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
    movie_ids = set(movies["movie_id"].dropna().astype(int).unique())

    # Вариант 1: если уже есть отдельные keywords.csv и movie_keywords.csv
    if KEYWORDS_PATH.exists() and MOVIE_KEYWORDS_PATH.exists():
        keywords = pd.read_csv(KEYWORDS_PATH, low_memory=False)
        movie_keywords = pd.read_csv(MOVIE_KEYWORDS_PATH, low_memory=False)

    # Вариант 2: если есть только keywords_clean.csv
    else:
        kw = pd.read_csv(KEYWORDS_CLEAN_PATH, low_memory=False)

        # keywords_clean должен содержать: movie_id, keyword_id, keyword
        required_cols = {"movie_id", "keyword_id"}
        missing = required_cols - set(kw.columns)
        assert not missing, f"Missing columns in keywords_clean.csv: {missing}"

        keywords = kw[["keyword_id"]].drop_duplicates().copy()

        if "keyword" in kw.columns:
            keywords["keyword"] = kw.groupby("keyword_id")["keyword"].first().values

        movie_keywords = kw[["movie_id", "keyword_id"]].drop_duplicates().copy()

    # Приводим типы
    keywords["keyword_id"] = pd.to_numeric(
        keywords["keyword_id"],
        errors="coerce"
    )

    movie_keywords["keyword_id"] = pd.to_numeric(
        movie_keywords["keyword_id"],
        errors="coerce"
    )

    movie_keywords["movie_id"] = pd.to_numeric(
        movie_keywords["movie_id"],
        errors="coerce"
    )

    # Убираем NaN для проверки
    keywords = keywords.dropna(subset=["keyword_id"])
    movie_keywords = movie_keywords.dropna(subset=["movie_id", "keyword_id"])

    keywords["keyword_id"] = keywords["keyword_id"].astype(int)
    movie_keywords["keyword_id"] = movie_keywords["keyword_id"].astype(int)
    movie_keywords["movie_id"] = movie_keywords["movie_id"].astype(int)

    keyword_ids = set(keywords["keyword_id"].unique())
    movie_keyword_ids = set(movie_keywords["keyword_id"].unique())
    movie_keywords_movie_ids = set(movie_keywords["movie_id"].unique())

    # Главные проверки внешних ключей
    missing_keywords = movie_keyword_ids - keyword_ids
    missing_movies = movie_keywords_movie_ids - movie_ids

    print("Foreign key check for movie_keywords\n")

    print("keywords keyword_id count:", len(keyword_ids))
    print("movie_keywords keyword_id count:", len(movie_keyword_ids))
    print("movies movie_id count:", len(movie_ids))
    print("movie_keywords movie_id count:", len(movie_keywords_movie_ids))

    print("\nFK check:")
    print("Missing keyword_id in keywords:", len(missing_keywords))
    print("Missing movie_id in movies:", len(missing_movies))

    if missing_keywords:
        print("\nExample missing keyword_id:")
        print(list(missing_keywords)[:20])

    if missing_movies:
        print("\nExample missing movie_id:")
        print(list(missing_movies)[:20])

    assert len(missing_keywords) == 0, (
        "ERROR: movie_keywords contains keyword_id values "
        "that do not exist in keywords"
    )

    assert len(missing_movies) == 0, (
        "ERROR: movie_keywords contains movie_id values "
        "that do not exist in movies"
    )

    print("\nOK: movie_keywords foreign keys are valid")


if __name__ == "__main__":
    main()
