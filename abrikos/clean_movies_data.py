"""
Скрипт очистки данных Movies Dataset для загрузки в PostgreSQL.

Запуск:
    python clean_movies_data.py --input ./data_Movies --output ./cleaned_data
"""

import argparse
import ast
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────


def parse_json_field(value):
    """Парсит строковое JSON-поле (одинарные или двойные кавычки)."""
    if pd.isna(value) or value in ("", "[]", "{}", "null", "None"):
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []


def extract_names(value, key="name"):
    """Извлекает список значений по ключу из JSON-поля."""
    items = parse_json_field(value)
    if not isinstance(items, list):
        return []
    return [item.get(key) for item in items if isinstance(item, dict) and key in item]


# ─────────────────────────────────────────────
# 1. movies_metadata.csv
# ─────────────────────────────────────────────


def clean_movies(path: str) -> pd.DataFrame:
    log.info("Очистка movies_metadata.csv ...")
    df = pd.read_csv(path, low_memory=False)
    log.info("  Строк до очистки: %d", len(df))

    # --- ID (TMDB) ---
    # 1) приводим к числу
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    before_drop_invalid = len(df)
    df = df[df["id"].notna()].copy()
    log.info("  Отброшено строк с невалидным id: %d", before_drop_invalid - len(df))

    df["id"] = df["id"].astype(int)

    # 2) логируем дубликаты по числовому id (если есть)
    dup_ids = df[df.duplicated(subset="id", keep=False)].sort_values("id")
    if not dup_ids.empty:
        log.warning("  Найдены дубликаты по id (TMDB): %d строк", len(dup_ids))
        # Можно сохранить для ручного разбора
        dup_ids.to_csv("movies_id_conflicts.csv", index=False)
        log.warning("  Конфликты по id сохранены в movies_id_conflicts.csv")

    # 3) удаляем дубликаты, оставляя первую запись по id
    before = len(df)
    df = df.drop_duplicates(subset="id", keep="first")
    log.info("  Удалено дубликатов по id: %d", before - len(df))

    # --- Числовые поля ---
    numeric_cols = ["budget", "revenue", "runtime",
                    "vote_count", "vote_average", "popularity"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 0 в бюджете / доходе считаем пропуском
    if "budget" in df.columns:
        df["budget"] = df["budget"].replace(0, np.nan)
    if "revenue" in df.columns:
        df["revenue"] = df["revenue"].replace(0, np.nan)

    # Выбросы runtime: < 1 мин или > 600 мин — пропуск
    if "runtime" in df.columns:
        df.loc[df["runtime"] < 1, "runtime"] = np.nan
        df.loc[df["runtime"] > 600, "runtime"] = np.nan

    # --- Даты ---
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year.astype("Int64")
        df["release_date"] = df["release_date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if not pd.isnull(x) else None
        )
    else:
        df["release_year"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["release_date"] = None

    # --- Булево ---
    for col in ["adult", "video"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
            )

    # --- Текстовые поля ---
    if "original_language" in df.columns:
        df["original_language"] = (
            df["original_language"].astype(str).str.strip().str[:10]
        )
    for col in ["status", "title", "original_title"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["overview", "tagline"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # --- Извлечение из JSON-полей ---
    if "genres" in df.columns:
        df["genre_names"] = df["genres"].apply(lambda x: extract_names(x, "name"))
    else:
        df["genre_names"] = [[] for _ in range(len(df))]

    if "production_companies" in df.columns:
        df["production_company_names"] = df["production_companies"].apply(
            lambda x: extract_names(x, "name")
        )
    else:
        df["production_company_names"] = [[] for _ in range(len(df))]

    if "production_countries" in df.columns:
        df["production_country_codes"] = df["production_countries"].apply(
            lambda x: extract_names(x, "iso_3166_1")
        )
    else:
        df["production_country_codes"] = [[] for _ in range(len(df))]

    if "spoken_languages" in df.columns:
        df["spoken_language_codes"] = df["spoken_languages"].apply(
            lambda x: extract_names(x, "iso_639_1")
        )
    else:
        df["spoken_language_codes"] = [[] for _ in range(len(df))]

    # Оставляем нужные плоские колонки
    keep = [
        "id", "imdb_id", "title", "original_title", "original_language",
        "overview", "tagline", "status", "adult", "video",
        "budget", "revenue", "runtime",
        "vote_count", "vote_average", "popularity",
        "release_date", "release_year",
        "genre_names", "production_company_names",
        "production_country_codes", "spoken_language_codes",
        "poster_path", "backdrop_path", "homepage", "belongs_to_collection",
    ]
    df = df[[c for c in keep if c in df.columns]]

    log.info("  Строк после очистки: %d", len(df))
    return df


# ─────────────────────────────────────────────
# 2. credits.csv
# ─────────────────────────────────────────────


def clean_credits(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает два датафрейма: cast и crew."""
    log.info("Очистка credits.csv ...")
    df = pd.read_csv(path)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset="id")

    cast_rows = []
    crew_rows = []

    for _, row in df.iterrows():
        movie_id = row["id"]

        for person in parse_json_field(row.get("cast", "[]")):
            if not isinstance(person, dict):
                continue
            cast_rows.append(
                {
                    "movie_id": movie_id,
                    "cast_id": person.get("cast_id"),
                    "person_id": person.get("id"),
                    "name": (person.get("name") or "").strip(),
                    "character": (person.get("character") or "").strip(),
                    "order": person.get("order"),
                    "gender": person.get("gender"),
                    "profile_path": person.get("profile_path"),
                }
            )

        for person in parse_json_field(row.get("crew", "[]")):
            if not isinstance(person, dict):
                continue
            crew_rows.append(
                {
                    "movie_id": movie_id,
                    "person_id": person.get("id"),
                    "name": (person.get("name") or "").strip(),
                    "department": (person.get("department") or "").strip(),
                    "job": (person.get("job") or "").strip(),
                    "gender": person.get("gender"),
                    "profile_path": person.get("profile_path"),
                }
            )

    cast_df = pd.DataFrame(cast_rows)
    crew_df = pd.DataFrame(crew_rows)

    # Типы
    for col in ["cast_id", "person_id", "order", "gender", "movie_id"]:
        if col in cast_df.columns:
            cast_df[col] = pd.to_numeric(cast_df[col], errors="coerce").astype("Int64")
    for col in ["person_id", "gender", "movie_id"]:
        if col in crew_df.columns:
            crew_df[col] = pd.to_numeric(crew_df[col], errors="coerce").astype("Int64")

    log.info("  Cast строк: %d, Crew строк: %d", len(cast_df), len(crew_df))
    return cast_df, crew_df


# ─────────────────────────────────────────────
# 3. keywords.csv
# ─────────────────────────────────────────────


def clean_keywords(path: str) -> pd.DataFrame:
    log.info("Очистка keywords.csv ...")
    df = pd.read_csv(path)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset="id")

    rows = []
    for _, row in df.iterrows():
        movie_id = row["id"]
        for kw in parse_json_field(row.get("keywords", "[]")):
            if isinstance(kw, dict):
                rows.append(
                    {
                        "movie_id": movie_id,
                        "keyword_id": kw.get("id"),
                        "keyword": (kw.get("name") or "").strip(),
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        log.info("  Keyword строк: 0")
        return result

    result["keyword_id"] = pd.to_numeric(
        result["keyword_id"], errors="coerce"
    ).astype("Int64")
    result["movie_id"] = result["movie_id"].astype(int)
    log.info("  Keyword строк: %d", len(result))
    return result


# ─────────────────────────────────────────────
# 4. ratings.csv / ratings_small.csv
# ─────────────────────────────────────────────


def clean_ratings(path: str) -> pd.DataFrame:
    log.info("Очистка %s ...", os.path.basename(path))
    df = pd.read_csv(path)

    df["userId"] = pd.to_numeric(df["userId"], errors="coerce").astype("Int64")
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df["timestamp"] = df["timestamp"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if not pd.isnull(x) else None
    )

    # Фильтр по допустимому диапазону рейтинга
    df = df[df["rating"].between(0.5, 5.0)]
    # Убираем записи без userId / movieId / rating
    df = df.dropna(subset=["userId", "movieId", "rating"])

    before = len(df)
    df = df.drop_duplicates(subset=["userId", "movieId"])
    log.info("  Строк после очистки: %d (удалено дубликатов: %d)", len(df), before - len(df))
    return df


# ─────────────────────────────────────────────
# 5. links.csv
# ─────────────────────────────────────────────


def clean_links(path: str) -> pd.DataFrame:
    log.info("Очистка %s ...", os.path.basename(path))
    df = pd.read_csv(path)

    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce").astype("Int64")
    df["imdbId"] = pd.to_numeric(df["imdbId"], errors="coerce").astype("Int64")
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["movieId"])
    before = len(df)
    df = df.drop_duplicates(subset=["movieId"])
    log.info("  Строк после очистки: %d (удалено дубликатов по movieId: %d)", len(df), before - len(df))
    return df


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Очистка Movies Dataset")
    parser.add_argument(
        "--input", default="./data_Movies", help="Папка с исходными CSV"
    )
    parser.add_argument(
        "--output", default="./cleaned_data", help="Папка для очищенных CSV"
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # movies_metadata
    movies_df = clean_movies(inp / "movies_metadata.csv")
    movies_df.to_csv(out / "movies_clean.csv", index=False)

    # credits
    cast_df, crew_df = clean_credits(inp / "credits.csv")
    cast_df.to_csv(out / "cast_clean.csv", index=False)
    crew_df.to_csv(out / "crew_clean.csv", index=False)

    # keywords
    kw_df = clean_keywords(inp / "keywords.csv")
    kw_df.to_csv(out / "keywords_clean.csv", index=False)

    # ratings
    ratings_df = clean_ratings(inp / "ratings.csv")
    ratings_df.to_csv(out / "ratings_clean.csv", index=False)

    # links
    links_df = clean_links(inp / "links.csv")
    links_df.to_csv(out / "links_clean.csv", index=False)

    log.info("Очистка завершена. Файлы сохранены в %s", str(out))


if __name__ == "__main__":
    main()