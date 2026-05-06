"""
Загрузка очищенных данных в PostgreSQL.

Требует: psycopg2-binary, pandas

Пример:
    python load_to_postgres.py `
        --host localhost --port 5432 `
        --db movies_db --user postgres --password secret `
        --data ./cleaned_data --schema ./schema.sql
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Подключение
# ─────────────────────────────────────────────


def get_connection(args):
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.db,
        user=args.user,
        password=args.password,
    )


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────


def df_to_list_of_tuples(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Преобразует DataFrame в список кортежей для execute_values."""
    sub = df[columns].copy()
    sub = sub.where(sub.notna(), other=None)
    return [tuple(row) for row in sub.itertuples(index=False, name=None)]


def parse_array_field(value) -> list:
    """Конвертирует строку/список обратно в список для PostgreSQL ARRAY."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        import ast

        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# ─────────────────────────────────────────────
# Загрузка movies
# ─────────────────────────────────────────────


def load_movies(conn, data_dir: Path):
    df = pd.read_csv(data_dir / "movies_clean.csv", low_memory=False)
    log.info("Загрузка movies: %d строк ...", len(df))

    # release_date -> DATE
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce").dt.date
    df = df.replace({np.nan: None})

    desired_columns = [
        "id", "imdb_id", "title", "original_title", "original_language",
        "overview", "tagline", "status", "adult", "video",
        "budget", "revenue", "runtime",
        "vote_count", "vote_average", "popularity",
        "release_date", "release_year",
        "poster_path", "backdrop_path", "homepage", "belongs_to_collection",
    ]

    missing = [c for c in desired_columns if c not in df.columns]
    if missing:
        log.warning("  Отсутствующие колонки (будут NULL): %s", set(missing))
        for c in missing:
            df[c] = None

    # title NOT NULL по схеме
    df["title"] = df["title"].fillna(df.get("original_title", "")).fillna("Unknown")
    df["title"] = df["title"].astype(str).str.strip().replace("", "Unknown")

    # Массивы
    arr_cols = [
        "genre_names",
        "production_company_names",
        "production_country_codes",
        "spoken_language_codes",
    ]
    for arr_col in arr_cols:
        if arr_col in df.columns:
            df[arr_col] = df[arr_col].apply(parse_array_field)
        else:
            df[arr_col] = [[] for _ in range(len(df))]

    df = df.where(df.notna(), other=None)

    all_cols = desired_columns + arr_cols
    sql = f"""
        INSERT INTO movies ({", ".join(all_cols)})
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            title         = EXCLUDED.title,
            vote_average  = EXCLUDED.vote_average,
            vote_count    = EXCLUDED.vote_count,
            popularity    = EXCLUDED.popularity
    """

    rows = [tuple(row) for row in df[all_cols].itertuples(index=False, name=None)]

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=500)
    conn.commit()
    log.info("  ✅ movies загружены")


# ─────────────────────────────────────────────
# Загрузка genres и movie_genres
# ─────────────────────────────────────────────


def load_genres_and_movie_genres(conn, data_dir: Path):
    """Строит справочник жанров и связи movie_genres по movies_clean.csv."""
    df = pd.read_csv(data_dir / "movies_clean.csv", low_memory=False)
    if "genre_names" not in df.columns:
        log.warning("В movies_clean.csv нет колонки genre_names — пропускаю genres/movie_genres")
        return

    log.info("Построение genres и movie_genres из movies_clean.csv ...")

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)
    df["genre_names"] = df["genre_names"].apply(parse_array_field)

    # 1. Справочник жанров
    genre_set: set[str] = set()
    for genres in df["genre_names"]:
        if not isinstance(genres, list):
            continue
        for g in genres:
            g = (g or "").strip()
            if g:
                genre_set.add(g)

    genres_df = pd.DataFrame(sorted(genre_set), columns=["name"])

    if not genres_df.empty:
        rows = df_to_list_of_tuples(genres_df, ["name"])
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO genres (name) VALUES %s
                ON CONFLICT (name) DO NOTHING
                """,
                rows,
                page_size=200,
            )
        conn.commit()
        log.info("  ✅ genres загружены (%d жанров)", len(genres_df))
    else:
        log.info("  Нет жанров для загрузки")
        return

    # 2. Маппинг name -> id из БД
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM genres")
        name_to_id = {name: gid for gid, name in cur.fetchall()}

    # 3. Связи movie_genres
    rel_rows: list[tuple[int, int]] = []
    for _, row in df[["id", "genre_names"]].iterrows():
        movie_id = int(row["id"])
        genres = row["genre_names"] or []
        for g in genres:
            g = (g or "").strip()
            gid = name_to_id.get(g)
            if gid is not None:
                rel_rows.append((movie_id, gid))

    if rel_rows:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO movie_genres (movie_id, genre_id) VALUES %s
                ON CONFLICT (movie_id, genre_id) DO NOTHING
                """,
                rel_rows,
                page_size=500,
            )
        conn.commit()
        log.info("  ✅ movie_genres загружены (%d связей)", len(rel_rows))
    else:
        log.info("  Нет связей movie_genres для загрузки")


# ─────────────────────────────────────────────
# Загрузка cast & crew
# ─────────────────────────────────────────────


def load_cast(conn, data_dir: Path):
    df = pd.read_csv(data_dir / "cast_clean.csv")
    log.info("Загрузка cast_members: %d строк ...", len(df))

    columns = [
        "movie_id",
        "person_id",
        "cast_id",
        "name",
        "character",
        "order",
        "gender",
        "profile_path",
    ]
    rows = df_to_list_of_tuples(df, columns)

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO cast_members (
                movie_id, person_id, cast_id, name,
                character, "order", gender, profile_path
            )
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            rows,
            page_size=1000,
        )
    conn.commit()
    log.info("  ✅ cast_members загружены")


def load_crew(conn, data_dir: Path):
    df = pd.read_csv(data_dir / "crew_clean.csv")
    log.info("Загрузка crew_members: %d строк ...", len(df))

    columns = [
        "movie_id",
        "person_id",
        "name",
        "department",
        "job",
        "gender",
        "profile_path",
    ]
    rows = df_to_list_of_tuples(df, columns)

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO crew_members (
                movie_id, person_id, name, department, job,
                gender, profile_path
            )
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            rows,
            page_size=1000,
        )
    conn.commit()
    log.info("  ✅ crew_members загружены")


# ─────────────────────────────────────────────
# Загрузка keywords / movie_keywords
# ─────────────────────────────────────────────


def load_keywords(conn, data_dir: Path):
    df = pd.read_csv(data_dir / "keywords_clean.csv")
    log.info("Загрузка keywords / movie_keywords: %d строк ...", len(df))

    if df.empty:
        log.info("  Пустой keywords_clean.csv, пропускаю")
        return

    # ---- Справочник keywords ----
    kw = df[["keyword_id", "keyword"]].copy()

    # чистим id и имя
    kw["keyword_id"] = pd.to_numeric(kw["keyword_id"], errors="coerce")
    kw = kw[kw["keyword_id"].notna()].copy()
    kw["keyword_id"] = kw["keyword_id"].astype(int)

    kw["keyword"] = kw["keyword"].astype(str).str.strip()
    kw = kw[kw["keyword"] != ""]

    # важное: уникальность по ID, а не по имени
    kw = kw.drop_duplicates(subset=["keyword_id"], keep="first")

    keywords_df = kw.rename(columns={"keyword_id": "id", "keyword": "name"})

    log.info("  Уникальных keywords по id: %d", len(keywords_df))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO keywords (id, name)
            VALUES %s
            ON CONFLICT (id) DO NOTHING
            """,
            df_to_list_of_tuples(keywords_df, ["id", "name"]),
            page_size=1000,
        )

    # множество реально существующих id
    valid_ids = set(keywords_df["id"].tolist())

    # ---- Связи movie_keywords ----
    links = df[["movie_id", "keyword_id"]].copy()
    links["movie_id"] = pd.to_numeric(links["movie_id"], errors="coerce")
    links["keyword_id"] = pd.to_numeric(links["keyword_id"], errors="coerce")

    links = links[links["movie_id"].notna() & links["keyword_id"].notna()].copy()
    links["movie_id"] = links["movie_id"].astype(int)
    links["keyword_id"] = links["keyword_id"].astype(int)

    # оставляем только те keyword_id, которые реально есть в keywords
    before = len(links)
    links = links[links["keyword_id"].isin(valid_ids)]
    log.info(
        "  Связей movie_keywords после фильтра по существующим keyword_id: %d (отброшено %d)",
        len(links),
        before - len(links),
    )

    links = links.drop_duplicates(subset=["movie_id", "keyword_id"])

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO movie_keywords (movie_id, keyword_id)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            df_to_list_of_tuples(links, ["movie_id", "keyword_id"]),
            page_size=1000,
        )

    conn.commit()
    log.info("  ✅ keywords и movie_keywords загружены")


# ─────────────────────────────────────────────
# Загрузка ratings
# ─────────────────────────────────────────────


def load_ratings(conn, data_dir: Path, filename: str = "ratings_clean.csv"):
    path = data_dir / filename
    if not path.exists():
        log.warning("Файл %s не найден, пропускаю ratings", path)
        return

    # Быстрая проверка, не загружены ли уже
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ratings")
        count = cur.fetchone()[0]

    if count > 0:
        log.info("ratings уже содержат %d строк, пропускаю загрузку", count)
        return

    log.info("Загрузка ratings из %s ...", path)
    with conn.cursor() as cur, open(path, "r", encoding="utf-8", newline="") as f:
        cur.copy_expert(
            """
            COPY ratings (user_id, movielens_id, rating, rated_at)
            FROM STDIN WITH (FORMAT csv, HEADER true)
            """,
            f,
        )
    conn.commit()
    log.info("  ✅ ratings загружены")


# ─────────────────────────────────────────────
# Загрузка links
# ─────────────────────────────────────────────


def load_links(conn, data_dir: Path, filename: str = "links_clean.csv"):
    path = data_dir / filename
    if not path.exists():
        log.warning("Файл %s не найден, пропускаю links", path)
        return

    df = pd.read_csv(path)
    log.info("Загрузка links из %s (%d строк) ...", path, len(df))

    df = df.rename(
        columns={
            "movieId": "movielens_id",
            "imdbId": "imdb_id",
            "tmdbId": "tmdb_id",
        }
    )

    # Приводим к числам
    for col in ["movielens_id", "imdb_id", "tmdb_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Убираем строки без movielens_id
    df = df[df["movielens_id"].notna()].copy()

    rows = []
    for movielens_id, imdb_id, tmdb_id in df[["movielens_id", "imdb_id", "tmdb_id"]].itertuples(index=False, name=None):
        rows.append(
            (
                int(movielens_id) if movielens_id is not None and not pd.isna(movielens_id) else None,
                int(imdb_id) if imdb_id is not None and not pd.isna(imdb_id) else None,
                int(tmdb_id) if tmdb_id is not None and not pd.isna(tmdb_id) else None,
            )
        )

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO links (movielens_id, imdb_id, tmdb_id)
            VALUES %s
            ON CONFLICT (movielens_id) DO NOTHING
            """,
            rows,
            page_size=1000,
        )
    conn.commit()
    log.info("  ✅ links загружены")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Загрузка Movies Dataset в PostgreSQL")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--db", default="movies_db")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="")
    parser.add_argument("--data", default="./cleaned_data")
    parser.add_argument(
        "--schema", default="./schema.sql", help="Файл с SQL-схемой (schema.sql)"
    )
    args = parser.parse_args()

    conn = get_connection(args)
    log.info("Подключение к PostgreSQL %s:%s/%s установлено", args.host, args.port, args.db)

    # Применяем схему
    if os.path.exists(args.schema):
        with open(args.schema, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
        log.info("Схема из %s применена", args.schema)
    else:
        log.warning("Файл схемы %s не найден, пропускаю", args.schema)

    data_dir = Path(args.data)

    load_movies(conn, data_dir)
    load_genres_and_movie_genres(conn, data_dir)
    load_cast(conn, data_dir)
    load_crew(conn, data_dir)
    load_keywords(conn, data_dir)
    load_ratings(conn, data_dir, "ratings_clean.csv")
    load_links(conn, data_dir, "links_clean.csv")

    conn.close()
    log.info("Готово: загрузка в БД завершена")


if __name__ == "__main__":
    main()