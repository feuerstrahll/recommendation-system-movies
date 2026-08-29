"""
Скрипт проверки подключения к PostgreSQL.
Проходит по всем таблицам базы данных и выводит первые 5 строк каждой.

Использование:
    python check_db.py \
        --host localhost --port 5432 \
        --db movies_db --user postgres --password secret
"""

import argparse
import psycopg2
from psycopg2.extras import RealDictCursor


EXPECTED_COLUMNS = {
    "movies": {"movie_id", "imdb_id", "title"},
    "ratings": {"user_id", "movie_id", "movielens_id", "rating", "timestamp"},
    "links": {"movielens_id", "imdb_id", "tmdb_id"},
    "cast_members": {"movie_id", "person_id", "name"},
    "crew_members": {"movie_id", "person_id", "name"},
    "keywords": {"keyword_id", "name"},
    "movie_keywords": {"movie_id", "keyword_id"},
}


FK_CHECKS = [
    {
        "name": "ratings.movie_id -> movies.movie_id",
        "tables": {"ratings", "movies"},
        "count_sql": """
            SELECT COUNT(*)
            FROM ratings r
            LEFT JOIN movies m ON m.movie_id = r.movie_id
            WHERE m.movie_id IS NULL
        """,
        "examples_sql": """
            SELECT DISTINCT r.movie_id
            FROM ratings r
            LEFT JOIN movies m ON m.movie_id = r.movie_id
            WHERE m.movie_id IS NULL
            LIMIT 10
        """,
    },
    {
        "name": "movie_keywords.movie_id -> movies.movie_id",
        "tables": {"movie_keywords", "movies"},
        "count_sql": """
            SELECT COUNT(*)
            FROM movie_keywords mk
            LEFT JOIN movies m ON m.movie_id = mk.movie_id
            WHERE m.movie_id IS NULL
        """,
        "examples_sql": """
            SELECT DISTINCT mk.movie_id
            FROM movie_keywords mk
            LEFT JOIN movies m ON m.movie_id = mk.movie_id
            WHERE m.movie_id IS NULL
            LIMIT 10
        """,
    },
    {
        "name": "movie_keywords.keyword_id -> keywords.keyword_id",
        "tables": {"movie_keywords", "keywords"},
        "count_sql": """
            SELECT COUNT(*)
            FROM movie_keywords mk
            LEFT JOIN keywords k ON k.keyword_id = mk.keyword_id
            WHERE k.keyword_id IS NULL
        """,
        "examples_sql": """
            SELECT DISTINCT mk.keyword_id
            FROM movie_keywords mk
            LEFT JOIN keywords k ON k.keyword_id = mk.keyword_id
            WHERE k.keyword_id IS NULL
            LIMIT 10
        """,
    },
    {
        "name": "cast_members.movie_id -> movies.movie_id",
        "tables": {"cast_members", "movies"},
        "count_sql": """
            SELECT COUNT(*)
            FROM cast_members c
            LEFT JOIN movies m ON m.movie_id = c.movie_id
            WHERE m.movie_id IS NULL
        """,
        "examples_sql": """
            SELECT DISTINCT c.movie_id
            FROM cast_members c
            LEFT JOIN movies m ON m.movie_id = c.movie_id
            WHERE m.movie_id IS NULL
            LIMIT 10
        """,
    },
    {
        "name": "crew_members.movie_id -> movies.movie_id",
        "tables": {"crew_members", "movies"},
        "count_sql": """
            SELECT COUNT(*)
            FROM crew_members c
            LEFT JOIN movies m ON m.movie_id = c.movie_id
            WHERE m.movie_id IS NULL
        """,
        "examples_sql": """
            SELECT DISTINCT c.movie_id
            FROM crew_members c
            LEFT JOIN movies m ON m.movie_id = c.movie_id
            WHERE m.movie_id IS NULL
            LIMIT 10
        """,
    },
]


def check_expected_schema(conn):
    problems = []

    with conn.cursor() as cur:
        for table, expected in EXPECTED_COLUMNS.items():
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                """,
                (table,),
            )
            actual = {row[0] for row in cur.fetchall()}

            if not actual:
                problems.append(f"{table}: table not found")
                continue

            missing = sorted(expected - actual)
            if missing:
                problems.append(f"{table}: missing columns {', '.join(missing)}")

            if table == "movies" and "id" in actual:
                problems.append("movies: old column id is still present; expected movie_id")

    return problems


def check_foreign_keys(conn, tables):
    problems = []

    with conn.cursor() as cur:
        for check in FK_CHECKS:
            missing_tables = sorted(check["tables"] - tables)
            if missing_tables:
                problems.append(
                    f"{check['name']}: skipped, missing tables "
                    f"{', '.join(missing_tables)}"
                )
                continue

            try:
                cur.execute(check["count_sql"])
                missing_count = cur.fetchone()[0]

                if missing_count == 0:
                    continue

                cur.execute(check["examples_sql"])
                examples = [str(row[0]) for row in cur.fetchall()]
                examples_text = ", ".join(examples) if examples else "no examples"
                problems.append(
                    f"{check['name']}: {missing_count} invalid rows; "
                    f"examples: {examples_text}"
                )
            except psycopg2.Error as e:
                conn.rollback()
                message = e.pgerror.strip() if e.pgerror else str(e)
                problems.append(f"{check['name']}: query failed: {message}")

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     default=5432, type=int)
    parser.add_argument("--db",       required=True)
    parser.add_argument("--user",     default="postgres")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    print(f"Подключаюсь к {args.host}:{args.port}/{args.db} ...")

    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.db,
            user=args.user,
            password=args.password,
        )
        print(f"✅ Подключено!\n")
    except psycopg2.OperationalError as e:
        print(f"❌ Ошибка подключения:\n{e}")
        return

    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = {row[0] for row in cur.fetchall()}

    print(f"Найдено таблиц: {len(tables)}")
    if not tables:
        print("⚠️  Таблиц нет — возможно схема не применялась")
        conn.close()
        return

    schema_problems = check_expected_schema(conn)
    if schema_problems:
        print("\nSchema consistency problems:")
        for problem in schema_problems:
            print(f"  - {problem}")
    else:
        print("\nSchema consistency: OK")

    fk_problems = check_foreign_keys(conn, tables)
    if fk_problems:
        print("\nForeign key consistency problems:")
        for problem in fk_problems:
            print(f"  - {problem}")
    else:
        print("\nForeign key consistency: OK")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for table in sorted(tables):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total = cur.fetchone()["count"]

            print(f"\nТаблица: {table}  (всего строк: {total})")
            print("-" * 60)

            cur.execute(f"SELECT * FROM {table} LIMIT 1")
            rows = cur.fetchall()

            if not rows:
                print("  (таблица пуста)")
            else:
                # Заголовок колонок
                cols = list(rows[0].keys())
                print("  " + " | ".join(cols))
                print("  " + "-" * (len(" | ".join(cols)) + 2))
                for row in rows:
                    values = []
                    for v in row.values():
                        s = str(v) if v is not None else "NULL"
                        # Обрезаем длинные значения для читаемости
                        values.append(s[:30] + "…" if len(s) > 30 else s)
                    print("  " + " | ".join(values))

    print("\n" + "=" * 60)
    print("✅ Проверка завершена")
    conn.close()

if __name__ == "__main__":
    main()
