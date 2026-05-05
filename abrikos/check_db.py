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
        tables = [row[0] for row in cur.fetchall()]

    print(f"Найдено таблиц: {len(tables)}")
    if not tables:
        print("⚠️  Таблиц нет — возможно схема не применялась")
        conn.close()

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for table in tables:
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