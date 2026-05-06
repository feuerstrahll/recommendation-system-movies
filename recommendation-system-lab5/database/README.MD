# Database

Папка содержит актуальные файлы для подготовки и проверки базы данных рекомендательной системы.

## Файлы

- `clean_movies_data.py` - очистка The Movies Dataset и сборка финальных CSV.
- `schema.sql` - актуальная PostgreSQL-схема.
- `check_db.py` - проверка структуры таблиц и ссылочной целостности в PostgreSQL.
- `database_architecture.md` - описание таблиц и связей.
- `er_diagram.mmd` - Mermaid ER-диаграмма.
- `recommender.db` - SQLite-снимок для локальных экспериментов.

## Очистка данных

Из папки `lab-5/recommendation-system-lab5`:

```powershell
python database/clean_movies_data.py
```

По умолчанию:

- input: `data/raw`
- output: `data/processed`

Явный запуск:

```powershell
python database/clean_movies_data.py --input data/raw --output data/processed
```

Скрипт ожидает исходные файлы:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `ratings.csv`
- `links.csv`

## Выходные CSV

- `movies.csv` - финальная таблица фильмов, ключ `movie_id` равен TMDB id.
- `ratings.csv` - финальные оценки, `movie_id` уже приведен к TMDB id.
- `movies_clean.csv`, `ratings_clean.csv`, `links_clean.csv`
- `keywords_clean.csv`, `cast_clean.csv`, `crew_clean.csv`

`movielens_id` в `ratings.csv` сохранен как исходный MovieLens id, но основной ключ для рекомендаций - `movie_id`.

## PostgreSQL

Применение схемы:

```powershell
psql -U postgres -d movies_db -f database/schema.sql
```

Проверка БД:

```powershell
python database/check_db.py --db movies_db --user postgres --password secret
```

`check_db.py` проверяет:

- наличие ожидаемых таблиц и колонок;
- `ratings.movie_id -> movies.movie_id`;
- `movie_keywords.movie_id -> movies.movie_id`;
- `movie_keywords.keyword_id -> keywords.keyword_id`;
- `cast_members.movie_id -> movies.movie_id`;
- `crew_members.movie_id -> movies.movie_id`.

Важно: `links.tmdb_id` не имеет FK на `movies.movie_id`, потому что `links_clean.csv` содержит часть TMDB id, которых нет в `movies.csv`.
