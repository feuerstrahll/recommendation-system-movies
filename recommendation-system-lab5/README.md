# Movie Recommendation System
подготовка датасета фильмов, схема базы данных и несколько подходов к рекомендациям.

## Структура

- `data/raw/` - исходные CSV из The Movies Dataset.
- `data/processed/` - актуальные очищенные данные.
- `database/` - `clean_movies_data.py`, `schema.sql`, `check_db.py`, ER-диаграмма и SQLite-снимок.
- `recommender/` - cold-start и hybrid recommender.
- `evaluation/` - метрики.
- `report/` - отчетные материалы.
- `demo.py` - старый демонстрационный entry point, может требовать обновления под текущие колонки.

## Актуальный контракт данных

Основной ключ фильмов в проекте:

```text
movies.movie_id = TMDB id
ratings.movie_id = TMDB id
ratings.movielens_id = исходный MovieLens id
```

Основные processed-файлы:

- `movies.csv`
- `ratings.csv`
- `movies_clean.csv`
- `ratings_clean.csv`
- `links_clean.csv`
- `keywords_clean.csv`
- `cast_clean.csv`
- `crew_clean.csv`

## Подготовка данных

Из папки `lab-5/recommendation-system-lab5`:

```powershell
python database/clean_movies_data.py
```

По умолчанию скрипт читает `data/raw` и пишет `data/processed`.

Можно явно указать пути:

```powershell
python database/clean_movies_data.py --input data/raw --output data/processed
```

## Проверка данных

Проверка внешних ключей `movie_keywords` на CSV:

```powershell
python data/check.py
```

Проверка PostgreSQL-схемы и ссылочной целостности:

```powershell
python database/check_db.py --db movies_db --user postgres --password secret
```

## Рекомендации

- `recommender/questionnaire.py` - cold-start для нового пользователя.
- `recommender/hybrid.py` - гибрид SVD + FAISS/content-based.
- `recommender/content_based&colaborative.ipynb` - исследовательский ноутбук.

SVD-модель должна быть обучена на текущих колонках:

```python
["user_id", "movie_id", "rating"]
```

