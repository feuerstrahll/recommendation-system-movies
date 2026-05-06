# Database Architecture

Документ описывает актуальную PostgreSQL-схему из `schema.sql` и processed CSV, которые создает `clean_movies_data.py`.

## Источник данных

Используется The Movies Dataset:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `ratings.csv`
- `links.csv`

После очистки данные сохраняются в `data/processed`.

## Главный принцип идентификаторов

В актуальной версии проекта все рекомендательные части используют единый ключ фильма:

```text
movie_id = TMDB id
```

`ratings.csv` дополнительно хранит:

```text
movielens_id = исходный MovieLens movieId
```

`movielens_id` нужен для трассировки к исходному MovieLens/links, но не является основным ключом рекомендаций.

## Таблицы

### movies

Главная таблица фильмов.

Ключевые поля:

| Поле | Тип | Назначение |
|---|---|---|
| `movie_id` | INTEGER, PK | TMDB id фильма |
| `imdb_id` | VARCHAR | IMDB id |
| `title` | TEXT | Название |
| `original_title` | TEXT | Оригинальное название |
| `original_language` | VARCHAR | Язык оригинала |
| `overview` | TEXT | Описание |
| `release_date` | DATE | Дата выхода |
| `release_year` | SMALLINT | Год выхода |
| `budget` | NUMERIC | Бюджет |
| `revenue` | NUMERIC | Сборы |
| `runtime` | NUMERIC | Длительность |
| `vote_average` | NUMERIC | Средняя оценка |
| `vote_count` | NUMERIC | Количество голосов |
| `popularity` | NUMERIC | Популярность |
| `genre_names` | TEXT | Строка Python-list с жанрами |
| `production_company_names` | TEXT | Строка Python-list с компаниями |
| `production_country_codes` | TEXT | Строка Python-list со странами |
| `spoken_language_codes` | TEXT | Строка Python-list с языками |

Поля со списками оставлены как `TEXT`, потому что текущий CSV хранит значения в виде строк `['Drama', 'Comedy']`, а не PostgreSQL-массивов.

### ratings

Финальные пользовательские оценки.

| Поле | Тип | Назначение |
|---|---|---|
| `user_id` | INTEGER | Пользователь |
| `movie_id` | INTEGER, FK | TMDB id фильма |
| `movielens_id` | INTEGER | Исходный MovieLens id |
| `rating` | NUMERIC | Оценка от 0.5 до 5.0 |
| `timestamp` | TIMESTAMP | Время оценки |

Первичный ключ: `(user_id, movie_id)`.

### links

Связь между MovieLens, IMDB и TMDB.

| Поле | Тип | Назначение |
|---|---|---|
| `movielens_id` | INTEGER, PK | MovieLens id |
| `imdb_id` | INTEGER | IMDB id без префикса `tt` |
| `tmdb_id` | INTEGER | TMDB id |

`tmdb_id` не имеет FK на `movies.movie_id`, потому что в `links_clean.csv` есть id, отсутствующие в `movies.csv`.

### cast_members

Актеры и роли.

| Поле | Тип | Назначение |
|---|---|---|
| `movie_id` | INTEGER, FK | TMDB id фильма |
| `cast_id` | INTEGER | id записи cast |
| `person_id` | INTEGER | TMDB id персоны |
| `name` | TEXT | Имя |
| `character` | TEXT | Роль |
| `order` | INTEGER | Порядок в cast |
| `gender` | SMALLINT | Код пола из датасета |
| `profile_path` | TEXT | Путь к профилю |

Первичный ключ: `(movie_id, cast_id, person_id, "order")`.

### crew_members

Съемочная группа.

| Поле | Тип | Назначение |
|---|---|---|
| `movie_id` | INTEGER, FK | TMDB id фильма |
| `person_id` | INTEGER | TMDB id персоны |
| `name` | TEXT | Имя |
| `department` | VARCHAR | Департамент |
| `job` | VARCHAR | Должность |
| `gender` | SMALLINT | Код пола из датасета |
| `profile_path` | TEXT | Путь к профилю |

Первичный ключ: `(movie_id, person_id, department, job)`.

### keywords

Справочник ключевых слов.

| Поле | Тип | Назначение |
|---|---|---|
| `keyword_id` | INTEGER, PK | id ключевого слова |
| `name` | TEXT | Текст ключевого слова |

### movie_keywords

Связь фильмов и ключевых слов.

| Поле | Тип | Назначение |
|---|---|---|
| `movie_id` | INTEGER, FK | TMDB id фильма |
| `keyword_id` | INTEGER, FK | id ключевого слова |

Первичный ключ: `(movie_id, keyword_id)`.

## Проверки

`check_db.py` проверяет структуру таблиц и ключевые связи:

- `ratings.movie_id -> movies.movie_id`;
- `movie_keywords.movie_id -> movies.movie_id`;
- `movie_keywords.keyword_id -> keywords.keyword_id`;
- `cast_members.movie_id -> movies.movie_id`;
- `crew_members.movie_id -> movies.movie_id`.

CSV-проверка для `movie_keywords` находится в `data/check.py`.

## Использование в рекомендациях

Content-based и cold-start используют `movies`, жанры, описания, популярность и оценки.

Collaborative/SVD использует `ratings` с колонками:

```python
["user_id", "movie_id", "rating"]
```

Hybrid объединяет content-based score и SVD score по `movie_id`.
