# Database Architecture

Describes the PostgreSQL schema in `schema.sql` and the processed CSVs
produced by `clean_movies_data.py`.

## Source data

The Movies Dataset:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `ratings.csv`
- `links.csv`

Cleaned output is written to `data/processed`.

## Identifier convention

Every recommendation component keys movies by the TMDB id:

```text
movie_id = TMDB id
```

`ratings.csv` additionally keeps:

```text
movielens_id = original MovieLens movieId
```

`movielens_id` exists only to trace back to the original MovieLens/links
data; it is not used as a join key by the models.

## Tables

### movies

Main movie table.

| Field | Type | Purpose |
|---|---|---|
| `movie_id` | INTEGER, PK | TMDB id |
| `imdb_id` | VARCHAR | IMDB id |
| `title` | TEXT | Title |
| `original_title` | TEXT | Original-language title |
| `original_language` | VARCHAR | Original language |
| `overview` | TEXT | Synopsis |
| `release_date` | DATE | Release date |
| `release_year` | SMALLINT | Release year |
| `budget` | NUMERIC | Budget |
| `revenue` | NUMERIC | Revenue |
| `runtime` | NUMERIC | Runtime |
| `vote_average` | NUMERIC | Average vote |
| `vote_count` | NUMERIC | Vote count |
| `popularity` | NUMERIC | Popularity score |
| `genre_names` | TEXT | Python-list string of genres |
| `production_company_names` | TEXT | Python-list string of companies |
| `production_country_codes` | TEXT | Python-list string of countries |
| `spoken_language_codes` | TEXT | Python-list string of languages |

List-like fields stay `TEXT` because the source CSV stores them as Python
list literals (e.g. `['Drama', 'Comedy']`), not native PostgreSQL arrays.

### ratings

Final user ratings.

| Field | Type | Purpose |
|---|---|---|
| `user_id` | INTEGER | User |
| `movie_id` | INTEGER, FK | TMDB id |
| `movielens_id` | INTEGER | Original MovieLens id |
| `rating` | NUMERIC | Rating, 0.5–5.0 |
| `timestamp` | TIMESTAMP | Rating time |

Primary key: `(user_id, movie_id)`.

### links

Maps MovieLens, IMDB, and TMDB ids.

| Field | Type | Purpose |
|---|---|---|
| `movielens_id` | INTEGER, PK | MovieLens id |
| `imdb_id` | INTEGER | IMDB id, without the `tt` prefix |
| `tmdb_id` | INTEGER | TMDB id |

`tmdb_id` has no FK to `movies.movie_id`: `links_clean.csv` contains ids
that are absent from `movies.csv`.

### cast_members

| Field | Type | Purpose |
|---|---|---|
| `movie_id` | INTEGER, FK | TMDB id |
| `cast_id` | INTEGER | Cast entry id |
| `person_id` | INTEGER | TMDB person id |
| `name` | TEXT | Actor name |
| `character` | TEXT | Role |
| `order` | INTEGER | Billing order |
| `gender` | SMALLINT | Gender code from the source dataset |
| `profile_path` | TEXT | Profile image path |

Primary key: `(movie_id, cast_id, person_id, "order")`.

### crew_members

| Field | Type | Purpose |
|---|---|---|
| `movie_id` | INTEGER, FK | TMDB id |
| `person_id` | INTEGER | TMDB person id |
| `name` | TEXT | Crew member name |
| `department` | VARCHAR | Department |
| `job` | VARCHAR | Job title |
| `gender` | SMALLINT | Gender code from the source dataset |
| `profile_path` | TEXT | Profile image path |

Primary key: `(movie_id, person_id, department, job)`.

### keywords

| Field | Type | Purpose |
|---|---|---|
| `keyword_id` | INTEGER, PK | Keyword id |
| `name` | TEXT | Keyword text |

### movie_keywords

| Field | Type | Purpose |
|---|---|---|
| `movie_id` | INTEGER, FK | TMDB id |
| `keyword_id` | INTEGER, FK | Keyword id |

Primary key: `(movie_id, keyword_id)`.

## Checks

`check_db.py` validates table structure and these relations:

- `ratings.movie_id -> movies.movie_id`;
- `movie_keywords.movie_id -> movies.movie_id`;
- `movie_keywords.keyword_id -> keywords.keyword_id`;
- `cast_members.movie_id -> movies.movie_id`;
- `crew_members.movie_id -> movies.movie_id`.

A CSV-level check for `movie_keywords` is in `data/check.py`.

## Use in recommendation models

Content-based and cold-start models use `movies`: genres, overview,
popularity, and vote data. Collaborative models (LightFM, LightGCN) use
`ratings` with columns `["user_id", "movie_id", "rating"]`.
