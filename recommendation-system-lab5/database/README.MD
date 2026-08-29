# Database

Tooling to build and validate the recommendation system's database.

## Files

- `clean_movies_data.py` — cleans The Movies Dataset and builds the final CSVs.
- `schema.sql` — PostgreSQL schema matching those CSVs.
- `check_db.py` — validates table structure and referential integrity in PostgreSQL.
- `database_architecture.md` — table and relationship reference.
- `er_diagram.mmd` — Mermaid ER diagram.
- `recommender.db` — SQLite snapshot for local experiments (also the target of `etl/etl_pipeline.py`).

## Cleaning the data

From `recommendation-system-lab5/`:

```powershell
python database/clean_movies_data.py
```

Defaults: reads from `data/raw`, writes to `data/processed`. Paths can be overridden:

```powershell
python database/clean_movies_data.py --input data/raw --output data/processed
```

Expected source files:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `ratings.csv`
- `links.csv`

## Output CSVs

- `movies.csv` — final movie table, keyed by `movie_id` (TMDB id).
- `ratings.csv` — final ratings, `movie_id` already mapped to TMDB id.
- `movies_clean.csv`, `ratings_clean.csv`, `links_clean.csv`
- `keywords_clean.csv`, `cast_clean.csv`, `crew_clean.csv`

`ratings.csv` also keeps `movielens_id` (the original MovieLens id) for
traceability, but `movie_id` is the key every model uses.

## PostgreSQL

Apply the schema:

```powershell
psql -U postgres -d movies_db -f database/schema.sql
```

Validate:

```powershell
python database/check_db.py --db movies_db --user postgres --password secret
```

`check_db.py` checks:

- expected tables and columns exist;
- `ratings.movie_id -> movies.movie_id`;
- `movie_keywords.movie_id -> movies.movie_id`;
- `movie_keywords.keyword_id -> keywords.keyword_id`;
- `cast_members.movie_id -> movies.movie_id`;
- `crew_members.movie_id -> movies.movie_id`.

Note: `links.tmdb_id` has no FK to `movies.movie_id`, because
`links_clean.csv` contains TMDB ids that are not present in `movies.csv`.
