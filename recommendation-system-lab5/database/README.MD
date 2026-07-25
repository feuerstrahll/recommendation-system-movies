# Data Preparation and Database Notes

This page summarizes the data information used in the presentation. Database
files can differ between branches, so paths and schemas should be checked
against the branch being run.

## Recorded dataset scale

| Movies | Ratings | MovieLens users | Features per movie |
|---:|---:|---:|---:|
| 45K+ | 270K+ | 671 | 20+ |

The project combines MovieLens interactions with The Movies Dataset (TMDB).
Documented processed files include:

- `movies.csv`;
- `ratings.csv`;
- `movies_clean.csv`;
- `ratings_clean.csv`;
- `links_clean.csv`;
- `keywords_clean.csv`;
- `cast_clean.csv`;
- `crew_clean.csv`.

## Identifier contract

The documented project convention is:

```text
movies.movie_id     = TMDB id
ratings.movie_id    = TMDB id
ratings.movielens_id = original MovieLens id, retained for mapping
```

Verify the actual column names before combining files from different branches.
The cold-start module in `feature/dev` may expect camel-case names such as
`movieId`, while other project files use snake-case names.

## Data checks

Before training or evaluating a model:

1. confirm that movie ids use the same identifier system;
2. remove or document unresolved identifier mappings;
3. check duplicate movies and ratings;
4. record filtering and missing-value rules;
5. verify that train/test splitting does not leak user interactions.

References to PostgreSQL, SQLite, indexes, or partitioning in the presentation
describe project storage choices or target design. They should not be read as a
deployment or scalability guarantee.
