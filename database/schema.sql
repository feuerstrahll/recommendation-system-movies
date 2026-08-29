-- ============================================================
-- Movies Dataset PostgreSQL schema
-- Compatible with CSV files produced by:
--   lab-5/database/clean_movies_data.py
--
-- Main processed files:
--   movies.csv
--   ratings.csv
--   links_clean.csv
--   cast_clean.csv
--   crew_clean.csv
--   keywords_clean.csv
--
-- Important:
--   movies.movie_id is the TMDB id.
--   ratings.movielens_id and links.movielens_id are MovieLens ids.
--   links_clean.csv is not filtered to movies.csv, so links.tmdb_id does
--   not have a foreign key to movies.movie_id.
-- ============================================================

CREATE TABLE IF NOT EXISTS movies (
    movie_id                  INTEGER PRIMARY KEY,
    imdb_id                   VARCHAR(20),
    title                     TEXT NOT NULL,
    original_title            TEXT,
    original_language         VARCHAR(10),
    overview                  TEXT,
    tagline                   TEXT,
    status                    VARCHAR(50),
    adult                     BOOLEAN,
    video                     BOOLEAN,
    budget                    NUMERIC(20, 2),
    revenue                   NUMERIC(20, 2),
    runtime                   NUMERIC(6, 1),
    vote_count                NUMERIC(12, 1),
    vote_average              NUMERIC(4, 2),
    popularity                NUMERIC(12, 6),
    release_date              DATE,
    release_year              SMALLINT,

    -- These columns are Python-list strings in the current CSV output,
    -- for example: ['Drama', 'Comedy']. Keep them as TEXT unless the
    -- loader converts them to PostgreSQL arrays or JSONB.
    genre_names               TEXT,
    production_company_names  TEXT,
    production_country_codes  TEXT,
    spoken_language_codes     TEXT,

    poster_path               TEXT,
    homepage                  TEXT,
    belongs_to_collection     TEXT
);

CREATE INDEX IF NOT EXISTS idx_movies_title
    ON movies (title);

CREATE INDEX IF NOT EXISTS idx_movies_imdb_id
    ON movies (imdb_id);

CREATE INDEX IF NOT EXISTS idx_movies_original_language
    ON movies (original_language);

CREATE INDEX IF NOT EXISTS idx_movies_release_year
    ON movies (release_year);

CREATE INDEX IF NOT EXISTS idx_movies_release_date
    ON movies (release_date);

CREATE INDEX IF NOT EXISTS idx_movies_vote_average
    ON movies (vote_average);

CREATE INDEX IF NOT EXISTS idx_movies_popularity
    ON movies (popularity);


CREATE TABLE IF NOT EXISTS ratings (
    user_id       INTEGER NOT NULL,
    movie_id      INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    movielens_id  INTEGER NOT NULL,
    rating        NUMERIC(3, 1) NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    timestamp     TIMESTAMP,

    PRIMARY KEY (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_user_id
    ON ratings (user_id);

CREATE INDEX IF NOT EXISTS idx_ratings_movie_id
    ON ratings (movie_id);

CREATE INDEX IF NOT EXISTS idx_ratings_movielens_id
    ON ratings (movielens_id);

CREATE INDEX IF NOT EXISTS idx_ratings_timestamp
    ON ratings (timestamp);


CREATE TABLE IF NOT EXISTS links (
    movielens_id  INTEGER PRIMARY KEY,
    imdb_id       INTEGER,
    tmdb_id       INTEGER
);

CREATE INDEX IF NOT EXISTS idx_links_imdb_id
    ON links (imdb_id);

CREATE INDEX IF NOT EXISTS idx_links_tmdb_id
    ON links (tmdb_id);


CREATE TABLE IF NOT EXISTS cast_members (
    movie_id      INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    cast_id       INTEGER NOT NULL,
    person_id     INTEGER NOT NULL,
    name          TEXT NOT NULL,
    character     TEXT,
    "order"       INTEGER NOT NULL,
    gender        SMALLINT,
    profile_path  TEXT,

    PRIMARY KEY (movie_id, cast_id, person_id, "order")
);

CREATE INDEX IF NOT EXISTS idx_cast_members_movie_id
    ON cast_members (movie_id);

CREATE INDEX IF NOT EXISTS idx_cast_members_person_id
    ON cast_members (person_id);

CREATE INDEX IF NOT EXISTS idx_cast_members_name
    ON cast_members (name);

CREATE INDEX IF NOT EXISTS idx_cast_members_movie_order
    ON cast_members (movie_id, "order");


CREATE TABLE IF NOT EXISTS crew_members (
    movie_id      INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    person_id     INTEGER NOT NULL,
    name          TEXT NOT NULL,
    department    VARCHAR(100) NOT NULL,
    job           VARCHAR(100) NOT NULL,
    gender        SMALLINT,
    profile_path  TEXT,

    PRIMARY KEY (movie_id, person_id, department, job)
);

CREATE INDEX IF NOT EXISTS idx_crew_members_movie_id
    ON crew_members (movie_id);

CREATE INDEX IF NOT EXISTS idx_crew_members_person_id
    ON crew_members (person_id);

CREATE INDEX IF NOT EXISTS idx_crew_members_department
    ON crew_members (department);

CREATE INDEX IF NOT EXISTS idx_crew_members_job
    ON crew_members (job);


CREATE TABLE IF NOT EXISTS keywords (
    keyword_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_keywords_name
    ON keywords (name);


CREATE TABLE IF NOT EXISTS movie_keywords (
    movie_id    INTEGER NOT NULL REFERENCES movies(movie_id) ON DELETE CASCADE,
    keyword_id  INTEGER NOT NULL REFERENCES keywords(keyword_id) ON DELETE CASCADE,

    PRIMARY KEY (movie_id, keyword_id)
);

CREATE INDEX IF NOT EXISTS idx_movie_keywords_movie_id
    ON movie_keywords (movie_id);

CREATE INDEX IF NOT EXISTS idx_movie_keywords_keyword_id
    ON movie_keywords (keyword_id);
