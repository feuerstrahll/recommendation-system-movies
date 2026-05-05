-- ============================================================
-- Movies Dataset — схема PostgreSQL
-- Запуск: psql -U <user> -d <db> -f schema.sql
-- ============================================================

-- ─────────────────────────────────────────────
-- Основная таблица фильмов
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS movies (
    id                      INTEGER PRIMARY KEY,
    imdb_id                 VARCHAR(20),
    title                   TEXT NOT NULL,
    original_title          TEXT,
    original_language       VARCHAR(10),
    overview                TEXT,
    tagline                 TEXT,
    status                  VARCHAR(50),
    adult                   BOOLEAN,
    video                   BOOLEAN,
    budget                  NUMERIC(20, 2),
    revenue                 NUMERIC(20, 2),
    runtime                 NUMERIC(6, 1),
    vote_count              INTEGER,
    vote_average            NUMERIC(4, 2),
    popularity              NUMERIC(12, 4),
    release_date            DATE,
    release_year            SMALLINT,
    poster_path             TEXT,
    backdrop_path           TEXT,
    homepage                TEXT,
    belongs_to_collection   TEXT,

    -- Нормализованные данные хранятся в отдельных таблицах (ниже),
    -- но для удобства быстрого поиска дублируем в JSON-колонках.
    genre_names             TEXT[],                  -- массив: {'Drama','Comedy',...}
    production_company_names TEXT[],
    production_country_codes CHAR(2)[],
    spoken_language_codes   CHAR(2)[]
);

CREATE INDEX IF NOT EXISTS idx_movies_release_year  ON movies (release_year);
CREATE INDEX IF NOT EXISTS idx_movies_vote_average  ON movies (vote_average);
CREATE INDEX IF NOT EXISTS idx_movies_popularity    ON movies (popularity);
CREATE INDEX IF NOT EXISTS idx_movies_imdb_id       ON movies (imdb_id);

-- ─────────────────────────────────────────────
-- Жанры (нормализованный справочник)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS genres (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS movie_genres (
    movie_id  INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    genre_id  INTEGER REFERENCES genres(id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, genre_id)
);

-- ─────────────────────────────────────────────
-- Актёры
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cast_members (
    id           SERIAL PRIMARY KEY,
    movie_id     INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    person_id    INTEGER,
    cast_id      INTEGER,
    name         TEXT NOT NULL,
    character    TEXT,
    "order"      SMALLINT,
    gender       SMALLINT,         -- 0=Неизвестно, 1=Женщина, 2=Мужчина
    profile_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_cast_movie_id   ON cast_members (movie_id);
CREATE INDEX IF NOT EXISTS idx_cast_person_id  ON cast_members (person_id);
CREATE INDEX IF NOT EXISTS idx_cast_name       ON cast_members (name);

-- ─────────────────────────────────────────────
-- Съёмочная группа
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS crew_members (
    id           SERIAL PRIMARY KEY,
    movie_id     INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    person_id    INTEGER,
    name         TEXT NOT NULL,
    department   VARCHAR(100),
    job          VARCHAR(100),
    gender       SMALLINT,
    profile_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_crew_movie_id   ON crew_members (movie_id);
CREATE INDEX IF NOT EXISTS idx_crew_person_id  ON crew_members (person_id);
CREATE INDEX IF NOT EXISTS idx_crew_department ON crew_members (department);

-- ─────────────────────────────────────────────
-- Ключевые слова
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keywords (
    id   INTEGER PRIMARY KEY,     -- keyword_id из датасета
    name VARCHAR(200) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS movie_keywords (
    movie_id    INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    keyword_id  INTEGER REFERENCES keywords(id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, keyword_id)
);

-- ─────────────────────────────────────────────
-- Оценки пользователей
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ratings (
    user_id    INTEGER NOT NULL,
    movie_id   INTEGER NOT NULL,   -- MovieLens movieId (не tmdbId)
    rating     NUMERIC(3, 1) NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
    rated_at   TIMESTAMP,          -- преобразуется из Unix-timestamp при загрузке
    PRIMARY KEY (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings (movie_id);
CREATE INDEX IF NOT EXISTS idx_ratings_user_id  ON ratings (user_id);

-- ─────────────────────────────────────────────
-- Связи MovieLens ↔ TMDB / IMDB
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS links (
    movielens_id BIGINT  PRIMARY KEY,
    imdb_id      BIGINT ,
    tmdb_id      BIGINT  REFERENCES movies(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_links_tmdb_id ON links (tmdb_id);
CREATE INDEX IF NOT EXISTS idx_links_imdb_id ON links (imdb_id);
