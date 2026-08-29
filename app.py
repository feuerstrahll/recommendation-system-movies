"""
Movie recommendation system — Streamlit front end.

Local-only "account" system (no real backend/DB for auth — profiles are
stored as JSON files under .local_profiles/). Three sections:
  - Личный кабинет   — login/signup, questionnaire, your ratings & favorites
  - Рекомендательные системы — content-based, collaborative (SVD), hybrid,
    and LightFM / LightGCN (from models/) as an additional strategy
  - Поиск фильмов    — search the catalog, rate movies, add to favorites

Data: data/processed/movies.csv, data/processed/ratings.csv
(movie_id = TMDB id; see database/database_architecture.md).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
PROFILES_DIR = ROOT / ".local_profiles"
PROFILES_DIR.mkdir(exist_ok=True)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
RATING_MIN, RATING_MAX, RATING_STEP = 0.5, 5.0, 0.5

st.set_page_config(page_title="Умный подбор фильмов", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Загружаем каталог фильмов...")
def load_movies():
    path = DATA_DIR / "movies.csv"
    if not path.exists():
        return pd.DataFrame(columns=["movie_id", "title", "genre_names", "release_year"])
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["movie_id", "title"]).copy()
    df["movie_id"] = df["movie_id"].astype(int)
    if "genre_names" not in df.columns:
        df["genre_names"] = ""
    df["genre_names"] = df["genre_names"].fillna("")
    return df


@st.cache_data(show_spinner="Загружаем оценки зрителей...")
def load_ratings():
    path = DATA_DIR / "ratings.csv"
    if not path.exists():
        return pd.DataFrame(columns=["user_id", "movie_id", "rating"])
    df = pd.read_csv(path, usecols=["user_id", "movie_id", "rating"])
    df["movie_id"] = df["movie_id"].astype(int)
    return df


def poster_url(row):
    path = row.get("poster_path")
    if isinstance(path, str) and path.strip():
        return f"{TMDB_IMAGE_BASE}{path}"
    return None


def genre_list(genre_str):
    if not isinstance(genre_str, str) or not genre_str.strip():
        return []
    # genre_names is stored either as "Action|Sci-Fi" or a Python-list-literal string
    cleaned = genre_str.strip("[]").replace("'", "").replace('"', "")
    parts = re.split(r"[|,]", cleaned)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Local "account" storage (JSON files — no real auth, matches project scope)
# ---------------------------------------------------------------------------

def _profile_path(email):
    safe = re.sub(r"[^a-zA-Z0-9_.@-]", "_", email)
    return PROFILES_DIR / f"{safe}.json"


def load_profile(email):
    path = _profile_path(email)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "email": email,
        "password": None,
        "favorite_genres": [],
        "age_preference": "any",
        "ratings": {},   # {movie_id: rating}
        "favorites": [], # [movie_id]
    }


def save_profile(profile):
    path = _profile_path(profile["email"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


GUEST_PROFILE = {
    "email": None,
    "password": None,
    "favorite_genres": [],
    "age_preference": "any",
    "ratings": {},
    "favorites": [],
}


def current_profile():
    return st.session_state.get("profile", GUEST_PROFILE)


def set_current_profile(profile):
    st.session_state["profile"] = profile


# ---------------------------------------------------------------------------
# Recommendation strategies
# ---------------------------------------------------------------------------

def recommend_content_based(movies, seed_movie_id, n=12):
    """Similar movies by shared genres to a chosen seed movie."""
    seed = movies[movies["movie_id"] == seed_movie_id]
    if seed.empty:
        return movies.head(n)
    seed_genres = set(genre_list(seed.iloc[0]["genre_names"]))
    if not seed_genres:
        return movies[movies["movie_id"] != seed_movie_id].head(n)

    def overlap_score(genre_str):
        return len(seed_genres & set(genre_list(genre_str)))

    scored = movies[movies["movie_id"] != seed_movie_id].copy()
    scored["_score"] = scored["genre_names"].apply(overlap_score)
    scored = scored[scored["_score"] > 0].sort_values("_score", ascending=False)
    return scored.head(n)


@st.cache_resource(show_spinner="Обучаем коллаборативную модель (SVD)...")
def fit_svd_model(ratings_hash_key):
    """
    SVD via models/svd_model.py's train_svd(), so the app uses the same
    factorization as the lab's SVD baseline (evaluation/compare_models.py)
    instead of a second, differently-tuned implementation.

    Note: unlike svd_model.py's own evaluation pipeline (which trains on
    implicit rating>=4.0 interactions), here we keep the app's explicit
    ratings — the app needs item_factors trained on the full catalog, then
    projects an ad-hoc user vector from the current profile's own ratings
    (see recommend_collaborative), including profiles/users that were never
    part of ratings.csv at all.
    """
    import sys
    sys.path.insert(0, str(ROOT / "models"))
    from svd_model import train_svd

    ratings = load_ratings()
    if ratings.empty:
        return None

    train_df = ratings[["user_id", "movie_id"]].drop_duplicates()
    user_factors, item_factors, user_map, item_map, item_inv = train_svd(train_df)
    return {
        "user_map": user_map,
        "item_map": item_map,
        "item_inv": item_inv,
        "user_factors": user_factors,
        "item_factors": item_factors,
    }


def recommend_collaborative(movies, user_ratings: dict, n=12):
    """
    Collaborative-style recommendation for the current profile: build a
    lightweight user vector from this profile's own ratings, projected onto
    the item-factor space fit on the full ratings.csv (approximates SVD
    without needing this exact user_id to be present in the training data).
    """
    model = fit_svd_model("static")
    if model is None or not user_ratings:
        return movies.sample(min(n, len(movies)))

    item_map, item_inv = model["item_map"], model["item_inv"]
    item_factors = model["item_factors"]

    known = [(item_map[int(mid)], r) for mid, r in user_ratings.items()
             if int(mid) in item_map]
    if not known:
        return movies.sample(min(n, len(movies)))

    idxs = np.array([i for i, _ in known])
    weights = np.array([r for _, r in known], dtype=np.float32)
    user_vec = (item_factors[idxs] * weights[:, None]).sum(axis=0) / weights.sum()

    scores = item_factors @ user_vec
    rated_idxs = set(idxs.tolist())
    order = np.argsort(scores)[::-1]

    rec_ids = []
    for idx in order:
        if idx in rated_idxs:
            continue
        rec_ids.append(item_inv[idx])
        if len(rec_ids) >= n:
            break

    return movies[movies["movie_id"].isin(rec_ids)]


def recommend_hybrid(movies, seed_movie_id, user_ratings, alpha=0.5, n=12):
    """Blend content-based (genre overlap) and collaborative (SVD) scores."""
    content = recommend_content_based(movies, seed_movie_id, n=n * 3)
    collab = recommend_collaborative(movies, user_ratings, n=n * 3)

    content_ids = list(content["movie_id"])
    collab_ids = list(collab["movie_id"])

    content_rank = {mid: 1.0 - i / max(len(content_ids), 1) for i, mid in enumerate(content_ids)}
    collab_rank = {mid: 1.0 - i / max(len(collab_ids), 1) for i, mid in enumerate(collab_ids)}

    all_ids = set(content_ids) | set(collab_ids)
    scored = [
        (mid, alpha * collab_rank.get(mid, 0) + (1 - alpha) * content_rank.get(mid, 0))
        for mid in all_ids
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [mid for mid, _ in scored[:n]]
    return movies[movies["movie_id"].isin(top_ids)]


@st.cache_resource(show_spinner="Обучаем LightFM/LightGCN (может занять минуту)...")
def fit_neural_models():
    """Train LightFM and LightGCN once per session, on the shared ratings data."""
    import sys
    sys.path.insert(0, str(ROOT / "models"))
    result = {"lightfm": None, "lightgcn": None, "error": None}

    ratings = load_ratings()
    if ratings.empty:
        result["error"] = "Нет данных в data/processed/ratings.csv"
        return result

    try:
        from lightfm import LightFM
        from lightfm.data import Dataset as LightFMDataset
        from scipy.sparse import coo_matrix

        dataset = LightFMDataset()
        dataset.fit(users=ratings["user_id"].unique(), items=ratings["movie_id"].unique())
        user_id_map, _, item_id_map, _ = dataset.mapping()
        item_inv = {v: k for k, v in item_id_map.items()}

        u_idx = ratings["user_id"].map(user_id_map).values.astype(np.int32)
        i_idx = ratings["movie_id"].map(item_id_map).values.astype(np.int32)
        w = ratings["rating"].values.astype(np.float32)
        shape = (len(user_id_map), len(item_id_map))
        interactions = coo_matrix((w, (u_idx, i_idx)), shape=shape)

        model = LightFM(no_components=32, loss="warp", random_state=42)
        model.fit(interactions, epochs=5, num_threads=1)

        result["lightfm"] = {
            "model": model,
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "item_inv": item_inv,
        }
    except Exception as exc:  # pragma: no cover - optional dependency path
        result["error"] = f"LightFM недоступен: {exc}"

    try:
        import torch
        from lightgcn_model import LightGCN, prepare_lightgcn_data, train_lightgcn

        gcn_ratings = ratings.rename(columns={"user_id": "userId", "movie_id": "movieId"})
        gcn_ratings = gcn_ratings[["userId", "movieId"]].drop_duplicates()

        edge_index, n_users, n_items, user_inv, item_inv, gt_dict, user_map, item_map = \
            prepare_lightgcn_data(gcn_ratings, min_user_interactions=2)

        model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=32, n_layers=2)
        user_embs, item_embs, _ = train_lightgcn(
            model, edge_index, gt_dict, n_users, n_items, user_map, item_map,
            epochs=5, batch_size=2048,
        )

        result["lightgcn"] = {
            "user_embs": user_embs.detach().cpu(),
            "item_embs": item_embs.detach().cpu(),
            "user_map": user_map,
            "item_map": item_map,
            "item_inv": item_inv,
        }
    except Exception as exc:  # pragma: no cover - optional dependency path
        note = f"LightGCN недоступен: {exc}"
        result["error"] = f"{result['error']}; {note}" if result["error"] else note

    return result


def recommend_neural(movies, user_id, engine_key, n=12):
    bundle = fit_neural_models()
    engine = bundle.get(engine_key)
    if engine is None:
        return None, bundle.get("error", "Модель недоступна.")

    if engine_key == "lightfm":
        user_id_map, item_id_map, item_inv = (
            engine["user_id_map"], engine["item_id_map"], engine["item_inv"]
        )
        if user_id not in user_id_map:
            return None, "Этот пользователь не встречался в обучающих данных LightFM."
        u_idx = user_id_map[user_id]
        all_items = np.arange(len(item_id_map), dtype=np.int32)
        scores = engine["model"].predict(np.full(len(all_items), u_idx, dtype=np.int32), all_items)
        top = np.argsort(scores)[::-1][:n]
        rec_ids = [item_inv[i] for i in top]
        return movies[movies["movie_id"].isin(rec_ids)], None

    if engine_key == "lightgcn":
        user_map, item_inv = engine["user_map"], engine["item_inv"]
        if user_id not in user_map:
            return None, "Этот пользователь не встречался в обучающих данных LightGCN."
        u_idx = user_map[user_id]
        u_emb = engine["user_embs"][u_idx]
        scores = u_emb @ engine["item_embs"].T
        top = np.argsort(scores.numpy())[::-1][:n]
        rec_ids = [item_inv[i] for i in top]
        return movies[movies["movie_id"].isin(rec_ids)], None

    return None, "Неизвестная модель."


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_movie_grid(df, movies, key_prefix, editable=True):
    if df is None or df.empty:
        st.info("Ничего не найдено.")
        return

    profile = current_profile()
    cols = st.columns(5)
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        with cols[i % 5]:
            url = poster_url(row)
            if url:
                st.image(url, use_container_width=True)
            else:
                st.markdown(
                    "<div style='background:#eee;height:220px;display:flex;"
                    "align-items:center;justify-content:center;color:#999;'>Нет фото</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**{row['title']}**")
            genres = ", ".join(genre_list(row.get("genre_names", "")))
            if genres:
                st.caption(genres)

            movie_id = int(row["movie_id"])
            fav_key = f"{key_prefix}_fav_{movie_id}"
            is_fav = movie_id in profile["favorites"]
            if st.button("★ В избранном" if is_fav else "В избранное", key=fav_key):
                if is_fav:
                    profile["favorites"].remove(movie_id)
                else:
                    profile["favorites"].append(movie_id)
                if profile["email"]:
                    save_profile(profile)
                set_current_profile(profile)
                st.rerun()

            if editable:
                rate_key = f"{key_prefix}_rate_{movie_id}"
                current = profile["ratings"].get(str(movie_id))
                options = ["—"] + [str(r) for r in np.arange(RATING_MIN, RATING_MAX + RATING_STEP, RATING_STEP)]
                default_idx = options.index(str(current)) if current and str(current) in options else 0
                chosen = st.selectbox("Оценка (0.5–5.0)", options, index=default_idx, key=rate_key)
                if chosen != "—":
                    profile["ratings"][str(movie_id)] = float(chosen)
                    if profile["email"]:
                        save_profile(profile)
                    set_current_profile(profile)


def sidebar_nav():
    st.sidebar.markdown("### ≡ Разделы")
    st.sidebar.caption("Куда перейти")
    for label in ["Личный кабинет", "Рекомендательные системы", "Поиск фильмов"]:
        if st.sidebar.button(label, key=f"nav_{label}", use_container_width=True,
                              type="primary" if st.session_state.get("page") == label else "secondary"):
            st.session_state["page"] = label
            st.rerun()
    return st.session_state.get("page", "Личный кабинет")


def header_banner(movies, ratings):
    profile = current_profile()
    who = profile["email"] or "гость"
    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,#f3e8ff,#e0e7ff);
                    padding:1.2rem 1.5rem;border-radius:0.6rem;margin-bottom:1rem;">
            <h2 style="margin:0;">Умный подбор фильмов</h2>
            <p style="margin:0;color:#444;">Текущий профиль: <b>{who}</b> —
               здесь хранятся ваши жанры, оценки и избранное</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Фильмов в каталоге", f"{len(movies):,}".replace(",", " "))
    c2.metric("Оценок в данных", f"{len(ratings):,}".replace(",", " "))
    c3.metric("Зрителей в базе", f"{ratings['user_id'].nunique():,}".replace(",", " ") if not ratings.empty else 0)
    c4.metric("В избранном у вас", len(profile["favorites"]))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

ALL_GENRES_CACHE_KEY = "_all_genres"


def page_account(movies):
    profile = current_profile()

    st.markdown("## Личный кабинет")
    st.write("Здесь вход/выход, анкета нового пользователя, ваши оценки и избранные фильмы.")

    if profile["email"]:
        st.success(f"Вы вошли как: **{profile['email']}**")
        if st.button("Выйти из аккаунта"):
            set_current_profile(GUEST_PROFILE)
            st.rerun()
    else:
        st.markdown("### Вход по почте")
        st.caption("Локальный аналог авторизации: данные аккаунта сохраняются только в файлах проекта.")
        # action is outside the form so switching it immediately reveals/hides
        # the "repeat password" field — st.form freezes widget values until submit,
        # so a radio placed inside it can't drive conditional fields in the same run.
        action = st.radio("Действие", ["Войти", "Создать аккаунт"], horizontal=True)
        with st.form("login_form"):
            email = st.text_input("Почта", placeholder="newname@example.com")
            password = st.text_input("Пароль", type="password")
            password2 = st.text_input("Повторите пароль", type="password") if action == "Создать аккаунт" else None
            submitted = st.form_submit_button(
                "Создать аккаунт" if action == "Создать аккаунт" else "Войти"
            )

        if submitted:
            if not email or "@" not in email:
                st.error("Введите корректную почту.")
            elif not password:
                st.error("Введите пароль.")
            elif action == "Создать аккаунт" and password != password2:
                st.error("Пароли не совпадают.")
            else:
                loaded = load_profile(email)
                if action == "Создать аккаунт":
                    if loaded["password"] is not None:
                        st.error("Аккаунт с такой почтой уже существует.")
                        return
                    loaded["password"] = password
                    save_profile(loaded)
                    set_current_profile(loaded)
                    st.rerun()
                else:
                    if loaded["password"] != password:
                        st.error("Неверная почта или пароль.")
                    else:
                        set_current_profile(loaded)
                        st.rerun()

        st.info("Сейчас вы в гостевом режиме. Для анкеты и персональных рекомендаций выполните вход.")
        return

    with st.expander("Изменить анкету", expanded=not profile["favorite_genres"]):
        st.markdown("### Анкета нового пользователя")
        st.caption("Анкета показывается при создании пользователя. По ней строится первая контентная подборка.")

        all_genres = sorted({g for gs in movies["genre_names"] for g in genre_list(gs)})
        genres = st.multiselect("Любимые жанры", all_genres, default=profile["favorite_genres"])
        age_pref = st.radio(
            "Какие фильмы вам ближе по годам?",
            ["any", "classic", "modern"],
            format_func=lambda x: {"any": "Любые годы", "classic": "Больше классики (до 2000 года)",
                                    "modern": "Больше нового (с 2000 года)"}[x],
            index=["any", "classic", "modern"].index(profile.get("age_preference", "any")),
        )
        if st.button("Сохранить анкету"):
            profile["favorite_genres"] = genres
            profile["age_preference"] = age_pref
            save_profile(profile)
            set_current_profile(profile)
            st.success("Анкета сохранена.")

    st.markdown("### Ваши оценки (0.5–5.0)")
    if not profile["ratings"]:
        st.info("Пока нет оценок — поставьте их в разделе **Поиск фильмов**.")
    else:
        rated_ids = [int(k) for k in profile["ratings"].keys()]
        rated_movies = movies[movies["movie_id"].isin(rated_ids)].copy()
        rated_movies["Оценка"] = rated_movies["movie_id"].map(
            lambda mid: profile["ratings"].get(str(mid))
        )
        table = rated_movies[["movie_id", "title", "genre_names", "Оценка"]].rename(
            columns={"movie_id": "ID", "title": "Название", "genre_names": "Жанры"}
        )
        edited = st.data_editor(table, use_container_width=True, num_rows="fixed", key="ratings_editor")
        if st.button("Сохранить изменения оценок"):
            for _, row in edited.iterrows():
                if pd.notna(row["Оценка"]):
                    profile["ratings"][str(int(row["ID"]))] = float(row["Оценка"])
            save_profile(profile)
            set_current_profile(profile)
            st.success(f"Сохранено оценок: {len(profile['ratings'])} — профиль «{profile['email']}».")

    if not profile["favorites"]:
        st.info("В избранном пока пусто. Добавьте фильмы в разделе **Рекомендательные системы** или "
                "**Поиск фильмов**, кнопка «В избранное».")
    else:
        st.markdown("### Избранное")
        fav_movies = movies[movies["movie_id"].isin(profile["favorites"])]
        render_movie_grid(fav_movies, movies, key_prefix="acc_fav", editable=False)


def page_recommendations(movies, ratings):
    profile = current_profile()
    st.markdown("## Рекомендательные системы")

    n = st.slider("Сколько фильмов показать", 4, 30, 12)

    n_user_ratings = len(profile["ratings"])
    default_method = "collab" if n_user_ratings >= 15 else "content"
    st.caption(f"Доступны все методы. По умолчанию при 15+ оценках выбрана коллаборативная (SVD).")

    method_labels = {
        "content": "По похожести на выбранный фильм",
        "collab": "По оценкам похожих зрителей",
        "hybrid": "Смешанный способ",
        "lightfm": "LightFM (исследование)",
        "lightgcn": "LightGCN (исследование)",
    }
    method = st.radio(
        "Выберите рекомендательную систему",
        list(method_labels.keys()),
        format_func=lambda k: method_labels[k],
        index=list(method_labels.keys()).index(default_method),
        horizontal=True,
    )

    seed_movie_id = None
    if method in ("content", "hybrid"):
        st.markdown("### Фильм-ориентир для контентного подбора")
        st.caption("Найдите фильм по названию — покажем похожие по жанрам.")
        query = st.text_input("Найти фильм по названию", key="seed_search")
        if query:
            matches = movies[movies["title"].str.contains(query, case=False, na=False)]
            if not matches.empty:
                options = {
                    f"{r['title']} ({int(r.get('release_year', 0)) or '?'}) · ID {r['movie_id']}": r["movie_id"]
                    for _, r in matches.head(20).iterrows()
                }
                choice = st.selectbox("Выберите фильм из результатов", list(options.keys()))
                seed_movie_id = options[choice]
                st.info(f"Сейчас выбран: **{choice}**")
            else:
                st.warning("Совпадений не найдено.")

    user_ratings = profile["ratings"]

    if method == "content":
        if seed_movie_id is None:
            st.info("Выберите фильм-ориентир выше, чтобы построить контентную подборку.")
            return
        result = recommend_content_based(movies, seed_movie_id, n=n)
        st.caption(f"Сейчас активна: Контентная система — похожие фильмы по жанрам. "
                   f"Оценок у вас: {n_user_ratings} · В подборке: {len(result)} фильмов")
    elif method == "collab":
        result = recommend_collaborative(movies, user_ratings, n=n)
        st.caption(f"Сейчас активна: Коллаборативная система (SVD, 15+ оценок). "
                   f"Оценок у вас: {n_user_ratings} · В подборке: {len(result)} фильмов")
    elif method == "hybrid":
        if seed_movie_id is None:
            st.info("Выберите фильм-ориентир выше, чтобы построить гибридную подборку.")
            return
        alpha = st.slider("Баланс: контент ↔ коллаборативная", 0.0, 1.0, 0.5)
        result = recommend_hybrid(movies, seed_movie_id, user_ratings, alpha=alpha, n=n)
        st.caption(f"Сейчас активна: Гибридная система (контент + SVD). "
                   f"Оценок у вас: {n_user_ratings} · В подборке: {len(result)} фильмов")
    else:
        if not profile["email"]:
            st.warning("LightFM/LightGCN используют user_id из обучающих данных ratings.csv — "
                       "войдите в аккаунт с email, который совпадает с известным user_id, "
                       "либо оцените фильмы, чтобы использовать другие методы.")
            return
        try:
            uid = int(re.sub(r"\D", "", profile["email"]) or "-1")
        except ValueError:
            uid = -1
        result, error = recommend_neural(movies, uid, method, n=n)
        if error:
            st.error(error)
            return
        st.caption(f"Сейчас активна: {method_labels[method]}. В подборке: {len(result)} фильмов")

    tab1, tab2 = st.tabs(["Карточки", "Таблица"])
    with tab1:
        render_movie_grid(result, movies, key_prefix=f"rec_{method}")
    with tab2:
        cols = ["movie_id", "title", "genre_names"]
        st.dataframe(result[cols].rename(
            columns={"movie_id": "ID", "title": "Название", "genre_names": "Жанры"}
        ), use_container_width=True)


def page_search(movies):
    st.markdown("## Поиск и оценки")
    st.caption("Оценки по шкале 0.5–5.0 (шаг 0.5). Они сохраняются в выбранном профиле и учитываются "
               "в «Анкете и подборе», а при необходимости — в разделе рекомендаций.")

    query = st.text_input("Найти фильм по названию")
    limit = st.slider("Сколько показать в результатах", 5, 50, 25)

    if query:
        results = movies[movies["title"].str.contains(query, case=False, na=False)].head(limit)
    else:
        results = movies.head(limit)

    st.write(f"Найдено: {len(results)}")
    st.markdown("### Карточки с постерами")
    st.caption("У каждого фильма: **В избранное** и **Оценка** — рядом с постером.")
    render_movie_grid(results, movies, key_prefix="search")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    movies = load_movies()
    ratings = load_ratings()

    if movies.empty:
        st.error(
            "data/processed/movies.csv не найден. Подтяните данные "
            "(`git lfs pull`) или запустите `python database/clean_movies_data.py`."
        )
        return

    page = sidebar_nav()
    header_banner(movies, ratings)

    if page == "Личный кабинет":
        page_account(movies)
    elif page == "Рекомендательные системы":
        page_recommendations(movies, ratings)
    else:
        page_search(movies)


if __name__ == "__main__":
    main()
