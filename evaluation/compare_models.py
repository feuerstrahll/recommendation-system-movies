"""
Unified evaluation of all 5 recommendation models.

Models:
  1. Content-Based (TF-IDF) – TF-IDF on genres/title, cosine similarity
  2. SVD                    – matrix factorization baseline (scipy truncated SVD)
  3. LightFM Collab         – factorization machines, WARP loss, interaction data only
  4. LightFM Hybrid         – WARP loss + TF-IDF item features
  5. LightGCN               – graph convolutional network, BPR loss

Note: "Content-Based (TF-IDF)" here is a lightweight TF-IDF + cosine
similarity baseline, not the more sophisticated SentenceTransformer +
FAISS content engine in models/content_based.py. That module is a
separate, more advanced content-based implementation used elsewhere in the
project (e.g. cold-start flows); it is not wired into this benchmark, so
its numbers are not reflected in the results table below.

All collaborative models (SVD, LightFM, LightGCN) are fit on the exact same
train_df, produced by one shared preprocessing pipeline (load_and_split):
  - Binarization: rating >= 4.0 counts as a positive interaction, everything
    else is dropped — every collaborative model sees the same 0/1
    interactions, not raw explicit ratings.
  - Filtering: only users with >= MIN_USER_INTERACTIONS positive interactions
    are kept.
  - Split: a strict 80/20 split per user (not a global random split), so
    every user has both train and test rows.
  - Evaluation: items already seen in train are excluded from a user's
    top-K at eval time, and test interactions on items never seen in train
    (cold-start items with no embedding) are dropped from ground truth
    before scoring, on every model.

Metrics for ALL models: Precision@K, Recall@K, NDCG@K, MAP@K  (K = 10, 20)

RMSE/MAE are no longer reported: with binarized (0/1) interactions there is
no rating scale left to normalize scores against, so a rating-prediction
error metric isn't meaningful for any of the models here anymore.

ROC-AUC is intentionally not part of this shared comparison (it is still
reported as a secondary diagnostic by models/lightfm_model.py standalone):
AUC compares scores across a user's whole candidate set globally, while
P/R/NDCG/MAP all judge within a user's own ranked list, which fits how
WARP/BPR-trained models are actually optimized. For LightFM specifically,
two more targeted diagnostics are generated instead — see below.

Beyond per-user ranking accuracy, two aggregate diagnostics are also
reported, computed from the same top-K lists every model already produces
(no retraining needed):
  - Catalog coverage: % of the catalog that appears at least once across
    all eval users' top-K. Precision/Recall/NDCG score per-user relevance
    and say nothing about whether a model just recommends the same
    popular items to everyone (a common failure mode, especially for
    collaborative/graph models) — coverage surfaces that directly.
  - Performance by user interaction count (cold/light/heavy, matching the
    lifecycle bands models/recommendation_router.py routes on): an overall
    average can hide a model that only works well for heavy users, which
    matters here since the router's whole design assumes different models
    suit different lifecycle stages.

For the two LightFM variants specifically, a score-distribution histogram
(positive vs. a sampled set of negative item scores per user) is saved to
results/score_hist_<model>.png — a diagnostic for a step-like/clustered ROC
curve (score ties or clusters show up directly as spikes here, more
intuitively than a single AUC scalar).

Usage:
  python evaluation/compare_models.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
K_VALUES = [10, 20]
MAX_K = max(K_VALUES)
N_EVAL_USERS = 300
MIN_USER_INTERACTIONS = 5
RANDOM_SEED = 42
LIGHTFM_EPOCHS = 10

# Interaction-count segments for the cold/light/heavy breakdown in
# analyze_segments_and_coverage. Boundaries follow the same lifecycle
# stages models/recommendation_router.py routes users through (cold start
# 1-5, warm start 5-20, mature 20+) so this table can be read directly
# against that routing logic.
INTERACTION_SEGMENTS = [
    ("cold (5-9)", 5, 10),
    ("light (10-19)", 10, 20),
    ("heavy (20+)", 20, float("inf")),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ranking_metrics(ranked_ids, relevant: set, k_values):
    """Return dict of P@k, R@k, NDCG@k for a single user."""
    return {
        f"P@{k}": precision_at_k(ranked_ids, relevant, k)
        for k in k_values
    } | {
        f"R@{k}": recall_at_k(ranked_ids, relevant, k)
        for k in k_values
    } | {
        f"NDCG@{k}": ndcg_at_k(ranked_ids, relevant, k)
        for k in k_values
    }


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

RATING_THRESHOLD = 4.0


def load_and_split(
    min_interactions: int = MIN_USER_INTERACTIONS,
    max_rows: int | None = None,
):
    """
    Load ratings.csv (TMDB movie IDs), binarize to implicit positive
    interactions (rating >= RATING_THRESHOLD), and split 80/20 per user.

    This is the single shared preprocessing step for every collaborative
    model (SVD, LightFM, LightGCN) — they all train on the exact same
    train_df, so any metric difference between them reflects the model,
    not the data it saw.

    Returns train_df, test_df with columns [user_id, movie_id]. Explicit
    rating values are intentionally dropped after binarization — every
    downstream model treats interactions as implicit 0/1 feedback.
    """
    path = DATA_DIR / "ratings.csv"
    if not path.exists():
        # Fallback to ratings_clean.csv (MovieLens IDs)
        path = DATA_DIR / "ratings_clean.csv"
        df = pd.read_csv(path, usecols=["userId", "movieId", "rating"], nrows=max_rows)
        df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    else:
        df = pd.read_csv(path, usecols=["user_id", "movie_id", "rating"], nrows=max_rows)

    df = df.dropna(subset=["user_id", "movie_id", "rating"])
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)

    # Binarization: only positive interactions survive from here on.
    df = df[df["rating"] >= RATING_THRESHOLD][["user_id", "movie_id"]].drop_duplicates()
    print(f"  Positive interactions (rating >= {RATING_THRESHOLD}): {len(df):,}")

    # Filtering: keep users with enough positive interactions
    counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_interactions].index)].copy()

    train_rows, test_rows = [], []
    rng = np.random.RandomState(RANDOM_SEED)
    for _, udf in df.groupby("user_id"):
        shuffled = udf.sample(frac=1, random_state=rng)
        n_train = max(1, int(len(shuffled) * 0.8))
        train_rows.append(shuffled.iloc[:n_train])
        test_rows.append(shuffled.iloc[n_train:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"  Users: {df['user_id'].nunique():,} | Items: {df['movie_id'].nunique():,}")
    return train_df, test_df


def drop_unseen_test_items(train_df, test_df, valid_movie_ids):
    """
    Drop test interactions on items never seen in train (or outside a
    model's own item vocabulary) — such items have no embedding and can
    never be recommended, so leaving them in ground truth would silently
    deflate Recall/NDCG for reasons unrelated to model quality. Applied
    identically for every model right before scoring.
    """
    known = test_df[test_df["movie_id"].isin(valid_movie_ids)]
    n_dropped = len(test_df) - len(known)
    if n_dropped > 0:
        print(f"  ⚠️  Dropped {n_dropped} test interactions on items unseen in train")
    return known


def sample_eval_users(test_df: pd.DataFrame, n: int = N_EVAL_USERS) -> np.ndarray:
    users = test_df["user_id"].unique()
    return np.random.choice(users, size=min(n, len(users)), replace=False)


MAX_NEG_SAMPLES_PER_USER = 20


def _collect_score_sample(store, scores, relevant, tr_items, item_map, rng):
    """
    Append this user's positive-item scores and a random sample of
    negative-item scores into store["positive"]/store["negative"], for a
    score-distribution histogram (positives vs. negatives). Diagnostic for
    why a ROC curve looks step-like: score clusters/ties show up directly
    as spikes here, more intuitively than an AUC scalar. Negatives are
    sampled (capped at MAX_NEG_SAMPLES_PER_USER) rather than using every
    non-relevant item, since there are always far more negatives than
    positives per user and using all of them would make the negative
    curve reflect the whole catalog's score distribution rather than a
    comparably-sized sample.
    """
    pos_indices = [item_map[i] for i in relevant if i in item_map]
    if pos_indices:
        store["positive"].extend(scores[pos_indices].tolist())

    neg_candidates = [
        idx for item_id, idx in item_map.items()
        if item_id not in relevant and item_id not in tr_items
    ]
    if neg_candidates:
        n_sample = min(MAX_NEG_SAMPLES_PER_USER, len(neg_candidates))
        neg_indices = rng.choice(neg_candidates, size=n_sample, replace=False)
        store["negative"].extend(scores[neg_indices].tolist())


# ---------------------------------------------------------------------------
# Model 1: Content-Based (TF-IDF) — TF-IDF cosine similarity
#
# This is a lightweight TF-IDF baseline, distinct from the SentenceTransformer
# + FAISS content engine in models/content_based.py, which is not used here.
# ---------------------------------------------------------------------------

def evaluate_content_based(train_df, test_df, eval_users, top_k_store: dict | None = None):
    print("\n[1/5] Content-Based (TF-IDF genres + title)...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    # Load movie metadata
    movies_path = DATA_DIR / "movies.csv"
    if movies_path.exists():
        movies = pd.read_csv(movies_path, low_memory=False)
        # Expect: movie_id, title, genre_names
        id_col = "movie_id"
        text_col = "genre_names"
    else:
        movies_path = DATA_DIR / "movies_clean.csv"
        movies = pd.read_csv(movies_path, low_memory=False)
        id_col = "id"
        text_col = "genres"

    movies = movies.dropna(subset=[id_col]).copy()
    movies[id_col] = movies[id_col].astype(int)
    movies = movies.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    content_text = (
        movies["title"].astype(str).str.lower()
        + " "
        + movies[text_col].astype(str).str.lower()
    )

    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    item_matrix = tfidf.fit_transform(content_text.fillna(""))

    # Index: movie_id (TMDB) → row in item_matrix
    mid_to_idx = {mid: i for i, mid in enumerate(movies[id_col])}
    all_movie_ids = movies[id_col].tolist()

    # Ground truth sets — drop test items this catalog doesn't even have metadata for
    test_df_known = drop_unseen_test_items(train_df, test_df, set(mid_to_idx.keys()))
    test_rel = test_df_known.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_items = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    per_k: dict = {k: {"precision": [], "recall": [], "ndcg": [], "map": []} for k in K_VALUES}
    inf_times = []

    for uid in eval_users:
        relevant = test_rel.get(uid)
        if not relevant:
            continue
        tr_items = train_items.get(uid, set())
        tr_idx = [mid_to_idx[m] for m in tr_items if m in mid_to_idx]
        if not tr_idx:
            continue

        t0 = time.perf_counter()
        user_profile = np.asarray(item_matrix[tr_idx].mean(axis=0))
        scores = cos_sim(user_profile, item_matrix).flatten()
        inf_times.append(time.perf_counter() - t0)

        ranked = [
            all_movie_ids[i]
            for i in np.argsort(scores)[::-1]
            if all_movie_ids[i] not in tr_items
        ]
        if top_k_store is not None:
            top_k_store[uid] = ranked[:MAX_K]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))
            per_k[k]["map"].append(average_precision_at_k(ranked, relevant, k))

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))
        result[f"MAP@{k}"] = float(np.mean(per_k[k]["map"]))
    result["Infer ms"] = float(np.mean(inf_times) * 1000) if inf_times else "N/A"
    result["Train s"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Model 2: SVD (matrix factorization baseline)
# ---------------------------------------------------------------------------

def evaluate_svd(train_df, test_df, eval_users, n_factors: int = 50, top_k_store: dict | None = None):
    print("\n[2/5] SVD (matrix factorization baseline)...")
    from models.svd_model import train_svd

    t0 = time.perf_counter()
    user_factors, item_factors, user_map, item_map, item_inv = train_svd(train_df, n_factors)
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s "
          f"(k={min(n_factors, min(len(user_map), len(item_map)) - 1)} latent factors)")

    test_df_known = drop_unseen_test_items(train_df, test_df, set(item_map.keys()))
    test_rel = test_df_known.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_sets = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    per_k = {k_: {"precision": [], "recall": [], "ndcg": [], "map": []} for k_ in K_VALUES}
    inf_times = []

    for uid in eval_users:
        if uid not in user_map:
            continue
        relevant = test_rel.get(uid)
        if not relevant:
            continue

        u_internal = user_map[uid]

        t0 = time.perf_counter()
        scores = item_factors @ user_factors[u_internal]
        inf_times.append(time.perf_counter() - t0)

        tr_items = train_sets.get(uid, set())
        ranked = [
            item_inv[i]
            for i in np.argsort(scores)[::-1]
            if item_inv[i] not in tr_items
        ]
        if top_k_store is not None:
            top_k_store[uid] = ranked[:MAX_K]

        for k_ in K_VALUES:
            per_k[k_]["precision"].append(precision_at_k(ranked, relevant, k_))
            per_k[k_]["recall"].append(recall_at_k(ranked, relevant, k_))
            per_k[k_]["ndcg"].append(ndcg_at_k(ranked, relevant, k_))
            per_k[k_]["map"].append(average_precision_at_k(ranked, relevant, k_))

    result: dict = {}
    for k_ in K_VALUES:
        result[f"P@{k_}"] = float(np.mean(per_k[k_]["precision"]))
        result[f"R@{k_}"] = float(np.mean(per_k[k_]["recall"]))
        result[f"NDCG@{k_}"] = float(np.mean(per_k[k_]["ndcg"]))
        result[f"MAP@{k_}"] = float(np.mean(per_k[k_]["map"]))
    result["Infer ms"] = float(np.mean(inf_times) * 1000) if inf_times else "N/A"
    result["Train s"] = train_time
    return result


# ---------------------------------------------------------------------------
# Model 3: LightFM Collaborative
# ---------------------------------------------------------------------------

def evaluate_lightfm_collab(
    train_df, test_df, eval_users, epochs: int = LIGHTFM_EPOCHS,
    top_k_store: dict | None = None,
    score_sample_store: dict | None = None,
):
    print("\n[3/5] LightFM Collaborative (WARP loss)...")
    from lightfm import LightFM

    all_users = train_df["user_id"].unique()
    all_items = train_df["movie_id"].unique()
    n_users = len(all_users)
    n_items = len(all_items)

    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}

    u_idx = train_df["user_id"].map(user_map).values.astype(np.int32)
    i_idx = train_df["movie_id"].map(item_map).values.astype(np.int32)
    # Binarized interactions — same 0/1 signal SVD and LightGCN train on.
    weights = np.ones(len(train_df), dtype=np.float32)
    train_matrix = coo_matrix((weights, (u_idx, i_idx)), shape=(n_users, n_items))

    model = LightFM(no_components=64, learning_rate=0.05, loss="warp", random_state=RANDOM_SEED)

    t0 = time.perf_counter()
    model.fit(train_matrix, epochs=epochs, num_threads=os.cpu_count() or 1, verbose=False)
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s")

    test_df_known = drop_unseen_test_items(train_df, test_df, set(item_map.keys()))
    test_rel = test_df_known.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_sets = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    all_item_ids = list(item_map.keys())
    all_item_internal = np.array([item_map[m] for m in all_item_ids], dtype=np.int32)

    per_k = {k: {"precision": [], "recall": [], "ndcg": [], "map": []} for k in K_VALUES}
    inf_times = []
    score_rng = np.random.RandomState(RANDOM_SEED)

    for uid in eval_users:
        if uid not in user_map:
            continue
        relevant = test_rel.get(uid)
        if not relevant:
            continue

        u_internal = user_map[uid]
        u_arr = np.full(n_items, u_internal, dtype=np.int32)

        t0 = time.perf_counter()
        scores = model.predict(u_arr, all_item_internal, num_threads=os.cpu_count() or 1)
        inf_times.append(time.perf_counter() - t0)

        tr_items = train_sets.get(uid, set())
        ranked = [
            all_item_ids[i]
            for i in np.argsort(scores)[::-1]
            if all_item_ids[i] not in tr_items
        ]
        if top_k_store is not None:
            top_k_store[uid] = ranked[:MAX_K]
        if score_sample_store is not None:
            _collect_score_sample(
                score_sample_store, scores, relevant, tr_items, item_map, score_rng
            )

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))
            per_k[k]["map"].append(average_precision_at_k(ranked, relevant, k))

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))
        result[f"MAP@{k}"] = float(np.mean(per_k[k]["map"]))
    result["Infer ms"] = float(np.mean(inf_times) * 1000)
    result["Train s"] = train_time
    return result


# ---------------------------------------------------------------------------
# Model 4: LightFM Hybrid (collaborative + TF-IDF item features)
# ---------------------------------------------------------------------------

def evaluate_lightfm_hybrid(
    train_df, test_df, eval_users, epochs: int = LIGHTFM_EPOCHS,
    top_k_store: dict | None = None,
    score_sample_store: dict | None = None,
):
    print("\n[4/5] LightFM Hybrid (WARP + TF-IDF item features)...")
    from lightfm import LightFM
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load movie content features
    movies_path = DATA_DIR / "movies.csv"
    if movies_path.exists():
        movies = pd.read_csv(movies_path, low_memory=False)
        id_col, text_col = "movie_id", "genre_names"
    else:
        movies = pd.read_csv(DATA_DIR / "movies_clean.csv", low_memory=False)
        id_col, text_col = "id", "genres"

    movies = movies.dropna(subset=[id_col]).copy()
    movies[id_col] = movies[id_col].astype(int)
    movies = movies.drop_duplicates(subset=[id_col])

    # Genres only, not title: title tokens are near-unique per movie, so
    # including them makes each item's feature row act like a noisy one-hot
    # item ID — it doesn't generalize across movies and adds weights that
    # slow WARP's convergence on the collaborative signal.
    content_text = movies[text_col].astype(str).fillna("").tolist()
    tfidf_all = TfidfVectorizer(max_features=200, stop_words="english")
    all_content_features = tfidf_all.fit_transform(content_text)
    mid_to_content_row = {int(mid): i for i, mid in enumerate(movies[id_col])}

    all_users = train_df["user_id"].unique()
    all_items = train_df["movie_id"].unique()
    n_users = len(all_users)
    n_items = len(all_items)

    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}

    # Build item feature matrix aligned to item_map order
    item_feature_rows = []
    for mid in all_items:
        if mid in mid_to_content_row:
            item_feature_rows.append(all_content_features[mid_to_content_row[mid]])
        else:
            import scipy.sparse as sp
            item_feature_rows.append(sp.csr_matrix((1, all_content_features.shape[1])))

    from scipy.sparse import identity as sp_identity
    from scipy.sparse import vstack as sp_vstack
    from scipy.sparse import hstack as sp_hstack

    genre_features = sp_vstack(item_feature_rows)  # (n_items, n_genre_features)
    # LightFM estimates one embedding per feature COLUMN and sums them for an
    # item's representation — passing only genre features means every item's
    # embedding is entirely a function of its genre, so items sharing a genre
    # become indistinguishable. Concatenating a per-item identity block keeps
    # each item's own learnable embedding (same as the no-features/Collab
    # case) in addition to the shared genre signal. Row-normalize (each
    # item's feature weights sum to 1), matching lightfm.data.Dataset's
    # build_item_features default, so the identity anchor and the genre
    # signal are on comparable footing regardless of genre count per movie.
    item_features = sp_hstack([sp_identity(n_items, format="csr"), genre_features]).tocsr()
    row_sums = np.asarray(item_features.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    item_features = item_features.multiply(1.0 / row_sums[:, None]).tocsr()

    u_idx = train_df["user_id"].map(user_map).values.astype(np.int32)
    i_idx = train_df["movie_id"].map(item_map).values.astype(np.int32)
    weights = np.ones(len(train_df), dtype=np.float32)
    train_matrix = coo_matrix((weights, (u_idx, i_idx)), shape=(n_users, n_items))

    model = LightFM(no_components=64, learning_rate=0.05, loss="warp", random_state=RANDOM_SEED)

    t0 = time.perf_counter()
    model.fit(train_matrix, item_features=item_features, epochs=epochs, num_threads=os.cpu_count() or 1, verbose=False)
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s")

    test_df_known = drop_unseen_test_items(train_df, test_df, set(item_map.keys()))
    test_rel = test_df_known.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_sets = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    all_item_ids = list(item_map.keys())
    all_item_internal = np.array([item_map[m] for m in all_item_ids], dtype=np.int32)

    per_k = {k: {"precision": [], "recall": [], "ndcg": [], "map": []} for k in K_VALUES}
    inf_times = []
    score_rng = np.random.RandomState(RANDOM_SEED)

    for uid in eval_users:
        if uid not in user_map:
            continue
        relevant = test_rel.get(uid)
        if not relevant:
            continue

        u_internal = user_map[uid]
        u_arr = np.full(n_items, u_internal, dtype=np.int32)

        t0 = time.perf_counter()
        scores = model.predict(u_arr, all_item_internal, item_features=item_features, num_threads=os.cpu_count() or 1)
        inf_times.append(time.perf_counter() - t0)

        tr_items = train_sets.get(uid, set())
        ranked = [
            all_item_ids[i]
            for i in np.argsort(scores)[::-1]
            if all_item_ids[i] not in tr_items
        ]
        if top_k_store is not None:
            top_k_store[uid] = ranked[:MAX_K]
        if score_sample_store is not None:
            _collect_score_sample(
                score_sample_store, scores, relevant, tr_items, item_map, score_rng
            )

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))
            per_k[k]["map"].append(average_precision_at_k(ranked, relevant, k))

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))
        result[f"MAP@{k}"] = float(np.mean(per_k[k]["map"]))
    result["Infer ms"] = float(np.mean(inf_times) * 1000)
    result["Train s"] = train_time
    return result


# ---------------------------------------------------------------------------
# Model 5: LightGCN
# ---------------------------------------------------------------------------

def evaluate_lightgcn(
    train_df, test_df, eval_users,
    top_k_store: dict | None = None,
    loss_history_store: list | None = None,
):
    print("\n[5/5] LightGCN (graph convolutional network, BPR loss)...")
    try:
        import torch
        from models.lightgcn_model import LightGCN, prepare_lightgcn_data, train_lightgcn
    except ImportError as exc:
        print(f"  Skipping LightGCN: {exc}")
        na = {f"P@{k}": "N/A" for k in K_VALUES}
        na |= {f"R@{k}": "N/A" for k in K_VALUES}
        na |= {f"NDCG@{k}": "N/A" for k in K_VALUES}
        na |= {"Infer ms": "N/A", "Train s": "N/A"}
        return na

    # prepare_lightgcn_data expects userId / movieId columns.
    # min_user_interactions=0: train_df is already filtered by load_and_split()
    # to the same MIN_USER_INTERACTIONS threshold every other model trains on —
    # filtering again here would silently shrink LightGCN's graph relative to
    # what SVD/LightFM see.
    train_gcn = train_df.rename(columns={"user_id": "userId", "movie_id": "movieId"})

    edge_index, n_users, n_items, user_inv, item_inv, gt_dict, user_map, item_map = (
        prepare_lightgcn_data(train_gcn.copy(), min_user_interactions=0)
    )

    # Seed torch before constructing the model: LightGCN.__init__ initializes
    # embeddings via nn.init.normal_, which draws from torch's RNG. Seeding
    # inside train_lightgcn would be too late — the model (and its random
    # initial embeddings) is already built by the time train_lightgcn runs.
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=64, n_layers=3).to(device)

    # epochs=10 is now comparable to LightFM's epochs=10: train_lightgcn
    # samples a (pos, neg) pair for EVERY positive interaction of every user
    # each epoch (not one triple per user total), so one LightGCN epoch and
    # one LightFM epoch both mean "one pass over every positive interaction"
    # — deliberately equalized, not copy-pasted from LightFM's default.
    #
    # batch_size=2048 (down from 8192): smaller batches mean noisier
    # per-step gradients, which can help escape the kind of plateau this
    # model was landing in — tradeoff is more optimizer.step() calls per
    # epoch (slower), acceptable at epochs=10.
    t0 = time.perf_counter()
    user_embs, item_embs, losses = train_lightgcn(
        model, edge_index, gt_dict, n_users, n_items,
        user_map, item_map, lr=0.001, epochs=10, batch_size=2048,
    )
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s")
    if loss_history_store is not None:
        loss_history_store[:] = losses

    test_df_known = drop_unseen_test_items(train_df, test_df, set(item_map.keys()))
    test_rel = test_df_known.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_sets = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    per_k = {k: {"precision": [], "recall": [], "ndcg": [], "map": []} for k in K_VALUES}

    # Batched scoring + torch.topk instead of a per-user Python loop with a
    # full argsort over all n_items: one matmul over every eval user at
    # once, then only the top candidates (not the whole catalog) come back
    # to CPU for the exclusion-filtering/metric step below.
    valid_uids = [uid for uid in eval_users if uid in user_map and test_rel.get(uid)]
    if not valid_uids:
        result: dict = {f"P@{k}": 0.0 for k in K_VALUES}
        result |= {f"R@{k}": 0.0 for k in K_VALUES}
        result |= {f"NDCG@{k}": 0.0 for k in K_VALUES}
        result["Infer ms"] = "N/A"
        result["Train s"] = train_time
        return result

    u_indices = torch.tensor([user_map[uid] for uid in valid_uids], device=device)
    batch_user_embs = user_embs[u_indices]  # (n_valid_users, emb_dim)

    # Buffer size guarantees at least MAX_K candidates survive exclusion of
    # each user's own train items — the same guarantee a full sort gave,
    # without actually sorting the whole catalog per user.
    max_train_count = max((len(train_sets.get(uid, set())) for uid in valid_uids), default=0)
    topk_buffer = min(n_items, MAX_K + max_train_count)

    t0 = time.perf_counter()
    with torch.no_grad():
        all_scores = batch_user_embs @ item_embs.T  # (n_valid_users, n_items)
        _, top_indices = torch.topk(all_scores, k=topk_buffer, dim=1)
    inf_time_total = time.perf_counter() - t0
    top_indices = top_indices.cpu().numpy()

    for row, uid in enumerate(valid_uids):
        relevant = test_rel[uid]
        tr_items = train_sets.get(uid, set())
        ranked = [
            item_inv[i]
            for i in top_indices[row]
            if item_inv.get(i) is not None and item_inv[i] not in tr_items
        ]
        if top_k_store is not None:
            top_k_store[uid] = ranked[:MAX_K]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))
            per_k[k]["map"].append(average_precision_at_k(ranked, relevant, k))

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))
        result[f"MAP@{k}"] = float(np.mean(per_k[k]["map"]))
    # Infer ms is now the amortized per-user cost of one batched matmul +
    # topk over all eval users, not a separately timed single-vector matmul
    # per user (that per-user loop was the thing being vectorized away) —
    # arguably closer to what a real batched-serving deployment would see.
    result["Infer ms"] = (inf_time_total / len(valid_uids)) * 1000
    result["Train s"] = train_time
    return result


# ---------------------------------------------------------------------------
# Production analysis (qualitative)
# ---------------------------------------------------------------------------

PRODUCTION_ANALYSIS = {
    "Content-Based (TF-IDF)": {
        "Cold Start": "Отлично",
        "Масштабируемость": "Хорошая O(I)",
        "Real-Time": "Да",
        "Обновление": "Простое (метаданные)",
        "Сложность": "Низкая",
    },
    "SVD": {
        "Cold Start": "Плохо",
        "Масштабируемость": "Средняя O(U×I)",
        "Real-Time": "Нет",
        "Обновление": "Переобучение",
        "Сложность": "Низкая",
    },
    "LightFM Collab": {
        "Cold Start": "Плохо",
        "Масштабируемость": "Средняя O(U×I)",
        "Real-Time": "Нет",
        "Обновление": "Переобучение",
        "Сложность": "Средняя",
    },
    "LightFM Hybrid": {
        "Cold Start": "Хорошо (item features)",
        "Масштабируемость": "Средняя",
        "Real-Time": "Частично",
        "Обновление": "Переобучение",
        "Сложность": "Средняя",
    },
    "LightGCN": {
        "Cold Start": "Плохо",
        "Масштабируемость": "Средняя O(E)",
        "Real-Time": "Нет",
        "Обновление": "Дорогое (граф)",
        "Сложность": "Высокая",
    },
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _expand_models(models: list[str] | None) -> set[str]:
    all_models = {"content", "svd", "lightfm_collab", "lightfm_hybrid", "lightgcn"}
    if not models:
        return all_models

    expanded: set[str] = set()
    for model in models:
        if model == "all":
            return all_models
        if model == "lightfm":
            expanded.update({"lightfm_collab", "lightfm_hybrid"})
        else:
            expanded.add(model)
    return expanded


def compute_coverage(top_k_store: dict, catalog_size: int, k: int = MAX_K) -> tuple[float, int]:
    """
    % of the catalog that appears at least once across all users' top-K —
    a low number flags a model that just recommends the same popular items
    to everyone (a common failure mode especially for graph/CF models),
    something Precision/Recall/NDCG (which score per-user relevance, not
    the aggregate item distribution) don't surface at all.

    Returns (coverage_fraction, n_unique_items_recommended).
    """
    if not top_k_store or catalog_size == 0:
        return 0.0, 0
    recommended_items = set()
    for ranked in top_k_store.values():
        recommended_items.update(ranked[:k])
    return len(recommended_items) / catalog_size, len(recommended_items)


def compute_segment_metrics(top_k_store: dict, test_rel: dict, train_counts: dict) -> dict:
    """
    Precision/Recall/NDCG@10, split by each user's TRAIN interaction count
    into the same cold/light/heavy bands recommendation_router.py routes
    on. Overall averages can hide a model that only works for heavy users
    — this is the same concern the router's whole design is built around,
    so it's worth measuring directly rather than asserting.
    """
    segment_results = {}
    for label, lo, hi in INTERACTION_SEGMENTS:
        per_metric = {"precision": [], "recall": [], "ndcg": []}
        for uid, ranked in top_k_store.items():
            count = train_counts.get(uid, 0)
            if not (lo <= count < hi):
                continue
            relevant = test_rel.get(uid)
            if not relevant:
                continue
            per_metric["precision"].append(precision_at_k(ranked, relevant, 10))
            per_metric["recall"].append(recall_at_k(ranked, relevant, 10))
            per_metric["ndcg"].append(ndcg_at_k(ranked, relevant, 10))

        n = len(per_metric["precision"])
        segment_results[label] = {
            "n_users": n,
            "P@10": float(np.mean(per_metric["precision"])) if n else "N/A",
            "R@10": float(np.mean(per_metric["recall"])) if n else "N/A",
            "NDCG@10": float(np.mean(per_metric["ndcg"])) if n else "N/A",
        }
    return segment_results


RESULTS_DIR = Path(__file__).parent.parent / "results"


def plot_score_histogram(model_name: str, score_sample: dict, out_dir: Path = RESULTS_DIR) -> Path | None:
    """
    Overlaid histogram of raw prediction scores for positive vs. sampled
    negative items. Diagnostic for a step-like/clustered ROC curve: score
    ties or clusters (e.g. many items scoring near-identically) show up
    directly as spikes here, which a single AUC number can't reveal.
    Returns the saved file path, or None if there's nothing to plot.
    """
    positive = score_sample.get("positive", [])
    negative = score_sample.get("negative", [])
    if not positive and not negative:
        return None

    import matplotlib
    matplotlib.use("Agg")  # headless — no display available on a training server
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 40
    if negative:
        ax.hist(negative, bins=bins, alpha=0.6, label=f"negative (n={len(negative)})", color="tab:blue", density=True)
    if positive:
        ax.hist(positive, bins=bins, alpha=0.6, label=f"positive (n={len(positive)})", color="tab:orange", density=True)
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name}: predicted score distribution (positive vs. negative)")
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    out_path = out_dir / f"score_hist_{safe_name}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_comparison(
    n_eval_users: int = N_EVAL_USERS,
    models: list[str] | None = None,
    max_rows: int | None = None,
    lightfm_epochs: int = LIGHTFM_EPOCHS,
) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)

    print("=" * 72)
    print("  СРАВНИТЕЛЬНАЯ ОЦЕНКА РЕКОМЕНДАТЕЛЬНЫХ СИСТЕМ")
    print("=" * 72)

    print("\nLoading & splitting data...")
    train_df, test_df = load_and_split(max_rows=max_rows)
    eval_users = sample_eval_users(test_df, n=n_eval_users)
    print(f"  Eval users: {len(eval_users)}, K = {K_VALUES}")

    selected_models = _expand_models(models)
    results = {}
    top_k_stores: dict = {}

    if "content" in selected_models:
        top_k_stores["Content-Based (TF-IDF)"] = {}
        results["Content-Based (TF-IDF)"] = evaluate_content_based(
            train_df, test_df, eval_users, top_k_store=top_k_stores["Content-Based (TF-IDF)"]
        )
    if "svd" in selected_models:
        top_k_stores["SVD"] = {}
        results["SVD"] = evaluate_svd(train_df, test_df, eval_users, top_k_store=top_k_stores["SVD"])
    score_sample_stores: dict = {}
    if "lightfm_collab" in selected_models:
        top_k_stores["LightFM Collab"] = {}
        score_sample_stores["LightFM Collab"] = {"positive": [], "negative": []}
        results["LightFM Collab"] = evaluate_lightfm_collab(
            train_df, test_df, eval_users, epochs=lightfm_epochs,
            top_k_store=top_k_stores["LightFM Collab"],
            score_sample_store=score_sample_stores["LightFM Collab"],
        )
    if "lightfm_hybrid" in selected_models:
        top_k_stores["LightFM Hybrid"] = {}
        score_sample_stores["LightFM Hybrid"] = {"positive": [], "negative": []}
        results["LightFM Hybrid"] = evaluate_lightfm_hybrid(
            train_df, test_df, eval_users, epochs=lightfm_epochs,
            top_k_store=top_k_stores["LightFM Hybrid"],
            score_sample_store=score_sample_stores["LightFM Hybrid"],
        )
    lightgcn_loss_history: list = []
    if "lightgcn" in selected_models:
        top_k_stores["LightGCN"] = {}
        results["LightGCN"] = evaluate_lightgcn(
            train_df, test_df, eval_users,
            top_k_store=top_k_stores["LightGCN"],
            loss_history_store=lightgcn_loss_history,
        )

    # Build comparison DataFrame
    col_order = (
        [f"P@{k}" for k in K_VALUES]
        + [f"R@{k}" for k in K_VALUES]
        + [f"NDCG@{k}" for k in K_VALUES]
        + [f"MAP@{k}" for k in K_VALUES]
        + ["Infer ms", "Train s"]
    )
    df = pd.DataFrame(results).T
    df = df[[c for c in col_order if c in df.columns]]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    df_str = df.applymap(fmt)

    print("\n" + "=" * 72)
    print("  МЕТРИКИ")
    print("=" * 72)
    print(df_str.to_string())

    if lightgcn_loss_history:
        print("\n" + "=" * 72)
        print("  LIGHTGCN LOSS CURVE (BPR loss, per epoch)")
        print("=" * 72)
        curve = ", ".join(f"{v:.4f}" for v in lightgcn_loss_history)
        print(f"  {curve}")
        if len(lightgcn_loss_history) >= 2:
            delta = lightgcn_loss_history[-2] - lightgcn_loss_history[-1]
            print(
                f"  Last-epoch improvement: {delta:.4f} "
                f"({'still declining' if delta > 1e-4 else 'plateaued'} "
                "— see RESULTS.md before treating epochs=10 as final)"
            )

    # Catalog coverage + performance-by-interaction-count segments: both
    # computed from the same top_k_stores every model already filled in, no
    # retraining needed. Coverage flags a model that just recommends
    # popular items to everyone; segments flag a model whose good overall
    # average hides poor performance for the very cold/light users
    # recommendation_router.py has to route.
    catalog_size = train_df["movie_id"].nunique()
    test_rel_all = drop_unseen_test_items(
        train_df, test_df, set(train_df["movie_id"].unique())
    ).groupby("user_id")["movie_id"].apply(set).to_dict()
    train_counts = train_df["user_id"].value_counts().to_dict()

    print("\n" + "=" * 72)
    print("  CATALOG COVERAGE (% of items ever recommended in top-{})".format(MAX_K))
    print("=" * 72)
    for name, store in top_k_stores.items():
        coverage, n_unique = compute_coverage(store, catalog_size)
        print(f"  {name:<24s} {coverage:.1%}  ({n_unique}/{catalog_size} items)")

    print("\n" + "=" * 72)
    print("  PERFORMANCE BY USER INTERACTION COUNT (cold/light/heavy)")
    print("=" * 72)
    for name, store in top_k_stores.items():
        print(f"\n  {name}:")
        segments = compute_segment_metrics(store, test_rel_all, train_counts)
        seg_df = pd.DataFrame(segments).T
        print("  " + seg_df.to_string().replace("\n", "\n  "))

    print("\n" + "=" * 72)
    print("  АНАЛИЗ ПРИГОДНОСТИ ДЛЯ ПРОДАКШЕНА")
    print("=" * 72)
    print(pd.DataFrame(PRODUCTION_ANALYSIS).T.to_string())

    print("\n" + "=" * 72)
    print("  ПРИМЕЧАНИЯ")
    print("=" * 72)
    print(
        f"Protocol:  SVD, LightFM и LightGCN обучены на ОДНОМ train_df — implicit\n"
        f"           interactions (rating >= {RATING_THRESHOLD}), пользователи с >= "
        f"{MIN_USER_INTERACTIONS} лайками,\n"
        "           80/20 split per user. Метрики поэтому напрямую сравнимы.\n"
        "Infer ms:  время предсказания для одного пользователя (миллисекунды).\n"
        "Train s:   полное время обучения на train_df (секунды)."
    )

    if score_sample_stores:
        print("\n" + "=" * 72)
        print("  SCORE DISTRIBUTION HISTOGRAMS (positive vs. negative)")
        print("=" * 72)
        for name, sample in score_sample_stores.items():
            path = plot_score_histogram(name, sample)
            if path is not None:
                print(f"  {name:<24s} saved to {path}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommendation models.")
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated models: all, content, svd, lightfm, lightfm_collab, "
            "lightfm_hybrid, lightgcn."
        ),
    )
    parser.add_argument("--eval-users", type=int, default=N_EVAL_USERS)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lightfm-epochs", type=int, default=LIGHTFM_EPOCHS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(
        n_eval_users=args.eval_users,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        max_rows=args.max_rows,
        lightfm_epochs=args.lightfm_epochs,
    )
