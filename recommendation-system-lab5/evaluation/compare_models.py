"""
Unified evaluation of all 4 recommendation models.

Models:
  1. Content-Based  – TF-IDF on genres/title, cosine similarity
  2. LightFM Collab – factorization machines, WARP loss, interaction data only
  3. LightFM Hybrid – WARP loss + TF-IDF item features
  4. LightGCN       – graph convolutional network, BPR loss

Metrics for ALL models:
  Precision@K, Recall@K, NDCG@K  (K = 10, 20)

Additional for LightFM models only (explicit ratings available):
  RMSE, MAE  via score normalization to [1, 5] scale

Reasoning: LightGCN and Content-Based optimize for ranking (not rating
prediction), so RMSE is not meaningful for them. LightFM outputs raw
logits that can be normalized to [1,5] as a heuristic.

Usage:
  cd recommendation-system-lab5
  python evaluation/compare_models.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
K_VALUES = [10, 20]
N_EVAL_USERS = 300
MIN_USER_INTERACTIONS = 5
RANDOM_SEED = 42
LIGHTFM_EPOCHS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmse_mae_normalized(raw_scores: np.ndarray, true_ratings: np.ndarray):
    """Normalize LightFM raw scores to [1,5] then compute RMSE and MAE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    lo, hi = raw_scores.min(), raw_scores.max()
    if hi == lo:
        norm = np.full_like(raw_scores, 3.0)
    else:
        norm = 1.0 + 4.0 * (raw_scores - lo) / (hi - lo)
    norm = np.clip(norm, 1.0, 5.0)
    rmse = float(np.sqrt(mean_squared_error(true_ratings, norm)))
    mae = float(mean_absolute_error(true_ratings, norm))
    return rmse, mae


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

def load_and_split(
    min_interactions: int = MIN_USER_INTERACTIONS,
    max_rows: int | None = None,
):
    """
    Load ratings.csv (TMDB movie IDs) and split 80/20 per user.
    Returns train_df, test_df with columns [user_id, movie_id, rating].
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

    # Keep users with enough interactions
    counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_interactions].index)].copy()

    # Keep items with at least 3 interactions
    icounts = df["movie_id"].value_counts()
    df = df[df["movie_id"].isin(icounts[icounts >= 3].index)].copy()

    train_rows, test_rows = [], []
    for _, udf in df.groupby("user_id"):
        shuffled = udf.sample(frac=1, random_state=RANDOM_SEED)
        n_train = max(1, int(len(shuffled) * 0.8))
        train_rows.append(shuffled.iloc[:n_train])
        test_rows.append(shuffled.iloc[n_train:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"  Users: {df['user_id'].nunique():,} | Items: {df['movie_id'].nunique():,}")
    return train_df, test_df


def sample_eval_users(test_df: pd.DataFrame, n: int = N_EVAL_USERS) -> np.ndarray:
    users = test_df["user_id"].unique()
    return np.random.choice(users, size=min(n, len(users)), replace=False)


# ---------------------------------------------------------------------------
# Model 1: Content-Based (TF-IDF cosine similarity)
# ---------------------------------------------------------------------------

def evaluate_content_based(train_df, test_df, eval_users):
    print("\n[1/4] Content-Based (TF-IDF genres + title)...")
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

    # Ground truth sets
    test_rel = test_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_items = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    all_movie_ids = movies[id_col].tolist()

    per_k: dict = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
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
        user_profile = item_matrix[tr_idx].mean(axis=0)
        scores = cos_sim(user_profile, item_matrix).flatten()
        inf_times.append(time.perf_counter() - t0)

        ranked = [
            all_movie_ids[i]
            for i in np.argsort(scores)[::-1]
            if all_movie_ids[i] not in tr_items
        ]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))
    result["RMSE"] = "N/A"
    result["MAE"] = "N/A"
    result["Infer ms"] = float(np.mean(inf_times) * 1000) if inf_times else "N/A"
    result["Train s"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Model 2: LightFM Collaborative
# ---------------------------------------------------------------------------

def evaluate_lightfm_collab(train_df, test_df, eval_users, epochs: int = LIGHTFM_EPOCHS):
    print("\n[2/4] LightFM Collaborative (WARP loss)...")
    from lightfm import LightFM

    all_users = train_df["user_id"].unique()
    all_items = train_df["movie_id"].unique()
    n_users = len(all_users)
    n_items = len(all_items)

    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}

    u_idx = train_df["user_id"].map(user_map).values.astype(np.int32)
    i_idx = train_df["movie_id"].map(item_map).values.astype(np.int32)
    weights = train_df["rating"].values.astype(np.float32)
    train_matrix = coo_matrix((weights, (u_idx, i_idx)), shape=(n_users, n_items))

    model = LightFM(no_components=64, learning_rate=0.05, loss="warp", random_state=RANDOM_SEED)

    t0 = time.perf_counter()
    model.fit(train_matrix, epochs=epochs, num_threads=1, verbose=False)
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s")

    test_rel = test_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_sets = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    all_item_ids = list(item_map.keys())
    all_item_internal = np.array([item_map[m] for m in all_item_ids], dtype=np.int32)

    per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    raw_preds, true_rats = [], []
    inf_times = []

    for uid in eval_users:
        if uid not in user_map:
            continue
        relevant = test_rel.get(uid)
        if not relevant:
            continue

        u_internal = user_map[uid]
        u_arr = np.full(n_items, u_internal, dtype=np.int32)

        t0 = time.perf_counter()
        scores = model.predict(u_arr, all_item_internal, num_threads=1)
        inf_times.append(time.perf_counter() - t0)

        tr_items = train_sets.get(uid, set())
        ranked = [
            all_item_ids[i]
            for i in np.argsort(scores)[::-1]
            if all_item_ids[i] not in tr_items
        ]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))

        # Collect raw scores on test items for RMSE
        test_items_list = list(relevant & set(item_map.keys()))
        if test_items_list:
            ti_internal = np.array([item_map[m] for m in test_items_list], dtype=np.int32)
            u_arr_rmse = np.full(len(ti_internal), u_internal, dtype=np.int32)
            raw = model.predict(u_arr_rmse, ti_internal, num_threads=1)
            true_r = (
                test_df[
                    (test_df["user_id"] == uid) & (test_df["movie_id"].isin(test_items_list))
                ]["rating"].values
            )
            min_len = min(len(raw), len(true_r))
            raw_preds.extend(raw[:min_len])
            true_rats.extend(true_r[:min_len])

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))

    if raw_preds:
        rmse, mae = _rmse_mae_normalized(np.array(raw_preds), np.array(true_rats))
        result["RMSE"] = rmse
        result["MAE"] = mae
    else:
        result["RMSE"] = "N/A"
        result["MAE"] = "N/A"

    result["Infer ms"] = float(np.mean(inf_times) * 1000)
    result["Train s"] = train_time
    return result


# ---------------------------------------------------------------------------
# Model 3: LightFM Hybrid (collaborative + TF-IDF item features)
# ---------------------------------------------------------------------------

def evaluate_lightfm_hybrid(train_df, test_df, eval_users, epochs: int = LIGHTFM_EPOCHS):
    print("\n[3/4] LightFM Hybrid (WARP + TF-IDF item features)...")
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

    content_text = (
        movies["title"].astype(str) + " " + movies[text_col].astype(str)
    ).fillna("").tolist()
    tfidf_all = TfidfVectorizer(max_features=500, stop_words="english")
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

    from scipy.sparse import vstack as sp_vstack
    item_features = sp_vstack(item_feature_rows)  # (n_items, n_features)

    u_idx = train_df["user_id"].map(user_map).values.astype(np.int32)
    i_idx = train_df["movie_id"].map(item_map).values.astype(np.int32)
    weights = train_df["rating"].values.astype(np.float32)
    train_matrix = coo_matrix((weights, (u_idx, i_idx)), shape=(n_users, n_items))

    model = LightFM(no_components=64, learning_rate=0.05, loss="warp", random_state=RANDOM_SEED)

    t0 = time.perf_counter()
    model.fit(train_matrix, item_features=item_features, epochs=epochs, num_threads=1, verbose=False)
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s")

    test_rel = test_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_sets = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    all_item_ids = list(item_map.keys())
    all_item_internal = np.array([item_map[m] for m in all_item_ids], dtype=np.int32)

    per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    raw_preds, true_rats = [], []
    inf_times = []

    for uid in eval_users:
        if uid not in user_map:
            continue
        relevant = test_rel.get(uid)
        if not relevant:
            continue

        u_internal = user_map[uid]
        u_arr = np.full(n_items, u_internal, dtype=np.int32)

        t0 = time.perf_counter()
        scores = model.predict(u_arr, all_item_internal, item_features=item_features, num_threads=1)
        inf_times.append(time.perf_counter() - t0)

        tr_items = train_sets.get(uid, set())
        ranked = [
            all_item_ids[i]
            for i in np.argsort(scores)[::-1]
            if all_item_ids[i] not in tr_items
        ]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))

        test_items_list = list(relevant & set(item_map.keys()))
        if test_items_list:
            ti_internal = np.array([item_map[m] for m in test_items_list], dtype=np.int32)
            u_arr_rmse = np.full(len(ti_internal), u_internal, dtype=np.int32)
            raw = model.predict(u_arr_rmse, ti_internal, item_features=item_features, num_threads=1)
            true_r = (
                test_df[
                    (test_df["user_id"] == uid) & (test_df["movie_id"].isin(test_items_list))
                ]["rating"].values
            )
            min_len = min(len(raw), len(true_r))
            raw_preds.extend(raw[:min_len])
            true_rats.extend(true_r[:min_len])

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))

    if raw_preds:
        rmse, mae = _rmse_mae_normalized(np.array(raw_preds), np.array(true_rats))
        result["RMSE"] = rmse
        result["MAE"] = mae
    else:
        result["RMSE"] = "N/A"
        result["MAE"] = "N/A"

    result["Infer ms"] = float(np.mean(inf_times) * 1000)
    result["Train s"] = train_time
    return result


# ---------------------------------------------------------------------------
# Model 4: LightGCN
# ---------------------------------------------------------------------------

def evaluate_lightgcn(train_df, test_df, eval_users):
    print("\n[4/4] LightGCN (graph convolutional network, BPR loss)...")
    try:
        import torch
        from models.lightgcn_model import LightGCN, prepare_lightgcn_data, train_lightgcn
    except ImportError as exc:
        print(f"  Skipping LightGCN: {exc}")
        na = {f"P@{k}": "N/A" for k in K_VALUES}
        na |= {f"R@{k}": "N/A" for k in K_VALUES}
        na |= {f"NDCG@{k}": "N/A" for k in K_VALUES}
        na |= {"RMSE": "N/A", "MAE": "N/A", "Infer ms": "N/A", "Train s": "N/A"}
        return na

    # prepare_lightgcn_data expects userId / movieId columns
    train_gcn = train_df.rename(columns={"user_id": "userId", "movie_id": "movieId"})

    edge_index, n_users, n_items, user_inv, item_inv, gt_dict, user_map, item_map = (
        prepare_lightgcn_data(train_gcn.copy(), min_user_interactions=2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=64, n_layers=3).to(device)

    t0 = time.perf_counter()
    user_embs, item_embs, _losses = train_lightgcn(
        model, edge_index, gt_dict, n_users, n_items,
        user_map, item_map, lr=0.001, epochs=10, batch_size=1024,
    )
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s")

    user_embs = user_embs.detach().cpu().numpy()
    item_embs = item_embs.detach().cpu().numpy()

    test_rel = test_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_sets = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    inf_times = []

    for uid in eval_users:
        if uid not in user_map:
            continue
        relevant = test_rel.get(uid)
        if not relevant:
            continue

        u_idx = user_map[uid]
        u_emb = user_embs[u_idx]

        t0 = time.perf_counter()
        scores = item_embs @ u_emb
        inf_times.append(time.perf_counter() - t0)

        tr_items = train_sets.get(uid, set())
        ranked = [
            item_inv[i]
            for i in np.argsort(scores)[::-1]
            if item_inv.get(i) is not None and item_inv[i] not in tr_items
        ]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))

    result: dict = {}
    for k in K_VALUES:
        result[f"P@{k}"] = float(np.mean(per_k[k]["precision"]))
        result[f"R@{k}"] = float(np.mean(per_k[k]["recall"]))
        result[f"NDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))
    result["RMSE"] = "N/A"
    result["MAE"] = "N/A"
    result["Infer ms"] = float(np.mean(inf_times) * 1000) if inf_times else "N/A"
    result["Train s"] = train_time
    return result


# ---------------------------------------------------------------------------
# Production analysis (qualitative)
# ---------------------------------------------------------------------------

PRODUCTION_ANALYSIS = {
    "Content-Based": {
        "Cold Start": "Отлично",
        "Масштабируемость": "Хорошая O(I)",
        "Real-Time": "Да",
        "Обновление": "Простое (метаданные)",
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
    if not models:
        return {"content", "lightfm_collab", "lightfm_hybrid", "lightgcn"}

    expanded: set[str] = set()
    for model in models:
        if model == "all":
            return {"content", "lightfm_collab", "lightfm_hybrid", "lightgcn"}
        if model == "lightfm":
            expanded.update({"lightfm_collab", "lightfm_hybrid"})
        else:
            expanded.add(model)
    return expanded


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
    if "content" in selected_models:
        results["Content-Based"] = evaluate_content_based(train_df, test_df, eval_users)
    if "lightfm_collab" in selected_models:
        results["LightFM Collab"] = evaluate_lightfm_collab(
            train_df, test_df, eval_users, epochs=lightfm_epochs
        )
    if "lightfm_hybrid" in selected_models:
        results["LightFM Hybrid"] = evaluate_lightfm_hybrid(
            train_df, test_df, eval_users, epochs=lightfm_epochs
        )
    if "lightgcn" in selected_models:
        results["LightGCN"] = evaluate_lightgcn(train_df, test_df, eval_users)

    # Build comparison DataFrame
    col_order = (
        [f"P@{k}" for k in K_VALUES]
        + [f"R@{k}" for k in K_VALUES]
        + [f"NDCG@{k}" for k in K_VALUES]
        + ["RMSE", "MAE", "Infer ms", "Train s"]
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

    print("\n" + "=" * 72)
    print("  АНАЛИЗ ПРИГОДНОСТИ ДЛЯ ПРОДАКШЕНА")
    print("=" * 72)
    print(pd.DataFrame(PRODUCTION_ANALYSIS).T.to_string())

    print("\n" + "=" * 72)
    print("  ПРИМЕЧАНИЯ")
    print("=" * 72)
    print(
        "RMSE/MAE:  только для LightFM моделей. Raw logits нормализованы в [1,5].\n"
        "           Сравнивать RMSE между LightFM и GNN/Content некорректно.\n"
        "Infer ms:  время предсказания для одного пользователя (миллисекунды).\n"
        "Train s:   полное время обучения на train_df (секунды)."
    )

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommendation models.")
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated models: all, content, lightfm, lightfm_collab, "
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
