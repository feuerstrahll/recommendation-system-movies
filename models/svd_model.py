"""
SVD collaborative filtering baseline (matrix factorization).

Classic latent-factor approach: truncated SVD on the user-item interaction
matrix. Uses the same implicit-feedback protocol as lightfm_model.py and
lightgcn_model.py, so the three are directly comparable:
  - implicit feedback: rating >= 4.0 counts as a positive interaction
  - users with < 5 positive interactions are dropped
  - 80/20 train/test split PER USER
  - primary metrics: Precision@K, Recall@K, NDCG@K on the held-out test set
  - test items unseen in train are excluded from ground truth (cold-start
    items — a factorized model has no latent vector for them)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RANDOM_SEED = 42
K_VALUES = [10, 20]
N_FACTORS = 50


# ---------------------------------------------------------------------------
# Metrics (same definitions as evaluation/metrics.py, kept local to avoid a
# cross-package import from models/ into evaluation/)
# ---------------------------------------------------------------------------

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(recommended_k)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended_k)
        if item in relevant
    )
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_ratings(max_rows=None):
    """Load ratings.csv (user_id, movie_id, rating), TMDB-keyed movie_id."""
    path = DATA_DIR / "ratings.csv"
    if path.exists():
        df = pd.read_csv(path, usecols=["user_id", "movie_id", "rating"], nrows=max_rows)
    else:
        print(f"⚠️  {path} not found, trying ratings_clean.csv")
        path = DATA_DIR / "ratings_clean.csv"
        df = pd.read_csv(path, usecols=["userId", "movieId", "rating"], nrows=max_rows)
        df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    df = df.dropna(subset=["user_id", "movie_id", "rating"])
    df["user_id"] = df["user_id"].astype(np.int32)
    df["movie_id"] = df["movie_id"].astype(np.int32)
    print(f"✓ Loaded {len(df)} ratings from {path.name}")
    return df


def prepare_implicit_interactions(ratings, min_user_interactions=5):
    """rating >= 4.0 -> positive interaction; users with too few dropped."""
    positives = ratings[ratings["rating"] >= 4.0][["user_id", "movie_id"]].drop_duplicates()
    print(f"✓ Filtered to {len(positives)} positive interactions (rating >= 4.0)")

    user_counts = positives["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    positives = positives[positives["user_id"].isin(valid_users)].copy()
    print(f"✓ Kept {len(positives)} interactions from {positives['user_id'].nunique()} "
          f"users with >= {min_user_interactions} interactions")
    return positives


def split_per_user(positives, seed=RANDOM_SEED):
    """80/20 train/test split within each user's interactions."""
    train_rows, test_rows = [], []
    rng = np.random.RandomState(seed)
    for _, udf in positives.groupby("user_id"):
        shuffled = udf.sample(frac=1, random_state=rng)
        n_train = max(1, int(len(shuffled) * 0.8))
        train_rows.append(shuffled.iloc[:n_train])
        test_rows.append(shuffled.iloc[n_train:])
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    print(f"✓ Train: {len(train_df)} interactions | Test: {len(test_df)} interactions")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_svd(train_df, n_factors=N_FACTORS):
    """
    Fit truncated SVD on the binarized (0/1) train interaction matrix.

    Returns:
        user_factors, item_factors: latent factor matrices
        user_map, item_map, item_inv: id <-> matrix-index mappings
    """
    all_users = train_df["user_id"].unique()
    all_items = train_df["movie_id"].unique()
    n_users = len(all_users)
    n_items = len(all_items)

    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}
    item_inv = {i: m for m, i in item_map.items()}

    u_idx = train_df["user_id"].map(user_map).values.astype(np.int32)
    i_idx = train_df["movie_id"].map(item_map).values.astype(np.int32)
    vals = np.ones(len(train_df), dtype=np.float32)
    matrix = csr_matrix((vals, (u_idx, i_idx)), shape=(n_users, n_items))

    k = min(n_factors, min(matrix.shape) - 1)
    u, s, vt = svds(matrix, k=k)
    user_factors = u * s
    item_factors = vt.T

    return user_factors, item_factors, user_map, item_map, item_inv


# ---------------------------------------------------------------------------
# Ranking evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(min_user_interactions=5, n_factors=N_FACTORS):
    import time

    ratings = load_ratings()
    positives = prepare_implicit_interactions(ratings, min_user_interactions)
    train_df, test_df = split_per_user(positives, seed=RANDOM_SEED)

    print(f"\nTraining SVD (k={min(n_factors, len(train_df['user_id'].unique()) - 1)} latent factors)...")
    t0 = time.perf_counter()
    user_factors, item_factors, user_map, item_map, item_inv = train_svd(train_df, n_factors)
    train_time = time.perf_counter() - t0
    print(f"✓ Trained in {train_time:.1f}s")

    # Test items unseen in train have no latent vector — drop from ground truth,
    # same as lightfm_model.py / lightgcn_model.py.
    test_df_known = test_df[test_df["movie_id"].isin(item_map.keys())]
    n_dropped = len(test_df) - len(test_df_known)
    if n_dropped > 0:
        print(f"⚠️  Dropped {n_dropped} test interactions on items unseen in train (cold-start items)")
    test_gt = test_df_known.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    print(f"\n📊 Ranking evaluation on held-out test set ({len(test_gt)} users)...")
    per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}

    for user_id, relevant in test_gt.items():
        if user_id not in user_map or not relevant:
            continue
        u_idx = user_map[user_id]
        scores = item_factors @ user_factors[u_idx]

        seen = train_seen.get(user_id, set())
        order = np.argsort(scores)[::-1]
        ranked = [item_inv[i] for i in order if item_inv[i] not in seen]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))

    print("\n" + "=" * 60)
    print("📈 SVD EVALUATION RESULTS (held-out test set)")
    print("=" * 60)
    metrics = {}
    for k in K_VALUES:
        p = float(np.mean(per_k[k]["precision"])) if per_k[k]["precision"] else 0.0
        r = float(np.mean(per_k[k]["recall"])) if per_k[k]["recall"] else 0.0
        n = float(np.mean(per_k[k]["ndcg"])) if per_k[k]["ndcg"] else 0.0
        metrics[k] = {"Precision@K": p, "Recall@K": r, "NDCG@K": n}
        print(f"\n🎯 Metrics @ K={k}:")
        print(f"  Precision@K    : {p:.4f}")
        print(f"  Recall@K       : {r:.4f}")
        print(f"  NDCG@K         : {n:.4f}")

    print("\n" + "=" * 60)
    return {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_map": user_map,
        "item_map": item_map,
        "item_inv": item_inv,
    }, metrics


if __name__ == "__main__":
    print("Running SVD recommendation pipeline...")
    train_and_evaluate()
    print("done")
