"""
LightFM collaborative filtering (factorization machines, WARP loss).

Trains on data/processed/ratings.csv (columns: user_id, movie_id, rating —
movie_id is the TMDB id, the project's canonical key). Uses the same
experimental protocol as models/lightgcn_model.py, so the two are directly
comparable:
  - implicit feedback: rating >= 4.0 counts as a positive interaction
  - users with < 5 positive interactions are dropped
  - 80/20 train/test split PER USER (not a global random split), so every
    user has both train and test interactions
  - primary metrics: Precision@K, Recall@K, NDCG@K on the held-out test set
  - test items unseen in train are excluded from ground truth (cold-start
    items — an item embedding-based model can never recommend them)

ROC AUC is reported as a secondary diagnostic only (see note at the bottom):
"unobserved" items are treated as negatives for AUC, which is a standard but
imperfect proxy for implicit-feedback ranking quality.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset
from lightfm.evaluation import auc_score
from scipy.sparse import coo_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RANDOM_SEED = 42
K_VALUES = [10, 20]


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
    """
    Same semantics as lightgcn_model.py: rating >= 4.0 -> positive interaction,
    everything else dropped. Keeps only users with enough positives to have
    both train and test rows after an 80/20 per-user split.
    """
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
# Training & ranking evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(min_user_interactions=5, epochs=5):
    ratings = load_ratings()
    positives = prepare_implicit_interactions(ratings, min_user_interactions)
    train_df, test_df = split_per_user(positives, seed=RANDOM_SEED)

    dataset = LightFMDataset()
    dataset.fit(users=positives["user_id"].unique(), items=positives["movie_id"].unique())
    user_id_map, _, item_id_map, _ = dataset.mapping()
    item_inv = {v: k for k, v in item_id_map.items()}

    print("Building sparse interaction matrix (train only)...")
    train_u = train_df["user_id"].map(user_id_map).values.astype(np.int32)
    train_i = train_df["movie_id"].map(item_id_map).values.astype(np.int32)
    shape = (len(user_id_map), len(item_id_map))
    train_interactions = coo_matrix((np.ones(len(train_u), dtype=np.float32), (train_u, train_i)), shape=shape)

    print("Training LightFM model (WARP loss, implicit feedback)...")
    model = LightFM(
        no_components=64,
        learning_rate=0.01,
        item_alpha=0.05,
        user_alpha=0.05,
        loss="warp",
        random_state=RANDOM_SEED,
    )
    model.fit(train_interactions, epochs=epochs, num_threads=os.cpu_count() or 1, verbose=True)

    # Test items unseen in train can never be recommended (no item embedding) —
    # drop them from ground truth, same as lightgcn_model.py's main().
    test_df_known = test_df[test_df["movie_id"].isin(item_id_map.keys())]
    n_dropped = len(test_df) - len(test_df_known)
    if n_dropped > 0:
        print(f"⚠️  Dropped {n_dropped} test interactions on items unseen in train (cold-start items)")
    test_gt = test_df_known.groupby("user_id")["movie_id"].apply(set).to_dict()
    train_seen = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    print(f"\n📊 Ranking evaluation on held-out test set ({len(test_gt)} users)...")
    all_item_idx = np.arange(len(item_id_map), dtype=np.int32)
    per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}

    for user_id, relevant in test_gt.items():
        if user_id not in user_id_map or not relevant:
            continue
        u_idx = user_id_map[user_id]
        scores = model.predict(np.full(len(all_item_idx), u_idx, dtype=np.int32), all_item_idx, num_threads=os.cpu_count() or 1)

        seen = train_seen.get(user_id, set())
        order = np.argsort(scores)[::-1]
        ranked = [item_inv[i] for i in order if item_inv[i] not in seen]

        for k in K_VALUES:
            per_k[k]["precision"].append(precision_at_k(ranked, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))

    print("\n" + "=" * 60)
    print("📈 LightFM EVALUATION RESULTS (held-out test set)")
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

    # --- Secondary diagnostic: ROC AUC ---
    # Treats every unobserved (user, item) pair as a negative, which is a
    # standard but imperfect proxy for implicit feedback (unobserved != truly
    # disliked). Not used as the primary comparison metric against LightGCN.
    print("\n--- Secondary diagnostic: ROC AUC (unobserved = negative) ---")
    test_u = test_df_known["user_id"].map(user_id_map).values.astype(np.int32)
    test_i = test_df_known["movie_id"].map(item_id_map).values.astype(np.int32)
    test_interactions = coo_matrix((np.ones(len(test_u), dtype=np.float32), (test_u, test_i)), shape=shape)

    auc = auc_score(model, test_interactions, train_interactions=train_interactions, num_threads=os.cpu_count() or 1)
    final_auc_score = float(auc.mean())
    print(f"ROC AUC Score (macro average over {len(auc)} users): {final_auc_score:.4f}")

    print("\n" + "=" * 60)
    return model, metrics, final_auc_score


if __name__ == "__main__":
    print("Running LightFM recommendation pipeline...")
    train_and_evaluate()
    print("done")
