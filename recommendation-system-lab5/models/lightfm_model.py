"""
LightFM collaborative filtering (factorization machines, WARP loss).

Trains on data/processed/ratings.csv (columns: user_id, movie_id, rating —
movie_id is the TMDB id, the project's canonical key). Reports ROC AUC on a
held-out test split and saves an ROC curve plot next to this script.
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


def train_and_evaluate():
    ratings = load_ratings()

    shuffled = ratings.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx]
    test_df = shuffled.iloc[split_idx:]
    print(f"✓ Train: {len(train_df)} | Test: {len(test_df)}")

    dataset = LightFMDataset()
    dataset.fit(users=shuffled["user_id"].unique(), items=shuffled["movie_id"].unique())
    user_id_map, _, item_id_map, _ = dataset.mapping()

    print("Building sparse interaction matrices...")
    train_u = train_df["user_id"].map(user_id_map).values.astype(np.int32)
    train_i = train_df["movie_id"].map(item_id_map).values.astype(np.int32)
    train_w = train_df["rating"].values.astype(np.float32)

    test_u = test_df["user_id"].map(user_id_map).values.astype(np.int32)
    test_i = test_df["movie_id"].map(item_id_map).values.astype(np.int32)
    test_w = test_df["rating"].values.astype(np.float32)

    shape = (len(user_id_map), len(item_id_map))
    train_interactions = coo_matrix((train_w, (train_u, train_i)), shape=shape)
    test_interactions = coo_matrix((test_w, (test_u, test_i)), shape=shape)

    print("Training LightFM model (WARP loss)...")
    model = LightFM(
        no_components=100,
        learning_rate=0.01,
        item_alpha=0.05,
        user_alpha=0.05,
        loss="warp",
        random_state=RANDOM_SEED,
    )
    model.fit(train_interactions, epochs=5, num_threads=1, verbose=True)

    print("Evaluating ROC AUC on a sample of test users...")
    test_users_with_interactions = np.unique(test_interactions.row)
    n_sample = min(500, len(test_users_with_interactions))
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_users = rng.choice(test_users_with_interactions, size=n_sample, replace=False)

    test_coo = test_interactions.tocoo()
    mask = np.isin(test_coo.row, sampled_users)
    sampled_test_interactions = coo_matrix(
        (test_coo.data[mask], (test_coo.row[mask], test_coo.col[mask])),
        shape=test_interactions.shape,
    )

    auc = auc_score(
        model,
        sampled_test_interactions,
        train_interactions=train_interactions,
        num_threads=1,
    )
    final_auc_score = auc.mean()

    print("Extracting prediction scores for ROC curve...")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sklearn_auc

    all_labels, all_predictions = [], []
    n_items = test_interactions.shape[1]
    all_items = np.arange(n_items, dtype=np.int32)
    test_csr = test_interactions.tocsr()

    for user_idx in sampled_users[:50]:
        user_test_row = test_csr[user_idx].toarray().flatten()
        true_labels = (user_test_row > 0).astype(int)
        if true_labels.sum() == 0:
            continue
        user_array = np.full(n_items, user_idx, dtype=np.int32)
        pred_scores = model.predict(user_array, all_items, num_threads=1)
        all_labels.extend(true_labels)
        all_predictions.extend(pred_scores)

    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc_val = sklearn_auc(fpr, tpr)

    print("Generating ROC curve plot...")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (Area = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) - LightFM Model")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    output_image_path = Path(__file__).resolve().parent / "lightfm_roc_curve.png"
    plt.savefig(output_image_path, dpi=300)
    plt.close()

    print("\n--- LightFM Native Metrics ---")
    print(f"ROC AUC Score (Macro Average, {n_sample} Users): {final_auc_score:.4f}")
    print(f"ROC Graph Area Under Curve (Sampled 50 Users): {roc_auc_val:.4f}")
    print(f"ROC curve saved to: {output_image_path}")

    return model, final_auc_score, roc_auc_val


if __name__ == "__main__":
    print("Running LightFM recommendation pipeline...")
    train_and_evaluate()
    print("done")
