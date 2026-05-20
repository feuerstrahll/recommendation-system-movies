import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset
from lightfm.evaluation import auc_score
import scipy
# Mock placeholders - replace these with your actual database reading logic
# df2 = pd.read_csv("movies.csv")
# ratings = pd.read_csv("ratings.csv")

def get_ratings_lightfm(ratings_df, movies_df, n_reviews=5):
    new_user_id = int(ratings_df['userId'].max() + 1) 

    sample_movies = movies_df.sample(n=100)
    random_movie_ids = sample_movies['movieId'].tolist() 

    new_user_ratings = [
        (new_user_id, random_movie_ids[i], float(random.randint(1, 5)))
        for i in range(n_reviews)
    ]

    new_user_ratings_df = pd.DataFrame(new_user_ratings, columns=['userId', 'movieId', 'rating'])
    updated_ratings = pd.concat([ratings_df, new_user_ratings_df], ignore_index=True)

    # Use explicit 32-bit types to protect C memory structures
    user_pool = updated_ratings['userId'].unique().astype(np.int32)
    item_pool = updated_ratings['movieId'].unique().astype(np.int32)

    dataset = LightFMDataset()
    dataset.fit(users=user_pool, items=item_pool)

    tuple_list = [
        (int(row.userId), int(row.movieId), float(row.rating)) 
        for row in updated_ratings.itertuples(index=False)
    ]
    (interactions, _) = dataset.build_interactions(tuple_list)
    interactions = interactions.tocsr()
    
    model = LightFM(no_components=100, learning_rate=0.01, item_alpha=0.05, user_alpha=0.05, loss='warp')
    
    # CRITICAL: Keep num_threads=1 for Windows script stability
    model.fit(interactions, epochs=30, num_threads=4)

    user_id_map, _, item_id_map, _ = dataset.mapping()
    internal_user_index = user_id_map[new_user_id]

    rated_items = set(new_user_ratings_df['movieId'])
    unrated_items = list(set(item_pool) - rated_items)

    internal_item_indices = np.array([item_id_map[item_id] for item_id in unrated_items], dtype=np.int32)
    user_array = np.full(len(internal_item_indices), internal_user_index, dtype=np.int32)

    predictions = model.predict(user_array, internal_item_indices, num_threads=1)

    results_df = pd.DataFrame({'movieId': unrated_items, 'lightfm_score': predictions})
    results_df = results_df.sort_values('lightfm_score', ascending=False)
    
    top_recs_df = results_df.head(10).merge(movies_df[['movieId', 'title']], on='movieId', how='left')
    print(top_recs_df[['title', 'lightfm_score']].round(2).to_string(index=False))

# Windows process isolation safety block
if __name__ == '__main__':
    print("Running LightFM recommendation pipeline...")

    movies = pd.read_csv('../abrikos/cleaned_data/movies_clean.csv', low_memory=False)
    ratings = pd.read_csv('../abrikos/cleaned_data/ratings_clean.csv', usecols=['userId', 'movieId', 'rating'], nrows=1_000_000)
    movies[movies['imdb_id'] == 'tt0114709']
    links = pd.read_csv('../abrikos/data_Movies/links.csv')
    movies1 = movies.copy()
    movies1['imdb_id'] = movies1['imdb_id'].apply(lambda x: str(x).lstrip('tt0'))
    movies1['imdb_id'] = pd.to_numeric(movies1['imdb_id'], errors='coerce')
    df1 = movies1.merge(links, left_on='imdb_id', right_on='imdbId')
    df1 = df1.drop_duplicates(subset=['title'])
    # keep only df1 rows that have movieId in ratings
    df2 = df1[df1['movieId'].isin(ratings['movieId'])]
    # keep only ratings rows that have movieId in df1
    ratings = ratings[ratings['movieId'].isin(df2['movieId'])]

    #get_ratings_lightfm(ratings, df2, n_reviews=1)
    print('done')

    shuffled_ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled_ratings) * 0.8)
    train_df = shuffled_ratings.iloc[:split_idx]
    test_df = shuffled_ratings.iloc[split_idx:]
    
    # 2. Fit mappings
    dataset = LightFMDataset()
    dataset.fit(users=shuffled_ratings['userId'].unique(), items=shuffled_ratings['movieId'].unique())
    
    # --- FIXED: Lightning Fast Matrix Conversion ---
    print("Building sparse interaction matrices (Vectorized style)...")
    user_id_map, _, item_id_map, _ = dataset.mapping()
    
    # Use pandas .map() to vectorize mapping instead of slow itertuples loop
    train_u_indices = train_df['userId'].map(user_id_map).values.astype(np.int32)
    train_i_indices = train_df['movieId'].map(item_id_map).values.astype(np.int32)
    train_weights = train_df['rating'].values.astype(np.float32)
    
    test_u_indices = test_df['userId'].map(user_id_map).values.astype(np.int32)
    test_i_indices = test_df['movieId'].map(item_id_map).values.astype(np.int32)
    test_weights = test_df['rating'].values.astype(np.float32)
    
    # Build the coo interaction format directly bypassing build_interactions
    from scipy.sparse import coo_matrix
    shape = (len(user_id_map), len(item_id_map))
    train_interactions = coo_matrix((train_weights, (train_u_indices, train_i_indices)), shape=shape)
    test_interactions = coo_matrix((test_weights, (test_u_indices, test_i_indices)), shape=shape)
    
    # 3. Model Training
    print("Training LightFM Model...")
    model = LightFM(no_components=100, learning_rate=0.01, item_alpha=0.05, user_alpha=0.05, loss='warp', random_state=42)
    
    # Added verbose=True to show progress bars per training epoch
    model.fit(train_interactions, epochs=5, num_threads=1, verbose=True)
    
                 # 4. Metric Evaluation & Plotting
    print("Evaluating LightFM Native AUC Metric (Optimized Sampling)...")
    
    # 1. Find all unique users in the test set
    test_users_with_interactions = np.unique(test_interactions.row)
    
    # 2. Pick a random sample of 500 users
    np.random.seed(42)
    sampled_users = np.random.choice(test_users_with_interactions, size=500, replace=False)
    
    # 3. Create a boolean mask to filter test matrix data rows quickly
    test_coo = test_interactions.tocoo()
    mask = np.isin(test_coo.row, sampled_users)
    
    from scipy.sparse import coo_matrix
    sampled_test_interactions = coo_matrix(
        (test_coo.data[mask], (test_coo.row[mask], test_coo.col[mask])),
        shape=test_interactions.shape  # Essential: Maintains the original shape!
    )
    
    # 4. Calculate AUC swiftly on the sparse slice
    auc = auc_score(
        model, 
        sampled_test_interactions, 
        train_interactions=train_interactions, # Pass the original intact train matrix
        num_threads=1
    )
    
    # The 'auc' array contains exactly one entry per user present in 'sampled_test_interactions'
    final_auc_score = auc.mean()
    
    # --- Generate Global ROC Curve Data Arrays ---
    print("Extracting prediction scores for ROC Curve generation...")
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sklearn_auc
    
    all_labels = []
    all_predictions = []
    
    # Build complete item index pool for predictions
    n_items = test_interactions.shape[1]
    all_items = np.arange(n_items, dtype=np.int32)
    
    # Extract binary labels and prediction scores user by user
    # Convert test_interactions to CSR once for ultra-fast row slicing
    test_csr = test_interactions.tocsr()
    
    for user_id in sampled_users[:50]:
        # Fast sparse row retrieval
        user_test_row = test_csr[user_id].toarray().flatten()
        true_labels = (user_test_row > 0).astype(int)
        
        # Skip user if they have zero ground-truth positive interactions
        if np.sum(true_labels) == 0:
            continue
            
        # Get raw ranking prediction scores from LightFM
        user_array = np.full(n_items, user_id, dtype=np.int32)
        pred_scores = model.predict(user_array, all_items, num_threads=1)
        
        all_labels.extend(true_labels)
        all_predictions.extend(pred_scores)
        
    # Compute global micro-ROC curve points
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc_val = sklearn_auc(fpr, tpr)
    
    # --- Plot and Render ROC Graphical Layout ---
    print("Generating ROC Curve plot...")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) - LightFM Model')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save chart as PNG graphic in current working directory
    output_image_path = "lightfm_roc_curve.png"
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    
    print("\n--- LightFM Native Metrics ---")
    print(f"ROC AUC Score (Macro Average 500 Users): {final_auc_score:.4f}")
    print(f"ROC Graph Area Under Curve (Sampled 50 Users): {roc_auc_val:.4f}")
    print(f"Success! ROC Curve graphic saved to: {os.path.abspath(output_image_path)}")
    print("done")

