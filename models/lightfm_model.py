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

    predictions = model.predict(user_array, internal_item_indices, num_threads=4)

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
    model.fit(train_interactions, epochs=5, num_threads=4, verbose=True)
    
    # 4. Metric Evaluation
    print("Evaluating LightFM Native AUC Metric...")
    # --- FIXED: Limit user count to prevent memory stalling ---
    # We sample evaluation on 2000 users. This provides a statistically stable 
    # score while cutting down evaluation time from hours to seconds.
    test_users_with_interactions = np.unique(test_interactions.row)
    if len(test_users_with_interactions) > 2000:
        sampled_users = np.random.choice(test_users_with_interactions, size=2000, replace=False).astype(np.int32)
    else:
        sampled_users = test_users_with_interactions.astype(np.int32)
        
    auc = auc_score(
        model, 
        test_interactions, 
        train_interactions=train_interactions, 
        user_features=None, 
        item_features=None,
        num_threads=4
    )
    
    # Calculate the mean score only for our targeted user sample group
    final_auc_score = auc[sampled_users].mean()
    
    print("\n--- LightFM Native Metrics ---")
    print(f"ROC AUC Score (Sampled): {final_auc_score:.4f}")
    print("done")
