import os
import random
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from lightfm import LightFM

def build_lightfm_content_engine(movies_df):
    print("Initializing SentenceTransformer Model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Clean Title string fields to avoid alignment gaps
    movies_df['title_clean'] = movies_df['title'].astype(str).str.strip()
    movies_df = movies_df.drop_duplicates(subset=['title_clean']).reset_index(drop=True)
    
    print("Generating text embeddings (Titles + Genres)...")
    # 1. Ensure columns exist and clean them as absolute strings
    titles = movies_df['title_clean'].astype(str).str.lower().str.strip()
    genres = movies_df['genre_names'].astype(str).str.lower().str.strip() if 'genre_names' in movies_df.columns else pd.Series('', index=movies_df.index)
    
    # 2. Re-verify and replace standard pandas 'nan' string indicators with true empty strings
    titles = titles.replace({'nan': '', 'none': ''})
    genres = genres.replace({'nan': '', 'none': ''})
    
    # 3. Create a temporary DataFrame to safely join text row-by-row
    temp_df = pd.DataFrame({'t': titles, 'g': genres})
    full_texts = temp_df.agg(lambda row: f"{row['t']} {row['g']}".strip(), axis=1).tolist()
    
    # 4. Final strict type validation pass
    full_texts = [str(text) if isinstance(text, str) else "" for text in full_texts]
    
    text_embeddings = encoder.encode(full_texts, show_progress_bar=True)

    full_texts = [str(text) for text in full_texts]
    
    text_embeddings = encoder.encode(full_texts, show_progress_bar=True)
    
    # 2. Extract and Normalize Release Year
    print("Processing release year distributions...")
    if 'year' in movies_df.columns:
        movies_df['year'] = movies_df['year'].astype(float)
    else:
        movies_df['year'] = movies_df['title_clean'].str.extract(r'\((\d{4})\)').astype(float)
        
    scaler = MinMaxScaler()
    year_normalized = scaler.fit_transform(movies_df[['year']].fillna(1900)).flatten().reshape(-1, 1)
    
    # 3. Stack text features and year features into an Item-Feature Matrix
    # Shape: (num_movies, 384_text_features + 1_year_feature)
    item_features_array = np.hstack((text_embeddings, year_normalized)).astype(np.float32)
    
    # LightFM requires features passed as a Scipy Sparse CSR Matrix
    item_features_sparse = csr_matrix(item_features_array)
    
    # 4. Train LightFM using exclusively item features
    # Since we don't have a massive interaction history, we use an identity matrix 
    # for dummy interactions to force LightFM to align the feature mappings.
    print("Training LightFM on Content Features...")
    dummy_interactions = eye(len(movies_df), format='coo')
    
    # Using 'bpr' loss here as it builds strong ranking representations for content embeddings
    model = LightFM(no_components=64, learning_rate=0.05, loss='bpr', random_state=42)
    model.fit(dummy_interactions, item_features=item_features_sparse, epochs=15, num_threads=1)
    
    print("LightFM content engine ready.")
    return model, item_features_sparse, movies_df

def get_content_recommendations_lightfm(user_liked_movies, model, item_features_sparse, movies_df, n=5):
    # Map title string queries using lower-case
    title_to_idx = pd.Series(movies_df.index, index=movies_df['title_clean'].str.lower()).to_dict()
    
    liked_indices = []
    for title in user_liked_movies:
        cleaned_query = str(title).strip().lower()
        if cleaned_query in title_to_idx:
            liked_indices.append(title_to_idx[cleaned_query])
        else:
            print(f"Warning: '{title}' not found in movie catalog.")
            
    if not liked_indices:
        print("Error: None of the input movies matched.")
        return pd.DataFrame(columns=['title', 'predicted_rating'])
    
    # 5. Build a dynamic User-Feature Profile
    # Instead of an internal User ID, we declare that this user profile is defined 
    # as the average representation of the features of the items they liked.
    user_feature_vector = np.asarray(item_features_sparse[liked_indices].mean(axis=0)).flatten()
    user_features_sparse = csr_matrix(user_feature_vector) # Shape: (1, num_features)
    
    # 6. Generate ranking scores across all movies using LightFM's prediction layer
    # We pass 0 as a dummy user index, because LightFM will derive the embedding 
    # exclusively from the user_features_sparse matrix provided.
    all_item_indices = np.arange(len(movies_df), dtype=np.int32)
    dummy_user_array = np.zeros(len(movies_df), dtype=np.int32)
    
    scores = model.predict(
        user_ids=dummy_user_array,
        item_ids=all_item_indices,
        user_features=user_features_sparse,
        item_features=item_features_sparse,
        num_threads=1
    )
    
    # 7. Filter out original training items and sort
    results_df = pd.DataFrame({
        'movie_idx': all_item_indices,
        'raw_score': scores
    })
    results_df = results_df[~results_df['movie_idx'].isin(liked_indices)]
    results_df = results_df.sort_values('raw_score', ascending=False)
    
    top_results = results_df.head(n).copy()
    
    # Scale raw model ranking metrics to a standard 1-5 star scale
    top_scores = top_results['raw_score'].values
    if len(top_scores) > 1 and top_scores.max() != top_scores.min():
        rating = 1 + 4 * (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min())
    else:
        rating = np.full_like(top_scores, 3.0)
        
    final_df = pd.DataFrame({
        "title": movies_df.iloc[top_results['movie_idx']]['title_clean'].tolist(), 
        "predicted_rating": rating
    })
    return final_df

if __name__ == '__main__':
    print("Starting Standalone LightFM Content-Based Pipeline...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.normpath(os.path.join(script_dir, '../abrikos/cleaned_data/movies_clean.csv'))
    
    print(f"Reading catalog data from: {movies_path}")
    raw_movies = pd.read_csv(movies_path, low_memory=False)
    
    # Run the setup matrix building pipeline
    model, item_features, clean_movies_df = build_lightfm_content_engine(raw_movies)
    
    # --- DIAGNOSTIC: Print sample titles to see their format ---
    print("\n[Diagnostic] Sample of 3 titles from your clean catalog:")
    print(clean_movies_df['title_clean'].head(3).tolist())
    
    # Raw target movies to find
    raw_test_likes = ["Toy Story (1995)", "Jumanji (1995)"] 
    test_user_likes = []
    
    # Map raw targets to what actually exists in clean_movies_df
    catalog_titles_lower = clean_movies_df['title_clean'].str.lower().tolist()
    catalog_titles_exact = clean_movies_df['title_clean'].tolist()
    
    print("\nMatching test items against catalog...")
    for target in raw_test_likes:
        target_lower = target.strip().lower()
        
        # 1. Try exact match (ignoring case)
        if target_lower in catalog_titles_lower:
            matched_title = catalog_titles_exact[catalog_titles_lower.index(target_lower)]
            test_user_likes.append(matched_title)
            print(f"-> Exact Match Found: '{matched_title}'")
        else:
            # 2. Try partial substring match (e.g., look for "toy story" anywhere in the title)
            # Remove year brackets like "(1995)" to make substring matching easier
            clean_target = target_lower.split('(')[0].strip()
            partial_matches = [t for t in catalog_titles_exact if clean_target in t.lower()]
            
            if partial_matches:
                fallback = partial_matches[0]
                test_user_likes.append(fallback)
                print(f"-> Partial Match Found: Redirecting '{target}' to catalog title: '{fallback}'")
            else:
                # If everything fails, keep original string to let get_content_recommendations log the warning
                test_user_likes.append(target)
    
    print(f"\nGenerating top LightFM content recommendations based on finalized list: {test_user_likes}")
    recs = get_content_recommendations_lightfm(test_user_likes, model, item_features, clean_movies_df, n=5)
    
    print("\n--- LightFM Content-Based Recommendations ---")
    print(recs.round(2).to_string(index=False))
    print("\ndone")

