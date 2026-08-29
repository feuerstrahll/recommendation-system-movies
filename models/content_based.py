"""
Content-Based Recommendation Strategy using SentenceTransformer embeddings.

This is the content-based component of the laboratory work, compared against:
  - SVD (classic collaborative filtering)
  - LightFM (ranking-based collaborative filtering, WARP loss)
  - LightGCN (graph-based collaborative filtering)

Content-based approach:
  - Encodes movie metadata (title + genres + year) into dense embeddings
  - Builds a user profile as the mean of liked items' embeddings
  - Ranks all catalog items by cosine similarity to the user profile
  - Optionally uses FAISS for fast approximate nearest neighbor search

This method can recommend cold-start items (movies never seen during
training), which is its main advantage over collaborative filtering methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Optional: FAISS for fast retrieval on large catalogs
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def build_content_engine(movies_df, use_faiss=True):
    """
    Build the content-based recommendation engine.

    Returns:
        item_features_array: np.ndarray of shape (num_movies, num_features)
        movies_df: cleaned DataFrame with title_clean and year columns
        faiss_index: optional FAISS index (None if FAISS unavailable or disabled)
    """
    print("Initializing SentenceTransformer Model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # 1. Clean title field
    movies_df['title_clean'] = movies_df['title'].astype(str).str.strip()
    movies_df = movies_df.drop_duplicates(subset=['title_clean']).reset_index(drop=True)

    # 2. Build text for embedding: title + genres
    print("Generating text embeddings (Titles + Genres)...")
    titles = movies_df['title_clean'].astype(str).str.lower().str.strip()
    genres = (
        movies_df['genre_names'].astype(str).str.lower().str.strip()
        if 'genre_names' in movies_df.columns
        else pd.Series('', index=movies_df.index)
    )

    titles = titles.replace({'nan': '', 'none': ''})
    genres = genres.replace({'nan': '', 'none': ''})

    full_texts = (titles + ' ' + genres).str.strip().tolist()

    text_embeddings = encoder.encode(full_texts, show_progress_bar=True)

    # 3. Extract and normalize release year
    print("Processing release year...")
    if 'release_year' in movies_df.columns:
        movies_df['year'] = pd.to_numeric(movies_df['release_year'], errors='coerce')
    elif 'year' in movies_df.columns:
        movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    else:
        movies_df['year'] = (
            movies_df['title_clean']
            .str.extract(r'\((\d{4})\)')
            .astype(float)
        )

    scaler = MinMaxScaler()
    year_normalized = scaler.fit_transform(
        movies_df[['year']].fillna(1900)
    ).astype(np.float32)

    # 4. Combine text embeddings + year into a single feature matrix
    item_features_array = np.hstack((
        text_embeddings.astype(np.float32),
        year_normalized
    ))

    # 5. Optionally build FAISS index for fast retrieval
    faiss_index = None
    if use_faiss and FAISS_AVAILABLE:
        print("Building FAISS index...")
        # Normalize embeddings for inner product = cosine similarity
        normalized = item_features_array.copy()
        faiss.normalize_L2(normalized)

        dim = normalized.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
        faiss_index.add(normalized)
        print(f"FAISS index built with {faiss_index.ntotal} items.")
    elif use_faiss and not FAISS_AVAILABLE:
        print("FAISS not installed — falling back to brute-force cosine similarity "
              "(pip install faiss-cpu to enable it).")

    print("Content engine ready.")
    return item_features_array, movies_df, faiss_index


def get_content_recommendations(user_liked_movies, item_features_array, movies_df,
                                 faiss_index=None, n=5):
    """
    Generate content-based recommendations for a user.

    Args:
        user_liked_movies: list of movie titles the user liked
        item_features_array: embedding matrix for all movies
        movies_df: cleaned movie catalog
        faiss_index: optional FAISS index for fast search
        n: number of recommendations to return

    Returns:
        DataFrame with columns [title, similarity_score]
    """
    # Map titles to indices
    title_to_idx = pd.Series(
        movies_df.index,
        index=movies_df['title_clean'].str.lower()
    ).to_dict()

    liked_indices = []
    for title in user_liked_movies:
        cleaned_query = str(title).strip().lower()
        if cleaned_query in title_to_idx:
            liked_indices.append(title_to_idx[cleaned_query])
        else:
            print(f"Warning: '{title}' not found in movie catalog.")

    if not liked_indices:
        print("Error: None of the input movies matched.")
        return pd.DataFrame(columns=['title', 'similarity_score'])

    # Build user profile: mean of liked items' embeddings
    user_vector = item_features_array[liked_indices].mean(axis=0).reshape(1, -1)

    # Score all items
    if faiss_index is not None:
        # Use FAISS for fast search
        query = user_vector.copy().astype(np.float32)
        faiss.normalize_L2(query)

        # Search more than needed to account for filtering liked items
        search_k = min(n + len(liked_indices) + 50, len(movies_df))
        scores, indices = faiss_index.search(query, search_k)
        scores = scores.flatten()
        indices = indices.flatten()
    else:
        # Use cosine similarity (slower but no extra dependency)
        scores = cosine_similarity(user_vector, item_features_array).flatten()
        indices = np.arange(len(movies_df))
        order = np.argsort(scores)[::-1]
        indices = indices[order]
        scores = scores[order]

    # Build results, excluding liked movies
    liked_indices_set = set(liked_indices)
    results = []
    for idx, score in zip(indices, scores):
        if idx not in liked_indices_set:
            results.append({'movie_idx': idx, 'similarity_score': float(score)})
        if len(results) >= n:
            break

    if not results:
        return pd.DataFrame(columns=['title', 'similarity_score'])

    results_df = pd.DataFrame(results)
    results_df['title'] = movies_df.iloc[results_df['movie_idx']]['title_clean'].values

    return results_df[['title', 'similarity_score']]


if __name__ == '__main__':
    print("Starting Content-Based Recommendation Pipeline...")

    movies_path = DATA_DIR / "movies.csv"
    if not movies_path.exists():
        movies_path = DATA_DIR / "movies_clean.csv"

    print(f"Reading catalog data from: {movies_path}")
    raw_movies = pd.read_csv(movies_path, low_memory=False)

    # Build the content engine
    item_features, clean_movies_df, faiss_index = build_content_engine(
        raw_movies, use_faiss=True
    )

    # Diagnostic
    print("\n[Diagnostic] Sample of 3 titles from catalog:")
    print(clean_movies_df['title_clean'].head(3).tolist())

    # Test movies
    raw_test_likes = ["Toy Story", "Jumanji"]
    test_user_likes = []

    catalog_titles_lower = clean_movies_df['title_clean'].str.lower().tolist()
    catalog_titles_exact = clean_movies_df['title_clean'].tolist()

    print("\nMatching test items against catalog...")
    for target in raw_test_likes:
        target_lower = target.strip().lower()
        if target_lower in catalog_titles_lower:
            matched_title = catalog_titles_exact[catalog_titles_lower.index(target_lower)]
            test_user_likes.append(matched_title)
            print(f"-> Exact Match Found: '{matched_title}'")
        else:
            clean_target = target_lower.split('(')[0].strip()
            partial_matches = [t for t in catalog_titles_exact if clean_target in t.lower()]
            if partial_matches:
                fallback = partial_matches[0]
                test_user_likes.append(fallback)
                print(f"-> Partial Match Found: '{target}' -> '{fallback}'")
            else:
                test_user_likes.append(target)

    print(f"\nGenerating recommendations based on: {test_user_likes}")
    recs = get_content_recommendations(
        test_user_likes, item_features, clean_movies_df,
        faiss_index=faiss_index, n=10
    )

    print("\n--- Content-Based Recommendations ---")
    print(recs.round(4).to_string(index=False))
    print("\ndone")
