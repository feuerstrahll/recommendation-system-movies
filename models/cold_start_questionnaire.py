"""
Cold Start Strategy for New Users

This module implements a holistic cold start approach:
1. Mini-Questionnaire: Collects genre preferences and favorite movies
2. Initial Recommendations: Content-based filtering (no retraining needed)
3. Progressive Strategy:
   - Ratings 0-5: Pure content-based recommendations
   - Ratings 5-20: Hybrid (content + collaborative)
   - Ratings 20+: Pure collaborative filtering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
import os
from pathlib import Path


class ColdStartQuestionnaireStrategy:
    """Mini-questionnaire for new users with progressive recommendation strategy."""
    
    def __init__(self, movies_df, ratings_df=None, model=None, item_features_sparse=None):
        """
        Initialize cold start strategy.
        
        Args:
            movies_df: DataFrame with columns [movieId, title, genre_names, release_year, etc.]
            ratings_df: Optional. DataFrame with historical ratings for collaborative filtering.
            model: Optional. Pre-trained LightFM model.
            item_features_sparse: Optional. Sparse matrix of item features.
        """
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df if ratings_df is not None else pd.DataFrame()
        self.model = model
        self.item_features_sparse = item_features_sparse
        
        # Initialize text encoder if model not provided
        if self.model is None:
            print("Initializing SentenceTransformer for content-based recommendations...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self._prepare_content_features()
        else:
            self.encoder = None
    
    def _prepare_content_features(self):
        """Prepare content features for content-based recommendations."""
        print("Preparing content features...")
        
        # Clean movie titles
        self.movies_df['title_clean'] = self.movies_df['title'].astype(str).str.strip()
        
        # Generate text embeddings
        titles = self.movies_df['title_clean'].astype(str).str.lower()
        genres = self.movies_df.get('genre_names', pd.Series('', index=self.movies_df.index)).astype(str).str.lower()
        
        titles = titles.fillna('').replace({'nan': '', 'none': ''})
        genres = genres.fillna('').replace({'nan': '', 'none': ''})
        
        full_texts = (titles + ' ' + genres).str.strip()
        text_embeddings = self.encoder.encode(full_texts.tolist(), show_progress_bar=False)
        
        # Normalize release year
        year_col = self.movies_df.get('release_year', 
                                       self.movies_df.get('year', 
                                                          pd.Series(2000, index=self.movies_df.index)))
        year_values = pd.to_numeric(year_col, errors='coerce').fillna(2000).values.reshape(-1, 1)
        
        scaler = MinMaxScaler()
        year_normalized = scaler.fit_transform(year_values)
        
        # Stack features
        self.item_features = np.hstack((text_embeddings, year_normalized)).astype(np.float32)
        print(f"Content features prepared: {self.item_features.shape}")
    
    def run_questionnaire(self):
        """
        Run interactive mini-questionnaire for new user.
        
        Returns:
            dict: User profile with:
                - favorite_genres: list of preferred genres
                - favorite_movies: list of movie IDs user likes
                - age_preference: 'classic', 'modern', 'any'
        """
        print("\n" + "="*70)
        print("🎬 WELCOME! LET'S FIND YOUR PERFECT MOVIES")
        print("="*70)
        
        profile = {
            'favorite_genres': self._ask_genres(),
            'favorite_movies': self._ask_favorite_movies(),
            'age_preference': self._ask_year_preference()
        }
        
        return profile
    
    def _ask_genres(self):
        """Ask user to select favorite genres."""
        print("\n📚 STEP 1: Select your favorite genres")
        print("-" * 70)
        
        # Extract unique genres
        all_genres = set()
        for genre_str in self.movies_df.get('genre_names', pd.Series()).fillna('').astype(str):
            if genre_str and genre_str.lower() != 'nan':
                all_genres.update(g.strip() for g in str(genre_str).split('|'))
        
        genres_list = sorted(list(all_genres))[:15]  # Limit to 15 for simplicity
        
        print("Available genres:")
        for i, genre in enumerate(genres_list, 1):
            print(f"  {i:2d}. {genre}")
        
        print("\nEnter genre numbers separated by commas (e.g., 1,3,5):")
        print("Or press Enter to skip (default: all genres):")
        
        try:
            user_input = input("> ").strip()
            if not user_input:
                return genres_list
            
            indices = [int(x.strip()) - 1 for x in user_input.split(',')]
            selected = [genres_list[i] for i in indices if 0 <= i < len(genres_list)]
            return selected if selected else genres_list
        except:
            print("Invalid input. Using all genres.")
            return genres_list
    
    def _ask_favorite_movies(self):
        """Ask user to name favorite movies."""
        print("\n🍿 STEP 2: What are 2-5 of your favorite movies?")
        print("-" * 70)
        
        print("(Type movie titles, one per line. Press Enter twice when done)")
        favorite_movies = []
        while len(favorite_movies) < 5:
            title = input(f"Movie {len(favorite_movies) + 1}: ").strip()
            
            if not title:
                if favorite_movies:
                    break
                else:
                    print("Please enter at least one movie.")
                    continue
            
            # Find matching movie
            title_lower = title.lower()
            matches = self.movies_df[
                self.movies_df['title'].astype(str).str.lower().str.contains(title_lower, na=False)
            ]
            
            if not matches.empty:
                matched_movie = matches.iloc[0]
                favorite_movies.append({
                    'movieId': matched_movie['movieId'],
                    'title': matched_movie['title']
                })
                print(f"  ✓ Added: {matched_movie['title']}")
            else:
                print(f"  ✗ '{title}' not found. Try another title.")
        
        return [m['movieId'] for m in favorite_movies]
    
    def _ask_year_preference(self):
        """Ask user preference for movie release year."""
        print("\n📅 STEP 3: Do you prefer older or newer movies?")
        print("-" * 70)
        print("  1. Classic (before 2000)")
        print("  2. Modern (2000-2015)")
        print("  3. Recent (after 2015)")
        print("  4. No preference")
        
        try:
            choice = input("Choose (1-4): ").strip()
            preferences = {'1': 'classic', '2': 'modern', '3': 'recent', '4': 'any'}
            return preferences.get(choice, 'any')
        except:
            return 'any'
    
    def generate_initial_recommendations(self, user_profile, n_recommendations=10):
        """
        Generate initial recommendations based on user profile and favorite movies.
        
        Strategy: Content-based (movie similarity to favorites)
        
        Args:
            user_profile: dict from run_questionnaire()
            n_recommendations: number of movies to recommend
            
        Returns:
            DataFrame with recommendations
        """
        print("\n" + "="*70)
        print("🎯 GENERATING YOUR INITIAL RECOMMENDATIONS...")
        print("="*70)
        
        favorite_movie_ids = user_profile.get('favorite_movies', [])
        
        if not favorite_movie_ids:
            print("No favorite movies found. Recommending popular movies...")
            return self._get_popular_movies(n_recommendations)
        
        # Find indices of favorite movies
        favorite_indices = []
        for movie_id in favorite_movie_ids:
            idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
            if len(idx) > 0:
                favorite_indices.append(idx[0])
        
        if not favorite_indices:
            return self._get_popular_movies(n_recommendations)
        
        # Build user profile = average of favorite movie features
        user_features = self.item_features[favorite_indices].mean(axis=0)
        
        # Compute similarity (cosine distance) to all movies
        similarities = np.dot(self.item_features, user_features)
        
        # Get top recommendations (excluding favorites)
        top_indices = np.argsort(similarities)[::-1]
        recommendations = []
        
        for idx in top_indices:
            if idx not in favorite_indices and len(recommendations) < n_recommendations:
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'movieId': movie['movieId'],
                    'title': movie.get('title', 'Unknown'),
                    'genres': movie.get('genre_names', ''),
                    'similarity_score': float(similarities[idx]),
                    'strategy': 'content-based'
                })
        
        result_df = pd.DataFrame(recommendations)
        
        print(f"\n✓ Generated {len(result_df)} recommendations")
        print("\nTop recommendations for you:")
        print("-" * 70)
        for i, (_, row) in enumerate(result_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['title']} ({row['genres']})")
        
        return result_df
    
    def _get_popular_movies(self, n=10):
        """Get popular movies as fallback."""
        # Try to use popularity column if exists
        if 'popularity' in self.movies_df.columns:
            top_movies = self.movies_df.nlargest(n, 'popularity')
        elif 'vote_average' in self.movies_df.columns:
            top_movies = self.movies_df.nlargest(n, 'vote_average')
        else:
            top_movies = self.movies_df.head(n)
        
        return pd.DataFrame({
            'movieId': top_movies['movieId'],
            'title': top_movies['title'],
            'genres': top_movies.get('genre_names', ''),
            'strategy': 'popular'
        })
    
    def recommend_for_new_user(self, user_profile, n_ratings_history=0):
        """
        Get recommendations based on user lifecycle stage.
        
        Strategy:
        - 0-5 ratings: Content-based (no collaboration yet)
        - 5-20 ratings: Hybrid (content + collaborative)
        - 20+ ratings: Pure collaborative filtering
        
        Args:
            user_profile: User's profile dict
            n_ratings_history: Number of ratings user has already given
            
        Returns:
            DataFrame with recommendations and strategy info
        """
        
        if n_ratings_history < 5:
            strategy = "content-based"
            print(f"\n📍 Cold Start Stage 1 (ratings: {n_ratings_history}/5)")
            print("Strategy: Pure content-based recommendations")
        elif n_ratings_history < 20:
            strategy = "hybrid"
            print(f"\n📍 Warm Start Stage 2 (ratings: {n_ratings_history}/20)")
            print("Strategy: Hybrid (content + collaborative)")
        else:
            strategy = "collaborative"
            print(f"\n📍 Full Cold Start Over! (ratings: {n_ratings_history})")
            print("Strategy: Pure collaborative filtering")
        
        recommendations = self.generate_initial_recommendations(user_profile)
        recommendations['recommendation_strategy'] = strategy
        
        return recommendations


def run_interactive_cold_start():
    """Run the complete interactive cold start flow."""
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    if not data_dir.exists():
        print("❌ Data directory not found. Please ensure data is in data/processed/")
        return
    
    print("Loading movie catalog...")
    movies = pd.read_csv(data_dir / "movies.csv", low_memory=False)
    
    if data_dir.joinpath("ratings.csv").exists():
        ratings = pd.read_csv(data_dir / "ratings.csv", nrows=100000)
    else:
        ratings = None
    
    # Initialize strategy
    strategy = ColdStartQuestionnaireStrategy(movies, ratings)
    
    # Run questionnaire
    user_profile = strategy.run_questionnaire()
    
    # Generate recommendations
    recommendations = strategy.generate_initial_recommendations(user_profile, n_recommendations=10)
    
    print("\n" + "="*70)
    print("✓ COLD START COMPLETE")
    print("="*70)
    print(f"\nRecommendations for new user:")
    print(recommendations[['title', 'genres', 'strategy']])
    
    return strategy, user_profile, recommendations


if __name__ == '__main__':
    strategy, profile, recs = run_interactive_cold_start()
