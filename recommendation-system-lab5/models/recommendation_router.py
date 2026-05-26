"""
Unified Recommendation Router

Coordinates between different recommendation strategies based on user lifecycle:
1. NEW USER (0 ratings) → Mini-Questionnaire + Content-Based
2. WARM START (1-20 ratings) → Hybrid (Content + Collaborative)
3. MATURE USER (20+ ratings) → Pure Collaborative Filtering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class UserStage(Enum):
    """User lifecycle stages."""
    NEW = "new"              # 0 ratings
    COLD_START = "cold_start"  # 1-5 ratings
    WARM_START = "warm_start"  # 5-20 ratings
    MATURE = "mature"        # 20+ ratings


@dataclass
class RecommendationResult:
    """Result of recommendation with strategy metadata."""
    recommendations: pd.DataFrame
    strategy: str
    user_stage: UserStage
    confidence_score: float
    n_user_ratings: int
    message: str


class UnifiedRecommendationRouter:
    """
    Routes recommendation requests to appropriate strategy based on user lifecycle.
    """
    
    def __init__(self, 
                 movies_df: pd.DataFrame,
                 ratings_df: Optional[pd.DataFrame] = None,
                 lightfm_model=None,
                 lightgcn_model=None,
                 item_features_sparse=None):
        """
        Initialize router with available models and data.
        
        Args:
            movies_df: Movie catalog
            ratings_df: Historical ratings (for collaborative filtering)
            lightfm_model: Pre-trained LightFM model
            lightgcn_model: Pre-trained LightGCN model
            item_features_sparse: Sparse feature matrix for LightFM
        """
        self.movies_df = movies_df
        self.ratings_df = ratings_df if ratings_df is not None else pd.DataFrame()
        self.lightfm_model = lightfm_model
        self.lightgcn_model = lightgcn_model
        self.item_features_sparse = item_features_sparse
        
        # Store user profiles (in production: use database)
        self.user_profiles = {}
        self.user_ratings_history = {}
        
        # Import strategies
        try:
            from cold_start_questionnaire import ColdStartQuestionnaireStrategy
            self.cold_start_strategy = ColdStartQuestionnaireStrategy(
                movies_df, ratings_df, lightfm_model, item_features_sparse
            )
        except ImportError:
            print("Warning: Cold start strategy not available")
            self.cold_start_strategy = None
    
    def determine_user_stage(self, user_id: int) -> Tuple[UserStage, int]:
        """
        Determine user's current lifecycle stage.
        
        Args:
            user_id: User ID
            
        Returns:
            (UserStage, number_of_ratings)
        """
        if user_id not in self.user_ratings_history:
            return UserStage.NEW, 0
        
        n_ratings = len(self.user_ratings_history[user_id])
        
        if n_ratings == 0:
            return UserStage.NEW, 0
        elif n_ratings <= 5:
            return UserStage.COLD_START, n_ratings
        elif n_ratings <= 20:
            return UserStage.WARM_START, n_ratings
        else:
            return UserStage.MATURE, n_ratings
    
    def register_new_user(self, user_id: int) -> Dict:
        """
        Register new user and run questionnaire.
        
        Args:
            user_id: New user ID
            
        Returns:
            User profile dict
        """
        if self.cold_start_strategy is None:
            raise RuntimeError("Cold start strategy not initialized")
        
        print(f"\n✨ Registering new user {user_id}...")
        profile = self.cold_start_strategy.run_questionnaire()
        
        self.user_profiles[user_id] = profile
        self.user_ratings_history[user_id] = []
        
        return profile
    
    def record_user_rating(self, user_id: int, movie_id: int, rating: float):
        """Record user's rating to track progress through stages."""
        if user_id not in self.user_ratings_history:
            self.user_ratings_history[user_id] = []
        
        self.user_ratings_history[user_id].append({
            'movie_id': movie_id,
            'rating': rating
        })
    
    def get_recommendations(self, 
                          user_id: int,
                          n_recommendations: int = 10,
                          force_strategy: Optional[str] = None) -> RecommendationResult:
        """
        Get recommendations for user, routing to appropriate strategy.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            force_strategy: Force specific strategy for testing
            
        Returns:
            RecommendationResult with recommendations and metadata
        """
        user_stage, n_ratings = self.determine_user_stage(user_id)
        
        print(f"\n📊 User {user_id} Stage: {user_stage.value} ({n_ratings} ratings)")
        
        # Determine which strategy to use
        if force_strategy:
            strategy = force_strategy
        else:
            strategy = self._select_strategy(user_stage)
        
        # Get profile (create if new user)
        if user_id not in self.user_profiles:
            if user_stage == UserStage.NEW:
                print("First time? Let's set up your profile...")
                self.register_new_user(user_id)
            else:
                raise ValueError(f"User {user_id} profile not found")
        
        profile = self.user_profiles[user_id]
        
        # Route to appropriate strategy
        if strategy == "content-based":
            result = self._recommend_content_based(user_id, profile, n_recommendations, user_stage)
        elif strategy == "hybrid":
            result = self._recommend_hybrid(user_id, profile, n_recommendations, user_stage)
        elif strategy == "collaborative":
            result = self._recommend_collaborative(user_id, n_recommendations, user_stage)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return result
    
    def _select_strategy(self, user_stage: UserStage) -> str:
        """Select recommendation strategy based on user stage."""
        if user_stage == UserStage.NEW or user_stage == UserStage.COLD_START:
            return "content-based"
        elif user_stage == UserStage.WARM_START:
            return "hybrid"
        else:  # MATURE
            return "collaborative"
    
    def _recommend_content_based(self, 
                                 user_id: int,
                                 profile: Dict,
                                 n_recommendations: int,
                                 user_stage: UserStage) -> RecommendationResult:
        """Generate content-based recommendations."""
        
        print(f"🎬 Using CONTENT-BASED strategy (movie similarity)")
        
        recommendations = self.cold_start_strategy.generate_initial_recommendations(
            profile, n_recommendations
        )
        
        return RecommendationResult(
            recommendations=recommendations,
            strategy="content-based",
            user_stage=user_stage,
            confidence_score=0.7,  # Content-based has moderate confidence
            n_user_ratings=len(self.user_ratings_history.get(user_id, [])),
            message=f"Based on your favorite genres and movies. Rate these to improve! ⭐"
        )
    
    def _recommend_hybrid(self,
                         user_id: int,
                         profile: Dict,
                         n_recommendations: int,
                         user_stage: UserStage) -> RecommendationResult:
        """Generate hybrid recommendations (content + collaborative)."""
        
        print(f"🎬 Using HYBRID strategy (content + collaborative)")
        
        # Get content-based recommendations
        content_recs = self.cold_start_strategy.generate_initial_recommendations(
            profile, n_recommendations // 2
        )
        
        # TODO: Add collaborative recommendations
        # For now, return content-based as fallback
        recommendations = content_recs
        recommendations['strategy'] = 'hybrid'
        
        return RecommendationResult(
            recommendations=recommendations,
            strategy="hybrid",
            user_stage=user_stage,
            confidence_score=0.80,  # Hybrid has good confidence
            n_user_ratings=len(self.user_ratings_history.get(user_id, [])),
            message=f"Based on your preferences AND similar users. Keep rating! ⭐"
        )
    
    def _recommend_collaborative(self,
                                user_id: int,
                                n_recommendations: int,
                                user_stage: UserStage) -> RecommendationResult:
        """Generate collaborative filtering recommendations."""
        
        print(f"🎬 Using COLLABORATIVE strategy (user-user similarity)")
        
        if self.lightgcn_model is None:
            print("⚠️  LightGCN model not available, falling back to content-based")
            profile = self.user_profiles.get(user_id, {})
            return self._recommend_content_based(user_id, profile, n_recommendations, user_stage)
        
        # TODO: Implement LightGCN recommendations
        recommendations = pd.DataFrame({
            'movieId': [],
            'title': [],
            'strategy': 'collaborative'
        })
        
        return RecommendationResult(
            recommendations=recommendations,
            strategy="collaborative",
            user_stage=user_stage,
            confidence_score=0.95,  # Collaborative has high confidence
            n_user_ratings=len(self.user_ratings_history.get(user_id, [])),
            message=f"Based on users with similar taste. Personalized just for you! 🎯"
        )
    
    def print_recommendation_summary(self, result: RecommendationResult):
        """Print nice summary of recommendations."""
        
        print("\n" + "="*70)
        print(f"📋 {result.strategy.upper()} RECOMMENDATIONS")
        print("="*70)
        print(f"Stage: {result.user_stage.value}")
        print(f"Confidence: {result.confidence_score:.0%}")
        print(f"Your ratings so far: {result.n_user_ratings}")
        print(f"Tip: {result.message}")
        print("-"*70)
        
        if not result.recommendations.empty:
            print(f"\nTop {len(result.recommendations)} recommendations:\n")
            for i, (_, row) in enumerate(result.recommendations.head(10).iterrows(), 1):
                title = row.get('title', 'Unknown')
                genres = row.get('genres', '')
                print(f"{i:2d}. {title}")
                if genres:
                    print(f"    Genres: {genres}\n")
        else:
            print("No recommendations available at this time.")


def demo_router():
    """Demo the unified recommendation router."""
    
    print("="*70)
    print("🎬 UNIFIED RECOMMENDATION ROUTER DEMO")
    print("="*70)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    print("\nLoading data...")
    movies = pd.read_csv(data_dir / "movies.csv", low_memory=False)
    ratings = None
    if data_dir.joinpath("ratings.csv").exists():
        ratings = pd.read_csv(data_dir / "ratings.csv", nrows=100000)
    
    # Initialize router
    router = UnifiedRecommendationRouter(movies, ratings)
    
    # Simulate new user
    print("\n" + "-"*70)
    print("SCENARIO: New User Registration")
    print("-"*70)
    
    user_id = 9999
    profile = router.register_new_user(user_id)
    
    # Get initial recommendations (cold start)
    print("\n" + "-"*70)
    print("SCENARIO 1: Initial Recommendations (Cold Start)")
    print("-"*70)
    result = router.get_recommendations(user_id, n_recommendations=10)
    router.print_recommendation_summary(result)
    
    # Simulate user rating some movies
    print("\n" + "-"*70)
    print("SCENARIO 2: After 3 ratings (Still Cold Start)")
    print("-"*70)
    for i in range(3):
        router.record_user_rating(user_id, movie_id=100+i, rating=4.0)
    
    result = router.get_recommendations(user_id, n_recommendations=10)
    router.print_recommendation_summary(result)
    
    # Simulate reaching warm start
    print("\n" + "-"*70)
    print("SCENARIO 3: After 10 ratings (Warm Start)")
    print("-"*70)
    for i in range(3, 10):
        router.record_user_rating(user_id, movie_id=100+i, rating=3.5)
    
    result = router.get_recommendations(user_id, n_recommendations=10)
    router.print_recommendation_summary(result)
    
    # Simulate mature user
    print("\n" + "-"*70)
    print("SCENARIO 4: After 25 ratings (Mature User)")
    print("-"*70)
    for i in range(10, 25):
        router.record_user_rating(user_id, movie_id=100+i, rating=3.8)
    
    result = router.get_recommendations(user_id, n_recommendations=10)
    router.print_recommendation_summary(result)
    
    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70)


if __name__ == '__main__':
    demo_router()
