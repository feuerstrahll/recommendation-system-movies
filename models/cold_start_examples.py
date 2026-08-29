"""
Example: Using Cold Start Strategy with New Users

This script demonstrates how to:
1. Register a new user via questionnaire
2. Generate content-based recommendations
3. Track progression through recommendation stages
4. Switch strategies as user gets more ratings
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cold_start_questionnaire import ColdStartQuestionnaireStrategy
from recommendation_router import UnifiedRecommendationRouter, UserStage


def example_1_questionnaire_only():
    """Example 1: Just run the questionnaire and get initial recommendations."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Questionnaire + Initial Content-Based Recommendations")
    print("="*70)
    
    # Load movies
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    movies = pd.read_csv(data_dir / "movies.csv", low_memory=False)
    
    # Initialize strategy
    strategy = ColdStartQuestionnaireStrategy(movies)
    
    # Run questionnaire
    user_profile = strategy.run_questionnaire()
    
    print("\n" + "-"*70)
    print("✓ User Profile Saved:")
    print("-"*70)
    print(f"Favorite genres: {user_profile['favorite_genres']}")
    print(f"Favorite movies: {user_profile['favorite_movies']}")
    print(f"Year preference: {user_profile['age_preference']}")
    
    # Get recommendations
    recommendations = strategy.generate_initial_recommendations(user_profile, n_recommendations=10)
    
    print("\n" + "-"*70)
    print("Top 5 Recommendations:")
    print("-"*70)
    for i, (_, row) in enumerate(recommendations.head(5).iterrows(), 1):
        print(f"{i}. {row['title']}")
        print(f"   Genres: {row['genres']}")
        print(f"   Score: {row['similarity_score']:.4f}\n")


def example_2_progressive_recommendations():
    """Example 2: Track user through different recommendation stages."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Progressive Recommendations Through User Lifecycle")
    print("="*70)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    movies = pd.read_csv(data_dir / "movies.csv", low_memory=False)
    ratings = pd.read_csv(data_dir / "ratings.csv", nrows=50000) if (data_dir / "ratings.csv").exists() else None
    
    # Initialize router
    router = UnifiedRecommendationRouter(movies, ratings)
    
    # Register new user
    user_id = 99999
    print("\n1️⃣  Registering new user...")
    profile = router.register_new_user(user_id)
    
    # Stage 1: Cold Start (0-5 ratings)
    print("\n" + "-"*70)
    print("STAGE 1: Cold Start (User has 0 ratings)")
    print("-"*70)
    result = router.get_recommendations(user_id, n_recommendations=5)
    router.print_recommendation_summary(result)
    
    # Simulate user rating 5 movies
    print("\n2️⃣  User rates 5 movies...")
    for i in range(5):
        movie_id = 100 + i
        router.record_user_rating(user_id, movie_id, 4.0)
    
    # Stage 2: Warm Start (5-20 ratings)
    print("\n" + "-"*70)
    print("STAGE 2: Warm Start (User has 5 ratings)")
    print("-"*70)
    result = router.get_recommendations(user_id, n_recommendations=5)
    router.print_recommendation_summary(result)
    
    # Simulate user rating more movies
    print("\n3️⃣  User rates 15 more movies...")
    for i in range(5, 20):
        movie_id = 100 + i
        router.record_user_rating(user_id, movie_id, 3.5 + (i % 2) * 0.5)
    
    # Stage 3: Mature (20+ ratings)
    print("\n" + "-"*70)
    print("STAGE 3: Mature (User has 20 ratings)")
    print("-"*70)
    result = router.get_recommendations(user_id, n_recommendations=5)
    router.print_recommendation_summary(result)
    
    print("\n✓ User progression complete!")


def example_3_strategy_switching():
    """Example 3: Demonstrating strategy switching based on user stage."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Strategy Switching Strategy")
    print("="*70)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    movies = pd.read_csv(data_dir / "movies.csv", low_memory=False)
    ratings = pd.read_csv(data_dir / "ratings.csv", nrows=50000) if (data_dir / "ratings.csv").exists() else None
    
    router = UnifiedRecommendationRouter(movies, ratings)
    
    # Register user
    user_id = 88888
    router.register_new_user(user_id)
    
    stages_info = [
        (0, "NEW", "No recommendations yet - questionnaire only"),
        (3, "COLD_START", "Content-based recommendations"),
        (10, "WARM_START", "Hybrid: Content + Collaborative"),
        (25, "MATURE", "Pure Collaborative")
    ]
    
    for n_ratings, stage_name, description in stages_info:
        print(f"\n{'='*70}")
        print(f"Stage: {stage_name}")
        print(f"Ratings: {n_ratings}")
        print(f"Description: {description}")
        print('='*70)
        
        # Add ratings
        if n_ratings > 0:
            current_n = len(router.user_ratings_history[user_id])
            for i in range(current_n, n_ratings):
                router.record_user_rating(user_id, movie_id=1000+i, rating=3.5 + (i % 3) * 0.5)
        
        # Get stage
        user_stage, n_hist = router.determine_user_stage(user_id)
        print(f"\n→ Detected Stage: {user_stage.value}")
        print(f"→ Ratings in history: {n_hist}")
        
        # Get recommendations
        try:
            result = router.get_recommendations(user_id, n_recommendations=3)
            print(f"→ Strategy: {result.strategy}")
            print(f"→ Confidence: {result.confidence_score:.0%}")
            print(f"→ Message: {result.message}")
        except Exception as e:
            print(f"Note: {e}")


def example_4_parallel_users():
    """Example 4: Managing multiple users at different stages."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Multiple Users at Different Stages")
    print("="*70)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    movies = pd.read_csv(data_dir / "movies.csv", low_memory=False)
    ratings = pd.read_csv(data_dir / "ratings.csv", nrows=50000) if (data_dir / "ratings.csv").exists() else None
    
    router = UnifiedRecommendationRouter(movies, ratings)
    
    # Create 3 users at different stages
    users = [
        {"id": 1001, "ratings": 0, "name": "Alice (NEW)"},
        {"id": 1002, "ratings": 7, "name": "Bob (WARM_START)"},
        {"id": 1003, "ratings": 30, "name": "Charlie (MATURE)"},
    ]
    
    for user_info in users:
        user_id = user_info["id"]
        n_ratings = user_info["ratings"]
        name = user_info["name"]
        
        print(f"\n{'='*70}")
        print(f"👤 {name}")
        print('='*70)
        
        # Register and add ratings
        router.register_new_user(user_id)
        
        for i in range(n_ratings):
            router.record_user_rating(user_id, movie_id=1000+i, rating=3.0 + (i % 5) * 0.4)
        
        # Get recommendations
        result = router.get_recommendations(user_id, n_recommendations=3)
        router.print_recommendation_summary(result)


def main():
    """Run all examples."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  COLD START STRATEGY - USAGE EXAMPLES".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    examples = [
        ("1", "Questionnaire + Content-Based Recommendations", example_1_questionnaire_only),
        ("2", "Progressive Recommendations (Full Lifecycle)", example_2_progressive_recommendations),
        ("3", "Strategy Switching Based on User Stage", example_3_strategy_switching),
        ("4", "Multiple Users at Different Stages", example_4_parallel_users),
    ]
    
    print("\nAvailable examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    
    print("\nWhich example would you like to run? (1-4 or 'all'):")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        for num, desc, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error in example {num}: {e}")
                import traceback
                traceback.print_exc()
    else:
        example_num = choice
        for num, desc, func in examples:
            if num == example_num:
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                return
        
        print(f"❌ Invalid choice: {choice}")


if __name__ == '__main__':
    main()
