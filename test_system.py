#!/usr/bin/env python3
"""
Test script for the movie recommendation system
"""

import sys
import os
import pandas as pd
from data_processor import DataProcessor
from tfidf_recommender import TFIDFRecommender
from semantic_recommender import SemanticRecommender
from hybrid_recommender import HybridRecommender

def test_data_processing():
    """Test data processing functionality"""
    print("=" * 50)
    print("Testing Data Processing...")
    print("=" * 50)
    
    processor = DataProcessor()
    
    # Load data
    if not os.path.exists('sample_movies.csv'):
        print("❌ Sample data not found. Please run create_sample_data.py first.")
        return False
    
    movies_df = processor.load_data('sample_movies.csv')
    if movies_df is None:
        print("❌ Failed to load data")
        return False
    
    print(f"✅ Loaded {len(movies_df)} movies")
    
    # Process data
    processed_df = processor.prepare_data(movies_df)
    print(f"✅ Processed {len(processed_df)} movies with valid features")
    
    # Check required columns
    required_cols = ['title', 'combined_features', 'id']
    for col in required_cols:
        if col not in processed_df.columns:
            print(f"❌ Missing required column: {col}")
            return False
    
    print("✅ Data processing test passed")
    return processed_df

def test_tfidf_recommender(movies_df):
    """Test TF-IDF recommender"""
    print("\n" + "=" * 50)
    print("Testing TF-IDF Recommender...")
    print("=" * 50)
    
    recommender = TFIDFRecommender()
    
    # Fit the model
    try:
        recommender.fit(movies_df)
        print("✅ TF-IDF model fitted successfully")
    except Exception as e:
        print(f"❌ Error fitting TF-IDF model: {e}")
        return False
    
    # Test recommendations
    test_movie = movies_df.iloc[0]['title']
    try:
        recommendations = recommender.get_recommendations(test_movie, 5)
        if recommendations:
            print(f"✅ Generated {len(recommendations)} TF-IDF recommendations for '{test_movie}'")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")
        else:
            print("❌ No TF-IDF recommendations generated")
            return False
    except Exception as e:
        print(f"❌ Error generating TF-IDF recommendations: {e}")
        return False
    
    # Test search
    try:
        search_results = recommender.search_movies("action adventure", 3)
        print(f"✅ TF-IDF search returned {len(search_results)} results")
    except Exception as e:
        print(f"❌ Error in TF-IDF search: {e}")
        return False
    
    print("✅ TF-IDF recommender test passed")
    return True

def test_semantic_recommender(movies_df):
    """Test Semantic recommender"""
    print("\n" + "=" * 50)
    print("Testing Semantic Recommender...")
    print("=" * 50)
    
    # Use a smaller subset for testing to speed up
    test_df = movies_df.head(50).copy()
    
    recommender = SemanticRecommender(model_name='all-MiniLM-L6-v2')
    
    # Fit the model
    try:
        recommender.fit(test_df)
        print("✅ Semantic model fitted successfully")
    except Exception as e:
        print(f"❌ Error fitting semantic model: {e}")
        return False
    
    # Test recommendations
    test_movie = test_df.iloc[0]['title']
    try:
        recommendations = recommender.get_recommendations(test_movie, 5)
        if recommendations:
            print(f"✅ Generated {len(recommendations)} semantic recommendations for '{test_movie}'")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")
        else:
            print("❌ No semantic recommendations generated")
            return False
    except Exception as e:
        print(f"❌ Error generating semantic recommendations: {e}")
        return False
    
    # Test search
    try:
        search_results = recommender.search_movies_semantic("space adventure", 3)
        print(f"✅ Semantic search returned {len(search_results)} results")
    except Exception as e:
        print(f"❌ Error in semantic search: {e}")
        return False
    
    print("✅ Semantic recommender test passed")
    return True

def test_hybrid_recommender(movies_df):
    """Test Hybrid recommender"""
    print("\n" + "=" * 50)
    print("Testing Hybrid Recommender...")
    print("=" * 50)
    
    # Use a smaller subset for testing
    test_df = movies_df.head(50).copy()
    
    recommender = HybridRecommender(
        tfidf_weight=0.4,
        semantic_weight=0.6
    )
    
    # Fit the model
    try:
        recommender.fit(test_df)
        print("✅ Hybrid model fitted successfully")
    except Exception as e:
        print(f"❌ Error fitting hybrid model: {e}")
        return False
    
    # Test different recommendation methods
    test_movie = test_df.iloc[0]['title']
    methods = ['tfidf', 'semantic', 'hybrid', 'ensemble']
    
    for method in methods:
        try:
            recommendations = recommender.get_recommendations(test_movie, 3, method=method)
            if recommendations:
                print(f"✅ {method.capitalize()} method: {len(recommendations)} recommendations")
            else:
                print(f"⚠️ {method.capitalize()} method: No recommendations")
        except Exception as e:
            print(f"❌ Error with {method} method: {e}")
            return False
    
    # Test method comparison
    try:
        comparison = recommender.compare_methods(test_movie, 3)
        print("✅ Method comparison completed")
    except Exception as e:
        print(f"❌ Error in method comparison: {e}")
        return False
    
    print("✅ Hybrid recommender test passed")
    return True

def main():
    """Run all tests"""
    print("🎬 Testing Movie Recommendation System")
    print("=" * 60)
    
    # Test data processing
    movies_df = test_data_processing()
    if movies_df is False:
        print("\n❌ Data processing test failed. Exiting.")
        return False
    
    # Test TF-IDF recommender
    if not test_tfidf_recommender(movies_df):
        print("\n❌ TF-IDF recommender test failed. Exiting.")
        return False
    
    # Test Semantic recommender
    if not test_semantic_recommender(movies_df):
        print("\n❌ Semantic recommender test failed. Exiting.")
        return False
    
    # Test Hybrid recommender
    if not test_hybrid_recommender(movies_df):
        print("\n❌ Hybrid recommender test failed. Exiting.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! The recommendation system is working correctly.")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run streamlit_app.py")
    print("2. Open your browser and navigate to the provided URL")
    print("3. Explore the different recommendation methods and features")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)