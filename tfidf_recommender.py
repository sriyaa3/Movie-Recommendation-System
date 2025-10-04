import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import pickle
import os

class TFIDFRecommender:
    """
    TF-IDF based content similarity recommender system
    """
    
    def __init__(self, max_features: int = 10000, stop_words: str = 'english'):
        """
        Initialize TF-IDF recommender
        
        Args:
            max_features: Maximum number of features for TF-IDF
            stop_words: Stop words to remove ('english' or None)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            lowercase=True,
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        self.tfidf_matrix = None
        self.movies_df = None
        self.feature_names = None
        
    def fit(self, movies_df: pd.DataFrame):
        """
        Fit the TF-IDF vectorizer on movie data
        
        Args:
            movies_df: DataFrame containing movies with combined_features column
        """
        self.movies_df = movies_df.copy()
        
        # Fit and transform the combined features
        print("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(movies_df['combined_features'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
    def get_recommendations(self, movie_title: str, n_recommendations: int = 10) -> List[Dict]:
        """
        Get movie recommendations based on TF-IDF cosine similarity
        
        Args:
            movie_title: Title of the movie to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find the movie in the dataset
        movie_idx = self._find_movie_index(movie_title)
        if movie_idx is None:
            return []
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(
            self.tfidf_matrix[movie_idx], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = cosine_similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            movie_info = self.movies_df.iloc[idx].to_dict()
            movie_info['similarity_score'] = float(cosine_similarities[idx])
            movie_info['recommendation_type'] = 'TF-IDF'
            recommendations.append(movie_info)
        
        return recommendations
    
    def get_similar_movies_by_id(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        Get movie recommendations by movie ID
        
        Args:
            movie_id: ID of the movie to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find the movie index by ID
        movie_idx = self._find_movie_index_by_id(movie_id)
        if movie_idx is None:
            return []
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(
            self.tfidf_matrix[movie_idx], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = cosine_similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            movie_info = self.movies_df.iloc[idx].to_dict()
            movie_info['similarity_score'] = float(cosine_similarities[idx])
            movie_info['recommendation_type'] = 'TF-IDF'
            recommendations.append(movie_info)
        
        return recommendations
    
    def _find_movie_index(self, movie_title: str) -> int:
        """
        Find the index of a movie by title
        
        Args:
            movie_title: Title of the movie
            
        Returns:
            Index of the movie or None if not found
        """
        # Try exact match first
        exact_match = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()]
        if not exact_match.empty:
            return exact_match.index[0]
        
        # Try partial match
        partial_match = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)
        ]
        if not partial_match.empty:
            return partial_match.index[0]
        
        return None
    
    def _find_movie_index_by_id(self, movie_id: int) -> int:
        """
        Find the index of a movie by ID
        
        Args:
            movie_id: ID of the movie
            
        Returns:
            Index of the movie or None if not found
        """
        movie_row = self.movies_df[self.movies_df['id'] == movie_id]
        if not movie_row.empty:
            return movie_row.index[0]
        return None
    
    def get_top_features(self, movie_title: str, n_features: int = 10) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a specific movie
        
        Args:
            movie_title: Title of the movie
            n_features: Number of top features to return
            
        Returns:
            List of tuples (feature, tfidf_score)
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        movie_idx = self._find_movie_index(movie_title)
        if movie_idx is None:
            return []
        
        # Get TF-IDF scores for the movie
        tfidf_scores = self.tfidf_matrix[movie_idx].toarray().flatten()
        
        # Get top features
        top_indices = tfidf_scores.argsort()[::-1][:n_features]
        top_features = [
            (self.feature_names[idx], tfidf_scores[idx]) 
            for idx in top_indices if tfidf_scores[idx] > 0
        ]
        
        return top_features
    
    def save_model(self, filepath: str):
        """
        Save the trained TF-IDF model
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'movies_df': self.movies_df,
            'feature_names': self.feature_names
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"TF-IDF model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """
        Load a trained TF-IDF model
        
        Args:
            filepath: Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.movies_df = model_data['movies_df']
            self.feature_names = model_data['feature_names']
            
            print(f"TF-IDF model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def search_movies(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Search for movies using TF-IDF similarity with query
        
        Args:
            query: Search query
            n_results: Number of search results to return
            
        Returns:
            List of movies matching the query
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Transform the query using the fitted vectorizer
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calculate similarity between query and all movies
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top matching movies
        top_indices = similarities.argsort()[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                movie_info = self.movies_df.iloc[idx].to_dict()
                movie_info['similarity_score'] = float(similarities[idx])
                movie_info['search_type'] = 'TF-IDF Search'
                results.append(movie_info)
        
        return results