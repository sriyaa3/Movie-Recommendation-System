import pandas as pd
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Semantic features will be limited.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class SemanticRecommender:
    """
    Semantic embeddings-based recommender using Sentence-Transformers
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic recommender
        
        Args:
            model_name: Name of the Sentence-Transformer model to use
                       Options: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'distilbert-base-nli-stsb-mean-tokens'
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.movies_df = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not available. Using TF-IDF fallback.")
            self.fallback_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.fallback_embeddings = None
            return
        
        # Load the sentence transformer model
        print(f"Loading Sentence-Transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to TF-IDF method...")
            self.fallback_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.fallback_embeddings = None
            self.model = None
    
    def fit(self, movies_df: pd.DataFrame):
        """
        Fit the semantic model on movie data
        
        Args:
            movies_df: DataFrame containing movies with combined_features column
        """
        self.movies_df = movies_df.copy()
        
        if self.model is None or not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Using TF-IDF fallback for semantic embeddings...")
            self.fallback_embeddings = self.fallback_vectorizer.fit_transform(movies_df['combined_features'])
            print(f"Generated TF-IDF embeddings shape: {self.fallback_embeddings.shape}")
            return
        
        print("Generating semantic embeddings...")
        print(f"Processing {len(movies_df)} movies...")
        
        # Prepare text data for embedding
        texts = movies_df['combined_features'].tolist()
        
        # Generate embeddings in batches to handle memory efficiently
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                convert_to_tensor=False,
                show_progress_bar=True if i == 0 else False
            )
            embeddings_list.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_texts)} movies...")
        
        # Combine all embeddings
        self.embeddings = np.vstack(embeddings_list)
        
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        print("Semantic model fitted successfully!")
    
    def get_recommendations(self, movie_title: str, n_recommendations: int = 10) -> List[Dict]:
        """
        Get movie recommendations based on semantic similarity
        
        Args:
            movie_title: Title of the movie to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with similarity scores
        """
        if self.embeddings is None:
            if hasattr(self, 'fallback_embeddings') and self.fallback_embeddings is not None:
                # Use TF-IDF fallback
                return self._get_recommendations_fallback(movie_title, n_recommendations)
            else:
                raise ValueError("Model not fitted. Call fit() first.")
        
        # Find the movie in the dataset
        movie_idx = self._find_movie_index(movie_title)
        if movie_idx is None:
            return []
        
        # Get the embedding for the target movie
        target_embedding = self.embeddings[movie_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all movies
        similarities = cosine_similarity(target_embedding, self.embeddings).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            movie_info = self.movies_df.iloc[idx].to_dict()
            movie_info['similarity_score'] = float(similarities[idx])
            movie_info['recommendation_type'] = 'Semantic'
            recommendations.append(movie_info)
        
        return recommendations
    
    def get_similar_movies_by_id(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        Get movie recommendations by movie ID using semantic similarity
        
        Args:
            movie_id: ID of the movie to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with similarity scores
        """
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find the movie index by ID
        movie_idx = self._find_movie_index_by_id(movie_id)
        if movie_idx is None:
            return []
        
        # Get the embedding for the target movie
        target_embedding = self.embeddings[movie_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all movies
        similarities = cosine_similarity(target_embedding, self.embeddings).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            movie_info = self.movies_df.iloc[idx].to_dict()
            movie_info['similarity_score'] = float(similarities[idx])
            movie_info['recommendation_type'] = 'Semantic'
            recommendations.append(movie_info)
        
        return recommendations
    
    def search_movies_semantic(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Search for movies using semantic similarity with query
        
        Args:
            query: Search query
            n_results: Number of search results to return
            
        Returns:
            List of movies matching the query semantically
        """
        if self.embeddings is None:
            if hasattr(self, 'fallback_embeddings') and self.fallback_embeddings is not None:
                # Use TF-IDF fallback
                query_vector = self.fallback_vectorizer.transform([query.lower()])
                similarities = cosine_similarity(query_vector, self.fallback_embeddings).flatten()
                
                # Get top matching movies
                top_indices = similarities.argsort()[::-1][:n_results]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.01:  # Lower threshold for TF-IDF
                        movie_info = self.movies_df.iloc[idx].to_dict()
                        movie_info['similarity_score'] = float(similarities[idx])
                        movie_info['search_type'] = 'Semantic Search (TF-IDF Fallback)'
                        results.append(movie_info)
                
                return results
            else:
                raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        
        # Calculate similarity between query and all movies
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top matching movies
        top_indices = similarities.argsort()[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for semantic similarity
                movie_info = self.movies_df.iloc[idx].to_dict()
                movie_info['similarity_score'] = float(similarities[idx])
                movie_info['search_type'] = 'Semantic Search'
                results.append(movie_info)
        
        return results
    
    def get_embedding_similarity(self, movie1_id: int, movie2_id: int) -> float:
        """
        Get semantic similarity between two movies
        
        Args:
            movie1_id: ID of the first movie
            movie2_id: ID of the second movie
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        idx1 = self._find_movie_index_by_id(movie1_id)
        idx2 = self._find_movie_index_by_id(movie2_id)
        
        if idx1 is None or idx2 is None:
            return 0.0
        
        similarity = cosine_similarity(
            self.embeddings[idx1].reshape(1, -1),
            self.embeddings[idx2].reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def _find_movie_index(self, movie_title: str) -> Optional[int]:
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
    
    def _find_movie_index_by_id(self, movie_id: int) -> Optional[int]:
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
    
    def save_model(self, filepath: str):
        """
        Save the trained semantic model and embeddings
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model_name': self.model_name,
            'embeddings': self.embeddings,
            'movies_df': self.movies_df
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Semantic model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """
        Load a trained semantic model
        
        Args:
            filepath: Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_name = model_data['model_name']
            self.embeddings = model_data['embeddings']
            self.movies_df = model_data['movies_df']
            
            # Reload the transformer model
            self.model = SentenceTransformer(self.model_name)
            
            print(f"Semantic model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def analyze_movie_themes(self, movie_title: str, n_similar: int = 5) -> Dict:
        """
        Analyze themes and concepts for a movie based on semantic similarity
        
        Args:
            movie_title: Title of the movie to analyze
            n_similar: Number of similar movies to find for analysis
            
        Returns:
            Dictionary containing theme analysis
        """
        similar_movies = self.get_recommendations(movie_title, n_similar)
        
        if not similar_movies:
            return {"error": "Movie not found"}
        
        # Extract genres and keywords from similar movies
        all_genres = []
        all_keywords = []
        
        for movie in similar_movies:
            if 'genres' in movie and movie['genres']:
                all_genres.extend(movie['genres'].split())
            if 'keywords' in movie and movie['keywords']:
                all_keywords.extend(movie['keywords'].split())
        
        # Count frequency
        from collections import Counter
        genre_counts = Counter(all_genres)
        keyword_counts = Counter(all_keywords)
        
        return {
            "similar_movies": [m['title'] for m in similar_movies],
            "common_genres": dict(genre_counts.most_common(5)),
            "common_themes": dict(keyword_counts.most_common(10)),
            "average_similarity": np.mean([m['similarity_score'] for m in similar_movies])
        }
    
    def _get_recommendations_fallback(self, movie_title: str, n_recommendations: int) -> List[Dict]:
        """Fallback method using TF-IDF when sentence-transformers is not available"""
        movie_idx = self._find_movie_index(movie_title)
        if movie_idx is None:
            return []
        
        # Calculate cosine similarity using TF-IDF
        similarities = cosine_similarity(
            self.fallback_embeddings[movie_idx], 
            self.fallback_embeddings
        ).flatten()
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            movie_info = self.movies_df.iloc[idx].to_dict()
            movie_info['similarity_score'] = float(similarities[idx])
            movie_info['recommendation_type'] = 'Semantic (TF-IDF Fallback)'
            recommendations.append(movie_info)
        
        return recommendations