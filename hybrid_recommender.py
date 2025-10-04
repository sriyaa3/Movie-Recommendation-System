import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
from tfidf_recommender import TFIDFRecommender
from semantic_recommender import SemanticRecommender
from data_processor import DataProcessor

class HybridRecommender:
    """
    Advanced hybrid recommender that combines TF-IDF and Semantic embeddings
    """
    
    def __init__(self, 
                 tfidf_weight: float = 0.4, 
                 semantic_weight: float = 0.6,
                 tfidf_params: Dict = None,
                 semantic_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize hybrid recommender
        
        Args:
            tfidf_weight: Weight for TF-IDF recommendations (0-1)
            semantic_weight: Weight for semantic recommendations (0-1)
            tfidf_params: Parameters for TF-IDF vectorizer
            semantic_model: Sentence-Transformer model name
        """
        if abs(tfidf_weight + semantic_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight
        
        # Initialize recommenders
        tfidf_params = tfidf_params or {}
        self.tfidf_recommender = TFIDFRecommender(**tfidf_params)
        self.semantic_recommender = SemanticRecommender(semantic_model)
        
        self.movies_df = None
        self.is_fitted = False
        
    def fit(self, movies_df: pd.DataFrame):
        """
        Fit both TF-IDF and semantic recommenders
        
        Args:
            movies_df: DataFrame containing movies with combined_features column
        """
        print("Fitting Hybrid Recommender...")
        self.movies_df = movies_df.copy()
        
        # Fit both recommenders
        print("\n1. Fitting TF-IDF Recommender...")
        self.tfidf_recommender.fit(movies_df)
        
        print("\n2. Fitting Semantic Recommender...")
        self.semantic_recommender.fit(movies_df)
        
        self.is_fitted = True
        print("\nHybrid Recommender fitted successfully!")
    
    def get_recommendations(self, 
                          movie_title: str, 
                          n_recommendations: int = 10,
                          method: str = 'hybrid') -> List[Dict]:
        """
        Get movie recommendations using specified method
        
        Args:
            movie_title: Title of the movie to get recommendations for
            n_recommendations: Number of recommendations to return
            method: 'hybrid', 'tfidf', 'semantic', or 'ensemble'
            
        Returns:
            List of recommended movies with combined scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if method == 'tfidf':
            return self.tfidf_recommender.get_recommendations(movie_title, n_recommendations)
        elif method == 'semantic':
            return self.semantic_recommender.get_recommendations(movie_title, n_recommendations)
        elif method == 'hybrid':
            return self._get_hybrid_recommendations(movie_title, n_recommendations)
        elif method == 'ensemble':
            return self._get_ensemble_recommendations(movie_title, n_recommendations)
        else:
            raise ValueError("Method must be 'hybrid', 'tfidf', 'semantic', or 'ensemble'")
    
    def _get_hybrid_recommendations(self, movie_title: str, n_recommendations: int) -> List[Dict]:
        """
        Get hybrid recommendations by combining scores
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with hybrid scores
        """
        # Get recommendations from both methods
        tfidf_recs = self.tfidf_recommender.get_recommendations(movie_title, n_recommendations * 2)
        semantic_recs = self.semantic_recommender.get_recommendations(movie_title, n_recommendations * 2)
        
        if not tfidf_recs or not semantic_recs:
            # Fallback to available method
            return tfidf_recs or semantic_recs
        
        # Create score dictionaries
        tfidf_scores = {rec['id']: rec['similarity_score'] for rec in tfidf_recs}
        semantic_scores = {rec['id']: rec['similarity_score'] for rec in semantic_recs}
        
        # Get all unique movie IDs
        all_movie_ids = set(tfidf_scores.keys()) | set(semantic_scores.keys())
        
        # Calculate hybrid scores
        hybrid_recommendations = []
        for movie_id in all_movie_ids:
            tfidf_score = tfidf_scores.get(movie_id, 0.0)
            semantic_score = semantic_scores.get(movie_id, 0.0)
            
            # Calculate weighted combination
            hybrid_score = (self.tfidf_weight * tfidf_score + 
                          self.semantic_weight * semantic_score)
            
            # Get movie info
            movie_info = self._get_movie_info(movie_id)
            if movie_info:
                movie_info['similarity_score'] = hybrid_score
                movie_info['tfidf_score'] = tfidf_score
                movie_info['semantic_score'] = semantic_score
                movie_info['recommendation_type'] = 'Hybrid'
                hybrid_recommendations.append(movie_info)
        
        # Sort by hybrid score and return top N
        hybrid_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return hybrid_recommendations[:n_recommendations]
    
    def _get_ensemble_recommendations(self, movie_title: str, n_recommendations: int) -> List[Dict]:
        """
        Get ensemble recommendations by ranking fusion
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with ensemble scores
        """
        # Get recommendations from both methods
        tfidf_recs = self.tfidf_recommender.get_recommendations(movie_title, n_recommendations * 2)
        semantic_recs = self.semantic_recommender.get_recommendations(movie_title, n_recommendations * 2)
        
        if not tfidf_recs or not semantic_recs:
            return tfidf_recs or semantic_recs
        
        # Create rank dictionaries (higher rank = better)
        tfidf_ranks = {rec['id']: len(tfidf_recs) - i for i, rec in enumerate(tfidf_recs)}
        semantic_ranks = {rec['id']: len(semantic_recs) - i for i, rec in enumerate(semantic_recs)}
        
        # Get all unique movie IDs
        all_movie_ids = set(tfidf_ranks.keys()) | set(semantic_ranks.keys())
        
        # Calculate ensemble scores using rank fusion
        ensemble_recommendations = []
        for movie_id in all_movie_ids:
            tfidf_rank = tfidf_ranks.get(movie_id, 0)
            semantic_rank = semantic_ranks.get(movie_id, 0)
            
            # Reciprocal rank fusion
            ensemble_score = (self.tfidf_weight / (1 + len(tfidf_recs) - tfidf_rank) + 
                            self.semantic_weight / (1 + len(semantic_recs) - semantic_rank))
            
            # Get movie info
            movie_info = self._get_movie_info(movie_id)
            if movie_info:
                movie_info['similarity_score'] = ensemble_score
                movie_info['tfidf_rank'] = tfidf_rank
                movie_info['semantic_rank'] = semantic_rank
                movie_info['recommendation_type'] = 'Ensemble'
                ensemble_recommendations.append(movie_info)
        
        # Sort by ensemble score and return top N
        ensemble_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return ensemble_recommendations[:n_recommendations]
    
    def search_movies(self, query: str, n_results: int = 10, method: str = 'hybrid') -> List[Dict]:
        """
        Search for movies using specified method
        
        Args:
            query: Search query
            n_results: Number of search results to return
            method: 'hybrid', 'tfidf', or 'semantic'
            
        Returns:
            List of movies matching the query
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if method == 'tfidf':
            return self.tfidf_recommender.search_movies(query, n_results)
        elif method == 'semantic':
            return self.semantic_recommender.search_movies_semantic(query, n_results)
        elif method == 'hybrid':
            return self._hybrid_search(query, n_results)
        else:
            raise ValueError("Method must be 'hybrid', 'tfidf', or 'semantic'")
    
    def _hybrid_search(self, query: str, n_results: int) -> List[Dict]:
        """
        Perform hybrid search combining TF-IDF and semantic results
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results with hybrid scores
        """
        # Get results from both methods
        tfidf_results = self.tfidf_recommender.search_movies(query, n_results * 2)
        semantic_results = self.semantic_recommender.search_movies_semantic(query, n_results * 2)
        
        if not tfidf_results and not semantic_results:
            return []
        
        # Create score dictionaries
        tfidf_scores = {res['id']: res['similarity_score'] for res in tfidf_results}
        semantic_scores = {res['id']: res['similarity_score'] for res in semantic_results}
        
        # Get all unique movie IDs
        all_movie_ids = set(tfidf_scores.keys()) | set(semantic_scores.keys())
        
        # Calculate hybrid scores
        hybrid_results = []
        for movie_id in all_movie_ids:
            tfidf_score = tfidf_scores.get(movie_id, 0.0)
            semantic_score = semantic_scores.get(movie_id, 0.0)
            
            # Calculate weighted combination
            hybrid_score = (self.tfidf_weight * tfidf_score + 
                          self.semantic_weight * semantic_score)
            
            # Get movie info
            movie_info = self._get_movie_info(movie_id)
            if movie_info:
                movie_info['similarity_score'] = hybrid_score
                movie_info['search_type'] = 'Hybrid Search'
                hybrid_results.append(movie_info)
        
        # Sort by hybrid score and return top N
        hybrid_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return hybrid_results[:n_results]
    
    def compare_methods(self, movie_title: str, n_recommendations: int = 5) -> Dict:
        """
        Compare recommendations from different methods
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations per method
            
        Returns:
            Dictionary containing recommendations from all methods
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = {
            'movie': movie_title,
            'tfidf': self.tfidf_recommender.get_recommendations(movie_title, n_recommendations),
            'semantic': self.semantic_recommender.get_recommendations(movie_title, n_recommendations),
            'hybrid': self._get_hybrid_recommendations(movie_title, n_recommendations),
            'ensemble': self._get_ensemble_recommendations(movie_title, n_recommendations)
        }
        
        return results
    
    def get_recommendation_explanation(self, movie_title: str, recommended_title: str) -> Dict:
        """
        Get explanation for why a movie was recommended
        
        Args:
            movie_title: Original movie title
            recommended_title: Recommended movie title
            
        Returns:
            Dictionary containing explanation details
        """
        # Get TF-IDF features for both movies
        tfidf_features_orig = self.tfidf_recommender.get_top_features(movie_title, 10)
        tfidf_features_rec = self.tfidf_recommender.get_top_features(recommended_title, 10)
        
        # Get movie IDs for semantic similarity
        orig_movie = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()]
        rec_movie = self.movies_df[self.movies_df['title'].str.lower() == recommended_title.lower()]
        
        semantic_similarity = 0.0
        if not orig_movie.empty and not rec_movie.empty:
            semantic_similarity = self.semantic_recommender.get_embedding_similarity(
                orig_movie.iloc[0]['id'], rec_movie.iloc[0]['id']
            )
        
        return {
            'original_movie': movie_title,
            'recommended_movie': recommended_title,
            'tfidf_features_original': tfidf_features_orig,
            'tfidf_features_recommended': tfidf_features_rec,
            'semantic_similarity': semantic_similarity,
            'explanation': self._generate_explanation(
                tfidf_features_orig, tfidf_features_rec, semantic_similarity
            )
        }
    
    def _generate_explanation(self, tfidf_orig: List, tfidf_rec: List, semantic_sim: float) -> str:
        """Generate human-readable explanation for recommendation"""
        # Find common TF-IDF features
        orig_features = set([feat[0] for feat in tfidf_orig[:5]])
        rec_features = set([feat[0] for feat in tfidf_rec[:5]])
        common_features = orig_features & rec_features
        
        explanation = f"Semantic similarity: {semantic_sim:.3f}. "
        
        if common_features:
            explanation += f"Common themes: {', '.join(list(common_features)[:3])}. "
        
        if semantic_sim > 0.7:
            explanation += "High semantic similarity indicates strong thematic overlap."
        elif semantic_sim > 0.4:
            explanation += "Moderate semantic similarity suggests related content."
        else:
            explanation += "Recommended based on specific keyword matches."
        
        return explanation
    
    def _get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """Get movie information by ID"""
        movie_row = self.movies_df[self.movies_df['id'] == movie_id]
        if not movie_row.empty:
            return movie_row.iloc[0].to_dict()
        return None
    
    def save_model(self, filepath: str):
        """Save the hybrid model"""
        model_data = {
            'tfidf_weight': self.tfidf_weight,
            'semantic_weight': self.semantic_weight,
            'movies_df': self.movies_df,
            'is_fitted': self.is_fitted
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save individual models
            base_path = filepath.replace('.pkl', '')
            self.tfidf_recommender.save_model(f"{base_path}_tfidf.pkl")
            self.semantic_recommender.save_model(f"{base_path}_semantic.pkl")
            
            print(f"Hybrid model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load the hybrid model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_weight = model_data['tfidf_weight']
            self.semantic_weight = model_data['semantic_weight']
            self.movies_df = model_data['movies_df']
            self.is_fitted = model_data['is_fitted']
            
            # Load individual models
            base_path = filepath.replace('.pkl', '')
            self.tfidf_recommender.load_model(f"{base_path}_tfidf.pkl")
            self.semantic_recommender.load_model(f"{base_path}_semantic.pkl")
            
            print(f"Hybrid model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def optimize_weights(self, test_cases: List[Tuple[str, List[str]]], 
                        weight_range: Tuple[float, float] = (0.1, 0.9),
                        step: float = 0.1) -> Tuple[float, float]:
        """
        Optimize weights based on test cases
        
        Args:
            test_cases: List of (movie_title, expected_recommendations)
            weight_range: Range of weights to test
            step: Step size for weight optimization
            
        Returns:
            Optimal (tfidf_weight, semantic_weight)
        """
        best_score = 0
        best_weights = (0.5, 0.5)
        
        for tfidf_w in np.arange(weight_range[0], weight_range[1], step):
            semantic_w = 1.0 - tfidf_w
            
            # Temporarily update weights
            original_tfidf_w, original_semantic_w = self.tfidf_weight, self.semantic_weight
            self.tfidf_weight, self.semantic_weight = tfidf_w, semantic_w
            
            # Evaluate on test cases
            total_score = 0
            for movie_title, expected in test_cases:
                recommendations = self.get_recommendations(movie_title, len(expected))
                rec_titles = [r['title'].lower() for r in recommendations]
                expected_lower = [e.lower() for e in expected]
                
                # Calculate overlap score
                overlap = len(set(rec_titles) & set(expected_lower))
                total_score += overlap / len(expected)
            
            avg_score = total_score / len(test_cases)
            
            if avg_score > best_score:
                best_score = avg_score
                best_weights = (tfidf_w, semantic_w)
            
            # Restore original weights
            self.tfidf_weight, self.semantic_weight = original_tfidf_w, original_semantic_w
        
        # Set optimal weights
        self.tfidf_weight, self.semantic_weight = best_weights
        print(f"Optimal weights: TF-IDF={best_weights[0]:.2f}, Semantic={best_weights[1]:.2f}")
        print(f"Best score: {best_score:.3f}")
        
        return best_weights