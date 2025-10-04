import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional
import nltk
import pickle
import os

class DataProcessor:
    """
    Handles data preprocessing for movie recommendation system
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.movies_df = None
        self.processed_features = None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load movie data from CSV file
        
        Args:
            data_path: Path to the CSV file containing movie data
            
        Returns:
            DataFrame with movie data
        """
        try:
            self.movies_df = pd.read_csv(data_path)
            print(f"Loaded {len(self.movies_df)} movies from {data_path}")
            return self.movies_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_combined_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined feature text for each movie
        
        Args:
            df: DataFrame containing movie data
            
        Returns:
            DataFrame with added combined_features column
        """
        df = df.copy()
        
        # Define feature columns to combine
        feature_columns = ['overview', 'genres', 'keywords', 'director', 'cast']
        
        # Clean each feature column
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
            else:
                df[col] = ""
        
        # Combine all features into a single text
        df['combined_features'] = (
            df['overview'].fillna('') + ' ' +
            df['genres'].fillna('') + ' ' +
            df['keywords'].fillna('') + ' ' +
            df['director'].fillna('') + ' ' +
            df['cast'].fillna('')
        )
        
        # Clean the combined features
        df['combined_features'] = df['combined_features'].apply(self.clean_text)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for recommendation system
        
        Args:
            df: Raw movie DataFrame
            
        Returns:
            Processed DataFrame ready for recommendations
        """
        # Create combined features
        processed_df = self.create_combined_features(df)
        
        # Filter out movies with empty combined features
        processed_df = processed_df[processed_df['combined_features'].str.len() > 0]
        
        # Reset index
        processed_df = processed_df.reset_index(drop=True)
        
        print(f"Processed {len(processed_df)} movies with valid features")
        
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """
        Save processed data to pickle file
        
        Args:
            df: Processed DataFrame
            filepath: Path to save the pickle file
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(df, f)
            print(f"Processed data saved to {filepath}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """
        Load processed data from pickle file
        
        Args:
            filepath: Path to the pickle file
            
        Returns:
            Processed DataFrame
        """
        try:
            with open(filepath, 'rb') as f:
                df = pickle.load(f)
            print(f"Loaded processed data from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None
    
    def get_movie_info(self, df: pd.DataFrame, movie_id: int) -> Dict:
        """
        Get detailed information about a specific movie
        
        Args:
            df: Movie DataFrame
            movie_id: ID of the movie
            
        Returns:
            Dictionary containing movie information
        """
        movie = df[df['id'] == movie_id]
        if movie.empty:
            return None
        
        movie_info = movie.iloc[0].to_dict()
        return movie_info