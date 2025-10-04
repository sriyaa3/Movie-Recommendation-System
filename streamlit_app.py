import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from typing import List, Dict

# Import our custom modules
from data_processor import DataProcessor
from tfidf_recommender import TFIDFRecommender
from semantic_recommender import SemanticRecommender
from hybrid_recommender import HybridRecommender

# Configure Streamlit page
st.set_page_config(
    page_title="🎬 Advanced Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading and data processing
@st.cache_resource
def load_and_process_data():
    """Load and process movie data"""
    processor = DataProcessor()
    
    # Check if processed data exists
    if os.path.exists('processed_movies.pkl'):
        try:
            movies_df = processor.load_processed_data('processed_movies.pkl')
            if movies_df is not None:
                return movies_df, processor
        except:
            pass
    
    # Load raw data
    if os.path.exists('sample_movies.csv'):
        movies_df = processor.load_data('sample_movies.csv')
    else:
        st.error("Sample movie data not found. Please run create_sample_data.py first.")
        return None, None
    
    # Process data
    movies_df = processor.prepare_data(movies_df)
    
    # Save processed data
    processor.save_processed_data(movies_df, 'processed_movies.pkl')
    
    return movies_df, processor

@st.cache_resource
def initialize_recommender(_movies_df):
    """Initialize and fit the hybrid recommender"""
    recommender = HybridRecommender(
        tfidf_weight=0.4,
        semantic_weight=0.6,
        semantic_model='all-MiniLM-L6-v2'
    )
    
    # Check if model exists
    if os.path.exists('hybrid_model.pkl'):
        try:
            recommender.load_model('hybrid_model.pkl')
            return recommender
        except:
            pass
    
    # Fit the model
    with st.spinner("Training recommendation models... This may take a few minutes."):
        recommender.fit(_movies_df)
        recommender.save_model('hybrid_model.pkl')
    
    return recommender

def display_movie_card(movie: Dict, show_score: bool = True):
    """Display a movie card with information"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Placeholder for movie poster
            st.image("https://via.placeholder.com/150x225/4ECDC4/white?text=🎬", width=100)
        
        with col2:
            st.markdown(f"**{movie['title']}** ({movie.get('release_year', 'N/A')})")
            st.markdown(f"*Director: {movie.get('director', 'Unknown')}*")
            
            if show_score and 'similarity_score' in movie:
                score_type = movie.get('recommendation_type', 'Similarity')
                st.markdown(f"**{score_type} Score:** {movie['similarity_score']:.3f}")
            
            # Rating and runtime
            col2a, col2b = st.columns(2)
            with col2a:
                st.markdown(f"⭐ **Rating:** {movie.get('rating', 'N/A')}")
            with col2b:
                st.markdown(f"⏱️ **Runtime:** {movie.get('runtime', 'N/A')} min")
            
            # Genres
            if movie.get('genres'):
                genres = movie['genres'].split()[:3]  # Show first 3 genres
                genre_badges = " ".join([f"`{genre}`" for genre in genres])
                st.markdown(f"**Genres:** {genre_badges}")
            
            # Overview
            overview = movie.get('overview', 'No overview available.')
            if len(overview) > 200:
                overview = overview[:200] + "..."
            st.markdown(f"*{overview}*")
            
            st.markdown("---")

def create_comparison_chart(comparison_results: Dict):
    """Create a comparison chart for different recommendation methods"""
    methods = ['tfidf', 'semantic', 'hybrid', 'ensemble']
    method_names = ['TF-IDF', 'Semantic', 'Hybrid', 'Ensemble']
    
    # Calculate average scores for each method
    avg_scores = []
    for method in methods:
        recommendations = comparison_results.get(method, [])
        if recommendations:
            avg_score = np.mean([r['similarity_score'] for r in recommendations])
            avg_scores.append(avg_score)
        else:
            avg_scores.append(0)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=method_names,
            y=avg_scores,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            text=[f'{score:.3f}' for score in avg_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Average Similarity Scores by Method",
        xaxis_title="Recommendation Method",
        yaxis_title="Average Similarity Score",
        showlegend=False,
        height=400
    )
    
    return fig

def create_genre_distribution_chart(movies_df: pd.DataFrame):
    """Create a genre distribution chart"""
    all_genres = []
    for genres in movies_df['genres'].dropna():
        all_genres.extend(genres.split())
    
    from collections import Counter
    genre_counts = Counter(all_genres)
    
    top_genres = dict(genre_counts.most_common(10))
    
    fig = px.bar(
        x=list(top_genres.keys()),
        y=list(top_genres.values()),
        labels={'x': 'Genre', 'y': 'Number of Movies'},
        title="Top 10 Genres in Dataset"
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Advanced Movie Recommendation System</h1>
        <p>Powered by TF-IDF Cosine Similarity & Sentence-Transformers Semantic Embeddings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and initialize models
    movies_df, processor = load_and_process_data()
    
    if movies_df is None:
        st.error("Failed to load movie data. Please check the data files.")
        return
    
    recommender = initialize_recommender(movies_df)
    
    # Sidebar
    st.sidebar.title("🎯 Recommendation Settings")
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎬 Get Recommendations", 
        "🔍 Search Movies", 
        "📊 Compare Methods",
        "📈 Analytics",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Get Movie Recommendations")
        
        # Movie selection
        movie_titles = movies_df['title'].tolist()
        selected_movie = st.selectbox(
            "Select a movie you like:",
            options=movie_titles,
            index=0
        )
        
        # Recommendation settings
        col1, col2 = st.columns(2)
        
        with col1:
            num_recommendations = st.slider(
                "Number of recommendations:",
                min_value=1, max_value=20, value=10
            )
        
        with col2:
            recommendation_method = st.selectbox(
                "Recommendation method:",
                options=['hybrid', 'tfidf', 'semantic', 'ensemble'],
                format_func=lambda x: {
                    'hybrid': 'Hybrid (TF-IDF + Semantic)',
                    'tfidf': 'TF-IDF Only',
                    'semantic': 'Semantic Only',
                    'ensemble': 'Ensemble (Rank Fusion)'
                }[x]
            )
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = recommender.get_recommendations(
                    selected_movie, 
                    num_recommendations,
                    method=recommendation_method
                )
            
            if recommendations:
                st.success(f"Found {len(recommendations)} recommendations for '{selected_movie}'")
                
                # Display selected movie info
                st.subheader("Selected Movie:")
                selected_movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0].to_dict()
                display_movie_card(selected_movie_info, show_score=False)
                
                # Display recommendations
                st.subheader("Recommended Movies:")
                for i, movie in enumerate(recommendations, 1):
                    st.markdown(f"### {i}. Recommendation")
                    display_movie_card(movie)
            else:
                st.warning("No recommendations found for the selected movie.")
    
    with tab2:
        st.header("Search Movies")
        
        # Search settings
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter search terms (genres, themes, keywords):",
                placeholder="e.g., 'space adventure', 'romantic comedy', 'superhero'"
            )
        
        with col2:
            search_method = st.selectbox(
                "Search method:",
                options=['hybrid', 'tfidf', 'semantic'],
                format_func=lambda x: {
                    'hybrid': 'Hybrid',
                    'tfidf': 'TF-IDF',
                    'semantic': 'Semantic'
                }[x]
            )
        
        num_results = st.slider("Number of results:", 1, 20, 10)
        
        if st.button("Search", type="primary") and search_query:
            with st.spinner("Searching movies..."):
                search_results = recommender.search_movies(
                    search_query, 
                    num_results,
                    method=search_method
                )
            
            if search_results:
                st.success(f"Found {len(search_results)} movies matching '{search_query}'")
                
                for i, movie in enumerate(search_results, 1):
                    st.markdown(f"### {i}. Search Result")
                    display_movie_card(movie)
            else:
                st.warning("No movies found matching your search.")
    
    with tab3:
        st.header("Compare Recommendation Methods")
        
        # Movie selection for comparison
        comparison_movie = st.selectbox(
            "Select a movie to compare methods:",
            options=movie_titles,
            key="comparison_movie"
        )
        
        num_comp_recs = st.slider("Number of recommendations per method:", 1, 10, 5)
        
        if st.button("Compare Methods", type="primary"):
            with st.spinner("Comparing recommendation methods..."):
                comparison_results = recommender.compare_methods(
                    comparison_movie, 
                    num_comp_recs
                )
            
            # Display comparison chart
            chart = create_comparison_chart(comparison_results)
            st.plotly_chart(chart, use_container_width=True)
            
            # Display detailed results
            methods = [
                ('tfidf', 'TF-IDF Based'),
                ('semantic', 'Semantic Embedding Based'),
                ('hybrid', 'Hybrid (Weighted Combination)'),
                ('ensemble', 'Ensemble (Rank Fusion)')
            ]
            
            for method_key, method_name in methods:
                if method_key in comparison_results:
                    st.subheader(f"{method_name} Recommendations")
                    recommendations = comparison_results[method_key]
                    
                    if recommendations:
                        for i, movie in enumerate(recommendations, 1):
                            with st.expander(f"{i}. {movie['title']} (Score: {movie['similarity_score']:.3f})"):
                                display_movie_card(movie, show_score=False)
                    else:
                        st.write("No recommendations found for this method.")
    
    with tab4:
        st.header("Dataset Analytics")
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", len(movies_df))
        
        with col2:
            avg_rating = movies_df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.1f}")
        
        with col3:
            year_range = f"{movies_df['release_year'].min()}-{movies_df['release_year'].max()}"
            st.metric("Year Range", year_range)
        
        with col4:
            total_directors = movies_df['director'].nunique()
            st.metric("Unique Directors", total_directors)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre distribution
            genre_chart = create_genre_distribution_chart(movies_df)
            st.plotly_chart(genre_chart, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig = px.histogram(
                movies_df, 
                x='rating', 
                nbins=20,
                title="Movie Rating Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Release year trend
        year_counts = movies_df.groupby('release_year').size().reset_index(name='count')
        fig = px.line(
            year_counts, 
            x='release_year', 
            y='count',
            title="Movies Released by Year"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 System Overview
        
        This advanced movie recommendation system combines two powerful approaches:
        
        1. **TF-IDF Cosine Similarity**: Analyzes textual features using term frequency-inverse document frequency
        2. **Sentence-Transformers Semantic Embeddings**: Captures deep semantic meaning using pre-trained neural networks
        
        ### 🔧 Technical Features
        
        - **Hybrid Approach**: Combines TF-IDF and semantic similarities with configurable weights
        - **Multiple Methods**: Choose from TF-IDF, Semantic, Hybrid, or Ensemble recommendations
        - **Advanced Search**: Semantic and keyword-based movie search capabilities
        - **Method Comparison**: Compare different recommendation approaches side-by-side
        - **Interactive Analytics**: Explore dataset characteristics and patterns
        
        ### 📊 Recommendation Methods
        
        - **TF-IDF**: Fast, keyword-based similarity using bag-of-words approach
        - **Semantic**: Deep understanding using Sentence-Transformers (all-MiniLM-L6-v2)
        - **Hybrid**: Weighted combination of TF-IDF and semantic scores
        - **Ensemble**: Rank fusion combining multiple approaches
        
        ### 🛠️ Technology Stack
        
        - **Frontend**: Streamlit
        - **ML/NLP**: scikit-learn, Sentence-Transformers
        - **Data Processing**: pandas, numpy
        - **Visualization**: Plotly
        
        ### 🎬 Dataset
        
        The system includes a diverse dataset of 500 movies with:
        - Multiple genres and themes
        - Rich metadata (director, cast, overview, keywords)
        - Both classic and modern films
        """)
        
        # System parameters
        st.subheader("Current System Configuration")
        
        config_data = {
            "Parameter": [
                "TF-IDF Weight",
                "Semantic Weight", 
                "Semantic Model",
                "TF-IDF Max Features",
                "Dataset Size"
            ],
            "Value": [
                f"{recommender.tfidf_weight:.1f}",
                f"{recommender.semantic_weight:.1f}",
                recommender.semantic_recommender.model_name,
                "10,000",
                f"{len(movies_df)} movies"
            ]
        }
        
        config_df = pd.DataFrame(config_data)
        st.table(config_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit • Advanced Content-Based Recommendation System"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()