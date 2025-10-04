# 🎬 Movie Recommendation System

An advanced content-based movie recommendation system that combines **TF-IDF cosine similarity** with **Sentence-Transformers semantic embeddings** to provide highly accurate movie recommendations.

## 🚀 Features

### 🎯 Recommendation Methods
- **TF-IDF Based**: Fast, keyword-based similarity using term frequency-inverse document frequency
- **Semantic Embeddings**: Deep semantic understanding using Sentence-Transformers (all-MiniLM-L6-v2)
- **Hybrid Approach**: Intelligently combines TF-IDF and semantic similarities with configurable weights
- **Ensemble Method**: Advanced rank fusion combining multiple approaches

### 🔍 Interactive Features
- **Movie Recommendations**: Get personalized recommendations based on a movie you like
- **Smart Search**: Search movies by themes, genres, or keywords using both TF-IDF and semantic methods
- **Method Comparison**: Compare different recommendation approaches side-by-side
- **Interactive Analytics**: Explore dataset characteristics and recommendation patterns
- **Real-time Results**: All recommendations generated instantly with similarity scores

### 📊 Advanced Analytics
- Genre distribution analysis
- Rating and year trend visualization
- Recommendation method performance comparison
- Similarity score visualizations

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **ML/NLP**: 
  - scikit-learn (TF-IDF, cosine similarity)
  - Sentence-Transformers (semantic embeddings)
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly (interactive charts)
- **Backend**: Python with modular architecture

## 📁 Project Structure

```
movie recommendation system/
├── streamlit_app.py              # Main Streamlit application
├── data_processor.py             # Data preprocessing and cleaning
├── tfidf_recommender.py          # TF-IDF based recommendations
├── semantic_recommender.py       # Semantic embeddings recommendations
├── hybrid_recommender.py         # Hybrid recommendation engine
├── create_sample_data.py         # Sample dataset generator
├── test_system.py               # System testing script
├── requirements.txt             # Python dependencies
├── sample_movies.csv            # Movie dataset (500 movies)
├── test_movies.csv              # Smaller test dataset (100 movies)
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd "movie recommendation system"

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data (if needed)

```bash
python create_sample_data.py
```

### 3. Test the System

```bash
python test_system.py
```

### 4. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8502`

## 🎮 How to Use

### Getting Recommendations
1. **Select a Movie**: Choose from 500+ movies in the dropdown
2. **Choose Method**: Pick from TF-IDF, Semantic, Hybrid, or Ensemble
3. **Set Number**: Select how many recommendations you want (1-20)
4. **Get Results**: Click "Get Recommendations" to see personalized suggestions

### Searching Movies
1. **Enter Keywords**: Type themes, genres, or descriptive terms
2. **Choose Search Method**: Select TF-IDF, Semantic, or Hybrid search
3. **View Results**: See movies ranked by relevance to your query

### Comparing Methods
1. **Select a Movie**: Choose a movie for comparison
2. **Set Recommendations**: Decide how many recommendations per method
3. **Compare Results**: See how different methods perform side-by-side

## 🧠 How It Works

### TF-IDF Approach
- Analyzes movie descriptions, genres, keywords, cast, and director information
- Creates feature vectors based on term frequency and document frequency
- Uses cosine similarity to find movies with similar textual characteristics
- **Best for**: Finding movies with similar keywords and descriptions

### Semantic Embeddings
- Uses pre-trained Sentence-Transformers (all-MiniLM-L6-v2) model
- Captures deep semantic meaning beyond just keywords
- Understands context and relationships between concepts
- **Best for**: Finding movies with similar themes and concepts

### Hybrid Method
- Combines TF-IDF and semantic approaches with configurable weights
- Default: 40% TF-IDF + 60% Semantic
- Balances keyword matching with semantic understanding
- **Best for**: Most accurate and well-rounded recommendations

### Ensemble Method
- Uses rank fusion to combine multiple recommendation approaches
- Considers ranking positions rather than just similarity scores
- Provides robust recommendations across different methods
- **Best for**: Most diverse and comprehensive recommendations

## 📊 Dataset

The system includes a diverse dataset of **500 movies** featuring:

- **Multiple Genres**: Action, Drama, Comedy, Sci-Fi, Horror, Romance, etc.
- **Rich Metadata**: Director, cast, overview, keywords, ratings, runtime
- **Time Range**: Movies from 1990-2024
- **Quality Content**: Mix of blockbusters and acclaimed films
- **Famous Movies**: Includes popular films like "The Dark Knight", "Inception", "Pulp Fiction"

## ⚙️ Configuration

### Recommendation Weights (Hybrid Method)
- **TF-IDF Weight**: Default 0.4 (40%)
- **Semantic Weight**: Default 0.6 (60%)
- Can be adjusted in `hybrid_recommender.py`

### Model Parameters
- **TF-IDF Max Features**: 10,000
- **Semantic Model**: all-MiniLM-L6-v2
- **N-gram Range**: (1, 2) for unigrams and bigrams

## 🔧 Customization

### Adding New Movies
1. Add movies to `sample_movies.csv` with required columns:
   - `id`, `title`, `overview`, `genres`, `keywords`, `director`, `cast`
2. Restart the application to reload data

### Changing Semantic Model
1. Modify `semantic_model` parameter in `HybridRecommender` initialization
2. Available models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, etc.

### Adjusting Weights
1. Change `tfidf_weight` and `semantic_weight` in `streamlit_app.py`
2. Ensure weights sum to 1.0

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

Tests include:
- Data processing functionality
- TF-IDF recommendation accuracy
- Semantic embedding generation
- Hybrid method integration
- Search functionality

## 🚀 Advanced Features

### Method Comparison
- Compare all four recommendation methods side-by-side
- Visualize average similarity scores
- Analyze recommendation overlap and differences

### Analytics Dashboard
- Dataset overview with key statistics
- Genre distribution charts
- Rating distribution histograms
- Release year trends
- Interactive Plotly visualizations

### Smart Caching
- Processed data and models are cached for faster loading
- Automatic cache invalidation when data changes
- Reduced startup time on subsequent runs

## 🎯 Performance

- **TF-IDF**: Near-instant recommendations (< 1 second)
- **Semantic**: First-time model loading (~30 seconds), then fast recommendations
- **Hybrid**: Combines both methods efficiently
- **Caching**: Significant performance improvement on repeat usage

## 📈 Future Enhancements

- **Collaborative Filtering**: Add user-based recommendations
- **Deep Learning Models**: Integrate advanced neural networks
- **Real-time Data**: Connect to live movie databases
- **User Profiles**: Personalized recommendation learning
- **A/B Testing**: Compare recommendation algorithm performance

## 🤝 Contributing

Feel free to enhance the system by:
1. Adding new recommendation algorithms
2. Improving the user interface
3. Expanding the dataset
4. Optimizing performance
5. Adding new visualization features

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Python, Streamlit, and cutting-edge NLP techniques**

Enjoy exploring movies with AI-powered recommendations! 🍿🎬
