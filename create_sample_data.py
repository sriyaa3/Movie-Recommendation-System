import pandas as pd
import numpy as np
import random
from typing import List

def create_sample_movie_dataset(n_movies: int = 500) -> pd.DataFrame:
    """
    Create a sample movie dataset for testing the recommendation system
    
    Args:
        n_movies: Number of movies to generate
        
    Returns:
        DataFrame with sample movie data
    """
    
    # Sample genres
    genres_list = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "Thriller", "War", "Western"
    ]
    
    # Sample directors
    directors = [
        "Christopher Nolan", "Steven Spielberg", "Martin Scorsese", "Quentin Tarantino",
        "James Cameron", "Ridley Scott", "David Fincher", "Tim Burton", "Wes Anderson",
        "Coen Brothers", "Paul Thomas Anderson", "Denis Villeneuve", "Jordan Peele",
        "Greta Gerwig", "Kathryn Bigelow", "Ari Aster", "Ryan Coogler", "Chloé Zhao"
    ]
    
    # Sample actors
    actors = [
        "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks", "Scarlett Johansson",
        "Robert Downey Jr.", "Jennifer Lawrence", "Brad Pitt", "Angelina Jolie",
        "Will Smith", "Emma Stone", "Ryan Gosling", "Natalie Portman", "Matt Damon",
        "Cate Blanchett", "Christian Bale", "Amy Adams", "Michael Fassbender",
        "Margot Robbie", "Oscar Isaac", "Lupita Nyong'o"
    ]
    
    # Keywords for different genres
    keywords_by_genre = {
        "Action": ["explosion", "chase", "fight", "hero", "villain", "rescue", "weapons", "combat"],
        "Adventure": ["journey", "quest", "exploration", "treasure", "discovery", "exotic", "survival"],
        "Animation": ["cartoon", "family", "colorful", "magical", "talking animals", "fairy tale"],
        "Comedy": ["funny", "humor", "laugh", "silly", "romantic comedy", "satire", "parody"],
        "Crime": ["detective", "police", "investigation", "murder", "criminal", "heist", "corruption"],
        "Documentary": ["real life", "factual", "interview", "truth", "history", "nature", "biography"],
        "Drama": ["emotional", "character study", "relationships", "tragedy", "family", "coming of age"],
        "Family": ["children", "wholesome", "adventure", "learning", "friendship", "values"],
        "Fantasy": ["magic", "mythical", "supernatural", "medieval", "dragons", "wizards", "prophecy"],
        "History": ["historical", "period piece", "war", "biography", "ancient", "revolution"],
        "Horror": ["scary", "supernatural", "monster", "ghost", "haunted", "fear", "suspense"],
        "Music": ["musical", "singing", "dance", "concert", "rock", "classical", "jazz"],
        "Mystery": ["puzzle", "clues", "detective", "secret", "investigation", "twist", "unknown"],
        "Romance": ["love", "relationship", "dating", "marriage", "heartbreak", "passion", "wedding"],
        "Science Fiction": ["future", "space", "technology", "aliens", "robots", "dystopia", "time travel"],
        "Thriller": ["suspense", "tension", "chase", "danger", "conspiracy", "psychological", "edge"],
        "War": ["battle", "military", "soldiers", "conflict", "victory", "sacrifice", "strategy"],
        "Western": ["cowboys", "frontier", "saloon", "desert", "horses", "gunfight", "sheriff"]
    }
    
    # Sample movie titles by genre
    title_templates = {
        "Action": ["The Last {}", "Rise of {}", "Mission: {}", "Operation {}", "The {} Protocol"],
        "Adventure": ["Quest for {}", "Journey to {}", "The {} Expedition", "Lost in {}", "Beyond {}"],
        "Animation": ["The {} Adventure", "{} and Friends", "Magic of {}", "The Little {}", "{}land"],
        "Comedy": ["The {} Comedy", "Crazy {}", "{} Happens", "The {} Show", "Life of {}"],
        "Crime": ["The {} Case", "{} Files", "Blood {}", "The {} Connection", "Dark {}"],
        "Drama": ["The {} Story", "Heart of {}", "The {} Chronicles", "Moments of {}", "Finding {}"],
        "Fantasy": ["The {} Kingdom", "Legends of {}", "The Magic {}", "Realm of {}", "The {} Prophecy"],
        "Horror": ["The {} Horror", "Night of {}", "The {} Curse", "Return of {}", "The {} Nightmare"],
        "Romance": ["Love in {}", "The {} Affair", "Heart of {}", "Finding {}", "The {} Romance"],
        "Science Fiction": ["2084: {}", "Star {}", "The {} Matrix", "Future {}", "Planet {}"],
        "Thriller": ["The {} Conspiracy", "Edge of {}", "The {} Game", "Midnight {}", "The {} Code"],
        "War": ["Battle of {}", "The {} War", "Victory at {}", "The {} Campaign", "Heroes of {}"]
    }
    
    # Generate movies
    movies = []
    
    for i in range(n_movies):
        # Randomly select 1-3 genres
        num_genres = random.randint(1, 3)
        movie_genres = random.sample(genres_list, num_genres)
        primary_genre = movie_genres[0]
        
        # Generate title
        title_template = random.choice(title_templates.get(primary_genre, ["The {} Story"]))
        title_word = random.choice(["Shadow", "Fire", "Crystal", "Storm", "Golden", "Silver", 
                                  "Dark", "Bright", "Silent", "Wild", "Secret", "Ancient"])
        title = title_template.format(title_word)
        
        # Generate overview
        genre_keywords = []
        for genre in movie_genres:
            genre_keywords.extend(random.sample(keywords_by_genre.get(genre, []), 2))
        
        overview_templates = [
            "A thrilling story about {} and {} in a world of {}.",
            "When {} meets {}, they must overcome {} to save the day.",
            "An epic tale of {} featuring {} and unexpected {}.",
            "A journey through {} where {} leads to {} and beyond.",
            "The story follows a hero dealing with {} while facing {} and {}."
        ]
        
        overview = random.choice(overview_templates).format(
            random.choice(genre_keywords),
            random.choice(genre_keywords),
            random.choice(genre_keywords)
        )
        
        # Select cast and director
        director = random.choice(directors)
        cast_size = random.randint(3, 6)
        cast = random.sample(actors, cast_size)
        
        # Generate other fields
        year = random.randint(1990, 2024)
        rating = round(random.uniform(3.0, 9.5), 1)
        runtime = random.randint(80, 180)
        
        # Combine keywords
        all_keywords = random.sample(genre_keywords, min(len(genre_keywords), 5))
        
        movie = {
            'id': i + 1,
            'title': title,
            'overview': overview,
            'genres': ' '.join(movie_genres),
            'keywords': ' '.join(all_keywords),
            'director': director,
            'cast': ' '.join(cast),
            'release_year': year,
            'rating': rating,
            'runtime': runtime,
            'budget': random.randint(1000000, 200000000) if random.random() > 0.3 else None,
            'revenue': random.randint(500000, 2000000000) if random.random() > 0.3 else None
        }
        
        movies.append(movie)
    
    df = pd.DataFrame(movies)
    return df

def create_enhanced_dataset() -> pd.DataFrame:
    """
    Create an enhanced dataset with more realistic movie data
    """
    
    # Famous movies data
    famous_movies = [
        {
            'id': 1, 'title': 'The Dark Knight',
            'overview': 'Batman faces the Joker in a battle for Gotham City\'s soul. A thrilling crime drama with psychological depth.',
            'genres': 'Action Crime Drama', 'keywords': 'superhero batman joker crime psychological',
            'director': 'Christopher Nolan', 'cast': 'Christian Bale Heath Ledger Aaron Eckhart',
            'release_year': 2008, 'rating': 9.0, 'runtime': 152
        },
        {
            'id': 2, 'title': 'Inception',
            'overview': 'A thief who enters people\'s dreams to steal secrets gets a chance to have his criminal record erased.',
            'genres': 'Action Science Fiction Thriller', 'keywords': 'dreams mind heist surreal layered',
            'director': 'Christopher Nolan', 'cast': 'Leonardo DiCaprio Ellen Page Tom Hardy',
            'release_year': 2010, 'rating': 8.8, 'runtime': 148
        },
        {
            'id': 3, 'title': 'Pulp Fiction',
            'overview': 'The lives of two mob hitmen, a boxer, and others intertwine in four tales of violence and redemption.',
            'genres': 'Crime Drama', 'keywords': 'nonlinear narrative violence redemption dialogue',
            'director': 'Quentin Tarantino', 'cast': 'John Travolta Samuel L. Jackson Uma Thurman',
            'release_year': 1994, 'rating': 8.9, 'runtime': 154
        },
        {
            'id': 4, 'title': 'The Shawshank Redemption',
            'overview': 'Two imprisoned men bond over years, finding solace and redemption through acts of common decency.',
            'genres': 'Drama', 'keywords': 'prison friendship hope redemption corruption',
            'director': 'Frank Darabont', 'cast': 'Tim Robbins Morgan Freeman Bob Gunton',
            'release_year': 1994, 'rating': 9.3, 'runtime': 142
        },
        {
            'id': 5, 'title': 'Interstellar',
            'overview': 'A team of explorers travel beyond this galaxy through a wormhole in an attempt to ensure humanity\'s survival.',
            'genres': 'Adventure Drama Science Fiction', 'keywords': 'space time wormhole family sacrifice',
            'director': 'Christopher Nolan', 'cast': 'Matthew McConaughey Anne Hathaway Jessica Chastain',
            'release_year': 2014, 'rating': 8.6, 'runtime': 169
        }
    ]
    
    # Generate additional movies
    additional_movies = create_sample_movie_dataset(495)
    
    # Combine famous movies with generated ones
    famous_df = pd.DataFrame(famous_movies)
    
    # Update IDs for additional movies
    additional_movies['id'] = additional_movies['id'] + 5
    
    # Combine datasets
    complete_df = pd.concat([famous_df, additional_movies], ignore_index=True)
    
    return complete_df

if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample movie dataset...")
    movies_df = create_enhanced_dataset()
    
    # Save to CSV
    movies_df.to_csv('/Users/sivaramand/Downloads/movie recommendation system/sample_movies.csv', index=False)
    print(f"Created dataset with {len(movies_df)} movies")
    print("Sample movies:")
    print(movies_df[['title', 'genres', 'director', 'release_year', 'rating']].head(10))
    
    # Save smaller test dataset
    test_df = movies_df.head(100)
    test_df.to_csv('/Users/sivaramand/Downloads/movie recommendation system/test_movies.csv', index=False)
    print(f"\nCreated test dataset with {len(test_df)} movies")