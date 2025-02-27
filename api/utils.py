# Import necessary libraries
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rich.console import Console
from rich.table import Table
from rich import box
from typing import List, Dict
import requests

# Determine whether to use a GPU (CUDA) if available or fall back to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
console = Console()
console.print(f"[bold green]Using device: {device}[/bold green]")

# Load the Netflix dataset and remove entries with missing descriptions
df = pd.read_csv('netflix_titles.csv')
df = df.dropna(subset=['description']).reset_index(drop=True)

# Load the pre-trained Sentence Transformer model on the specified device
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

# Variables to get movie preview from TMDB
TMDB_API_KEY = ""
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Extract descriptions from the DataFrame and compute embeddings
descriptions = df['description'].tolist()
desc_embeddings = model.encode(descriptions, convert_to_tensor=True, device=device)

def get_movie_poster(movie_title: str) -> str:
    params = {"query": movie_title, "api_key": TMDB_API_KEY}
    response = requests.get(TMDB_SEARCH_URL, params=params)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"{IMAGE_BASE_URL}{poster_path}"
    return ""

def recommend_movie(query, model, desc_embeddings, df, top_n=5):
     # Compute embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)

    # Calculate cosine similarity between the query and all movie descriptions
    cosine_scores = util.cos_sim(query_embedding, desc_embeddings)[0].cpu().numpy()

    # Move scores to CPU and convert to a numpy array for further processing
    top_indices = np.argpartition(-cosine_scores, range(top_n))[:top_n]
    top_indices = top_indices[np.argsort(-cosine_scores[top_indices])]

    # Prepare a DataFrame with the top recommendations and compute match percentage
    recommended_df = df.iloc[top_indices][['title', 'description', 'listed_in', 'release_year']].copy()
    recommended_df['match_percentage'] = cosine_scores[top_indices] * 100
    return recommended_df

def display_recommendations(recommendations: pd.DataFrame) -> List[Dict[str, str]]:
    recommended_movies = []

    # Iterate through the recommendations and add them to the list
    for _, row in recommendations.iterrows():
        movie_title = row['title']
        poster_url = get_movie_poster(movie_title)
        movie_details = {
            "title": movie_title,
            "description": row['description'],
            "listed_in": row['listed_in'],
            "release_year": str(row['release_year']),
            "match_percentage": f"{row['match_percentage']:.2f}%",
            "poster_url": poster_url
        }
        recommended_movies.append(movie_details)
    return recommended_movies