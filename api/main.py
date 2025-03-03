from typing import Union
from fastapi import FastAPI
from utils import recommend_movie
from utils import model
from utils import desc_embeddings
from utils import df
from utils import display_recommendations
from utils import getTop4RandomRecommendations
from typing import List, Dict

app = FastAPI()

@app.get("/getRecommendations")
def getRecommendations(movie_description: str) -> List[Dict[str, str]]:
    recommendations = recommend_movie(movie_description, model, desc_embeddings, df, top_n=4)
    return display_recommendations(recommendations)

@app.get("/getRandomRecommendations")
def getRandomRecommendations() -> List[Dict[str, str]]:
    return getTop4RandomRecommendations(df)