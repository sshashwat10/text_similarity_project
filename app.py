# app.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn

app = FastAPI()

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Request structure
class SimilarityRequest(BaseModel):
    text1: str
    text2: str

# Response structure
class SimilarityResponse(BaseModel):
    similarity_score: float

@app.post("/predict", response_model=SimilarityResponse)
async def predict_similarity(req: SimilarityRequest):
    embeddings = model.encode([req.text1, req.text2], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return {"similarity_score": round(score.item(), 4)}

# Vercel serverless handler
from mangum import Mangum
handler = Mangum(app, lifespan="off")  # Disable lifespan to reduce memory usage
