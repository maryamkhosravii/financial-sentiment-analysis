from fastapi import FastAPI
from api.schemas import PredictionRequest, PredictionResponse
from src.inference import InferenceService


app=FastAPI(title="Financial Sentiment Analysis API", version="1.0.0", description="Financial sentiment analysis using FinBERT")

service = InferenceService()

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "FinBERT"
    }


@app.post ("/predict", response_model=PredictionResponse)
def predict (request: PredictionRequest):
    return service.predict (request.text)