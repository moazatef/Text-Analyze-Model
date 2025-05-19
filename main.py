from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re
import os
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Prediction API",
    description="API for predicting mental health conditions from text",
    version="1.0.0"
)

# Enable CORS (important for frontend or mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "model.h5"))
with open(os.path.join(BASE_DIR, "tokenizer.pickle"), "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_LENGTH = 512
MOODS = ["Anxiety", "Depression", "Normal", "Stress"]

# Request and Response models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_mood: str
    confidence: float
    top_predictions: List[dict]

# Text preprocessing
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

# Prediction logic
def get_predictions(text: str) -> tuple:
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    predictions = model.predict(padded, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1]
    main_prediction = MOODS[top_indices[0]]
    confidence = float(predictions[top_indices[0]]) * 100

    all_predictions = [
        {"mood": MOODS[idx], "confidence": float(predictions[idx]) * 100}
        for idx in top_indices
    ]

    return main_prediction, confidence, all_predictions

# Routes
@app.get("/")
async def root():
    return {
        "message": "Mental Health Prediction API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/docs": "Swagger documentation"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text is too short. Please provide a longer description."
        )

    try:
        predicted_mood, confidence, all_predictions = get_predictions(request.text)
        return PredictionResponse(
            predicted_mood=predicted_mood,
            confidence=confidence,
            top_predictions=all_predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
