from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pickle
import re
import requests
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Prediction API",
    description="API for predicting mental health conditions from text",
    version="1.0.0"
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

# Define request model
class PredictionRequest(BaseModel):
    text: str

# Define response model
class PredictionResponse(BaseModel):
    predicted_mood: str
    confidence: float
    top_predictions: List[dict]

# Load model and tokenizer
try:
    model = load_model("model.h5")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ Model and tokenizer loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model or tokenizer: {e}")
    raise

# Constants
MAX_LENGTH = 512
MOODS = ["Anxiety", "Depression", "Normal", "Stress"]

def clean_text(text: str) -> str:
    """Clean and preprocess the input text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def get_predictions(text: str) -> tuple:
    """Get model predictions for the input text"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # Pad sequence
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # Get predictions
    predictions = model.predict(padded, verbose=0)[0]
    
    # Get top predictions
    top_indices = np.argsort(predictions)[::-1]
    main_prediction = MOODS[top_indices[0]]
    confidence = float(predictions[top_indices[0]])
    
    # Get all predictions with confidences
    all_predictions = [
        {"mood": MOODS[idx], "confidence": float(predictions[idx]) * 100}
        for idx in top_indices
    ]
    
    return main_prediction, confidence * 100, all_predictions

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict mental health condition from text
    """
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
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mental Health Prediction API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
