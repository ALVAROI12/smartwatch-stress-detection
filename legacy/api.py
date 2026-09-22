
"""FastAPI endpoint for Stress Detection Model"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List, Optional
import json

app = FastAPI(
    title="Stress Detection API",
    description="Real-time stress detection from wearable sensor features",
    version="1.0.0"
)

# Load models at startup
model = joblib.load("outputs/models/xgboost_optimized.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")  # Need to save this
with open("outputs/model_card.json") as f:
    model_card = json.load(f)

FEATURE_NAMES = [  # 39 features
    "bvp_mean", "bvp_std", "bvp_min", "bvp_max", "bvp_range",
    "hr_mean", "hr_std", "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
    "eda_mean", "eda_std", "eda_min", "eda_max", "eda_range",
    "eda_scr_count", "eda_scr_amp_mean", "eda_tonic_mean", "eda_phasic_mean",
    "temp_mean", "temp_std", "temp_min", "temp_max", "temp_range", "temp_slope",
    "acc_x_mean", "acc_y_mean", "acc_z_mean", "acc_x_std", "acc_y_std", "acc_z_std",
    "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max",
    "acc_sma", "acc_energy", "acc_entropy", "eda_slope"
]

CLASS_NAMES = ["Aerobic", "Amusement", "Anaerobic", "Baseline", "Emotion", "Stress"]


class SensorFeatures(BaseModel):
    """Input features from wearable sensors"""
    features: List[float]  # 39 features
    return_probabilities: Optional[bool] = True
    confidence_threshold: Optional[float] = None  # Rejection option


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: str
    confidence: float
    probabilities: Optional[dict] = None
    rejected: bool = False
    message: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Stress Detection API", "version": "1.0.0"}


@app.get("/model-info")
async def get_model_info():
    return {
        "model_name": model_card["model_details"]["name"],
        "version": model_card["model_details"]["version"],
        "accuracy": model_card["performance_metrics"]["overall"]["accuracy"],
        "classes": CLASS_NAMES,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorFeatures):
    # Validate input
    if len(data.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features, got {len(data.features)}"
        )
    
    # Preprocess
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Predict
    proba = model.predict_proba(X_scaled)[0]
    pred_idx = np.argmax(proba)
    confidence = float(proba[pred_idx])
    prediction = CLASS_NAMES[pred_idx]
    
    # Rejection option
    rejected = False
    message = None
    if data.confidence_threshold and confidence < data.confidence_threshold:
        rejected = True
        message = f"Prediction rejected: confidence {confidence:.3f} < threshold {data.confidence_threshold}"
        prediction = "uncertain"
    
    # Response
    response = PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        rejected=rejected,
        message=message
    )
    
    if data.return_probabilities:
        response.probabilities = {cls: float(p) for cls, p in zip(CLASS_NAMES, proba)}
    
    return response


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Run with: uvicorn api:app --reload --port 8000
