from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from typing import Dict, Any

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
ARTIFACT_PATH = "diabetes_model_20250715_130113.pkl"
artifacts = joblib.load(ARTIFACT_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']

print("Model loaded with features:", feature_names)  # Debugging

# Pydantic model based on selected features
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float
    Glucose_BMI_Ratio: float
    Age_Glucose_Int: float
    Insulin_BMI_Ratio: float
    Age_BMI_Int: float
    Is_Obese: float
    Is_Young: float
    Glucose2: float
    BMI2: float
    Pregnancies_log1p: float
    Insulin_log1p: float

@app.get("/")
async def home():
    return {"message": "Diabetes Prediction API is running."}

@app.post("/predict")
async def predict(input_data: Dict[str, Any] = Body(...)):
    try:
        print("Received input:", input_data)  # Debugging

        # Verify we have all required features
        missing_features = [feat for feat in feature_names if feat not in input_data]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )

        # Create DataFrame with correct feature order
        ordered_input = [input_data[feat] for feat in feature_names]
        df = pd.DataFrame([ordered_input], columns=feature_names)

        # Scale features
        scaled = scaler.transform(df)

        # Make prediction
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        return {
            "predicted_class": "Diabetes" if prediction == 1 else "No Diabetes",
            "probability": round(probability * 100, 2),
            "risk_level": get_risk_level(probability),
            "used_features": feature_names
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "expected_features": feature_names,
                "received_data": input_data
            }
        )

def get_risk_level(probability):
    if probability > 0.7:
        return "High risk - consult a doctor"
    elif probability > 0.3:
        return "Moderate risk - consider lifestyle changes"
    return "Low risk - maintain healthy habits"