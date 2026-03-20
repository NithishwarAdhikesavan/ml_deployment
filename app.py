from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

# Load model and scaler using pickle (single, consistent method)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

print("Loading model from:", model_path)
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Loading scaler from:", scaler_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

print("Model and Scaler loaded successfully.")

# Input schema
class DiabetesFeatures(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Root endpoint (important for Render health checks)
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}

@app.post("/predict")
def predict_diabetes(features: DiabetesFeatures):
    # Convert input to numpy array
    input_data = np.array([
        features.Pregnancies,
        features.Glucose,
        features.BloodPressure,
        features.SkinThickness,
        features.Insulin,
        features.BMI,
        features.DiabetesPedigreeFunction,
        features.Age
    ]).reshape(1, -1)

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_input)
    outcome = int(prediction[0])

    return {
        "prediction": outcome,
        "result": "Diabetic" if outcome == 1 else "Not Diabetic"
    }
