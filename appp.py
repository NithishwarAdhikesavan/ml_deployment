from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# Load model and scaler using joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "logistic_regression_model.joblib")
scaler_path = os.path.join(BASE_DIR, "standard_scaler.joblib")


print("Loading model from:", model_path)
model = joblib.load(model_path)

print("Loading scaler from:", scaler_path)
scaler = joblib.load(scaler_path)

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

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    outcome = int(prediction[0])

    return {
        "prediction": outcome,
        "result": "Diabetic" if outcome == 1 else "Not Diabetic"
    }
