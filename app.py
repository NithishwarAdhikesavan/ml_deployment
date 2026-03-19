from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import joblib

app = FastAPI()
# Load the pre-trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')

# Load the pre-trained StandardScaler
scaler = joblib.load('standard_scaler.joblib')

print("Model and Scaler loaded successfully.")


# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


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

    # Better response
    return {
        "prediction": outcome,
        "result": "Diabetic" if outcome == 1 else "Not Diabetic"
    }
