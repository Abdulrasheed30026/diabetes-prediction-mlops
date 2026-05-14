"""
app.py — FastAPI Diabetes Prediction API
MLOps Assignment 1
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
import uvicorn

# ── Load saved artifacts ──────────────────────────────────────────────────────
model           = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")

# ── FastAPI app instance ──────────────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Prediction API",
    description="MLOps SP26 — Assignment 1: Predict diabetes class from patient data.",
    version="1.0.0",
)

# ── Pydantic input model ──────────────────────────────────────────────────────
class PatientData(BaseModel):
    age:    float
    urea:   float
    cr:     float
    hba1c:  float
    chol:   float
    tg:     float
    hdl:    float
    ldl:    float
    vldl:   float
    bmi:    float
    gender: str   # must be "M" or "F"

    @validator("gender")
    def gender_must_be_valid(cls, v):
        v = v.strip().upper()
        if v not in ("M", "F"):
            raise ValueError("gender must be 'M' or 'F'")
        return v

# ── Health-check endpoint ─────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Health check — confirms the API is running."""
    return {"status": "API is running"}

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict(patient: PatientData):
    """
    Receive patient data via POST and return a diabetes class prediction.

    - **gender**: "M" or "F" (case-insensitive)
    - Returns the predicted class and a human-readable label.
    """
    # 1. Build a raw DataFrame from incoming data
    input_dict = {
        "AGE":   [patient.age],
        "Urea":  [patient.urea],
        "Cr":    [patient.cr],
        "HbA1c": [patient.hba1c],
        "Chol":  [patient.chol],
        "TG":    [patient.tg],
        "HDL":   [patient.hdl],
        "LDL":   [patient.ldl],
        "VLDL":  [patient.vldl],
        "BMI":   [patient.bmi],
        "Gender":[patient.gender],
    }
    input_df = pd.DataFrame(input_dict)

    # 2. One-hot encode the Gender column
    input_df = pd.get_dummies(input_df, columns=["Gender"], drop_first=False)

    # 3. Align columns with training data (add missing dummies as 0)
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only columns seen during training, in the correct order
    input_df = input_df[training_columns]

    # 4. Predict
    prediction = model.predict(input_df)[0]

    # 5. Build readable label
    label_map = {
        "Y": "Diabetic",
        "N": "Non-Diabetic",
        "P": "Pre-Diabetic",
    }
    label = label_map.get(str(prediction).upper(), str(prediction))

    return {
        "prediction": str(prediction),
        "label":      label,
        "message":    f"Patient is predicted to be: {label}",
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
