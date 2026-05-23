# ============================================
# server.py
# ============================================

"""
FastAPI Server

Responsibilities:
- receive API requests
- validate schemas
- call inference engine
- return predictions

The actual ML logic lives in:
inference.py

This keeps architecture modular.
"""

from fastapi import FastAPI

from app.schemas import (
    PKEvent,
    PredictionResponse
)

from app.inference import PKInferenceEngine

from app.state_manager import reset_patient


# ============================================
# CREATE FASTAPI APP
# ============================================

app = FastAPI(

    title="Remifentanil PK Prediction API",

    description=(
        "Stateful XGBoost-based "
        "pharmacokinetic system for remifentanil plasma concentration prediction."
    ),

    version="1.0"
)


# ============================================
# LOAD INFERENCE ENGINE
# ============================================

inference_engine = PKInferenceEngine(

    model_path="app/models/xgboost_model.json"
)


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
def home():

    return {

        "message":
        "Remifentanil PK API Running"
    }


# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(event: PKEvent):
    """
    Predict remifentanil concentration
    from sequential infusion event.
    """

    result = inference_engine.predict(event)

    return PredictionResponse(

        patient_id=event.patient_id,

        predicted_concentration=round(
            result["prediction"],
            4
        ),

        cumulative_dose=round(
            result["CumulativeDose"],
            4
        ),

        infusion_active=result[
            "infusion_active"
        ]
    )


# ============================================
# RESET PATIENT STATE
# ============================================

@app.delete("/reset/{patient_id}")
def reset(patient_id: str):
    """
    Clears stored patient state.
    """

    success = reset_patient(patient_id)

    if success:

        return {

            "message":
            f"Patient {patient_id} reset."
        }

    return {

        "message":
        "Patient not found."
    }
