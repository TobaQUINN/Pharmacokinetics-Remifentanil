import uvicorn
from fastapi import FastAPI
from xgboost import XGBRegressor

import pandas as pd

from schemas import PKInput
from feature_engineering import engineer_features


# Create API
app = FastAPI(
    title="Remifentanil PK Predictor"
)

# Load trained model
xgboost_model = XGBRegressor()
xgboost_model.load_model(
    "src/app/models/xgboost_model.json"
)

# Feature order
FEATURES = [

    'Time',
    'delta_time',

    'Rate',
    'Amt',

    'cum_dose',

    'Age',
    'Sex',

    'Ht',
    'Wt',

    'BSA',
    'LBM',

    'InfusionActive'
]


@app.get("/")
def home():

    return {
        "message":
        "Remifentanil PK Prediction API"
    }


@app.post("/predict")
def predict(data: PKInput):

    # Convert request to dictionary
    input_data = data.dict()

    # Engineer features
    df = engineer_features(input_data)

    # Select model features
    X = df[FEATURES]

    # Predict concentration
    prediction = xgboost_model.predict(X)[0]

    return {
        "predicted_concentration":
        round(float(prediction), 4)
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
