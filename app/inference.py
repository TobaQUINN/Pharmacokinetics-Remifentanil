# ============================================
# inference.py
# ============================================

"""
Handles:
- model loading
- feature engineering
- prediction logic

Separating inference from API routes
makes the system:
- cleaner
- scalable
- maintainable
"""

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from state_manager import (
    get_patient_state,
    update_patient_state
)


# ============================================
# FEATURE ORDER
# ============================================

FEATURES = [

    'Time',
    'TimeDelta',

    'Rate',
    'Amt',

    'CumulativeDose',

    'Age',
    'Sex',

    'Ht',
    'Wt',

    'BSA',
    'LBM',

    'InfusionActive'
]


# ============================================
# PK INFERENCE ENGINE
# ============================================

class PKInferenceEngine:

    def __init__(self, model_path):
        """
        Load trained XGBoost model.
        """

        self.xgboost_model = XGBRegressor()

        self.xgboost_model.load_model(model_path)

    # ========================================
    # BODY SURFACE AREA
    # ========================================

    def compute_bsa(self, height, weight):
        """
        Mosteller BSA Formula
        """

        return np.sqrt(
            (height * weight) / 3600
        )

    # ========================================
    # LEAN BODY MASS
    # ========================================

    def compute_lbm(
        self,
        sex,
        weight,
        height
    ):
        """
        Lean Body Mass estimation.
        """

        if sex == 1:

            return (
                1.1 * weight
                - 128 * (weight / height) ** 2
            )

        else:

            return (
                1.07 * weight
                - 148 * (weight / height) ** 2
            )

    def compute_bmi(self, Ht, Wt):

        height_m = Ht / 100

        bmi = Wt / (height_m ** 2)

        return bmi

    # ========================================
    # ENCODING SEX
    # ========================================
    def encode_sex(self, sex):
        """
        Encode biological sex
        for model compatibility.
        """

        sex = sex.lower()

        if sex == "male":

            return 1

        elif sex == "female":

            return 0

        else:

            raise ValueError(
                "Sex must be Male or Female"
            )

    # ========================================
    # FEATURE ENGINEERING
    # ========================================

    def engineer_features(self, event):
        """
        Converts raw event data
        into model-ready PK features.
        """

        patient_id = event.patient_id

        # Retrieve stored patient state
        state = get_patient_state(patient_id)

        # ================================
        # TEMPORAL FEATURES
        # ================================

        delta_time = (
            event.Time - state["last_time"]
        )

        delta_time = max(delta_time, 0)

        # ================================
        # CUMULATIVE DOSE
        # ================================

        updated_cum_dose = (
            state["cum_dose"] + event.Amt
        )
        # ================================
        # BODY MASS INDEX (BMI)
        # ================================

        bmi = self.compute_bmi(
            event.Ht,
            event.Wt
        )

        # ================================
        # ENCODE SEX
        # ================================
        sex_encoded = self.encode_sex(
            event.Sex
        )

        # ================================
        # INFUSION STATUS
        # ================================

        infusion_active = int(
            event.Rate > 0
        )

        # ================================
        # BODY SURFACE AREA
        # ================================

        bsa = self.compute_bsa(
            event.Ht,
            event.Wt
        )

        # ================================
        # LEAN BODY MASS
        # ================================

        lbm = self.compute_lbm(
            event.Sex,
            event.Wt,
            event.Ht
        )

        # ================================
        # MODEL INPUT
        # ================================

        row = {

            'Time': event.Time,

            'TimeDelta': delta_time,

            'Rate': event.Rate,

            'Amt': event.Amt,

            'CumulativeDose': updated_cum_dose,

            'Age': event.Age,

            'Sex_Encoded': sex_encoded,

            'Ht': event.Ht,

            'Wt': event.Wt,

            'BSA': bsa,

            'LBM': lbm,

            'InfusionActive': infusion_active,

            'BMI': bmi
        }

        # ================================
        # UPDATE PATIENT STATE
        # ================================

        update_patient_state(

            patient_id=patient_id,

            last_time=event.Time,

            cum_dose=updated_cum_dose,

            last_rate=event.Rate
        )

        return pd.DataFrame([row])

    # ========================================
    # PREDICTION
    # ========================================

    def predict(self, event):
        """
        Full inference pipeline.
        """

        # Engineer features
        df = self.engineer_features(event)

        # Select model features
        X = df[FEATURES]

        # Predict concentration
        prediction = self.xgboost_model.predict(X)[0]

        return {

            "prediction": float(prediction),

            "cum_dose":
            float(df['CumulativeDose'].iloc[0]),

            "infusion_active":
            bool(df['InfusionActive'].iloc[0])
        }
