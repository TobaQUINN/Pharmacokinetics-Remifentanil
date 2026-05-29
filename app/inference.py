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
