# ============================================
# schemas.py
# ============================================

"""
Defines API request/response schemas.

Schemas:
- validate incoming data
- enforce structure
- improve API reliability
- auto-generate Swagger docs

FastAPI + Pydantic handle validation automatically.
"""

from pydantic import BaseModel, Field


# ============================================
# PK EVENT INPUT
# ============================================

class PKEvent(BaseModel):

    """
    Represents a single infusion event
    for a patient at a specific timepoint.
    """

    # Unique patient/session identifier
    patient_id: str = Field(
        ...,
        example="P001"
    )

    # Time since infusion started
    Time: float = Field(
        ...,
        example=15.0
    )

    # Current infusion rate
    Rate: float = Field(
        ...,
        example=80.0
    )

    # Drug amount delivered
    # during current interval
    Amt: float = Field(
        ...,
        example=400.0
    )

    # Patient covariates
    Age: float = Field(
        ...,
        example=45
    )

    # 1 = Male
    # 0 = Female
    Sex: str = Field(
        ...,
        example="Male"
    )

    # Height (cm)
    Ht: float = Field(
        ...,
        example=172
    )

    # Weight (kg)
    Wt: float = Field(
        ...,
        example=75
    )


# ============================================
# PREDICTION RESPONSE
# ============================================

class PredictionResponse(BaseModel):

    patient_id: str

    predicted_concentration: float

    cumulative_dose: float

    infusion_active: bool
