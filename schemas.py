"""
schemas.py — Pydantic Data Contracts
=====================================
Strict input validation and structured output models for the
Crop Recommendation ML Pipeline.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List

from config import INPUT_BOUNDS


class CropInput(BaseModel):
    """
    Raw environmental input from the user.
    Only the 6 base features are required — all derived features
    (ratios, interactions, bins) are computed by the FeatureEngineer.
    """
    n: float = Field(..., description="Nitrogen content in soil (kg/ha)")
    p: float = Field(..., description="Phosphorus content in soil (kg/ha)")
    k: float = Field(..., description="Potassium content in soil (kg/ha)")
    soil_ph: float = Field(..., description="Soil pH level (0-14)")
    temp: float = Field(..., description="Average temperature in °C")
    relative_humidity: float = Field(..., description="Relative humidity in %")

    @field_validator("n", "p", "k", "soil_ph", "temp", "relative_humidity")
    @classmethod
    def validate_bounds(cls, v, info):
        field_name = info.field_name
        lo, hi = INPUT_BOUNDS[field_name]
        if not (lo <= v <= hi):
            raise ValueError(
                f"{field_name} must be between {lo} and {hi}, got {v}"
            )
        return v


class CropPrediction(BaseModel):
    """A single crop recommendation with confidence and explanation."""
    rank: int = Field(..., description="Rank position (1 = best match)")
    crop: str = Field(..., description="Predicted crop name")
    confidence: float = Field(..., description="Model confidence (0.0 – 1.0)")
    confidence_pct: str = Field(..., description="Confidence as formatted %")
    explanation: str = Field(..., description="Human-readable reason for selection")


class PredictionResponse(BaseModel):
    """Top-K crop recommendations returned by the pipeline."""
    input_summary: dict = Field(..., description="Echo of the user-provided inputs")
    recommendations: List[CropPrediction] = Field(
        ..., description="Top-K ranked crop predictions"
    )
    model_name: str = Field(..., description="Name of the model that produced this")
    model_accuracy: float = Field(..., description="Test accuracy of the model")
