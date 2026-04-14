"""
config.py — Pipeline Constants & Human-Readable Feature Mappings
================================================================
Central configuration for the Crop Recommendation ML Pipeline.
All feature names, bin edges, and display mappings are derived directly
from the trained model artifact to ensure 1:1 consistency.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best trained crop recommendation model.pkl"

# ---------------------------------------------------------------------------
# Feature column ordering (must match model training exactly)
# ---------------------------------------------------------------------------
RAW_FEATURES = ["soil_ph", "temp", "relative_humidity", "n", "p", "k"]

FEATURE_COLUMNS = [
    "soil_ph", "temp", "relative_humidity",
    "n", "p", "k",
    "npk_total",
    "n_to_p_ratio", "n_to_k_ratio", "p_to_k_ratio",
    "temp_humidity_interaction", "ph_temp_interaction",
    "temp_squared", "ph_squared", "humidity_squared",
    "temp_ph_ratio", "humidity_temp_ratio",
    "temp_bin", "ph_bin", "humidity_bin",
]

# ---------------------------------------------------------------------------
# Bin edges (copied verbatim from model_dict['bin_edges'])
# ---------------------------------------------------------------------------
BIN_EDGES = {
    "temp": [4.958, 12.0, 19.0, 26.0, 33.0, 40.0, 47.0],
    "soil_ph": [
        4.9965, 5.583333333333333, 6.166666666666667,
        6.75, 7.333333333333334, 7.916666666666667, 8.5,
    ],
    "relative_humidity": [
        14.915, 29.166666666666664, 43.33333333333333,
        57.5, 71.66666666666666, 85.83333333333333, 100.0,
    ],
}

# ---------------------------------------------------------------------------
# Human-readable feature name mapping (for explanation generation)
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    "soil_ph":                    "Soil pH",
    "temp":                       "Temperature",
    "relative_humidity":          "Relative Humidity",
    "n":                          "Nitrogen (N)",
    "p":                          "Phosphorus (P)",
    "k":                          "Potassium (K)",
    "npk_total":                  "Total NPK",
    "n_to_p_ratio":               "Nitrogen-to-Phosphorus Ratio",
    "n_to_k_ratio":               "Nitrogen-to-Potassium Ratio",
    "p_to_k_ratio":               "Phosphorus-to-Potassium Ratio",
    "temp_humidity_interaction":   "Temperature × Humidity Interaction",
    "ph_temp_interaction":        "pH × Temperature Interaction",
    "temp_squared":               "Temperature²",
    "ph_squared":                 "Soil pH²",
    "humidity_squared":           "Humidity²",
    "temp_ph_ratio":              "Temperature-to-pH Ratio",
    "humidity_temp_ratio":        "Humidity-to-Temperature Ratio",
    "temp_bin":                   "Temperature Range Category",
    "ph_bin":                     "pH Range Category",
    "humidity_bin":               "Humidity Range Category",
}

# ---------------------------------------------------------------------------
# Input validation bounds
# ---------------------------------------------------------------------------
INPUT_BOUNDS = {
    "n":                 (0.0, 300.0),
    "p":                 (0.0, 300.0),
    "k":                 (0.0, 300.0),
    "soil_ph":           (0.0, 14.0),
    "temp":              (-10.0, 60.0),
    "relative_humidity": (0.0, 100.0),
}

# ---------------------------------------------------------------------------
# Top-K predictions to return
# ---------------------------------------------------------------------------
TOP_K = 3
