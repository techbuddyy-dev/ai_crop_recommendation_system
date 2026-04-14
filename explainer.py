"""
explainer.py — SHAP-Powered Explainability Engine
==================================================
Uses SHAP TreeExplainer to compute local feature contributions
for each prediction, then translates them into concise,
human-readable explanations.

This is the core of the Explainable AI (XAI) layer.
"""

import numpy as np
import pandas as pd
import shap

from config import FEATURE_DISPLAY_NAMES, FEATURE_COLUMNS
from model_loader import get_model


class ExplainabilityEngine:
    """
    Wraps SHAP TreeExplainer around the loaded Random Forest and
    provides human-readable explanations for crop predictions.
    """

    def __init__(self):
        self._model = get_model()
        # TreeExplainer is optimised for tree-based models;
        # it computes exact SHAP values in polynomial time.
        self._explainer = shap.TreeExplainer(self._model)

    def get_shap_values(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for a single-row feature DataFrame.

        Returns
        -------
        np.ndarray
            Shape (n_classes, n_features) — SHAP contribution of each
            feature towards each class prediction.
        """
        sv = self._explainer.shap_values(features_df)
        # shap_values returns list[ndarray] for multi-class:
        #   len = n_classes, each element shape (1, n_features)
        # Stack into (n_classes, n_features) for easy indexing.
        if isinstance(sv, list):
            return np.array([s[0] for s in sv])  # (n_classes, n_features)
        # Newer SHAP versions may return (1, n_features, n_classes)
        if sv.ndim == 3:
            return sv[0].T  # (n_classes, n_features)
        return sv

    def explain_prediction(
        self,
        features_df: pd.DataFrame,
        class_index: int,
        crop_name: str,
        confidence: float,
        raw_inputs: dict,
        top_n_features: int = 2,
    ) -> str:
        """
        Generate a human-readable explanation for why a specific crop
        was predicted, based on SHAP local feature contributions.

        Parameters
        ----------
        features_df : pd.DataFrame
            The full 20-feature engineered row.
        class_index : int
            The integer class index for which to explain.
        crop_name : str
            Display name of the crop.
        confidence : float
            Model confidence probability.
        raw_inputs : dict
            Original user inputs for value display.
        top_n_features : int
            Number of top contributing features to mention.

        Returns
        -------
        str
            A concise, professional explanation string.
        """
        shap_matrix = self.get_shap_values(features_df)
        # shap_matrix[class_index] → shape (n_features,)
        contributions = shap_matrix[class_index]

        # Pair feature names with their SHAP contribution
        feature_contribs = list(zip(FEATURE_COLUMNS, contributions))

        # Sort by absolute contribution (most impactful first)
        feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

        # Take top N positive contributors
        top_positive = [
            (fname, val) for fname, val in feature_contribs if val > 0
        ][:top_n_features]

        # Take top negative contributor (if any significant one exists)
        top_negative = [
            (fname, val) for fname, val in feature_contribs if val < 0
        ][:1]

        # Build the explanation
        explanation_parts = []

        if top_positive:
            drivers = []
            for fname, _ in top_positive:
                display = FEATURE_DISPLAY_NAMES.get(fname, fname)
                value = self._get_feature_value_str(fname, raw_inputs)
                if value:
                    drivers.append(f"{display} ({value})")
                else:
                    drivers.append(display)

            if confidence >= 0.5:
                explanation_parts.append(
                    f"{crop_name} is strongly recommended due to favorable "
                    f"{' and '.join(drivers)}."
                )
            elif confidence >= 0.1:
                explanation_parts.append(
                    f"{crop_name} is a good alternative, supported by "
                    f"{' and '.join(drivers)}."
                )
            else:
                explanation_parts.append(
                    f"{crop_name} is a potential match based on "
                    f"{' and '.join(drivers)}."
                )

        if top_negative and confidence < 0.5:
            neg_name = FEATURE_DISPLAY_NAMES.get(
                top_negative[0][0], top_negative[0][0]
            )
            explanation_parts.append(
                f"However, {neg_name} is slightly sub-optimal for this crop."
            )

        return " ".join(explanation_parts) if explanation_parts else (
            f"{crop_name} matched based on the overall input profile."
        )

    @staticmethod
    def _get_feature_value_str(feature_name: str, raw_inputs: dict) -> str:
        """
        Return a human-friendly value string for a feature,
        using the original raw inputs where possible.
        """
        units = {
            "soil_ph": "",
            "temp": "°C",
            "relative_humidity": "%",
            "n": " kg/ha",
            "p": " kg/ha",
            "k": " kg/ha",
        }
        if feature_name in raw_inputs and feature_name in units:
            val = raw_inputs[feature_name]
            unit = units[feature_name]
            return f"{val}{unit}"
        return ""
