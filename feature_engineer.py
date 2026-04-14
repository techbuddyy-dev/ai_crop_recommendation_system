"""
feature_engineer.py — Dynamic Feature Engineering
===================================================
Replicates the EXACT feature transformations that were applied during
model training. Accepts the 6 raw inputs and produces the full 20-column
feature vector expected by the Random Forest.

IMPORTANT: All ratio denominators use (+1) smoothing and binning uses
pd.cut — matching the original training pipeline verbatim.
"""

import pandas as pd

from config import FEATURE_COLUMNS, BIN_EDGES


class FeatureEngineer:
    """
    Stateless transformer that converts 6 raw environmental features
    into the 20-column engineered feature vector.
    """

    def __init__(self, bin_edges: dict | None = None):
        """
        Parameters
        ----------
        bin_edges : dict, optional
            Override bin edges (useful if loading from model artifact at runtime).
            Falls back to config.BIN_EDGES if not provided.
        """
        self.bin_edges = bin_edges or BIN_EDGES

    def transform(self, soil_ph: float, temp: float, relative_humidity: float,
                  n: float, p: float, k: float) -> pd.DataFrame:
        """
        Build the full 20-feature DataFrame from raw inputs.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns ordered exactly as FEATURE_COLUMNS.
        """
        # ── 1. Raw features ────────────────────────────────────────────
        frame = pd.DataFrame(
            [
                {
                    "soil_ph": float(soil_ph),
                    "temp": float(temp),
                    "relative_humidity": float(relative_humidity),
                    "n": float(n),
                    "p": float(p),
                    "k": float(k),
                }
            ]
        )

        # ── 2. Aggregate ───────────────────────────────────────────────
        frame["npk_total"] = frame["n"] + frame["p"] + frame["k"]

        # ── 3. Ratios (+1 smoothing to avoid division by zero) ─────────
        frame["n_to_p_ratio"] = frame["n"] / (frame["p"] + 1)
        frame["n_to_k_ratio"] = frame["n"] / (frame["k"] + 1)
        frame["p_to_k_ratio"] = frame["p"] / (frame["k"] + 1)
        frame["temp_ph_ratio"] = frame["temp"] / (frame["soil_ph"] + 1)
        frame["humidity_temp_ratio"] = frame["relative_humidity"] / (frame["temp"] + 1)

        # ── 4. Interaction terms ───────────────────────────────────────
        frame["temp_humidity_interaction"] = frame["temp"] * frame["relative_humidity"]
        frame["ph_temp_interaction"] = frame["soil_ph"] * frame["temp"]

        # ── 5. Polynomial (squared) terms ──────────────────────────────
        frame["temp_squared"] = frame["temp"] ** 2
        frame["ph_squared"] = frame["soil_ph"] ** 2
        frame["humidity_squared"] = frame["relative_humidity"] ** 2

        # ── 6. Binning (pd.cut against stored edges) ──────────────────
        frame["temp_bin"] = pd.cut(
            frame["temp"], bins=self.bin_edges["temp"],
            labels=False, include_lowest=True,
        )
        frame["ph_bin"] = pd.cut(
            frame["soil_ph"], bins=self.bin_edges["soil_ph"],
            labels=False, include_lowest=True,
        )
        frame["humidity_bin"] = pd.cut(
            frame["relative_humidity"], bins=self.bin_edges["relative_humidity"],
            labels=False, include_lowest=True,
        )

        # ── 7. Return in exact column order ────────────────────────────
        return frame[FEATURE_COLUMNS]
