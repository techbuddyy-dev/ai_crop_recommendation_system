"""
tests/test_pipeline.py — Unit Tests for the Crop Recommendation Pipeline
=========================================================================
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FEATURE_COLUMNS, BIN_EDGES, INPUT_BOUNDS
from schemas import CropInput, PredictionResponse
from feature_engineer import FeatureEngineer
from model_loader import get_model_artifact, get_crop_names
from pipeline import CropRecommendationPipeline


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """Load the pipeline once for all tests in this module."""
    return CropRecommendationPipeline()


@pytest.fixture
def sample_inputs():
    return {"n": 80, "p": 48, "k": 40, "soil_ph": 6.5, "temp": 23.0, "relative_humidity": 82.0}


# ── Schema Validation Tests ─────────────────────────────────────────────

class TestSchemaValidation:
    def test_valid_input(self, sample_inputs):
        inp = CropInput(**sample_inputs)
        assert inp.n == 80
        assert inp.soil_ph == 6.5

    def test_ph_out_of_bounds(self):
        with pytest.raises(ValueError, match="soil_ph"):
            CropInput(n=80, p=48, k=40, soil_ph=15.0, temp=23.0, relative_humidity=82.0)

    def test_negative_nitrogen(self):
        with pytest.raises(ValueError, match="n"):
            CropInput(n=-5, p=48, k=40, soil_ph=6.5, temp=23.0, relative_humidity=82.0)

    def test_humidity_over_100(self):
        with pytest.raises(ValueError, match="relative_humidity"):
            CropInput(n=80, p=48, k=40, soil_ph=6.5, temp=23.0, relative_humidity=110.0)


# ── Feature Engineering Tests ─────────────────────────────────────────

class TestFeatureEngineering:
    def test_output_columns(self, sample_inputs):
        fe = FeatureEngineer()
        df = fe.transform(**sample_inputs)
        assert list(df.columns) == FEATURE_COLUMNS, \
            "Feature columns must exactly match model training order"

    def test_output_shape(self, sample_inputs):
        fe = FeatureEngineer()
        df = fe.transform(**sample_inputs)
        assert df.shape == (1, 20), "Must produce exactly 1 row × 20 columns"

    def test_npk_total(self, sample_inputs):
        fe = FeatureEngineer()
        df = fe.transform(**sample_inputs)
        assert df["npk_total"].values[0] == 80 + 48 + 40

    def test_ratios_plus_one_smoothing(self):
        fe = FeatureEngineer()
        # With +1 smoothing: n/(k+1) = 80/(0+1) = 80.0
        df = fe.transform(soil_ph=6.5, temp=23.0, relative_humidity=82.0, n=80, p=48, k=0)
        assert df["n_to_k_ratio"].values[0] == 80.0  # n / (k + 1)
        assert df["p_to_k_ratio"].values[0] == 48.0  # p / (k + 1)

    def test_binning_produces_integers(self, sample_inputs):
        fe = FeatureEngineer()
        df = fe.transform(**sample_inputs)
        for col in ["temp_bin", "ph_bin", "humidity_bin"]:
            assert np.issubdtype(type(df[col].values[0]), np.integer)


# ── Model Loading Tests ──────────────────────────────────────────────

class TestModelLoading:
    def test_artifact_keys(self):
        art = get_model_artifact()
        expected_keys = {
            "model_name", "model", "label_encoder", "feature_columns",
            "selected_crops", "bin_edges", "test_accuracy",
            "benchmark_scores", "top_confusions", "feature_importances",
        }
        assert expected_keys.issubset(set(art.keys()))

    def test_25_crop_classes(self):
        crops = get_crop_names()
        assert len(crops) == 25

    def test_feature_columns_match_config(self):
        art = get_model_artifact()
        assert art["feature_columns"] == FEATURE_COLUMNS


# ── End-to-End Pipeline Tests ─────────────────────────────────────────

class TestPipeline:
    def test_returns_3_recommendations(self, pipeline, sample_inputs):
        result = pipeline.predict(**sample_inputs)
        assert len(result.recommendations) == 3

    def test_response_structure(self, pipeline, sample_inputs):
        result = pipeline.predict(**sample_inputs)
        assert isinstance(result, PredictionResponse)
        assert result.model_name == "Best Random Forest Top 25 India Crops"
        assert 0 < result.model_accuracy <= 1.0

    def test_confidences_sum_to_reasonable(self, pipeline, sample_inputs):
        result = pipeline.predict(**sample_inputs)
        total = sum(r.confidence for r in result.recommendations)
        # Top-3 should capture a meaningful share of probability
        assert total > 0.1, "Top-3 confidences should sum to > 10%"

    def test_ranks_are_ordered(self, pipeline, sample_inputs):
        result = pipeline.predict(**sample_inputs)
        ranks = [r.rank for r in result.recommendations]
        assert ranks == [1, 2, 3]

    def test_confidence_descending(self, pipeline, sample_inputs):
        result = pipeline.predict(**sample_inputs)
        confs = [r.confidence for r in result.recommendations]
        assert confs == sorted(confs, reverse=True), \
            "Recommendations must be sorted by descending confidence"

    def test_explanations_not_empty(self, pipeline, sample_inputs):
        result = pipeline.predict(**sample_inputs)
        for rec in result.recommendations:
            assert len(rec.explanation) > 10, \
                f"Explanation for {rec.crop} is too short"

    def test_crop_names_are_valid(self, pipeline, sample_inputs):
        result = pipeline.predict(**sample_inputs)
        valid_crops = {c.title() for c in get_crop_names()}
        for rec in result.recommendations:
            assert rec.crop in valid_crops, f"{rec.crop} not in valid crop list"
