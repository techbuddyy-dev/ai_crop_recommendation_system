"""
model_loader.py — Thread-Safe Singleton Model Loader
=====================================================
Loads the ~160 MB Random Forest pickle exactly once and caches it
for the lifetime of the process. All downstream consumers import
`get_model_artifact()` to obtain the dict without redundant I/O.
"""

import pickle
import warnings
import threading
from pathlib import Path
from typing import Any, Dict

from config import MODEL_PATH

_lock = threading.Lock()
_model_cache: Dict[str, Any] | None = None


def get_model_artifact(path: Path | None = None) -> Dict[str, Any]:
    """
    Return the full model dictionary from the pickle file.

    Keys available:
        model_name, model, label_encoder, feature_columns,
        selected_crops, bin_edges, test_accuracy, benchmark_scores,
        top_confusions, feature_importances

    Parameters
    ----------
    path : Path, optional
        Override the default MODEL_PATH (useful for testing).

    Returns
    -------
    dict
        The complete model artifact dictionary.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    with _lock:
        # Double-checked locking pattern
        if _model_cache is not None:
            return _model_cache

        resolved = path or MODEL_PATH
        if not resolved.exists():
            raise FileNotFoundError(
                f"Model file not found at: {resolved}\n"
                "Ensure 'best trained crop recommendation model.pkl' "
                "is in the project root."
            )

        # Suppress sklearn version mismatch warnings (trained on 1.7.2)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(resolved, "rb") as f:
                _model_cache = pickle.load(f)

    return _model_cache


def get_model():
    """Convenience: return just the RandomForestClassifier."""
    return get_model_artifact()["model"]


def get_label_encoder():
    """Convenience: return just the LabelEncoder."""
    return get_model_artifact()["label_encoder"]


def get_crop_names():
    """Convenience: return the ordered list of crop class names."""
    return get_model_artifact()["label_encoder"].classes_.tolist()


def get_model_metadata() -> dict:
    """Return non-model metadata (name, accuracy, benchmarks)."""
    art = get_model_artifact()
    return {
        "model_name": art["model_name"],
        "test_accuracy": art["test_accuracy"],
        "n_estimators": art["model"].n_estimators,
        "n_classes": len(art["label_encoder"].classes_),
        "n_features": len(art["feature_columns"]),
        "benchmark_scores": art.get("benchmark_scores", {}),
    }
