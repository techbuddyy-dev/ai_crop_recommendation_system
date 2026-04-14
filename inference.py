from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ARTIFACT_PATH = Path(__file__).parent / "model.pkl.gz"

FEATURE_COLUMNS = [
    "soil_ph",
    "temp",
    "relative_humidity",
    "n",
    "p",
    "k",
    "npk_total",
    "n_to_p_ratio",
    "n_to_k_ratio",
    "p_to_k_ratio",
    "temp_humidity_interaction",
    "ph_temp_interaction",
    "temp_squared",
    "ph_squared",
    "humidity_squared",
    "temp_ph_ratio",
    "humidity_temp_ratio",
    "temp_bin",
    "ph_bin",
    "humidity_bin",
]

TEMP_BIN_EDGES = [4.958, 12.0, 19.0, 26.0, 33.0, 40.0, 47.0]
PH_BIN_EDGES = [4.9965, 5.583333333333333, 6.166666666666667, 6.75, 7.333333333333334, 7.916666666666667, 8.5]
HUMIDITY_BIN_EDGES = [14.915, 29.166666666666664, 43.33333333333333, 57.5, 71.66666666666666, 85.83333333333333, 100.0]


def load_artifact(artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    path_str = str(artifact_path)
    if path_str.endswith(".gz"):
        import gzip
        with gzip.open(path_str, "rb") as file:
            artifact = pickle.load(file)
    else:
        with open(Path(artifact_path), "rb") as file:
            artifact = pickle.load(file)
    return artifact


def build_features(
    *,
    soil_ph: float,
    temp: float,
    relative_humidity: float,
    n: float,
    p: float,
    k: float,
    bin_edges: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    edges = bin_edges or {
        "temp": TEMP_BIN_EDGES,
        "soil_ph": PH_BIN_EDGES,
        "relative_humidity": HUMIDITY_BIN_EDGES,
    }

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

    frame["npk_total"] = frame["n"] + frame["p"] + frame["k"]
    frame["n_to_p_ratio"] = frame["n"] / (frame["p"] + 1)
    frame["n_to_k_ratio"] = frame["n"] / (frame["k"] + 1)
    frame["p_to_k_ratio"] = frame["p"] / (frame["k"] + 1)
    frame["temp_humidity_interaction"] = frame["temp"] * frame["relative_humidity"]
    frame["ph_temp_interaction"] = frame["soil_ph"] * frame["temp"]
    frame["temp_squared"] = frame["temp"] ** 2
    frame["ph_squared"] = frame["soil_ph"] ** 2
    frame["humidity_squared"] = frame["relative_humidity"] ** 2
    frame["temp_ph_ratio"] = frame["temp"] / (frame["soil_ph"] + 1)
    frame["humidity_temp_ratio"] = frame["relative_humidity"] / (frame["temp"] + 1)

    frame["temp_bin"] = pd.cut(frame["temp"], bins=edges["temp"], labels=False, include_lowest=True)
    frame["ph_bin"] = pd.cut(frame["soil_ph"], bins=edges["soil_ph"], labels=False, include_lowest=True)
    frame["humidity_bin"] = pd.cut(
        frame["relative_humidity"],
        bins=edges["relative_humidity"],
        labels=False,
        include_lowest=True,
    )

    return frame[FEATURE_COLUMNS]


def predict_crop(
    *,
    soil_ph: float,
    temp: float,
    relative_humidity: float,
    n: float,
    p: float,
    k: float,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    top_k: int = 3,
) -> dict[str, Any]:
    artifact = load_artifact(artifact_path)
    features = build_features(
        soil_ph=soil_ph,
        temp=temp,
        relative_humidity=relative_humidity,
        n=n,
        p=p,
        k=k,
        bin_edges=artifact["bin_edges"],
    )

    probabilities = artifact["model"].predict_proba(features)[0]
    prediction_index = int(probabilities.argmax())
    prediction = artifact["label_encoder"].inverse_transform([prediction_index])[0]
    confidence = float(probabilities[prediction_index])

    ranked = sorted(
        zip(artifact["label_encoder"].classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    top_predictions = [{"crop": crop, "probability": float(probability)} for crop, probability in ranked]

    return {
        "model_name": artifact["model_name"],
        "artifact_path": str(Path(artifact_path)),
        "prediction": prediction,
        "confidence": confidence,
        "top_3_predictions": top_predictions,
    }


def prompt_for_float(label: str) -> float:
    while True:
        raw_value = input(f"Enter {label}: ").strip()
        try:
            return float(raw_value)
        except ValueError:
            print(f"Invalid value for {label}. Please enter a number.")


if __name__ == "__main__":
    user_inputs = {
        "soil_ph": prompt_for_float("soil_ph"),
        "temp": prompt_for_float("temp"),
        "relative_humidity": prompt_for_float("relative_humidity"),
        "n": prompt_for_float("n"),
        "p": prompt_for_float("p"),
        "k": prompt_for_float("k"),
    }

    result = predict_crop(**user_inputs)

    print()
    print(f"Model: {result['model_name']}")
    print(f"Artifact: {result['artifact_path']}")
    print(f"Best prediction: {result['prediction']} ({result['confidence']:.6f})")
    print("Top 3 predictions:")
    for index, item in enumerate(result["top_3_predictions"], start=1):
        print(f"{index}. {item['crop']} - {item['probability']:.6f}")
