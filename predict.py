"""
predict.py — CLI Entry Point for Crop Recommendation
======================================================
Demonstrates the full pipeline with sample inputs and pretty-printed
JSON output. Also supports custom inputs via command-line arguments.

Usage
-----
    # Default demo (rice-friendly conditions)
    python predict.py

    # Custom inputs
    python predict.py --n 90 --p 42 --k 43 --ph 6.5 --temp 20.8 --humidity 82.0
"""

import argparse
import json
import sys
import time

from pipeline import CropRecommendationPipeline


def main():
    # Reconfigure stdout to UTF-8 to safely handle special characters on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Crop Recommendation ML Pipeline - Top-3 Predictions with XAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py
  python predict.py --n 90 --p 42 --k 43 --ph 6.5 --temp 20.8 --humidity 82.0
  python predict.py --n 20 --p 60 --k 20 --ph 7.0 --temp 30.0 --humidity 50.0
        """,
    )
    parser.add_argument("--n", type=float, default=None, help="Nitrogen (kg/ha)")
    parser.add_argument("--p", type=float, default=None, help="Phosphorus (kg/ha)")
    parser.add_argument("--k", type=float, default=None, help="Potassium (kg/ha)")
    parser.add_argument("--ph", type=float, default=None, help="Soil pH (0-14)")
    parser.add_argument("--temp", type=float, default=None, help="Temperature (C)")
    parser.add_argument("--humidity", type=float, default=None, help="Relative Humidity (0-100)")
    args = parser.parse_args()

    # -- Demo scenarios --
    demo_scenarios = [
        {
            "label": "[1] Rice-Friendly Conditions",
            "params": {"n": 80, "p": 48, "k": 40, "soil_ph": 6.5, "temp": 23.0, "relative_humidity": 82.0},
        },
        {
            "label": "[2] Cotton Belt Profile",
            "params": {"n": 120, "p": 60, "k": 60, "soil_ph": 7.5, "temp": 32.0, "relative_humidity": 65.0},
        },
        {
            "label": "[3] Wheat Region Conditions",
            "params": {"n": 60, "p": 35, "k": 30, "soil_ph": 7.0, "temp": 18.0, "relative_humidity": 55.0},
        },
    ]

    # -- Initialise pipeline --
    print("=" * 70)
    print("  CROP RECOMMENDATION ML PIPELINE")
    print("  Powered by Random Forest + SHAP Explainable AI")
    print("=" * 70)

    print("\n>> Loading model (this may take a moment on first run)...")
    t0 = time.time()
    pipe = CropRecommendationPipeline()
    load_time = time.time() - t0
    print(f">> Model loaded in {load_time:.2f}s\n")

    # -- Custom input mode --
    if args.n is not None:
        custom = {
            "n": args.n,
            "p": args.p,
            "k": args.k,
            "soil_ph": args.ph,
            "temp": args.temp,
            "relative_humidity": args.humidity,
        }
        # Validate all args provided
        missing = [k for k, v in custom.items() if v is None]
        if missing:
            print(f"[ERROR] Missing required arguments: {', '.join(missing)}")
            sys.exit(1)

        print(">> Custom Input Prediction")
        print("-" * 40)
        _run_prediction(pipe, custom)
        return

    # -- Demo mode --
    for scenario in demo_scenarios:
        print(f"\n{scenario['label']}")
        print("-" * 40)
        _run_prediction(pipe, scenario["params"])


def _run_prediction(pipe: CropRecommendationPipeline, params: dict):
    """Run a single prediction and pretty-print results."""
    t0 = time.time()
    result = pipe.predict(**params)
    infer_time = time.time() - t0

    output = result.model_dump()
    print(json.dumps(output, indent=2))
    print(f"\n>> Inference time: {infer_time:.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
