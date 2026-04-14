import os
import concurrent.futures
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from inference import predict_crop
from rag.graph import run_pipeline

# Model path
MODEL_PATH = Path(__file__).parent / "model.pkl.gz"

app = FastAPI(
    title="Agronomist AI Server",
    description="Single API endpoint for predicting crops and returning RAG evaluations for the top 3 recommendations.",
    version="1.0"
)

class CropPredictionRequest(BaseModel):
    n: float
    p: float
    k: float
    soil_ph: float
    temp: float
    humidity: float

class CropEvaluation(BaseModel):
    crop: str
    probability: float
    rag_evaluation: str | None = None

class CropPredictionResponse(BaseModel):
    top_prediction: str
    confidence: float
    top_3_predictions: list[CropEvaluation]

def evaluate_single_crop(crop_name: str, prob: float, req: CropPredictionRequest) -> CropEvaluation:
    """Helper to run the LangGraph RAG pipeline for a single crop."""
    try:
        rag_result = run_pipeline(
            n=req.n, p=req.p, k=req.k,
            soil_ph=req.soil_ph, temp=req.temp, humidity=req.humidity,
            target_crop=crop_name, target_prob=prob
        )
        return CropEvaluation(
            crop=crop_name, 
            probability=prob, 
            rag_evaluation=rag_result["explanation"]
        )
    except Exception as e:
        print(f"Error evaluating {crop_name}: {e}")
        return CropEvaluation(
            crop=crop_name, 
            probability=prob, 
            rag_evaluation=f"Failed to generate evaluation: {e}"
        )

@app.post("/predict", response_model=CropPredictionResponse)
def predict_and_evaluate(req: CropPredictionRequest):
    """
    Takes 6 inputs, predicts the top 3 crops using the ML model, 
    and runs the RAG pipeline in parallel to evaluate each crop.
    """
    # 1. Run the base ML model prediction
    result = predict_crop(
        soil_ph=req.soil_ph, temp=req.temp, relative_humidity=req.humidity,
        n=req.n, p=req.p, k=req.k, artifact_path=MODEL_PATH
    )
    
    top_3_raw = result["top_3_predictions"]
    
    # 2. Check for API key before running RAG
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=500, 
            detail="GOOGLE_API_KEY environment variable is missing. It must be set before calling this endpoint."
        )

    # 3. Run all 3 RAG evaluations concurrently (cuts response time from ~10s to ~3s)
    top_3_evaluated = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(evaluate_single_crop, item["crop"], item["probability"], req)
            for item in top_3_raw
        ]
        
        for future in concurrent.futures.as_completed(futures):
            top_3_evaluated.append(future.result())
            
    # Sort them back by probability descending since as_completed returns them randomly
    top_3_evaluated.sort(key=lambda x: x.probability, reverse=True)

    return CropPredictionResponse(
        top_prediction=result["prediction"],
        confidence=result["confidence"],
        top_3_predictions=top_3_evaluated
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Starting the Single-Endpoint Agronomist API Server...")
    print("=" * 60)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
