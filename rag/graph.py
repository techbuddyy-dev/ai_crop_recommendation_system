"""
rag/graph.py — LangGraph Agentic Workflow
==========================================
Orchestrates: ML Prediction → RAG Retrieval → LLM Synthesis
using a directed LangGraph StateGraph.
"""

import os
import sys
from pathlib import Path
from typing import Any, TypedDict
import threading

import numpy as np
import shap
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import predict_crop, load_artifact, build_features
from rag.prompts import AGRONOMIST_SYSTEM_PROMPT, SYNTHESIS_PROMPT_TEMPLATE

# ── Paths ───────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl.gz"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "crop_knowledge"

# Human-readable feature names
FEATURE_DISPLAY = {
    "soil_ph": "Soil pH", "temp": "Temperature", "relative_humidity": "Humidity",
    "n": "Nitrogen (N)", "p": "Phosphorus (P)", "k": "Potassium (K)",
    "npk_total": "Total NPK", "n_to_p_ratio": "N-to-P Ratio",
    "n_to_k_ratio": "N-to-K Ratio", "p_to_k_ratio": "P-to-K Ratio",
    "temp_humidity_interaction": "Temp×Humidity", "ph_temp_interaction": "pH×Temp",
    "temp_squared": "Temp²", "ph_squared": "pH²", "humidity_squared": "Humidity²",
    "temp_ph_ratio": "Temp/pH Ratio", "humidity_temp_ratio": "Humidity/Temp Ratio",
    "temp_bin": "Temp Range", "ph_bin": "pH Range", "humidity_bin": "Humidity Range",
}


# ── State Definition ────────────────────────────────────────────────────
class AgriState(TypedDict):
    # Inputs
    n: float
    p: float
    k: float
    soil_ph: float
    temp: float
    humidity: float
    target_crop: str
    target_prob: float
    # Intermediate
    shap_summary: str
    retrieved_context: str
    # Output
    explanation: str


# ── Cached Resources ────────────────────────────────────────────────────
_artifact = None
_explainer = None
_vectorstore = None
_llm = None
_init_lock = threading.Lock()


def _get_artifact():
    global _artifact
    if _artifact is None:
        with _init_lock:
            if _artifact is None:
                _artifact = load_artifact(MODEL_PATH)
    return _artifact


def _get_explainer():
    global _explainer
    if _explainer is None:
        with _init_lock:
            if _explainer is None:
                artifact = _get_artifact()
                _explainer = shap.TreeExplainer(artifact["model"])
    return _explainer


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        with _init_lock:
            if _vectorstore is None:
                api_key = os.environ.get("GOOGLE_API_KEY", "")
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001", google_api_key=api_key,
                )
                _vectorstore = Chroma(
                    persist_directory=str(CHROMA_DIR),
                    embedding_function=embeddings,
                    collection_name=COLLECTION_NAME,
                )
    return _vectorstore


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        with _init_lock:
            if _llm is None:
                api_key = os.environ.get("GOOGLE_API_KEY", "")
                _llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.3,
                )
    return _llm


# ── Node 1: ML Prediction + SHAP ───────────────────────────────────────
def predict_node(state: AgriState) -> dict:
    """Compute SHAP drivers for the target crop."""
    # 1. Compute SHAP for the requested crop
    artifact = _get_artifact()
    features_df = build_features(
        soil_ph=state["soil_ph"], temp=state["temp"],
        relative_humidity=state["humidity"],
        n=state["n"], p=state["p"], k=state["k"],
        bin_edges=artifact["bin_edges"],
    )

    explainer = _get_explainer()
    shap_values = explainer.shap_values(features_df)

    # Get SHAP for target crop
    crop_name = state["target_crop"]
    crop_classes = list(artifact["label_encoder"].classes_)
    class_idx = crop_classes.index(crop_name)

    if isinstance(shap_values, list):
        sv = shap_values[class_idx][0]
    else:
        sv = shap_values[0, :, class_idx]

    feature_names = list(features_df.columns)
    contribs = sorted(zip(feature_names, sv), key=lambda x: -abs(x[1]))

    # Build SHAP summary text
    lines = []
    for feat, val in contribs[:5]:
        direction = "POSITIVE" if val > 0 else "NEGATIVE"
        display = FEATURE_DISPLAY.get(feat, feat)
        lines.append(f"- {display}: {direction} impact (SHAP value: {val:+.4f})")

    shap_text = "\n".join(lines) if lines else "No significant drivers identified."

    return {"shap_summary": shap_text}


# ── Node 2: RAG Retrieval ──────────────────────────────────────────────
def retrieve_node(state: AgriState) -> dict:
    """Query ChromaDB for relevant crop knowledge, filtered by crop name."""
    crop_name = state["target_crop"]
    vectorstore = _get_vectorstore()

    # Build a rich query from the prediction context
    query = (
        f"{crop_name} soil pH {state['soil_ph']} temperature {state['temp']} "
        f"humidity {state['humidity']} nitrogen {state['n']} "
        f"phosphorus {state['p']} potassium {state['k']} "
        f"suitability nutrient climate requirements"
    )

    # Retrieve with metadata filter for this specific crop
    results = vectorstore.similarity_search(
        query, k=8,
        filter={"crop_name": crop_name.lower()},
    )

    # If no filtered results, try without filter
    if not results:
        results = vectorstore.similarity_search(query, k=6)

    context = "\n\n".join([
        f"[{r.metadata.get('section', 'general')}] {r.page_content}"
        for r in results
    ])

    return {"retrieved_context": context if context else "No relevant documents found."}


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def _call_llm(llm, messages):
    return llm.invoke(messages)


# ── Node 3: LLM Synthesis ──────────────────────────────────────────────
def synthesize_node(state: AgriState) -> dict:
    """Send prediction + SHAP + retrieved context to Gemini for synthesis."""

    user_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        n=state["n"], p=state["p"], k=state["k"],
        soil_ph=state["soil_ph"], temp=state["temp"], humidity=state["humidity"],
        crop_name=state["target_crop"],
        confidence=state["target_prob"],
        shap_summary=state["shap_summary"],
        context=state["retrieved_context"],
    )

    llm = _get_llm()
    response = _call_llm(llm, [
        SystemMessage(content=AGRONOMIST_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    return {"explanation": response.content}


# ── Build the Graph ─────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """Construct the LangGraph workflow."""
    graph = StateGraph(AgriState)

    graph.add_node("predict", predict_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("predict")
    graph.add_edge("predict", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ── Convenience Runner ──────────────────────────────────────────────────
def run_pipeline(
    n: float, p: float, k: float,
    soil_ph: float, temp: float, humidity: float,
    target_crop: str, target_prob: float,
) -> dict:
    """Run the full LangGraph pipeline end-to-end for a specific crop."""
    app = build_graph()
    result = app.invoke({
        "n": n, "p": p, "k": k,
        "soil_ph": soil_ph, "temp": temp, "humidity": humidity,
        "target_crop": target_crop, "target_prob": target_prob,
    })
    return result
