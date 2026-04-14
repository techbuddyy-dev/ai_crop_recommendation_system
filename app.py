import streamlit as st
from pathlib import Path
from inference import predict_crop, load_artifact, build_features
import shap
import numpy as np
import os
import time

# Model path (local)
MODEL_PATH = Path(__file__).parent / "model.pkl.gz"

# Human-readable feature names for explanations
FEATURE_DISPLAY = {
    "soil_ph": "Soil pH", "temp": "Temperature", "relative_humidity": "Relative Humidity",
    "n": "Nitrogen (N)", "p": "Phosphorus (P)", "k": "Potassium (K)",
    "npk_total": "Total NPK", "n_to_p_ratio": "N-to-P Ratio",
    "n_to_k_ratio": "N-to-K Ratio", "p_to_k_ratio": "P-to-K Ratio",
    "temp_humidity_interaction": "Temp x Humidity", "ph_temp_interaction": "pH x Temp",
    "temp_squared": "Temperature Squared", "ph_squared": "pH Squared",
    "humidity_squared": "Humidity Squared", "temp_ph_ratio": "Temp-to-pH Ratio",
    "humidity_temp_ratio": "Humidity-to-Temp Ratio", "temp_bin": "Temp Range",
    "ph_bin": "pH Range", "humidity_bin": "Humidity Range",
}

st.set_page_config(page_title="Crop Predictor", page_icon="🌱", layout="centered")


@st.cache_resource
def get_artifact():
    return load_artifact(MODEL_PATH)


@st.cache_resource
def get_explainer():
    artifact = get_artifact()
    return shap.TreeExplainer(artifact["model"])


try:
    artifact = get_artifact()
    explainer = get_explainer()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

st.title("🌱 Crop Recommendation")
st.markdown("Enter the soil and environmental conditions below to get the top 3 crop recommendations with AI-powered explanations.")

if model_loaded:
    # Sidebar: API key + toggle
    with st.sidebar:
        st.header("⚙️ Settings")
        enable_rag = st.toggle("Enable AI Deep Explanation", value=False,
                               help="Uses RAG + Gemini to generate detailed agronomic explanations")
        if enable_rag:
            api_key = st.text_input("Google API Key", type="password",
                                    value=os.environ.get("GOOGLE_API_KEY", ""),
                                    help="Required for Gemini LLM and embeddings")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key

    with st.form("prediction_form"):
        st.subheader("Soil Nutrients")
        col1, col2, col3 = st.columns(3)
        with col1:
            n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=80.0, step=1.0)
        with col2:
            p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=300.0, value=48.0, step=1.0)
        with col3:
            k = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=40.0, step=1.0)

        st.subheader("Environmental Factors")
        col4, col5, col6 = st.columns(3)
        with col4:
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with col5:
            temp = st.number_input("Temperature (C)", min_value=-10.0, max_value=60.0, value=23.0, step=0.5)
        with col6:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.5)

        submitted = st.form_submit_button("Predict Best Crops", type="primary")

    if submitted:
        with st.spinner("Predicting..."):
            t0 = time.time()

            # 1. Run the original predict_crop (untouched)
            result = predict_crop(
                soil_ph=ph, temp=temp, relative_humidity=humidity,
                n=n, p=p, k=k, artifact_path=MODEL_PATH,
            )

            # 2. Build features using the original build_features (untouched)
            features_df = build_features(
                soil_ph=ph, temp=temp, relative_humidity=humidity,
                n=n, p=p, k=k, bin_edges=artifact["bin_edges"],
            )

            # 3. SHAP explanation
            shap_values = explainer.shap_values(features_df)
            crop_classes = artifact["label_encoder"].classes_
            feature_names = list(features_df.columns)

            infer_time = time.time() - t0

        st.success(f"Best match: **{result['prediction']}** (confidence: {result['confidence']:.4f}) | Time: {infer_time:.2f}s")

        st.subheader("Top 3 Predictions")
        for i, item in enumerate(result["top_3_predictions"], start=1):
            crop_name = item["crop"]
            prob = item["probability"]

            # Find class index for this crop
            class_idx = list(crop_classes).index(crop_name)

            # Get SHAP values for this crop class
            if isinstance(shap_values, list):
                sv = shap_values[class_idx][0]
            else:
                sv = shap_values[0, :, class_idx]

            # Find top 2 positive drivers and top 1 negative
            contribs = list(zip(feature_names, sv))
            positive = sorted([c for c in contribs if c[1] > 0], key=lambda x: -x[1])[:2]
            negative = sorted([c for c in contribs if c[1] < 0], key=lambda x: x[1])[:1]

            # Build explanation text
            if positive:
                drivers = " and ".join([FEATURE_DISPLAY.get(f, f) for f, _ in positive])
                if prob >= 0.5:
                    explanation = f"Strongly recommended due to favorable {drivers}."
                elif prob >= 0.05:
                    explanation = f"A good alternative, supported by {drivers}."
                else:
                    explanation = f"A potential match based on {drivers}."

                if negative and prob < 0.5:
                    neg_name = FEATURE_DISPLAY.get(negative[0][0], negative[0][0])
                    explanation += f" However, {neg_name} is slightly sub-optimal."
            else:
                explanation = f"Matched based on overall input profile."

            with st.expander(f"#{i} - **{crop_name}** (probability: {prob:.4f})", expanded=(i == 1)):
                st.progress(prob)
                st.info(f"**Why?** {explanation}")

                # ── RAG Deep Evaluation per Crop ─────────────────────────────────────
                if enable_rag:
                    api_key_set = os.environ.get("GOOGLE_API_KEY", "")
                    if not api_key_set:
                        st.warning("Please enter your Google API Key in the sidebar to enable AI explanations.")
                    else:
                        st.divider()
                        with st.spinner(f"Generating Agronomist evaluation for {crop_name}..."):
                            try:
                                from rag.graph import run_pipeline
                                rag_result = run_pipeline(
                                    n=n, p=p, k=k,
                                    soil_ph=ph, temp=temp, humidity=humidity,
                                    target_crop=crop_name, target_prob=prob
                                )
                                st.markdown("### 👨‍🌾 Agronomist Evaluation")
                                st.markdown(rag_result["explanation"])
                            except Exception as e:
                                st.error(f"RAG pipeline error: {e}")
                                st.info("Make sure you've run the ingestion script first: `python -m rag.ingest`")
