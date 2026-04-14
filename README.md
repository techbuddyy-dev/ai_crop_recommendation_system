# 🌱 AI Crop Recommendation System: The Digital Agronomist

Welcome to the **AI Crop Recommendation System**, a next-generation "Digital Agronomist" built to solve the black-box problem in traditional agricultural machine learning.

This platform bridges the gap between raw predictive data and actionable human advice. 

---

## 🚀 The "Triple-Threat" Architecture

Instead of just predicting a crop name with no explanation, this platform utilizes a 3-stage agentic pipeline:

1. **Deterministic AI (Random Forest)**: An optimized machine learning model processes 6 environmental factors (N, P, K, pH, Temp, Humidity) to securely predict the **Top 3** most viable crops.
2. **Explainable AI (SHAP)**: Feature drivers are reverse-engineered instantly. The system calculates *exactly* why the model selected the crop (e.g., "High Potassium was a driving factor").
3. **Agentic Generative AI (RAG + Gemini)**: We pipe the raw prediction and the SHAP mathematical drivers into a LangGraph workflow. Using a pre-embedded vector database (ChromaDB) containing agronomic rules for 25 different crops, **Google Gemini** synthesizes a professional, structured advisory report customized to the farmer's specific land profile.

## 🛠 Features Included

- **`app.py`**: A fully visual, interactive **Streamlit Dashboard** featuring real-time crop analysis and inline AI explanations.
- **`api.py`**: A cloud-ready, asynchronous **FastAPI Server** capable of taking remote POST requests and responding with the full RAG explanation in JSON format.
- **Zero-Latency Vector Storage**: Utilizing a bundled `chroma_db`, making cloud deployment (like Railway) lightning-fast and 100% free.

## 💻 Tech Stack
- **Models:** Random Forest, Google Gemini 1.5 Flash (Synthesis), Gemini Embedding 001 (Vectorization)
- **Frameworks:** LangChain, LangGraph, FastAPI, Streamlit
- **Data/Math:** Pandas, Numpy, SHAP, ChromaDB

---

## ⚙️ How to Deploy (Railway / Cloud)

This repository is optimized for **Zero-Config Deployment on Railway.app**.

1. Connect this repo to a new Railway project.
2. Railway will automatically find the included `Procfile` and use `requirements.txt` to install the backend.
3. In your Railway dashboard, add your API key heavily secured under the **Variables** tab:
   - Key: `GOOGLE_API_KEY`
   - Value: `Your_Paid_Google_Key`
4. The deployment will initialize locally embedded Vector stores and boot the FastAPI gateway instantly. *(Access `/docs` for testing).*

## 🧪 How to Run Locally

If you clone this repository to test locally:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Environment secret (Or create a .env file locally)
set GOOGLE_API_KEY=your_google_key_here

# 3a. Start the Visual Dashboard
streamlit run app.py

# 3b. OR Start the Swagger API Gateway
python api.py
```

### Note on Large Files
The trained Random Forest model has been statically gzipped (`model.pkl.gz`) resulting in a tiny 11 MB tracking footprint to respect GitHub constraints and speed up cloud Docker builds. The Python backend organically decompresses this at runtime.
