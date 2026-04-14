"""
rag/ingest.py — Document Ingestion Pipeline
=============================================
Loads crop JSON knowledge files, extracts meaningful text chunks
with metadata, embeds them via Google Generative AI, and stores
in a persistent ChromaDB vector store.

Run once:
    python -m rag.ingest
"""

import json
import os
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ── Config ──────────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "crop_knowledge"


def _flatten_profile(crop_id: str, crop_name: str, profile: dict) -> list[Document]:
    """Extract structured text chunks from master_profile section."""
    docs = []

    # Short description
    if desc := profile.get("short_description"):
        docs.append(Document(
            page_content=f"{crop_name}: {desc}",
            metadata={"crop_name": crop_id, "section": "overview"},
        ))

    # Soil suitability
    if soil := profile.get("soil_suitability"):
        parts = []
        if soil.get("suitable_soil_textures"):
            parts.append(f"Suitable soils: {', '.join(soil['suitable_soil_textures'])}.")
        if soil.get("unsuitable_soil_textures"):
            parts.append(f"Unsuitable soils: {', '.join(soil['unsuitable_soil_textures'])}.")
        parts.append(f"Ideal pH: {soil.get('ideal_ph_min', '?')}-{soil.get('ideal_ph_max', '?')}.")
        if soil.get("drainage_requirement"):
            parts.append(f"Drainage: {soil['drainage_requirement']}")
        if soil.get("salinity_tolerance_note"):
            parts.append(soil["salinity_tolerance_note"])
        docs.append(Document(
            page_content=f"{crop_name} Soil Requirements: " + " ".join(parts),
            metadata={"crop_name": crop_id, "section": "soil"},
        ))

    # Climate suitability
    if climate := profile.get("climate_suitability"):
        parts = [
            f"Ideal temperature: {climate.get('ideal_temperature_c_min', '?')}-{climate.get('ideal_temperature_c_max', '?')}°C.",
            f"Ideal humidity: {climate.get('ideal_humidity_min', '?')}-{climate.get('ideal_humidity_max', '?')}%.",
            f"Ideal rainfall: {climate.get('ideal_rainfall_mm_min', '?')}-{climate.get('ideal_rainfall_mm_max', '?')} mm.",
        ]
        if climate.get("frost_tolerance_note"):
            parts.append(f"Frost: {climate['frost_tolerance_note']}")
        if climate.get("heat_tolerance_note"):
            parts.append(f"Heat: {climate['heat_tolerance_note']}")
        docs.append(Document(
            page_content=f"{crop_name} Climate Requirements: " + " ".join(parts),
            metadata={"crop_name": crop_id, "section": "climate"},
        ))

    # Nutrient profile
    if nutrients := profile.get("nutrient_profile"):
        parts = []
        for key in ["nitrogen_demand", "phosphorus_demand", "potassium_demand"]:
            if v := nutrients.get(key):
                parts.append(f"{key.replace('_', ' ').title()}: {v}")
        if nutrients.get("micronutrient_notes"):
            parts.extend(nutrients["micronutrient_notes"])
        docs.append(Document(
            page_content=f"{crop_name} Nutrient Requirements: " + " ".join(parts),
            metadata={"crop_name": crop_id, "section": "nutrients"},
        ))

    # Water profile
    if water := profile.get("water_profile"):
        parts = []
        if water.get("drought_tolerance"):
            parts.append(f"Drought tolerance: {water['drought_tolerance']}")
        if water.get("waterlogging_tolerance"):
            parts.append(f"Waterlogging tolerance: {water['waterlogging_tolerance']}")
        if water.get("critical_water_stages"):
            parts.append(f"Critical stages: {', '.join(water['critical_water_stages'])}")
        docs.append(Document(
            page_content=f"{crop_name} Water Profile: " + " ".join(parts),
            metadata={"crop_name": crop_id, "section": "water"},
        ))

    # Recommendation logic
    if rec := profile.get("recommendation_logic"):
        parts = []
        if rec.get("why_recommend_conditions"):
            parts.append("Recommended when: " + "; ".join(rec["why_recommend_conditions"]) + ".")
        if rec.get("why_not_recommend_conditions"):
            parts.append("Not recommended when: " + "; ".join(rec["why_not_recommend_conditions"]) + ".")
        if rec.get("hard_rejection_conditions"):
            parts.append("Hard reject: " + "; ".join(rec["hard_rejection_conditions"]) + ".")
        docs.append(Document(
            page_content=f"{crop_name} Recommendation Logic: " + " ".join(parts),
            metadata={"crop_name": crop_id, "section": "recommendation_logic"},
        ))

    return docs


def _extract_retrieval_chunks(crop_id: str, chunks: list[dict]) -> list[Document]:
    """Use the pre-built retrieval_chunks from each JSON file."""
    docs = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        docs.append(Document(
            page_content=f"{chunk.get('title', '')}: {text}",
            metadata={
                "crop_name": crop_id,
                "section": chunk.get("chunk_type", "general"),
                "chunk_id": chunk.get("chunk_id", ""),
                "season": chunk.get("season", ""),
            },
        ))
    return docs


def _extract_advisories(crop_id: str, crop_name: str, advisories: list[dict]) -> list[Document]:
    """Extract key advisory sections as documents."""
    docs = []
    for adv in advisories:
        region = adv.get("state_region", "India")

        # Fertilizer advice
        if fert := adv.get("fertilizer"):
            parts = []
            for k, v in fert.items():
                if v:
                    parts.append(f"{k.replace('_', ' ').title()}: {v}")
            if parts:
                docs.append(Document(
                    page_content=f"{crop_name} Fertilizer Advisory ({region}): " + " ".join(parts),
                    metadata={"crop_name": crop_id, "section": "fertilizer", "region": region},
                ))

        # Pest management
        if pests := adv.get("pest_management"):
            for pest in pests:
                text = f"{pest['name']}: {pest.get('symptoms', '')} Management: {pest.get('management', '')}"
                docs.append(Document(
                    page_content=f"{crop_name} Pest ({region}): {text}",
                    metadata={"crop_name": crop_id, "section": "pest", "region": region},
                ))

        # Disease management
        if diseases := adv.get("disease_management"):
            for disease in diseases:
                text = f"{disease['name']}: {disease.get('symptoms', '')} Management: {disease.get('management', '')}"
                docs.append(Document(
                    page_content=f"{crop_name} Disease ({region}): {text}",
                    metadata={"crop_name": crop_id, "section": "disease", "region": region},
                ))

        # FAQ
        if faqs := adv.get("faq"):
            for faq in faqs:
                docs.append(Document(
                    page_content=f"{crop_name} FAQ: Q: {faq['question']} A: {faq['answer']}",
                    metadata={"crop_name": crop_id, "section": "faq", "region": region},
                ))

    return docs


def _extract_risks(crop_id: str, crop_name: str, risk_lib: dict) -> list[Document]:
    """Extract risk library entries."""
    docs = []
    for category in ["pests", "diseases", "abiotic_risks"]:
        for item in risk_lib.get(category, []):
            name = item.get("name", "Unknown")
            parts = []
            if item.get("favorable_conditions"):
                parts.append(f"Favorable conditions: {', '.join(item['favorable_conditions'])}.")
            if item.get("trigger_conditions"):
                parts.append(f"Triggers: {', '.join(item['trigger_conditions'])}.")
            if item.get("symptoms"):
                syms = item["symptoms"] if isinstance(item["symptoms"], list) else [item["symptoms"]]
                parts.append(f"Symptoms: {', '.join(syms)}.")
            if item.get("impact"):
                parts.append(f"Impact: {item['impact']}")
            if item.get("management"):
                mgmt = item["management"] if isinstance(item["management"], list) else [item["management"]]
                parts.append(f"Management: {', '.join(mgmt)}.")
            if item.get("mitigation"):
                mit = item["mitigation"] if isinstance(item["mitigation"], list) else [item["mitigation"]]
                parts.append(f"Mitigation: {', '.join(mit)}.")

            docs.append(Document(
                page_content=f"{crop_name} Risk - {name}: " + " ".join(parts),
                metadata={"crop_name": crop_id, "section": category},
            ))
    return docs


def load_all_documents() -> list[Document]:
    """Load and chunk all 25 crop knowledge files."""
    all_docs = []

    for fpath in sorted(KNOWLEDGE_DIR.glob("*.txt*")):
        try:
            raw = fpath.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  [SKIP] {fpath.name}: {e}")
            continue

        crop_id = data.get("crop_id", fpath.stem.lower())
        crop_name = data.get("crop_name", crop_id.title())
        print(f"  Processing: {crop_name} ({fpath.name})")

        # 1. Master profile sections
        if profile := data.get("master_profile"):
            all_docs.extend(_flatten_profile(crop_id, crop_name, profile))

        # 2. Pre-built retrieval chunks
        if chunks := data.get("retrieval_chunks"):
            all_docs.extend(_extract_retrieval_chunks(crop_id, chunks))

        # 3. Regional advisories
        if advisories := data.get("region_season_advisories"):
            all_docs.extend(_extract_advisories(crop_id, crop_name, advisories))

        # 4. Risk library
        if risk_lib := data.get("risk_library"):
            all_docs.extend(_extract_risks(crop_id, crop_name, risk_lib))

    return all_docs


def build_vector_store(docs: list[Document]) -> Chroma:
    """Embed documents in small batches and persist to ChromaDB."""
    import time
    import shutil

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # Clear old DB if it exists
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"  Cleared old ChromaDB at {CHROMA_DIR}")

    print(f"\n  Embedding {len(docs)} chunks into ChromaDB (batches of 20, ~3 min total)...")

    # Create the store with the first batch
    batch_size = 20
    first_batch = docs[:batch_size]
    vectorstore = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"    Batch 1/{(len(docs) - 1) // batch_size + 1}: {len(first_batch)} chunks embedded")

    # Add remaining batches
    for i in range(batch_size, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(docs) - 1) // batch_size + 1

        time.sleep(20)  # Rate limit pause (Gemini free tier is 100 RPM)
        vectorstore.add_documents(batch)
        print(f"    Batch {batch_num}/{total_batches}: {len(batch)} chunks embedded")

    print(f"\n  Done! {len(docs)} chunks stored.\n")
    return vectorstore


if __name__ == "__main__":
    print("=" * 60)
    print("  Crop Knowledge Ingestion Pipeline")
    print("=" * 60)

    docs = load_all_documents()
    print(f"\n  Total documents extracted: {len(docs)}")

    store = build_vector_store(docs)

    # Quick sanity check
    results = store.similarity_search("soybean soil requirements", k=2)
    print("  Sanity check — top 2 results for 'soybean soil requirements':")
    for r in results:
        print(f"    [{r.metadata.get('crop_name')}] {r.page_content[:100]}...")
