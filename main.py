"""
FastAPI application — the single entry point for NestJS to call.
Chains all three teammates' work:
    Aayush (preprocess) → Karthik (translate_and_normalize) → Yshasvee (classify_and_route)

Endpoints:
    POST /analyze         — single complaint
    POST /analyze/batch   — multiple complaints (for demo/testing)
    GET  /health          — readiness check

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8001
"""
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# ─── Team imports ─────────────────────────────────────────────────────────────
from preprocess import preprocess                  # Aayush — Steps 1-2
from postprocess import translate_and_normalize    # Karthik — Steps 3-4
from classify_route import classify_and_route      # Yshasvee — Steps 5-6
from groq_client import check_groq_status          # Hybrid Groq AI status
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InfraLynx NLP Service",
    description="Multilingual complaint analysis pipeline for InfraLynx CRIMS",
    version="1.0.0",
)

# Allow NestJS backend (typically localhost:3000) to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "front-end")
STUDIO_DIR = os.path.join(BASE_DIR, "nlp-demo-frontend", "dist")

# Mount InfraLynx civic portal static directories
if os.path.isdir(FRONTEND_DIR):
    styles_path = os.path.join(FRONTEND_DIR, "styles")
    scripts_path = os.path.join(FRONTEND_DIR, "scripts")
    assets_path = os.path.join(FRONTEND_DIR, "assets")
    pages_path = os.path.join(FRONTEND_DIR, "pages")

    if os.path.isdir(styles_path):
        app.mount("/styles", StaticFiles(directory=styles_path), name="styles")
    if os.path.isdir(scripts_path):
        app.mount("/scripts", StaticFiles(directory=scripts_path), name="scripts")
    if os.path.isdir(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
    if os.path.isdir(pages_path):
        app.mount("/pages", StaticFiles(directory=pages_path), name="pages")

    @app.get("/")
    @app.get("/index.html")
    def serve_frontend_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount dedicated NLP Studio model inspection dashboard under /nlp-studio
if os.path.isdir(STUDIO_DIR):
    app.mount("/nlp-studio", StaticFiles(directory=STUDIO_DIR, html=True), name="nlp-studio")
    if not os.path.isdir(FRONTEND_DIR):
        @app.get("/")
        @app.get("/index.html")
        def redirect_to_studio():
            return RedirectResponse(url="/nlp-studio/")

# ─── Request/Response models ─────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str

class BatchAnalyzeRequest(BaseModel):
    texts: List[str]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

# ─── Pipeline ────────────────────────────────────────────────────────────────
def full_pipeline(text: str, debug: bool = False) -> dict:
    """
    Run the complete NLP pipeline: detect → transliterate → translate →
    normalize → classify → extract entities → estimate urgency.
    """
    timings = {}

    # Steps 1-2: Language detection + transliteration (Aayush)
    t0 = time.time()
    step_1_2 = preprocess(text)
    timings["detection_transliteration_ms"] = round((time.time() - t0) * 1000)

    # Steps 3-4: Translation + normalization (Karthik)
    t1 = time.time()
    step_3_4 = translate_and_normalize(step_1_2)
    timings["translation_normalization_ms"] = round((time.time() - t1) * 1000)

    # Steps 5-6: Classification + entities + urgency + routing (Yshasvee)
    t2 = time.time()
    step_5_6 = classify_and_route(step_3_4)
    timings["classification_routing_ms"] = round((time.time() - t2) * 1000)

    # Total processing time
    step_5_6["processing_time_ms"] = round((time.time() - t0) * 1000)
    # Ensure Step 1-2 metadata is explicitly preserved
    step_5_6["script"] = step_1_2.get("script")
    step_5_6["is_english"] = step_1_2.get("is_english")
    step_5_6["native_script_text"] = step_1_2.get("native_script_text")
    step_5_6["confidence"] = step_1_2.get("confidence")

    if debug:
        step_5_6["pipeline_log"] = timings

    return step_5_6

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Readiness check — NestJS should poll this before sending real requests."""
    return {"status": "ready", "models_loaded": True}

@app.get("/groq/status")
def groq_status(refresh: bool = False):
    """
    Check if Groq AI API is working and responsive.
    Cached by default to prevent continuous pings to the API.
    Pass ?refresh=true when user clicks manual refresh.
    """
    return check_groq_status(force_refresh=refresh)

@app.post("/analyze")
def analyze(payload: AnalyzeRequest, debug: bool = Query(False)):
    """
    Analyze a single citizen complaint.
    NestJS calls this with: POST /analyze {"text": "..."}
    Add ?debug=true to include per-step timing breakdown.
    """
    return full_pipeline(payload.text, debug=debug)

@app.post("/analyze/batch")
def analyze_batch(payload: BatchAnalyzeRequest):
    """
    Analyze multiple complaints at once — for demo/testing.
    Body: {"texts": ["complaint 1", "complaint 2", ...]}
    """
    return [full_pipeline(t) for t in payload.texts]

# ─── Startup warm-up ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def warmup():
    """
    Run a dummy sentence through the full pipeline on startup.
    This forces all models (classifier, spaCy, etc.) into memory so the
    first real citizen request doesn't eat a 30-90 second delay.
    """
    print("[...] Warming up NLP pipeline...")
    full_pipeline("test sentence for warming up the models")
    print("[OK] All models loaded and warm. NLP service is ready.")
