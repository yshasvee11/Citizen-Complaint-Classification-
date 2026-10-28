"""
InfraLynx CRIMS — NLP Layer
Groq AI Integration for Hybrid Classification & Urgency Analysis

Uses Groq LPU API for contextual Request Type and Urgency Analysis.
Falls back seamlessly to local models when Groq is unreachable or disabled.
Status is cached to prevent frequent API calls.
"""
import os
import time
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

_cached_status: Dict[str, Any] = {
    "checked": False,
    "working": False,
    "model": GROQ_MODEL,
    "latency_ms": None,
    "last_checked": None,
    "error": None
}

def get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None

def check_groq_status(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Check if the Groq API key is valid and working.
    Caches the result so it does not ping repeatedly.
    Only pings when force_refresh=True or upon the very first check.
    """
    global _cached_status

    if _cached_status["checked"] and not force_refresh:
        return _cached_status

    if not GROQ_API_KEY:
        _cached_status = {
            "checked": True,
            "working": False,
            "model": GROQ_MODEL,
            "latency_ms": None,
            "last_checked": time.strftime("%H:%M:%S"),
            "error": "GROQ_API_KEY is not configured"
        }
        return _cached_status

    client = get_groq_client()
    if client is None:
        _cached_status = {
            "checked": True,
            "working": False,
            "model": GROQ_MODEL,
            "latency_ms": None,
            "last_checked": time.strftime("%H:%M:%S"),
            "error": "groq library not installed"
        }
        return _cached_status

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=3,
            timeout=4.0
        )
        latency = round((time.time() - t0) * 1000)
        _cached_status = {
            "checked": True,
            "working": True,
            "model": GROQ_MODEL,
            "latency_ms": latency,
            "last_checked": time.strftime("%H:%M:%S"),
            "error": None
        }
    except Exception as e:
        latency = round((time.time() - t0) * 1000)
        _cached_status = {
            "checked": True,
            "working": False,
            "model": GROQ_MODEL,
            "latency_ms": latency,
            "last_checked": time.strftime("%H:%M:%S"),
            "error": str(e)
        }

    return _cached_status

def groq_analyze_request_type_and_urgency(english_text: str, original_text: str = "") -> Optional[Dict[str, Any]]:
    """
    Use Groq AI to classify request_type and urgency.
    Returns structured dict if successful, or None to trigger local fallback.
    """
    if not GROQ_API_KEY:
        return None

    client = get_groq_client()
    if client is None:
        return None

    # Don't try if recently checked and failed
    if _cached_status["checked"] and not _cached_status["working"]:
        return None

    system_prompt = (
        "You are an expert civic complaint triage analyst. "
        "Classify the citizen text into:\n"
        "1. request_type: 'Complaint' (for defects, breakdown, outages, dirty areas, leakages, damage, hazards) "
        "OR 'Improvement' (for requests for new infrastructure, new installation, upgrades, benches, beautification).\n"
        "2. request_type_confidence: float between 0.70 and 0.99.\n"
        "3. urgency_score: float from 0.0 to 5.0 reflecting real civic hazard and duration "
        "(Critical: 3.0 to 5.0, High: 2.0 to 2.9, Medium: 1.0 to 1.9, Low: 0.0 to 0.9).\n"
        "4. urgency_label: 'Critical' | 'High' | 'Medium' | 'Low'.\n"
        "5. rationale: concise one-sentence explanation.\n"
        "Output ONLY valid JSON."
    )

    user_text = f"Citizen Submission:\nOriginal: {original_text}\nEnglish: {english_text}"

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=250,
            timeout=4.5
        )
        content = resp.choices[0].message.content
        data = json.loads(content)

        req_type = data.get("request_type", "Complaint")
        if req_type not in ("Complaint", "Improvement"):
            req_type = "Complaint" if "complaint" in req_type.lower() else "Improvement"

        conf = float(data.get("request_type_confidence", 0.92))
        conf = max(0.60, min(0.99, conf))

        score = float(data.get("urgency_score", 1.5))
        score = max(0.0, min(5.0, round(score, 1)))

        label = data.get("urgency_label", "")
        if label not in ("Critical", "High", "Medium", "Low"):
            if score >= 3.0:
                label = "Critical"
            elif score >= 2.0:
                label = "High"
            elif score >= 1.0:
                label = "Medium"
            else:
                label = "Low"

        rationale = data.get("rationale", "")

        return {
            "source": "groq-ai",
            "model": GROQ_MODEL,
            "request_type": req_type,
            "request_type_confidence": conf,
            "urgency": {
                "score": score,
                "label": label,
                "rationale": rationale
            },
            "groq_latency_ms": round((time.time() - t0) * 1000)
        }
    except Exception as e:
        # Transparently fallback to local models on error or timeout
        print(f"[groq_client] Notice: Groq API call failed or timed out ({e}). Using local fallback.")
        return None
