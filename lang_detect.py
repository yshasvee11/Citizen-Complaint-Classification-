"""
Step 1: Language Detection
Owner: Aayush
"""
import os
import sys
from typing import Dict, Any, Optional

# Ensure IndicLID Inference directory is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_INDICLID_PATH = os.path.join(_CURRENT_DIR, "IndicLID", "Inference")
if os.path.isdir(_INDICLID_PATH) and _INDICLID_PATH not in sys.path:
    sys.path.insert(0, _INDICLID_PATH)

_model_instance: Optional[Any] = None

def get_indiclid_model():
    global _model_instance
    if _model_instance is None:
        from ai4bharat.IndicLID import IndicLID
        _model_instance = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
    return _model_instance

# Script normalization
SCRIPT_MAP = {
    "Deva": "Deva",
    "Devanagari": "Deva",
    "Taml": "Taml",
    "Tamil": "Taml",
    "Telu": "Telu",
    "Telugu": "Telu",
    "Latn": "Latn",
    "Latin": "Latn",
}

# Indo-Aryan / North Indic languages closely aligned with Hindi in Roman script
HINDI_RELATED = {"hin", "pan", "mai", "bho", "urd", "awa", "mag"}

def detect_language(text: str) -> dict:
    """
    Detects language and script of input text using IndicLID.
    Returns: {
        "language": "hin" | "tam" | "tel" | "eng" | "other",
        "script": "Deva" | "Taml" | "Telu" | "Latn" | "other",
        "confidence": float
    }
    """
    if not text or not text.strip():
        return {"language": "other", "script": "other", "confidence": 0.0}

    clean_text = text.strip()
    model = get_indiclid_model()
    results = model.batch_predict([clean_text], batch_size=1)
    if not results:
        return {"language": "other", "script": "other", "confidence": 0.0}

    # Format from IndicLID: (text, label, confidence, model_name)
    res = results[0]
    raw_label = str(res[1])
    raw_confidence = float(res[2])
    confidence = max(0.0, min(1.0, round(raw_confidence, 4)))

    if raw_label == "other" or "_" not in raw_label:
        return {"language": "other", "script": "other", "confidence": confidence}

    raw_lang, raw_script = raw_label.split("_", 1)
    normalized_script = SCRIPT_MAP.get(raw_script, "other")

    # Map detected language into target contract: "hin", "tam", "tel", "eng", "other"
    if normalized_script == "Deva":
        detected_lang = "hin"
    elif normalized_script == "Taml":
        detected_lang = "tam"
    elif normalized_script == "Telu":
        detected_lang = "tel"
    elif normalized_script == "Latn":
        if raw_lang == "eng":
            detected_lang = "eng"
        elif raw_lang in HINDI_RELATED:
            detected_lang = "hin"
        elif raw_lang == "tam":
            detected_lang = "tam"
        elif raw_lang == "tel":
            detected_lang = "tel"
        else:
            detected_lang = "other"
    else:
        detected_lang = "other"

    return {
        "language": detected_lang,
        "script": normalized_script,
        "confidence": confidence,
    }
