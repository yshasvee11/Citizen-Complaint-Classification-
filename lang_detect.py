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
        try:
            from ai4bharat.IndicLID import IndicLID
            _model_instance = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
        except Exception:
            _model_instance = "heuristic"
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
    Detects language and script of input text using IndicLID (or heuristic fallback).
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

    if model == "heuristic" or not hasattr(model, "batch_predict"):
        clean_lower = clean_text.lower()
        if any('\u0900' <= ch <= '\u097F' for ch in clean_text):
            return {"language": "hin", "script": "Deva", "confidence": 0.98}
        if any('\u0B80' <= ch <= '\u0BFF' for ch in clean_text):
            return {"language": "tam", "script": "Taml", "confidence": 0.98}
        if any('\u0C00' <= ch <= '\u0C7F' for ch in clean_text):
            return {"language": "tel", "script": "Telu", "confidence": 0.98}

        # Romanized Indic detection heuristics (Tanglish, Telugu-English, Hinglish)
        words = clean_lower.split()
        word_set = set(words)

        tamil_roman_markers = {
            "en", "pakuthiyil", "naatkalaga", "thanni", "illai", "engal", "theruvil",
            "kudi", "kudineer", "kuzhai", "udaithu", "thannir", "veenagirathu", "theru",
            "vilakku", "oru", "vaaramaga", "eriyavillai", "saalaiyil", "periya", "kuzhi",
            "ullathu", "kuppai", "theangi", "kidakkirathu", "moonu", "naal", "vanakkam"
        }
        if sum(1 for w in words if w in tamil_roman_markers) >= 2 or any(w in word_set for w in ("pakuthiyil", "naatkalaga", "eriyavillai", "veenagirathu", "saalaiyil")):
            return {"language": "tam", "script": "Latn", "confidence": 0.92}

        telugu_roman_markers = {
            "maa", "veedhilo", "moodu", "rojula", "nunchi", "neeru", "ledu", "colonylo",
            "veedhi", "deepalu", "varam", "rojuluga", "velagadam", "raatri", "cheekatiga",
            "undi", "undhi", "maku", "roddupai", "pedda", "guntha", "chetta", "perukupoyindi",
            "deepam", "namaskaram"
        }
        if sum(1 for w in words if w in telugu_roman_markers) >= 2 or any(w in word_set for w in ("veedhilo", "rojula", "velagadam", "roddupai", "perukupoyindi", "cheekatiga")):
            return {"language": "tel", "script": "Latn", "confidence": 0.92}

        hinglish_words = {
            "meri", "mera", "mere", "mujhe", "hum", "hamare", "nahi", "nhi", "rahi", "raha",
            "rahe", "hai", "hain", "tha", "thi", "the", "din", "bijli", "sadak", "sarak",
            "gaddha", "gadha", "pani", "paani", "kachra", "badbu", "aur", "abhi", "tak",
            "karna", "karne", "kare", "karo", "pad", "pada", "padi", "se", "pe", "par",
            "ke", "ka", "ki", "ko", "bahut", "bohot", "bhi", "aaj", "kal", "gaya", "gayi",
            "ho", "ilaka", "ilake", "gali", "mohalla", "pichli", "samadhan", "shikayat"
        }
        if sum(1 for w in words if w in hinglish_words) >= 2 or any(w in word_set for w in ("bijli", "gaddha", "kachra", "badbu", "mohalla")):
            return {"language": "hin", "script": "Latn", "confidence": 0.92}

        return {"language": "eng", "script": "Latn", "confidence": 0.95}

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
