"""
Steps 1 & 2: Language Detection + Transliteration
Owner: Aayush

This is the main entry point for Aayush's work.
"""
from lang_detect import detect_language
from transliterate import transliterate_to_native

LANG_CODE_MAP = {
    "hin": "hi",
    "tam": "ta",
    "tel": "te",
}

def preprocess(text: str) -> dict:
    """
    Preprocess citizen complaint text:
    Detects language and script, and transliterates Roman Indic text to native script.

    Input: raw citizen complaint text.
    Output contract:
    {
      "original_text": str,
      "detected_language": "hin" | "tam" | "tel" | "eng" | "other",
      "script": "Deva" | "Taml" | "Telu" | "Latn" | "other",
      "is_english": bool,
      "native_script_text": str,
      "confidence": float
    }
    """
    if text is None:
        text = ""

    lang_info = detect_language(text)
    language = lang_info["language"]
    script = lang_info["script"]
    confidence = lang_info["confidence"]

    is_english = (language == "eng")

    if script == "Latn" and language in LANG_CODE_MAP:
        native_text = transliterate_to_native(text, LANG_CODE_MAP[language])
    else:
        native_text = text  # already native script, or English, or "other"

    return {
        "original_text": text,
        "detected_language": language,
        "script": script,
        "is_english": is_english,
        "native_script_text": native_text,
        "confidence": confidence,
    }
