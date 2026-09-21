"""
Step 2: Transliteration
Owner: Aayush
"""
from typing import Dict, Optional
from ai4bharat.transliteration import XlitEngine

_engines: Dict[str, XlitEngine] = {}

LANG_CODE_MAP = {
    "hin": "hi",
    "hi": "hi",
    "tam": "ta",
    "ta": "ta",
    "tel": "te",
    "te": "te",
}

def get_engine(lang_code: str) -> Optional[XlitEngine]:
    norm_code = LANG_CODE_MAP.get(lang_code, lang_code)
    if norm_code not in ("hi", "ta", "te"):
        return None
    if norm_code not in _engines:
        _engines[norm_code] = XlitEngine(norm_code, beam_width=10, rescore=True)
    return _engines[norm_code]

def transliterate_to_native(text: str, lang_code: str) -> str:
    """
    Converts Roman-script Indic text to native script.
    lang_code: "hi" / "ta" / "te" or "hin" / "tam" / "tel"
    """
    if not text or not text.strip():
        return text

    target_lang = LANG_CODE_MAP.get(lang_code, lang_code)
    engine = get_engine(target_lang)
    if engine is None:
        return text

    words = text.split()
    converted_words = []
    for word in words:
        try:
            result = engine.translit_word(word, topk=1)
            if target_lang in result and result[target_lang]:
                converted_words.append(result[target_lang][0])
            else:
                converted_words.append(word)
        except Exception:
            converted_words.append(word)

    return " ".join(converted_words)
