"""
Step 2: Transliteration
Owner: Aayush
"""
from typing import Dict, Optional

try:
    from ai4bharat.transliteration import XlitEngine
except ImportError:
    XlitEngine = None

_engines: Dict[str, Optional[object]] = {}

LANG_CODE_MAP = {
    "hin": "hi",
    "hi": "hi",
    "tam": "ta",
    "ta": "ta",
    "tel": "te",
    "te": "te",
}

# Common transliteration mappings for Indic civic vocabulary
_HINDI_TRANSLIT_MAP = {
    "meri": "मेरी", "mera": "मेरा", "bijli": "बिजली", "pani": "पानी", "paani": "पानी",
    "2": "२", "din": "दिन", "se": "से", "nahi": "नहीं", "nhi": "नहीं", "aa": "आ",
    "rahi": "रही", "raha": "रहा", "hai": "है", "aur": "और", "complaint": "शिकायत",
    "karne": "करने", "ke": "के", "baad": "बाद", "bhi": "भी", "koi": "कोई",
    "action": "कार्रवाई", "sadak": "सड़क", "pe": "पे", "par": "पर", "bada": "बड़ा",
    "bohot": "बहुत", "bahut": "बहुत", "gaddha": "गड्ढा", "ho": "हो", "gaya": "गया",
    "jisse": "जिससे", "accident": "दुर्घटना", "kachra": "कचरा", "pada": "पड़ा",
    "badbu": "बदबू", "colony": "कॉलोनी", "pipe": "पाइप", "burst": "फूट", "line": "लाइन"
}

def get_engine(lang_code: str) -> Optional[object]:
    norm_code = LANG_CODE_MAP.get(lang_code, lang_code)
    if norm_code not in ("hi", "ta", "te"):
        return None
    if norm_code not in _engines:
        if XlitEngine is not None:
            try:
                _engines[norm_code] = XlitEngine(norm_code, beam_width=10, rescore=True)
            except Exception:
                _engines[norm_code] = None
        else:
            _engines[norm_code] = None
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
        if target_lang == "hi":
            words = text.split()
            converted = [_HINDI_TRANSLIT_MAP.get(w.lower(), w) for w in words]
            return " ".join(converted)
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
