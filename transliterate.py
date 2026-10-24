"""
Step 2: Transliteration
Owner: Aayush
"""
from typing import Dict, Optional, Set
import re

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as sanscript_transliterate
except ImportError:
    sanscript = None
    sanscript_transliterate = None

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

# English loanwords commonly used in Hinglish / Tanglish / Telugu civic complaints.
# These should be kept in English or cleanly preserved.
ENGLISH_LOANWORDS: Set[str] = {
    "complaint", "pending", "action", "hospital", "clinic", "pest", "infestation",
    "face", "water", "pipeline", "pipe", "meter", "electric", "wire", "transformer",
    "light", "street", "road", "gate", "drainage", "sewage", "overflow", "urgent",
    "critical", "danger", "hazard", "problem", "issue", "station", "park", "track",
    "status", "online", "application", "service", "area", "zone", "ward", "colony"
}

# Common conversational & civic vocabulary for Hindi (Hinglish -> Devanagari)
_HINDI_TRANSLIT_MAP: Dict[str, str] = {
    "meri": "मेरी", "mera": "मेरा", "mere": "मेरे", "mujhe": "मुझे", "hum": "हम", "hamare": "हमारे",
    "bijli": "बिजली", "pani": "पानी", "paani": "पानी", "2": "२", "din": "दिन", "se": "से",
    "nahi": "नहीं", "nhi": "नहीं", "na": "ना", "mat": "मत", "aa": "आ", "rahi": "रही", "raha": "रहा",
    "rahe": "रहे", "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी", "the": "थे", "aur": "और",
    "karne": "करने", "karna": "करना", "kare": "करें", "karo": "करो", "kiya": "किया", "ke": "के",
    "ka": "का", "ki": "की", "ko": "को", "baad": "बाद", "bhi": "भी", "koi": "कोई", "sadak": "सड़क",
    "sarak": "सड़क", "pe": "पे", "par": "पर", "bada": "बड़ा", "bohot": "बहुत", "bahut": "बहुत",
    "gaddha": "गड्ढा", "gadha": "गड्ढा", "ho": "हो", "gaya": "गया", "gayi": "गई", "jisse": "जिससे",
    "kachra": "कचरा", "kooda": "कूड़ा", "pada": "पड़ा", "badbu": "बदबू", "colony": "कॉलोनी",
    "line": "लाइन", "pad": "पड़", "do": "दो", "teen": "तीन", "char": "चार", "paanch": "पाँच",
    "abhi": "अभी", "tak": "तक", "ilaka": "इलाका", "ilake": "इलाके", "gali": "गली", "mohalla": "मोहल्ले",
    "pichli": "पिछली", "samadhan": "समाधान", "gaon": "गाँव", "pichle": "पिछले", "sewer": "सीवर",
    "naali": "नाली", "nali": "नाली", "batti": "बत्ती", "pole": "खंभा", "khamba": "खंभा",
    "ped": "पेड़", "shikayat": "शिकायत", "durghatna": "दुर्घटना", "khatra": "खतरा", "safai": "सफाई"
}

# Common conversational & civic vocabulary for Tamil (Tanglish -> Tamil script)
_TAMIL_TRANSLIT_MAP: Dict[str, str] = {
    "en": "என்", "pakuthiyil": "பகுதியில்", "3": "3", "naatkalaga": "நாட்களாக",
    "thanni": "தண்ணீர்", "illai": "இல்லை", "engal": "எங்கள்", "theruvil": "தெருவில்",
    "kudi": "குடி", "kudineer": "குடிநீர்", "kuzhai": "குழாய்", "udaithu": "உடைந்து",
    "thannir": "தண்ணீர்", "veenagirathu": "வீணாகிறது", "naatkal": "நாட்கள்", "theru": "தெரு",
    "vilakku": "விளக்கு", "oru": "ஒரு", "vaaramaga": "வாரமாக", "eriyavillai": "எரியவில்லை",
    "saalaiyil": "சாலையில்", "periya": "பெரிய", "kuzhi": "குழி", "ullathu": "உள்ளது",
    "kuppai": "குப்பை", "theangi": "தேங்கி", "kidakkirathu": "கிடக்கிறது", "moonu": "மூன்று",
    "naal": "நாள்", "vanakkam": "வணக்கம்"
}

# Common conversational & civic vocabulary for Telugu (Telugu Roman -> Telugu script)
_TELUGU_TRANSLIT_MAP: Dict[str, str] = {
    "maa": "మా", "veedhilo": "వీధిలో", "moodu": "మూడు", "rojula": "రోజుల", "nunchi": "నుండి",
    "neeru": "నీరు", "ledu": "లేదు", "colonylo": "కాలనీలో", "veedhi": "వీధి", "deepalu": "దీపాలు",
    "varam": "వారం", "rojuluga": "రోజులుగా", "velagadam": "వెలగడం", "raatri": "రాత్రి",
    "cheekatiga": "చీకటిగా", "undi": "ఉంది", "undhi": "ఉంది", "maku": "మాకు", "roddupai": "రోడ్డుపై",
    "pedda": "పెద్ద", "guntha": "గుంత", "chetta": "చెత్త", "perukupoyindi": "పేరుకుపోయింది",
    "deepam": "దీపం", "namaskaram": "నమస్కారం"
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

def _transliterate_word_phonetic(word: str, target_lang: str) -> str:
    """Phonetic fallback for words not in the civic dictionary."""
    if sanscript is not None and sanscript_transliterate is not None:
        try:
            target_scheme = sanscript.DEVANAGARI
            if target_lang == "ta":
                target_scheme = sanscript.TAMIL
            elif target_lang == "te":
                target_scheme = sanscript.TELUGU
            return sanscript_transliterate(word, sanscript.ITRANS, target_scheme)
        except Exception:
            return word
    return word

import urllib.request
import urllib.parse
import json

def _neural_transliterate(text: str, target_lang: str) -> Optional[str]:
    """
    High-accuracy neural transliteration model (Google Input Tools engine).
    Provides native script representation for colloquial Hinglish, Tanglish, and Telugu-Roman.
    """
    lang_map = {
        "hi": "hi-t-i0-und",
        "hin": "hi-t-i0-und",
        "ta": "ta-t-i0-und",
        "tam": "ta-t-i0-und",
        "te": "te-t-i0-und",
        "tel": "te-t-i0-und",
    }
    itc = lang_map.get(target_lang)
    if not itc:
        return None

    url = "https://inputtools.google.com/request?text=" + urllib.parse.quote(text) + f"&itc={itc}&num=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and data[0] == "SUCCESS" and len(data) > 1 and data[1]:
                result_parts = []
                for item in data[1]:
                    if item and len(item) > 1 and item[1]:
                        result_parts.append(item[1][0])
                    elif item and len(item) > 0 and isinstance(item[0], str):
                        result_parts.append(item[0])
                if result_parts:
                    return "".join(result_parts)
    except Exception:
        return None
    return None

def transliterate_to_native(text: str, lang_code: str) -> str:
    """
    Converts Roman-script Indic text to native script.
    lang_code: "hi" / "ta" / "te" or "hin" / "tam" / "tel"
    Uses Google Input Tools neural transliterator with local dictionary + sanscript fallback.
    """
    if not text or not text.strip():
        return text

    target_lang = LANG_CODE_MAP.get(lang_code, lang_code)

    # 1. Primary: Neural transliteration model
    neural_res = _neural_transliterate(text, target_lang)
    if neural_res and neural_res.strip():
        return neural_res

    # 2. Secondary fallback: Local vocabulary dictionary + IndicXlit / sanscript
    engine = get_engine(target_lang)

    vocab_map: Dict[str, str] = {}
    if target_lang == "hi":
        vocab_map = _HINDI_TRANSLIT_MAP
    elif target_lang == "ta":
        vocab_map = _TAMIL_TRANSLIT_MAP
    elif target_lang == "te":
        vocab_map = _TELUGU_TRANSLIT_MAP

    words = text.split()
    converted_words = []

    for word in words:
        clean_w = word.strip(".,?!।;:()\"'")
        lower_w = clean_w.lower()

        # 1. Preserve English loanwords commonly present in code-mixed complaints
        if lower_w in ENGLISH_LOANWORDS:
            converted_words.append(word)
            continue

        # 2. Check curated civic & conversational vocabulary dictionary
        if lower_w in vocab_map:
            mapped = vocab_map[lower_w]
            # Preserve punctuation on original word
            if clean_w != word:
                converted_words.append(word.replace(clean_w, mapped))
            else:
                converted_words.append(mapped)
            continue

        # 3. Try official IndicXlit neural engine if available
        if engine is not None:
            try:
                result = engine.translit_word(clean_w, topk=1)
                if target_lang in result and result[target_lang]:
                    res_word = result[target_lang][0]
                    if clean_w != word:
                        converted_words.append(word.replace(clean_w, res_word))
                    else:
                        converted_words.append(res_word)
                    continue
            except Exception:
                pass

        # 4. Phonetic transliteration fallback via sanscript
        phonetic_word = _transliterate_word_phonetic(clean_w, target_lang)
        if clean_w != word:
            converted_words.append(word.replace(clean_w, phonetic_word))
        else:
            converted_words.append(phonetic_word)

    return " ".join(converted_words)

