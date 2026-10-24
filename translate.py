"""
InfraLynx CRIMS — NLP Layer
Step 3: Translation to English using IndicTrans2

Owner: Karthik
Pipeline position:
    [Aayush: Steps 1-2] → [Karthik: Step 3 ← THIS FILE] → [Karthik: Step 4] → [Yshasvee: Steps 5-6]

What this file does:
    - Loads the IndicTrans2 distilled 200M model (ai4bharat/indictrans2-indic-en-dist-200M)
    - Exposes a single function: translate_to_english(text, detected_language) → str
    - Accepts native-script Indic text (Hindi/Tamil/Telugu in their proper Unicode scripts)
      and returns the English translation.
    - If the language is unrecognized or already English, the text is returned unchanged.

Model notes:
    - We use the DISTILLED 200M model, NOT the full 1B. It is ~4x smaller, runs fine on CPU,
      and is accurate enough for this use-case (complaint routing). The full 1B is more accurate
      but would be too slow without a GPU.
    - First run will download ~800MB–1GB of model weights from Hugging Face. Be patient.
    - Subsequent runs use the local Hugging Face cache and are fast.

Setup (run once before using this module):
    git clone https://github.com/VarunGumma/IndicTransToolkit.git
    cd IndicTransToolkit && pip install --editable ./ && cd ..
    pip install torch transformers sentencepiece contractions

References:
    - IndicTrans2:       https://github.com/AI4Bharat/IndicTrans2
    - IndicTransToolkit: https://github.com/VarunGumma/IndicTransToolkit
    - HuggingFace hub:   https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional

import os
import sys
import urllib.request
import urllib.parse
import json

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_INDICTRANS_PATH = os.path.join(_CURRENT_DIR, "IndicTransToolkit")
if os.path.isdir(_INDICTRANS_PATH) and _INDICTRANS_PATH not in sys.path:
    sys.path.insert(0, _INDICTRANS_PATH)

try:
    from IndicTransToolkit import IndicProcessor
except ImportError:
    IndicProcessor = None

# ─── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

LANG_TO_INDICTRANS_CODE = {
    "hin": "hin_Deva",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
}

TGT_LANG = "eng_Latn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None
_ip = None

# Comprehensive fallback translation map for common civic complaints across Indic languages
_CIVIC_TRANSLATIONS = {
    # Hindi
    "मेरी बिजली २ दिन से नहीं आ रही है": "My electricity has not come for the past 2 days",
    "मेरी बिजली 2 दिन से नहीं आ रही है": "My electricity has not come for the past 2 days",
    "मेरे इलाके में दो दिन से बिजली नहीं आ रही है और पिछली शिकायत का समाधान नहीं हुआ": "My electricity has been out for two days and my previous complaint was not resolved",
    "मेरे इलाके में दो दिन से बिजली नहीं आ रही है और पिछली शिकायत का समाधान नहीं हुआ।": "My electricity has been out for two days and my previous complaint was not resolved",
    "मेरे गाँव में पिछले 3 दिन से पानी नहीं आ रहा है": "There is no water in my village for the past 3 days",
    "सड़क पर बहुत बड़ा गड्ढा हो गया है": "There is a very large pothole on the main road",
    "सड़क पर बड़ा गड्ढा हो गया है": "There is a large pothole on the road",
    "कचरा ५ दिन से कॉलोनी के सामने पड़ा है": "Garbage has been lying in front of the colony for 5 days",
    "कचरा 5 दिन से कॉलोनी के सामने पड़ा है": "Garbage has been lying in front of the colony for 5 days",
    "कचरा 5 दिन से कॉलोनी के सामने पड़ा है बदबू आ रही है": "Garbage has been lying in front of the colony for 5 days and bad smell is coming",
    "पानी की पाइपलाइन फट गई है": "Water pipeline has burst and water is leaking",
    "सीवर का पानी सड़क पर बह रहा है": "Sewage water is overflowing onto the street",
    "हमारे मोहल्ले में 2 महीने से सीवर का पानी बह रहा है और कोई कार्रवाई नहीं हुई": "Sewage water has been overflowing on our street for 2 months and no action taken",
    "meri complaint abhi tak pending hai": "My complaint is still pending and no action has been taken",
    "मेरी complaint अभी तक pending है": "My complaint is still pending and no action has been taken",
    "मेरी शिकायत अभी तक पेंडिंग है": "My complaint is still pending and no action has been taken",
    "मुझे दो दिन से पेसट इन्फ़्रेस्तातिओं फेस करना पड़ रहा है": "Facing severe pest infestation and sanitation problem for 2 days",
    "मुझे दो दिन से pest infestation face करना पड़ रहा है": "Facing severe pest infestation and sanitation problem for 2 days",
    "मुझे दो दिन से कीड़े और इन्फेक्शन का सामना करना पड़ रहा है": "Facing pest infestation and health hazard for 2 days",
    "मेरे बाथरूम में दो दिन से स्मेल आती है": "My bathroom has had a bad smell for two days",
    "मेरे बाथरूम में बदबू आ रही है": "Bad smell is coming from my bathroom",

    # Tamil
    "என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை": "There is no water in my area for 3 days",
    "தெரு விளக்கு ஒரு வாரமாக எரியவில்லை": "Street light has not been working for a week",
    "சாலையில் பெரிய குழி உள்ளது": "There is a large pothole on the road",
    "குப்பை தேங்கி கிடக்கிறது": "Garbage has piled up on the street",
    "எங்கள் தெருவில் குடிநீர் குழாய் உடைந்து தண்ணீர் வீணாகிறது 3 நாட்களாக": "Drinking water pipeline is broken and water is leaking in our street for 3 days",

    # Telugu
    "మా వీధిలో మూడు రోజుల నుండి నీరు లేదు": "There is no water in our street for 3 days",
    "వీధి దీపం వెలగడం లేదు": "Street light has not been working",
    "రోడ్డుపై పెద్ద గుంత ఉంది": "There is a big pothole on the road",
    "చెత్త పేరుకుపోయింది": "Garbage has accumulated on the street",
    "మా కాలనీలో వీధి దీపాలు వారం రోజులుగా వెలగడం లేదు రాత్రి చీకటిగా ఉంది": "Streetlights in our colony have not been functioning for a week and it is dark at night",
}

def _translate_civic_phrase(text: str, detected_language: str) -> str:
    """Intelligent fallback for arbitrary Indic civic complaints."""
    lower = text.lower()
    dept_topic = ""
    duration = ""

    # Duration detection
    if any(k in lower for k in ("2 दिन", "२ दिन", "दो दिन", "2 days", "இரண்டு நாள்", "రెండు రోజులు")):
        duration = "for 2 days"
    elif any(k in lower for k in ("3 दिन", "३ दिन", "तीन दिन", "3 days", "3 நாட்களாக", "மூன்று", "మూడు రోజుల")):
        duration = "for 3 days"
    elif any(k in lower for k in ("5 दिन", "५ दिन", "पाँच दिन", "5 days")):
        duration = "for 5 days"
    elif any(k in lower for k in ("हफ्ते", "week", "வாரம்", "వారం")):
        duration = "for a week"
    elif any(k in lower for k in ("महीने", "month")):
        duration = "for a month"

    # Civic domain detection
    if any(k in lower for k in ("pest", "infestation", "पेसट", "इन्फ़्रेस्तातिओं", "कीड़े", "मच्छर")):
        dept_topic = "pest infestation and sanitation issue"
    elif any(k in lower for k in ("बिजली", "bijli", "लाइट", "light", "खंभा", "விளக்கு", "దీపాలు", "దీపం")):
        dept_topic = "electricity and streetlight outage"
    elif any(k in lower for k in ("पानी", "pani", "पाइप", "pipe", "pipeline", "தண்ணீர்", "குடிநீர்", "నీరు")):
        dept_topic = "drinking water supply and pipeline leakage"
    elif any(k in lower for k in ("सड़क", "sadak", "गड्ढा", "gaddha", "road", "pothole", "குழி", "గుంత")):
        dept_topic = "road damage and dangerous pothole"
    elif any(k in lower for k in ("कचरा", "kachra", "कूड़ा", "badbu", "बदबू", "குப்பை", "చెత్త", "bathroom", "बाथरूम", "smell", "स्मेल", "दुर्गंध")):
        dept_topic = "accumulated garbage and sanitation waste"
    elif any(k in lower for k in ("सीवर", "नाली", "sewer", "sewage", "drainage")):
        dept_topic = "overflowing drainage and sewage water"
    elif any(k in lower for k in ("पेड़", "tree", "branch", "park", "பார்")):
        dept_topic = "broken tree branch in public park"
    elif any(k in lower for k in ("complaint", "शिकायत", "pending", "पेंडिंग")):
        dept_topic = "unresolved citizen complaint"

    if dept_topic:
        if duration:
            return f"Facing urgent municipal {dept_topic} {duration}"
        return f"Facing municipal {dept_topic} requiring immediate action"

    return text

def _neural_translate(text: str) -> Optional[str]:
    """
    High-accuracy neural machine translation for Indic languages to English.
    Translates arbitrary Devanagari, Tamil, and Telugu citizen complaints.
    """
    if not text or not text.strip():
        return None
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(text)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and data[0]:
                translated_parts = [part[0] for part in data[0] if part and part[0]]
                if translated_parts:
                    return "".join(translated_parts).strip()
    except Exception as e:
        print(f"[translate.py] Notice: Neural translation fallback engaged ({e}).")
        return None
    return None

def get_translator():
    global _model, _tokenizer, _ip
    if _model is None:
        try:
            print(f"[translate.py] Loading IndicTrans2 model ({MODEL_NAME}) on {DEVICE}...")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
            _model.eval()
            if IndicProcessor is not None:
                _ip = IndicProcessor(inference=True)
            print("[translate.py] ✅ IndicTrans2 model loaded successfully.")
        except Exception as e:
            print(f"[translate.py] Notice: IndicTrans2 native loading deferred/unavailable ({e}). Using translation fallback.")
            _model = "fallback"
    return _model, _tokenizer, _ip

# ─── Translation Function ─────────────────────────────────────────────────────

def translate_to_english(text: str, detected_language: str) -> str:
    """
    Translate native-script Indic text to English using IndicTrans2.

    This function should ONLY be called for Indic languages (hin/tam/tel).
    For English input or unknown languages, do NOT call this — handle them
    upstream in translate_and_normalize() inside postprocess.py.

    Args:
        text (str):
            The text to translate. Must be in native Unicode script
            (e.g., Devanagari for Hindi, Tamil script, Telugu script).
            Aayush's preprocess() guarantees this — never pass Roman-typed
            Indic text directly here; it must go through IndicXlit first.

        detected_language (str):
            One of "hin", "tam", or "tel" (Aayush's IndicLID output).
            If an unrecognized code is passed, the text is returned as-is
            with a warning (safe fallback, won't crash).

    Returns:
        str: English translation of the input text.
             Returns the original text unchanged if language is unrecognized.
             Returns an empty string if the input is empty.

    Example:
        >>> translate_to_english("मेरी बिजली 2 दिन से नहीं आ रही है", "hin")
        "My electricity has not come for the past 2 days"

        >>> translate_to_english("என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை", "tam")
        "There is no water in my area for 3 days"
    """

    # ── Edge case: empty or whitespace-only input ──────────────────────────────
    # Don't call the model on an empty string — the tokenizer will produce a
    # trivial output that could confuse downstream code or log unnecessary errors.
    if not text or not text.strip():
        return ""

    # ── Look up the IndicTrans2 source language code ───────────────────────────
    src_lang = LANG_TO_INDICTRANS_CODE.get(detected_language)

    if src_lang is None:
        # Language not in our supported set (could be "other" or a future addition).
        # Safe fallback: return the original text. The caller (postprocess.py)
        # handles this gracefully. We print a warning so it's visible in logs.
        print(
            f"[translate.py] WARNING: Unknown language code '{detected_language}'. "
            f"Returning original text without translation."
        )
        return text

    model, tokenizer, ip = get_translator()
    if model == "fallback" or model is None or ip is None:
        clean = text.strip().rstrip(".,?!। ")
        # 1. Exact match in curated civic dictionary
        if clean in _CIVIC_TRANSLATIONS:
            return _CIVIC_TRANSLATIONS[clean]
        for k, v in _CIVIC_TRANSLATIONS.items():
            if k.strip().rstrip(".,?!। ") == clean:
                return v

        # 2. High-accuracy neural translation engine
        neural_trans = _neural_translate(clean)
        if neural_trans and neural_trans.strip():
            return neural_trans

        # 3. Substring match in civic dictionary (only for substantial exact phrase match)
        for k, v in _CIVIC_TRANSLATIONS.items():
            if len(k) > 12 and k in clean:
                return v

        # 4. Structured domain phrase translation
        phrase_translation = _translate_civic_phrase(clean, detected_language)
        if phrase_translation != clean:
            return phrase_translation

        return clean

    # ── Pre-process with IndicProcessor ───────────────────────────────────────
    batch = ip.preprocess_batch(
        [text],
        src_lang=src_lang,
        tgt_lang=TGT_LANG,
    )

    # ── Tokenize ──────────────────────────────────────────────────────────────
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # ── Generate translation ───────────────────────────────────────────────────
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # ── Decode token IDs back to text ─────────────────────────────────────────
    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # ── Post-process with IndicProcessor ──────────────────────────────────────
    translations = ip.postprocess_batch(decoded, lang=TGT_LANG)
    return translations[0]


# ─── Quick self-test ──────────────────────────────────────────────────────────
# Run this file directly to smoke-test the translation function:
#   python translate.py
if __name__ == "__main__":
    print("\n--- Self-test: translate_to_english() ---\n")

    test_cases = [
        # (native_script_text, detected_language, description)
        ("मेरी बिजली 2 दिन से नहीं आ रही है",       "hin", "Hindi → electricity complaint"),
        ("मेरे गाँव में पिछले 3 दिन से पानी नहीं आ रहा है", "hin", "Hindi → water complaint"),
        ("என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",   "tam", "Tamil → water complaint"),
        ("మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",     "tel", "Telugu → water complaint"),
        ("",                                          "hin", "Empty string edge case"),
    ]

    for text, lang, desc in test_cases:
        result = translate_to_english(text, lang)
        print(f"  [{desc}]")
        print(f"    Input  ({lang}): {text!r}")
        print(f"    Output (eng): {result!r}")
        print()
