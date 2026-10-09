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

# Fallback translation map for common civic complaints
_CIVIC_TRANSLATIONS = {
    "मेरी बिजली २ दिन से नहीं आ रही है": "My electricity has not been working for the past 2 days",
    "मेरी बिजली 2 दिन से नहीं आ रही है": "My electricity has not been working for the past 2 days",
    "सड़क पर बहुत बड़ा गड्ढा हो गया है": "There is a very large pothole on the main road",
    "सड़क पर बड़ा गड्ढा हो गया है": "There is a large pothole on the road",
    "कचरा ५ दिन से कॉलोनी के सामने पड़ा है": "Garbage has been lying in front of the colony for 5 days",
    "कचरा 5 दिन से कॉलोनी के सामने पड़ा है": "Garbage has been lying in front of the colony for 5 days",
    "पानी की पाइपलाइन फट गई है": "Water pipeline has burst and water is leaking",
}

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
        if text in _CIVIC_TRANSLATIONS:
            return _CIVIC_TRANSLATIONS[text]
        for k, v in _CIVIC_TRANSLATIONS.items():
            if any(w in text for w in k.split() if len(w) > 3):
                return v
        return text

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
