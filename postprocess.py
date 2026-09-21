"""
InfraLynx CRIMS — NLP Layer
Steps 3 & 4: Translation + Normalization — Main Entry Point

Owner: Karthik
Pipeline position:
    [Aayush: preprocess()] → [Karthik: translate_and_normalize() ← THIS FILE] → [Yshasvee: classify_and_route()]

This is Karthik's main deliverable. Yshasvee calls:
    from postprocess import translate_and_normalize

What this file does:
    - Takes Aayush's preprocess() output dict as input
    - Decides whether translation is needed (based on detected_language)
    - Calls translate_to_english() from translate.py for Indic languages
    - Calls normalize_text() from normalize.py for all cases
    - Returns a clean output dict matching the exact contract that Yshasvee expects

Input contract (from Aayush's preprocess()):
    {
        "original_text":      str,   # raw citizen text exactly as received
        "detected_language":  str,   # "hin" | "tam" | "tel" | "eng" | "other"
        "script":             str,   # "Latn" (Roman) | "Deva" | "Taml" | "Telu"
        "is_english":         bool,  # True if English, False if Indic
        "native_script_text": str,   # always in proper native script (or same as original if English)
        "confidence":         float  # IndicLID detection confidence [0.0, 1.0]
    }

Output contract (for Yshasvee's classify_and_route()):
    {
        "original_text":     str,   # passed through unchanged from Aayush
        "detected_language": str,   # passed through unchanged from Aayush
        "english_text":      str,   # ← Karthik's main deliverable: translated + normalized English
        "confidence":        float  # passed through unchanged from Aayush
    }

DO NOT change the output contract without notifying both Aayush and Yshasvee.
"""

from translate import translate_to_english   # Step 3: IndicTrans2 translation
from normalize import normalize_text          # Step 4: spaCy + contraction normalization

# ─── Constants ────────────────────────────────────────────────────────────────

# Languages supported by our IndicTrans2 model.
# Must stay in sync with LANG_TO_INDICTRANS_CODE in translate.py.
SUPPORTED_INDIC_LANGUAGES = {"hin", "tam", "tel"}


# ─── Main deliverable ─────────────────────────────────────────────────────────

def translate_and_normalize(input_dict: dict) -> dict:
    """
    Translate Indic text to English (if needed) and normalize the result.

    This is Karthik's final deliverable function. It is the single function
    Yshasvee imports and calls. Do not rename or change its signature.

    Routing logic:
        1. detected_language == "eng":
              Skip translation. Just normalize the original_text.
              (Citizen wrote in English — IndicTrans2 not needed.)

        2. detected_language in {"hin", "tam", "tel"}:
              Translate native_script_text → English using IndicTrans2,
              then normalize the result.

        3. detected_language == "other" (or any unknown code):
              Translation not possible — IndicTrans2 is not trained for unknown
              languages and would produce garbage. Pass the original_text through
              untranslated, then normalize what we can. The classifier will
              either fail to classify it (→ "Unclassified") or get lucky if the
              text is partially English. This is an acceptable known limitation.

    Args:
        input_dict (dict): Output from Aayush's preprocess() function.
                           Must contain at minimum:
                               - "original_text": str
                               - "detected_language": str
                               - "native_script_text": str  (used for Indic translation)
                               - "confidence": float

    Returns:
        dict: Clean structured output for Yshasvee's classifier:
            {
                "original_text":     str,   # unchanged from input
                "detected_language": str,   # unchanged from input
                "english_text":      str,   # translated + normalized English
                "confidence":        float  # unchanged from input
            }

    Example — Hindi complaint:
        Input:
            {
                "original_text": "meri bijli 2 din se nahi aa rahi hai",
                "detected_language": "hin",
                "script": "Latn",
                "is_english": False,
                "native_script_text": "मेरी बिजली 2 दिन से नहीं आ रही है",
                "confidence": 0.91
            }
        Output:
            {
                "original_text": "meri bijli 2 din se nahi aa rahi hai",
                "detected_language": "hin",
                "english_text": "my electricity has not come for the past 2 days",
                "confidence": 0.91
            }

    Example — English complaint:
        Input:
            {
                "original_text": "Water supply has been cut for 3 days",
                "detected_language": "eng",
                "script": "Latn",
                "is_english": True,
                "native_script_text": "Water supply has been cut for 3 days",
                "confidence": 0.99
            }
        Output:
            {
                "original_text": "Water supply has been cut for 3 days",
                "detected_language": "eng",
                "english_text": "water supply has been cut for 3 days",
                "confidence": 0.99
            }
    """

    # ── Extract fields from Aayush's output ────────────────────────────────────
    detected_language = input_dict.get("detected_language", "other")
    original_text     = input_dict.get("original_text", "")
    native_script_text = input_dict.get("native_script_text", original_text)
    confidence        = input_dict.get("confidence", 1.0)

    # ── Step 3: Determine whether translation is needed ────────────────────────
    if detected_language == "eng":
        # ── Case 1: Already English ────────────────────────────────────────────
        # Citizen wrote in English. No translation needed.
        # We normalize the original_text directly (lowercase, fix contractions, etc.)
        print(f"[postprocess.py] Language is English — skipping translation.")
        raw_english = original_text

    elif detected_language in SUPPORTED_INDIC_LANGUAGES:
        # ── Case 2: Supported Indic language (Hindi / Tamil / Telugu) ──────────
        # We translate the native_script_text (Devanagari / Tamil / Telugu script)
        # to English using IndicTrans2.
        #
        # IMPORTANT: we translate native_script_text, NOT original_text.
        # Aayush's preprocess() guarantees native_script_text is always in proper
        # Unicode native script (IndicXlit already handled Roman→native conversion).
        # Passing Roman-typed Hindi to IndicTrans2 produces garbage output.
        print(f"[postprocess.py] Translating '{detected_language}' text using IndicTrans2...")
        raw_english = translate_to_english(native_script_text, detected_language)

        if not raw_english or not raw_english.strip():
            # Translation returned empty — this can happen if native_script_text
            # was very short or the model produced no output for this input.
            # Fall back to original_text to avoid a downstream empty-string crash.
            print(
                f"[postprocess.py] WARNING: Translation returned empty for input: "
                f"{native_script_text!r}. Falling back to original_text."
            )
            raw_english = original_text

    else:
        # ── Case 3: Unknown language ("other" or future language codes) ─────────
        # IndicTrans2 is not trained on this language. Attempting translation
        # would produce nonsense. Pass through the original text instead.
        # Yshasvee's classifier can handle this gracefully (→ "Unclassified").
        print(
            f"[postprocess.py] Language '{detected_language}' is not supported for translation. "
            f"Passing original_text through untranslated."
        )
        raw_english = original_text

    # ── Step 4: Normalize the English text ────────────────────────────────────
    # Applies: lowercase, contraction expansion, elongation fix, symbol stripping,
    # and whitespace collapse. See normalize.py for full details.
    english_text = normalize_text(raw_english)

    # ── Assemble and return output dict ────────────────────────────────────────
    # IMPORTANT: original_text, detected_language, and confidence are passed
    # through UNCHANGED. Only english_text is new output from Karthik's work.
    return {
        "original_text":     original_text,
        "detected_language": detected_language,
        "english_text":      english_text,
        "confidence":        confidence,
    }


# ─── Quick self-test ──────────────────────────────────────────────────────────
# Run this file directly to smoke-test the full Steps 3+4 pipeline:
#   python postprocess.py
if __name__ == "__main__":
    import json

    print("\n--- Self-test: translate_and_normalize() ---\n")

    # Simulate what Aayush's preprocess() produces for each test sentence
    test_inputs = [
        {
            "description": "Hindi (native script) → electricity complaint",
            "input": {
                "original_text": "मेरी बिजली 2 दिन से नहीं आ रही है",
                "detected_language": "hin",
                "script": "Deva",
                "is_english": False,
                "native_script_text": "मेरी बिजली 2 दिन से नहीं आ रही है",
                "confidence": 0.91,
            },
        },
        {
            "description": "Hindi (was Roman, now native) → water complaint",
            "input": {
                "original_text": "mere gaon mein pichle 3 din se paani nhi aa rha hai",
                "detected_language": "hin",
                "script": "Latn",
                "is_english": False,
                "native_script_text": "मेरे गाँव में पिछले 3 दिन से पानी नहीं आ रहा है",
                "confidence": 0.88,
            },
        },
        {
            "description": "Tamil (native script) → water complaint",
            "input": {
                "original_text": "என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",
                "detected_language": "tam",
                "script": "Taml",
                "is_english": False,
                "native_script_text": "என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",
                "confidence": 0.88,
            },
        },
        {
            "description": "Telugu (native script) → water complaint",
            "input": {
                "original_text": "మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",
                "detected_language": "tel",
                "script": "Telu",
                "is_english": False,
                "native_script_text": "మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",
                "confidence": 0.89,
            },
        },
        {
            "description": "English input → water complaint (no translation)",
            "input": {
                "original_text": "Water supply has been cut for 3 days",
                "detected_language": "eng",
                "script": "Latn",
                "is_english": True,
                "native_script_text": "Water supply has been cut for 3 days",
                "confidence": 0.99,
            },
        },
        {
            "description": "Unknown language → pass-through",
            "input": {
                "original_text": "meri complaint abhi tak pending hai",
                "detected_language": "other",
                "script": "Latn",
                "is_english": False,
                "native_script_text": "meri complaint abhi tak pending hai",
                "confidence": 0.45,
            },
        },
    ]

    for case in test_inputs:
        print(f"  [{case['description']}]")
        result = translate_and_normalize(case["input"])
        print(f"  Output: {json.dumps(result, ensure_ascii=False, indent=4)}")
        print()
