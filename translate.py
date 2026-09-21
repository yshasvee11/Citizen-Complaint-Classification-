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

# IndicTransToolkit provides IndicProcessor, which handles the Indic-specific
# pre/post-processing around the transformer model (sentence splitting, tag injection, etc.)
from IndicTransToolkit import IndicProcessor

# ─── Model Configuration ──────────────────────────────────────────────────────

# Distilled 200M model — lighter weight, CPU-friendly, good enough for complaint classification
MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

# ─── Language Code Mapping ────────────────────────────────────────────────────
# Maps Aayush's IndicLID language codes (ISO 639-3) to IndicTrans2's internal codes.
# Format: "<lang>_<script>" where script is the BCP-47 script subtag.
#
# Aayush's output        → IndicTrans2 src_lang
# ─────────────────────────────────────────────
# "hin"  (Hindi)         → "hin_Deva"  (Hindi, Devanagari script)
# "tam"  (Tamil)         → "tam_Taml"  (Tamil, Tamil script)
# "tel"  (Telugu)        → "tel_Telu"  (Telugu, Telugu script)
#
# Target language is always English in Latin script.
LANG_TO_INDICTRANS_CODE = {
    "hin": "hin_Deva",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
}

TGT_LANG = "eng_Latn"  # English in Latin script — always our translation target

# ─── Model Loading (done ONCE at import time, not per request) ────────────────
# Loading here at module level means: the first `from translate import ...` will
# trigger the download/load (30–90 seconds on first run, <10s thereafter from cache).
# This avoids re-loading the model on every call, which would be prohibitively slow.

print(f"[translate.py] Loading IndicTrans2 model: {MODEL_NAME} ...")

# Detect device: use GPU if available, otherwise fall back to CPU.
# On a course server or laptop without a GPU, CPU is fine — just a few seconds per sentence.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[translate.py] Using device: {DEVICE}")

# Load tokenizer with trust_remote_code=True because IndicTrans2 has custom tokenizer code
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load seq2seq model (encoder-decoder architecture, similar to mBART/mT5)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = model.to(DEVICE)   # move to GPU if available
model.eval()               # set to inference mode (disables dropout, etc.)

# IndicProcessor handles Indic-specific pre/post-processing:
# - Sentence normalization for each script
# - Language tag injection that IndicTrans2 expects
# - Post-processing cleanup after decoding
ip = IndicProcessor(inference=True)

print(f"[translate.py] ✅ IndicTrans2 model loaded successfully.")

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

    # ── Pre-process with IndicProcessor ───────────────────────────────────────
    # ip.preprocess_batch() normalizes the Indic text and injects the language
    # tags that IndicTrans2 uses to condition its encoder. It expects a LIST of
    # strings (we pass a single-element list for single-sentence mode).
    batch = ip.preprocess_batch(
        [text],
        src_lang=src_lang,
        tgt_lang=TGT_LANG,
    )

    # ── Tokenize ──────────────────────────────────────────────────────────────
    # truncation=True: complaints > 256 tokens are silently truncated.
    # padding="longest": pads to the longest sequence in the batch (just 1 here).
    # return_tensors="pt": return PyTorch tensors (needed for model.generate()).
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # ── Generate translation ───────────────────────────────────────────────────
    # torch.no_grad() disables gradient tracking — required for inference,
    # reduces memory usage significantly.
    #
    # Beam search (num_beams=5) explores 5 translation candidates in parallel
    # and returns the highest-scoring one. More beams = better quality but slower.
    # 5 is a good balance for a CPU-only setup.
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,        # KV-cache speeds up decoding
            min_length=0,          # allow very short outputs (e.g., single-word answers)
            max_length=256,        # max output length in tokens
            num_beams=5,           # beam search width
            num_return_sequences=1,  # return only the best translation
        )

    # ── Decode token IDs back to text ─────────────────────────────────────────
    # We use tokenizer.as_target_tokenizer() context manager because IndicTrans2
    # uses a shared vocabulary with language-specific tokenization; the target
    # tokenizer handles the English output side correctly.
    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,          # remove <s>, </s>, language tags
            clean_up_tokenization_spaces=True,  # fix spaces around punctuation
        )

    # ── Post-process with IndicProcessor ──────────────────────────────────────
    # ip.postprocess_batch() removes any remaining Indic-specific artifacts,
    # normalizes Unicode, and returns clean English strings.
    translations = ip.postprocess_batch(decoded, lang=TGT_LANG)

    # Return the single translation (index 0 since we sent one sentence in)
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
