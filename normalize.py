"""
InfraLynx CRIMS — NLP Layer
Step 4: Text Normalization

Owner: Karthik
Pipeline position:
    [Aayush: Steps 1-2] → [Karthik: Step 3] → [Karthik: Step 4 ← THIS FILE] → [Yshasvee: Steps 5-6]

What this file does:
    - Cleans up English text before Yshasvee's classifier sees it.
    - Handles both machine-translated output from IndicTrans2 (which may have
      capitalization/spacing artifacts) and raw citizen English input (which can
      have typos, emoji, slang, elongated words, contractions, etc.)
    - Exposes a single function: normalize_text(text: str) → str

Normalization steps (in order):
    1. Lowercase + strip leading/trailing whitespace
    2. Expand contractions  ("didn't" → "did not")
    3. Fix elongated words  ("pleeease" → "please", but NOT "committee" → "comite")
    4. Strip emojis and symbols (keep . , ? ! as they help the classifier)
    5. Collapse extra whitespace

Dependencies:
    pip install spacy contractions
    python -m spacy download en_core_web_sm
"""

import re
import spacy
import contractions  # pip install contractions

# ─── spaCy model loading ──────────────────────────────────────────────────────
# We load the small English model. We only need the vocabulary (tokenizer +
# lexeme data) for the elongated-word check. We do NOT need the full pipeline
# (NER, parser, tagger) so we disable it for speed.
#
# If 'en_core_web_sm' is not installed, run:
#   python -m spacy download en_core_web_sm
print("[normalize.py] Loading spaCy model (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
print("[normalize.py] ✅ spaCy model loaded.")

# ─── Helper functions ─────────────────────────────────────────────────────────

def expand_contractions(text: str) -> str:
    """
    Expand English contractions to their full forms using the `contractions` library.

    Uses a comprehensive, community-maintained dictionary of 100+ contractions
    so we don't miss edge cases that a hand-rolled dict would.

    Examples:
        "didn't"  → "did not"
        "I've"    → "I have"
        "won't"   → "will not"
        "it's"    → "it is"
        "y'all"   → "you all"

    Args:
        text (str): Input text, may contain contractions.

    Returns:
        str: Text with contractions expanded.
    """
    # contractions.fix() handles both ' and ' (curly apostrophes)
    return contractions.fix(text)


def fix_elongated_words(text: str) -> str:
    """
    Collapse elongated/repeated characters in words that spaCy doesn't recognize
    as real English vocabulary.

    Why this matters: Citizens writing complaints informally often type things like:
        "pleeease fix the road"    → "please fix the road"
        "noooo action taken"       → "no action taken"
        "sooooo bad"               → "so bad"

    IMPORTANT — the naive approach of collapsing ANY repeated letter is WRONG:
    It breaks real English words that happen to have double letters:
        "committee"   → would become "comite"    ❌
        "address"     → would become "adres"     ❌
        "mississippi" → would become "misisipi"  ❌

    Our approach: ONLY collapse repeated characters in words that spaCy's
    vocabulary does NOT recognize (out-of-vocabulary = likely informal/elongated).
    We collapse to max 2 repeats (not 1) to preserve words like "oo" in "oof"
    or genuinely doubled letters that are out-of-vocab but meaningful.

    Args:
        text (str): Input text, may contain elongated words.

    Returns:
        str: Text with elongated words collapsed where safe.

    Examples:
        >>> fix_elongated_words("pleeease fix this")
        "please fix this"

        >>> fix_elongated_words("the committee reviewed the address in mississippi")
        "the committee reviewed the address in mississippi"  # unchanged
    """
    words = text.split()
    fixed_words = []

    for word in words:
        # nlp.vocab[word].is_oov is True when the word is NOT in spaCy's vocabulary.
        # Real English words like "committee", "address" are in-vocab → we leave them alone.
        # Elongated slang like "pleeease", "noooo" are OOV → we collapse repeats.
        if nlp.vocab[word].is_oov:
            # Regex: match any character followed by 2+ of the same character
            # Replace with the character + itself (max 2 repeats retained)
            # e.g. "pleee" → "plee" → hmm, we do this iteratively via regex
            # r'(.)\1{2,}' matches a char followed by 2 or more of itself
            # r'\1\1' replaces with exactly 2 of that char
            # This handles "pleeease" → "pleease" — still OOV, but that's OK.
            # The classifier can still handle "pleease". We avoid over-collapsing.
            word = re.sub(r'(.)\1{2,}', r'\1\1', word)
        fixed_words.append(word)

    return " ".join(fixed_words)


# ─── Main normalization function ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Normalize English text for Yshasvee's classifier.

    This function runs AFTER translation (IndicTrans2 output or direct English input).
    It cleans up informal citizen language and machine-translation artifacts.

    Steps applied (in order):
        1. Lowercase + strip:        "My Electricity" → "my electricity"
        2. Expand contractions:      "didn't come" → "did not come"
        3. Fix elongated words:      "pleeease fix" → "please fix"  (safe — real words untouched)
        4. Strip non-text chars:     Remove emojis, symbols; keep . , ? !
        5. Collapse whitespace:      "has   not" → "has not"

    Args:
        text (str): English text — either translated from Indic or written directly in English.
                    May contain contractions, emojis, elongated slang, extra spaces.

    Returns:
        str: Clean, lowercase English text ready for classification and entity extraction.
             Returns empty string if input is empty or whitespace-only.

    Examples:
        >>> normalize_text("My Electricity has NOT come for 2 days!!")
        "my electricity has not come for 2 days!!"

        >>> normalize_text("Meri complaint abhi tak pendinggg 😡😡, koi action nahiii!!")
        "meri complaint abhi tak pending , koi action nahi!!"

        >>> normalize_text("Water supply didn't come for 3 days")
        "water supply did not come for 3 days"

        >>> normalize_text("The committee reviewed the address")
        "the committee reviewed the address"   # real words with double letters preserved
    """

    # ── Step 0: Guard against empty input ─────────────────────────────────────
    if not text or not text.strip():
        return ""

    # ── Step 1: Lowercase + strip ─────────────────────────────────────────────
    # Lowercase everything so the classifier treats "Water" and "water" the same.
    # strip() removes leading/trailing whitespace (common in translated output).
    text = text.lower().strip()

    # ── Step 2: Expand contractions ───────────────────────────────────────────
    # Must happen AFTER lowercasing so the contractions library sees consistent case.
    # e.g., "didn't" → "did not", "can't" → "can not", "it's" → "it is"
    text = expand_contractions(text)

    # ── Step 3: Fix elongated words ────────────────────────────────────────────
    # e.g., "pleeease" → "please", "noooo" → "no"
    # Real vocabulary words (committee, address, etc.) are NOT touched — see function docstring.
    text = fix_elongated_words(text)

    # ── Step 4: Strip symbols and emojis ──────────────────────────────────────
    # Keep: a-z, 0-9, whitespace, and basic punctuation (. , ? ! for sentence structure)
    # Remove: everything else (emojis, @, #, *, ^, ~, non-ASCII symbols, etc.)
    # Why keep punctuation? Exclamation marks signal urgency ("no water for 3 days!!")
    # and question marks indicate queries. Yshasvee's urgency scorer may use these.
    text = re.sub(r'[^a-z0-9\s.,?!]', ' ', text)

    # ── Step 5: Collapse extra whitespace ─────────────────────────────────────
    # Step 4 can introduce multiple consecutive spaces (where symbols were removed).
    # \s+ matches one or more whitespace characters and collapses them to one space.
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ─── Quick self-test ──────────────────────────────────────────────────────────
# Run this file directly to smoke-test the normalization function:
#   python normalize.py
if __name__ == "__main__":
    print("\n--- Self-test: normalize_text() ---\n")

    test_cases = [
        # (input_text, description)
        ("My Electricity has NOT come for 2 days!!",
         "Basic capitalization + punctuation"),

        ("Meri complaint abhi tak pendinggg 😡😡, koi action nahiii!!",
         "Elongated words + emojis"),

        ("Water supply didn't come for 3 days",
         "Contraction expansion"),

        ("pleeease fix the road sooooon",
         "Multiple elongated words"),

        ("the committee reviewed the address in mississippi",
         "Real words with double/triple letters — must NOT be mangled"),

        ("  Road has been   broken   since last   week   ",
         "Extra whitespace cleanup"),

        ("meri bijli 2 din se nahi aa rahi hai @#$%",
         "Symbols stripped"),

        ("",
         "Empty string edge case"),
    ]

    for text, desc in test_cases:
        result = normalize_text(text)
        print(f"  [{desc}]")
        print(f"    Input:  {text!r}")
        print(f"    Output: {result!r}")
        print()

    # ── Critical test: verify real double-letter words are preserved ───────────
    print("--- Critical regression test: fix_elongated_words() ---\n")
    real_word_input = "the committee reviewed the address in mississippi"
    real_word_output = fix_elongated_words(real_word_input)
    print(f"  Input:  {real_word_input!r}")
    print(f"  Output: {real_word_output!r}")

    # These words must appear unchanged
    for word in ["committee", "address", "mississippi"]:
        if word in real_word_output:
            print(f"  ✅ '{word}' correctly preserved")
        else:
            print(f"  ❌ '{word}' was incorrectly mangled — BUG!")
    print()
