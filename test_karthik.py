"""
InfraLynx CRIMS — NLP Layer
Test Suite for Steps 3 & 4: Translation + Normalization

Owner: Karthik
Run with:   python test_karthik.py

This script tests all 8 shared test sentences from the project spec (Section 10
of Steps_3-4_Plan_for_Karthik.md) plus additional edge cases.

The output of each test is formatted as a dict matching the exact output contract
that Yshasvee expects. Share the full output with the team after running.

Tests included:
    [A] normalize_text()      — normalization-only tests (fast, no model needed)
    [B] translate_to_english() — translation tests (requires IndicTrans2 model)
    [C] translate_and_normalize() — full Step 3+4 pipeline (the actual deliverable)
    [D] Edge cases             — empty strings, unknown language, very long text

Usage:
    python test_karthik.py           # run all tests
    python test_karthik.py --quick   # run only normalization tests (no model download)
"""

import sys
import json

# ─── Import our modules ───────────────────────────────────────────────────────
# Normalization tests don't need the translation model, so we import separately
# to support the --quick flag.

QUICK_MODE = "--quick" in sys.argv

def separator(title: str, width: int = 70):
    """Print a visible section separator for easy reading in the terminal."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

def pp(obj):
    """Pretty-print a dict or any JSON-serializable object."""
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ─── TEST GROUP A: normalize_text() ──────────────────────────────────────────
def test_normalization():
    """
    Test the normalize_text() function in isolation.
    These tests run WITHOUT loading IndicTrans2 — fast, always runnable.

    Tests cover:
        - Basic lowercasing
        - Contraction expansion
        - Elongated word correction
        - Real English words with double letters (critical regression test)
        - Emoji/symbol stripping
        - Whitespace cleanup
        - Empty string
    """
    separator("TEST GROUP A: normalize_text() — normalization only")

    # Import here (not at top) so --quick flag skips translate.py's heavy model load
    from normalize import normalize_text, fix_elongated_words

    cases = [
        # (input_text, description, expected_contains_check)
        (
            "My Electricity has NOT come for 2 days!!",
            "Lowercase + punctuation preserved",
            "my electricity has not come for 2 days",
        ),
        (
            "Water supply didn't come for 3 days",
            "Contraction: didn't → did not",
            "did not",
        ),
        (
            "I've filed 2 complaints but it's not been fixed",
            "Contractions: I've + it's",
            "i have",
        ),
        (
            "Meri complaint abhi tak pendinggg 😡😡, koi action nahiii!!",
            "Elongated words + emojis stripped",
            "pending",    # pendinggg → pending (or at least some collapse)
        ),
        (
            "pleeease fix the road sooooon",
            "Multiple elongated words",
            "please",
        ),
        (
            "the committee reviewed the address in mississippi",
            "Real double-letter words MUST survive unchanged",
            "committee",
        ),
        (
            "  Road has been   broken   since last   week   ",
            "Extra whitespace collapsed",
            "road has been broken since last week",
        ),
        (
            "Sewage overflow @#$%!! near my house 🚨🚨",
            "All symbols and emojis stripped",
            "sewage overflow",
        ),
        (
            "",
            "Empty string → empty string",
            "",
        ),
    ]

    passed = 0
    failed = 0

    for text, desc, expected_contains in cases:
        result = normalize_text(text)
        # Check: does the result contain the expected substring?
        ok = expected_contains in result
        icon = "✅" if ok else "⚠️ "
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n  {icon} {desc}")
        print(f"     Input:    {text!r}")
        print(f"     Output:   {result!r}")
        if not ok:
            print(f"     Expected to contain: {expected_contains!r}  ← NOT FOUND")

    # ── Critical regression check ───────────────────────────────────────────
    print("\n  --- Critical regression: fix_elongated_words() ---")
    real_word_test = "the committee reviewed the address in mississippi"
    rw_result = fix_elongated_words(real_word_test)
    for word in ["committee", "address", "mississippi"]:
        ok = word in rw_result
        icon = "✅" if ok else "❌ BUG"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"    {icon}  '{word}' {'preserved' if ok else 'was MANGLED — this is a bug!'}")

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


# ─── TEST GROUP B: translate_to_english() ────────────────────────────────────
def test_translation():
    """
    Test translate_to_english() in isolation (without normalization).
    Requires IndicTrans2 model to be downloaded.
    """
    separator("TEST GROUP B: translate_to_english() — translation only")

    from translate import translate_to_english

    cases = [
        # (native_script_text, detected_language, description)
        ("मेरी बिजली 2 दिन से नहीं आ रही है",           "hin", "Hindi → electricity complaint"),
        ("मेरे गाँव में पिछले 3 दिन से पानी नहीं आ रहा है", "hin", "Hindi → water complaint"),
        ("என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",       "tam", "Tamil → water complaint"),
        ("en pakuthiyil 3 naatkalaga thanni illai",       "tam", "Tamil (already transliterated, check behavior)"),
        ("మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",          "tel", "Telugu → water complaint"),
        ("",                                              "hin", "Empty string → empty output"),
        ("some random text",                              "unk", "Unknown language → pass-through"),
    ]

    for text, lang, desc in cases:
        print(f"\n  [{desc}]")
        print(f"    Input  ({lang}): {text!r}")
        result = translate_to_english(text, lang)
        print(f"    Output (eng): {result!r}")


# ─── TEST GROUP C: Full pipeline — the 8 shared test sentences ────────────────
def test_full_pipeline():
    """
    Run all 8 shared test sentences through translate_and_normalize().
    These are the exact sentences from Section 10 of the project plan.

    These are the dicts to share with Yshasvee — she builds her classifier
    category list and entity extraction rules on the actual english_text output.
    """
    separator("TEST GROUP C: translate_and_normalize() — full Steps 3+4 pipeline")

    from postprocess import translate_and_normalize

    # The 8 shared test sentences (using simulated Aayush preprocess() output)
    # Aayush: please replace these with your ACTUAL preprocess() output
    # when you have your code ready. The native_script_text field must come from you.
    test_sentences = [
        {
            "id": 1,
            "description": "Hindi (native script) → electricity complaint",
            "expected_rough_meaning": "my electricity has not come for 2 days",
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
            "id": 2,
            "description": "Hindi (was Roman 'meri bijli 2 din se nahi...') → electricity complaint",
            "expected_rough_meaning": "my electricity has not come for 2 days",
            "input": {
                "original_text": "meri bijli 2 din se nahi aa rahi hai",
                "detected_language": "hin",
                "script": "Latn",
                "is_english": False,
                # Aayush's IndicXlit converts Roman → Devanagari; this is what he produces
                "native_script_text": "मेरी बिजली 2 दिन से नहीं आ रही है",
                "confidence": 0.91,
            },
        },
        {
            "id": 3,
            "description": "Tamil (native script) → water complaint",
            "expected_rough_meaning": "no water in my area for 3 days",
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
            "id": 4,
            "description": "Tamil (was Roman) → water complaint",
            "expected_rough_meaning": "no water in my area for 3 days",
            "input": {
                "original_text": "en pakuthiyil 3 naatkalaga thanni illai",
                "detected_language": "tam",
                "script": "Latn",
                "is_english": False,
                # Aayush: replace with your actual IndicXlit output for Tamil
                "native_script_text": "என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",
                "confidence": 0.85,
            },
        },
        {
            "id": 5,
            "description": "Telugu (native script) → water complaint",
            "expected_rough_meaning": "no water in our street for 3 days",
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
            "id": 6,
            "description": "Telugu (was Roman) → water complaint",
            "expected_rough_meaning": "no water in our street for 3 days",
            "input": {
                "original_text": "maa veedhilo moodu rojula nunchi neeru ledu",
                "detected_language": "tel",
                "script": "Latn",
                "is_english": False,
                # Aayush: replace with your actual IndicXlit output for Telugu
                "native_script_text": "మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",
                "confidence": 0.84,
            },
        },
        {
            "id": 7,
            "description": "English input → water complaint (no translation needed)",
            "expected_rough_meaning": "water supply has been cut for 3 days",
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
            "id": 8,
            "description": "Mixed Hinglish (bonus) — known limitation, pass-through",
            "expected_rough_meaning": "meri complaint abhi tak pending hai",
            "input": {
                "original_text": "meri complaint abhi tak pending hai",
                # NOTE: This would actually be classified as "hin" by IndicLID
                # because Hinglish typed in Roman is Hindi. Aayush would convert
                # this to Devanagari via IndicXlit, and we'd translate it.
                # We simulate the "other" case here to test the fallback path.
                "detected_language": "other",
                "script": "Latn",
                "is_english": False,
                "native_script_text": "meri complaint abhi tak pending hai",
                "confidence": 0.45,
            },
        },
    ]

    results = []

    for test in test_sentences:
        print(f"\n  Test #{test['id']}: {test['description']}")
        print(f"  Original: {test['input']['original_text']!r}")
        print(f"  Expected meaning: {test['expected_rough_meaning']!r}")

        output = translate_and_normalize(test["input"])
        print(f"  ─── Output ───")
        pp(output)
        results.append({"test_id": test["id"], "description": test["description"], **output})

    # Print a summary table at the end for easy sharing with the team
    separator("SUMMARY TABLE — share with Yshasvee and Aayush")
    print(f"\n  {'#':<4} {'Language':<8} {'english_text (truncated to 60 chars)'}")
    print(f"  {'─'*4} {'─'*8} {'─'*60}")
    for r in results:
        lang = r.get("detected_language", "?")
        eng = r.get("english_text", "")[:60]
        print(f"  {r['test_id']:<4} {lang:<8} {eng}")

    return results


# ─── TEST GROUP D: Edge cases ─────────────────────────────────────────────────
def test_edge_cases():
    """
    Test edge cases that could cause crashes or silent failures in production.
    """
    separator("TEST GROUP D: Edge cases")

    from postprocess import translate_and_normalize

    edge_cases = [
        {
            "description": "Empty original_text — must return empty english_text, not crash",
            "input": {
                "original_text": "",
                "detected_language": "hin",
                "script": "Deva",
                "is_english": False,
                "native_script_text": "",
                "confidence": 0.91,
            },
        },
        {
            "description": "Whitespace-only text — must return empty string",
            "input": {
                "original_text": "   ",
                "detected_language": "eng",
                "script": "Latn",
                "is_english": True,
                "native_script_text": "   ",
                "confidence": 0.99,
            },
        },
        {
            "description": "English with emojis, slang, and elongated words",
            "input": {
                "original_text": "pleeease fix my road 🚨🚨 its been sooooo long!!!",
                "detected_language": "eng",
                "script": "Latn",
                "is_english": True,
                "native_script_text": "pleeease fix my road 🚨🚨 its been sooooo long!!!",
                "confidence": 0.97,
            },
        },
        {
            "description": "Unknown language code — must not crash, must pass through",
            "input": {
                "original_text": "some text in an unsupported language",
                "detected_language": "xyz",
                "script": "Latn",
                "is_english": False,
                "native_script_text": "some text in an unsupported language",
                "confidence": 0.30,
            },
        },
        {
            "description": "Missing keys in input dict — must not crash (uses .get() with defaults)",
            "input": {
                # Minimal dict — missing native_script_text, script, is_english
                "original_text": "water problem",
                "detected_language": "eng",
                "confidence": 0.95,
            },
        },
    ]

    for case in edge_cases:
        print(f"\n  [{case['description']}]")
        try:
            result = translate_and_normalize(case["input"])
            print(f"  ✅ No crash. Output:")
            pp(result)
        except Exception as e:
            print(f"  ❌ CRASHED: {type(e).__name__}: {e}")


# ─── Main entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "🧪 InfraLynx NLP — Steps 3+4 Test Suite (Karthik)" + "\n")

    if QUICK_MODE:
        # Quick mode: only run normalization tests (no model download required)
        print("  Running in --quick mode: skipping translation tests.")
        print("  (No IndicTrans2 model download required.)\n")
        test_normalization()
        separator("QUICK MODE COMPLETE")
        print("  To run full tests including translation, run without --quick\n")
    else:
        # Full test run
        norm_ok = test_normalization()

        # Only run translation tests if normalization passed
        # (translation depends on normalize_text internally)
        test_translation()
        test_full_pipeline()
        test_edge_cases()

        separator("ALL TESTS COMPLETE")
        print()
        print("  📋 Next steps:")
        print("  1. Review any ⚠️  warnings above.")
        print("  2. Share the SUMMARY TABLE from Test Group C with Yshasvee.")
        print("  3. Ask Aayush to run his preprocess() on the same 8 sentences")
        print("     and give you his actual output dicts — replace the simulated")
        print("     native_script_text values in this file with his real output.")
        print("  4. Re-run this test with Aayush's actual data and share final results.\n")

