"""
Test suite for Steps 1 & 2: Language Detection + Transliteration
Owner: Aayush

Runs all 8 test cases from Section 9 of Steps_1-2_Plan_for_Aayush.md.
"""
import json
import sys
from preprocess import preprocess

TEST_CASES = [
    {
        "id": 1,
        "language": "Hindi",
        "script": "Devanagari",
        "sample_text": "मेरी बिजली 2 दिन से नहीं आ रही है",
        "expected_lang": "hin",
        "expected_script": "Deva",
        "is_english": False,
    },
    {
        "id": 2,
        "language": "Hindi",
        "script": "Roman",
        "sample_text": "meri bijli 2 din se nahi aa rahi hai",
        "expected_lang": "hin",
        "expected_script": "Latn",
        "is_english": False,
    },
    {
        "id": 3,
        "language": "Tamil",
        "script": "Tamil script",
        "sample_text": "என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",
        "expected_lang": "tam",
        "expected_script": "Taml",
        "is_english": False,
    },
    {
        "id": 4,
        "language": "Tamil",
        "script": "Roman",
        "sample_text": "en pakuthiyil 3 naatkalaga thanni illai",
        "expected_lang": "tam",
        "expected_script": "Latn",
        "is_english": False,
    },
    {
        "id": 5,
        "language": "Telugu",
        "script": "Telugu script",
        "sample_text": "మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",
        "expected_lang": "tel",
        "expected_script": "Telu",
        "is_english": False,
    },
    {
        "id": 6,
        "language": "Telugu",
        "script": "Roman",
        "sample_text": "maa veedhilo moodu rojula nunchi neeru ledu",
        "expected_lang": "tel",
        "expected_script": "Latn",
        "is_english": False,
    },
    {
        "id": 7,
        "language": "English",
        "script": "Roman",
        "sample_text": "Water supply has been cut for 3 days",
        "expected_lang": "eng",
        "expected_script": "Latn",
        "is_english": True,
    },
    {
        "id": 8,
        "language": "Mixed (bonus)",
        "script": "Roman",
        "sample_text": "meri complaint abhi tak pending hai",
        "expected_lang": "hin",
        "expected_script": "Latn",
        "is_english": False,
    },
]

def separator(title: str):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")

def run_tests():
    separator("InfraLynx CRIMS NLP — Steps 1-2 Test Suite (Aayush)")
    results = []

    for test in TEST_CASES:
        output = preprocess(test["sample_text"])
        results.append(output)
        lang_match = output["detected_language"] == test["expected_lang"]
        script_match = output["script"] == test["expected_script"]
        status = "✅ PASS" if (lang_match and script_match) else "⚠️ CHECK"

        print(f"\nTest #{test['id']}: [{test['language']} - {test['script']}] {status}")
        print(f"  Original:           \"{output['original_text']}\"")
        print(f"  Detected Language:  {output['detected_language']} (expected: {test['expected_lang']})")
        print(f"  Detected Script:    {output['script']} (expected: {test['expected_script']})")
        print(f"  Is English:         {output['is_english']}")
        print(f"  Confidence:         {output['confidence']}")
        print(f"  Native Script Text: \"{output['native_script_text']}\"")

    separator("FULL JSON OUTPUTS FOR KARTHIK & TEAM")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    return results

if __name__ == "__main__":
    run_tests()
