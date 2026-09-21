# InfraLynx CRIMS — NLP Layer
## Steps 1 & 2: Language Detection + Transliteration
**Owner: Aayush**

---

## 0. Where this fits in the full pipeline

```
Citizen text (any language, any script)
        │
   [STEP 1-2: Aayush]  ← THIS DOCUMENT
   Detect language + script → Convert Roman text to native script
        │
   [STEP 3-4: Karthik]
   Translate to English → Normalize text
        │
   [STEP 5-6: Yshasvee]
   Classify department → Extract entities/urgency → Route
        │
   Structured complaint → NestJS /requests
```

Your output feeds directly into Karthik's translation step. So the **exact shape of your output matters** — stick to the contract below.

---

## 1. Your Goal

Write one Python function:

```python
def preprocess(text: str) -> dict
```

**Input:** raw citizen complaint text. Could be:
- English ("water supply has been cut for 3 days")
- Hindi in Devanagari ("मेरी बिजली नहीं आ रही है")
- Hindi typed in English letters ("meri bijli nahi aa rahi hai")
- Same for Tamil and Telugu (native script or Roman letters)

**Output:** a dictionary telling Karthik what language it is, and the text converted to native script if it was typed in Roman letters.

---

## 2. Output contract (must match exactly — Karthik will build on this)

```json
{
  "original_text": "meri bijli 2 din se nahi aa rahi hai",
  "detected_language": "hin",
  "script": "Latn",
  "is_english": false,
  "native_script_text": "मेरी बिजली २ दिन से नहीं आ रही है",
  "confidence": 0.91
}
```

| Field | Meaning |
|---|---|
| `original_text` | exactly what the citizen typed, unchanged |
| `detected_language` | one of: `"hin"`, `"tam"`, `"tel"`, `"eng"`, `"other"` |
| `script` | one of: `"Deva"` (Devanagari), `"Taml"`, `"Telu"`, `"Latn"` (Roman/English letters) |
| `is_english` | `true` if the whole sentence is English |
| `native_script_text` | if script was `"Latn"` and language isn't English → the transliterated native-script version. Otherwise, same as `original_text`. **This is the field Karthik's translator will actually consume.** |
| `confidence` | confidence score from the language detector (0 to 1) |

If you need to change this contract for any reason, tell Karthik and me before changing it — don't change it silently mid-project.

---

## 3. Tools you'll use (both free, open-source, made by AI4Bharat/IIT Madras — built specifically for this exact problem)

1. **IndicLID** — detects language + script (native or Roman) for 22 Indian languages
   - GitHub: https://github.com/AI4Bharat/IndicLID
2. **ai4bharat-transliteration** — converts Roman-typed Indic text to native script
   - PyPI: https://pypi.org/project/ai4bharat-transliteration/

Both are pip-installable, no GPU required, no training needed. You are only doing **inference** (using pre-trained models), not training anything.

---

## 4. One-time setup

```bash
python3 -m venv nlp-env
source nlp-env/bin/activate      # on Windows: nlp-env\Scripts\activate
pip3 install fasttext transformers
pip install ai4bharat-transliteration
```

---

## 5. PART A — Step 1: Language & Script Detection (IndicLID)

### 5.1 Install the model

```bash
git clone https://github.com/AI4Bharat/IndicLID.git
cd IndicLID/Inference
mkdir models
cd models
wget https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.zip
wget https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip
wget https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip
unzip indiclid-bert.zip
unzip indiclid-ftn.zip
unzip indiclid-ftr.zip
cd ..
```

(If `wget` doesn't work on your OS, just paste the URL into a browser and download manually, then unzip into the `models` folder.)

### 5.2 First test — run this exactly, and LOOK at the printed output

```python
from ai4bharat.IndicLID import IndicLID

IndicLID_model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)

test_samples = [
    "meri bijli 2 din se nahi aa rahi hai",
    "मेरी बिजली नहीं आ रही है",
    "enaku kudi thanni moonu naal aayidhu varala",
    "water supply has been cut for 3 days",
]

outputs = IndicLID_model.batch_predict(test_samples, batch_size=1)
print(outputs)
```

**Important:** the exact structure of `outputs` (list of tuples vs list of dicts) can vary slightly by version. Run this first, print it, and look at what you actually get before writing the wrapper function in 5.3. Paste the raw output in the team chat so we all know the exact shape.

### 5.3 Label codes you'll see

IndicLID returns labels like `hin_Deva`, `hin_Latn`, `tam_Taml`, `tam_Latn`, `tel_Telu`, `tel_Latn`, `eng_Latn`, or `other`. The part before `_` is the language, the part after is the script. Relevant ones for us:

| Label | Language | Script |
|---|---|---|
| `hin_Deva` | Hindi | Devanagari (native) |
| `hin_Latn` | Hindi | Roman |
| `tam_Taml` | Tamil | Tamil (native) |
| `tam_Latn` | Tamil | Roman |
| `tel_Telu` | Telugu | Telugu (native) |
| `tel_Latn` | Telugu | Roman |
| `eng_Latn` | English | Roman |
| `other` | anything else / unclear | — |

### 5.4 Write the wrapper function

```python
def detect_language(text: str) -> dict:
    """
    Returns: {"language": "hin"/"tam"/"tel"/"eng"/"other",
              "script": "Deva"/"Taml"/"Telu"/"Latn",
              "confidence": float}
    """
    result = IndicLID_model.batch_predict([text], batch_size=1)[0]
    # Adjust the next lines once you've seen the real structure of `result` from 5.2
    label = result[1]          # e.g. "hin_Latn"  — INDEX MAY DIFFER, CHECK
    confidence = result[2]     # CHECK actual index/key

    if label == "other":
        return {"language": "other", "script": "other", "confidence": confidence}

    lang, script = label.split("_")
    return {"language": lang, "script": script, "confidence": confidence}
```

Test it on all 4 sample sentences from 5.2 and confirm the language+script come back correctly before moving on.

---

## 6. PART B — Step 2: Transliteration (Roman → native script)

You only need this when Step 1 says `script == "Latn"` AND `language != "eng"`.

### 6.1 Install

Already done in section 4 (`pip install ai4bharat-transliteration`). No need to clone a repo or install fairseq — this package is a simpler, ready-to-use wrapper.

### 6.2 Basic usage

```python
from ai4bharat.transliteration import XlitEngine

engine = XlitEngine("hi", beam_width=10, rescore=True)
out = engine.translit_word("namasthe", topk=1)
print(out)
# → {'hi': ['नमस्ते']}
```

`XlitEngine` takes a language code: `"hi"` (Hindi), `"ta"` (Tamil), `"te"` (Telugu).

It transliterates **word by word**, not full sentences directly. So for a full sentence, split into words, transliterate each, and join back:

```python
def transliterate_to_native(text: str, lang_code: str) -> str:
    """
    lang_code: "hi" / "ta" / "te"
    """
    engine = XlitEngine(lang_code, beam_width=10, rescore=True)
    words = text.split()
    converted_words = []
    for word in words:
        result = engine.translit_word(word, topk=1)
        if lang_code in result and result[lang_code]:
            converted_words.append(result[lang_code][0])
        else:
            converted_words.append(word)  # fallback: keep original if it fails
    return " ".join(converted_words)
```

**Performance tip:** creating `XlitEngine(...)` loads a model — don't create it fresh for every word/sentence. Initialize it once (e.g. at module load) and reuse it. Better: initialize all three engines once (`hi`, `ta`, `te`) in a dictionary and pick the right one per request.

```python
engines = {
    "hin": XlitEngine("hi", beam_width=10, rescore=True),
    "tam": XlitEngine("ta", beam_width=10, rescore=True),
    "tel": XlitEngine("te", beam_width=10, rescore=True),
}
```

### 6.3 Test it

Run `transliterate_to_native()` on the Roman-script test sentences from section 5.2 and check the native-script output looks reasonable (doesn't need to be perfect — this is a statistical model, not a dictionary).

---

## 7. Combine Steps 1 + 2 into your final deliverable function

```python
LANG_CODE_MAP = {"hin": "hi", "tam": "ta", "tel": "te"}

def preprocess(text: str) -> dict:
    lang_info = detect_language(text)
    language = lang_info["language"]
    script = lang_info["script"]

    is_english = (language == "eng")

    if script == "Latn" and language in LANG_CODE_MAP:
        native_text = transliterate_to_native(text, LANG_CODE_MAP[language])
    else:
        native_text = text  # already native script, or English, or "other"

    return {
        "original_text": text,
        "detected_language": language,
        "script": script,
        "is_english": is_english,
        "native_script_text": native_text,
        "confidence": lang_info["confidence"],
    }
```

This `preprocess()` function is your final deliverable. Karthik will `import` and call it directly.

---

## 8. Edge cases you must handle

- **Very short text (1-2 words):** language detection is unreliable on very short input. If confidence is low (e.g. below 0.5), still return your best guess but don't worry about perfect accuracy here — flag it in `confidence` and move on.
- **Code-mixed sentences** (Hindi + English words mixed in one sentence, e.g. "meri complaint still pending hai"): IndicLID will give ONE label for the whole sentence. This is a known limitation — don't try to solve it, just document it as a limitation in your section of the report.
- **Numbers/punctuation:** test whether IndicLID handles a sentence with numbers in it (e.g. "2 din se" — "2" is a number). If it errors out, strip numbers/punctuation before passing to the model, but keep the original text intact in `original_text`.
- **Already English:** skip transliteration entirely — `native_script_text = original_text`.
- **`language == "other"`:** pass the text through unchanged, keep confidence low, don't try to transliterate.

---

## 9. Test cases — run all of these, save the output, share with the team

| # | Language | Script | Sample text |
|---|---|---|---|
| 1 | Hindi | Devanagari | मेरी बिजली 2 दिन से नहीं आ रही है |
| 2 | Hindi | Roman | meri bijli 2 din se nahi aa rahi hai |
| 3 | Tamil | Tamil script | என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை |
| 4 | Tamil | Roman | en pakuthiyil 3 naatkalaga thanni illai |
| 5 | Telugu | Telugu script | మా వీధిలో మూడు రోజుల నుండి నీరు లేదు |
| 6 | Telugu | Roman | maa veedhilo moodu rojula nunchi neeru ledu |
| 7 | English | Roman | Water supply has been cut for 3 days |
| 8 | Mixed (bonus, don't need to fix) | Roman | meri complaint abhi tak pending hai |

For each, run `preprocess(text)` and record the full output dict. Send this to me and Karthik once done — I need to know what `native_script_text` looks like before I finalize my classifier's input assumptions, and Karthik needs it to test his translation step.

---

## 10. Suggested folder structure

```
nlp/
├── lang_detect.py       # detect_language() from section 5
├── transliterate.py     # transliterate_to_native() from section 6
├── preprocess.py        # preprocess() from section 7 — imports the above two
└── test_samples.py      # runs section 9's test table, prints results
```

Karthik will do: `from nlp.preprocess import preprocess`

---

## 11. Out of scope for you (don't do these — someone else owns them)

- ❌ Translating native-script text to English → Karthik (Step 3, uses IndicTrans2)
- ❌ Cleaning/normalizing text, fixing typos → Karthik (Step 4)
- ❌ Deciding which department the complaint belongs to → Yshasvee (Step 5)
- ❌ Assigning the complaint to a specific officer → Yshasvee (Step 6)

Stay focused on: detect language/script → convert Roman to native script → return the dict in section 2's exact format.

---

## 12. Checklist

- [x] Virtual environment set up, both packages installed
- [x] IndicLID model files downloaded and unzipped
- [x] `detect_language()` tested on all 4 samples from section 5.2, output structure confirmed
- [x] `transliterate_to_native()` tested and working for hi/ta/te
- [x] `preprocess()` written and matches the exact output contract in section 2
- [x] All 8 test cases from section 9 run, results shared with team
- [x] Code pushed to repo under `nlp/` with this checklist ticked off
