# InfraLynx CRIMS — NLP Layer
## Steps 3 & 4: Translation to English + Normalization
**Owner: Karthik**

---

## 0. Where this fits in the full pipeline

```
Citizen text (any language, any script)
        │
   [STEP 1-2: Aayush]
   Detect language + script → Convert Roman text to native script
        │
   [STEP 3-4: Karthik]  ← THIS DOCUMENT
   Translate to English → Normalize/clean the text
        │
   [STEP 5-6: Yshasvee]
   Classify department → Extract entities/urgency → Route
        │
   Structured complaint → NestJS /requests
```

You receive Aayush's output as input. Your output feeds Yshasvee's classifier. Stick to the contracts below so nobody's code breaks when the pieces are combined.

---

## 1. Your Goal

Write one Python function:

```python
def translate_and_normalize(input_dict: dict) -> dict
```

You take what Aayush's `preprocess()` produces, translate it to clean English, and hand off a clean structured English sentence.

---

## 2. Input you'll receive (from Aayush — already built, don't rebuild it)

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

The field you actually translate is **`native_script_text`** — Aayush guarantees this is always in proper native script (Devanagari/Tamil/Telugu) or already English, never Roman-typed Indic text. If `is_english` is `true`, skip translation entirely and just normalize `original_text`.

---

## 3. Output contract (must match exactly — Yshasvee will build the classifier on this)

```json
{
  "original_text": "meri bijli 2 din se nahi aa rahi hai",
  "detected_language": "hin",
  "english_text": "my electricity has not come for the past 2 days",
  "confidence": 0.91
}
```

| Field | Meaning |
|---|---|
| `original_text` | passed through unchanged from Aayush's input |
| `detected_language` | passed through unchanged from Aayush's input |
| `english_text` | **your main deliverable** — translated (if needed) AND normalized/cleaned English text. This is what Yshasvee's classifier will read. |
| `confidence` | passed through unchanged from Aayush's input |

If you need to change this contract, tell me and Aayush before changing it — don't change it silently.

---

## 4. Tools you'll use

1. **IndicTrans2 (distilled 200M model)** — translates Indic languages → English
   - Model: `ai4bharat/indictrans2-indic-en-dist-200M` on Hugging Face
   - GitHub toolkit: https://github.com/VarunGumma/IndicTransToolkit
   - We're using the **distilled 200M version**, not the full 1B version — it's much lighter and runs fine on CPU, which matters since you likely don't have a GPU. Accuracy is slightly lower than the 1B model but more than good enough for a course project.
2. **spaCy** — for text normalization/cleanup
   - https://spacy.io

---

## 5. One-time setup

```bash
python3 -m venv nlp-env
source nlp-env/bin/activate      # on Windows: nlp-env\Scripts\activate

# For translation
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
pip install --editable ./
cd ..
pip install torch transformers

# For normalization
pip install spacy
python -m spacy download en_core_web_sm
```

You do NOT need a GPU. The code will automatically use CPU if no GPU is found — it'll just be a bit slower (a few seconds per sentence instead of milliseconds). That's fine for a course project.

---

## 6. PART A — Step 3: Translation to English (IndicTrans2)

### 6.1 Load the model (do this once, not per-request)

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
ip = IndicProcessor(inference=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
```

### 6.2 Language code mapping

IndicTrans2 expects specific language codes for the `src_lang` argument. Map Aayush's `detected_language` values to these:

| Aayush's `detected_language` | IndicTrans2 code |
|---|---|
| `"hin"` | `"hin_Deva"` |
| `"tam"` | `"tam_Taml"` |
| `"tel"` | `"tel_Telu"` |

Target is always `"eng_Latn"` (English).

### 6.3 Translation function

```python
LANG_TO_INDICTRANS_CODE = {
    "hin": "hin_Deva",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
}

def translate_to_english(text: str, detected_language: str) -> str:
    src_lang = LANG_TO_INDICTRANS_CODE.get(detected_language)
    if src_lang is None:
        # detected_language was "eng" or "other" — should not reach here,
        # this function is only called for hin/tam/tel. See section 8.
        return text

    tgt_lang = "eng_Latn"

    batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return translations[0]
```

### 6.4 Test it

```python
print(translate_to_english("मेरी बिजली २ दिन से नहीं आ रही है", "hin"))
# expect something like: "My electricity has not come for the past 2 days"
```

Try it on a Tamil and Telugu native-script sentence too before moving on. Translation quality won't be perfect — that's expected, this is a compact model. As long as the department/problem/duration come through clearly, it's good enough for the next step.

---

## 7. PART B — Step 4: Normalization

Translation output can be a bit rough (capitalization inconsistencies, extra spaces, etc.), and English-input complaints from citizens will be informal/typo-ridden. Clean this up before Yshasvee's classifier sees it.

### 7.1 What normalization means here

- Lowercase the text
- Fix extra whitespace
- Expand contractions (`"didn't"` → `"did not"`)
- Fix repeated/elongated characters (`"pleeease"` → `"please"`, `"noooo action"` → `"no action"`)
- Strip stray symbols/emojis (keep basic punctuation like `.` `,` `?`)

### 7.2 Normalization function

**⚠️ UPDATE: install one more package first** — a hand-rolled contraction dictionary misses too many cases. Use a proper library instead:

```bash
pip install contractions
```

```python
import re
import spacy
import contractions

nlp = spacy.load("en_core_web_sm")

def expand_contractions(text: str) -> str:
    return contractions.fix(text)   # handles 100+ contractions automatically

def fix_elongated_words(text: str) -> str:
    """
    Collapse elongated words like "pleeease" -> "please", "noooo" -> "no".
    IMPORTANT: only touch words spaCy doesn't recognize as real English words.
    A naive "collapse any 3+ repeated letter" regex is WRONG — it also wrecks
    genuine words with double/triple letters, e.g. "committee" -> "comite",
    "address" -> "adres", "mississippi" -> "misisipi". Don't use that version.
    """
    words = text.split()
    fixed_words = []
    for word in words:
        if nlp.vocab[word].is_oov:  # not a recognized English word — likely elongated slang
            word = re.sub(r'(.)\1{2,}', r'\1\1', word)  # collapse to max 2 repeats, not 1
        fixed_words.append(word)
    return " ".join(fixed_words)

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = expand_contractions(text)
    text = fix_elongated_words(text)
    text = re.sub(r'[^a-z0-9\s.,?!]', ' ', text)   # strip emojis/symbols, keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()        # collapse extra whitespace
    return text
```

### 7.3 Test it

```python
print(normalize_text("Meri complaint abhi tak pendinggg 😡😡, koi action nahiii!!"))
# expect something like: "meri complaint abhi tak pending, koi action nahi!"

# Also test that real words with double/triple letters survive:
print(fix_elongated_words("the committee reviewed the address in mississippi"))
# must NOT become "comite reviewd the adres in misisipi" — should stay mostly unchanged
```

(The first example tests the cleanup logic on messy text; the second confirms the elongation fix doesn't break real English words. Run both — this was a real bug in an earlier version of this function, so don't skip the second test.)

---

## 8. Combine Steps 3 + 4 into your final deliverable function

```python
def translate_and_normalize(input_dict: dict) -> dict:
    detected_language = input_dict["detected_language"]

    if detected_language == "eng":
        raw_english = input_dict["original_text"]
    elif detected_language in LANG_TO_INDICTRANS_CODE:
        raw_english = translate_to_english(
            input_dict["native_script_text"], detected_language
        )
    else:
        # detected_language == "other" — nothing reliable to translate.
        # Pass the original text through untranslated; flag it via low confidence.
        raw_english = input_dict["original_text"]

    english_text = normalize_text(raw_english)

    return {
        "original_text": input_dict["original_text"],
        "detected_language": detected_language,
        "english_text": english_text,
        "confidence": input_dict["confidence"],
    }
```

This `translate_and_normalize()` function is your final deliverable. Yshasvee will `import` and call it directly.

---

## 9. Edge cases you must handle

- **`detected_language == "eng"`:** skip translation, just normalize.
- **`detected_language == "other"`:** don't attempt translation (the model isn't trained for whatever this is) — pass the text through as-is, so the pipeline doesn't crash. Yshasvee's step can decide how to handle low-confidence/untranslated complaints.
- **Very long complaints:** IndicTrans2 truncates input beyond 256 tokens (`max_length=256` in the code above). Citizen complaints are usually short, so this shouldn't matter, but don't be surprised if a very long paragraph gets cut off.
- **Empty or near-empty text:** guard against calling the model on an empty string — return an empty `english_text` instead of letting it error out.
- **Translation gives an odd/garbled result:** this will happen sometimes since it's a compact 200M model, not the full 1B one. Don't try to "fix" translation errors yourself — just pass through what the model gives you. If it becomes a real problem, we can discuss switching to the 1B model later.

---

## 10. Test cases — run all of these, save the output, share with the team

Ask Aayush to give you his actual `preprocess()` output for these 8 sentences (he has the same list in his document). Run each one through `translate_and_normalize()` and record the result.

| # | Language | Sample (native script, from Aayush) | Expected rough meaning |
|---|---|---|---|
| 1 | Hindi | मेरी बिजली 2 दिन से नहीं आ रही है | my electricity has not come for 2 days |
| 2 | Hindi (was Roman) | मेरी बिजली 2 दिन से नहीं आ रही है | same as above |
| 3 | Tamil | என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை | no water in my area for 3 days |
| 4 | Tamil (was Roman) | same as #3 after transliteration | same as above |
| 5 | Telugu | మా వీధిలో మూడు రోజుల నుండి నీరు లేదు | no water in our street for 3 days |
| 6 | Telugu (was Roman) | same as #5 after transliteration | same as above |
| 7 | English | Water supply has been cut for 3 days | water supply has been cut for 3 days |
| 8 | Mixed (bonus) | meri complaint abhi tak pending hai | (pass-through, known limitation) |

Send the full output dicts to me and Aayush once done — I need to know what real `english_text` looks like before finalizing my classifier's category list and entity extraction rules.

---

## 11. Suggested folder structure

```
nlp/
├── translate.py       # translate_to_english() from section 6
├── normalize.py        # normalize_text() from section 7
├── postprocess.py       # translate_and_normalize() from section 8 — imports the above two
└── test_samples.py      # runs section 10's test table, prints results
```

Yshasvee will do: `from nlp.postprocess import translate_and_normalize`

---

## 12. Out of scope for you (don't do these — someone else owns them)

- ❌ Detecting language/script, converting Roman to native script → Aayush (Steps 1-2)
- ❌ Deciding which department the complaint belongs to → Yshasvee (Step 5)
- ❌ Extracting duration/location/previous-complaint entities → Yshasvee (Step 5)
- ❌ Estimating urgency → Yshasvee (Step 5)
- ❌ Assigning the complaint to a specific officer → Yshasvee (Step 6)

Stay focused on: translate native-script text to English → clean/normalize it → return the dict in section 3's exact format.

---

## 13. Checklist

- [ ] Virtual environment set up, IndicTransToolkit + transformers + spaCy installed
- [ ] IndicTrans2 distilled model downloads and loads successfully (first run will download ~800MB-1GB, be patient)
- [ ] `translate_to_english()` tested on Hindi, Tamil, and Telugu native-script sentences
- [ ] `normalize_text()` tested and cleaning output correctly — **including the "committee/address/mississippi" test in 7.3**, confirming real words aren't mangled
- [ ] `translate_and_normalize()` written and matches the exact output contract in section 3
- [ ] All 8 test cases from section 10 run (using Aayush's actual outputs), results shared with team
- [ ] Code pushed to repo under `nlp/` with this checklist ticked off
