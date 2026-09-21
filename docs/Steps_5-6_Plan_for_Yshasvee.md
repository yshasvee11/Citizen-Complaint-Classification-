# InfraLynx CRIMS — NLP Layer
## Steps 5 & 6: Classification, Entity Extraction, Urgency, Routing
**Owner: Yshasvee (you)**

*(Incorporates review feedback — see section 12 for what changed and why.)*

---

## 0. Where this fits in the full pipeline

```
Citizen text (any language, any script)
        │
   [STEP 1-2: Aayush]
   Detect language + script → Convert Roman text to native script
        │
   [STEP 3-4: Karthik]
   Translate to English → Normalize/clean the text
        │
   [STEP 5-6: You]  ← THIS DOCUMENT
   Classify department → Extract entities → Estimate urgency → Route
        │
   Structured complaint → NestJS /requests
```

By the time text reaches you, it's already clean English. You don't need any multilingual/Indic-specific models — everything here is standard English NLP.

---

## 1. Your Goal

Write one function:

```python
def classify_and_route(input_dict: dict) -> dict
```

This produces the final structured complaint object that gets attached to the request in InfraLynx CRIMS. You will also own the single FastAPI endpoint that chains all three teammates' work together (section 11) — since your step is last before the result goes back to NestJS.

---

## 2. Input you'll receive (from Karthik)

```json
{
  "original_text": "meri bijli 2 din se nahi aa rahi hai",
  "detected_language": "hin",
  "english_text": "my electricity has not come for the past 2 days",
  "confidence": 0.91
}
```

You work off `english_text`.

---

## 3. Output contract (this is what gets attached to the InfraLynx request)

```json
{
  "original_text": "meri bijli 2 din se nahi aa rahi hai",
  "detected_language": "hin",
  "english_text": "my electricity has not come for the past 2 days",
  "department": "Electricity",
  "department_confidence": 0.87,
  "needs_manual_review": false,
  "entities": {
    "duration": "2 days",
    "previous_complaint": false,
    "location": null
  },
  "urgency": {
    "score": 2.0,
    "label": "Medium"
  }
}
```

New field vs. before: **`needs_manual_review`** — set `true` when the classifier isn't confident (see section 6.3). This is what your Python service returns to NestJS, which attaches it to the request before it enters the existing `RECEIVED → UNDER_REVIEW → ...` workflow.

Which specific *officer* gets assigned is a separate concern — see section 9. Your contract's job stops at `department`, `entities`, `urgency`.

---

## 4. Tools you'll use

1. **Zero-shot text classification** — `facebook/bart-large-mnli` via HuggingFace `transformers`. (If your machine is CPU-only and this feels slow, `typeform/distilbart-mnli-12-3` is a 3x smaller drop-in replacement with the same API.)
2. **spaCy** — for entity extraction (DATE/TIME and location), same package Karthik already installed.
3. **Plain regex + a keyword table** — for urgency scoring. (Not VADER/sentiment — see section 8 for why.)

---

## 5. One-time setup

```bash
python3 -m venv nlp-env
source nlp-env/bin/activate      # on Windows: nlp-env\Scripts\activate

pip install transformers torch
pip install spacy
python -m spacy download en_core_web_sm
```

---

## 6. PART A — Step 5a: Department Classification

### 6.1 Sync your category list first

Your candidate labels must match InfraLynx's actual `serviceCategories` from `back-end/src/data/seed.data.ts`. Pull the real list before finalizing — below is a placeholder based on your project plan's examples:

```python
DEPARTMENTS = [
    "Electricity",
    "Water Supply",
    "Roads",
    "Sanitation",
    "Street Lights",
    "Drainage",
    # replace/extend with the exact names from seed.data.ts's serviceCategories
]
```

### 6.2 Load the classifier (once, not per-request)

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# CPU-only and it's too slow? swap the model string for:
# classifier = pipeline("zero-shot-classification", model="typeform/distilbart-mnli-12-3")
```

### 6.3 Classification function — with a confidence threshold

A zero-shot classifier **always** picks something, even for garbage input like `"asdfghjkl"`. Without a minimum-confidence check, you'll silently mis-route nonsense or ambiguous complaints. Add a threshold:

```python
CONFIDENCE_THRESHOLD = 0.4

def classify_department(text: str, candidate_labels: list = DEPARTMENTS) -> dict:
    result = classifier(text, candidate_labels)
    top_label = result["labels"][0]
    top_score = round(result["scores"][0], 2)

    if top_score < CONFIDENCE_THRESHOLD:
        return {"department": "Unclassified", "confidence": top_score, "needs_manual_review": True}

    return {"department": top_label, "confidence": top_score, "needs_manual_review": False}
```

### 6.4 Test it

```python
print(classify_department("my electricity has not come for the past 2 days"))
# expect: {"department": "Electricity", "confidence": <high>, "needs_manual_review": False}

print(classify_department("asdfghjkl qwerty"))
# expect: {"department": "Unclassified", "confidence": <low>, "needs_manual_review": True}
```

Try it on all your department categories with a couple of example sentences each — zero-shot classifiers can be confidently wrong on ambiguous text, so check a handful of cases manually, not just the happy path.

---

## 7. PART B — Step 5b: Entity Extraction

spaCy's built-in NER plus a couple of regex/keyword checks is enough here — no separate model needed.

### 7.1 Duration and location via spaCy NER, with a regex fallback for duration

```python
import re
import spacy

nlp = spacy.load("en_core_web_sm")

DURATION_REGEX = re.compile(r'(\d+)\s*(days?|hours?|weeks?|months?)', re.IGNORECASE)

def extract_duration_and_location(text: str) -> dict:
    doc = nlp(text)
    duration = None
    location = None

    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME") and duration is None:
            duration = ent.text
        elif ent.label_ in ("GPE", "LOC", "FAC") and location is None:
            location = ent.text

    # Regex fallback if spaCy's NER missed the duration
    if duration is None:
        match = DURATION_REGEX.search(text)
        if match:
            duration = match.group(0)

    return {"duration": duration, "location": location}
```

Heads up: spaCy's English NER was trained on general text, not civic complaints. It'll catch named places reasonably well ("Marikina", "MG Road") but will usually miss vague references like "my area" or "our street" — those aren't named entities. Document this as a known limitation, don't try to fix it for a course project.

### 7.2 Previous-complaint indicator (keyword check — not something spaCy NER covers)

```python
PREVIOUS_COMPLAINT_PHRASES = [
    "already complained", "complained before", "previous complaint",
    "no action", "again", "still not resolved", "reported earlier", "still pending",
]

def detect_previous_complaint(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in PREVIOUS_COMPLAINT_PHRASES)
```

### 7.3 Combine

```python
def extract_entities(text: str) -> dict:
    result = extract_duration_and_location(text)
    result["previous_complaint"] = detect_previous_complaint(text)
    return result
```

### 7.4 Test it

```python
print(extract_entities("my electricity has not come for the past 2 days and no action was taken after I complained"))
# expect: {"duration": "2 days" (or similar), "location": None, "previous_complaint": True}
```

---

## 8. PART C — Step 5c: Urgency Estimation

**Design change from the original plan:** don't use sentiment analysis (e.g. VADER) for urgency. Sentiment measures *emotional tone* ("I'm furious"), not *actual severity* ("gas leak"). A calmly worded gas leak report and an angrily worded minor pothole complaint should NOT get the same urgency ranking that sentiment analysis would give them. Use a deterministic, explainable keyword scorer instead — it's also much easier to justify in your report/demo than a black-box sentiment score.

### 8.1 Weighted keyword scorer

```python
URGENCY_SIGNALS = {
    # Critical — safety/life-threatening
    "fire": 3, "flood": 3, "collapse": 3, "gas leak": 3, "electric shock": 3,
    # High — service-down / health-risk
    "no water": 2, "no electricity": 2, "sewage": 2, "overflow": 2,
    # Contributing factors — push score up, don't decide it alone
    "week": 1.5, "month": 2, "no action": 1.5, "complained before": 2,
}

def score_urgency(text: str) -> float:
    text_lower = text.lower()
    return sum(weight for keyword, weight in URGENCY_SIGNALS.items() if keyword in text_lower)

def urgency_label(score: float) -> str:
    if score >= 3:
        return "Critical"
    elif score >= 2:
        return "High"
    elif score >= 1:
        return "Medium"
    else:
        return "Low"

def estimate_urgency(text: str) -> dict:
    score = score_urgency(text)
    return {"score": score, "label": urgency_label(score)}
```

### 8.2 Test it

```python
print(estimate_urgency("there is a gas leak near my house"))
# expect: {"score": 3, "label": "Critical"}

print(estimate_urgency("my street light has been off for a week"))
# expect: {"score": 1.5, "label": "Medium"}
```

### 8.3 Tune the keyword list against your real test sentences

The exact keywords/weights above are a starting point — once you have Karthik's actual translated outputs for your test set, check whether real complaints trigger sensible scores, and adjust the dictionary. Keep the scoring logic itself unchanged; just tune the dictionary contents.

---

## 9. PART D — Step 6: Routing

### 9.1 Department routing (yours)

Once you have `department` from section 6, that's your routing output. Make sure the string matches an actual entry in InfraLynx's `departments` collection — coordinate with whoever owns the backend seed data.

### 9.2 Assigning a specific officer (NOT an NLP task)

Deciding *which specific* Department Officer handles a request is a backend business rule, not something NLP predicts — it needs to read/write live `officialAccounts` state, so it belongs in NestJS, not your Python service:

```javascript
// Illustrative only — this goes in NestJS, not your Python code
function assignOfficer(department, officialAccounts) {
  const officersInDept = officialAccounts.filter(o => o.department === department);
  const nextOfficer = officersInDept[roundRobinCounter[department] % officersInDept.length];
  roundRobinCounter[department]++;
  return nextOfficer;
}
```

Your Python service's job stops at outputting `department`. Flag this split clearly in your report so it's obvious the officer-assignment step is deliberately kept out of the NLP layer.

---

## 10. Combine everything into your final deliverable function

```python
def classify_and_route(input_dict: dict) -> dict:
    text = input_dict["english_text"]

    dept_result = classify_department(text)
    entities = extract_entities(text)
    urgency = estimate_urgency(text)

    return {
        "original_text": input_dict["original_text"],
        "detected_language": input_dict["detected_language"],
        "english_text": text,
        "department": dept_result["department"],
        "department_confidence": dept_result["confidence"],
        "needs_manual_review": dept_result["needs_manual_review"],
        "entities": entities,
        "urgency": urgency,
    }
```

---

## 11. The single FastAPI endpoint (you own this — it's the team's integration point)

Expose **exactly one** endpoint for NestJS to call. Internally it chains all three of your team's functions. Add a startup warm-up (models take 30–90 seconds to load on first use — you don't want the *first real citizen request* to eat that delay), basic timing per step, and a batch endpoint for demoing all 8 test cases at once.

```python
import time
from fastapi import FastAPI
from nlp.preprocess import preprocess                  # Aayush
from nlp.postprocess import translate_and_normalize     # Karthik
from nlp.classify_route import classify_and_route         # You

app = FastAPI()

def full_pipeline(text: str, debug: bool = False) -> dict:
    timings = {}

    t0 = time.time()
    step_1_2 = preprocess(text)
    timings["detection_transliteration_ms"] = round((time.time() - t0) * 1000)

    t1 = time.time()
    step_3_4 = translate_and_normalize(step_1_2)
    timings["translation_normalization_ms"] = round((time.time() - t1) * 1000)

    t2 = time.time()
    step_5_6 = classify_and_route(step_3_4)
    timings["classification_routing_ms"] = round((time.time() - t2) * 1000)

    step_5_6["processing_time_ms"] = round((time.time() - t0) * 1000)
    if debug:
        step_5_6["pipeline_log"] = timings

    return step_5_6

@app.on_event("startup")
async def warmup():
    full_pipeline("test sentence for warming up")
    print("✅ All models loaded")

@app.post("/analyze")
def analyze(payload: dict, debug: bool = False):
    return full_pipeline(payload["text"], debug=debug)

@app.post("/analyze/batch")
def analyze_batch(payload: dict):
    # payload = {"texts": ["...", "...", ...]}
    return [full_pipeline(t) for t in payload["texts"]]
```

Run with:

```bash
pip install fastapi uvicorn
uvicorn main:app --port 8001
```

NestJS calls `POST http://localhost:8001/analyze` with `{"text": "<raw citizen complaint>"}` and gets the structured object back. Adding `?debug=true` to the URL also returns a `pipeline_log` showing per-step timing — useful to show in your demo ("processed in 1.2s — 45ms detection, 890ms translation, 180ms classification").

**Tell your backend teammate (whoever owns the NestJS side) this important point:** if this Python service is down or times out, complaint submission should NOT break entirely. NestJS should call this endpoint with a ~5 second timeout, and if it fails, still create the request with `department: "Unclassified"` so a human can route it manually. Don't let the NLP layer become a single point of failure for the whole app.

---

## 12. What changed from the original plan, and why

| Change | Reason |
|---|---|
| Urgency: keyword scorer instead of VADER sentiment | Sentiment measures tone, not severity. "Gas leak" said calmly is still critical; a minor complaint said angrily isn't. A deterministic scorer is also easier to explain/defend in a demo than a sentiment black box. |
| Added `needs_manual_review` + confidence threshold on classification | Zero-shot classification always picks *something*, even for nonsense input. Without a floor, garbage-in silently becomes garbage-out (wrong department). |
| Entity extraction uses spaCy's DATE/TIME + GPE/LOC tags directly, regex only as fallback | Simpler and reuses spaCy's built-in NER rather than hand-writing everything from scratch. |
| Single `/analyze` endpoint + startup warm-up + `processing_time_ms`/debug log | One clean contract for NestJS to call; avoids the first real request eating a 30-90s model load; timing breakdown is a strong, concrete thing to show in a demo. |
| Explicit note on NestJS-side timeout + fallback | If your Python service crashes or is slow, the whole citizen complaint flow shouldn't break. This is a backend reliability point, flag it to whoever owns NestJS integration. |
| Officer assignment stays out of Python, explicitly | Confirmed as the right call — it's live in-memory state (`officialAccounts`), which is NestJS's job, not an NLP prediction. |

---

## 13. Edge cases you must handle

- **Low classification confidence:** now handled via `needs_manual_review` (section 6.3) — don't silently auto-route these.
- **`detected_language == "other"` complaints:** unreliable classification/urgency on whatever text came through untranslated — this will usually also trip the confidence threshold and get flagged for manual review, which is the right outcome.
- **Empty `english_text`:** guard against running the classifier on an empty string.
- **Multiple valid departments:** a complaint could plausibly span two categories (e.g. a broken pipe causing water logging on a road). Zero-shot only gives you the top-1 label. If you want to go further: return the top-2 labels and flag for manual review when the gap between them is small (e.g. < 0.15) — otherwise, document it as a known limitation and move on.

---

## 14. Test cases — run all of these, save the output

Get Karthik's actual `translate_and_normalize()` output for the shared 8 test sentences, run each through `classify_and_route()`, and confirm department/entities/urgency look sensible.

| # | Text (from Karthik, English) | Expected department | Expected duration | Expected urgency |
|---|---|---|---|---|
| 1-2 | my electricity has not come for 2 days | Electricity | 2 days | Low-Medium |
| 3-4 | no water in my area for 3 days | Water Supply | 3 days | High (matches "no water") |
| 5-6 | no water in our street for 3 days | Water Supply | 3 days | High |
| 7 | water supply has been cut for 3 days | Water Supply | 3 days | Low-Medium (doesn't hit "no water" exactly — check if you need to add "cut" as a keyword) |
| 8 | my complaint is still pending | Unclassified (low confidence — flag for review) | None | Medium (hits "still pending" if you added it to the phrase list) |

---

## 15. Suggested folder structure

```
nlp/
├── classify.py           # classify_department() from section 6
├── entities.py            # extract_entities() from section 7
├── urgency.py               # estimate_urgency() from section 8
├── classify_route.py         # classify_and_route() from section 10 — imports the above three
├── main.py                     # FastAPI app from section 11 — chains all three team members' work
└── test_samples.py               # runs section 14's test table, prints results
```

---

## 16. Team-level items — not yours alone, but you should raise them

These affect the whole project, not just steps 5-6. Bring them up with the team:

- **`docker-compose.yml`** for one-command startup (NestJS backend + Python NLP service together). Makes examiner demos much smoother.
- **One `nlp/requirements.txt`** with pinned versions, shared by all three of you, so everyone's environment behaves identically. Add `nlp-env/` to `.gitignore`.
- **A simple demo UI**: a text box + submit button + results panel (can be served directly from FastAPI). Typing a Tamil complaint and watching it get detected → transliterated → translated → classified live is the single highest-impact thing for your presentation — worth the ~30 minutes it takes to build.
- **Document known limitations honestly** in your report — examiners respect this more than pretending the system is flawless:
  - Code-mixed sentences get one language label (not split per-word)
  - The 200M translation model isn't perfect, but key nouns (water/electricity/road) usually survive, which is what classification depends on
  - Transliteration can mangle numbers/proper nouns — `original_text` is preserved so nothing is lost
  - No learning from admin corrections — flag as future work

---

## 17. Checklist

- [ ] Real `serviceCategories` list pulled from `seed.data.ts`, `DEPARTMENTS` updated to match
- [ ] `classify_department()` tested, including the confidence-threshold/`Unclassified` case
- [ ] `extract_entities()` tested — duration, location, previous-complaint all working
- [ ] `estimate_urgency()` tested and tuned against real translated test sentences
- [ ] `classify_and_route()` written and matches the output contract in section 3
- [ ] Officer-assignment split (section 9.2) discussed with whoever owns NestJS
- [ ] Single `/analyze` FastAPI endpoint built, with startup warmup and `processing_time_ms`
- [ ] `/analyze/batch` endpoint added for demoing all 8 test cases at once
- [ ] NestJS-side timeout + `"Unclassified"` fallback discussed with backend owner
- [ ] All 8 test cases from section 14 run, results recorded for the report/demo
- [ ] `docker-compose.yml`, shared `requirements.txt`, and demo UI discussed/assigned within the team
