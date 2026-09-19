# InfraLynx CRIMS — NLP Layer

This is the standalone NLP pipeline project repository. It exposes a FastAPI endpoint to chain the outputs of three different components.

## Team Structure

**1. Steps 1 & 2: Language Detection + Transliteration (Owner: Aayush)**
* **Files to edit:** `lang_detect.py`, `transliterate.py`, `preprocess.py`
* **Goal:** Detect language and script. If roman text is provided for indicative languages, convert it to native script.
* **Input:** Raw citizen text.
* **Output:** JSON object with detected language and native script text.

**2. Steps 3 & 4: Translation + Normalization (Owner: Karthik)**
* **Files to edit:** `translate.py`, `normalize.py`, `postprocess.py`
* **Goal:** Translate native text to English and clean up the text.
* **Input:** Output from Aayush's `preprocess.py`.
* **Output:** JSON object with `english_text`.

**3. Steps 5 & 6: Classification, Entity Extraction, Urgency (Owner: Yshasvee)**
* **Files to edit:** `classify.py`, `entities.py`, `urgency.py`, `classify_route.py`
* **Goal:** Classify into departments, extract location/duration, and score urgency. 
* **Input:** Output from Karthik's `postprocess.py`.
* **Output:** Final JSON object ready for routing.

## How to use

1. Set up your virtual environment:
```bash
python -m venv nlp-env
source nlp-env/bin/activate  # On Windows: nlp-env\Scripts\activate
pip install -r requirements.txt
```

2. Run the test suite:
```bash
python test_samples.py
```

3. Run the FastAPI Server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```
