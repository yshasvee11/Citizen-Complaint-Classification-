"""
Step 5b: Entity Extraction
Uses spaCy NER for duration/location, regex fallback for duration,
keyword matching for previous-complaint detection.
"""
import re
import spacy

# Load once at module level
print("Loading spaCy en_core_web_sm for entity extraction...")
nlp = spacy.load("en_core_web_sm")
print("[OK] spaCy model loaded.")

# Regex fallback for duration — catches "3 days", "2 hours", "1 week", etc.
DURATION_REGEX = re.compile(
    r"(\d+)\s*(days?|hours?|weeks?|months?|years?)", re.IGNORECASE
)

# Keywords/phrases indicating the citizen has complained about this before
PREVIOUS_COMPLAINT_PHRASES = [
    "already complained",
    "complained before",
    "previous complaint",
    "no action",
    "again",
    "still not resolved",
    "reported earlier",
    "still pending",
    "follow up",
    "followup",
    "reminder",
    "second time",
    "third time",
    "multiple times",
    "repeatedly",
    "no response",
    "ignored",
]

def extract_duration_and_location(text: str, original_text: str = "") -> dict:
    """
    Extract duration (how long the problem has lasted) and location from text.
    Uses spaCy NER for DATE/TIME and GPE/LOC/FAC entities, with a regex
    fallback for duration patterns that spaCy's general English model misses.
    """
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

    # Check original_text if duration still missing
    if duration is None and original_text:
        match_orig = re.search(r'(\d+)\s*(days?|दिन|din|naatkal|rojula)', original_text, re.IGNORECASE)
        if match_orig:
            duration = f"{match_orig.group(1)} days"

    # Civic location pattern heuristic fallback if spaCy NER missed non-named locations
    if location is None:
        comb = f"{text or ''} {original_text or ''}".lower()
        if any(h in comb for h in ["ghar", "my house", "at home", "in house", "our house"]):
            location = "Residence / House"
        elif "market" in comb or "gate" in comb:
            location = "Market Area"
        elif "colony" in comb:
            location = "Residential Colony"
        elif "park" in comb:
            location = "Public Park"
        elif "clinic" in comb or "hospital" in comb:
            location = "Health Clinic"

    return {"duration": duration, "location": location}

def detect_previous_complaint(text: str, original_text: str = "") -> bool:
    """
    Check if the complaint text indicates the citizen has raised this
    issue before (follow-up / repeated complaint).
    """
    combined = f"{text or ''} {original_text or ''}".lower()
    return any(phrase in combined for phrase in PREVIOUS_COMPLAINT_PHRASES)

def extract_entities(text: str, original_text: str = "") -> dict:
    """
    Combined entity extraction: duration, location, and previous-complaint flag.
    Returns:
        {
            "duration": "2 days" | None,
            "location": "MG Road" | None,
            "previous_complaint": True | False
        }
    """
    if not text and not original_text:
        return {"duration": None, "location": None, "previous_complaint": False}

    result = extract_duration_and_location(text, original_text=original_text)
    result["previous_complaint"] = detect_previous_complaint(text, original_text=original_text)
    return result
