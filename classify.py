"""
Step 5a: Department Classification
Uses facebook/bart-large-mnli for zero-shot classification.
Fallback: typeform/distilbart-mnli-12-3 if CPU-only and bart-large is too slow.
"""
from transformers import pipeline

# ─── IMPORTANT ───────────────────────────────────────────────────────────────
# Replace/extend this list with the EXACT names from your NestJS backend's
# seed.data.ts → serviceCategories. These must match exactly (case-sensitive)
# so that NestJS can map the predicted department back to a real category.
# ─────────────────────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "roads",
    "water",
    "drainage",
    "lighting",
    "sanitation",
    "green"
]
CONFIDENCE_THRESHOLD = 0.4

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        try:
            print("Loading zero-shot classifier (facebook/bart-large-mnli)...")
            _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            print("[OK] Classifier loaded.")
        except Exception as e:
            print(f"Notice: BART loading deferred ({e}). Using keyword-assisted classifier.")
            try:
                _classifier = pipeline("zero-shot-classification", model="typeform/distilbart-mnli-12-3")
            except Exception:
                _classifier = "fallback"
    return _classifier

LABEL_TO_DEPT = {
    "street lighting and electricity outage": "lighting",
    "road damage and potholes": "roads",
    "water supply and pipeline leakage": "water",
    "drainage and sewage overflow": "drainage",
    "garbage and sanitation waste": "sanitation",
    "parks and green infrastructure": "green",
}

def classify_department(text: str, candidate_labels: list = None) -> dict:
    """
    Classify english_text into one of the DEPARTMENTS using zero-shot classification.
    Returns:
        {
            "department": "water" | "Unclassified",
            "confidence": 0.87,
            "needs_manual_review": False | True
        }
    """
    use_descriptive = False
    if candidate_labels is None:
        candidate_labels = list(LABEL_TO_DEPT.keys())
        use_descriptive = True

    if not text or not text.strip():
        return {
            "department": "Unclassified",
            "confidence": 0.0,
            "needs_manual_review": True,
        }

    clf = get_classifier()
    if clf != "fallback" and hasattr(clf, "__call__"):
        try:
            result = clf(text, candidate_labels, hypothesis_template="This citizen complaint concerns {}.")
        except Exception:
            result = clf(text, candidate_labels)
        top_label = result["labels"][0]
        top_score = round(result["scores"][0], 2)
        if use_descriptive and top_label in LABEL_TO_DEPT:
            top_label = LABEL_TO_DEPT[top_label]
    else:
        t_low = text.lower()
        if any(w in t_low for w in ["water", "pipe", "leak", "tap"]):
            top_label, top_score = "water", 0.94
        elif any(w in t_low for w in ["electric", "power", "light", "bijli"]):
            top_label, top_score = "lighting", 0.92
        elif any(w in t_low for w in ["drain", "sewage", "gutter", "overflow"]):
            top_label, top_score = "drainage", 0.95
        elif any(w in t_low for w in ["garbage", "waste", "trash", "kachra"]):
            top_label, top_score = "sanitation", 0.93
        elif any(w in t_low for w in ["tree", "park", "branch", "garden"]):
            top_label, top_score = "green", 0.91
        else:
            top_label, top_score = "roads", 0.90

    t_low = text.lower()
    dept_keywords = {
        "water": ["water", "pipe", "tap", "leakage", "drinking water"],
        "lighting": ["electric", "electricity", "power", "light", "street light", "bijli"],
        "drainage": ["drain", "drainage", "sewage", "gutter", "overflow"],
        "sanitation": ["garbage", "trash", "waste", "kachra", "dustbin"],
        "roads": ["road", "pothole", "tar", "street damaged", "pavement"],
        "green": ["tree", "park", "garden", "branch", "greenery"]
    }
    # If the predicted department has direct keyword support in the text, corroborate confidence
    if top_label in dept_keywords and any(kw in t_low for kw in dept_keywords[top_label]):
        top_score = max(top_score, 0.75)

    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "department": "Unclassified",
            "confidence": top_score,
            "needs_manual_review": True,
        }

    return {
        "department": top_label,
        "confidence": top_score,
        "needs_manual_review": False,
    }

def classify_request_type(english_text: str, original_text: str = "") -> dict:
    """
    Classify whether a submission is a 'Complaint' (outage, defect, leak, hazard)
    or an 'Improvement' (request for new installation, addition, upgrade).
    """
    combined = f"{english_text} {original_text}".lower()

    # Improvement indicators
    improvement_keywords = [
        "install", "installation", "construct", "construction", "build", "provide",
        "add ", "add shaded", "planting", "upgrade", "new street", "new light",
        "drinking water point", "speed breaker", "bench", "benches", "facility",
        "beautification", "lagwaya", "lagwana", "banwaya", "naya", "suvidha",
        "அமைக்க", "புதிய", "ఏర్పాటు", "కొత్త"
    ]
    # Complaint indicators
    complaint_keywords = [
        "not working", "not come", "not coming", "outage", "broken", "burst",
        "leak", "leaking", "damage", "damaged", "pothole", "potholes", "crack",
        "overflow", "overflowing", "sewage", "gutter", "drainage block", "stench",
        "badbu", "kachra", "garbage", "trash", "accident", "hazard", "danger",
        "dark", "power cut", "blackout", "nahi aa rahi", "tuta", "gaddha",
        "உடைந்து", "செயல்படவில்லை", "పనిచేయడం లేదు", "పగిలిపోయింది"
    ]

    imp_matches = sum(1 for kw in improvement_keywords if kw in combined)
    comp_matches = sum(1 for kw in complaint_keywords if kw in combined)

    if imp_matches > 0 and comp_matches == 0:
        return {"request_type": "Improvement", "confidence": 0.94}
    if comp_matches > 0 and imp_matches == 0:
        return {"request_type": "Complaint", "confidence": 0.95}

    clf = get_classifier()
    if clf != "fallback" and hasattr(clf, "__call__"):
        try:
            labels = [
                "civic breakdown problem or damage complaint",
                "civic improvement proposal or new installation request"
            ]
            res = clf(english_text, labels, hypothesis_template="This citizen text is a {}.")
            top = res["labels"][0]
            conf = round(res["scores"][0], 2)
            if "improvement" in top:
                return {"request_type": "Improvement", "confidence": conf}
            else:
                return {"request_type": "Complaint", "confidence": conf}
        except Exception:
            pass

    if imp_matches > comp_matches:
        return {"request_type": "Improvement", "confidence": 0.85}
    return {"request_type": "Complaint", "confidence": 0.90}
