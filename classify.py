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
    "street lighting, electricity outage, and dark roads": "lighting",
    "road damage, potholes, and speed breakers": "roads",
    "drinking water supply, water cut, and pipeline leakage": "water",
    "drainage block, sewage overflow, and gutter problems": "drainage",
    "garbage collection, street cleaning, sanitation waste, and pest infestation": "sanitation",
    "parks, fallen tree branches, and green infrastructure": "green",
}

def classify_department(text: str, candidate_labels: list = None, original_text: str = "") -> dict:
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

    combined_low = f"{text or ''} {original_text or ''}".strip().lower()

    if not combined_low:
        return {
            "department": "Unclassified",
            "confidence": 0.0,
            "needs_manual_review": True,
        }

    dept_keywords = {
        "water": ["water", "pipe", "pipeline", "tap", "leak", "leaking", "leakage", "drinking water", "pani", "neeru", "thanni", "தண்ணீர்", "நீరు", "पानी"],
        "lighting": ["electric", "electricity", "power", "power cut", "light", "lights", "street light", "streetlight", "streetlights", "lamp", "khamba", "bijli", "current", "dark", "चमक", "बिजली", "விளக்கு", "దీపాలు"],
        "drainage": ["drain", "drainage", "sewage", "gutter", "overflow", "overflowing", "sewer", "dirty water", "naali", "सीवर", "नाली"],
        "sanitation": ["garbage", "trash", "waste", "kachra", "dustbin", "clean", "cleaned", "cleaning", "cleanliness", "safai", "sweeping", "litter", "stench", "badbu", "smell", "pest", "infestation", "mosquito", "बदबू", "कचरा", "सफाई", "कूड़ा", "குப்பை", "చెత్త"],
        "roads": ["road", "roads", "pothole", "potholes", "tar", "pavement", "footpath", "speed breaker", "gaddha", "sadak", "traffic", "गड्ढा", "सड़क", "குழி", "గుంత"],
        "green": ["tree", "trees", "park", "parks", "garden", "gardens", "branch", "branches", "greenery", "grass", "plant", "पेड़", "पार्क", "மரங்கள்", "చెట్లు"]
    }

    clf = get_classifier()
    if clf != "fallback" and hasattr(clf, "__call__"):
        try:
            # Use english_text if present, otherwise original_text
            eval_text = text if (text and text.strip()) else original_text
            result = clf(eval_text, candidate_labels, hypothesis_template="This citizen complaint concerns {}.")
        except Exception:
            result = clf(text or original_text, candidate_labels)
        top_label = result["labels"][0]
        top_score = round(result["scores"][0], 2)
        if use_descriptive and top_label in LABEL_TO_DEPT:
            top_label = LABEL_TO_DEPT[top_label]
    else:
        if any(w in combined_low for w in dept_keywords["water"]):
            top_label, top_score = "water", 0.94
        elif any(w in combined_low for w in dept_keywords["lighting"]):
            top_label, top_score = "lighting", 0.92
        elif any(w in combined_low for w in dept_keywords["drainage"]):
            top_label, top_score = "drainage", 0.95
        elif any(w in combined_low for w in dept_keywords["sanitation"]):
            top_label, top_score = "sanitation", 0.93
        elif any(w in combined_low for w in dept_keywords["green"]):
            top_label, top_score = "green", 0.91
        elif any(w in combined_low for w in dept_keywords["roads"]):
            top_label, top_score = "roads", 0.90
        else:
            top_label, top_score = "Unclassified", 0.25

    # Check keyword corroboration
    matched_depts = [dept for dept, kws in dept_keywords.items() if any(kw in combined_low for kw in kws)]

    # If predicted department has keyword support, boost confidence
    if top_label in matched_depts:
        top_score = max(top_score, 0.88)
    elif matched_depts:
        # Action-oriented keywords take precedence (e.g., street cleaning -> sanitation, not roads)
        if "sanitation" in matched_depts and any(sw in combined_low for sw in ["clean", "cleaned", "cleaning", "safai", "garbage", "kachra", "कूड़ा", "सफाई"]):
            top_label = "sanitation"
            top_score = max(top_score, 0.90)
        elif "water" in matched_depts and any(ww in combined_low for ww in ["water", "pipe", "pipeline", "leak", "pani"]):
            top_label = "water"
            top_score = max(top_score, 0.90)
        elif "lighting" in matched_depts and any(lw in combined_low for lw in ["electric", "power", "light", "bijli"]):
            top_label = "lighting"
            top_score = max(top_score, 0.90)
        elif top_score < CONFIDENCE_THRESHOLD:
            # If BART was uncertain (< 0.4), fallback to the first matched keyword department
            top_label = matched_depts[0]
            top_score = 0.85

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
        "complaint", "pending", "unresolved", "no action",
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
