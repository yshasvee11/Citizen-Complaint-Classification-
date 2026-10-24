"""
Step 5c: Urgency Estimation
Uses a deterministic, explainable weighted-keyword scorer.
NOT using sentiment analysis (VADER etc.) — sentiment measures emotional
tone ("I'm furious"), not actual severity ("gas leak"). A calmly worded
gas leak report should rank higher than an angry pothole complaint.
"""

import re

# ─── Keyword weights ─────────────────────────────────────────────────────────
# Tune these after testing with Karthik's real translated outputs.
# Keep the scoring logic unchanged; just adjust dictionary contents.
# ──────────────────────────────────────────────────────────────────────────────

URGENCY_SIGNALS = {
    # Critical — immediate life safety / disaster hazards (weight 3.0+)
    "fire": 3.5,
    "gas leak": 3.5,
    "electric shock": 3.5,
    "electrocution": 3.5,
    "collapse": 3.0,
    "collapsed": 3.0,
    "life threatening": 3.5,
    "severe epidemic": 3.5,
    "fatal accident": 3.0,
    "fatal": 3.0,
    "fatality": 3.0,
    "death": 3.0,
    "open high voltage": 3.0,
    "जानलेवा": 3.5,

    # High — severe service outages or active hazards (weight 2.0 - 2.5)
    "burst": 2.5,
    "flooding": 2.5,
    "hospital": 2.5,
    "clinic": 2.5,
    "severe accident": 2.5,
    "serious accident": 2.5,
    "major accident": 2.5,
    "traffic accident": 2.5,
    "traffic accidents": 2.5,
    "can cause a serious accident": 2.5,
    "can cause serious accident": 2.5,
    "can cause an accident": 2.5,
    "can cause accident": 2.5,
    "cause an accident": 2.5,
    "cause accident": 2.5,
    "causing accidents": 2.5,
    "causing accident": 2.5,
    "prone to accident": 2.5,
    "prone to accidents": 2.5,
    "road accident": 2.5,
    "road accidents": 2.5,
    "accident": 2.2,
    "accidents": 2.2,
    "dangerous pothole": 2.2,
    "huge pothole": 2.0,
    "massive pothole": 2.0,
    "deep pothole": 2.0,
    "hazard": 2.0,
    "hazardous": 2.2,
    "dangerous": 2.0,
    "danger": 2.0,
    "contaminated water": 2.5,
    "short circuit": 2.5,
    "emergency": 2.0,

    # Multilingual / Indic hazard terms
    "गंभीर दुर्घटना": 2.5,
    "दुर्घटना": 2.2,
    "खतरा": 2.0,
    "विபத்து": 2.2,
    "ప్రమాదం": 2.2,

    # Medium — standard utility failures & service outages (weight 1.0 - 1.5)
    # Notice: a 1-2 day power/water outage is Medium civic priority
    "no electricity": 1.5,
    "no power": 1.5,
    "power cut": 1.5,
    "outage": 1.5,
    "not working": 1.5,
    "not been working": 1.5,
    "not come": 1.5,
    "not coming": 1.5,
    "has not come": 1.5,
    "has not been coming": 1.5,
    "cut for": 1.5,
    "has been cut": 1.5,
    "supply cut": 1.5,
    "supply has been cut": 1.5,
    "water cut": 1.5,
    "no water": 1.5,
    "sewage": 1.5,
    "overflow": 1.5,
    "pothole": 1.2,
    "potholes": 1.2,
    "broken": 1.2,
    "damaged": 1.2,
    "leak": 1.2,
    "leaking": 1.2,
    "blocked": 1.2,
    "garbage": 1.2,
    "stench": 1.2,
    "dark": 1.0,
    "urgent": 1.5,
}

def _is_multi_day_duration(text_lower: str) -> bool:
    """Check if the text specifies a prolonged duration (>= 2 days, weeks, months)."""
    multi_day_patterns = [
        "2 day", "2 days", "two day", "two days", "3 day", "3 days", "three day", "three days",
        "4 day", "4 days", "four day", "four days", "5 day", "5 days", "five day", "five days",
        "6 day", "6 days", "six day", "six days", "7 day", "7 days", "seven day", "seven days",
        "week", "weeks", "month", "months", "multiple days", "several days",
        "2 din", "do din", "3 din", "teen din", "5 din", "दो दिन", "तीन दिन", "पाँच दिन",
        "3 naatkalaga", "moodu rojula"
    ]
    if any(p in text_lower for p in multi_day_patterns):
        return True
    m = re.search(r'(\d+)\s*days?', text_lower)
    if m and int(m.group(1)) >= 2:
        return True
    return False

def score_urgency(text: str, original_text: str = "") -> float:
    """
    Calculate urgency score without double-counting nested substrings.
    Sorts keywords by length descending so longer matching phrases take precedence.
    Evaluates both normalized English text and original citizen text for signals.
    """
    combined = f"{text or ''} {original_text or ''}".strip().lower()
    text_lower = combined
    matched_score = 0.0
    matched_spans = []

    # Sort signals longest first
    sorted_keywords = sorted(URGENCY_SIGNALS.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        idx = text_lower.find(keyword)
        if idx != -1:
            end = idx + len(keyword)
            # Check if this span overlaps with an already matched longer phrase
            overlap = any(s <= idx < e or s < end <= e for s, e in matched_spans)
            if not overlap:
                matched_spans.append((idx, end))
                # Only take the maximum single signal weight per category or highest severity
                matched_score = max(matched_score, URGENCY_SIGNALS[keyword])

    # Escalation: Essential domestic utility outage (electricity/power or drinking water supply)
    # A prolonged domestic power blackout or water supply cut (>= 2 days) is High urgency (2.2).
    # Single-pole outdoor street lights ("street light off for a week") remain Medium (1.5).
    is_street_light_only = any(sl in text_lower for sl in ["street light", "streetlight", "street lights", "streetlights", "lamp post", "khamba", "batti", "vilakku", "deepalu", "deepam"])
    is_power_or_water = any(u in text_lower for u in ["electricity", "power", "water", "bijli", "pani", "current", "drinking water", "pipeline"])
    has_outage_signal = any(s in text_lower for s in ["not come", "has not come", "no electricity", "no power", "no water", "cut", "outage", "not working", "nahi aa rahi", "ledu", "illai", "broken", "burst"])

    if is_power_or_water and has_outage_signal and not is_street_light_only:
        if _is_multi_day_duration(text_lower):
            matched_score = max(matched_score, 2.2)  # High urgency (>= 2.0)
        else:
            matched_score = max(matched_score, 1.5)  # Medium for brief/single-day outage
    elif is_power_or_water and any(d in text_lower for d in ["day", "days", "din", "naatkal", "rojula"]):
        matched_score = max(matched_score, 1.5)

    # Escalation factor for multi-week/month chronic neglect or repeated follow-ups
    if any(k in text_lower for k in ["week", "weeks", "month", "months", "multiple complaints", "complained before", "no action taken", "no action"]):
        matched_score = min(5.0, matched_score + 1.0)

    return round(matched_score, 1)

def urgency_label(score: float) -> str:
    """
    Map numeric score to human-readable priority:
    - Critical: >= 3.0
    - High: >= 2.0
    - Medium: >= 1.0
    - Low: < 1.0
    """
    if score >= 3.0:
        return "Critical"
    elif score >= 2.0:
        return "High"
    elif score >= 1.0:
        return "Medium"
    else:
        return "Low"

def estimate_urgency(text: str, original_text: str = "") -> dict:
    """
    Estimate how urgent/serious a complaint is based on keyword signals.
    Returns:
        {
            "score": 2.5,
            "label": "High"
        }
    """
    if not text or not text.strip():
        if original_text and original_text.strip():
            score = score_urgency(original_text)
            return {"score": score, "label": urgency_label(score)}
        return {"score": 0.0, "label": "Low"}

    score = score_urgency(text, original_text=original_text)
    return {"score": score, "label": urgency_label(score)}
