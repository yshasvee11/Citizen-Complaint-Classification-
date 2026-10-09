"""
Step 5c: Urgency Estimation
Uses a deterministic, explainable weighted-keyword scorer.
NOT using sentiment analysis (VADER etc.) — sentiment measures emotional
tone ("I'm furious"), not actual severity ("gas leak"). A calmly worded
gas leak report should rank higher than an angry pothole complaint.
"""

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

    # High — severe service outages or active hazards (weight 2.0 - 2.5)
    "burst": 2.5,
    "flooding": 2.5,
    "hospital": 2.5,
    "clinic": 2.5,
    "severe accident": 2.5,
    "contaminated water": 2.5,
    "open high voltage": 3.0,

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
    "emergency": 2.0,
}

def score_urgency(text: str) -> float:
    """
    Calculate urgency score without double-counting nested substrings.
    Sorts keywords by length descending so longer matching phrases take precedence.
    """
    text_lower = text.lower()
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

    # Essential utility outage lasting days (e.g., electricity or water out for 2-3 days)
    # Calibrated to Medium (1.5)
    utility_words = ["electricity", "power", "water", "light", "bijli", "pani", "current"]
    duration_words = ["day", "days", "din", "naatkal", "rojula", "hours", "ghante"]
    if any(u in text_lower for u in utility_words) and any(d in text_lower for d in duration_words):
        matched_score = max(matched_score, 1.5)

    # Escalation factor for multi-day complaints (e.g. 5+ days or weeks)
    if any(k in text_lower for k in ["week", "weeks", "month", "multiple complaints", "complained before"]):
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

def estimate_urgency(text: str) -> dict:
    """
    Estimate how urgent/serious a complaint is based on keyword signals.
    Returns:
        {
            "score": 1.5,
            "label": "Medium"
        }
    """
    if not text or not text.strip():
        return {"score": 0.0, "label": "Low"}

    score = score_urgency(text)
    return {"score": score, "label": urgency_label(score)}
