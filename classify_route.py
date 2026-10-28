"""
Steps 5 & 6 combined: Classification + Entity Extraction + Urgency + Routing.
This is Yshasvee's final deliverable function.
Takes Karthik's output (clean English text) and produces the structured
complaint object that gets attached to the InfraLynx request.

Usage:
    from classify_route import classify_and_route
"""
from classify import classify_department, classify_request_type
from entities import extract_entities
from urgency import estimate_urgency
from groq_client import groq_analyze_request_type_and_urgency

def classify_and_route(input_dict: dict) -> dict:
    """
    Full Step 5-6 pipeline with Hybrid Groq AI + Local Fallback:
    1. Try Groq AI for contextual Request Type and Urgency analysis
    2. Fallback to local zero-shot + heuristic models if Groq is unavailable
    3. Classify complaint into a department (zero-shot + keyword corroboration)
    4. Extract entities (duration, location, previous complaint)
    5. Package everything into the output contract
    """
    text = input_dict.get("english_text", "")
    orig_text = input_dict.get("original_text", "")

    # Step 5a & 5d: Hybrid Groq AI analysis for Request Type and Urgency
    groq_res = groq_analyze_request_type_and_urgency(text, original_text=orig_text)

    if groq_res is not None:
        req_type = groq_res["request_type"]
        req_type_conf = groq_res["request_type_confidence"]
        urgency = groq_res["urgency"]
        urgency_source = "groq-ai"
    else:
        # Local fallback
        req_type_res = classify_request_type(text, original_text=orig_text)
        req_type = req_type_res["request_type"]
        req_type_conf = req_type_res["confidence"]
        urgency = estimate_urgency(text, original_text=orig_text)
        urgency_source = "local-rule-based"

    # Step 5b: Department classification
    dept_result = classify_department(text, original_text=orig_text)

    # Step 5c: Entity extraction
    entities = extract_entities(text, original_text=orig_text)

    # Step 6: Package for routing
    return {
        "original_text": orig_text,
        "detected_language": input_dict.get("detected_language", ""),
        "script": input_dict.get("script", "other"),
        "is_english": input_dict.get("is_english", False),
        "native_script_text": input_dict.get("native_script_text", orig_text),
        "confidence": input_dict.get("confidence", 0.95),
        "english_text": text,
        "request_type": req_type,
        "request_type_confidence": req_type_conf,
        "department": dept_result["department"],
        "department_confidence": dept_result["confidence"],
        "needs_manual_review": dept_result["needs_manual_review"],
        "entities": entities,
        "urgency": urgency,
        "urgency_source": urgency_source,
    }
