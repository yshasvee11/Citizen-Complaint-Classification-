"""
InfraLynx CRIMS — NLP Layer
Comprehensive Batch Evaluation & Stress Test Suite

Evaluates the complete multilingual pipeline:
    Aayush (Steps 1-2: Detect & Transliterate)
    Karthik (Steps 3-4: Translate & Normalize)
    Yshasvee (Steps 5-6: Classify, Extract & Route)

Run with:
    ./nlp-env/bin/python test_batch_evaluation.py
"""

import time
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import full_pipeline

BENCHMARK_DATASET = [
    # ── 1. Electricity / Lighting (English, Hindi Devanagari, Hinglish) ────────
    {
        "text": "meri bijli 2 din se nahi aa rahi hai",
        "expected_lang": "hin",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": "2 days"
    },
    {
        "text": "मेरे इलाके में दो दिन से बिजली नहीं आ रही है और पिछली शिकायत का समाधान नहीं हुआ।",
        "expected_lang": "hin",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": True,
        "expected_duration": "two days"
    },
    {
        "text": "Street light has been broken for 1 week outside my house",
        "expected_lang": "eng",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": "1 week"
    },

    # ── 2. Water Supply (English, Tamil, Telugu, Hindi) ─────────────────────────
    {
        "text": "Water supply has been cut for 3 days",
        "expected_lang": "eng",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": "3 days"
    },
    {
        "text": "என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",
        "expected_lang": "tam",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": "3 days"
    },
    {
        "text": "మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",
        "expected_lang": "tel",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": "3 days"
    },
    {
        "text": "पानी की पाइपलाइन फट गई है और पानी बह रहा है",
        "expected_lang": "hin",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": None
    },

    # ── 3. Roads & Infrastructure (Hindi, Hinglish, English) ─────────────────────
    {
        "text": "sadak par bahut bada gaddha ho gaya hai",
        "expected_lang": "hin",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": None
    },
    {
        "text": "Huge dangerous pothole on MG Road causing traffic accidents",
        "expected_lang": "eng",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": None
    },

    # ── 4. Sanitation & Waste (Hindi, Hinglish, English) ────────────────────────
    {
        "text": "kachra 5 din se colony ke samne pada hai badbu aa rahi hai",
        "expected_lang": "hin",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": "5 days"
    },
    {
        "text": "Garbage has not been collected for 4 days in sector 12",
        "expected_lang": "eng",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": "4 days"
    },

    # ── 5. Drainage & Sewerage ──────────────────────────────────────────────────
    # Pair A: Fresh sewage overflow report without duration -> Medium urgency in both languages
    {
        "text": "sewage water is overflowing onto the street",
        "expected_lang": "eng",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium",
        "expected_prev_complaint": False,
        "expected_duration": None
    },
    {
        "text": "सीवर का पानी सड़क पर बह रहा है",
        "expected_lang": "hin",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium",
        "expected_prev_complaint": False,
        "expected_duration": None
    },
    # Pair B: Chronic 2-month neglected sewage overflow -> High urgency in both languages
    {
        "text": "sewage overflow in our street for 2 months, no action taken",
        "expected_lang": "eng",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "High",
        "expected_prev_complaint": False,
        "expected_duration": "2 months"
    },
    {
        "text": "हमारे मोहल्ले में 2 महीने से सीवर का पानी बह रहा है और कोई कार्रवाई नहीं हुई",
        "expected_lang": "hin",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "High",
        "expected_prev_complaint": True,
        "expected_duration": "2 months"
    },

    # ── 6. Green Infrastructure / Parks (English, Hinglish) ───────────────────────
    {
        "text": "Big tree branch fell down and blocking the residential road",
        "expected_lang": "eng",
        "expected_dept": "green",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": False,
        "expected_duration": None
    },

    # ── 7. Civic Improvement Proposals (Requests for New Infrastructure) ────────
    {
        "text": "Please install new street lights and benches in the central park",
        "expected_lang": "eng",
        "expected_dept": "green",
        "expected_req_type": "Improvement",
        "expected_prev_complaint": False,
        "expected_duration": None
    },
    {
        "text": "Request to construct a speed breaker near the public school",
        "expected_lang": "eng",
        "expected_dept": "roads",
        "expected_req_type": "Improvement",
        "expected_prev_complaint": False,
        "expected_duration": None
    },

    # ── 8. Ambiguous / Edge Case (Needs Manual Review) ───────────────────────────
    {
        "text": "meri complaint abhi tak pending hai",
        "expected_lang": "hin",
        "expected_dept": "Unclassified",
        "expected_req_type": "Complaint",
        "expected_prev_complaint": True,
        "expected_duration": None
    }
]

def run_batch_evaluation():
    print("\n" + "=" * 78)
    print("  INFRALYNX CRIMS -- MULTILINGUAL NLP PIPELINE BATCH EVALUATION SUITE")
    print("=" * 78)

    dept_correct = 0
    type_correct = 0
    total = len(BENCHMARK_DATASET)
    latencies = []

    print(f"\nEvaluating {total} benchmark complaints across English, Hindi, Hinglish, Tamil, Telugu...\n")
    print(f"{'#':<3} {'Input (truncated)':<30} {'Pred Dept':<12} {'Exp Dept':<12} {'Req Type':<12} {'Urgency':<10} {'Latency':<8} {'Status'}")
    print("-" * 105)

    for i, item in enumerate(BENCHMARK_DATASET, start=1):
        t0 = time.time()
        res = full_pipeline(item["text"], debug=True)
        elapsed_ms = round((time.time() - t0) * 1000)
        latencies.append(elapsed_ms)

        pred_dept = res.get("department", "Unclassified")
        pred_type = res.get("request_type", "Complaint")
        urgency_label = res.get("urgency", {}).get("label", "Low")

        dept_match = (pred_dept == item["expected_dept"])
        type_match = (pred_type == item["expected_req_type"])

        if dept_match:
            dept_correct += 1
        if type_match:
            type_correct += 1

        status = "OK" if dept_match and type_match else "WARN"
        short_text = (item["text"][:27] + "...") if len(item["text"]) > 27 else item["text"]

        print(f"{i:<3} {short_text:<30} {pred_dept:<12} {item['expected_dept']:<12} {pred_type:<12} {urgency_label:<10} {elapsed_ms:>5}ms   [{status}]")

    dept_acc = (dept_correct / total) * 100.0
    type_acc = (type_correct / total) * 100.0
    avg_lat = sum(latencies) / len(latencies)

    print("-" * 105)
    print(f"\nEVALUATION SUMMARY:")
    print(f"  • Total Test Samples:             {total}")
    print(f"  • Department Classification Acc: {dept_acc:.1f}% ({dept_correct}/{total})")
    print(f"  • Request Type (Comp/Imp) Acc:   {type_acc:.1f}% ({type_correct}/{total})")
    print(f"  • Average Latency per Complaint: {avg_lat:.1f} ms")
    print(f"  • Max Latency:                   {max(latencies)} ms")
    print(f"  • Min Latency:                   {min(latencies)} ms")
    print("\n" + "=" * 78 + "\n")

if __name__ == "__main__":
    run_batch_evaluation()

