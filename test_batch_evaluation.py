import sys
import os
import time
import json
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import full_pipeline

# ─── 54-Item Comprehensive Civic Benchmark Dataset ───────────────────────────
BENCHMARK_DATASET = [
    # ── 1. Electricity & Lighting (9 items) ──────────────────────────────────
    {
        "id": 1,
        "text": "Street light has been broken for 1 week outside my house",
        "expected_lang": "eng",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 2,
        "text": "High voltage wire snapped and sparking on the pedestrian walkway",
        "expected_lang": "eng",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 3,
        "text": "meri bijli 2 din se nahi aa rahi hai",
        "expected_lang": "hin",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 4,
        "text": "gali ki light kharab hai andhera rehta hai",
        "expected_lang": "hin",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium"
    },
    {
        "id": 5,
        "text": "मेरे इलाके में दो दिन से बिजली नहीं आ रही है और पिछली शिकायत का समाधान नहीं हुआ।",
        "expected_lang": "hin",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 6,
        "text": "ट्रांसफार्मर में आग लग गई है तुरंत बिजली विभाग को सूचित करें।",
        "expected_lang": "hin",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 7,
        "text": "எங்கள் தெருவில் தெருவிளக்கு எரியவில்லை",
        "expected_lang": "tam",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium"
    },
    {
        "id": 8,
        "text": "మా వీధిలో వీధి దీపాలు వెలగడం లేదు",
        "expected_lang": "tel",
        "expected_dept": "lighting",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium"
    },
    {
        "id": 9,
        "text": "Please install solar LED streetlights along the park jogging track",
        "expected_lang": "eng",
        "expected_dept": "lighting",
        "expected_req_type": "Improvement",
        "expected_urgency": "Low"
    },

    # ── 2. Water Supply (10 items) ───────────────────────────────────────────
    {
        "id": 10,
        "text": "Water supply has been cut for 3 days",
        "expected_lang": "eng",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 11,
        "text": "Main water pipeline burst near metro station flooding the road",
        "expected_lang": "eng",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 12,
        "text": "Tap water is coming dirty and smelling contaminated since morning",
        "expected_lang": "eng",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 13,
        "text": "pani ka pipeline toot gaya hai aur sara peene ka pani beh raha hai",
        "expected_lang": "hin",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 14,
        "text": "hamare yahan 4 din se pani nahi aa raha peene ka pani nahi hai",
        "expected_lang": "hin",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 15,
        "text": "पानी की पाइपलाइन फट गई है और पानी बह रहा है",
        "expected_lang": "hin",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 16,
        "text": "घरों में गंदा और बदबूदार पीने का पानी सप्लाई हो रहा है।",
        "expected_lang": "hin",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 17,
        "text": "என் பகுதியில் 3 நாட்களாக தண்ணீர் இல்லை",
        "expected_lang": "tam",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 18,
        "text": "మా వీధిలో మూడు రోజుల నుండి నీరు లేదు",
        "expected_lang": "tel",
        "expected_dept": "water",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 19,
        "text": "Request to set up a community water purification ATM near the bus stand",
        "expected_lang": "eng",
        "expected_dept": "water",
        "expected_req_type": "Improvement",
        "expected_urgency": "Low"
    },

    # ── 3. Roads & Traffic Infrastructure (10 items) ─────────────────────────
    {
        "id": 20,
        "text": "Huge dangerous pothole on MG Road causing traffic accidents",
        "expected_lang": "eng",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 21,
        "text": "Road has completely caved in after rainfall near school crossing",
        "expected_lang": "eng",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 22,
        "text": "Broken concrete footpath slabs making walking hazardous for seniors",
        "expected_lang": "eng",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium"
    },
    {
        "id": 23,
        "text": "sadak par bahut bada gaddha ho gaya hai",
        "expected_lang": "hin",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 24,
        "text": "main road tooti hui hai gaadiyan fisal rahi hain",
        "expected_lang": "hin",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 25,
        "text": "सड़क पर गहरा गड्ढा है जिससे कई दोपहिया वाहन गिर चुके हैं।",
        "expected_lang": "hin",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 26,
        "text": "पैदल चलने वाला फुटपाथ पूरी तरह टूटा हुआ है।",
        "expected_lang": "hin",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium"
    },
    {
        "id": 27,
        "text": "சாலையில் பெரிய பள்ளம் உள்ளது விபத்து ஏற்படும் அபாயம்",
        "expected_lang": "tam",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 28,
        "text": "రోడ్డుపై పెద్ద గుంతలు ఉన్నాయి ప్రమాదాలు జరుగుతున్నాయి",
        "expected_lang": "tel",
        "expected_dept": "roads",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 29,
        "text": "Request to construct a speed breaker near the public school",
        "expected_lang": "eng",
        "expected_dept": "roads",
        "expected_req_type": "Improvement",
        "expected_urgency": "Medium"
    },

    # ── 4. Sanitation & Waste Management (9 items) ───────────────────────────
    {
        "id": 30,
        "text": "Garbage has not been collected for 4 days in sector 12",
        "expected_lang": "eng",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 31,
        "text": "Dead animal lying on the public street creating unbearable stink and disease risk",
        "expected_lang": "eng",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 32,
        "text": "Illegal open dumping yard created opposite residential apartment complex",
        "expected_lang": "eng",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 33,
        "text": "kachra 5 din se colony ke samne pada hai badbu aa rahi hai",
        "expected_lang": "hin",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 34,
        "text": "mere ghar pe 1000 din se safai nahi hui.",
        "expected_lang": "hin",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 35,
        "text": "कॉलोनी के मुख्य द्वार पर कूड़े का ढेर लगा है कोई उठाने नहीं आ रहा।",
        "expected_lang": "hin",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 36,
        "text": "सड़क पर कचरा गाड़ी पिछले एक हफ्ते से नहीं आई है।",
        "expected_lang": "hin",
        "expected_dept": "sanitation",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 37,
        "text": "Please place separate wet and dry waste public dustbins in the market area",
        "expected_lang": "eng",
        "expected_dept": "sanitation",
        "expected_req_type": "Improvement",
        "expected_urgency": "Low"
    },
    {
        "id": 38,
        "text": "Request to start daily door-to-door waste collection in newly developed layout",
        "expected_lang": "eng",
        "expected_dept": "sanitation",
        "expected_req_type": "Improvement",
        "expected_urgency": "Low"
    },

    # ── 5. Drainage & Sewerage (8 items) ─────────────────────────────────────
    {
        "id": 39,
        "text": "sewage water is overflowing onto the street",
        "expected_lang": "eng",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 40,
        "text": "sewage overflow in our street for 2 months, no action taken",
        "expected_lang": "eng",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 41,
        "text": "Stormwater drain is completely choked causing heavy waterlogging during rain",
        "expected_lang": "eng",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 42,
        "text": "Open uncovered manhole on main walking street risk to children",
        "expected_lang": "eng",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 43,
        "text": "ganda naala ufan raha hai aur paani gharon mein ghus raha hai",
        "expected_lang": "hin",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 44,
        "text": "सीवर का पानी सड़क पर बह रहा है",
        "expected_lang": "hin",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 45,
        "text": "हमारे मोहल्ले में 2 महीने से सीवर का पानी बह रहा है और कोई कार्रवाई नहीं हुई",
        "expected_lang": "hin",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 46,
        "text": "सड़क पर खुला हुआ मैनहोल है तुरंत ढक्कन लगवाया जाए।",
        "expected_lang": "hin",
        "expected_dept": "drainage",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },

    # ── 6. Green Infrastructure & Parks (5 items) ────────────────────────────
    {
        "id": 47,
        "text": "Big tree branch fell down and blocking the residential road",
        "expected_lang": "eng",
        "expected_dept": "green",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 48,
        "text": "Heavy storm uprooted dangerous tree leaning against electrical transformer",
        "expected_lang": "eng",
        "expected_dept": "green",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 49,
        "text": "barish ki wajah se bada ped raste par gir gaya hai rasta band hai",
        "expected_lang": "hin",
        "expected_dept": "green",
        "expected_req_type": "Complaint",
        "expected_urgency": "Critical"
    },
    {
        "id": 50,
        "text": "कॉलोनी के पार्क में सूखा पेड़ गिरने की कगार पर है इसे कटवाया जाए।",
        "expected_lang": "hin",
        "expected_dept": "green",
        "expected_req_type": "Complaint",
        "expected_urgency": "High"
    },
    {
        "id": 51,
        "text": "Please plant native shade trees along the newly widened avenue",
        "expected_lang": "eng",
        "expected_dept": "green",
        "expected_req_type": "Improvement",
        "expected_urgency": "Low"
    },

    # ── 7. Ambiguous / Review Cases (3 items) ─────────────────────────────────
    {
        "id": 52,
        "text": "meri complaint abhi tak pending hai",
        "expected_lang": "hin",
        "expected_dept": "Unclassified",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium"
    },
    {
        "id": 53,
        "text": "I submitted grievance number 9482 last month but got no response from office",
        "expected_lang": "eng",
        "expected_dept": "Unclassified",
        "expected_req_type": "Complaint",
        "expected_urgency": "Medium"
    },
    {
        "id": 54,
        "text": "Hello municipal corporation kindly share contact details of ward counselor",
        "expected_lang": "eng",
        "expected_dept": "Unclassified",
        "expected_req_type": "Complaint",
        "expected_urgency": "Low"
    }
]

def run_batch_evaluation():
    print("=" * 80)
    print("  RUNNING 54-ITEM MULTILINGUAL CIVIC COMPLAINT BENCHMARK SUITE")
    print("=" * 80)

    total = len(BENCHMARK_DATASET)
    dept_correct = 0
    type_correct = 0
    lang_correct = 0
    latencies = []
    results = []

    for item in BENCHMARK_DATASET:
        idx = item["id"]
        txt = item["text"]
        t0 = time.time()
        res = full_pipeline(txt, debug=True)
        lat = round((time.time() - t0) * 1000)
        latencies.append(lat)

        pred_dept = res.get("department", "Unclassified")
        pred_type = res.get("request_type", "Complaint")
        pred_lang = res.get("detected_language", "eng")
        urg_label = res.get("urgency", {}).get("label", "Medium")

        d_ok = (pred_dept == item["expected_dept"])
        t_ok = (pred_type == item["expected_req_type"])
        l_ok = (pred_lang == item["expected_lang"])

        if d_ok:
            dept_correct += 1
        if t_ok:
            type_correct += 1
        if l_ok:
            lang_correct += 1

        status = "OK" if (d_ok and t_ok and l_ok) else "WARN"
        short_txt = (txt[:26] + "...") if len(txt) > 26 else txt
        print(f"[{idx:2d}/54] {short_txt:<30} | Dept: {pred_dept:<12} ({item['expected_dept']:<12}) | Type: {pred_type:<11} | Lat: {lat:>5}ms [{status}]")

        results.append({
            "id": idx,
            "text": txt,
            "pred_dept": pred_dept,
            "expected_dept": item["expected_dept"],
            "dept_ok": d_ok,
            "pred_type": pred_type,
            "expected_type": item["expected_req_type"],
            "type_ok": t_ok,
            "pred_lang": pred_lang,
            "expected_lang": item["expected_lang"],
            "lang_ok": l_ok,
            "urgency": urg_label,
            "latency_ms": lat
        })

    dept_acc = (dept_correct / total) * 100.0
    type_acc = (type_correct / total) * 100.0
    lang_acc = (lang_correct / total) * 100.0
    avg_lat = sum(latencies) / len(latencies)

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS (N = {total}):")
    print(f"  • Department Accuracy:   {dept_acc:.1f}% ({dept_correct}/{total})")
    print(f"  • Request Type Accuracy: {type_acc:.1f}% ({type_correct}/{total})")
    print(f"  • Language Accuracy:     {lang_acc:.1f}% ({lang_correct}/{total})")
    print(f"  • Avg Latency:           {avg_lat:.1f} ms (~{avg_lat/1000:.2f} s)")
    print(f"  • Min Latency:           {min(latencies)} ms")
    print(f"  • Max Latency:           {max(latencies)} ms")
    print("=" * 80)

    # Save to JSON
    with open("benchmark_54_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "dept_acc": dept_acc,
            "dept_correct": dept_correct,
            "type_acc": type_acc,
            "type_correct": type_correct,
            "lang_acc": lang_acc,
            "lang_correct": lang_correct,
            "avg_latency_ms": avg_lat,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print("Saved benchmark results to 'benchmark_54_results.json'")

if __name__ == "__main__":
    run_batch_evaluation()
