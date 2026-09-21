# InfraLynx CRIMS — NestJS Backend Integration Guide

This document explains how to connect the **NestJS backend** (complaint intake API) with this **FastAPI NLP Service** (`http://localhost:8001/analyze`).

---

## 1. Architecture & Service Boundaries

```
Citizen Mobile App / Web Portal
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│              NestJS Backend (/requests)                  │
│                                                          │
│  1. Receives raw complaint text from citizen             │
│  2. Calls NLP Service (POST http://localhost:8001/analyze│
│  3. Maps department to serviceCategories                 │
│  4. Assigns Officer via round-robin (backend state)      │
│  5. Saves complaint in database with NLP metadata        │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼ (HTTP with 5s timeout & fallback)
┌──────────────────────────────────────────────────────────┐
│              Python FastAPI NLP Service (:8001)          │
│                                                          │
│  • Step 1-2 (Aayush): IndicLID + IndicXlit               │
│  • Step 3-4 (Karthik): IndicTrans2 + spaCy Normalizer    │
│  • Step 5-6 (Yshasvee): BART Zero-shot + Urgency + NER   │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Department Category Mapping

The NLP service outputs one of 6 canonical department keys (or `"Unclassified"` if confidence is low):

| NLP Predicted Department | Meaning | NestJS `serviceCategories` (`seed.data.ts`) | Fallback / Review Required |
|---|---|---|---|
| `"roads"` | Potholes, damaged roads, road signs | `Roads & Infrastructure` / `roads` | `false` |
| `"water"` | Water pipeline leaks, no water supply, dirty water | `Water Supply` / `water` | `false` |
| `"drainage"` | Blocked drains, sewage overflow | `Drainage & Sewerage` / `drainage` | `false` |
| `"lighting"` | Street lights not working, power failure | `Electricity & Lighting` / `lighting` | `false` |
| `"sanitation"` | Garbage pile-up, waste collection | `Sanitation & Waste` / `sanitation` | `false` |
| `"green"` | Broken tree branches, park maintenance | `Parks & Horticulture` / `green` | `false` |
| `"Unclassified"` | Low confidence (< 0.40) or ambiguous text | `Unclassified` (or general triage queue) | `true` (`needs_manual_review`) |

> [!IMPORTANT]
> If your NestJS backend uses custom category UUIDs or different string names in `seed.data.ts`, use the mapping dictionary in `nlp-client.service.ts` (shown in Section 4) to map them without modifying the Python service.

---

## 3. FastAPI NLP Service Endpoints

### 3.1 Health Check
* **Endpoint**: `GET http://localhost:8001/health`
* **Response**:
```json
{
  "status": "ready",
  "models_loaded": true
}
```

### 3.2 Analyze Single Complaint
* **Endpoint**: `POST http://localhost:8001/analyze?debug=false`
* **Request Body**:
```json
{
  "text": "meri bijli 2 din se nahi aa rahi hai"
}
```
* **Response Body**:
```json
{
  "original_text": "meri bijli 2 din se nahi aa rahi hai",
  "detected_language": "hin",
  "english_text": "my electricity has not come for the past 2 days",
  "department": "lighting",
  "department_confidence": 0.86,
  "needs_manual_review": false,
  "entities": {
    "duration": "2 days",
    "location": null,
    "previous_complaint": false
  },
  "urgency": {
    "score": 2.0,
    "label": "Medium"
  },
  "processing_time_ms": 320
}
```

---

## 4. NestJS Client Implementation (`nlp.service.ts`)

Drop this service into your NestJS backend (e.g., `src/modules/nlp/nlp.service.ts`):

```typescript
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';

export interface NlpAnalysisResult {
  original_text: string;
  detected_language: string;
  english_text: string;
  department: string;
  department_confidence: number;
  needs_manual_review: boolean;
  entities: {
    duration: string | null;
    location: string | null;
    previous_complaint: boolean;
  };
  urgency: {
    score: number;
    label: 'Low' | 'Medium' | 'High' | 'Critical';
  };
  processing_time_ms?: number;
}

@Injectable()
export class NlpService {
  private readonly logger = new Logger(NlpService.name);
  private readonly nlpServiceUrl = process.env.NLP_SERVICE_URL || 'http://localhost:8001';

  /**
   * Calls the Python FastAPI NLP pipeline.
   * If the service is unreachable or times out (> 5s), fails gracefully
   * so citizen complaint creation is NEVER blocked.
   */
  async analyzeComplaint(text: string): Promise<NlpAnalysisResult> {
    try {
      const response = await axios.post<NlpAnalysisResult>(
        `${this.nlpServiceUrl}/analyze`,
        { text },
        { timeout: 5000 } // 5-second circuit breaker
      );
      return response.data;
    } catch (error) {
      this.logger.warn(`NLP Service unavailable (${error.message}). Falling back to Unclassified.`);
      return {
        original_text: text,
        detected_language: 'unknown',
        english_text: text,
        department: 'Unclassified',
        department_confidence: 0.0,
        needs_manual_review: true,
        entities: { duration: null, location: null, previous_complaint: false },
        urgency: { score: 0, label: 'Low' },
      };
    }
  }

  /**
   * Maps NLP department string to the database Category Name / ID
   */
  mapDepartmentToCategory(nlpDepartment: string): string {
    const categoryMap: Record<string, string> = {
      roads: 'Roads & Infrastructure',
      water: 'Water Supply',
      drainage: 'Drainage & Sewerage',
      lighting: 'Electricity & Lighting',
      sanitation: 'Sanitation & Waste',
      green: 'Parks & Horticulture',
      Unclassified: 'General Review',
    };
    return categoryMap[nlpDepartment] || 'General Review';
  }
}
```

---

## 5. Round-Robin Officer Assignment (NestJS Business Rule)

In accordance with project specifications, **Officer assignment belongs in NestJS**, not in Python NLP models. Here is the recommended handler in your Requests Service:

```typescript
// in requests.service.ts
async assignOfficerToComplaint(departmentCategory: string) {
  // 1. Fetch available department officers for this category from database
  const officers = await this.userRepository.find({
    where: { role: 'DEPARTMENT_OFFICER', department: departmentCategory, isActive: true },
    order: { assignedComplaintsCount: 'ASC' }
  });

  if (!officers || officers.length === 0) {
    return null; // Routed to department unassigned pool
  }

  // 2. Assign to officer with lowest active load (or round-robin)
  const assignedOfficer = officers[0];
  return assignedOfficer;
}
```

---

## 6. How to Start Both Services for Demo

1. **Terminal 1 (Python NLP Service)**:
   ```bash
   cd Citizen-Complaint-Classification-
   .\nlp-env\Scripts\activate
   uvicorn main:app --host 0.0.0.0 --port 8001 --reload
   ```

2. **Terminal 2 (NestJS Backend)**:
   ```bash
   npm run start:dev
   ```
