# V1 Stage Diagnosis V2

> [!NOTE]
> **Historical Forensic Record**: This document captures early Stage 2 retrieval diagnosis.
> For current canonical platform state and benchmark results, see [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

Cases: **15**

## Core questions

| Measurement | Result |
|---|---:|
| Dense found relevant | 0/15 |
| BM25 found relevant | 0/15 |
| RRF retained relevant | 0/15 |
| Relevant candidate verified | 0/15 |
| Relevant candidate uncertain | 0/15 |
| Relevant candidate rejected | 0/15 |
| Relevant candidate missing from verification buckets | 0/15 |
| Ranking preserved relevant | 0/15 |
| Evidence boundary preserved relevant | 0/15 |
| Package ready | 0/15 |
| Final context non-empty | 0/15 |

## Case table

| Case | Dense | BM25 | RRF | Verification | Rank | Evidence | Scope removed | Package | Context |
|---|---|---|---|---|---:|---|---:|---|---|
| mtech_regular_eligibility | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| mtech_percentage | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| mtech_gate | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| mtech_qualification | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| mtech_written_test | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| mtech_work_experience | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| msc_eligibility | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| msc_percentage | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| msc_jam | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| msc_bachelor_route | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| phd_eligibility | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| phd_application_fee | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| mba_steps | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| registration_documents | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |
| registration_provisional | ❌ | ❌ | ❌ | — | — | ❌ | 0 | ❌ | ❌ |

## First failures

- `resolve_conversation_node`: 15

## Interpretation rules

- Dense/BM25 answer whether each raw lane surfaced benchmark-relevant evidence.
- RRF answer whether that evidence remained in `fused_candidates`.
- Verification reports the actual bucket containing the relevant candidate.
- Ranking reports whether the relevant candidate remains in `ranked_candidates`.
- Evidence grouping and expansion are measured separately from scope filtering.
- Claim auditing is measured as an audit stage; the current graph does not expose a separate claim-filter node.
- Final context is saved separately per case under `final_context/`.

JSON report: `/Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/test_diagnosis/reports/stage_diagnosis_v2/diagnosis_v2.json`