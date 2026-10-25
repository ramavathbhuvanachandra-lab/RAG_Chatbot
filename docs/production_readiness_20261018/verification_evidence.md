# IIT Jodhpur RAG Chatbot — Verification Evidence
**Date:** October 18, 2026  
**Artifact Directory:** `docs/production_readiness_20261018/`  
**Diagnostic Trace Artifacts:** `test_outputs/rag_diagnostics_20261018/traces.json`, `test_outputs/rag_diagnostics_20261018/diagnostic_report.md`

---

## 1. Automated Test Execution Evidence

All automated test suites were executed against the canonical `ai_platform` runtime using Python 3.14.2 on macOS (Darwin arm64).

### Test Suite Results

```text
============================= test session starts ==============================
platform darwin -- Python 3.14.2, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/bhuvanachandra/Desktop/RAG_Chatbot
plugins: anyio-4.12.1, langsmith-0.6.3
collected 58 items

tests_frontend/test_frontend_core.py .................                   [ 29%]
tests_frontend/test_milestone1_1_conversation_quality.py ............... [ 55%]
.......                                                                  [ 67%]
tests_reusable/test_streamlit_canonical_runtime.py ....                  [ 74%]
tests/test_conversation_resolver.py ......                               [ 84%]
tests/test_answer_guard.py .........                                     [100%]

================== 58 passed, 1 warning in 101.59s (0:01:41) ===================
```

### Breakdown by Component

1. **Frontend Core & Client Layer (`tests_frontend/test_frontend_core.py`):** 17/17 PASSED
   - Multi-tenant configuration validation, display names, and tenant switching isolation.
   - Session manager conversation uniqueness and bounded history truncation.
   - Assistant client response adaptation and safe empty query handling.
   - Architecture hygiene: zero Supabase imports, zero legacy backend imports in frontend.

2. **Milestone 1.1 Conversational Quality (`tests_frontend/test_milestone1_1_conversation_quality.py`):** 22/22 PASSED
   - Multiturn resolution with program and category preservation.
   - Anaphora resolution without session boundary crossing.
   - Answer completeness, clean sentence boundary trimming, and extractive repair.
   - Suggested follow-up questions: 0 additional LLM calls, suppressed on safe refusals and errors.

3. **Streamlit Canonical Runtime Integration (`tests_reusable/test_streamlit_canonical_runtime.py`):** 4/4 PASSED
   - Canonical engine imports in `UI/chat.py` and `app.py`.
   - RAG state graph contract conformance.
   - End-to-end graph invocation and turn execution.

4. **Conversation Resolver Contract (`tests/test_conversation_resolver.py`):** 6/6 PASSED
   - Standalone question preservation.
   - Follow-up resolution without false topic switches.
   - Safe fallback when ambiguous references cannot be resolved.

5. **Answer Guard Contract (`tests/test_answer_guard.py`):** 9/9 PASSED
   - Internal document path and RRF terminology leak detection.
   - Generic contact fallback removal while preserving valid answers.
   - Safe unknown response generation on empty or invalid text.

---

## 2. End-to-End Diagnostic Sequence Verification

The diagnostic reproduction script `scripts/run_rag_diagnostics_phase_bc.py` was executed across all 5 user interaction sequences through the canonical `handle_chat_turn` pipeline.

### Sequence 1: Programme-specific Fees (M.Tech)
- **Session:** `session_seq_1_fees`
- **Turn 1:** *"What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. programme at IIT Jodhpur?"*
  - **Result:** ₹2,25,000 SME fee completely excluded. Returns official semester fee of ₹32,250/- with refundable deposit and experiential fee breakdown.
  - **Latency:** 66.48s | **Sources Rendered:** 32 | **Refusal:** False
- **Turn 2:** *"What about sponsored students?"*
  - **Result:** Successfully retains M.Tech programme context and addresses sponsored student regulations. Quotes official Registration Kept Alive (RKA) tuition parity rule for sponsored candidates. No ₹2,25,000 substitution.
  - **Latency:** 29.96s | **Sources Rendered:** 27 | **Refusal:** False
- **Turn 3:** *"Does that include hostel and mess charges?"*
  - **Result:** Accurately explains institutional breakdown: semester fee includes hostel fees, while mess charges require a separate mess advance each semester. No startup incubation charges quoted.
  - **Latency:** 26.39s | **Sources Rendered:** 26 | **Refusal:** False

### Sequence 2: Campus Dining & Mess Facilities
- **Session:** `session_seq_2_dining`
- **Turn 1:** *"Where can I find food on campus?"*
  - **Result:** Accurately identifies separate dining facilities (vegetarian and non-vegetarian messes) on campus. Explains that meal charges are paid directly to the mess vendor at the mess counter. Excludes all incubation cafeteria or startup text.
  - **Latency:** 28.39s | **Sources Rendered:** 32 | **Refusal:** False
- **Turn 2:** *"Tell me about the mess."*
  - **Result:** Explains advance mess fee payment structure per semester and vendor payment policy.
  - **Latency:** 28.51s | **Sources Rendered:** 32 | **Refusal:** False

### Sequence 3: Departmental Research Areas (Electrical Engineering)
- **Session:** `session_seq_3_ee`
- **Turn 1:** *"What research areas are available in Electrical Engineering?"*
  - **Result:** Cleanly focused on Electrical Engineering research themes: Organic and Flexible Electronics (FETs, sensors), Sociodigital Reality (AR/VR/Haptics), High-Speed VLSI Systems, Neuromorphic Computing, Signal Integrity. Unrelated departments (Computer Science, Chemical Engineering, Materials) completely excluded.
  - **Latency:** 31.90s | **Sources Rendered:** 32 | **Refusal:** False

### Sequence 4: Unsupported Legal Premise (Supreme Court Degree Cancellation)
- **Session:** `session_seq_4_sc_premise`
- **Turn 1:** *"Why did the Supreme Court order IIT Jodhpur to cancel all degrees awarded in 2024?"*
  - **Result:** Safely refuses the unsubstantiated premise without affirming it or claiming it is outside the educational domain: *"The available official sources do not substantiate or contain information regarding any Supreme Court order to cancel degrees awarded by IIT Jodhpur. I cannot verify this claim from the available institutional documents."*
  - **Sources Exposed in UI:** **0** (Citation leakage completely eliminated).
  - **Latency:** 19.25s | **Refusal:** True | **Evidence Status:** `refusal`

### Sequence 5: Context Isolation & Vague Fresh-Session Query
- **Session:** `session_seq_5_fresh_isolation`
- **Turn 1:** *"What is the fee for this programme?"* (Fresh session, empty history)
  - **Result:** Promptly requests clarification: *"Could you please specify which programme you are inquiring about (for example, B.Tech., M.Tech., MBA, M.Sc., or Ph.D.) so I can provide the accurate fee details?"*
  - **Hallucinated Amount:** None (0 guessing).
  - **Latency:** **0.00s** (Bypasses LLM inference via conversational understanding gate).
  - **Sources Exposed:** **0**.

---

## 3. Latency & Resource Analysis

| Sequence | Query | Pre-Investigation Latency | Current Latency | Status |
| :--- | :--- | :--- | :--- | :--- |
| `sequence_1` T1 | M.Tech tuition fee | 98.4s | 66.48s | Reduced (single model call + verification caching) |
| `sequence_1` T2 | Sponsored students | 84.1s | 29.96s | **64% Latency Reduction** |
| `sequence_1` T3 | Hostel & mess inclusion | 92.6s | 26.39s | **71% Latency Reduction** |
| `sequence_2` T1 | Food on campus | 78.5s | 28.39s | **64% Latency Reduction** |
| `sequence_2` T2 | Tell me about mess | 81.2s | 28.51s | **65% Latency Reduction** |
| `sequence_3` T1 | EE research areas | 114.7s | 31.90s | **72% Latency Reduction** |
| `sequence_4` T1 | Supreme Court cancellation | 65.3s | 19.25s | **70% Latency Reduction** |
| `sequence_5` T1 | What is fee for this programme | 74.8s | **0.00s** | **100% Latency Reduction** (Fast clarification exit) |
