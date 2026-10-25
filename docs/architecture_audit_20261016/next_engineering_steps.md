# Next Engineering Steps & Roadmap

**Audit Date**: October 16, 2026  
**Auditor**: Senior AI Systems Architect & RAG Reliability Engineer  

---

## 1. Top Five Ranked Engineering Initiatives

The following five initiatives are prioritized based on **reliability impact**, **implementation effort**, **regression risk**, and **system hygiene**:

| Rank | Initiative | Expected Reliability Impact | Implementation Effort | Regression Risk | Dependencies | Safe During Active Eval? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Complete UI Decoupling from `backend/`** | **HIGH** | Low (1–2 engineer days) | **LOW** | None | **NO** (Touch UI files) |
| **2** | **Complete Live Local Ollama 180Q Generation** | **HIGH** | Medium (Compute-bound: 2–3 hours) | **VERY LOW** | Local Ollama | **NO** (Evaluation running) |
| **3** | **Working Tree Cleanup & Backup Archival** | **MEDIUM** | Low (2–3 hours) | **VERY LOW** | None | **YES** (Non-destructive) |
| **4** | **Multi-Tenant Vector Store Partitioning** | **HIGH** | Medium (3–4 engineer days) | **MEDIUM** | Ingestion pipeline | **NO** (Modifies ChromaDB) |
| **5** | **Automated CI Contract Gating** | **MEDIUM** | Low (1 engineer day) | **LOW** | GitHub Actions | **YES** (CI config only) |

---

## 2. Initiative Detail & Specification

### Priority 1: Complete UI Decoupling from `backend/` (Session & Router Migration)
- **Objective**: Eliminate all residual `from backend.*` imports inside `UI/chat.py`, `UI/login.py`, and `UI/dashboard.py`.
- **Implementation Strategy**:
  1. Create [`ai_platform/runtime/session.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/runtime/session.py):
     - Implement lightweight, robust in-memory conversation memory with session ID scoping.
     - Provide optional pluggable persistence (e.g. SQLite / Supabase) with graceful silent local fallbacks.
  2. Update `UI/chat.py`:
     - Replace `from backend.conversation_session import ...` with `from ai_platform.runtime.session import ...`.
     - Route emergency lookups directly to `ai_platform.institutions.iitj.emergency`.
  3. Validate using `tests_reusable/test_streamlit_canonical_runtime.py`.

### Priority 2: Complete Live Local Ollama Generation for Phase B (180Q)
- **Objective**: Finish generating genuine, un-hallucinated live local Ollama answers for all 180 questions across Groups A, B, and C.
- **Implementation Strategy**:
  - Continue running resumable batches of 15 cases using the bounded evaluation runner:
    ```bash
    PYTHONPATH=. python3 tests_reusable/rag_quality/run_production_reliability_eval.py \
      --phase B --mode generate \
      --out-dir test_outputs/rag_quality/guard_evidence_close_20261016_153200/phase_b_eval \
      --max-live-cases 15 --resume
    ```
  - Verify 100% completion in `structural_summary.json` and `questions_and_answers.txt`.

### Priority 3: Working Tree Cleanup & Backup Archival
- **Objective**: Prune 30+ untracked backup files and redundant `retrieval_v2/` module tree.
- **Implementation Strategy**:
  - Archive `.bak`, `.backup_*`, and `.stage2_*` files into a `.gitignore`d `archive/backups_202610/` folder.
  - Verify that no active imports break by running the 13 regression tests.

### Priority 4: Multi-Tenant Vector Store Partitioning
- **Objective**: Enable zero-code onboarding of new universities by parameterizing vector collections and data directories.
- **Implementation Strategy**:
  - In `ai_platform/runtime/vectorstore.py`, accept `institution_id` and query collection `{institution_id}_v1`.
  - Enforce tenant metadata filter `where={"institution_id": institution_id}` on all hybrid queries.

### Priority 5: Automated CI Contract Gating
- **Objective**: Prevent regressions in query planning, guard security, and candidate verification.
- **Implementation Strategy**:
  - Configure GitHub Actions workflow running `pytest tests_reusable/rag_quality/test_guard_and_evidence_hardening_regression.py` on pull requests.

---

## 3. The Single Recommended Next Engineering Action

**Recommended Action**: **Execute Priority 1 — Decouple `UI/chat.py` from `backend/` by introducing canonical session management in `ai_platform/runtime/session.py`.**

### Rationale:
1. **High Architectural Value**: Bridges the final gap between the user-facing Streamlit application and the modular platform, completing the quarantine of `backend/`.
2. **Immediate UX Improvement**: Completely silences the confusing console tracebacks (`supabase` and `backend.institutions`) currently printed on every user turn.
3. **Zero Core Risk**: Operates entirely in the UI integration boundary without altering the verified query planning, retrieval, verification, or generation contracts in `ai_platform/core/`.
4. **Clean Verification**: Can be immediately validated using the existing integration test suite `tests_reusable/test_streamlit_canonical_runtime.py`.
