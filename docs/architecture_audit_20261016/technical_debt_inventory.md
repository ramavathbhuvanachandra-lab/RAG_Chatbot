# Technical Debt & Architecture Defect Inventory

**Audit Date**: October 16, 2026  
**Auditor**: Senior AI Systems Architect & Code Auditor  
**Audit Scope**: Repository-wide code inspection, import analysis, and configuration audit  

---

## 1. Executive Summary of Technical Debt

The core RAG engine in `ai_platform/` has achieved high structural reliability (98%+ recall, 96%+ candidate verification, 0% control false acceptances). However, the repository has accumulated significant architectural technical debt from rapid evolutionary prototyping, including **legacy UI couplings**, **redundant subsystem copies**, **untracked backup clutter**, and **single-tenant hardcoding**.

---

## 2. Structured Technical Debt Findings

| Finding ID | Severity | Category | Confidence | Impact Summary |
| :---: | :---: | :--- | :---: | :--- |
| **TD-01** | **CRITICAL** | Legacy UI Coupling | **CONFIRMED** | `UI/chat.py`, `login.py`, and `dashboard.py` import obsolete `backend/` packages. |
| **TD-02** | **HIGH** | Repository Bloat | **CONFIRMED** | Over 30 untracked backup files (`.bak`, `.stage2_*`, `.zip`) clutter the active working tree. |
| **TD-03** | **HIGH** | Dead Architecture | **CONFIRMED** | Monolithic `backend/` directory (50+ files) remains in tree, causing developer confusion. |
| **TD-04** | **HIGH** | Subsystem Duplication| **CONFIRMED** | Parallel `ai_platform/core/retrieval_v2/` exists alongside canonical `retrieval/`. |
| **TD-05** | **MEDIUM** | Runtime Exceptions | **CONFIRMED** | Missing optional dependencies (`supabase`, `backend.institutions`) trigger console tracebacks. |
| **TD-06** | **MEDIUM** | Multi-Tenant Blocker| **CONFIRMED** | `ai_platform/runtime/config.py` hardcodes single-tenant collection `iitj_v1` and data paths. |
| **TD-07** | **LOW** | Stale Documentation | **CONFIRMED** | Root `README.md` and historical diagnostic reports describe obsolete architectures. |

---

## 3. Detailed Technical Debt Specifications

### TD-01: Auxiliary UI Modules Import Quarantined `backend/` (CRITICAL)
- **Evidence**:
  - `UI/chat.py:45`: `from backend.assistant_router import assistant_router`
  - `UI/chat.py:49`: `from backend.conversation_session import ensure_chat_session, load_persistent_chat_history...`
  - `UI/login.py:3`: `from backend.authentication import authenticate_user`
  - `UI/dashboard.py:3`: `from backend.dashboard_db import ...`
- **Confidence**: **CONFIRMED** (Verified by source grep and test output inspection).
- **User / System Impact**:
  - Prevents the removal of `backend/`.
  - Causes console tracebacks during UI startup.
  - Couples session memory to obsolete storage schemas instead of the canonical `ai_platform` runtime.
- **Recommended Action**:
  - Introduce canonical `ai_platform/runtime/session.py` to handle session memory and chat history formatting.
  - Refactor `UI/chat.py`, `UI/login.py`, and `UI/dashboard.py` to import exclusively from `ai_platform/runtime/`.
- **Runtime Confirmation Required**: No (proven via static import graph).

---

### TD-02: Accumulation of 30+ Untracked Backup & Scratch Files (HIGH)
- **Evidence**:
  - In `ai_platform/core/graph/`: `nodes.py.stage2_hotfix_v2_backup_20261005_172002`, `nodes.py.phase1_v5_backup_20261006_062022`, `nodes.py.before_v41.bak`, and 10 other backup files.
  - In `ai_platform/core/retrieval/`: 8 versioned backup files.
  - In root: `rag_v3_final_bundle.zip`, `rag_v3_final_bundle/`, `retrieval_v4_bundle/`, `phase1_backup_*/`.
- **Confidence**: **CONFIRMED** (Verified via `git status --short`).
- **User / System Impact**:
  - Severe cognitive overhead for developers.
  - Risk of accidental imports from backup files or editing the wrong file.
  - Clutters grep and global code search results.
- **Recommended Action**:
  - Safely archive all backup files into an external git-ignored archive directory (e.g. `archive/backups_202610/`) and clean the working tree.
- **Runtime Confirmation Required**: No.

---

### TD-03: Quarantined Monolithic `backend/` Directory Retained in Root (HIGH)
- **Evidence**:
  - Directory `backend/` contains 55 Python files representing the legacy pre-modular prototype (`backend/graph.py`, `backend/retriever.py`, `backend/nodes.py`, etc.).
  - `ai_platform/core/` has zero active imports from `backend/`.
- **Confidence**: **CONFIRMED** (Verified via directory listing and import analysis).
- **User / System Impact**:
  - New contributors or coding agents frequently mistake `backend/` for the active production engine.
  - Duplication of models and prompts between `backend/` and `ai_platform/`.
- **Recommended Action**:
  - Once TD-01 is completed, deprecate and move `backend/` into `legacy_quarantine/backend/` or remove it entirely from production deployments.
- **Runtime Confirmation Required**: No.

---

### TD-04: Subsystem Duplication: `retrieval_v2/` vs `retrieval/` (HIGH)
- **Evidence**:
  - `ai_platform/core/retrieval/`: Active canonical retrieval package used by `nodes.py` (contains `hybrid.py`, `rrf.py`, `verification.py`, `ranking.py`).
  - `ai_platform/core/retrieval_v2/`: Parallel implementation containing `retrieve.py`, `rank.py`, `evidence_units.py`, `models.py`.
- **Confidence**: **CONFIRMED** (Verified by checking imports in `ai_platform/core/graph/nodes.py`).
- **User / System Impact**:
  - Confusion as to which retrieval module is canonical.
  - Dead code maintenance burden.
- **Recommended Action**:
  - Audit any unique utility functions in `retrieval_v2/`, merge necessary components into `retrieval/`, and delete `retrieval_v2/`.
- **Runtime Confirmation Required**: No.

---

### TD-05: Missing Optional Dependencies Triggering Console Tracebacks (MEDIUM)
- **Evidence**:
  - Console warning on every chat session initialization:
    `Chat history load failed for session test_session_123: No module named 'supabase'`
  - Console warning on emergency lookup:
    `Failed to load institution service backend.institutions.iitj.emergency.find_emergency: ModuleNotFoundError`
- **Confidence**: **CONFIRMED** (Captured during execution of `test_streamlit_canonical_runtime.py`).
- **User / System Impact**:
  - Degrades developer experience and clutters log files.
  - Masks genuine application errors with noisy stack traces.
- **Recommended Action**:
  - Make external persistence providers (Supabase) cleanly optional with silent local fallbacks unless explicitly configured via environment variables.
  - Fix emergency service routing to point to `ai_platform.institutions.iitj.emergency`.
- **Runtime Confirmation Required**: No.

---

### TD-06: Hardcoded Single-Tenant Configuration (MEDIUM)
- **Evidence**:
  - `ai_platform/runtime/config.py`:
    `CHROMA_COLLECTION = "iitj_v1"`
    `DATA_DIR = Path("data/data_iitj/iitj_rag_v1_docs_production")`
- **Confidence**: **CONFIRMED** (Verified by viewing `config.py`).
- **User / System Impact**:
  - Blocks zero-code onboarding of new institutions.
  - Prevents multi-tenant vector store partitioning.
- **Recommended Action**:
  - Refactor `config.py` to support dynamic institutional collection names (`f"{institution_id}_v1"`) and data paths.
- **Runtime Confirmation Required**: No.

---

### TD-07: Stale Documentation & Historical Benchmark Claims (LOW)
- **Evidence**:
  - Root `README.md` describes the old monolithic architecture and lacks mention of `ai_platform/`.
  - Historical reports in `reports/` and `tests_reusable/test_diagnosis/` describe resolved defects (e.g. C2 polarity inversion, adm14 planner misclassification) without cross-linking to their verified solutions.
- **Confidence**: **CONFIRMED**.
- **User / System Impact**:
  - Misleading documentation for users and engineers reading the repository root.
- **Recommended Action**:
  - Update root `README.md` to reference `flow.md` and the modular `ai_platform/` runtime.
- **Runtime Confirmation Required**: No.
