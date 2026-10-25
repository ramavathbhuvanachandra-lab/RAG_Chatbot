# Project Status & Codebase Audit: RAG_Chatbot
**Date**: October 12, 2026  
**Auditor**: Senior AI Systems Engineer & Repository Auditor  
**Repository**: `RAG_Chatbot` (Branch: `ai-platform`, commit: `ecf4981`)  
**Scope**: Non-destructive, read-only architectural audit and verification baseline

> [!NOTE]
> **Historical Audit Record**: This report documents findings from October 12, 2026.
> For the current project state, active architecture, fresh 50-question baseline (October 13, 2026), and active engineering roadmap, see the canonical [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

---

## 1. Executive Summary

This audit assesses the state of `RAG_Chatbot`, an institution-agnostic retrieval-augmented generation (RAG) platform currently parameterized for the Indian Institute of Technology Jodhpur (IIT Jodhpur) as its reference institution.

### Key Conclusions:
1. **Architectural Split & Runtime Blocker**: The repository contains two competing graph architectures:
   - Canonical implementation in `ai_platform/core/graph/graph.py` (6 cohesive composite nodes).
   - Production entrypoint in `app.py` wiring directly to `backend/graph.py` (13 legacy nodes).
   - **CRITICAL DEFECT**: `backend/graph.py:57` imports `from ai_platform.core.query_pipeline import (process_query,)`, but `ai_platform.core.query_pipeline` does not exist on disk (the actual module is `ai_platform.core.query.pipeline`). **The application (`app.py`) crashes on startup.**
2. **Phase 1 Contracts Status**: **Substantially Complete & Functionally Verified**.
   - Contract test suite (`test_phase1_contracts.py`, `test_phase1_adversarial.py`, `test_phase1_runtime_contract.py`): **12/12 PASS** (100%).
   - Grounding, evidence boundary enforcement, answer packaging, and citation verification adhere to strict mathematical invariants.
3. **Phase 2 Implementation Status**: **NOT INSTALLED (0% Installed)**.
   - Comprehensive codebase grep for `LosslessSemanticQuery`, `LSQC`, `lossless_semantic`, and `SemanticContract` yielded **zero occurrences**.
   - Note: `tests/test_phase2_identity.py` is a misnamed legacy test that imports `backend.retriever` and verifies document identifier integrity; it is not the Lossless Semantic Query Contract (LSQC).
4. **Diagnostic Baseline (50-Question Benchmark)**:
   - Evaluated from `tests/diagnostics_50_canonical.txt`: **43 PASS_STRUCTURAL, 7 CHECK**.
   - The 7 CHECK cases (cases 6, 14, 16, 21, 22, 40, 41) fail due to **channel misrouting or empty channel selection (`channels=[]`)**, yielding zero retrieved chunks and forced system abstentions.
   - 2 structural passes (cases 25 and 33) exhibit semantic hallucination ("first-class not accepted" when evidence indicates it is accepted).
5. **Dead Code**:
   - `ai_platform/core/graph/nodes.py:1025-1035`: Code block following `raise ValueError("Question cannot be empty")` inside `if not question:` is completely unreachable.

---

## 2. Architecture & Execution Flow

### Canonical Flow (`ai_platform/core/graph/graph.py`)
```
                         User Query
                             │
                             ▼
                 [resolve_conversation]
                 (Query history & context)
                             │
                             ▼
                 [understand_query]
          (Intent, entities, institutional scope)
                             │
                             ▼
                 [plan_multi_intent]
     (Sub-queries, metadata filters, channel routing)
                             │
                             ▼
               [retrieve_and_qualify]
     (Channel search, dense/BM25, score qualification)
                             │
                             ▼
                         [answer]
   (Evidence verification, citation checking, packaging)
                             │
                             ▼
                        Final Answer
```

### Production Runtime Graph (`backend/graph.py` via `app.py`)
```
[User Input] ──► [validate_input] ──► [route_query]
                       │
                       ▼
               [expand_context]
                       │
                       ▼
              [retrieve_documents]
                       │
                       ▼
             [process_hybrid_query] ◄── [CRITICAL IMPORT ERROR AT LINE 57]
                       │
                       ▼
             [verify_evidence] ──► [generate_answer] ──► [format_response]
```

### Flow Breakdown & Disconnect
- `app.py` runs FastAPI or Streamlit serving `backend.graph.rag_graph`.
- Canonical logic (`ai_platform/core/`) is only partially invoked via ad-hoc bridge imports in `backend/nodes.py` and `backend/graph.py`.
- The canonical pipeline uses strict `AgentState` typing (`ai_platform/core/graph/state.py`) with `EvidenceBundle`, `QualificationVerdict`, and `GroundingVerification`. The legacy graph uses unstructured dictionary state passing.

---

## 3. Directory & Module Map

### Active Canonical Implementation (`ai_platform/`)
- `ai_platform/core/graph/graph.py`: Canonical compiled LangGraph workflow.
- `ai_platform/core/graph/nodes.py`: `CoreNodes` class encapsulating stage execution.
- `ai_platform/core/graph/state.py`: Typed canonical pipeline state definitions.
- `ai_platform/core/answering/`:
  - `evidence_boundary.py`: Evidence verification and boundary enforcement.
  - `grounding.py`: Fact-checking and claim attribution engine.
  - `packaging.py`: Response structuring, citation formatting, fallback generation.
  - `verification.py`: Contract validator ensuring strict grounding before emit.
- `ai_platform/core/planning/planner.py`: Multi-intent decomposition, channel resolution, query analyzer.
- `ai_platform/core/query/`:
  - `pipeline.py`: Actual query preprocessing pipeline.
  - `parser.py`, `rewriter.py`: Semantic normalization and entity extraction.
- `ai_platform/core/institution/`:
  - `config.py`: Institutional configuration schema.
  - `registry.py`: Dynamic registry for multi-institution onboarding.

### Legacy / Bridge Layer (`backend/`)
- `backend/app.py`: Web service API entrypoint.
- `backend/graph.py`: Legacy 13-node LangGraph workflow (Broken import at line 57).
- `backend/nodes.py`: Legacy nodes bridging into some `ai_platform` classes.
- `backend/retriever/`: Legacy Chroma/BM25 retrieval mechanisms.

### Historical Bundles & Orphans (Not in Canonical Execution)
- `backend_snapshot/`: Point-in-time snapshot of pre-migration backend.
- `rag_v3_final_bundle/`: Standalone delivery package for version 3.
- `retrieval_v4_bundle/`: Standalone retrieval engine; README states explicitly: *"Not wired into production runtime"*.
- `*.bak` files across root: Leftover migration backups.

---

## 4. Component Status Table

| Component | Status | Evidence Class | Primary Files | Remaining Risk / Issues |
|---|---|---|---|---|
| **Production Entrypoint** | ❌ Broken | Class 1 (Source) | `backend/graph.py:57`, `app.py` | Crashes on startup: `ModuleNotFoundError: No module named 'ai_platform.core.query_pipeline'`. |
| **Phase 1 Contracts** | ✅ Verified | Class 2 (Tests) | `ai_platform/core/answering/*` | Fully compliant with 12/12 contract tests passing. |
| **Phase 2 (LSQC)** | ❌ Not Installed | Class 1 (Source) | None (zero files/classes found) | Complete absence of lossless semantic query contracts. |
| **Planner / Router** | ⚠️ Partial Bug | Class 3 (Diagnostics) | `ai_platform/core/planning/planner.py` | 5/50 canonical queries emit `channels=[]`, causing complete retrieval abstention. |
| **Retriever & Qualification** | ⚠️ Partial Bug | Class 1 & 3 | `ai_platform/core/graph/nodes.py:512-700` | Sub-query result fusion drops context when multiple sub-queries target disparate entities. |
| **Canonical LangGraph** | ⚠️ Untested in Prod | Class 1 (Source) | `ai_platform/core/graph/graph.py` | Not wired to `app.py`; runs in isolation via diagnostic test harness. |
| **Dead Code Block** | ⚠️ Defect | Class 1 (Source) | `ai_platform/core/graph/nodes.py:1025-1035` | 11 lines unreachable following unconditional `raise ValueError`. |
| **Multi-Institution Config** | 🟡 Functional Draft | Class 1 (Source) | `ai_platform/core/institution/*`, `INSTITUTION_CHANGE_MAP.md` | Single institution hardcoded in current planner defaults (IIT Jodhpur). |

*Evidence Classes:*
- Class 1: Verified directly in source code.
- Class 2: Executed and verified via test run.
- Class 3: Pre-recorded diagnostic logs in repository.
- Class 4: Systemic inference based on architectural layout.

---

## 5. Phase-by-Phase Progress

### Phase 1: Canonical Foundation & Answering Invariants
- **Target**: Establish verified evidence boundaries, strict answer packaging, attribution, and anti-hallucination verification.
- **Status**: **Substantially Complete**.
- **Evidence**:
  - `tests/diagnostics/test_phase1_contracts.py` (5 tests): PASS.
  - `tests/diagnostics/test_phase1_adversarial.py` (4 tests): PASS.
  - `tests/diagnostics/test_phase1_runtime_contract.py` (3 tests): PASS.
  - All 12/12 tests executed in Python 3.11 conda environment passed cleanly in 0.68s.
  - Verified invariants: Zero ungrounded sentences emitted; non-empty citations mandatory; fallback trigger on qualification failure.

### Phase 2: Lossless Semantic Query Contract (LSQC)
- **Target**: Ensure zero semantic loss during multi-intent decomposition and channel routing; maintain entity-attribute bindings without destructive query rewriting.
- **Status**: **Not Installed (0%)**.
- **Evidence**:
  - Grep across entire repository returned 0 matches for `LosslessSemanticQuery`, `LSQC`, `lossless_semantic`, `SemanticContract`.
  - No contract validation exists between `understand_query` and `retrieve_and_qualify`.
  - Note: `tests/test_phase2_identity.py` was inspected and verified to be a legacy unit test testing document ID format in `backend/retriever/`, not LSQC.

---

## 6. Known Failure Modes

1. **FM-1: Startup Import Crash**
   - *Location*: `backend/graph.py:57`
   - *Cause*: Invalid import path `ai_platform.core.query_pipeline`.
   - *Impact*: Any execution of `app.py` terminates immediately with `ModuleNotFoundError`.
2. **FM-2: Empty Channel Assignment (`channels=[]`)**
   - *Location*: `ai_platform/core/planning/planner.py`
   - *Cause*: Heuristic routing classifier fails on compound academic queries (Cases 14, 21, 22, 40, 41).
   - *Impact*: Downstream retriever receives no target channels, retrieves 0 chunks, forces safe fallback abstention.
3. **FM-3: Dead Code in CoreNodes**
   - *Location*: `ai_platform/core/graph/nodes.py:1025-1035`
   - *Cause*: Post-exception statements inside `if not question:` block.
   - *Impact*: Harmless at runtime but violates static code quality standards.
4. **FM-4: Semantic Hallucination under Low Evidence Threshold**
   - *Location*: Answer generator prompt / thresholding
   - *Evidence*: Benchmark Cases 25 & 33 (e.g., asserting "First class degree is not accepted" when evidence specifies it is required/accepted).
   - *Impact*: Inverted fact generation passing structural checks.
5. **FM-5: Package Isolation & Import Sensitivity**
   - *Location*: Repository Root / `pyproject.toml`
   - *Cause*: `ai_platform` is not installed as an editable package (`pip install -e .`).
   - *Impact*: Tests and scripts fail with `ModuleNotFoundError: No module named 'ai_platform'` unless `PYTHONPATH=.` is manually prepended.
6. **FM-6: SC/ST Threshold Evidence Incompleteness**
   - *Evidence*: Benchmark Cases 30–32.
   - *Cause*: Ingestion corpus lacks explicit departmental cutoff tables for reserved categories.
   - *Impact*: Correct abstention ("not specified in evidence"), but user query intent remains unserved.
7. **FM-7: Architecture Divergence**
   - *Location*: `backend/graph.py` vs `ai_platform/core/graph/graph.py`
   - *Cause*: Dual implementations maintained concurrently.
   - *Impact*: Enhancements made to `ai_platform` are bypassed by legacy production callers.
8. **FM-8: Disconnected Retrieval Bundles**
   - *Location*: `retrieval_v4_bundle/`
   - *Cause*: Experimental retrieval improvements packaged as external bundle without production wiring.
   - *Impact*: Code drift and wasted maintenance surface.

---

## 7. Test and Diagnostic Results

### 1. Active Test Suite Execution (2026-10-12)
- **Command**:
  ```bash
  PYTHONPATH=. pytest tests/diagnostics/test_phase1_contracts.py \
                      tests/diagnostics/test_phase1_adversarial.py \
                      tests/diagnostics/test_phase1_runtime_contract.py -v
  ```
- **Outcome**: **12 Passed in 0.68s**
  - `test_phase1_contracts.py`: 5 passed
  - `test_phase1_adversarial.py`: 4 passed
  - `test_phase1_runtime_contract.py`: 3 passed

### 2. Pre-Recorded Diagnostic Analysis (`tests/diagnostics_50_canonical.txt`)
- **Total Queries**: 50
- **Summary**:
  - `PASS_STRUCTURAL`: 43 / 50 (86%)
  - `CHECK`: 7 / 50 (14%)
- **Detailed CHECK breakdown**:
  - Cases 14, 21, 22, 40, 41: `channels=[]` routing failure → 0 evidence retrieved → safe fallback.
  - Case 6: Sub-query coordination failure on dual-topic query.
  - Case 16: Multi-intent threshold underflow.

### 3. Forensic Log Verification (`reports/forensic_*.txt`)
- `forensic_stage2_summary_2026-10-06.txt`: 12 test cases evaluated; 12/12 evaluated GOOD with 0 invariant breaches.
- `forensic_production_run_summary_2026-10-06.txt`: 5 test cases evaluated; all 5 confirmed EVIDENCE_COVERAGE FIT.

---

## 8. Five Highest-Priority Engineering Risks

1. **Risk 1 (CRITICAL): Production Service is Inoperable**
   - The primary application entrypoint (`app.py` -> `backend/graph.py`) cannot launch due to a broken module reference (`ai_platform.core.query_pipeline`). Any deployment or container startup will crash instantly.
2. **Risk 2 (HIGH): Architectural Divergence / Dual Codebase Drift**
   - Core development is occurring in `ai_platform/core/`, while runtime routing is still anchored in legacy `backend/`. This leads to false confidence where passing unit tests in `ai_platform` do not reflect runtime production behavior.
3. **Risk 3 (HIGH): Silent Information Loss via Empty Routing (`channels=[]`)**
   - 10% of benchmark queries fail silently into abstention because the query planner fails to identify a matching channel. The system safely falls back, but user utility is compromised.
4. **Risk 4 (MEDIUM): Inverted Facts Passing Structural Invariants**
   - Cases 25 and 33 demonstrate that answers can satisfy Phase 1 structural and citation invariants while conveying contradictory facts ("first class not accepted").
5. **Risk 5 (MEDIUM): Non-Standardized Environment Setup**
   - Absence of an editable `pyproject.toml` or `setup.py` forces hardcoded `PYTHONPATH` dependencies, making CI/CD pipelines and developer onboarding brittle.

---

## 9. Recommended Next-Step Sequence

To transition the repository to a reliable, deployable state, execute the following steps in strict chronological order:

```
[Step 1: Fix Import Crash]
          │
          ▼
[Step 2: Add pyproject.toml]
          │
          ▼
[Step 3: Unify Runtime Graph to Canonical ai_platform]
          │
          ▼
[Step 4: Fix Planner channels=[] Empty Routing]
          │
          ▼
[Step 5: Run 50Q Regression Suite]
          │
          ▼
[Step 6: Implement Phase 2 (LSQC)]
```

### Step 1: Repair Critical Runtime Import (Immediate)
- Correct `backend/graph.py:57` to import from `ai_platform.core.query.pipeline` instead of `ai_platform.core.query_pipeline`. Verify `app.py` can load its application graph without import errors.

### Step 2: Formalize Packaging via `pyproject.toml`
- Add a minimal `pyproject.toml` exposing `ai_platform` and `backend` as editable packages. Eliminate the requirement for manual `PYTHONPATH` exports.

### Step 3: Unify Runtime Entrypoint to Canonical Graph
- Deprecate `backend/graph.py` and redirect `app.py` to invoke `ai_platform.core.graph.graph.app`. Eliminate the 13-node legacy graph to ensure all runtime calls benefit from Phase 1 contracts.

### Step 4: Resolve Planner Routing Default
- In `ai_platform/core/planning/planner.py`, add a fallback channel heuristic: if classifier outputs `channels=[]`, default to institutional general search (`channels=["general", "academic"]`) rather than empty list.

### Step 5: Run Full 50-Question Benchmark Regression
- Re-run the canonical diagnostic benchmark to confirm that the 7 CHECK cases resolve into proper retrieval and answer qualification.

### Step 6: Begin Phase 2 Implementation (LSQC)
- Proceed with the Lossless Semantic Query Contract implementation once the foundation is stabilized.

---

## 10. Phase 2 Recommendation & Technical Specification

Phase 2 must be developed as an **additive, strictly typed contract** between `understand_query` and `retrieve_and_qualify`.

### Proposed Contract Architecture:
1. **Contract Definition (`ai_platform/core/planning/semantic_contract.py`)**:
   - Class `LosslessSemanticQuery`:
     - `raw_query`: str
     - `atomic_intents`: List[str]
     - `entity_bindings`: Dict[str, List[str]] (preserves exact institutional codes like "M.Tech CSE", "SC/ST", "CPI 6.5")
     - `temporal_constraints`: Optional[Dict[str, str]]
     - `lossless_invariants_hash`: str
2. **Integration Touchpoints**:
   - `ai_platform/core/planning/planner.py`: Planner must generate `LosslessSemanticQuery` alongside routing channels.
   - `ai_platform/core/graph/nodes.py` (`retrieve_and_qualify`): Retrieve across all atomic intents without dropping entity-attribute bindings.
   - `ai_platform/core/answering/verification.py`: Validate that every entity binding in the contract is addressed in either evidence qualification or safe fallback.
3. **Acceptance Criteria**:
   - 0% entity drop between query ingestion and retriever query generation.
   - 100% pass on 50Q benchmark with zero `channels=[]` abstentions.
   - Existing Phase 1 test suite (12/12) continues to pass without modification.
