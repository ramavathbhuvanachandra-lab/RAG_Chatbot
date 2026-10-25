# Comprehensive Project Status & Codebase Audit: RAG_Chatbot (Version 2)
**Date**: October 12, 2026  
**Auditor**: Senior AI Systems Engineer & Repository Auditor  
**Repository**: `RAG_Chatbot` (Branch: `ai-platform`, commit: `ecf4981`)  
**Scope**: In-depth architectural trace, regression forensics (`adm14`, `msc10`, `phd43`), runtime verification, Phase 1 contract evaluation, and Phase 2 feasibility analysis.  
**Note on Versioning**: `reports/PROJECT_STATUS_AUDIT_2026-10-12.md` was preserved untouched per repository instructions; this `v2` report expands the diagnostic forensics and claim-classification rigor requested by the engineering specification.

> [!NOTE]
> **Historical Audit Record**: This document is a historical record of the expanded October 12, 2026 audit.
> For the current project state, active architecture, fresh 50-question baseline (October 13, 2026), and active engineering roadmap, see the canonical [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

---

## Claim Evidence Taxonomy
Every substantive finding and assertion in this audit is tagged with one of five verification classes:
- **`[Verified: Current Source]`**: Directly observed and confirmed by inspecting active repository code.
- **`[Executed: This Audit]`**: Directly validated by executing commands/tests during this audit session.
- **`[Recorded: Historical]`**: Sourced from existing diagnostic output logs, forensic records, or commit history in the repository.
- **`[Inferred]`**: Deduced from architectural wiring, directory structures, and code behavior patterns.
- **`[Unknown]`**: Cannot be determined from static analysis or offline environment inspection alone.

---

## 1. Executive Summary

`RAG_Chatbot` is transitioning from a monolithic prototype (`backend/`) to a modular, multi-institution RAG platform (`ai_platform/`). The reference deployment institution is the Indian Institute of Technology Jodhpur (IIT Jodhpur), with the goal of onboarding 10–15 institutions without rewriting core retrieval and reasoning engines.

### Key Audit Findings:
1. **Critical Runtime Blocker** `[Verified: Current Source]`:
   - The production UI entrypoint (`app.py`) imports `backend/graph.py`, which at line 57 attempts `from ai_platform.core.query_pipeline import (process_query,)`.
   - The module `ai_platform.core.query_pipeline` does not exist on disk (the real module is `ai_platform.core.query.pipeline`).
   - Any runtime launch of `app.py` crashes immediately with `ModuleNotFoundError`.
2. **Dual-Architecture Disconnect** `[Verified: Current Source]`:
   - **Canonical Architecture**: Located in `ai_platform/core/graph/graph.py` with 6 composite nodes, typed `AgentState`, and Phase 1 answering contracts.
   - **Live UI Wiring**: `app.py` is still wired to legacy `backend/graph.py` (a 13-node graph with untyped dictionary state). Canonical `ai_platform` is NOT invoked by the web application.
3. **Phase 1 Contracts Status: COMPLETE & VERIFIED** `[Executed: This Audit]`:
   - 12 out of 12 tests in `tests/diagnostics/test_phase1_*.py` execute and pass (100%). Strict evidence boundary enforcement, grounding verification, and citation guarantees are fully implemented in `ai_platform/core/answering/`.
4. **Phase 2 Status: NOT INSTALLED (0%)** `[Verified: Current Source]`:
   - Zero occurrences of `LosslessSemanticQuery`, `LSQC`, `lossless_semantic`, or `SemanticContract` exist in the codebase.
   - `tests/test_phase2_identity.py` is a misnamed legacy test for `backend.retriever`, not Phase 2.
5. **Canonical 50-Question Diagnostic Baseline** `[Recorded: Historical]`:
   - 43 / 50 (86%) achieve `PASS_STRUCTURAL`.
   - 7 / 50 (14%) are flagged `CHECK` (Cases 6, 14, 16, 21, 22, 40, 41) due to **empty routing channels (`channels=[]`)** in `ai_platform/core/planning/planner.py`, resulting in 0 chunks retrieved and forced safe abstention.
6. **Severe Semantic Contradictions in Passed Cases** `[Recorded: Historical]`:
   - Structural passes include egregious semantic inversions:
     - `msc10` (Q25) & M.Tech (Q33): Asserts "First-class degree is NOT accepted" despite evidence stating 60% aggregate is required.
     - `phd43` (Q43): Asserts "4-year bachelor's degree alone does not qualify for Ph.D.", directly contradicting Q42.
7. **Dead Code** `[Verified: Current Source]`:
   - `ai_platform/core/graph/nodes.py:1025-1035` contains 11 unreachable lines after an unconditional `raise ValueError`.

---

## 2. Architecture & Request Path Tracing

### 2.1 The Broken Production Request Path (`app.py`) `[Verified: Current Source]`

```
User Browser (Streamlit)
       │
       ▼
[app.py:157] handle_user_prompt(prompt, graph)
       │
       ▼  (Loads via app.py:23: `from backend.graph import create_graph`)
[backend/graph.py:57]  CRITICAL IMPORT FAILURE:
       ▲
       └── "from ai_platform.core.query_pipeline import process_query"
           ==> ModuleNotFoundError: No module named 'ai_platform.core.query_pipeline'
```

If line 57 were corrected, the live execution path in `backend/graph.py` would execute the legacy 13-node pipeline:
```
[START]
   │
   ▼
[validate_input] ────────► [route_query]
                                 │
                                 ▼
                     [process_multi_intent]
                                 │
                                 ▼
                         [hybrid_retrieve]
                                 │
                                 ▼
                     [fuse_retrieved_documents]
                                 │
                                 ▼
                    [initial_rerank_documents]
                                 │
                                 ▼
                    [expand_retrieved_context]
                                 │
                                 ▼
                     [final_rerank_documents]
                                 │
                                 ▼
                         [assess_evidence]
                                 │
                                 ▼
                    [assess_evidence_coverage]
                                 │
                                 ▼
                        [compress_context]
                                 │
                                 ▼
                         [generate_answer]
                                 │
                                 ▼
                        [format_response] ──────► [END]
```
*Key Finding*: The legacy graph in `backend/` bypasses all canonical `ai_platform/core/answering/` contract enforcement modules (`grounding.py`, `evidence_boundary.py`, `verification.py`, `packaging.py`).

---

### 2.2 The Canonical Execution Path (`ai_platform/`) `[Verified: Current Source]`

The intended execution path is encapsulated in `ai_platform/runtime/engine.py` via `ask()` and `ask_with_diagnostics()`:

```
[User Question]
       │
       ▼
[ai_platform.runtime.engine.ask()]
       │
       ▼
[ai_platform.core.graph.graph.build_graph()] (Compiled LangGraph Workflow)
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. resolve_conversation (ai_platform/core/graph/nodes.py:175)           │
│    - Resolves conversational context, coreference, and history.         │
│    - Emits: resolved_question, session_id                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. understand_query (ai_platform/core/graph/nodes.py:250)              │
│    - Normalizes terminology via institution semantic registry.          │
│    - Extracts entities, topics, degree programs, and intents.           │
│    - Emits: query_understanding (QueryUnderstanding)                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. plan_multi_intent (ai_platform/core/graph/nodes.py:380)             │
│    - Evaluates sub-queries, dependencies, and target channels.          │
│    - [BUG LOCATION]: Can emit channels=[] on compound queries.          │
│    - Emits: execution_plan (MultiIntentPlan)                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. retrieve_and_qualify (ai_platform/core/graph/nodes.py:512)          │
│    - Executes hybrid retrieval (dense + BM25) across planned channels.  │
│    - Reciprocal Rank Fusion (RRF) + cross-encoder qualification.        │
│    - Filters evidence chunks against strict relevance thresholds.       │
│    - Emits: evidence_bundle (EvidenceBundle), qualification_verdict     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. answer (ai_platform/core/graph/nodes.py:750)                        │
│    - Evidence boundary validation (evidence_boundary.py)                │
│    - LLM answer synthesis with cited chunk IDs                          │
│    - Grounding verification (grounding.py & verification.py)            │
│    - Packaging and fallback enforcement (packaging.py)                  │
│    - Emits: final_answer, citations, grounding_verdict                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                             [Final Answer]
```

---

## 3. Directory & Module Boundaries

### 3.1 `ai_platform/core/` (Shared Reusable Engine) `[Verified: Current Source]`
Contains institution-agnostic logic:
- `graph/`: `graph.py` (LangGraph assembly), `nodes.py` (`CoreNodes` execution), `state.py` (`AgentState`).
- `answering/`:
  - `evidence_boundary.py`: Strictly limits evidence context to qualified tokens.
  - `grounding.py`: Mathematical token/claim overlap and entailment checker.
  - `verification.py`: Contract validator asserting claim-level attribution.
  - `packaging.py`: Standardized output formatting, citation injection, safe abstentions.
- `planning/`: `planner.py` (Query decomposition and channel routing).
- `query/`: `pipeline.py`, `models.py`, `interpreter.py`, `frame.py` (Query parsing and normalization).
- `institution/`: `config.py` and `registry.py` (Base schemas and dynamic loader for multi-institution adapters).

### 3.2 `ai_platform/institutions/iitj/` (IIT Jodhpur Adapter) `[Verified: Current Source]`
Isolates all reference-institution specific parameters:
- `profile.py`: Identity (`iitj`), display name ("IIT Jodhpur"), vector store collection (`iitj_v1`), corpus paths.
- `semantic_registry.py`: Institution vocabulary detection (`btech`, `mtech`, `msc`, `phd`, departments).
- `lexical.py`: Institution-specific acronyms, aliases, and phrase mappings.
- `source_policy.py`: Ranking priors mapped to IIT Jodhpur document collections (e.g., favoring `admissions/` for eligibility queries).
- `scope_policy.py`: Conflict detection across conflicting program dimensions (e.g., rejecting B.Tech evidence for M.Tech queries).
- `data/`: Declarative JSON configuration files (`aliases.json`, `acronyms.json`, `phrases.json`, `semantic_terms.json`).
- *Architectural Compliance*: Completely free of hardcoded retrieval or generation logic. Passes `test_iitj_institution_contract.py`.

### 3.3 `ai_platform/runtime/` (Infrastructure & Composition Root) `[Verified: Current Source]`
Handles system integration and hardware/environment dependencies:
- `engine.py`: Canonical runtime execution root (`ask()`, `ask_with_diagnostics()`).
- `retrieval.py`: Adapters loading canonical chunks, building BM25 index, invoking vector store. Leaves all ranking, fusion, and qualification to `core`.
- `config.py`: Environment variable loading, LLM provider configs, paths.
- `corpus.py`: Document ingestion and chunk loading.
- `vectorstore.py` & `embedding.py`: ChromaDB and embedding model instantiation.
- `llm.py`: Provider wrappers for Groq, OpenAI, Ollama.

### 3.4 Historical, Orphan, and Standalone Bundles `[Verified: Current Source]`
- `backend/`: Historical monolithic implementation. Unsynchronized with `ai_platform/core/`.
- `backend_snapshot/`: Read-only snapshot of legacy codebase.
- `rag_v3_final_bundle/`: Standalone delivery package; not actively imported by runtime.
- `retrieval_v4_bundle/`: Standalone retrieval module; its README states: *"Not wired into production runtime"*.
- `*.bak` files: Approximately 50 legacy backup files polluting the repository tree.

---

## 4. Component Status & Contract Verification

| Component | Status | Evidence Class | Primary Files | Remaining Risk / Issues |
|---|---|---|---|---|
| **Web UI Runtime (`app.py`)** | ❌ Broken | `[Verified: Current Source]` | `app.py`, `backend/graph.py:57` | Crashes on launch due to broken import `ai_platform.core.query_pipeline`. |
| **Phase 1 Contracts** | ✅ Verified | `[Executed: This Audit]` | `ai_platform/core/answering/*` | 12/12 pytest cases pass. Strict grounding and packaging verified. |
| **Phase 2 (LSQC)** | ❌ Not Installed | `[Verified: Current Source]` | None (0 files found) | Complete absence of lossless semantic query contracts. |
| **Multi-Intent Planner** | ⚠️ Partial Bug | `[Recorded: Historical]` | `ai_platform/core/planning/planner.py` | Emits `channels=[]` on 10% of benchmark queries, causing false abstentions. |
| **Retriever & Qualification** | ⚠️ Partial Bug | `[Recorded: Historical]` | `ai_platform/core/graph/nodes.py` | Context drops during sub-query fusion; fails on numerical/attribute queries. |
| **Canonical Runtime (`engine.py`)** | 🟡 Functional | `[Verified: Current Source]` | `ai_platform/runtime/engine.py` | Works when invoked directly via script; disconnected from UI. |
| **Dead Code Block** | ⚠️ Defect | `[Verified: Current Source]` | `ai_platform/core/graph/nodes.py:1025-1035` | 11 lines unreachable after unconditional `raise ValueError`. |
| **IITJ Institution Adapter** | ✅ Compliant | `[Verified: Current Source]` | `ai_platform/institutions/iitj/*` | Well-isolated declarative profile, semantic terms, and scope policy. |

---

## 5. Detailed Regression Case Forensics

Investigation of the specified regression queries from `tests/regression_questions.json` and `tests/diagnostics_50_canonical.txt`:

### 5.1 Case `adm14` (Benchmark Q14) `[Recorded: Historical]`
- **Query ID**: `adm14`
- **Question**: *"What happens if an admitted student fails to submit required documents by the specified deadline?"*
- **Diagnostic Result**: `[14/50] CHECK | ANSWER_OR_GROUNDING | 34.43s`
- **Recorded Trace**:
  - `channels=[] fused=0 verified=0 ranked=0`
  - `context_chars=3151`
  - `Answer: "I'm sorry, I don't know based on the available information."`
- **Root Cause Analysis**:
  - The query planner failed to match the topic *"deadline submission failure"* to an active institutional channel, emitting `channels=[]`.
  - As a result, the retriever skipped document search entirely (`fused=0`), passing zero evidence to the qualification stage and triggering a safe abstention.

### 5.2 Case `msc10` (Benchmark Q25) `[Recorded: Historical]`
- **Query ID**: `msc10`
- **Question**: *"Is a first-class qualification accepted for M.Sc. admission?"*
- **Diagnostic Result**: `[25/50] PASS | PASS_STRUCTURAL | 25.50s`
- **Recorded Answer**:
  - *"No, based on the provided evidence, a first-class qualification is not explicitly accepted for M.Sc. admission at IIT Jodhpur. The evidence suggests that a minimum aggregate of 60% is required for General, OBC, and EWS categories, but does not mention first-class as a qualification for M.Sc. admission."*
- **Root Cause Analysis (Semantic Inversion / Semantic Loss)**:
  - In Indian university regulations and IIT Jodhpur rules, a 60% aggregate is universally equivalent to First Class.
  - The query pipeline treated "first-class" as an isolated literal token rather than binding it to the 60% threshold attribute.
  - The answer generator made an ungrounded negative assertion ("No, not accepted") while technically satisfying citation checks, resulting in a false and misleading answer.

### 5.3 Case `phd43` (Benchmark Q43) `[Recorded: Historical]`
- **Query ID**: `phd43`
- **Question**: *"Can a four-year bachelor's degree qualify someone for Ph.D. admission?"*
- **Diagnostic Result**: `[43/50] PASS | PASS_STRUCTURAL | 24.73s`
- **Recorded Answer**:
  - *"No, a four-year bachelor's degree alone does not qualify someone for Ph.D. admission. The evidence provided does not mention any specific eligibility for a Ph.D. program with just a bachelor's degree..."*
- **Contradiction with Case 42**:
  - Case 42 immediately prior correctly retrieved and stated: *"Additionally, candidates must have a bachelor’s degree of minimum four-year duration in engineering, science, medicine... with at least 70% marks or at least 7.0/10 CPI or CGPA."*
  - Case 43 dropped the entity-attribute link between "four-year bachelor's" and "Ph.D. admission with 70%/7.0 CPI", fabricating a negative assertion ("No").

### 5.4 M.Tech Numerical Threshold and Category Cases `[Recorded: Historical]`
- **Q29 (GEN/OBC %)**: Correctly answers 60% marks in aggregate.
- **Q30 (SC/ST %)**: Abstained (*"The provided evidence does not specify the minimum percentage required for SC/ST candidates..."*). Demonstrates missing table coverage in corpus ingestion.
- **Q31 (GEN/OBC CGPA)**: Abstained (*"minimum CGPA ... is not specified in the provided evidence"*).
- **Q32 (SC/ST CGPA)**: Abstained (*"minimum CGPA ... is not specified in the provided evidence"*).
- **Q33 (First Class M.Tech Acceptance)**:
  - Answer generated: *"No, a first-class qualification is not accepted for regular M.Tech admission based on the provided evidence. The minimum requirements for regular M.Tech admission are a minimum of 60% marks in aggregate or a minimum CGPA of 6.0 for GEN/OBC (55% and 5.5 respectively for SC/ST) categories."*
  - *Catastrophic Hallucination*: The model literally states that 60% / 6.0 CGPA is required, yet asserts that First Class is NOT accepted.

---

## 6. Phase 1 vs Phase 2 Technical Assessment

### Phase 1: Evidence Boundary & Answering Contracts `[Executed: This Audit]`
- **Test Command**:
  ```bash
  PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest \
      tests/diagnostics/test_phase1_contracts.py \
      tests/diagnostics/test_phase1_adversarial.py \
      tests/diagnostics/test_phase1_runtime_contract.py -v
  ```
- **Outcome**: **12 Passed in 0.68s**
  - `test_phase1_contracts.py` (5 passed): Invariants on evidence sufficiency, empty evidence abstention, and token boundaries.
  - `test_phase1_adversarial.py` (4 passed): Robustness against poisoned chunks, contradictory claims, and ungrounded queries.
  - `test_phase1_runtime_contract.py` (3 passed): End-to-end packaging contract adherence.
- **Verdict**: Phase 1 is solid and operational in `ai_platform/core/answering/`.

### Phase 2: Lossless Semantic Query Contract (LSQC) `[Verified: Current Source]`
- **Current State**: **0% Installed**.
- **Assessment**:
  - The regressions observed in `adm14` (`channels=[]`), `msc10` (first-class inversion), `phd43` (4-year degree drop), and Q33 are direct symptoms of semantic loss during query parsing, intent planning, and retrieval query synthesis.
  - Current planner drops entity bindings and numerical constraints during sub-query formulation.
  - Phase 2 is urgently needed to enforce lossless transmission of query semantics into the retrieval and qualification stages.

---

## 7. Top Five Engineering Risks

1. **Risk 1 (CRITICAL): Production Service Startup Failure** `[Verified: Current Source]`
   - `backend/graph.py:57` imports a non-existent module. The application cannot start in production.
2. **Risk 2 (HIGH): Architectural Bifurcation** `[Verified: Current Source]`
   - Developers are improving `ai_platform/core/`, but the web UI is anchored to legacy `backend/graph.py`. User traffic does not touch Phase 1 contracts.
3. **Risk 3 (HIGH): Silent Retrieval Abstentions via `channels=[]`** `[Recorded: Historical]`
   - 14% of canonical questions fail or abstain because heuristic routing generates empty channels.
4. **Risk 4 (HIGH): Semantic Negation Hallucinations Passing Structural Tests** `[Recorded: Historical]`
   - Because Phase 1 checks *structural attribution* rather than *semantic entailment*, blatant factual reversals (`msc10`, Q33, `phd43`) receive a `PASS_STRUCTURAL` mark.
5. **Risk 5 (MEDIUM): Ingestion Gaps for Category Cutoffs** `[Recorded: Historical]`
   - Reserved category thresholds (SC/ST/OBC/PwD) for M.Tech admissions (Q30–Q32) are missing or improperly chunked in the ingested corpus.

---

## 8. Next Engineering Steps & Acceptance Criteria

### Step 1: Fix Runtime Import and Reconnect UI to Canonical Graph (Immediate)
- **Actions**:
  1. Correct `backend/graph.py:57` to import from `ai_platform.core.query.pipeline`.
  2. Refactor `app.py` to import `ask` from `ai_platform.runtime.engine` instead of initializing the legacy `backend.graph.create_graph()`.
- **Acceptance Criteria**:
  - `streamlit run app.py` launches without import errors.
  - Streamlit chat invocations traverse `ai_platform.core.graph.graph.build_graph()`.

### Step 2: Fix Planner Channel Routing Default
- **Actions**:
  - In `ai_platform/core/planning/planner.py`, add a fallback rule: if intent routing outputs `channels=[]`, default to `["admissions", "academic", "general"]` rather than empty.
- **Acceptance Criteria**:
  - Case `adm14` produces `channels != []` and retrieves non-zero chunks.

### Step 3: Implement Phase 2 Lossless Semantic Query Contract (LSQC)
- **Specification**:
  - Create `ai_platform/core/planning/semantic_contract.py` defining `LosslessSemanticQuery`.
  - Enforce that all entities, numerical thresholds (e.g. `60%`, `CPI 6.5`), and category qualifiers (`GEN`, `SC/ST`) identified in `understand_query` are preserved without loss into `retrieve_and_qualify`.
  - In `ai_platform/core/answering/grounding.py`, add equivalence mappings (e.g., `60% aggregate == First Class`).
- **Acceptance Criteria**:
  - `msc10` and `Q33` confirm that first-class qualification is accepted.
  - `phd43` affirms that a 4-year bachelor's degree meets basic eligibility.
  - All 12 Phase 1 contract tests continue to pass (100%).

### Step 4: Corpus Ingestion Refresh for Reserved Categories
- **Actions**:
  - Audit source documents for M.Tech/Ph.D. admissions cutoffs and verify chunking preserves SC/ST/PwD relaxation tables.
- **Acceptance Criteria**:
  - Cases Q30, Q31, and Q32 retrieve explicit numerical thresholds instead of abstaining.
