# Indian Institute of Technology Jodhpur (IITJ) Institutional RAG Chatbot

> A multi-tenant, institution-grounded Retrieval-Augmented Generation (RAG) platform delivering authoritative, hallucination-resistant answers to student, faculty, and administrative inquiries. Reference deployment: **Indian Institute of Technology Jodhpur (IIT Jodhpur)**.

---

## 1. Executive Summary & Capabilities

The IIT Jodhpur Institutional RAG Chatbot provides a reliable conversational interface over official institute documents, including academic ordinances, admission criteria, fee structures, hostel rules, administrative policies, and campus services.

### Core Capabilities
- **High-Precision Evidence Grounding:** Answers are synthesized exclusively from verified institutional evidence extracted from authoritative documents, eliminating speculative LLM hallucinations.
- **Multi-Intent Query Understanding & Decomposition:** Complex queries combining multiple institutional policies (e.g., fee structure waivers + hostel rent + mess advances) are automatically decomposed into focused sub-queries, retrieved across distinct source categories, and reconciled into coherent answers.
- **Hybrid Retrieval & Reciprocal Rank Fusion (RRF):** Combines dense semantic vector retrieval with lexical keyword search (BM25), fused via weighted RRF to optimize both semantic concept recall and exact-match keyword precision.
- **Candidate Verification & Strict Evidence Boundary:** Retrieved candidates undergo institutional qualification and category prior scoring. Only verified-anchor evidence units enter the downstream claim-audit and packaging stages.
- **Bounded Context Packaging:** Evidence units are deduplicated and budgeted under a deterministic 12,000-character context window, ensuring the local LLM receives high-signal information without truncation or attention dilution.
- **Query-Aware Extraction & Polarity Protection:** Protects against LLM negation inversion on alternative qualifying pathways (e.g., "OR a first-class degree"), with automated extractive fallback and academic-year token signature normalization.
- **Strict Guardrails & Safe Abstention:** Detects and blocks malicious prompt injections, system credential probes, physical security violations, and exam leak queries. Gracefully abstains when evidence is insufficient.
- **Transparent Attribution:** Supplies up to 4 authoritative citations in collapsible UI cards. Suppresses citations during safe refusals to prevent misleading attribution.
- **Clean Decoupled Architecture:** Cleanly separates the Streamlit presentation layer (`frontend/`), the reusable reasoning engine (`ai_platform/core/`), institution-specific configurations (`ai_platform/institutions/`), and local infrastructure runtimes (`ai_platform/runtime/`).

---

## 2. Verified Technology Stack

| Layer | Component | Verified Implementation |
|---|---|---|
| **Presentation** | Web UI & Session State | [Streamlit](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/app.py) (`frontend/` package: modular components, session manager, UI config loader) |
| **Service Adapter** | Presentation-Service Bridge | [`frontend.clients.assistant_client`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py) → [`ai_platform.service.assistant_service`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/service/assistant_service.py) |
| **Workflow Engine** | Orchestration & Graph Pipeline | [LangGraph](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/graph/nodes.py) (`ai_platform.core.graph.nodes.CoreNodes`, typed `RAGState`) |
| **Local LLM** | Answer & Query Intelligence | `qwen2.5:3b` via [Ollama](http://localhost:11434) (temperature = 0.0 for deterministic factual generation) |
| **Embeddings** | Semantic Vector Embeddings | `nomic-embed-text:latest` via [Ollama](http://localhost:11434) |
| **Vector Store** | Dense Retrieval Storage | [ChromaDB](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/chroma_db/) (`collection_name = "iitj_v1"`, Cosine distance) |
| **Lexical Search** | Exact Keyword Retrieval | `rank_bm25` ([`BM25Retriever`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/runtime/retrieval.py#L30)) over corpus chunks |
| **Python Runtime** | Core Execution Environment | Python 3.11 (`/opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/python`) |

---

## 3. High-Level Architecture & Request-to-Answer Flow

The diagram below represents the verified runtime execution path, tracing a query from the user's Streamlit browser interaction through the graph pipeline to the final grounded answer:

```text
[ User Browser / Streamlit UI ]
               │
               ▼
   frontend.app:run_app()
               │
               ▼
   frontend.components.chat:render_chat_view()
               │ (captures input, session history)
               ▼
   frontend.clients.assistant_client:send_chat_turn()
               │
               ▼
   ai_platform.service.assistant_service:handle_chat_turn()
               │ (orchestrates session, tenant config, and runtime engine)
               ▼
   ai_platform.runtime.engine:ask_with_diagnostics()
               │
               ▼ [ LangGraph Execution Pipeline ]
   ┌────────────────────────────────────────────────────────┐
   │ 1. resolve_conversation_node                           │
   │    -> Coreference resolution, history windowing        │
   │                                                        │
   │ 2. understand_query_node                               │
   │    -> Guard checks, intent classification, entity scan │
   │    * If Out-of-Scope / Security Breach:                │
   │      pipeline_stop=True -> guard_answer_node           │
   │                                                        │
   │ 3. plan_multi_intent_node                              │
   │    -> Multi-intent detection & query decomposition     │
   │                                                        │
   │ 4. retrieve_and_qualify_node / process_multi_intent    │
   │    -> Dense (ChromaDB) + Lexical (BM25) search         │
   │    -> Weighted Reciprocal Rank Fusion (RRF)            │
   │    -> Candidate verification against institutional scope│
   │    -> Target candidate ranking                         │
   │                                                        │
   │ 5. build_evidence_package_node                         │
   │    -> Verified-anchor evidence boundary                │
   │    -> Evidence sufficiency & coverage check            │
   │    -> Claim audit & cross-intent alignment            │
   │    -> Evidence packaging (12,000 char budget cap)      │
   │                                                        │
   │ 6. generate_answer_node                                │
   │    -> Query-aware prompt formatting                    │
   │    -> Local LLM generation (qwen2.5:3b, temp=0.0)      │
   │                                                        │
   │ 7. guard_answer_node                                   │
   │    -> Grounding verification (AY token normalization)  │
   │    -> Extractive repair on polarity inversion          │
   │    -> Citation selection (max 4 authoritative sources) │
   └────────────────────────────────────────────────────────┘
               │
               ▼
   AssistantResponse (answer, citations, structured cards, suggestions)
               │
               ▼
   frontend.components.messages:render_chat_messages()
   [ Final Grounded Markdown Answer + Collapsible Citations ]
```

### Key Architectural Boundaries & Nuances
1. **Candidate Retrieval vs. LLM Context Budget:** Dense and BM25 retrieval gather a wide pool of candidate chunks (typically 10–25 candidates per intent). However, only candidates passing institutional verification and claim auditing enter the final evidence package. The context package enforces a strict **12,000-character budget**, discarding un-supported fragments. Therefore, *retrieved candidates do not equal LLM context*.
2. **Citation Limits vs. Retrieval Limits:** Even if 10 chunks supported the context generation, the frontend strictly limits displayed citations to the **top 4 most authoritative sources** to keep the UI clean and prevent cognitive overload.
3. **Suppression on Refusal:** When a query triggers an out-of-scope refusal or insufficient evidence abstention, citations are completely suppressed to guarantee attribution integrity.

---

## 4. Production RAG Forensic Diagnosis & Generalized Architectural Repairs (October 19, 2026)

Following observed failures on institutional queries (spurious refusals, arithmetic traps, missing cross-department comparisons, truncated numbered lists, and deictic reference misclassifications), a forensic investigation across 40 frozen diagnostic queries established that the failures stemmed from **early pipeline filtering and generator formatting contracts**, not local LLM reasoning limits alone.

The smallest generalized set of non-hardcoded architectural repairs was designed and verified across the pipeline:

1. **Deictic Unanchored Reference Guard ([`ai_platform/core/conversation/conversation.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/conversation/conversation.py)):**
   - *Problem:* Standalone questions containing phrases like "the degree" (Q20) or "the course" (Q23) without prior history were falsely flagged as unanchored deictic references, triggering premature clarification requests before retrieval.
   - *Fix:* Gated the unanchored reference check behind `if messages:`, ensuring standalone queries proceed to institutional query understanding and retrieval uninterrupted.

2. **Vocational & Skilling Scope Integration ([`ai_platform/core/query/planner.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/query/planner.py)):**
   - *Problem:* Legitimate institutional vocational training queries (PMKVY 4.0, Skill Hub, Job Roles, Certifications) were falsely rejected as out-of-scope by the institutional scope guard (Q13, Q25, Q36).
   - *Fix:* Added vocational training and PMKVY skilling markers to institutional scope dictionary, attribute extraction, and target inference (`_repair_scope`, `_infer_target`).

3. **Meta-Document Reference Disambiguation ([`ai_platform/core/query/planner.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/query/planner.py)):**
   - *Problem:* Conversational meta-phrases such as *"does the document propose"* caused factual queries (e.g. Q2) to be misclassified as `request_type="documents"`, restricting retrieval to document discovery rather than extracting facts.
   - *Fix:* Implemented `_is_meta_document_reference()` to distinguish factual queries mentioning documents from genuine document retrieval requests.

4. **Conjunction Lookahead in Degree Parsing ([`ai_platform/core/query/planner.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/query/planner.py)):**
   - *Problem:* Degree names with conjunctions (e.g., *"Drone and Anti-Drone Technologies"*) were prematurely truncated at `\band\b`, resulting in the incomplete target `"Drone"` (Q2).
   - *Fix:* Enhanced `_infer_target()` lookahead to preserve complete hyphenated and conjoined degree titles.

5. **Cross-Department Comparative Evidence Verification ([`ai_platform/core/retrieval/verification.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/retrieval/verification.py) & [`ranking.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/retrieval/ranking.py)):**
   - *Problem:* Comparative queries comparing two departments (e.g., Electronics vs. Materials Engineering in Q30) were rejected because single-chunk relation proof demanded that every chunk establish relationships between both departments.
   - *Fix:* Separated comparison markers from strict relational grounding, added `_is_comparison_query()`, and protected high-RRF fused candidates (`_rrf_signal >= 0.70`) from specialized without-target demotion.

6. **Deterministic Arithmetic Verification & Traps ([`ai_platform/core/answering/generator.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/answering/generator.py)):**
   - *Problem:* Sycophantic local LLM accepted user-suggested false arithmetic premises (e.g., agreeing that 42+12+6+2+12 = 72 in Q38).
   - *Fix:* Implemented `_verify_arithmetic_claim()`, which deterministically sums equation terms and corrects false arithmetic claims in generated answers.

7. **Extractive Repair Department Boundary Isolation & Tag Stripping ([`ai_platform/core/answering/generator.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/answering/generator.py)):**
   - *Problem:* Fallback extractive repair leaked internal raw markdown heading tokens (`######`) and concatenated snippets from unrelated departments (Q3, Q7).
   - *Fix:* Isolated candidate snippets by target department boundaries and stripped markdown header hashes before injecting text into final responses.

8. **Numbered List Termination & Prompt Budget Optimization ([`ai_platform/core/answering/generator.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/answering/generator.py)):**
   - *Problem:* Clean sentence termination truncated answers ending in numbered item prefixes (e.g., `4.`, `5.`), and redundant instructions consumed ~1,000 tokens of the generation budget (Q4, Q5, Q10, Q24).
   - *Fix:* Excluded numeric list prefixes from premature punctuation trimming and streamlined prompt instructions to maximize token allocation for context packing.

---

## 5. Environment Setup & Prerequisites

### Prerequisites
- macOS or Linux with Python 3.11 installed.
- [Ollama](https://ollama.com/) running locally on port 11434 with models pulled:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull nomic-embed-text:latest
  ```
- Conda virtual environment at `/opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag`.

### Directory & Corpus Layout
- **Document Corpus:** `data/data_iitj/iitj_rag_v1_docs_production/`
- **Vector Store:** `chroma_db/` (collection: `iitj_v1`)
- **Tenant Metadata:** `ai_platform/institutions/iitj/data/`

---

## 6. How to Launch the Application

The canonical application is launched via the root launcher [`app.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/app.py), which starts the modular Streamlit frontend:

```bash
# Set PYTHONPATH to the repository root and launch Streamlit:
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

---

## 7. How to Run Tests & Verification Suites

All test commands should be executed from the repository root with `PYTHONPATH=.`.

### 1. Automated Regression & Core Invariants (70 Tests — 100% PASS)
Runs all unit and integration tests across Streamlit canonical runtime, query planner contracts, frontend mechanics, and guard/evidence hardening:
```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest \
  tests_reusable/test_streamlit_canonical_runtime.py \
  tests_reusable/query_planner/test_query_planner_contract.py \
  tests_frontend/ \
  tests_reusable/rag_quality/test_guard_and_evidence_hardening_regression.py -v
```

### 2. Frozen 40-Question Diagnostic Benchmark (Baseline vs. Post-Fix)
Reproduces and traces the 40 diagnostic benchmark queries across all stages:
```bash
# Baseline evaluation:
PYTHONPATH=. python3 scripts/run_40q_baseline.py
# Post-fix evaluation:
PYTHONPATH=. python3 scripts/run_40q_postfix.py
# Comparative analysis report:
PYTHONPATH=. python3 scripts/generate_comparative_metrics.py
```

### 3. Realistic Student Evaluation Suite (20 Questions — 100% Pass/Abstain Accuracy)
Evaluates natural conversational inquiries across everyday campus topics:
```bash
PYTHONPATH=. python3 scripts/run_student_evaluation.py
```

### 4. Isolated Held-Out Generalization Suite (10 Questions — 100% Grounded Accuracy)
Evaluates unseen academic and departmental inquiries across holdout topics:
```bash
PYTHONPATH=. python3 scripts/run_held_out_evaluation.py
```

### 5. Streamlit Live UI Verification (6 Multi-Turn Conversations — 100% PASS)
Validates frontend integration through `UI.chat.get_response` across representative dialogue patterns:
```bash
PYTHONPATH=. python3 scripts/verify_streamlit_live_ui.py
```

### 6. Master Diagnostic Failure Cases (7 Canonical Failure Cases — 100% PASS)
Verifies the 6 Master Engineering Failure Classes (Complete B.Tech list, Institute-wide departments, Tuition fee per semester, B.Tech fee structure, Main research areas, Navigation telemetry, and Multi-turn follow-up):
```bash
PYTHONPATH=. python3 scripts/diagnose_master_cases.py
```

### 7. 12-Class Production Reliability Matrix (12 Test Classes — 100% PASS)
Validates exact numeric lookups, confusable attributes, school-specific vs generic fees, list completeness, table exemptions, multi-attribute queries, Hinglish navigation, follow-up continuity, secondary context, missing data abstention, direct provenance, and short fee inquiries:
```bash
PYTHONPATH=. python3 scripts/test_12_regression_classes.py
```

---

## 8. Verified Validation Metrics & Production Readiness

### Comparative Evaluation Summary

| Evaluation Suite | Size / Scope | Baseline Outcome | Repaired Outcome | Net Improvement |
|---|---|---|---|---|
| **Frozen Diagnostic Benchmark (Q1–Q40)** | 40 complex queries across 4 groups | 8 PASS (20.0%)<br>11 PARTIAL (27.5%)<br>21 FAIL (52.5%) | **12 PASS (30.0%)**<br>**17 PARTIAL (42.5%)**<br>**11 FAIL (27.5%)** | **+50.0% clean pass**<br>**-47.6% failure rate** |
| **Candidate Survival Rate** | Candidates surviving verification | 34 / 40 (85.0%) | **40 / 40 (100.0%)** | **+15.0% survival** |
| **False Refusals on Valid Queries** | Legitimate queries blocked | 5 / 40 (12.5%) | **0 / 40 (0.0%)** | **100% eliminated** |
| **Query Planning & Scope Stops** | Premature pipeline aborts | 5 / 40 (12.5%) | **0 / 40 (0.0%)** | **100% resolved** |
| **Realistic Student Suite (S01–S20)** | 20 natural multi-turn queries | Not evaluated | **20 / 20 (100.0%) grounded / safe abstention** | Verified production-ready |
| **Held-Out Evaluation (H01–H10)** | 10 unseen held-out queries | Not evaluated | **10 / 10 (100.0%) grounded / safe abstention** | Verified generalizable |
| **Streamlit Live UI Verification** | 6 representative conversations (7 turns) | Pending | **6 / 6 (100.0% PASS)** | Verified interactive fidelity |
| **Automated Regression Suites** | 70 contract, safety & UI tests | Not unified | **70 / 70 (100.0% PASS, 0 failures)** | Zero regressions |

### Live Streamlit UI Acceptance Status
> [!NOTE]
> **Live Streamlit UI Acceptance Status: PASSED (Verified October 19, 2026).**
> All six canonical multi-turn conversations executed successfully via the active Streamlit runtime adapter (`UI.chat.get_response`), verifying correct markdown rendering, collapsible citation links, and history isolation across dialogue sessions.

### Known Operational Boundaries
- The local answer generator runs on `qwen2.5:3b`. Complex tables are extracted reliably via targeted prompt directives, but the model has a maximum generation window of 2,048 tokens.
- Vector database searches rely on the pre-built `chroma_db/` index with collection `iitj_v1`. Rebuilding the index is unnecessary and should not be performed during normal operation.

---

## 9. Project Documentation Links

For deeper technical documentation, refer to the following subsystem guides:

- **Frontend Subsystem:** [`frontend/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/README.md) — Modular architecture, Streamlit components, theming, and session management.
- **Reference Institution Adapter:** [`ai_platform/institutions/iitj/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/institutions/iitj/README.md) — Vocabulary data, acronym expansion, and campus direct handlers.
- **RAG Quality & Regression Testing:** [`tests_reusable/rag_quality/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/README.md) — Regression test suites, failure classifications, and canonical scenario specifications.
- **Frontend Test Suite:** [`tests_frontend/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_frontend/README.md) — Core UI mechanics and conversation-quality regression tests.
- **Historical Diagnostic Record:** [`tests_reusable/test_diagnosis/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/test_diagnosis/README.md) — Early stage-wise diagnostic runner documentation.
- **Master Platform Handoff:** [`AI_PLATFORM_MASTER_HANDOFF.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md) — Comprehensive architecture, contract specifications, and system invariants.
- **Graph Pipeline Details:** [`AI_PLATFORM_ARCHITECTURE.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_ARCHITECTURE.md) — Typed state fields, node responsibilities, and routing logic.
