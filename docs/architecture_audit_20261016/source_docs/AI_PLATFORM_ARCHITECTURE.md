# AI Platform Architecture

> Comprehensive architectural specification for the reusable institutional AI/RAG platform.
> Canonical engineering status and handover details: [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

---

## 1. System Philosophy & Reusability Goal

The College AI Platform is designed to support 10–15 educational institutions with **~2 days of configuration and data onboarding** rather than 20 days of custom code rewrites per campus.

To achieve this, the system enforces a strict boundary between:
1. **Generic RAG Intelligence** (`ai_platform/core/`): Query normalization, intent planning, hybrid retrieval, reciprocal rank fusion (RRF), deterministic candidate verification, multi-signal ranking, evidence boundary enforcement, answer generation, grounding checks, and output guarding.
2. **Institutional Knowledge & Adapters** (`ai_platform/institutions/<id>/`): Curated vocabulary, acronyms, program names, degree codes, local policies, and campus aliases.
3. **Runtime Infrastructure** (`ai_platform/runtime/`): Pluggable LLMs (Ollama), embeddings, vector database (ChromaDB), and engine composition roots.
4. **Historical Quarantine** (`backend/`): The pre-migration IIT Jodhpur chatbot codebase. It is retained strictly as read-only historical reference. It must never be imported, extended, repaired, or used as a runtime dependency.

---

## 2. High-Level Architectural Topology

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COLLEGE AI PLATFORM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ai_platform/core/                                                         │
│   ├── graph/       LangGraph orchestration (6 composite nodes)              │
│   ├── query/       Understanding, classification, planning                  │
│   ├── retrieval/   Dense + BM25, RRF fusion, candidate verification, ranking│
│   ├── evidence/    Sufficiency, coverage, claim audit, context packaging    │
│   └── answering/   Grounded generation, inversion detection, guard          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ai_platform/institutions/                                                 │
│   └── iitj/        Reference institution: vocabulary, aliases, policies     │
│   └── <future>/    Onboarding directory per new university                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ai_platform/runtime/                                                      │
│   └── engine.py    Composition root: ask() & ask_with_diagnostics()         │
│   └── llm.py       Ollama integration (qwen2.5:3b)                          │
│   └── embeddings.py nomic-embed-text:latest                                 │
│   └── vectorstore.py ChromaDB client                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [QUARANTINED] backend/ (Legacy codebase — DO NOT IMPORT OR MODIFY)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Canonical Graph Execution Flow

The canonical engine (`ai_platform/core/graph/graph.py`) compiles a deterministic 6-node LangGraph:

```text
                             User Question
                                   │
                                   ▼
                       [resolve_conversation_node]
                                   │
                                   ▼
                    [understand_and_plan_query_node]
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
 [retrieve_and_qualify_node]                   [process_multi_intent_node]
     (Single-Intent Path)                         (Multi-Intent Path)
            │                                             │
            └──────────────────────┬──────────────────────┘
                                   │
                                   ▼
                     [build_evidence_package_node]
                                   │
                                   ▼
                         [generate_answer_node]
                                   │
                                   ▼
                          [guard_answer_node]
                                   │
                                   ▼
                             Final Answer
```

### Typed State Contract (`RAGState`)

The graph operates on an immutable, typed state (`ai_platform/core/graph/state.py`):

| State Field | Type | Description |
|---|---|---|
| `question` | `str` | Raw input query from user |
| `resolved_question` | `str` | Conversational resolution incorporating history |
| `query_frame` | `QueryFrame` | Extracted intents, targets, attributes, constraints |
| `query_plan` | `QueryPlan` | Execution strategy (single vs. multi-intent, search queries) |
| `retrieval_results` | `tuple` | Multi-channel raw candidate lists (Dense, BM25, Lexical) |
| `fused_candidates` | `tuple` | Reciprocal Rank Fusion output |
| `verified_candidates`| `tuple` | Safety-gated candidates passing deterministic criteria |
| `ranked_candidates` | `tuple` | Verified candidates scored by multi-signal ranker |
| `evidence_package` | `EvidencePackage` | Final compressed context, claim audits, and status |
| `answer` | `str` | Generated user-facing answer |
| `answer_grounding` | `GroundingResult` | Polarity, numeric, currency, and causality checks |
| `answer_guard` | `GuardResult` | Sanitized output or fail-closed fallback |

---

## 4. Detailed Core Pipeline Stages

### Stage 1: Query Understanding & Intent Planning
- **Module**: `ai_platform/core/query/understanding.py` and `planner.py`
- Resolves pronoun references and conversation context.
- Extracts target entity, requested attribute, category, conditions, and constraints.
- Builds search queries across dense and lexical channels.
- Distinguishes genuine multi-intent questions from single questions with multiple clauses.

### Stage 2: Multi-Channel Hybrid Retrieval
- **Module**: `ai_platform/core/retrieval/`
- **Dense Channel**: Cosine similarity against `nomic-embed-text` vectors in ChromaDB.
- **BM25 Channel**: Lexical keyword matching over full tokenized corpus.
- **Lexical Rescue Channel**: Acronym and exact-phrase rescue for campus-specific codes.

### Stage 3: Reciprocal Rank Fusion (RRF)
- **Module**: `ai_platform/core/retrieval/rrf.py`
- Combines rank lists deterministically: $RRF\_Score(d) = \sum_{c \in Channels} \frac{w_c}{k + rank_c(d)}$ ($k=60$).

### Stage 4: Pre-LLM Candidate Verification Boundary
- **Module**: `ai_platform/core/retrieval/verification.py`
- Hard safety/relevance gate before ranking.
- Rejects explicit scope mismatches, contradictory degree levels, and off-target chunks.
- Permits multi-chunk complementary evidence for complex criteria.

### Stage 5: Multi-Signal Ranking
- **Module**: `ai_platform/core/retrieval/ranking.py`
- Deterministically scores verified candidates using target overlap, attribute alignment, lexical density, and source policy priors.

### Stage 6: Evidence Boundary & Packaging
- **Module**: `ai_platform/core/evidence/`
- Assembles verified and ranked chunks into an `EvidencePackage`.
- Enforces character/token budgets, eliminates redundancy, and assesses sufficiency.
- If evidence is genuinely absent or insufficient, marks `ready_for_generation=False`.

### Stage 7: Grounded Answer Generation
- **Module**: `ai_platform/core/answering/generator.py`
- Invokes `qwen2.5:3b` with strict system instructions:
  - Return ONLY the final answer.
  - Zero internal process narration or chunk enumeration.
  - Faithful to provided evidence; no speculation.
  - Explicit recognition of qualifying options connected by `OR`.

### Stage 8: Deterministic Grounding Verification
- **Module**: `ai_platform/core/answering/grounding.py`
- Inspects generated answer against verified evidence without an LLM call:
  - Numeric value consistency.
  - Currency and fee amounts.
  - Scope boundaries.
  - Polarity inversion detection (preventing false negative claims when evidence supports eligibility).

### Stage 9: Fail-Closed Output Guarding
- **Module**: `ai_platform/core/answering/guard.py`
- Sanitizes leaked metadata (e.g. `Evidence 1`, `RRF`, internal paths).
- If answer is ungrounded (`status="review"`), empty, or unsafe, replaces with safe fallback: *"I'm sorry, I don't have enough authoritative information to answer this question."*

---

## 5. Testing & Verification Architecture

Testing follows a strict hierarchy under `tests_reusable/`:
- **`tests_reusable/rag_quality/`**: Dedicated regression test suite covering documented failure classes (C2, C4, C5, runner safety).
- **`tests/diagnostics/`**: Phase 1 mathematical invariant contracts and 50-question canonical benchmark runner (`run_50_canonical.py`).
- **`test_outputs/rag_quality/`**: Isolated directory for all benchmark runs.
