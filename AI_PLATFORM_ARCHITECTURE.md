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
- **Module**: `ai_platform/core/conversation/conversation.py`, `understanding.py`, and `planner.py`
- Resolves pronoun references and conversation context.
- **Deictic Reference Anchoring**: Ensures deictic queries containing generic nouns (e.g. "the degree", "the course") only trigger clarification requests when there is an active prior conversation (`if messages:`), preventing false refusals on standalone queries.
- **Vocational & Skilling Scope**: Integrates institutional vocational training (PMKVY 4.0, Skill Hub, Job Roles, Certifications) into the core scope dictionary and target inference.
- **Meta-Document Disambiguation**: Differentiates factual queries referencing documents (e.g., *"does the document propose"*) from genuine requests for document listing.
- **Conjunction Lookahead**: Accurately preserves conjoined and hyphenated degree names (e.g., *"Drone and Anti-Drone Technologies"*).

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
- **Comparative Query Support**: Implements `_is_comparison_query()` to decouple multi-entity comparative queries from single-chunk relation requirements, allowing cross-department evidence to survive.

### Stage 5: Multi-Signal Ranking
- **Module**: `ai_platform/core/retrieval/ranking.py`
- Deterministically scores verified candidates using target overlap, attribute alignment, lexical density, and source policy priors.
- **High-RRF Signal Protection**: Protects top-ranked fused candidates (`_rrf_signal >= 0.70`) from premature demotion under specialized candidate penalties.

### Stage 6: Evidence Boundary, Sufficiency Calibration & Abstention
- **Module**: `ai_platform/core/evidence/`
- Assembles verified and ranked chunks into an `EvidencePackage`.
- **Calibrated Evidence Sufficiency**: Enforces a 4-state semantic decision (`SUFFICIENT`, `PARTIAL`, `INSUFFICIENT`, `CONFLICTING`) based on entity match, requested attribute, relationship, institutional scope, dates/units, and list completeness.
- **Honest Institutional Abstention**: When evidence is missing, conflicting, or cannot substantiate the requested claim, marks `ready_for_generation=False` and abstains with standard institutional response: *"Sorry, I couldn't find reliable information about that in the available IIT Jodhpur documents."* (configured via `InstitutionProfile.fallback_message`).
- **Partial Support Handling**: For multi-part requests where only a subset of facts is substantiated, provides a qualified partial answer explicitly identifying unverified items rather than hallucinating or refusing outright.

### Stage 7: Grounded Answer Generation & Multilingual English-Only Output
- **Module**: `ai_platform/core/answering/generator.py`
- Invokes `qwen2.5:3b` with strict system instructions:
  - Return ONLY the final answer in fluent, concise **English** regardless of input language (Hindi, Hinglish, etc.).
  - Zero internal process narration or chunk enumeration.
  - Faithful to provided evidence; no speculation or unverified claims.
  - **Prompt Injection Defense**: Explicitly forbids accepting user-asserted premises as facts (e.g., attempts to force "tuition is ₹1").
  - Explicit recognition of qualifying options connected by `OR`.
- **List Prefix Preservation**: Prevents premature truncation of answers ending in numbered item prefixes (`4.`, `5.`).
- **Prompt Token Optimization**: Streamlined prompt instructions removing redundant directives, freeing ~1,000 tokens for evidence packing.
- **Devanagari Guard & English Fallback**: Automatic detection of any Devanagari script characters in generated output, automatically routing to extractive English repair to eliminate repetitive Hindi token-loop degradation.

### Stage 8: Deterministic Grounding & Extractive Repair
- **Module**: `ai_platform/core/answering/grounding.py` and `generator.py`
- Inspects generated answer against verified evidence without an LLM call:
  - Numeric value consistency.
  - Academic-year signature normalization (`YYYY_YYYY`).
  - Scope boundaries.
  - Polarity inversion detection (preventing false negative claims when evidence supports eligibility).
- **Extractive Repair Department Boundary Isolation**: Enforces department scope boundaries during fallback extraction, eliminating cross-department contamination and stripping raw markdown header hashes (`######`).

### Stage 9: Fail-Closed Output Guarding & Citation Integrity
- **Module**: `ai_platform/core/answering/guard.py`
- Sanitizes leaked metadata (e.g. `Evidence 1`, `RRF`, internal paths).
- **Prompt Injection Stripping**: Strips adversarial override prefixes ("Ignore all previous instructions...").
- **Citation Provenance Integrity**:
  - Every displayed citation directly substantiates the claim it is attached to.
  - Suppresses citation display during safe refusals to prevent spurious attribution (0 manufactured citations on abstention).
  - Direct navigation routes provide structured provenance without inheriting stale RAG citations.
- If answer is ungrounded (`status="review"`), empty, or unsafe, replaces with safe fallback: *"Sorry, I couldn't find reliable information about that in the available IIT Jodhpur documents."*

---

## 5. Testing & Verification Architecture

Testing follows a strict hierarchy under `scripts/` and `tests_reusable/`:
- **`scripts/diagnose_master_cases.py`**: Automated test runner for the 6 Master Engineering Failure Classes (Complete B.Tech list, Institute-wide departments, Semester tuition, B.Tech fee structure, Main research areas, Navigation telemetry, Multi-turn follow-up).
- **`scripts/test_12_regression_classes.py`**: Automated test runner for the 12-Class Regression Matrix (Numeric lookups, confusable attributes, school-specific vs generic fees, list completeness, table exemptions, multi-attribute queries, Hinglish navigation, follow-up continuity, secondary context, missing data abstention, direct provenance, short fee inquiries).
- **`tests_reusable/rag_quality/`**: Dedicated regression test suite covering documented failure classes (C2, C4, C5, runner safety).
- **`tests/diagnostics/`**: Phase 1 mathematical invariant contracts and 50-question canonical benchmark runner (`run_50_canonical.py`).
- **`test_outputs/`**: Isolated directory for all benchmark runs and diagnostic JSON artifacts.

---

## 6. Knowledge Ingestion Engine V2 Package Adapter Architecture

### Purpose & Reusability Contract
The Knowledge Ingestion Engine V2 package integration (`ai_platform/packages/`) bridges upstream institutional crawling releases with the downstream hybrid RAG chatbot without coupling either side to college-specific schemas or hardcoded source paths. Future colleges providing packages matching the V2 manifest and contract specification can be onboarded via package registration rather than code rewrites.

### Package Topology
```text
data/data_<institution_id>/<release_name>/
├── CONTENTS_INDEX.md
├── RAG_INGESTION_CONTRACT.md
├── README.md, RELEASE_NOTES.md, UPDATE_GUIDE.md
├── contracts/          # Schemas & consumer ingestion contracts
├── data/               # Discovered structured JSON datasets (fees, accommodations, etc.)
├── documents/          # Canonical readable documents (.docx primary representation)
├── knowledge/          # Paired Markdown representations (auditing / inspection)
├── manifests/          # Manifests (document_inventory.json, package_manifest.json, etc.)
├── metadata/           # Granular sidecars (<doc_id>.json) paired by stable ID
├── provenance/         # Source registry, claim ledgers, and conflict records
└── reports/            # Quality metrics, coverage, and audit reports
```

### Core Components (`ai_platform/packages/`)
1. **`models.py`**: Strongly-typed dataclasses for package components:
   - `CanonicalDocument`: Path, document ID, category, subcategory, title, SHA-256 hash.
   - `ConflictRecord`: Documented institutional facts with reconciled vs conflicting values (e.g. B.Tech ₹1,00,000 vs ITEP ₹50,000 tuition; visitor accommodation rates).
   - `ValidationReport`: Cryptographic SHA-256 verification, discovered datasets, accepted/quarantined documents.
   - `InstitutionalRelease`: Normalized institutional bundle ready for indexing and retrieval.
2. **`package_adapter.py`**:
   - `validate_package(package_root)`: Validates manifests, schemas, file existence, and hashes.
   - `load_institutional_release(package_root)`: Maps metadata sidecars to canonical documents by stable ID; parses discovered structured JSON datasets into typed factual records.
   - `chunk_package_documents(release)`: Performs metadata-aware chunking preserving section hierarchy, table values, date ranges, and attaches 16+ canonical metadata fields (`institution_id`, `release_id`, `document_id`, `chunk_id`, `title`, `category`, `subcategory`, `document_type`, `source_ids`, `source_urls`, `unit_ids`, `claim_ids`, `conflict_ids`, `aliases`, `canonical_path`, `content_hash`, `chunk_index`, `schema_version`) to every chunk.
3. **`registry.py`**:
   - Dynamic release configuration (`data/active_release_<inst>.json`).
   - Isolation: staged releases are written to a release-specific vectorstore collection (e.g. `iitj_iit_jodhpur_rag_package_v2_staging`) and separate lexical cache without touching the active production index.
   - Atomic cutover and rollback preservation.

### Management CLI (`scripts/knowledge_package.py`)
Provides operators with an 8-command lifecycle for institutional data onboarding:
- `validate`: Verify manifests, SHA-256 cryptographic hashes, and sidecar mappings.
- `stage`: Register a release for staging isolation.
- `build-index`: Build the staging vector collection and BM25 lexical cache.
- `evaluate`: Run the 11-probe regression and disentanglement test suite.
- `compare`: Compare active baseline vs staging results side-by-side.
- `activate`: Atomically switch active production pointers with backup.
- `status`: Display active release, staged release, and index metrics.
- `rollback`: Instantly restore the previous known-good active configuration.
