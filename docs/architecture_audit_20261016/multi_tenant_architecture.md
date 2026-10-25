# Multi-Tenant Architecture & Institutional Isolation

**Audit Date**: October 16, 2026  
**Implementation**: `ai_platform/core/institution/`, `ai_platform/institutions/`  
**Reference Document**: [`INSTITUTION_CHANGE_MAP.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/INSTITUTION_CHANGE_MAP.md)  

---

## 1. Multi-Tenant Architectural Vision

The stated architectural objective of `ai_platform/` is to serve as a **reusable institutional AI engine** capable of being deployed across 10–15 universities and colleges with **~2 days of configuration/data preparation** instead of 20 days of code changes.

```
┌─────────────────────────────────────────────────────────────┐
│       Generic, Reusable Core (ai_platform/core/)            │
│  - Query Planner & Guard     - 3-Channel Retrieval & RRF    │
│  - Candidate Verification    - Evidence Packaging           │
│  - Generator & Grounding     - LangGraph State Machine      │
└──────────────────────────────┬──────────────────────────────┘
                               │ Pluggable Adapters
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Institution  │       │ Institution  │       │ Institution  │
│   "iitj"     │       │   "univ_b"   │       │   "univ_c"   │
│ (Active Ref) │       │  (Planned)   │       │  (Planned)   │
└──────────────┘       └──────────────┘       └──────────────┘
```

---

## 2. Separation of Concerns (Implemented)

The codebase exhibits clear structural separation between generic core logic and university-specific assets:

### 2.1 Generic Core (`ai_platform/core/`)
- Contains zero institution-specific strings (e.g. no hardcoded references to "Jodhpur", "Karwar", "CYL1010", or "B.Tech").
- Consumes abstract `InstitutionProfile`, `ScopePolicy`, and `LexicalRegistry` protocols.

### 2.2 Institutional Adapter (`ai_platform/institutions/iitj/`)
- Contains IIT Jodhpur-specific vocabulary, degree program taxonomy, campus locations, emergency contacts, and administrative policies:
  - `profile.py`: Declares institution name, ID, degree programs, and grading scale.
  - `lexical.py`: Custom token normalizers and acronym expansions.
  - `data/aliases.json`: Equivalence mappings (e.g. "first class" $\leftrightarrow$ "60% aggregate").
  - `data/scope_policy.json`: Supported degree levels, departments, and administrative domains.
  - `navigation.py` & `emergency.py`: Campus landmark coordinates and emergency phone directories.

### 2.3 Dynamic Institution Loader (`ai_platform/institutions/loader.py`)
- Provides `load_institution_profile(institution_id: str) -> InstitutionProfile`.
- Dynamically imports and instantiates the target institution package at runtime.

---

## 3. Honest Multi-Tenant Readiness Assessment

> [!WARNING]
> **Multi-Tenancy Is An Architecture Pattern, NOT An Implemented Multi-Tenant Service.**
> The system is currently a **single-institution reference deployment** configured for IIT Jodhpur. True multi-tenant production operation is UNTESTED and INCOMPLETE.

### 3.1 Implemented vs. Planned Capabilities

| Capability | Status | Evidence / Implementation Notes |
| :--- | :---: | :--- |
| **Separation of Core vs Configuration** | **VERIFIED** | Clean boundary between `ai_platform/core/` and `ai_platform/institutions/`. |
| **Dynamic Profile Loader Interface** | **VERIFIED** | `loader.py` dynamically resolves profile modules by ID. |
| **Pluggable Lexical Synonyms** | **VERIFIED** | `aliases.json` and `semantic_terms.json` are isolated per institution directory. |
| **Vector Database Isolation** | **INCOMPLETE** | ChromaDB is hardcoded to a single collection `iitj_v1` in `runtime/config.py`. No tenant-partitioned namespaces or collection-per-tenant routing exists. |
| **Corpus Ingestion Isolation** | **INCOMPLETE** | Corpus paths default to `data/data_iitj/`. Ingestion scripts do not accept a tenant parameter. |
| **Concurrent Multi-Tenant Serving** | **UNVERIFIED** | The runtime has never been benchmarked or tested with concurrent requests targeting different institutions. |
| **Cross-Tenant Contamination Defense**| **NOT PROVEN** | If multiple institutions were indexed into the current vector store, Dense and BM25 channels lack tenant metadata filters, creating high risk of cross-tenant data leakage. |

---

## 4. Minimum Engineering Requirements for Multi-Tenant Production

Before onboarding a second institution (e.g. `univ_b`), the following changes are mandatory:

1. **Vector Store Tenant Partitioning (`runtime/vectorstore.py`)**:
   - Update `ChromaDBStore` to dynamically route queries to collection `{institution_id}_v1` or enforce a mandatory metadata filter:
     ```python
     collection.query(where={"institution_id": active_institution_id})
     ```
2. **Corpus Registry (`runtime/config.py`)**:
   - Replace hardcoded `DATA_DIR = Path("data/data_iitj/...")` with a parameterized repository registry: `DATA_DIR / f"data_{institution_id}"`.
3. **Multi-Tenant Ingestion Pipeline**:
   - Parameterize ingestion scripts to ingest institution-specific documents into isolated tenant collections.
4. **Cross-Tenant Isolation Tests**:
   - Add automated regression tests proving that a query for Institution A cannot retrieve documents, chunks, or emergency numbers from Institution B.
