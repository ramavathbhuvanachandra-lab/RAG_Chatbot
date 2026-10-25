# IIT Jodhpur Institution Adapter (`ai_platform/institutions/iitj/`)

This package provides the reference implementation of the institution boundary for the Indian Institute of Technology Jodhpur (IIT Jodhpur) within the reusable AI platform.

---

## 1. Architectural Responsibilities & Boundaries

The adapter encapsulates all institution-specific knowledge, nomenclature, and configuration:
- **Deployment & Vector Store Configuration:** Corpus storage locations, ChromaDB collection identity (`iitj_v1`), and embedding configuration.
- **Institutional Vocabulary:** Program identifiers (e.g., B.Tech, M.Tech, M.Sc., Ph.D., ITEP), departmental terminology, and campus facility names.
- **Acronym & Phrase Expansion:** Abbreviations (`acronyms.json`), entity aliases (`aliases.json`), and recognized multi-word phrases (`phrases.json`).
- **Scope & Guard Policies:** In-scope topic boundaries, out-of-scope intent definitions, and institutional fallback messaging.
- **Ranking Priors:** Source-category metadata weighting priors (`source_policy.json`).
- **Structured Direct Handlers:** Deterministic location wayfinding (`campus_navigation.py`) and emergency response services (`emergency.py`).

### Strict Architectural Boundaries
- **No Retrieval or Generation Overrides:** This module does **not** own dense/BM25 retrieval, RRF candidate fusion, verification, ranking algorithms, evidence boundary packaging, LLM prompt templates, grounding, or refusal guard mechanics. All algorithms reside in [`ai_platform/core/`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/).
- **Configuration, Not Hardcoded Logic:** Do **not** embed procedural query branches (such as `if query == ...`) in this module. All institutional signals are declared through structured data schemas.

---

## 2. Package Structure

```text
ai_platform/institutions/iitj/
├── __init__.py               # Exports PROFILE and adapter accessors
├── profile.py                # Defines the IITJ InstitutionProfile instance
├── semantic_registry.py      # Semantic term mapping and entity registry
├── lexical.py                # Lexical regex patterns and token normalizers
├── source_policy.py          # Weak ranking priors per source category
├── scope_policy.py           # In-scope and out-of-scope policy definitions
├── navigation.py             # Campus navigation and landmark registry
├── campus_navigation.py      # Deterministic location card generator
├── emergency.py              # Structured emergency contacts handler
├── prompts.py                # IITJ system tone and greeting definitions
└── data/                     # Structured configuration datasets
    ├── acronyms.json         # Acronym expansions (e.g., ITEP, APM, CSE)
    ├── aliases.json          # Entity and office aliases
    ├── phrases.json          # Institutional idioms and recognized multi-word phrases
    ├── semantic_terms.json   # Domain taxonomy and topic mapping
    ├── source_policy.json    # Source document priority weights
    ├── scope_policy.json     # Institutional scope boundary terms
    └── navigation.json       # Campus geographic coordinates and building metadata
```

---

## 3. Data vs. Corpus Distinction

> [!IMPORTANT]
> The files under `data/` in this package contain **metadata and vocabulary configuration only**, not the factual document corpus.

- **Configuration:** `ai_platform/institutions/iitj/data/` defines JSON files loaded at runtime to configure semantic understanding and lexical matching.
- **Factual Knowledge Base:** The actual indexed source documents reside in `data/data_iitj/iitj_rag_v1_docs_production/`.
- **Vector Database:** Indexed dense embeddings reside in `chroma_db/` under the Chroma collection `iitj_v1`.

---

## 4. Integration with Core Runtime

The core platform discovers and loads this adapter via [`ai_platform.institutions.loader.load_institution_profile()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/institutions/loader.py):

```python
from ai_platform.institutions.loader import load_institution_profile

# Loads the active IIT Jodhpur profile instance:
profile = load_institution_profile("iitj")
print(profile.display_name)             # "Indian Institute of Technology Jodhpur"
print(profile.vectorstore_collection)   # "iitj_v1"
```

The profile supplies the core LangGraph runtime nodes with tenant-specific boundary rules, fallback messages, and search priors without tightly coupling the reasoning engine to IIT Jodhpur.