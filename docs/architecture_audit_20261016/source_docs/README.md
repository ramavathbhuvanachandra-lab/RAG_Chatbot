# College AI Platform (RAG_Chatbot)

> A reusable, institution-agnostic RAG platform for building authoritative, knowledge-grounded AI assistants for colleges and universities. Reference institution: **IIT Jodhpur**.

---

## Quick Navigation

- **Primary Project Handoff**: [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md) — **Start here!** Single maintained engineering record and handover guide.
- **Architecture Reference**: [AI_PLATFORM_ARCHITECTURE.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_ARCHITECTURE.md) — Core pipeline stages, contracts, and graph architecture.
- **Institution Change Map**: [INSTITUTION_CHANGE_MAP.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/INSTITUTION_CHANGE_MAP.md) — Multi-institution onboarding guide and boundary rules.
- **RAG Quality Tests**: [tests_reusable/rag_quality/README.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/README.md) — Test conventions, failure classes, and regression suites.

---

## Architecture Overview

```text
ai_platform/
├── core/           <- SHARED, institution-agnostic RAG intelligence
│   ├── graph/      <- LangGraph wiring, node orchestration, typed RAGState
│   ├── query/      <- Query understanding, planning, classification
│   ├── retrieval/  <- Dense + BM25 retrieval, RRF fusion, candidate verification, ranking
│   ├── evidence/   <- Context grouping, coverage, boundary enforcement, packaging
│   └── answering/  <- Answer generation, grounding verification, output guarding
├── institutions/
│   └── iitj/       <- Reference institution (IIT Jodhpur): vocabulary, aliases, policies
└── runtime/        <- LLM clients (Ollama), embeddings, ChromaDB vector store, composition root
```

### Architectural Boundaries (Strictly Enforced)

- `ai_platform/core/` is reusable across any college or university; it contains **zero** hardcoded institutional facts or rules.
- `ai_platform/institutions/<id>/` contains institution-specific configuration, aliases, and policies — **no** retrieval or generation logic.
- `ai_platform/runtime/` handles infrastructure connections (Ollama, ChromaDB); it owns no retrieval policy.
- `backend/` is the **historical legacy codebase**. Retained for reference only. **Never import, repair, extend, or use as a dependency.**
- The existing UI (`app.py`, `UI/`) is legacy and intentionally not migrated yet.

---

## Environment & Prerequisites

- **Python Environment**: `/opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/python`
- **Ollama Service**: Running locally at `http://localhost:11434`
  - Active LLM: `qwen2.5:3b` (query understanding, query planning, answer generation)
  - Active Embeddings: `nomic-embed-text:latest`
- **Corpus & Vector Store**:
  - Raw documents: `data/data_iitj/iitj_rag_v1_docs_production/`
  - ChromaDB store: `chroma_db/` (collection: `iitj_v1`)

---

## Fast Verification Commands

All commands must be run from the repository root with `PYTHONPATH=.`:

```bash
# 1. RAG quality unit tests (fast, deterministic, <1s, no LLM required):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/ -v -m "not integration"

# 2. Benchmark runner safety tests (5 tests, <1s):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/test_runner_safety.py -v

# 3. Phase 1 contract tests (12 tests, 100% PASS):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest \
  tests/diagnostics/test_phase1_contracts.py \
  tests/diagnostics/test_phase1_adversarial.py \
  tests/diagnostics/test_phase1_runtime_contract.py -v

# 4. Canonical 50-Question Benchmark (safe, isolated output directory):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/python tests/diagnostics/run_50_canonical.py \
  --output-dir test_outputs/rag_quality/baseline_20261013
```

---

## Current Project Status (Fresh Baseline: Oct 13, 2026)

- **Benchmark Results**: 50/50 executed | **43 Structural PASS**, **7 CHECK**, **0 ENGINE_ERROR** (~27.97s/query).
- **Benchmark Reports**: Stored safely in `test_outputs/rag_quality/baseline_20261013/` (Historical Oct 6 baseline preserved untouched).
- **Diagnosed Defect Priorities**:
  1. **Fix A (C2 & C5)**: Answering model polarity inversion on explicit OR alternatives (M.Sc. first-class and Ph.D. 4-year degree). Proposed patch ready in `generator.py` and `grounding.py`.
  2. **Fix B (C4)**: Planner misclassification of document deadline policy (`adm14`) causing retrieval query misrouting.
  3. **Fix C**: Scope classifier false out-of-scope halts on institutional percentage queries (`msc06`, `msc07`, `phd03`, `phd04`).
