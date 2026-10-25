# Institutional RAG Chatbot Platform

> A lean, multi-tenant, institution-grounded Retrieval-Augmented Generation (RAG) platform delivering authoritative, hallucination-resistant answers to student, faculty, and administrative inquiries. Reference deployment: **Indian Institute of Technology Jodhpur (IIT Jodhpur)**.

---

## 1. Executive Summary & Capabilities

The Institutional RAG Chatbot provides a reliable conversational interface over official institutional documents, including academic ordinances, admission criteria, fee structures, hostel rules, administrative policies, and campus services.

### Core Capabilities
- **High-Precision Evidence Grounding:** Answers are synthesized exclusively from verified institutional evidence extracted from authoritative documents, eliminating speculative LLM hallucinations.
- **Multi-Intent Query Understanding & Decomposition:** Complex queries combining multiple institutional policies (e.g., fee structure waivers + hostel rent + mess advances) are automatically decomposed into focused sub-queries, retrieved across distinct source categories, and reconciled into coherent answers.
- **Hybrid Retrieval & Reciprocal Rank Fusion (RRF):** Combines dense semantic vector retrieval with lexical keyword search (BM25), fused via weighted RRF to optimize both semantic concept recall and exact-match keyword precision.
- **Candidate Verification & Strict Evidence Boundary:** Retrieved candidates undergo institutional qualification and category prior scoring. Only verified-anchor evidence units enter downstream claim auditing and context packaging.
- **Bounded Context Packaging:** Evidence units are deduplicated and budgeted under a deterministic 12,000-character context window, ensuring the local LLM receives high-signal information without truncation or attention dilution.
- **Query-Aware Extraction & Polarity Protection:** Protects against LLM negation inversion on alternative qualifying pathways (e.g., "OR a first-class degree"), with automated extractive fallback and academic-year token signature normalization.
- **Strict Guardrails & Safe Abstention:** Detects and blocks malicious prompt injections, system credential probes, physical security violations, and exam leak queries. Gracefully abstains when evidence is insufficient.
- **Transparent Attribution:** Supplies up to 4 authoritative citations in collapsible UI cards. Suppresses citations during safe refusals to prevent misleading attribution.
- **Clean Decoupled Architecture:** Cleanly separates the Streamlit presentation layer (`frontend/`), the reusable reasoning engine (`ai_platform/core/`), institution-specific configurations (`ai_platform/institutions/`), local infrastructure runtimes (`ai_platform/runtime/`), and knowledge package management (`scripts/knowledge_package.py`).

---

## 2. Verified Technology Stack

| Layer | Component | Verified Implementation |
|---|---|---|
| **Presentation** | Web UI & Session State | [Streamlit](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/app.py) (`frontend/` package: modular components, session manager, UI config loader) |
| **Service Adapter** | Presentation-Service Bridge | [`frontend.clients.assistant_client`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py) → [`ai_platform.service.assistant_service`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/service/assistant_service.py) |
| **Workflow Engine** | Orchestration & Graph Pipeline | [LangGraph](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/graph/nodes.py) (`ai_platform.core.graph.nodes.CoreNodes`, typed `RAGState`) |
| **Local LLM** | Answer & Query Intelligence | `qwen2.5:3b` via [Ollama](http://localhost:11434) (temperature = 0.0 for deterministic factual generation) |
| **Embeddings** | Semantic Vector Embeddings | `nomic-embed-text:latest` via [Ollama](http://localhost:11434) |
| **Vector Store** | Dense Retrieval Storage | [ChromaDB](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/chroma_db/) (`collection_name = "iitj_v1"`, Cosine distance; staging: `iitj_iit_jodhpur_rag_package_v2_staging`) |
| **Lexical Search** | Exact Keyword Retrieval | `rank_bm25` ([`BM25Retriever`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/runtime/retrieval.py#L30)) over corpus chunks |
| **Release Management** | Knowledge Ingestion Engine V2 | [`scripts/knowledge_package.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/scripts/knowledge_package.py) (Package validator, indexer, staging evaluator) |
| **Python Runtime** | Core Execution Environment | Python 3.11 (`/opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/python`) |

---

## 3. Repository Architecture & Layout

The codebase has undergone full architectural consolidation. The obsolete monolithic prototype (`backend/`), historical phase test folders, and temporary task dumps have been completely removed.

```text
RAG_Chatbot/
├── app.py                     # Canonical Streamlit web application entry point
├── requirements.txt           # Verified Python production & test dependencies
├── Dockerfile                 # Production container image definition
├── docker-compose.yml         # Container orchestration specification
├── .dockerignore              # Strict production image exclusion filters
├── .gitignore                 # Machine-specific and disposable artifact filters
│
├── frontend/                  # Modular Streamlit UI presentation layer
│   ├── app.py                 # Core UI runner and multi-turn state controller
│   ├── clients/               # Service client adapters (AssistantClient)
│   ├── components/            # UI components (chat, cards, sidebar, error modal)
│   ├── config/                # UI theming, branding, and quick-prompt loaders
│   └── state/                 # Session state manager and conversation storage
│
├── ai_platform/               # Reusable multi-institution AI reasoning engine
│   ├── core/                  # Core RAG algorithmic modules (answering, query, retrieval)
│   │   ├── answering/         # Evidence boundary, claim audit, generator, grounding
│   │   ├── conversation/      # Multi-turn history, coreference resolution
│   │   ├── graph/             # Typed LangGraph nodes, state, and pipeline workflow
│   │   ├── query/             # Query planner, intent decomposition, scope guard
│   │   └── retrieval/         # Hybrid fusion (RRF), ranking, candidate verification
│   ├── institutions/          # Multi-tenant institution configuration & adapters
│   │   ├── base.py            # Institution contract definition
│   │   ├── registry.py        # Institution adapter registry
│   │   └── iitj/              # Reference tenant: IIT Jodhpur adapters & assets
│   ├── packages/              # KIE V2 package loader, parser, and manifest adapter
│   ├── runtime/               # Local execution engine, ChromaDB client, BM25 retriever
│   └── service/               # Headless assistant service API (AssistantService)
│
├── data/                      # Knowledge packages and release manifests
│   ├── staging_release_iitj.json   # Release configuration (active: v1, staged: v2)
│   └── data_iitj/             # IIT Jodhpur knowledge assets
│       └── iit_jodhpur_rag_package_v2/  # Verified KIE V2 package (161 accepted docs)
│           ├── manifest.json  # Cryptographic document manifest & dataset signatures
│           ├── documents/     # Canonical markdown documents by category
│           └── knowledge/     # Structured JSON metadata (departments, programs)
│
├── tests_frontend/            # Automated UI mechanics and chat flow tests (41 tests)
├── tests_reusable/            # Reusable platform contracts and regression suites (94 tests)
│   ├── production_readiness/  # Tenant isolation, synthetic institution, service tests
│   ├── query_planner/         # Query planner contracts and decomposition tests
│   ├── rag_quality/           # Evidence packaging, hardening, and regression tests
│   └── standalone/            # Zero-backend dependency & runtime invariant tests
├── tests/                     # Diagnostic benchmarks and contract tests (12 tests)
│   ├── regression_questions.json  # Canonical benchmark question bank
│   └── diagnostics/           # 50-question benchmark runner and contract tests
│
└── scripts/                   # Operational and release management CLI utilities
    ├── knowledge_package.py   # KIE V2 package validation, indexing, and staging CLI
    ├── verify_streamlit_live_ui.py  # Headless live Streamlit UI verification
    └── run_student_evaluation.py    # Natural student query evaluation runner
```

---

## 4. Environment Setup & Dependency Installation

### Prerequisites
- macOS or Linux with Python 3.11 installed.
- [Ollama](https://ollama.com/) running locally on port 11434 with models pulled:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull nomic-embed-text:latest
  ```

### Dependency Installation
Create and activate your Python 3.11 virtual environment or Conda environment, then install dependencies:

```bash
# Using Conda:
conda create -n iitj_rag python=3.11 -y
conda activate iitj_rag

# Install dependencies:
pip install -r requirements.txt
```

---

## 5. How to Launch Streamlit

### Production Launch (Default)
The canonical application is launched via the root launcher [`app.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/app.py), which starts the modular Streamlit frontend connected to the active production knowledge base (`iitj_v1`):

```bash
PYTHONPATH=. streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Staging Release Preview
To preview and test a staged KIE V2 release without modifying production database pointers:

```bash
RAG_USE_STAGING=1 PYTHONPATH=. streamlit run app.py
```
When `RAG_USE_STAGING=1` is set:
- The assistant service dynamically points to the staging ChromaDB collection (e.g., `iitj_iit_jodhpur_rag_package_v2_staging`).
- The UI header displays a visual badge indicating that staging preview mode is active.

---

## 6. KIE V2 Knowledge Package Management

The repository includes a comprehensive CLI tool [`scripts/knowledge_package.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/scripts/knowledge_package.py) for validating, staging, indexing, and evaluating Knowledge Ingestion Engine (KIE) V2 packages.

### 1. Validate a Knowledge Package
Validates package directory structure, `manifest.json`, SHA256 checksums, and dataset schemas:
```bash
python scripts/knowledge_package.py validate --package-root data/data_iitj/iit_jodhpur_rag_package_v2
```

### 2. Stage a Package
Stages a validated package in the institutional release manifest (`data/staging_release_<institution_id>.json`) without activating it in production:
```bash
python scripts/knowledge_package.py stage \
  --institution iitj \
  --package-root data/data_iitj/iit_jodhpur_rag_package_v2 \
  --release-id iit_jodhpur_rag_package_v2
```

### 3. Build Staging Indexes
Builds dense ChromaDB embeddings and BM25 lexical index for the staged package in an isolated staging collection:
```bash
python scripts/knowledge_package.py build-index --institution iitj
```

### 4. Evaluate Staging Release
Runs benchmark probes against the staging release and verifies evidence survival:
```bash
python scripts/knowledge_package.py evaluate --institution iitj
```

### 5. Check Release Status
Inspect current production active release and staging release details:
```bash
python scripts/knowledge_package.py status --institution iitj
```

### 6. Activate Staging Release
Promotes the staging release to active production status:
```bash
python scripts/knowledge_package.py activate --institution iitj
```

---

## 7. How to Onboard Another Institution

The platform is designed for clean multi-tenant reuse. To onboard a new university or college (e.g., `srgi` or `nitk`):

1. **Create Institution Configuration:**
   Create a new package directory under `ai_platform/institutions/<institution_id>/` containing:
   - `adapter.py`: Subclass of `InstitutionAdapter` defining institution metadata, acronyms, and aliases.
   - `campus_navigation.py`: Structured campus locations and aliases.
   - `emergency.py`: Structured campus emergency contacts and helplines.
   - `data/`: Structured JSON files (contacts, navigation targets, vocabulary).

2. **Ingest Institutional Documents:**
   Assemble your official documents into a valid KIE V2 package under `data/data_<institution_id>/<package_name>/`:
   - `manifest.json`: List of accepted documents, categories, and hashes.
   - `documents/`: Markdown or text files grouped by category (academics, fees, admissions, hostels).
   - `knowledge/`: Optional structured datasets (programs, departments).

3. **Validate and Index:**
   ```bash
   python scripts/knowledge_package.py validate --package-root data/data_<institution_id>/<package_name>
   python scripts/knowledge_package.py stage --institution <institution_id> --package-root data/data_<institution_id>/<package_name>
   python scripts/knowledge_package.py build-index --institution <institution_id>
   ```

4. **Verify Tenant Isolation:**
   Run the synthetic institution tenant isolation suite to ensure that data, queries, and vectorstores never cross-contaminate between institutions:
   ```bash
   PYTHONPATH=. pytest tests_reusable/production_readiness/test_synthetic_institution.py -v
   ```

---

## 8. Automated Regression Testing

The test suite has been consolidated into 147 clean, passing automated tests with zero collection errors.

### Run All Core Regression Suites (147 Tests — 100% PASS)
Run the consolidated regression suite across UI mechanics, platform contracts, query planning, and evidence hardening:

```bash
PYTHONPATH=. pytest \
  tests_frontend/ \
  tests/ \
  tests_reusable/production_readiness/ \
  tests_reusable/query_planner/test_query_planner_contract.py \
  tests_reusable/test_streamlit_canonical_runtime.py \
  tests_reusable/standalone/ \
  tests_reusable/rag_quality/test_guard_and_evidence_hardening_regression.py -v
```

### Run 50-Question Canonical Benchmark
Runs the canonical 50-question institutional question bank and reports grounding accuracy:
```bash
PYTHONPATH=. python tests/diagnostics/run_50_canonical.py
```

### Run Headless Streamlit Live UI Verification
Simulates multi-turn interactive chat sessions through `UI.chat.get_response`:
```bash
PYTHONPATH=. python scripts/verify_streamlit_live_ui.py
```

---

## 9. Directory Hygiene & Disposable Artifacts

To maintain repository cleanliness and deployment performance:

### Disposable Artifacts (Never Commit)
The following directories and files are strictly generated at runtime and are ignored by `.gitignore`:
- `chroma_db/`: Local ChromaDB vector database index files.
- `test_outputs/`, `test_debug/`: Temporary test run dumps, traces, and metrics.
- `.pytest_cache/`, `.ruff_cache/`: Test and linter runner caches.
- `*.canonical_chunks_cache*.json`: Local lexical BM25 chunk caches.
- `.env`, `.env.*`: Local environment secrets and API keys.
- `__pycache__/`, `*.pyc`: Python bytecode caches.

### Docker Build Hygiene
The production [`Dockerfile`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/Dockerfile) utilizes [`.dockerignore`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/.dockerignore) to ensure test files (`tests/`, `tests_frontend/`, `tests_reusable/`), one-off scripts (`scripts/`), documentation (`docs/`, `reports/`), and cache directories are excluded from the production container image, keeping container builds small, secure, and fast.

---

## 10. Subsystem Documentation Links

- **Frontend Subsystem:** [`frontend/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/README.md) — Modular architecture, Streamlit components, theming, and session management.
- **Frontend Test Suite:** [`tests_frontend/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_frontend/README.md) — Core UI mechanics and conversation-quality regression tests.
- **Reference Institution Adapter:** [`ai_platform/institutions/iitj/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/institutions/iitj/README.md) — Vocabulary data, acronym expansion, and campus direct handlers.
- **RAG Quality & Regression Testing:** [`tests_reusable/rag_quality/README.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/README.md) — Regression test suites, failure classifications, and canonical scenario specifications.
- **Master Platform Handoff:** [`AI_PLATFORM_MASTER_HANDOFF.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md) — Comprehensive architecture, contract specifications, and system invariants.
- **Graph Pipeline Details:** [`AI_PLATFORM_ARCHITECTURE.md`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_ARCHITECTURE.md) — Typed state fields, node responsibilities, and routing logic.
