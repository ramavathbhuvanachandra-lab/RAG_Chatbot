# Retrieval, Evidence Preservation & Verification

**Audit Date**: October 16, 2026  
**Implementation**: `ai_platform/core/retrieval/` & `ai_platform/core/evidence/`  

---

## 1. The 3-Channel Hybrid Retrieval Architecture

To achieve robust recall across both semantic concepts (e.g. "continuation criteria") and strict token lookups (e.g. course codes "MEL1010", acronyms "APM", or specific percentages "75%"), the platform employs a three-channel retrieval strategy:

```
                          User Query / Plan
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  [Dense Channel]          [BM25 Channel]          [Lexical Channel]
  VectorStore: ChromaDB    RankBM25 Token Match    Exact N-Gram Match
  Model: nomic-embed-text  Tokenized Corpus        aliases.json & synonyms
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                     [Reciprocal Rank Fusion (RRF)]
                     Score = ∑ w_i / (k + rank_i)
                                  ▼
                   [Harmonized Verification Gate]
                      Relevance >= 0.25 Threshold
                                  ▼
                   [Evidence Packaging (32 Units)]
```

### Channel 1: Dense Semantic Retrieval
- **Implementation**: [`ai_platform/runtime/vectorstore.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/runtime/vectorstore.py)
- **Model**: Local Ollama embedding model `nomic-embed-text:latest`
- **Corpus**: ChromaDB persistent collection `iitj_v1` (1,228 canonical document chunks).
- **Strengths**: Captures conceptual similarity when query words differ from policy headings.
- **Limitations**: Weak on exact numerical strings, course codes, and short acronyms.

### Channel 2: BM25 Token Retrieval
- **Implementation**: In-memory `rank_bm25.BM25Okapi` over whitespace/punctuation-tokenized documents.
- **Strengths**: High precision on specific institutional terms and administrative terminology.

### Channel 3: Lexical N-Gram & Alias Recall
- **Implementation**: [`ai_platform/core/retrieval/lexical_recall.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/retrieval/lexical_recall.py)
- **Data**: Static institutional dictionaries [`aliases.json`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/institutions/iitj/data/aliases.json) and [`semantic_terms.json`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/institutions/iitj/data/semantic_terms.json).
- **Strengths**: Recovers critical chunks containing domain synonyms (e.g. expanding "first class" to "60% aggregate" or "BTech" to "Bachelor of Technology").

---

## 2. Reciprocal Rank Fusion (RRF) Formulation

Outputs from all three channels are fused in [`ai_platform/core/retrieval/rrf.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/retrieval/rrf.py) using weighted Reciprocal Rank Fusion:

$$RRF(d) = \sum_{c \in \{\text{Dense, BM25, Lexical}\}} \frac{w_c}{k + \text{rank}_c(d)}$$

Where:
- Constant smoothing factor $k = 60$.
- Channel weights: $w_{\text{BM25}} = 0.40$, $w_{\text{Lexical}} = 0.30$, $w_{\text{Dense}} = 0.30$.
- Deduplication: Chunks with identical doc IDs are fused into a single candidate preserving multi-channel provenance flags.

---

## 3. Candidate Verification & Threshold Harmonization

Candidate verification in [`ai_platform/core/retrieval/verification.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/retrieval/verification.py) evaluates candidate suitability before ranking:

1. **Degree & Entity Constraints**: Rejects candidates referencing disjoint academic levels (e.g. discarding Ph.D. thesis submission chunks when the query explicitly asks for B.Tech freshman rules).
2. **Attribute Alignment**: Evaluates whether requested attributes (e.g. `fee`, `attendance`, `deadline`) appear in candidate text or headings.
3. **Harmonized Verification Threshold ($S \ge 0.25$)**:
   - *The Cliff Problem*: Prior to Stage 5, `verify_candidate` enforced $S \ge 0.28$ at line 428, while earlier filters used 0.25. High-quality policy chunks (such as `Attendance Requirements` 4.1.9 with score 0.2778) fell into this 0.002 gap and were discarded as `insufficient_relevance`.
   - *Resolution*: Harmonizing the threshold to 0.25 enabled valid institutional evidence to survive consistently without increasing false acceptances on adversarial controls.

---

## 4. Evidence Packaging & Context Budgets

Surviving candidates are processed into generation context in [`ai_platform/core/evidence/packaging.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/evidence/packaging.py):
- **Unit Capacity**: Bounded to **32 atomic evidence units** (increased from legacy limit of 16).
- **Character Budget**: Hard ceiling of **12,000 characters**.
- **Provenance Tags**: Each packaged unit includes its source document, heading breadcrumb, and unit identifier:
  ```
  Evidence [1] [Academic Progression -> Attendance Requirements]
  A student must maintain a minimum attendance of 75% in each course...
  ```

---

## 5. Formal Taxonomy of Evidence Loss

Evidence can fail to reach the generator at seven distinct stages:

```
[Raw Documents in Corpus]
   │
   ├── 1. RAW_RETRIEVAL_MISS          (Target chunk absent from top-20 across all 3 channels)
   ▼
[Fused Candidates]
   │
   ├── 2. VERIFICATION_FALSE_REJECTION (Candidate filtered out by threshold or entity mismatch)
   ▼
[Verified Candidates]
   │
   ├── 3. RANKING_BUDGET_LOSS         (Candidate dropped during top-k budget truncation / MMR)
   ▼
[Ranked Candidates]
   │
   ├── 4. EVIDENCE_BOUNDARY_DROP      (Boundary expander failed to preserve heading/context)
   ▼
[Evidence Units]
   │
   ├── 5. CLAIM_AUDIT_DROP            (Atomic unit extraction failed to extract required claim)
   ▼
[Evidence Package]
   │
   ├── 6. PACKAGING_BUDGET_DROP       (Context exceeded 32 units or 12,000 char budget)
   ▼
[Prompt Sent to LLM]
   │
   └── 7. PROMPT_TRUNCATION_LOSS      (Prompt tokens exceeded context window)
```

---

## 6. Empirical Retrieval & Evidence Benchmarks

### 6.1 Unseen Challenge 40Q Benchmark (`test_outputs/.../challenge_eval/`)
- Total Scenarios: 40 | Answerable: 25 | Controls: 15
- **Recall@10**: **96.0%** (24 / 25 answerable questions retrieved gold evidence in top 10).
- **Verification Survival**: **96.0%** (24 / 25 survived candidate verification).
- **`VERIFICATION_FALSE_REJECTION`**: **0** (Eliminated from 2 down to 0).
- **Channel Breakdown**:
  - Dense-only Recall@10: 36.0%
  - BM25-only Recall@10: 68.0%
  - Lexical-only Recall@10: 72.0%
  - Configured Hybrid (RRF): 68.0%

### 6.2 Phase B 180Q Suite Bounded Evaluation (`test_outputs/.../phase_b_eval/`)
- Scenarios Evaluated: 67 (bounded to 516.0s / 540s limit)
- **Recall@10**: **98.4%** (63 / 64 answerable questions).
- **Verification Candidate Survival**: **96.9%** (62 / 64 survived).
- **Final Context Inclusion**: **85.9%** (55 / 64 answerable questions had gold evidence in final context).
- **Channel Breakdown**:
  - Dense-only Recall@10: 56.2%
  - BM25-only Recall@10: 85.9%
  - Lexical-only Recall@10: 92.2%
  - Configured Hybrid (RRF): 92.2%
