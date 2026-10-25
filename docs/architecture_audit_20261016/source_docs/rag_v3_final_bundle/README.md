# RAG V3 — Verification + Final Context Builder

> [!NOTE]
> **Historical Patch Bundle**: This directory documents the historical RAG V3 verification and context builder bundle.
> The integrated canonical codebase resides under `ai_platform/core/`.
> For active engineering status and benchmarks, see [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

This bundle freezes the single-intent critical path into a small, production-oriented flow:

```text
USER QUERY
   -> Query Understanding
   -> Dense + BM25
   -> RRF
   -> Candidate Verification (hard safety/relevance boundary)
   -> Ranking (verified candidates only)
   -> Final Context Builder
   -> ONE Answer LLM
   -> Answer Grounding
   -> Answer Guard
   -> RESPONSE
```

## What changed

### 1. Verification is now a pre-LLM safety/relevance boundary

`ai_platform/core/retrieval/verification.py`

Verification no longer requires one chunk to contain the complete answer. Normal eligibility, requirements, descriptive, process, and list questions may use multiple complementary verified chunks.

Hard rejection remains for explicit conflicts, required scope mismatches, clearly insufficient relevance, and explicit relational/comparison questions where the relation is absent locally.

Direct factual lookups such as a percentage, fee, GATE requirement, or years of experience do **not** require a target↔attribute proof inside one chunk.

### 2. The old evidence/coverage/claim/package gates are removed from the single-intent critical path

`retrieve_and_qualify_node` now runs:

```text
understand -> retrieve -> RRF -> verify -> rank -> final context
```

No post-ranking evidence sufficiency gate, coverage gate, claim audit, or second scope filter is used before generation.

The existing `EvidencePackage` object is retained only as the answer-generator transport contract.

### 3. Final Context Builder

`ai_platform/core/retrieval/final_context_builder.py`

This module never retrieves or verifies candidates. It only works on `ranked_candidates` and performs:

- query-aware sentence/paragraph selection
- exact/near duplicate removal
- local heading awareness for program/admission scope
- lightweight diversity selection
- obvious generic noise suppression
- deterministic token/character budget enforcement
- no new candidate creation

No LLM call is made by the context builder.

## Install

From the repository root:

```bash
bash /path/to/rag_v3_final_bundle/install_rag_v3.sh
```

The installer backs up the active `ai_platform/core/nodes.py` or `ai_platform/core/graph/nodes.py` before modifying it.

## Verification tests

Run the component and real-source checks:

```bash
PYTHONPATH=. pytest -q \
  tests_reusable/test_diagnosis/test_verification_v3.py \
  tests_reusable/test_diagnosis/test_verification_v3_adversarial.py \
  tests_reusable/test_diagnosis/test_final_context_builder.py \
  tests_reusable/test_diagnosis/test_real_iitj_context_v3.py \
  tests_reusable/test_diagnosis/test_real_iitj_15_v3.py
```

Run the active graph contract check:

```bash
PYTHONPATH=. pytest -q \
  tests_reusable/test_diagnosis/test_node_pipeline_contract_v3.py
```

Inspect exactly what would be sent to the answer LLM for two real IIT Jodhpur questions:

```bash
PYTHONPATH=. python -u show_two_real_contexts.py
```

This inspection stops **before** the answer LLM. It prints the verified candidate count and the exact final context string.

## Important validation boundary

The real 15-case tests use the actual IIT Jodhpur DOCX knowledge sources and exercise verification + ranking-style ordering + final context construction deterministically. They do not claim to replace a live Ollama/Chroma end-to-end run.

The answer-generation path, answer-grounding path, and answer-guard path are intentionally not redesigned in this bundle.
