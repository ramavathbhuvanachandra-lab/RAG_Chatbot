# Architecture Audit & Documentation Library (October 16, 2026)

**Repository**: `RAG_Chatbot`  
**Active Branch**: `ai-platform`  
**Audit Milestone**: Stage 5 — Guard Hardening, Evidence Preservation, & UI Migration  
**Auditor**: Senior AI Systems Architect, RAG Reliability Engineer & Documentation Specialist  
**Audit Mode**: Strictly Read-Only Architecture Inspection & Documentation Consolidation  

---

## 1. Executive Purpose

This dedicated documentation library was created to provide a complete, authoritative, and evidence-based architectural health audit of the `RAG_Chatbot` repository. It consolidates scattered architectural references, reconciles historical claims against the active working tree, inventories technical debt, and outlines actionable next engineering steps without modifying any production source code or disrupting active background evaluation tasks.

---

## 2. Source-of-Truth Hierarchy

When navigating this codebase and evaluating architectural assertions, the following hierarchy of authority MUST be strictly applied:

```
[Level 1: Canonical Source Code] (Highest Authority)
   ai_platform/core/          -> Reusable, institution-agnostic RAG engine
   ai_platform/institutions/  -> University-specific policies, vocabulary, and aliases
   ai_platform/runtime/       -> Ollama LLM, embeddings, ChromaDB, and composition root
   app.py                     -> Streamlit user-facing entry point

[Level 2: Canonical System Flow & Master Handoff]
   flow.md                    -> Authoritative end-to-end execution flow & graph topology
   AI_PLATFORM_MASTER_HANDOFF.md -> Single canonical project status, milestone scorecard & defect registry

[Level 3: Verified Test & Benchmark Suites]
   tests_reusable/rag_quality/ -> 180Q generalization & 40Q challenge test suites
   tests_reusable/test_streamlit_canonical_runtime.py -> Canonical UI integration verification

[Level 4: Historical Reports & Archived Bundles] (Informational Only)
   reports/                   -> Point-in-time forensic snapshots (Oct 12-14, 2026)
   tests_reusable/test_diagnosis/ -> Early candidate retrieval diagnostics
   rag_v3_final_bundle/       -> Archived legacy bundle snapshots
   backend/                   -> STRICTLY QUARANTINED prototype reference code
```

> [!IMPORTANT]
> **No Historical Overwrites**: Documents in `reports/` or `tests_reusable/test_diagnosis/` represent frozen point-in-time historical diagnostic snapshots. They are preserved for auditability and must never be interpreted as current system contracts.

---

## 3. Directory Navigation Map

The documentation library is organized into the following specialized modules:

| Document | Purpose & Scope |
| :--- | :--- |
| [manifest.md](manifest.md) | Inventory of all copied source documents, excluded generated logs, and duplicate candidates. |
| [architecture_map.md](architecture_map.md) | Structural subsystem breakdown, directory boundaries, and component responsibilities. |
| [runtime_call_flow.md](runtime_call_flow.md) | End-to-end execution flow tracing from user prompt to final grounded response. |
| [retrieval_and_evidence.md](retrieval_and_evidence.md) | 3-channel retrieval (Dense, BM25, Lexical), RRF fusion, verification, and evidence survival. |
| [guards_and_failure_handling.md](guards_and_failure_handling.md) | Input-level scope/security guarding, answer grounding checks, fail-closed handling, and abstention. |
| [multi_tenant_architecture.md](multi_tenant_architecture.md) | Multi-institutional scaling, isolation boundaries, institution profiles, and current readiness. |
| [ui_and_integration.md](ui_and_integration.md) | Streamlit UI integration status, runtime entry points, and residual legacy bindings. |
| [technical_debt_inventory.md](technical_debt_inventory.md) | Comprehensive, evidence-based inventory of 7 confirmed technical debt findings. |
| [next_engineering_steps.md](next_engineering_steps.md) | Top 5 ranked engineering initiatives and the single recommended next action. |
| [source_docs/](source_docs/) | Exact replicas of eligible Markdown documents preserving their relative repository paths. |

---

## 4. Operational & Safety Invariants Enforced During Audit

1. **Zero Working-Tree Modifications**: No existing Python code, tests, configs, git state, or test outputs were modified, renamed, or moved.
2. **Preservation of Live Background Evaluation**: A live local Ollama generation process (`PID 61670` running `--mode generate --max-live-cases 15 --resume`) is actively executing in the environment. This audit made zero competing CPU/GPU calls, zero vector store writes, and zero file access to active evaluation checkpoints.
3. **Dedicated Artifact Sandbox**: All newly generated audit artifacts reside strictly within `docs/architecture_audit_20261016/`.
