# Documentation Manifest & Inventory

**Audit Date**: October 16, 2026  
**Destination Root**: `docs/architecture_audit_20261016/`  
**Copied Documentation Directory**: `docs/architecture_audit_20261016/source_docs/`  

---

## 1. Inventory of Copied Markdown Documents

Exactly 17 eligible Markdown documents were copied from the repository into `source_docs/` while strictly preserving their relative paths. Original files remain untouched in place.

| Original Relative Path | Copy Path in `source_docs/` | Status | Architectural Relationship | Alignment & Conflict Analysis |
| :--- | :--- | :---: | :--- | :--- |
| `flow.md` | `source_docs/flow.md` | **Current** | Canonical runtime execution flow & graph topology. | Fully aligned with active working tree on `ai-platform`. Updated to reflect UI migration. |
| `AI_PLATFORM_MASTER_HANDOFF.md` | `source_docs/AI_PLATFORM_MASTER_HANDOFF.md` | **Current** | Single master handoff document, defect registry, and benchmark scorecard. | Primary project state reference. Contains comprehensive history and latest Stage 5 results. |
| `README.md` | `source_docs/README.md` | **Historical** | Root repository overview. | Outdated. References legacy `backend/` and monolithic structure rather than modular `ai_platform/`. |
| `AI_PLATFORM_ARCHITECTURE.md` | `source_docs/AI_PLATFORM_ARCHITECTURE.md` | **Current** | Canonical architectural blueprint and modular design specifications. | Aligned with `ai_platform/` core abstractions, though UI was listed as pending cutover. |
| `INSTITUTION_CHANGE_MAP.md` | `source_docs/INSTITUTION_CHANGE_MAP.md` | **Current** | Implementation guide for multi-institutional onboarding. | Aligned. Specifies the ~2-day target for new institution configuration. |
| `ai_platform/institutions/iitj/README.md` | `source_docs/ai_platform/institutions/iitj/README.md` | **Current** | IIT Jodhpur institutional data, aliases, and policies documentation. | Aligned with active `ai_platform/institutions/iitj/` implementation. |
| `tests_reusable/rag_quality/README.md` | `source_docs/tests_reusable/rag_quality/README.md` | **Current** | Test runner guide for reproducible quality benchmarks. | Aligned with `run_production_reliability_eval.py` runner contracts. |
| `tests_reusable/test_diagnosis/README.md` | `source_docs/tests_reusable/test_diagnosis/README.md` | **Historical** | Documentation of early candidate retrieval diagnosis setup. | Historical context for Stage 2 diagnostics. |
| `tests_reusable/test_diagnosis/stagewise_rebuild_plan.md` | `source_docs/tests_reusable/test_diagnosis/stagewise_rebuild_plan.md` | **Historical** | Plan for rebuilding retrieval and verification stages. | Historical. The planned fixes have since been implemented in `ai_platform/core/retrieval/`. |
| `tests_reusable/test_diagnosis/reports/stage_diagnosis_v2/diagnosis_v2.md` | `source_docs/tests_reusable/test_diagnosis/reports/stage_diagnosis_v2/diagnosis_v2.md` | **Historical** | Stage 2 candidate diagnosis forensic report. | Frozen diagnostic record. |
| `tests_reusable/test_diagnosis/reports/stage_diagnosis_v2/stage2_forensic_gate.md` | `source_docs/tests_reusable/test_diagnosis/reports/stage_diagnosis_v2/stage2_forensic_gate.md` | **Historical** | Forensic gate assessment for Stage 2 candidate verification. | Frozen diagnostic record. |
| `reports/PROJECT_STATUS_AUDIT_2026-10-12.md` | `source_docs/reports/PROJECT_STATUS_AUDIT_2026-10-12.md` | **Historical** | Pre-refactoring project status audit (October 12, 2026). | Superseded by `_v2.md` and subsequent handoffs. |
| `reports/PROJECT_STATUS_AUDIT_2026-10-12_v2.md` | `source_docs/reports/PROJECT_STATUS_AUDIT_2026-10-12_v2.md` | **Historical** | Revised project status audit detailing legacy backend defects. | Useful forensic record of initial architectural defects. |
| `reports/RAG_CORE_RELIABILITY_ASSESSMENT.md` | `source_docs/reports/RAG_CORE_RELIABILITY_ASSESSMENT.md` | **Historical** | Deep reliability assessment of candidate retrieval and ranking. | Informational baseline. Documents earlier failure modes that informed current code. |
| `rag_v3_final_bundle/README.md` | `source_docs/rag_v3_final_bundle/README.md` | **Archived** | Archived snapshot documentation for RAG V3 candidate bundle. | Archived artifact. Does not represent active codebase. |
| `rag_v3_final_bundle/VALIDATION_REPORT.md` | `source_docs/rag_v3_final_bundle/VALIDATION_REPORT.md` | **Archived** | Validation report accompanying the RAG V3 candidate bundle. | Archived artifact. |
| `retrieval_v4_bundle/README.md` | `source_docs/retrieval_v4_bundle/README.md` | **Archived** | Archived snapshot documentation for Retrieval V4 bundle. | Archived artifact. |

---

## 2. Intentionally Excluded Files (Referenced by Path)

To adhere strictly to audit safety rules, prevent repository bloat, and avoid duplicating massive generated evaluation outputs or raw datasets, the following files were cataloged but intentionally excluded from `source_docs/`:

1. **Massive Diagnostic Dump**:
   - `tests_reusable/test_diagnosis/reports/retrieval_diagnosis.md` (909 KB). Contains raw candidate dumps across 50+ queries.
2. **Runtime Evaluation Reports & Logs**:
   - `test_outputs/rag_quality/*/*.md` (e.g. `guard_report.md`, `generalization_report.md`, `starting_limitations.md`). These are point-in-time outputs of automated test runs.
3. **Benchmark Datasets**:
   - `tests_reusable/rag_quality/generalization_180q_scenarios.json` (SHA-256: `90c8b46a...`)
   - `tests_reusable/rag_quality/unseen_challenge_40q_scenarios.json` (SHA-256: `50969800...`)
   - `tests_reusable/rag_quality/retrieval_120q_scenarios.json` (SHA-256: `75f6a1d1...`)
4. **Binary & Cache Artifacts**:
   - `chroma_db/` (Vector database collection `iitj_v1` with 1,228 chunks)
   - `rag_v3_final_bundle.zip` (Compressed archive)
   - `.env` and environment configuration files.

---

## 3. Duplicate Candidates & Obsolete Bundles

The repository contains several parallel, duplicate, or historical snapshot directories:

1. **Archived Bundles vs Active Code**:
   - `rag_v3_final_bundle/` (contains redundant copies of core graph and retrieval files from Oct 4–6).
   - `retrieval_v4_bundle/` (contains redundant copies of retrieval contracts from Oct 13).
   - `phase1_backup_20261006_*/` (untracked directories containing dated backups).
2. **Duplicate Reports**:
   - `reports/PROJECT_STATUS_AUDIT_2026-10-12.md` is an older, partial version of `reports/PROJECT_STATUS_AUDIT_2026-10-12_v2.md`.
3. **Backup Source Code**:
   - More than 25 untracked files with extensions `.before_*`, `.bak`, `.stage2_backup_*`, and `.stage2_hotfix_*` exist inside `ai_platform/core/` directories.

---

## 4. Stale or Contradictory Documentation Findings

1. **Root `README.md` Stale Architecture**:
   - Root `README.md` instructs users to start the app via `streamlit run app.py` and claims the architecture uses `backend/retriever.py` and `backend/graph.py`. It fails to mention `ai_platform/` entirely.
2. **Historical Handoff UI Claims**:
   - Prior sections of `AI_PLATFORM_MASTER_HANDOFF.md` (pre-October 16, 2026) state that `app.py` crashes due to `backend/graph.py` and that UI migration is deferred. This was resolved in Stage 5, but uncurated readers of historical handoff entries could be confused without reading the latest change log.
3. **Defect-1 Status vs Actual Fix**:
   - Historical notes describe DEFECT-1 (`backend/graph.py:57` importing non-existent `ai_platform.core.query_pipeline`). While `app.py` bypassed this by pointing directly to `ai_platform.runtime.engine`, `backend/graph.py` still contains the broken import.
