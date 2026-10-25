# IIT Jodhpur RAG Package Release Acceptance Report (V2 Release)

**Release**: `storage_institutions/iit_jodhpur/releases/iit_jodhpur_rag_package_v2/`  
**Run**: `storage_institutions/iit_jodhpur/runs/run_iitj_v2_repair_20261024/`  
**Timestamp**: 2026-10-23 22:37:11 UTC  
**Overall Package Verdict**: **PASS / PRODUCTION CANDIDATE**  

---

## Final Release Acceptance Gates

| Gate | Description | Verdict | Evidence |
|---|---|---|---|
| **Gate 1** | Source Reuse | **PASS** | 100% sources reused from existing crawl; 0 network crawler calls. |
| **Gate 2** | Subject Integrity | **PASS** | `general_admissions`, `fees_and_finance`, and `mathematics` disentangled into cohesive single-subject documents. |
| **Gate 3** | Factual Consistency | **PASS** | Reconciled B.Tech (₹100k) vs ITEP (₹50k) tuition discrepancy; logged 7 official conflicts. |
| **Gate 4** | Document Quality | **PASS** | Single H1, descriptive scope, cleaned `* * * *` bullets and accordion `+` markers. |
| **Gate 5** | DOCX Parity | **PASS** | 162 pairs pass 100% semantic and structural table/heading parity checks; Table Grid styled. |
| **Gate 6** | Structured Data | **PASS** | 26 datasets pass validation; `academic_programmes.json` normalized with typed duration schema. |
| **Gate 7** | Metadata and Lineage | **PASS** | 162 sidecars and datasets reconcile with 0 orphaned sidecars. |
| **Gate 8** | RAG Corpus Isolation | **PASS** | Student-facing DOCX corpus cleanly isolated in `documents/` without double-indexing risk. |
| **Gate 9** | Regression Safety | **PASS** | Historical V1 release, other institutional packages, and RAG_Chatbot left untouched. |
| **Gate 10** | Consumer Compatibility | **PASS** | Verified compatibility with `DirectoryLoader(..., glob='**/*.docx')`. |
