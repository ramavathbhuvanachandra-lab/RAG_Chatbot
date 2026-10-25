# IIT Jodhpur Crawl Reuse Inventory Report (V2 Release)

**Institution**: Indian Institute of Technology Jodhpur (`iitj`)  
**Date**: 2026-10-23 22:37:04 UTC  
**Crawl Status**: **100% REUSED FROM EXISTING LOCAL CRAWL DATA (ZERO RECRAWL)**  

---

## 1. Source Inventory Summary

| Source Category | Location | Total Count | Reused in V2 | Status |
|---|---|---|---|---|
| Crawled Web Pages | `storage/output/www.iitj.ac.in` | 1,096 | 1,096 | Usable (100%) |
| Unique Crawled URLs | `storage/output/www.iitj.ac.in` | 1,093 | 1,093 | Unique |
| Structured Knowledge JSONs | `storage/structured_knowledge/www.iitj.ac.in` | 1,046 | 1,046 | Validated |
| Downloaded PDFs | `storage/documents/www.iitj.ac.in` | 19 | 19 | Co-located in Run |
| Normalized Knowledge Units | `storage/test6_iitj_knowledge_units.json` | 2,641 | 2,641 | Classified |

---

## 2. Invariants Certified

1. **Zero Recrawl**: No network crawler (Crawl4AI or similar) was launched.
2. **Deterministic Lineage**: Every canonical document traces to exact source URLs and hashes.
3. **Evidence Preservation**: Frozen source data remains intact in `storage/` and co-located in `storage_institutions/iit_jodhpur/runs/`.
