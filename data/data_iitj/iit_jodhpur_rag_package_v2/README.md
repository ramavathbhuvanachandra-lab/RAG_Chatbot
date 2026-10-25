# Indian Institute of Technology Jodhpur — RAG Knowledge Package (V2 Release)

**Institution**: Indian Institute of Technology Jodhpur (`iitj`)  
**Package Version**: 2.0.0  
**Release Location**: `storage_institutions/iit_jodhpur/releases/iit_jodhpur_rag_package_v2`  
**Run Location**: `storage_institutions/iit_jodhpur/runs/run_iitj_v2_repair_20261024`  
**Historical Baseline**: `storage_institutions/iit_jodhpur/releases/iit_jodhpur_rag_package_v1` (Preserved Intact)

---

## Package Overview

This package represents the corrected, factually reconciled, and retrieval-optimized V2 knowledge release for IIT Jodhpur.

### Core Architectural Deliverables
1. **Canonical Markdown Documents (`knowledge/`)**: 162 subject-specific documents with single H1, structured scope, and provenance blocks.
2. **Faithful Word Documents (`documents/`)**: 162 companion DOCX documents styled with native Word `Table Grid` and OpenXML hyperlinks.
3. **Metadata Sidecars (`metadata/`)**: 162 JSON sidecars implementing Schema 2.0 with cryptographic SHA-256 hashes.
4. **Structured Institutional Datasets (`data/`)**: 26 rich institutional JSON datasets covering departments, schools, programmes, fees, visitor accommodation, and emergency contacts.
5. **Source Registry & Lineage (`source_registry/`)**: Complete discovered URL inventory, disposition manifest, and source-to-document lineage mappings.
6. **Provenance & Evidence (`provenance/`)**: Claim-level evidence ledger, source coverage audit, and formal conflict register (7 conflicts logged).
7. **Consumer Contracts (`contracts/`)**: Ingestion contract and migration guide for LangChain `DirectoryLoader(..., glob='**/*.docx')`.
