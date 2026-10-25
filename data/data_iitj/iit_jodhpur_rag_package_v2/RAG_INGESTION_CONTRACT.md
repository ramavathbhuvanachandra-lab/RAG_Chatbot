# IIT Jodhpur RAG Consumer Ingestion Contract (V2 Release)

## Consumer Identification and Verified Contract
- **Consumer System**: React RAG Chatbot (`/Users/bhuvanachandra/Desktop/RAG_Chatbot`)
- **Loader Implementation**: `DirectoryLoader(..., glob="**/*.docx", loader_cls=Docx2txtLoader)`
- **Target Ingestion Directory**: `storage_institutions/iit_jodhpur/releases/iit_jodhpur_rag_package_v2/documents/`
- **Total Ingestion Corpus**: 162 native Word `.docx` documents.

## Critical Ingestion Boundaries
1. **Isolated Discovery Tree**: The consumer MUST point specifically to the `documents/` directory. It MUST NOT ingest from `knowledge/` or the package root.
2. **Double-Indexing Protection**: `knowledge/` contains `.md` files; `documents/` contains `.docx` files. By pointing strictly to `documents/`, the chatbot indexes exactly 1 copy per subject.
3. **Exclusion of Administrative Files**: All manifests, provenance ledgers, and audit reports are placed in `manifests/`, `provenance/`, and `reports/` outside the `documents/` folder.
4. **Sidecar Preservation**: 152+ granular metadata sidecars exist in `metadata/<doc_id>.json`. The current `Docx2txtLoader` ignores these sidecars safely. If metadata filtering is introduced upstream, the sidecars provide stable document IDs, hashes, and topics.
