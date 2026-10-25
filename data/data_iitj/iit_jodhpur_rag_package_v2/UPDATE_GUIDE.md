# Update Guide — IIT Jodhpur RAG Package V2

To update the RAG chatbot to V2:
1. Re-point `ingest_documents` doc_dir to `storage_institutions/iit_jodhpur/releases/iit_jodhpur_rag_package_v2/documents`.
2. Build index in a staging collection.
3. Verify probe answers for visitor accommodation, ITEP fees, and math minor.
4. Swap staging index to production.
