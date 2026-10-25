# IIT Jodhpur Knowledge Package V2 Consumer Migration Plan

## 1. Zero Production Disruption Principle
The React chatbot remains strictly untouched. Migration is performed via separate staging index creation.

## 2. Step-by-Step Execution Sequence

### Step 1: Staging Index Creation (Dry Run)
Execute document ingestion pointing exclusively to the V2 `documents/` directory:
```bash
python -c "
from backend.ingestion import ingest_documents
ingest_documents(
    doc_dir='/Users/bhuvanachandra/Documents/knowledge-ingestion-engine/storage_institutions/iit_jodhpur/releases/iit_jodhpur_rag_package_v2/documents',
    persist_dir='/Users/bhuvanachandra/Desktop/RAG_Chatbot/data/chroma_iitj_v2_staging'
)
"
```

### Step 2: Verification of Repaired Canonical Subjects
Run canonical retrieval probes against the staging index:
- Probe 1: "What are the visitor hostel room charges?" -> Expects `visitor_hostel_accommodation_and_charges.docx`
- Probe 2: "What is the ITEP tuition fee?" -> Expects `academic_fees_and_scholarships.docx` (₹50,000)
- Probe 3: "What is the B.Tech tuition fee?" -> Expects `academic_fees_and_scholarships.docx` (₹1,00,000)
- Probe 4: "What is the curriculum for Minor in Mathematics?" -> Expects `minor_mathematics_and_computing.docx` (19 credits)
- Probe 5: "What is the emergency hospital number?" -> Expects `primary_health_centre.docx` (0291-280-1190)

### Step 3: Atomic Cutover
Symlink or swap the active Chroma vector collection pointer.

### Step 4: Instant Reversibility
If rollback is desired, revert the pointer to the previous index backup:
```bash
ln -sfn /Users/bhuvanachandra/Desktop/RAG_Chatbot/data/chroma_db_backup /Users/bhuvanachandra/Desktop/RAG_Chatbot/data/chroma_db
```
