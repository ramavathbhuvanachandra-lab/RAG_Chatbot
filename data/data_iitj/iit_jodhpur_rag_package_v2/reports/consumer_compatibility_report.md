# IIT Jodhpur Consumer Compatibility Report (V2 Release)

- **Target Consumer**: React RAG Chatbot
- **Loader Compatibility**: `DirectoryLoader(..., glob='**/*.docx', loader_cls=Docx2txtLoader)`
- **Total Consumable DOCX Files**: 162
- **Zero Double-Indexing Guarantee**: Validated by clean directory isolation between `documents/` and `knowledge/`.
