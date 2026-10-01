# AI Platform Architecture

A reusable institutional AI/RAG platform designed to support multiple
institutions while keeping the core retrieval and answering intelligence
institution-agnostic.

---

## Architecture Overview

```text
┌──────────────────────────────────────────────────────────────┐
│                        AI PLATFORM                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CORE                                                        │
│  Reusable AI / RAG intelligence                              │
│                                                              │
│  Query Understanding → Retrieval → Verification → Evidence   │
│  → Grounded Answer Generation → Answer Guard                │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INSTITUTIONS                                                │
│  Institution-specific knowledge, policies and adapters       │
│                                                              │
│  IIT Jodhpur / Future Institutions                           │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  RUNTIME                                                     │
│  Replaceable infrastructure                                  │
│                                                              │
│  LLM / Embeddings / Vector Store / Configuration             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
