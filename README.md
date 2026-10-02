# College AI Platform

> A reusable, institution-agnostic RAG platform for building knowledge-grounded AI assistants for colleges and educational institutions.

## What is this?

College AI Platform is a reusable AI/RAG system designed to power institution-specific AI assistants.

Instead of rebuilding a complete RAG system for every college, the platform separates reusable AI/RAG intelligence from institution-specific knowledge, policies, configuration, and adapters.

The current implementation is onboarded for **IIT Jodhpur** and serves as the first institutional implementation of the platform.

## How it works

```text
                    COLLEGE AI PLATFORM
                            │
             ┌──────────────┼──────────────┐
             │              │              │
            CORE       INSTITUTIONS      RUNTIME
             │              │              │
     Reusable RAG      College-specific   LLM
     intelligence      knowledge          Embeddings
             │          policies           Vector Store
             │          adapters           Config
             │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                  Institution AI Assistant
