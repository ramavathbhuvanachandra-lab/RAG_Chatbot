# Retrieval V4 — side-by-side proposal

> [!NOTE]
> **Historical Proposal Bundle**: This bundle is a side-by-side architectural proposal, not wired into production.
> For the active canonical engine implementation, see `ai_platform/core/`.
> For current engineering status, see [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

This bundle is intentionally **not wired into production**.

It provides a reusable, institution-agnostic design for:

1. `query_features.py` — generic target/attribute/relation feature extraction.
2. `verification.py` — candidate participation boundary; no completeness proof.
3. `ranking.py` — query-aware deterministic feature scoring.
4. `context_compression.py` — extractive compression with relevance filtering,
   exact/near-duplicate removal, MMR-style diversity, local continuity, source
   caps, and character budget.
5. `final_context_builder.py` — wraps the compressed context into the existing
   `EvidencePackage` transport contract.

## Intended path

`query understanding -> hybrid retrieval -> RRF -> V4 verification -> V4 ranking -> V4 compression -> answer LLM`

For multi-intent queries, the same retrieval/rank/compress pipeline should be
run **per intent**, then the intent contexts should be merged with explicit
intent labels before generation. Conversational follow-ups should first be
resolved into a standalone question and only then enter this pipeline.

## Promotion rule

Do not replace the current production modules until V4 beats the existing
baseline on the same 15+ real IITJ cases and adversarial tests.
