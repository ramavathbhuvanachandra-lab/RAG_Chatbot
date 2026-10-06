"""
Stage 2 — verification contract tests.

These tests do not tune the verifier. They protect the verification boundary
while we inspect the real 54 -> 16 behavior.
"""

from __future__ import annotations

from typing import Any

from ai_platform.core.graph.nodes import CoreNodes, default_dependencies


QUERY = "What are the M.Tech eligibility requirements?"
GOLD_SUFFIX = "/admissions/mtech_admissions.docx"


def _source(candidate: Any) -> str:
    source = str(getattr(candidate, "source", "") or "")
    if source:
        return source
    document = getattr(candidate, "document", candidate)
    metadata = getattr(document, "metadata", {}) or {}
    return str(metadata.get("source", "") or "")


def test_real_iitj_verification_boundary() -> None:
    nodes = CoreNodes(default_dependencies())
    state = {
        "question": QUERY,
        "resolved_question": QUERY,
        "chat_history": [],
    }

    state.update(nodes.understand_query_node(state))
    state.update(nodes.hybrid_retrieve_node(state))
    state.update(nodes.fuse_retrieved_documents_node(state))

    fused = tuple(state.get("fused_candidates", ()))
    assert fused, "Real IITJ retrieval produced zero fused candidates."

    result = nodes.verify_and_rank_node(state)

    verified = tuple(result.get("verified_candidates", ()))
    uncertain = tuple(result.get("uncertain_candidates", ()))
    rejected = tuple(result.get("rejected_candidates", ()))

    assert verified or uncertain or rejected
    assert len(result.get("verification_trace", ())) >= len(fused)

    all_bucket_ids = {
        str(getattr(item, "document_id", "") or _source(item))
        for item in (*verified, *uncertain, *rejected)
    }
    fused_ids = {
        str(getattr(item, "document_id", "") or _source(item))
        for item in fused
    }

    assert fused_ids <= all_bucket_ids


def test_mtech_admissions_source_survives_verification() -> None:
    nodes = CoreNodes(default_dependencies())
    state = {
        "question": QUERY,
        "resolved_question": QUERY,
        "chat_history": [],
    }

    state.update(nodes.understand_query_node(state))
    state.update(nodes.hybrid_retrieve_node(state))
    state.update(nodes.fuse_retrieved_documents_node(state))

    fused = tuple(state.get("fused_candidates", ()))
    result = nodes.verify_and_rank_node(state)
    verified = tuple(result.get("verified_candidates", ()))

    gold_fused = [
        item for item in fused
        if _source(item).casefold().endswith(GOLD_SUFFIX)
    ]
    gold_verified = [
        item for item in verified
        if _source(item).casefold().endswith(GOLD_SUFFIX)
    ]

    assert gold_fused, "M.Tech admissions source did not reach verification."
    assert gold_verified, (
        "M.Tech admissions source reached verification but did not survive "
        "verification."
    )
