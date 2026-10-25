"""Production Generalization and Core Reliability Tests.

Verifies:
1. Multi-intent institutional finance decomposition (fellowship + refundable deposit, hostel + caution deposit).
2. Metamorphic consistency: semantically equivalent questions retrieve equivalent evidence units.
3. Negative controls: out-of-scope and non-existent entities safely abstain / reject.
4. Request-scoped institution isolation: parameterized retrieval caches return appropriate data without collision.
5. Fail-closed behavior on empty or whitespace queries.
"""

from __future__ import annotations

import pytest

from ai_platform.runtime.engine import inspect_context
from ai_platform.runtime.retrieval import get_canonical_chunks, get_bm25_retriever, get_dense_retriever
from ai_platform.runtime.vectorstore import get_vectorstore


def test_multi_intent_institutional_finance_decomposition():
    """Verify that multi-intent finance queries decompose and package evidence for both intents."""
    query = (
        "What is the minimum CGPA required for Ph.D. fellowship continuation, "
        "and what is the one-time refundable institute deposit?"
    )
    result = inspect_context(query)

    assert result["is_multi_intent"] is True, "Query should be recognized as multi-intent"
    assert len(result["intent_questions"]) == 2, f"Expected 2 intents, got {result['intent_questions']}"

    headings = [u.get("heading", "") for u in result.get("evidence_units", [])]
    texts = " ".join(u.get("text", "") for u in result.get("evidence_units", []))

    # Verify fellowship continuation evidence is present
    has_fellowship = (
        "fellowship" in texts.casefold()
        or "assistantship" in texts.casefold()
        or any("Progression" in h or "Academics" in h for h in headings)
    )
    assert has_fellowship, "Expected fellowship / academic progression evidence in final package"

    # Verify refundable deposit evidence is present
    has_deposit = (
        "refundable deposit" in texts.casefold()
        or "₹8,000" in texts
        or "8000" in texts
        or any("Finance" in h for h in headings)
    )
    assert has_deposit, "Expected refundable deposit / finance evidence in final package"


def test_multi_intent_hostel_and_caution_deposit():
    """Verify hostel inquiry combined with caution money decomposes into distinct intents."""
    query = "What are the hostel room allotment rules, and what is the caution deposit?"
    result = inspect_context(query)

    assert result["is_multi_intent"] is True, "Hostel + caution deposit should split into multi-intent"
    assert len(result["intent_questions"]) == 2


def test_metamorphic_paraphrase_equivalence():
    """Verify semantically equivalent phrasings retrieve equivalent core evidence."""
    q1 = "What is the CGPA required for continuing Ph.D. fellowship?"
    q2 = "What minimum grade point average is needed to retain the doctoral assistantship?"

    res1 = inspect_context(q1)
    res2 = inspect_context(q2)

    assert res1["ready_for_generation"] is True
    assert res2["ready_for_generation"] is True

    # Both should retrieve academic progression / financial assistance documents
    headings1 = set(u.get("heading", "") for u in res1.get("evidence_units", []))
    headings2 = set(u.get("heading", "") for u in res2.get("evidence_units", []))

    common_headings = headings1.intersection(headings2)
    assert len(common_headings) > 0, (
        f"Paraphrased queries should share supporting headings: {headings1} vs {headings2}"
    )


def test_negative_control_abstention():
    """Verify that unanswerable, fabricated, or out-of-corpus queries abstain safely."""
    query = "What is the fee for the commercial pilot flight training program at IITJ?"
    result = inspect_context(query)

    assert result["will_abstain"] is True or result["evidence_status"] == "insufficient", (
        "Nonexistent program query must abstain or report insufficient evidence"
    )


def test_retrieval_institution_parameterization():
    """Verify get_canonical_chunks and retrievers accept institution_id and cache consistently."""
    chunks_default = get_canonical_chunks()
    chunks_explicit = get_canonical_chunks("iitj")

    assert len(chunks_default) == len(chunks_explicit)
    assert chunks_default[0].page_content == chunks_explicit[0].page_content

    # Vector store cache
    vs1 = get_vectorstore("iitj")
    vs2 = get_vectorstore()
    assert vs1 is vs2, "Cached vectorstore instance should be identical for same institution"

    # BM25 cache
    bm25_1 = get_bm25_retriever("iitj")
    bm25_2 = get_bm25_retriever()
    assert bm25_1 is bm25_2, "Cached BM25 instance should be identical for same institution"


def test_fail_closed_on_empty_and_whitespace():
    """Verify inspect_context fails closed on invalid inputs."""
    with pytest.raises(ValueError, match="question cannot be empty"):
        inspect_context("")

    with pytest.raises(ValueError, match="question cannot be empty"):
        inspect_context("   \n\t  ")
