"""
Evidence Group Integration Tests

Validate grouped evidence against the real IIT Jodhpur corpus.

The test intentionally uses real retrieval results as a stress test.
Production logic must remain generic.
"""

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    rerank_documents,
    FINAL_CONTEXT_DOCUMENTS,
    get_source,
)

from backend.local_context import (
    expand_local_context,
)

from backend.evidence_groups import (
    build_evidence_groups,
    attach_local_context,
    rank_evidence_groups,
    flatten_evidence_groups,
)


# =========================================================
# Retrieval helper
# =========================================================

def initial_retrieve(
    question: str,
):
    dense = dense_retrieve(
        question
    )

    bm25 = keyword_retrieve(
        question
    )

    fused = reciprocal_rank_fusion(
        [
            dense,
            bm25,
        ]
    )

    fused = deduplicate_documents(
        fused
    )

    return rerank_documents(
        query=question,
        documents=fused,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )


# =========================================================
# Test 1 — Ph.D. continuation stays with anchor
# =========================================================

def test_phd_regular_eligibility_group_keeps_continuation():

    question = (
        "What are the eligibility requirements "
        "for regular Ph.D. admission?"
    )

    anchors = initial_retrieve(
        question
    )

    expanded = expand_local_context(
        anchors
    )

    groups = build_evidence_groups(
        anchors
    )

    groups = attach_local_context(
        groups,
        expanded,
    )

    ranked = rank_evidence_groups(
        query=question,
        groups=groups,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )

    flattened = flatten_evidence_groups(
        ranked
    )

    combined = " ".join(
        document.page_content.lower()
        for document in flattened
        if "phd"
        in get_source(document).lower()
    )

    assert (
        "four-year"
        in combined
        or "70%"
        in combined
    )


# =========================================================
# Test 2 — Hostel evidence stays grouped
# =========================================================

def test_hostel_fee_group_keeps_accommodation_context():

    question = "What are the hostel fees?"

    anchors = initial_retrieve(
        question
    )

    expanded = expand_local_context(
        anchors
    )

    groups = build_evidence_groups(
        anchors
    )

    groups = attach_local_context(
        groups,
        expanded,
    )

    ranked = rank_evidence_groups(
        query=question,
        groups=groups,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )

    flattened = flatten_evidence_groups(
        ranked
    )

    finance_documents = [
        document
        for document in flattened
        if "finance/fees_and_finance.docx"
        in get_source(document)
    ]

    assert finance_documents

    combined = " ".join(
        document.page_content.lower()
        for document in finance_documents
    )

    assert (
        "hostel room rent"
        in combined
    )


# =========================================================
# Test 3 — EE research group remains department-scoped
# =========================================================

def test_electrical_research_groups_keep_electrical_evidence():

    question = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    anchors = initial_retrieve(
        question
    )

    expanded = expand_local_context(
        anchors
    )

    groups = build_evidence_groups(
        anchors
    )

    groups = attach_local_context(
        groups,
        expanded,
    )

    ranked = rank_evidence_groups(
        query=question,
        groups=groups,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )

    flattened = flatten_evidence_groups(
        ranked
    )

    combined = " ".join(
        document.page_content.lower()
        for document in flattened
    )

    assert (
        "electrical engineering"
        in combined
    )

    assert (
        "research"
        in combined
    )


# =========================================================
# Test 4 — Group count stays bounded
# =========================================================

def test_group_count_remains_bounded():

    question = (
        "What are the eligibility requirements "
        "for regular M.Tech. admission?"
    )

    anchors = initial_retrieve(
        question
    )

    groups = build_evidence_groups(
        anchors
    )

    assert (
        len(groups)
        <= FINAL_CONTEXT_DOCUMENTS
    )


# =========================================================
# Test 5 — Flattening does not duplicate evidence
# =========================================================

def test_group_flattening_removes_duplicates():

    question = (
        "What research areas are available "
        "in Electrical Engineering?"
    )

    anchors = initial_retrieve(
        question
    )

    expanded = expand_local_context(
        anchors
    )

    groups = build_evidence_groups(
        anchors
    )

    groups = attach_local_context(
        groups,
        expanded,
    )

    ranked = rank_evidence_groups(
        query=question,
        groups=groups,
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )

    flattened = flatten_evidence_groups(
        ranked
    )

    keys = [
        (
            get_source(document),
            document.page_content,
        )
        for document in flattened
    ]

    assert len(keys) == len(
        set(keys)
    )