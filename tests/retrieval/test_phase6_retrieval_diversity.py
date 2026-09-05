"""
Phase 6 — Retrieval Diversity / Duplicate Refinement

These tests deliberately use difficult synthetic retrieval candidates.

The goal is to prove that Phase 6:

    - preserves strong relevance
    - reduces redundant evidence
    - improves source diversity when relevance is comparable
    - keeps legitimate same-source information
    - does not confuse similar facts with duplicates
    - remains deterministic
"""

from langchain_core.documents import Document

from backend.retrieval_diversity import (
    CONTENT_REDUNDANCY_THRESHOLD,
    select_diverse_documents,
)


def candidate(
    text: str,
    source: str,
    score: float,
    original_rank: int,
):
    return {
        "document": Document(
            page_content=text,
            metadata={
                "source": source,
            },
        ),
        "score": score,
        "original_rank": original_rank,
        "source": source,
    }


def texts(documents):
    return [
        document.page_content
        for document in documents
    ]


# =========================================================
# Strong relevance must remain dominant
# =========================================================

def test_strongly_better_document_is_not_displaced():

    scored = [
        candidate(
            "Hostel fee is ₹300 per day.",
            "fees_a.json",
            100.0,
            1,
        ),
        candidate(
            "Hostel rules include quiet hours.",
            "rules_b.json",
            70.0,
            2,
        ),
    ]

    result = select_diverse_documents(
        scored,
        top_k=1,
    )

    assert texts(
        result
    ) == [
        "Hostel fee is ₹300 per day."
    ]


# =========================================================
# Same-source saturation
# =========================================================

def test_competitive_other_source_breaks_source_saturation():

    scored = [
        candidate(
            "Hostel fee is ₹300 per day. "
            "Double occupancy is available.",
            "hostel_a.json",
            100.0,
            1,
        ),
        candidate(
            "Hostel fee is ₹300 per day. "
            "Double occupancy is available.",
            "hostel_a.json",
            99.0,
            2,
        ),
        candidate(
            "Double occupancy rooms cost ₹300 per day "
            "and include shared facilities.",
            "hostel_b.json",
            98.0,
            3,
        ),
        candidate(
            "Hostel rules include quiet hours.",
            "hostel_c.json",
            96.0,
            4,
        ),
    ]

    result = select_diverse_documents(
        scored,
        top_k=4,
        max_documents_per_source=2,
    )

    selected_sources = [
        document.metadata["source"]
        for document in result
    ]

    assert selected_sources.count(
        "hostel_a.json"
    ) <= 2

    assert (
        "hostel_b.json"
        in selected_sources
    )

    assert (
        "hostel_c.json"
        in selected_sources
    )


# =========================================================
# Legitimate same-source evidence
# =========================================================

def test_distinct_same_source_information_is_preserved_when_needed():

    scored = [
        candidate(
            "Hostel single occupancy costs ₹500 per day.",
            "hostel.json",
            100.0,
            1,
        ),
        candidate(
            "Hostel double occupancy costs ₹300 per day.",
            "hostel.json",
            99.0,
            2,
        ),
        candidate(
            "Hostel residents have access to Wi-Fi.",
            "hostel.json",
            98.0,
            3,
        ),
    ]

    result = select_diverse_documents(
        scored,
        top_k=3,
        max_documents_per_source=2,
    )

    assert texts(
        result
    ) == [
        "Hostel single occupancy costs ₹500 per day.",
        "Hostel double occupancy costs ₹300 per day.",
        "Hostel residents have access to Wi-Fi.",
    ]


# =========================================================
# Similar content but materially different facts
# =========================================================

def test_similar_fee_chunks_are_not_treated_as_duplicates():

    scored = [
        candidate(
            "Single occupancy hostel room costs ₹500 per day.",
            "hostel.json",
            100.0,
            1,
        ),
        candidate(
            "Double occupancy hostel room costs ₹300 per day.",
            "hostel.json",
            99.0,
            2,
        ),
    ]

    result = select_diverse_documents(
        scored,
        top_k=2,
        max_documents_per_source=2,
    )

    result_texts = texts(
        result
    )

    assert (
        "Single occupancy hostel room costs ₹500 per day."
        in result_texts
    )

    assert (
        "Double occupancy hostel room costs ₹300 per day."
        in result_texts
    )


# =========================================================
# Redundant content should not consume the whole budget
# =========================================================

def test_near_identical_evidence_yields_to_distinct_competitive_evidence():

    repeated_text_a = (
        "Hostel fees include accommodation charges "
        "for student residents."
    )

    repeated_text_b = (
        "Hostel fees include accommodation charges "
        "for student residents and hostel rooms."
    )

    scored = [
        candidate(
            repeated_text_a,
            "hostel_a.json",
            100.0,
            1,
        ),
        candidate(
            repeated_text_b,
            "hostel_a.json",
            99.0,
            2,
        ),
        candidate(
            "Hostel residents can use dining and laundry facilities.",
            "hostel_b.json",
            98.0,
            3,
        ),
    ]

    result = select_diverse_documents(
        scored,
        top_k=2,
        max_documents_per_source=2,
    )

    result_texts = texts(
        result
    )

    assert (
        repeated_text_a
        in result_texts
    )

    assert (
        "Hostel residents can use dining and laundry facilities."
        in result_texts
    )


# =========================================================
# No competitive alternative means preserve evidence
# =========================================================

def test_same_source_documents_are_not_artificially_removed():

    scored = [
        candidate(
            "Academic regulations cover attendance requirements.",
            "rules.json",
            100.0,
            1,
        ),
        candidate(
            "Academic regulations cover examination rules.",
            "rules.json",
            97.0,
            2,
        ),
        candidate(
            "Academic regulations cover grading policy.",
            "rules.json",
            94.0,
            3,
        ),
    ]

    result = select_diverse_documents(
        scored,
        top_k=3,
        max_documents_per_source=2,
    )

    assert len(
        result
    ) == 3


# =========================================================
# Determinism
# =========================================================

def test_selection_is_deterministic():

    scored = [
        candidate(
            "Hostel fee is ₹300 per day.",
            "a.json",
            100.0,
            1,
        ),
        candidate(
            "Hostel rooms include Wi-Fi.",
            "b.json",
            99.0,
            2,
        ),
        candidate(
            "Hostel residents follow quiet-hour rules.",
            "c.json",
            98.0,
            3,
        ),
        candidate(
            "Hostel fee is ₹300 per day.",
            "a.json",
            97.0,
            4,
        ),
    ]

    first = texts(
        select_diverse_documents(
            scored,
            top_k=3,
        )
    )

    second = texts(
        select_diverse_documents(
            scored,
            top_k=3,
        )
    )

    assert first == second


# =========================================================
# Empty input
# =========================================================

def test_empty_candidates_return_empty():

    assert (
        select_diverse_documents(
            [],
            top_k=5,
        )
        == []
    )


# =========================================================
# Invalid budget
# =========================================================

def test_non_positive_budget_returns_empty():

    scored = [
        candidate(
            "Hostel fees are ₹300 per day.",
            "hostel.json",
            100.0,
            1,
        )
    ]

    assert (
        select_diverse_documents(
            scored,
            top_k=0,
        )
        == []
    )


# =========================================================
# Existing redundancy threshold remains respected
# =========================================================

def test_redundancy_threshold_is_below_existing_near_duplicate_threshold():

    from backend.retriever import (
        NEAR_DUPLICATE_THRESHOLD,
    )

    assert (
        CONTENT_REDUNDANCY_THRESHOLD
        < NEAR_DUPLICATE_THRESHOLD
    )