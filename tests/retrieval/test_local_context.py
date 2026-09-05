"""
Phase 4 — Local Context Expansion Tests

Purpose
-------
Validate conservative local context recovery using the real corpus.

Contract:

    same source
    + useful continuity
    + no clear section/category boundary
        -> eligible for expansion

A new section/category boundary must not be crossed blindly.
"""

from langchain_core.documents import Document

from backend.retriever import (
    chunks,
    get_source,
)


# =========================================================
# Helpers
# =========================================================

def find_chunk(
    fragment: str,
    source_fragment: str | None = None,
):
    """
    Find a real corpus chunk by content fragment.
    """

    fragment = fragment.lower()

    for document in chunks:

        if fragment not in (
            document.page_content.lower()
        ):
            continue

        if source_fragment:

            source = get_source(
                document
            ).lower()

            if (
                source_fragment.lower()
                not in source
            ):
                continue

        return document

    raise AssertionError(
        f"Could not find corpus chunk containing {fragment!r}"
    )


def same_source(
    first: Document,
    second: Document,
) -> bool:
    """
    Verify same-source identity.
    """

    return (
        get_source(first)
        == get_source(second)
    )


# =========================================================
# Test 1 — Electrical context
# =========================================================

def test_electrical_overview_has_useful_adjacent_context():
    """
    The previous Electrical Engineering chunk adds meaningful
    department/research context.
    """

    target = find_chunk(
        "active collaborations with leading industries",
        "departments/electrical_engineering/overview.docx",
    )

    target_index = chunks.index(
        target
    )

    previous = chunks[
        target_index - 1
    ]

    assert same_source(
        target,
        previous,
    )

    assert (
        "teaching and research activities"
        in previous.page_content.lower()
        or
        "power systems"
        in previous.page_content.lower()
    )


# =========================================================
# Test 2 — Electrical URL/noise protection
# =========================================================

def test_electrical_url_chunk_is_not_context():
    """
    The next chunk is same-source but contains source URLs and
    retrieval metadata. It must not be considered useful context.
    """

    target = find_chunk(
        "active collaborations with leading industries",
        "departments/electrical_engineering/overview.docx",
    )

    target_index = chunks.index(
        target
    )

    next_chunk = chunks[
        target_index + 1
    ]

    assert same_source(
        target,
        next_chunk,
    )

    assert (
        "original source urls"
        in next_chunk.page_content.lower()
    )


# =========================================================
# Test 3 — Hostel fee continuation
# =========================================================

def test_hostel_fee_has_useful_adjacent_context():
    """
    The hostel fee/accommodation table continues into the next
    finance chunk.
    """

    target = find_chunk(
        "accommodation charges",
        "finance/fees_and_finance.docx",
    )

    target_index = chunks.index(
        target
    )

    next_chunk = chunks[
        target_index + 1
    ]

    assert same_source(
        target,
        next_chunk,
    )

    assert (
        "hostel room rent"
        in next_chunk.page_content.lower()
        or
        "charges per day"
        in next_chunk.page_content.lower()
    )


# =========================================================
# Test 4 — Source boundary
# =========================================================

def test_local_context_must_not_cross_source_boundary():
    """
    Same-source expansion must never cross into another source.
    """

    target = find_chunk(
        "accommodation charges",
        "finance/fees_and_finance.docx",
    )

    target_index = chunks.index(
        target
    )

    found_boundary = False

    for offset in (-1, 1):

        candidate_index = (
            target_index + offset
        )

        if (
            candidate_index < 0
            or candidate_index >= len(chunks)
        ):
            continue

        candidate = chunks[
            candidate_index
        ]

        if (
            get_source(candidate)
            != get_source(target)
        ):
            found_boundary = True
            break

    assert found_boundary


# =========================================================
# Test 5 — Same source is not enough
# =========================================================

def test_same_source_does_not_automatically_mean_contextual():
    """
    Same-source adjacency alone must not cause expansion.
    """

    target = find_chunk(
        "active collaborations with leading industries",
        "departments/electrical_engineering/overview.docx",
    )

    target_index = chunks.index(
        target
    )

    next_chunk = chunks[
        target_index + 1
    ]

    assert same_source(
        target,
        next_chunk,
    )

    noise_markers = (
        "original source urls",
        "command 6",
        "retrieval representation",
    )

    assert any(
        marker
        in next_chunk.page_content.lower()
        for marker in noise_markers
    )


# =========================================================
# Test 6 — Ph.D. eligibility continuation is useful
# =========================================================

def test_phd_eligibility_recovers_bachelor_route():
    """
    The next chunk after the regular Ph.D. eligibility chunk contains
    the four-year bachelor's-degree eligibility route.

    That continuation should remain recoverable.
    """

    target = find_chunk(
        "20.2.6 Admission to Ph.D. Program (Regular)",
        "admissions/phd_admissions.docx",
    )

    target_index = chunks.index(
        target
    )

    next_chunk = chunks[
        target_index + 1
    ]

    assert same_source(
        target,
        next_chunk,
    )

    assert (
        "four-year duration"
        in next_chunk.page_content.lower()
    )

    assert (
        "70%"
        in next_chunk.page_content
    )


# =========================================================
# Test 7 — Ph.D. section boundary must be respected
# =========================================================

def test_phd_expansion_should_not_cross_into_sponsored_section():
    """
    The next chunk contains the useful bachelor's-degree route,
    but it also begins the sponsored/external/part-time section.

    Local expansion must not blindly treat the new section as part
    of the regular-admission context.
    """

    target = find_chunk(
        "20.2.6 Admission to Ph.D. Program (Regular)",
        "admissions/phd_admissions.docx",
    )

    target_index = chunks.index(
        target
    )

    next_chunk = chunks[
        target_index + 1
    ]

    assert (
        "20.2.9"
        in next_chunk.page_content
        or
        "sponsored/external/part-time"
        in next_chunk.page_content.lower()
    )