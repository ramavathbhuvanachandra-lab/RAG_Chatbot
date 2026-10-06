"""
Retrieval V2 — Evidence Filtering Tests

These tests validate the core evidence-filtering invariant:

    Evidence must support the requested target and facet together.

In particular:

    M.Tech eligibility != Ph.D. eligibility

A document containing information about multiple programs must not cause
evidence for one program to be incorrectly accepted for another program.
"""

from ai_platform.core.retrieval_v2 import (
    QuerySpec,
    RetrievalIntent,
    RetrievedDocument,
    RankedCandidate,
    build_evidence_units,
    score_evidence_unit,
)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _query() -> QuerySpec:
    """
    User asks specifically about M.Tech admission eligibility.
    """

    return QuerySpec(
        raw_query=(
            "What are the eligibility criteria for admission "
            "to the M.Tech program?"
        ),
        institution_id="iitj",
        intents=(
            RetrievalIntent(
                target="mtech",
                facets=("eligibility",),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Candidate helper
# ---------------------------------------------------------------------------


def _candidate(
    text: str,
    *,
    program: str = "mtech",
    semantic: float = 0.85,
    lexical: float = 0.85,
) -> RankedCandidate:
    """
    Build a synthetic ranked retrieval candidate.
    """

    return RankedCandidate(
        document=RetrievedDocument(
            document_id="test",
            text=text,
            metadata={
                "institution_id": "iitj",
                "program": program,
            },
            dense_score=semantic,
            lexical_score=lexical,
        ),
        semantic_score=semantic,
        lexical_score=lexical,
    )


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def _score(
    query: QuerySpec,
    candidate: RankedCandidate,
):
    """
    Convert the candidate into local evidence units and score each unit.
    """

    units = build_evidence_units(
        candidate,
        max_sentences=3,
        sentence_stride=1,
    )

    return [
        score_evidence_unit(
            query,
            unit,
        )
        for unit in units
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_direct_mtech_eligibility_is_relevant() -> None:
    """
    Direct evidence for M.Tech eligibility must be accepted.
    """

    decisions = _score(
        _query(),
        _candidate(
            "For admission to the M.Tech program, "
            "candidates must hold a relevant bachelor's degree."
        ),
    )

    assert any(
        decision.keep
        for decision in decisions
    )


def test_phd_eligibility_is_not_mtech_evidence() -> None:
    """
    Ph.D. eligibility must not satisfy an M.Tech eligibility question.
    """

    decisions = _score(
        _query(),
        _candidate(
            "For admission to the Ph.D. program, "
            "candidates must satisfy the prescribed eligibility criteria.",
            program="phd",
        ),
    )

    assert not any(
        decision.keep
        for decision in decisions
    )


def test_mtech_document_with_phd_section_does_not_mix_programs() -> None:
    """
    A document containing both M.Tech and Ph.D. information must not cause
    Ph.D. eligibility evidence to be accepted as M.Tech eligibility.

    The two statements are intentionally placed in separate local sections.
    """

    decisions = _score(
        _query(),
        _candidate(
            "The academic catalogue lists the M.Tech program.\n"
            "General information about postgraduate programs is provided "
            "throughout the catalogue.\n"
            "The Ph.D. program has separate eligibility requirements."
            "\n"
            "Additional information about departments and research areas "
            "is provided in the catalogue.",
            program="mtech",
        ),
    )

    assert not any(
        decision.keep
        for decision in decisions
    )