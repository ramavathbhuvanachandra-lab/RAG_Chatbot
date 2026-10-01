"""
Generic tests for candidate verification.

These tests verify the reusable core behavior only.

Core principle:

    retrieval may find related evidence,
    but exact requests must only accept candidates
    that actually satisfy the requested target/attribute.
"""

from types import SimpleNamespace

from backend.core.candidate_verification import verify_candidate
from backend.core.query_frame import (
    QueryRequirement,
    SemanticQueryFrame,
)
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
    RetrievalProvenance,
)


def make_candidate(
    document_id: str,
    text: str,
    alignment: CandidateAlignment,
) -> RetrievalCandidate:
    """
    Build a deterministic RetrievalCandidate using the
    current retrieval contracts.
    """

    document = SimpleNamespace(
        page_content=text,
        metadata={
            "source": document_id,
        },
    )

    meaning = DocumentMeaning(
        programs=(),
        entities=(),
        topics=(),
        attributes=(),
        intent="cost",
        scope=(),
        qualifiers=(),
        constraints=(),
    )

    provenance = RetrievalProvenance(
        dense_rank=1,
        dense_score=1.0,
        bm25_rank=1,
        bm25_score=1.0,
        rrf_score=1.0,
        alternate_score_contribution=0.0,
        primary_query="What is the price of item A?",
        retrieval_queries=(
            "What is the price of item A?",
        ),
        signals=(),
    )

    quality = EvidenceQuality(
        content_quality=1.0,
        structural_quality=1.0,
        noise=0.0,
    )

    return RetrievalCandidate(
        document=document,
        document_id=document_id,
        source=document_id,
        provenance=provenance,
        meaning=meaning,
        alignment=alignment,
        quality=quality,
        final_score=1.0,
    )


def make_exact_cost_frame(
    target: str,
) -> SemanticQueryFrame:
    """
    Build an exact information request.

    Example:
        What is the price of Item A?
    """

    return SemanticQueryFrame(
        original_query=f"What is the price of {target}?",
        normalized_query=(
            f"what is the price of {target.lower()}"
        ),
        semantic_query=f"price of {target}",
        target=target,
        request_type="cost",
        facets=(),
        qualifiers=(),
        conditions=(),
        relations=(),
        temporal_context=None,
        comparison_targets=(),
        preserved_terms=(target,),
        is_list_question=False,
        is_comparison_question=False,
        is_multi_part=False,
        requirement=QueryRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            require_scope_alignment=False,
            reject_explicit_conflict=True,
            allow_partial_evidence=False,
            max_unmatched_required_facets=0,
        ),
        confidence=0.96,
        needs_clarification=False,
        clarification_reason="",
        language="en",
        interpretation_status="trusted",
    )


def test_exact_target_is_accepted():
    """
    Correct target + requested attribute should pass.
    """

    frame = make_exact_cost_frame("Item A")

    alignment = CandidateAlignment(
        program_match=0.0,
        entity_match=0.0,
        topic_match=0.0,
        attribute_match=1.0,
        intent_match=1.0,
        scope_match=1.0,
        coverage=1.0,
        semantic_match=1.0,
        conflicts=(),
    )

    candidate = make_candidate(
        "item_a",
        "Item A costs ₹500.",
        alignment,
    )

    decision = verify_candidate(
        frame,
        candidate,
    )

    assert decision.accepted is True
    assert decision.status == "verified"
    assert decision.target_grounded is True
    assert decision.attribute_grounded is True
    assert decision.conflict_detected is False


def test_related_target_is_rejected():
    """
    Classic A-vs-B failure case.

    User asks:
        What is the price of Item A?

    Retrieved candidate:
        Item B costs ₹700.

    The candidate is related to price, but it is not
    evidence for the requested target.
    """

    frame = make_exact_cost_frame("Item A")

    alignment = CandidateAlignment(
        program_match=0.0,
        entity_match=0.0,
        topic_match=1.0,
        attribute_match=1.0,
        intent_match=1.0,
        scope_match=1.0,
        coverage=0.5,
        semantic_match=1.0,
        conflicts=(),
    )

    candidate = make_candidate(
        "item_b",
        "Item B costs ₹700.",
        alignment,
    )

    decision = verify_candidate(
        frame,
        candidate,
    )

    assert decision.accepted is False
    assert decision.status == "rejected"
    assert decision.target_grounded is False


def test_exact_target_phrase_does_not_require_semantic_registry():
    """
    Arbitrary targets should work even when they do not appear
    in the institution's semantic registry.

    Example:
        Model ZX-200
    """

    frame = make_exact_cost_frame(
        "Model ZX-200"
    )

    alignment = CandidateAlignment(
        program_match=0.0,
        entity_match=0.0,
        topic_match=0.0,
        attribute_match=1.0,
        intent_match=1.0,
        scope_match=1.0,
        coverage=1.0,
        semantic_match=0.0,
        conflicts=(),
    )

    candidate = make_candidate(
        "model_zx_200",
        "The price of Model ZX-200 is ₹42,000.",
        alignment,
    )

    decision = verify_candidate(
        frame,
        candidate,
    )

    assert decision.accepted is True
    assert decision.status == "verified"
    assert decision.target_grounded is True


def test_explicit_conflict_is_rejected():
    """
    Explicit conflicts must reject an exact request.
    """

    frame = make_exact_cost_frame(
        "Item A"
    )

    alignment = CandidateAlignment(
        program_match=0.0,
        entity_match=0.0,
        topic_match=1.0,
        attribute_match=1.0,
        intent_match=1.0,
        scope_match=1.0,
        coverage=1.0,
        semantic_match=1.0,
        conflicts=(
            "candidate refers to a different target",
        ),
    )

    candidate = make_candidate(
        "conflicting_document",
        "Item A costs ₹500, but this section applies to Item B.",
        alignment,
    )

    decision = verify_candidate(
        frame,
        candidate,
    )

    assert decision.accepted is False
    assert decision.status == "rejected"
    assert decision.conflict_detected is True


def test_missing_attribute_is_rejected_for_exact_request():
    """
    Correct target but missing requested attribute.

    User asks:
        price of Item A

    Candidate only says:
        Item A is available in three variants.
    """

    frame = make_exact_cost_frame(
        "Item A"
    )

    alignment = CandidateAlignment(
        program_match=0.0,
        entity_match=0.0,
        topic_match=1.0,
        attribute_match=0.0,
        intent_match=0.0,
        scope_match=1.0,
        coverage=0.5,
        semantic_match=0.5,
        conflicts=(),
    )

    candidate = make_candidate(
        "item_a_description",
        "Item A is available in three variants.",
        alignment,
    )

    decision = verify_candidate(
        frame,
        candidate,
    )

    assert decision.accepted is False
    assert decision.status == "rejected"
    assert decision.target_grounded is True
    assert decision.attribute_grounded is False


if __name__ == "__main__":
    test_exact_target_is_accepted()
    test_related_target_is_rejected()
    test_exact_target_phrase_does_not_require_semantic_registry()
    test_explicit_conflict_is_rejected()
    test_missing_attribute_is_rejected_for_exact_request()

    print(
        "CANDIDATE VERIFICATION CORE TESTS: PASS"
    )