"""Stage 2 regression tests for target/attribute relation grounding."""

from __future__ import annotations

from types import SimpleNamespace

from ai_platform.core.retrieval.contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
)
from ai_platform.core.retrieval.verification import (
    verify_candidate,
)
from ai_platform.core.query.models import (
    Entity,
    QueryFacet,
    RetrievalRequirement,
    SemanticQueryFrame,
)


def _frame() -> SemanticQueryFrame:
    return SemanticQueryFrame(
        original_query="What are the M.Tech eligibility requirements?",
        target="M.Tech",
        request_type="eligibility",
        facets=(
            QueryFacet(
                "requested_attribute",
                "eligibility",
            ),
        ),
        entities=(
            Entity(
                name="mtech",
                entity_type="program",
                entity_id="mtech",
                confidence=1.0,
                resolution_state="resolved",
            ),
        ),
        requirement=RetrievalRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            reject_explicit_conflict=True,
        ),
    )


def _candidate(
    text: str,
    *,
    programs=(),
    attributes=("eligibility",),
) -> RetrievalCandidate:
    document = SimpleNamespace(
        page_content=text,
        metadata={
            "source": "data/admissions/test.docx",
        },
    )

    return RetrievalCandidate(
        document=document,
        document_id="test|" + text,
        source="data/admissions/test.docx",
        meaning=DocumentMeaning(
            programs=programs,
            attributes=attributes,
        ),
        alignment=CandidateAlignment(
            conflicts=(),
            semantic_match=0.95,
            coverage=0.95,
            attribute_match=0.95,
            scope_match=1.0,
        ),
        quality=EvidenceQuality(
            content_quality=1.0,
            structural_quality=1.0,
            noise=0.0,
        ),
    )


def test_nearby_target_and_attribute_are_verified() -> None:
    result = verify_candidate(
        _frame(),
        _candidate(
            "Master of Technology (M.Tech.) is offered by the institute. "
            "Eligibility requirements include a qualifying degree and the "
            "minimum marks specified for the admission category.",
            programs=("mtech",),
        ),
    )
    assert result.status == "verified"


def test_distant_unrelated_attribute_is_not_joined() -> None:
    result = verify_candidate(
        _frame(),
        _candidate(
            "Master of Technology (M.Tech.) programs are listed in the "
            "academic catalogue. "
            + ("General information. " * 80)
            + "Ph.D. eligibility requirements are provided in a separate "
              "section for doctoral applicants.",
            programs=("mtech",),
            attributes=(),
        ),
    )
    assert result.status == "rejected"


def test_exact_meaning_metadata_can_establish_relation() -> None:
    result = verify_candidate(
        _frame(),
        _candidate(
            "Program information.",
            programs=("mtech",),
            attributes=("eligibility",),
        ),
    )
    assert result.status == "verified"
