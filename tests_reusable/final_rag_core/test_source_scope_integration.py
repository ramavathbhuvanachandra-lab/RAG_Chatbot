from __future__ import annotations

from types import SimpleNamespace

from backend.core.candidate_verification import verify_candidate
from backend.core.nodes import _apply_scope_conflicts
from backend.core.query.models import Entity, QueryFacet, RetrievalRequirement, SemanticQueryFrame
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
)
from backend.institutions.iitj.scope_policy import SCOPE_POLICY


def _candidate(text: str, source: str) -> RetrievalCandidate:
    document = SimpleNamespace(page_content=text, metadata={"source": source})
    return RetrievalCandidate(
        document=document,
        document_id=source + text,
        source=source,
        meaning=DocumentMeaning(),
        alignment=CandidateAlignment(),
        quality=EvidenceQuality(
            content_quality=1.0,
            structural_quality=1.0,
            noise=0.0,
        ),
    )


def _mtech_frame() -> SemanticQueryFrame:
    entity = Entity(
        name="mtech",
        entity_type="program",
        entity_id="mtech",
        confidence=1.0,
        resolution_state="resolved",
    )
    return SemanticQueryFrame(
        original_query="What are the M.Tech eligibility requirements?",
        target="M.Tech",
        request_type="eligibility",
        facets=(QueryFacet("requested_attribute", "eligibility"),),
        entities=(entity,),
        requirement=RetrievalRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            reject_explicit_conflict=True,
        ),
    )


def test_program_scoped_phd_source_is_rejected_for_mtech() -> None:
    candidate = _candidate(
        "(b) The requirement of GATE is exempted for an applicant with M.Tech.",
        "data/data_iitj/iitj_rag_v1_docs_production/programs/phd/general_information.docx",
    )
    scoped = _apply_scope_conflicts(
        candidate,
        SCOPE_POLICY,
        "What are the M.Tech eligibility requirements?",
    )

    assert "program" in tuple(scoped.alignment.conflicts)
    decision = verify_candidate(_mtech_frame(), scoped)
    assert decision.status == "rejected"


def test_program_scoped_mtech_source_has_no_program_conflict() -> None:
    candidate = _candidate(
        "M.Tech eligibility requirements are stated here.",
        "data/data_iitj/iitj_rag_v1_docs_production/admissions/mtech_admissions.docx",
    )
    scoped = _apply_scope_conflicts(
        candidate,
        SCOPE_POLICY,
        "What are the M.Tech eligibility requirements?",
    )

    assert "program" not in tuple(scoped.alignment.conflicts)


if __name__ == "__main__":
    test_program_scoped_phd_source_is_rejected_for_mtech()
    print("PASS test_program_scoped_phd_source_is_rejected_for_mtech")
    test_program_scoped_mtech_source_has_no_program_conflict()
    print("PASS test_program_scoped_mtech_source_has_no_program_conflict")
    print("SOURCE SCOPE INTEGRATION TESTS: PASS (2/2)")