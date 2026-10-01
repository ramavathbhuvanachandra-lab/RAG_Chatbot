from __future__ import annotations

from types import SimpleNamespace

from backend.core.answering.generator import AnswerGenerationRequest, generate_answer
from backend.core.candidate_verification import verify_candidate
from backend.core.evidence.packaging import EvidencePackage, PackagedEvidenceItem
from backend.core.query.models import (
    Entity,
    QueryFacet,
    RetrievalRequirement,
    SemanticQueryFrame,
)
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
    document_identity,
)


def candidate(text: str, source: str, programs=(), entities=()):
    document = SimpleNamespace(
        page_content=text,
        metadata={"source": source},
    )
    return RetrievalCandidate(
        document=document,
        document_id=source + text,
        source=source,
        meaning=DocumentMeaning(
            programs=programs,
            entities=entities,
            topics=("research",),
            attributes=("eligibility",),
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


def qframe(
    question: str,
    target: str,
    name: str,
    kind: str,
    attr: str = "eligibility",
):
    entity = Entity(
        name=name,
        entity_type=kind,
        entity_id=name,
        confidence=1.0,
        resolution_state="resolved",
    )
    return SemanticQueryFrame(
        original_query=question,
        target=target,
        request_type=attr,
        facets=(QueryFacet("requested_attribute", attr),),
        entities=(entity,),
        requirement=RetrievalRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            reject_explicit_conflict=True,
        ),
    )


def test_wrong_program_rejected() -> None:
    frame = qframe(
        "What are the M.Tech eligibility requirements?",
        "M.Tech",
        "mtech",
        "program",
    )
    result = verify_candidate(
        frame,
        candidate(
            "Physics eligibility information.",
            "data/physics/programs.docx",
            programs=("msc",),
            entities=("physics",),
        ),
    )
    assert result.status == "rejected"


def test_exact_program_verified() -> None:
    frame = qframe(
        "What are the M.Tech eligibility requirements?",
        "M.Tech",
        "mtech",
        "program",
    )
    result = verify_candidate(
        frame,
        candidate(
            "M.Tech eligibility requirements are stated here.",
            "data/admissions/mtech_admissions.docx",
            programs=("mtech",),
        ),
    )
    assert result.status == "verified"


def test_wrong_department_rejected() -> None:
    frame = qframe(
        "What are the research areas in Electrical Engineering?",
        "research areas",
        "electrical_engineering",
        "entity",
        attr="research",
    )
    result = verify_candidate(
        frame,
        candidate(
            "Physics research areas are listed here.",
            "data/physics/research.docx",
            entities=("physics",),
        ),
    )
    assert result.status == "rejected"


class _FakeModel:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def invoke(self, _payload):
        self.calls += 1
        return self.response


def test_answer_rejects_unrequested_program() -> None:
    package = EvidencePackage(
        status="ready",
        context="MBA admission uses a separate process.",
        items=(
            PackagedEvidenceItem(
                "e1",
                "MBA admission uses a separate process.",
                "data/admission.docx",
                (),
            ),
        ),
        source_count=1,
        selected_count=1,
    )
    result = generate_answer(
        AnswerGenerationRequest(
            question="How do I apply for admission?",
            evidence=package,
            known_program_names=("MBA", "M.Tech", "M.Sc.", "B.Tech"),
        ),
        _FakeModel("To apply for admission to the MBA program, submit the form."),
    )
    assert result.answer == ""
    assert result.reason == "model_output_rejected_entity_contract"


def test_equivalent_local_paths_share_identity() -> None:
    relative = SimpleNamespace(
        page_content="M.Tech eligibility",
        metadata={"source": "data/admissions/mtech_admissions.docx"},
    )
    absolute = SimpleNamespace(
        page_content="M.Tech eligibility",
        metadata={
            "source": "/Users/example/project/data/admissions/mtech_admissions.docx"
        },
    )
    assert document_identity(relative) == document_identity(absolute)


TESTS = [
    test_wrong_program_rejected,
    test_exact_program_verified,
    test_wrong_department_rejected,
    test_answer_rejects_unrequested_program,
    test_equivalent_local_paths_share_identity,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"TARGET INTEGRITY TESTS: PASS ({len(TESTS)}/{len(TESTS)})")
