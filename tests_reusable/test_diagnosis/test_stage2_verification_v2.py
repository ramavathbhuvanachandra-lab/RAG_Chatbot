"""Stage-2 V2 verification gate: 15 positive + adversarial cases.

The production verifier must remain institution-agnostic. These tests provide
candidate evidence/metadata explicitly; they do not encode institution logic
inside the verifier.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_platform.core.query.models import Entity, QueryFacet, RetrievalRequirement, SemanticQueryFrame
from ai_platform.core.retrieval.contracts import CandidateAlignment, DocumentMeaning, EvidenceQuality, RetrievalCandidate
from ai_platform.core.retrieval.verification import verify_candidate, verify_candidates


def candidate(
    text: str,
    *,
    programs: tuple[str, ...] = (),
    attributes: tuple[str, ...] = (),
    entities: tuple[str, ...] = (),
    topics: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
    semantic_match: float = 0.92,
    scope_match: float = 1.0,
) -> RetrievalCandidate:
    document = SimpleNamespace(
        page_content=text,
        metadata={"source": "data/test/source.docx"},
    )
    return RetrievalCandidate(
        document=document,
        document_id=f"id::{text}",
        source="data/test/source.docx",
        meaning=DocumentMeaning(
            programs=programs,
            attributes=attributes,
            entities=entities,
            topics=topics,
        ),
        alignment=CandidateAlignment(
            semantic_match=semantic_match,
            coverage=0.90,
            attribute_match=0.90,
            scope_match=scope_match,
            conflicts=conflicts,
        ),
        quality=EvidenceQuality(),
    )


def frame(
    question: str,
    target: str,
    attribute: str,
    request_type: str,
    *,
    aliases: tuple[str, ...] = (),
) -> SemanticQueryFrame:
    entity = Entity(
        name=target,
        entity_type="program",
        entity_id=target.casefold().replace(" ", "_"),
        aliases=aliases,
        confidence=1.0,
        resolution_state="resolved",
    )
    return SemanticQueryFrame(
        original_query=question,
        normalized_query=question,
        semantic_query=f"{target} {attribute}",
        target=target,
        request_type=request_type,
        facets=(QueryFacet("requested_attribute", attribute),),
        entities=(entity,),
        requirement=RetrievalRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            require_scope_alignment=False,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
        ),
    )


POSITIVE_CASES = [
    (
        "mtech_eligibility",
        frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",)),
        candidate("The M.Tech program requires a relevant bachelor's degree and the stated eligibility requirements.", programs=("mtech",), attributes=("eligibility",)),
    ),
    (
        "mtech_percentage",
        frame("What percentage or CGPA is required for M.Tech admission?", "M.Tech", "percentage or CGPA", "eligibility", aliases=("m tech",)),
        candidate("For M.Tech admission, the minimum requirement is 60% marks or the specified CGPA.", programs=("mtech",), attributes=("percentage", "cgpa")),
    ),
    (
        "mtech_gate",
        frame("Is GATE required for M.Tech admission?", "M.Tech", "GATE", "admission", aliases=("m tech",)),
        candidate("Regular M.Tech admission requires a valid GATE score or an equivalent qualification.", programs=("mtech",), attributes=("gate",)),
    ),
    (
        "mtech_qualification",
        frame("Which bachelor's degree qualifications are accepted for M.Tech admission?", "M.Tech", "accepted bachelor's degree qualifications", "eligibility", aliases=("m tech",)),
        candidate("M.Tech admission accepts the specified relevant bachelor's degree qualifications, including engineering and science backgrounds.", programs=("mtech",), attributes=("bachelor",)),
    ),
    (
        "mtech_written_test",
        frame("Is there a written test for M.Tech admission?", "M.Tech", "written test", "admission", aliases=("m tech",)),
        candidate("The M.Tech admission process includes a written test for eligible applicants.", programs=("mtech",), attributes=("written test",)),
    ),
    (
        "mtech_work_experience",
        frame("How much work experience is required for Executive M.Tech?", "Executive M.Tech", "work experience", "eligibility", aliases=("executive m tech", "m tech")),
        candidate("Executive M.Tech applicants require the prescribed work experience before admission.", programs=("executive mtech",), attributes=("work experience",)),
    ),
    (
        "msc_eligibility",
        frame("What are the eligibility requirements for M.Sc admission?", "M.Sc", "eligibility", "eligibility", aliases=("m sc",)),
        candidate("The M.Sc eligibility requirements include the relevant academic qualification and the applicable admission route.", programs=("msc",), attributes=("eligibility",)),
    ),
    (
        "msc_percentage",
        frame("What percentage or CGPA is required for M.Sc admission?", "M.Sc", "percentage or CGPA", "eligibility", aliases=("m sc",)),
        candidate("For M.Sc admission, the required percentage or CGPA must meet the stated minimum threshold.", programs=("msc",), attributes=("percentage", "cgpa")),
    ),
    (
        "msc_jam",
        frame("Is JAM required for M.Sc admission?", "M.Sc", "JAM", "admission", aliases=("m sc",)),
        candidate("JAM is the entrance route used for the regular M.Sc admission process.", programs=("msc",), attributes=("jam",)),
    ),
    (
        "msc_bachelor_route",
        frame("Can I apply for M.Sc through a bachelor's degree route with a test or interview?", "M.Sc", "bachelor's degree route", "admission", aliases=("m sc",)),
        candidate("Applicants with the bachelor's-degree route may be considered through the written test or interview process for M.Sc admission.", programs=("msc",), attributes=("bachelor route", "written test", "interview")),
    ),
    (
        "phd_eligibility",
        frame("What are the eligibility requirements for Ph.D. admission?", "Ph.D.", "eligibility", "eligibility", aliases=("ph d",)),
        candidate("Ph.D. admission eligibility requires the prescribed master's or bachelor's qualification and the stated marks.", programs=("phd",), attributes=("eligibility",)),
    ),
    (
        "phd_application_fee",
        frame("What is the Ph.D. application processing fee?", "Ph.D.", "processing fee", "cost", aliases=("ph d",)),
        candidate("The Ph.D. application processing fee is the amount specified for the applicable applicant category.", programs=("phd",), attributes=("processing fee",)),
    ),
    (
        "mba_steps",
        frame("What are the steps for MBA admission?", "MBA", "steps", "procedure"),
        candidate("MBA admission proceeds through the listed application, shortlisting, and selection steps.", programs=("mba",), attributes=("steps",)),
    ),
    (
        "registration_documents",
        frame("What documents do I need to bring for registration?", "registration", "documents to bring", "registration"),
        candidate("During registration, students must bring the required certificates, score cards, proof of address, and identity proof.", topics=("registration",), attributes=("documents", "certificates", "identity proof")),
    ),
    (
        "registration_provisional",
        frame("What happens during provisional registration?", "provisional registration", "procedure", "registration", aliases=("provisionally registered",)),
        candidate("During provisional registration, students submit the required documents and complete the applicable registration procedure.", topics=("registration",), attributes=("procedure",)),
    ),
]


@pytest.mark.parametrize("case_id, query_frame, good", POSITIVE_CASES)
def test_fifteen_positive_cases_are_verified(case_id, query_frame, good):
    result = verify_candidate(query_frame, good)
    assert result.status == "verified", (case_id, result.to_dict())
    assert result.accepted is True
    assert result.evidence is not None
    assert result.evidence.relation_score >= 0.70


def test_distant_target_and_attribute_are_not_verified_by_semantic_score():
    q = frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",))
    bad = candidate(
        "M.Tech programs are listed in the catalogue.\n\n"
        + ("General information. " * 40)
        + "\n\nPh.D. eligibility requirements are described separately.",
        programs=("mtech",),
        attributes=(),
        semantic_match=0.99,
    )
    result = verify_candidate(q, bad)
    assert result.status != "verified"
    assert "required_target_attribute_relation_not_grounded" in result.reasons


def test_semantic_similarity_cannot_replace_missing_target():
    q = frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",))
    bad = candidate(
        "Eligibility requirements for admission are described here.",
        programs=(),
        attributes=("eligibility",),
        semantic_match=1.0,
    )
    result = verify_candidate(q, bad)
    assert result.status != "verified"
    assert "required_target_not_grounded" in result.reasons


def test_structured_meaning_can_support_both_sides_without_lexical_target():
    q = frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",))
    good = candidate(
        "Program admission information.",
        programs=("mtech",),
        attributes=("eligibility",),
        semantic_match=0.95,
    )
    result = verify_candidate(q, good)
    assert result.status == "verified"
    assert result.evidence.relation_score >= 0.70


def test_explicit_scope_conflict_is_hard_rejection_when_required():
    q = frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",))
    # Scope enforcement is represented by the alignment conflict at this
    # stage; it should remain a hard evidence failure.
    bad = candidate(
        "M.Tech eligibility requirements are discussed here.",
        programs=("mtech",),
        attributes=("eligibility",),
        conflicts=("program_scope",),
    )
    result = verify_candidate(q, bad)
    assert result.status == "rejected"
    assert result.conflict_detected is True


def test_batch_partition_is_lossless_and_disjoint():
    q = frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",))
    items = [
        POSITIVE_CASES[0][2],
        candidate("Eligibility requirements are given, but no requested program is identified.", attributes=("eligibility",), semantic_match=0.99),
        candidate("An unrelated research program is described.", programs=("research",), attributes=("research",), semantic_match=0.10),
    ]
    batch = verify_candidates(q, items)
    all_ids = [x.document_id for x in batch.verified + batch.uncertain + batch.rejected]
    assert len(all_ids) == len(items)
    assert len(set(all_ids)) == len(items)
    assert not (set(x.document_id for x in batch.verified) & set(x.document_id for x in batch.rejected))


def test_hinglish_query_uses_structured_target_and_local_factual_evidence():
    q = frame(
        "M.Tech admission ke liye GATE compulsory hai kya?",
        "M.Tech",
        "GATE",
        "admission",
        aliases=("m tech",),
    )
    good = candidate(
        "M.Tech admission requires a valid GATE score or equivalent qualification.",
        programs=("mtech",),
        attributes=("gate",),
    )
    result = verify_candidate(q, good)
    assert result.status == "verified"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
