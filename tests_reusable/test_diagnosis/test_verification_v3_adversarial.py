from __future__ import annotations

from dataclasses import dataclass

from ai_platform.core.query.models import QueryFacet, RetrievalRequirement, SemanticQueryFrame
from ai_platform.core.retrieval.contracts import CandidateAlignment, RetrievalCandidate
from ai_platform.core.retrieval.verification import verify_candidate, verify_candidates


@dataclass
class Doc:
    page_content: str
    metadata: dict


def cand(cid: str, text: str, *, semantic: float = 0.8, coverage: float = 0.5, conflicts: tuple[str, ...] = ()):
    return RetrievalCandidate(
        document=Doc(text, {"source": f"source/{cid}.docx"}),
        document_id=cid,
        source=f"source/{cid}.docx",
        alignment=CandidateAlignment(semantic_match=semantic, coverage=coverage, conflicts=conflicts),
    )


def frame(query: str, target: str = "M.Tech", request_type: str = "eligibility", facets=("eligibility",), *, mode="standard", require_scope=False):
    return SemanticQueryFrame(
        original_query=query,
        normalized_query=query,
        semantic_query=query,
        target=target,
        request_type=request_type,
        facets=tuple(QueryFacet(name="requested", value=f) for f in facets),
        requirement=RetrievalRequirement(
            mode=mode,
            require_target_alignment=bool(target),
            require_attribute_alignment=bool(facets),
            require_scope_alignment=require_scope,
            allow_partial_evidence=True,
        ),
    )


def test_partial_fact_chunk_is_allowed_without_target_repeat():
    f = frame("How much work experience is required for Executive M.Tech?", facets=("work experience", "Executive"))
    c = cand("experience", "The applicant must have a minimum of two years of work experience in industry/R&D laboratories.")
    d = verify_candidate(f, c)
    assert d.status == "verified"
    assert d.accepted


def test_explicit_conflict_remains_hard_rejection():
    f = frame("What are the eligibility requirements for M.Tech admission?")
    c = cand("phd", "PhD admission requires a master's degree.", conflicts=("program_mismatch",))
    d = verify_candidate(f, c)
    assert d.status == "rejected"
    assert d.conflict_detected


def test_relation_question_requires_local_relation():
    f = frame(
        "Is the hostel fee included in the semester fee?",
        target="hostel fee",
        request_type="comparison",
        facets=("hostel fee", "semester fee"),
        mode="exact",
    )
    good = cand("good", "Hostel charges are listed separately from semester fees.")
    bad = cand("bad", "Hostel accommodation is available to students.")
    assert verify_candidate(f, good).status == "verified"
    assert verify_candidate(f, bad).status == "rejected"


def test_direct_scalar_lookup_does_not_require_relationship_proof():
    f = frame("What percentage is required for M.Tech admission?", facets=("percentage",), mode="exact")
    c = cand("pct", "A minimum of 60% marks or a minimum CGPA of 6.0 is required.")
    d = verify_candidate(f, c)
    assert d.status == "verified"


def test_batch_has_no_uncertain_queue():
    f = frame("What are the eligibility requirements for M.Tech admission?")
    batch = verify_candidates(
        f,
        (
            cand("good", "M.Tech applicants must have a relevant bachelor's degree."),
            cand("bad", "The dining hall serves lunch to students.", semantic=0.1),
        ),
    )
    assert batch.uncertain == ()
    assert {x.document_id for x in batch.verified} == {"good"}


def test_candidate_score_is_diagnostic_not_a_second_generation_gate():
    f = frame("What are the eligibility requirements for M.Tech admission?")
    c = cand("partial", "GATE qualification is required for M.Tech admission.")
    d = verify_candidate(f, c)
    assert d.status == "verified"
    assert 0.0 <= d.score <= 1.0
    assert any("partial_evidence_allowed" in reason for reason in d.reasons)
