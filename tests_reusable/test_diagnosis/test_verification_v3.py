from __future__ import annotations

from dataclasses import dataclass

from ai_platform.core.query.models import QueryFacet, RetrievalRequirement, SemanticQueryFrame
from ai_platform.core.retrieval.contracts import CandidateAlignment, RetrievalCandidate
from ai_platform.core.retrieval.verification import verify_candidate, verify_candidates


@dataclass
class Doc:
    page_content: str
    metadata: dict


def candidate(text: str, *, cid: str = "c1", conflict: tuple[str, ...] = ()) -> RetrievalCandidate:
    return RetrievalCandidate(
        document=Doc(text, {"source": f"source/{cid}.docx"}),
        document_id=cid,
        source=f"source/{cid}.docx",
        alignment=CandidateAlignment(
            semantic_match=0.80,
            coverage=0.70,
            conflicts=conflict,
        ),
    )


def frame(
    query: str,
    *,
    target: str = "M.Tech",
    request_type: str = "eligibility",
    facets=("eligibility",),
    mode: str = "standard",
) -> SemanticQueryFrame:
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
            allow_partial_evidence=True,
        ),
    )


def test_partial_requirement_chunk_is_verified():
    f = frame("What are the eligibility requirements for regular M.Tech admission?")
    c = candidate("Regular M.Tech applicants are required to have a relevant bachelor's degree.")
    d = verify_candidate(f, c)
    assert d.status == "verified"
    assert d.accepted is True


def test_one_candidate_does_not_need_every_requirement():
    f = frame("What are the eligibility requirements for regular M.Tech admission?", facets=("eligibility",))
    c = candidate("GATE qualification is required for admission to the M.Tech programme.")
    d = verify_candidate(f, c)
    assert d.status == "verified"


def test_explicit_conflict_is_rejected():
    f = frame("What are the eligibility requirements for regular M.Tech admission?")
    c = candidate("PhD admissions require a master's degree.", conflict=("program_mismatch",))
    d = verify_candidate(f, c)
    assert d.status == "rejected"
    assert d.conflict_detected is True


def test_relation_question_stays_strict():
    f = frame(
        "Is the hostel fee included in the semester fee?",
        target="hostel fee",
        request_type="comparison",
        facets=("hostel fee", "semester fee"),
        mode="exact",
    )
    c = candidate("Hostel charges are listed separately from semester fees.", cid="relation")
    d = verify_candidate(f, c)
    assert d.status == "verified"


def test_relation_question_rejects_generic_semantic_match():
    f = frame(
        "Is the hostel fee included in the semester fee?",
        target="hostel fee",
        request_type="comparison",
        facets=("hostel fee", "semester fee"),
        mode="exact",
    )
    c = candidate("Hostel facilities are available to students.", cid="generic")
    d = verify_candidate(f, c)
    assert d.status == "rejected"


def test_batch_has_no_soft_uncertain_queue():
    f = frame("What are the eligibility requirements for M.Tech admission?")
    batch = verify_candidates(
        f,
        (
            candidate("M.Tech applicants are required to have a relevant bachelor's degree.", cid="a"),
            candidate("Completely unrelated dining information.", cid="b"),
        ),
    )
    assert all(x.status != "uncertain" for x in batch.decisions)
    assert batch.uncertain == ()
    assert {x.document_id for x in batch.verified} == {"a"}
