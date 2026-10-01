"""Production-style hard tests for the reusable RAG core.

The test file intentionally imports only from the repository itself. It never
references model-workspace paths such as /mnt/data/core_work, so it is safe to
run from a developer laptop or CI checkout.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from backend.core.answering.generator import (
    AnswerGenerationRequest,
    generate_answer,
)
from backend.core.answering.guard import guard_answer
from backend.core.candidate_verification import verify_candidates
from backend.core.evidence.claims import (
    Claim,
    ClaimAudit,
    ClaimEvidenceMatch,
    EvidenceUnit,
    audit_claims,
)
from backend.core.evidence.packaging import build_evidence_package
from backend.core.query.models import QueryFacet, SemanticQueryFrame
from backend.core.retrieval.lexical_recall import retrieve_lexical, score_document
from backend.core.retrieval_contracts import (
    RetrievalCandidate,
    RetrievalProvenance,
)
from backend.core.nodes import CoreNodes, NodeDependencies


class Doc:
    def __init__(self, text: str, *, source: str = "source-a", title: str = "") -> None:
        self.page_content = text
        self.metadata = {"source": source}
        if title:
            self.metadata["title"] = title


def _frame(
    question: str,
    *,
    target: str | None = None,
    request_type: str | None = None,
    facets: tuple[QueryFacet, ...] = (),
    list_question: bool = False,
) -> SemanticQueryFrame:
    return SemanticQueryFrame(
        original_query=question,
        normalized_query=question,
        target=target,
        request_type=request_type,
        facets=facets,
        is_list_question=list_question,
        confidence=0.95,
        interpretation_status="resolved",
    )


def _candidate(text: str, *, source: str = "source-a") -> RetrievalCandidate:
    document = Doc(text, source=source)
    return RetrievalCandidate.from_document(
        document,
        source=source,
        provenance=RetrievalProvenance(
            primary_query=text,
            retrieval_queries=(text,),
        ),
    )


def _unit(
    evidence_id: str,
    text: str,
    *,
    source: str = "source-a",
    position: int = 0,
    heading: str | None = None,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        text=text,
        source=source,
        position=position,
        heading=heading,
    )


def _audit_for(evidence_id: str, status: str) -> ClaimAudit:
    claim = Claim(
        claim_id="request_1",
        text="What is the process?",
        kind="request",
        name="process",
    )
    return ClaimAudit(
        claims=(claim,),
        matches=(ClaimEvidenceMatch("request_1", evidence_id, 0.9 if status == "supported" else 0.3, status),),
        supported_claim_ids=("request_1",) if status == "supported" else (),
        partial_claim_ids=("request_1",) if status == "partial" else (),
        unsupported_claim_ids=(),
        conflicting_claim_ids=(),
    )


def test_lexical_recall_direct_topic():
    docs = [
        Doc("The institute has several departments and academic programs.", title="Academic Programs"),
        Doc("Hostel laundry facilities are available.", title="Hostel Facilities"),
    ]
    frame = _frame(
        "What academic programs are available?",
        target="academic programs",
        request_type="information",
        list_question=True,
    )
    hits = retrieve_lexical(
        "What academic programs are available?",
        docs,
        query_frame=frame,
        limit=5,
        min_score=0.28,
    )
    assert hits
    assert "academic programs" in hits[0].document.page_content.casefold()


def test_verifier_accepts_strong_word_to_word_recall():
    frame = _frame(
        "What academic programs are available?",
        target="academic programs",
        request_type="information",
        list_question=True,
    )
    candidates = [
        _candidate("The institute has several departments and academic programs."),
    ]
    batch = verify_candidates(frame, candidates)
    assert batch.verified, batch.to_dict()


def test_verifier_recall_family_positive_cases():
    cases = (
        (
            "Can parents visit the campus during admission?",
            "campus admission visit parents are permitted during the admission period.",
            "visit",
        ),
        (
            "Does the campus have a Student Activity Centre?",
            "The Student Activity Centre is available on campus for student activities.",
            "student activity centre",
        ),
        (
            "Can undergraduate students do research?",
            "Undergraduate students can participate in research projects and research activities.",
            "research",
        ),
        (
            "How do I reach IIT Jodhpur from the airport?",
            "Visitors can reach the institute from the airport using the campus transport route.",
            "reach",
        ),
        (
            "What comes after JEE Advanced for B.Tech admission?",
            "After JEE Advanced, the B.Tech admission process proceeds through counselling and seat allocation.",
            "counselling",
        ),
        (
            "Who is a faculty advisor?",
            "A faculty advisor helps students with academic planning and guidance.",
            "faculty advisor",
        ),
        (
            "What extracurricular clubs are available?",
            "The institute has several extracurricular clubs for students.",
            "extracurricular clubs",
        ),
        (
            "Where is the institute located?",
            "IIT Jodhpur is located in Jodhpur, Rajasthan.",
            "located",
        ),
    )

    for question, text, anchor in cases:
        frame = _frame(
            question,
            request_type="information" if question.startswith("What extracurricular") or question.startswith("Who is") else None,
            target=None,
        )
        batch = verify_candidates(frame, [_candidate(text)])
        assert batch.verified, {
            "question": question,
            "anchor": anchor,
            "result": batch.to_dict(),
        }


def test_relation_question_does_not_accept_unrelated_keyword_overlap():
    frame = _frame(
        "Is hostel fee included in the semester fee?",
        target="hostel fee",
        request_type="cost",
        facets=(QueryFacet("relation", "included in semester fee"),),
    )
    unrelated = _candidate("The hostel has WiFi, laundry, and a gym.")
    batch = verify_candidates(frame, [unrelated])
    assert not batch.verified, batch.to_dict()


def test_lexical_score_prefers_exact_phrase():
    frame = _frame(
        "What are the admission requirements for B.Tech?",
        target="B.Tech",
        request_type="eligibility",
    )
    exact = score_document(
        "What are the admission requirements for B.Tech?",
        Doc("B.Tech admission requirements include the prescribed academic qualification."),
        query_frame=frame,
    )
    weak = score_document(
        "What are the admission requirements for B.Tech?",
        Doc("The campus has admission information for several programs."),
        query_frame=frame,
    )
    assert exact.score > weak.score
    assert exact.target_overlap > weak.target_overlap


def test_generator_repairs_reasoning_style_output_without_second_model_call():
    evidence = build_evidence_package(
        [
            _unit("e1", "The applicant must have a bachelor’s degree in engineering or science.")
        ],
        verified_evidence=True,
    )
    request = AnswerGenerationRequest(
        question="What is the eligibility requirement?",
        evidence=evidence,
    )

    class BadModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, payload: Any) -> Any:
            self.calls += 1
            return "Let me check the evidence. Evidence 1 says the applicant needs a bachelor’s degree."

    model = BadModel()
    result = generate_answer(request, model)
    assert model.calls == 1
    assert result.generated
    assert result.answer_mode == "extractive_repair"
    assert "Evidence 1" not in result.answer
    assert "bachelor" in result.answer.casefold()


def test_guard_fails_closed_on_reasoning_leak():
    result = guard_answer(
        "First, I need to check Evidence 1. The admission process is online.",
        fallback="I’m sorry, I don’t know based on the available information.",
    )
    assert result.fallback_used
    assert result.status == "fallback"
    assert "first" not in result.answer.casefold()


def test_verified_packaging_preserves_verified_evidence_when_claim_audit_misses():
    evidence = [_unit("e1", "The application process is completed online.")]
    audit = _audit_for("e2", "supported")
    package = build_evidence_package(
        evidence,
        claim_audit=audit,
        verified_evidence=True,
    )
    assert package.items
    assert package.items[0].evidence_id == "e1"


def test_unverified_packaging_still_respects_claim_audit():
    evidence = [_unit("e1", "The application process is completed online.")]
    audit = _audit_for("e2", "supported")
    package = build_evidence_package(
        evidence,
        claim_audit=audit,
        verified_evidence=False,
    )
    assert not package.items


def test_contextless_numeric_fragment_is_not_promoted_alone():
    evidence = [
        _unit("e1", "Tuition Fee", position=10),
        _unit("e2", "₹50,000/-", position=40),
    ]
    package = build_evidence_package(evidence, verified_evidence=True)
    assert all("₹50,000" not in item.text for item in package.items)


def test_generic_cost_claim_with_distinct_values_is_marked_conflicting():
    evidence = [
        _unit("e1", "Tuition fee is ₹50,000/-.", source="source-a"),
        _unit("e2", "Tuition fee is ₹2,25,000/- per semester.", source="source-b"),
    ]
    claim = Claim(
        claim_id="request_1",
        text="What is the tuition fee?",
        kind="request",
        name="cost",
    )
    audit = audit_claims((claim,), tuple(evidence))
    assert audit.conflicting_claim_ids == ("request_1",)


def test_nodes_lexical_retrieval_lane_is_present_and_rescues_zero_verified():
    docs = (
        Doc("The institute offers academic programs including B.Tech.", source="programs"),
        Doc("Hostel laundry facilities are available.", source="hostel"),
    )
    frame = _frame(
        "What academic programs are available?",
        target="academic programs",
        request_type="information",
        list_question=True,
    )
    query = SimpleNamespace(
        original_query=frame.original_query,
        search_query="academic programs",
        target=SimpleNamespace(text="academic programs"),
        request_type="information",
    )

    base_candidate = _candidate("The institute provides unrelated campus services.", source="other")

    class FakeBatch:
        def __init__(self, verified=(), uncertain=(), rejected=()):
            self.verified = tuple(verified)
            self.uncertain = tuple(uncertain)
            self.rejected = tuple(rejected)
            self.decisions = ()

    def fake_verify(_frame, candidates):
        # Deliberately reject the semantic lane and accept the exact lexical
        # candidate. This isolates the fallback behavior.
        accepted = tuple(
            candidate
            for candidate in candidates
            if "academic programs" in candidate.document.page_content.casefold()
        )
        rejected = tuple(candidate for candidate in candidates if candidate not in accepted)
        return FakeBatch(accepted, (), rejected)

    deps = NodeDependencies(
        conversation_resolver=lambda **_: {},
        multi_intent_decomposer=lambda _: (),
        query_understander=lambda **_: query,
        dense_retrieve=lambda _: (),
        keyword_retrieve=lambda _: (),
        fuse_ranked_lists=lambda lists, **__: tuple(item for group in lists for item in group),
        align_query_to_document=lambda *args, **kwargs: (None, SimpleNamespace(), SimpleNamespace()),
        rank_candidates=lambda _query, candidates, **__: tuple(candidates),
        verify_candidates=fake_verify,
        build_evidence_groups=lambda docs: tuple(docs),
        expand_group_context=lambda docs, **__: tuple(docs),
        flatten_evidence_groups=lambda groups: tuple(groups),
        filter_scope_conflicts=lambda docs, **__: tuple(docs),
        assess_evidence=lambda *args, **__: SimpleNamespace(status="supported"),
        assess_coverage=lambda *args, **__: SimpleNamespace(status="supported"),
        extract_claims=lambda _: (),
        extract_evidence_units=lambda docs: tuple(docs),
        audit_claims=lambda *args, **__: SimpleNamespace(),
        build_evidence_package=lambda *args, **__: SimpleNamespace(),
        answer_generator=lambda *args, **__: SimpleNamespace(),
        answer_request_type=object,
        assess_answer_grounding=lambda *args, **__: SimpleNamespace(status="grounded"),
        guard_answer=lambda answer, **__: SimpleNamespace(answer=answer, status="clean", fallback_used=False, reason=""),
        institution_provider=lambda _: SimpleNamespace(),
        answer_model_provider=lambda _: None,
        fallback_provider=lambda *_: "fallback",
        canonical_chunks_provider=lambda _: docs,
    )

    nodes = CoreNodes(deps)
    result = nodes.verify_and_rank_node(
        {
            "query": query,
            "query_frame": frame,
            "fused_candidates": (base_candidate,),
        }
    )
    assert result["lexical_fallback_used"] is True
    assert result["verified_candidates"]
    assert any("academic programs" in c.document.page_content.casefold() for c in result["verified_candidates"])


TESTS = [
    test_lexical_recall_direct_topic,
    test_verifier_accepts_strong_word_to_word_recall,
    test_verifier_recall_family_positive_cases,
    test_relation_question_does_not_accept_unrelated_keyword_overlap,
    test_lexical_score_prefers_exact_phrase,
    test_generator_repairs_reasoning_style_output_without_second_model_call,
    test_guard_fails_closed_on_reasoning_leak,
    test_verified_packaging_preserves_verified_evidence_when_claim_audit_misses,
    test_unverified_packaging_still_respects_claim_audit,
    test_contextless_numeric_fragment_is_not_promoted_alone,
    test_generic_cost_claim_with_distinct_values_is_marked_conflicting,
    test_nodes_lexical_retrieval_lane_is_present_and_rescues_zero_verified,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS  {test.__name__}")
    print(f"FINAL RAG CORE HARD TESTS: PASS ({len(TESTS)}/{len(TESTS)})")