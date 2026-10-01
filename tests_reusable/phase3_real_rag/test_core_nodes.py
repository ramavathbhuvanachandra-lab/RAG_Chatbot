"""Adversarial hard tests for backend.core.nodes orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.nodes import CoreNodes, NodeDependencies  # noqa: E402


@dataclass(frozen=True)
class Doc:
    page_content: str
    metadata: dict


@dataclass(frozen=True)
class Candidate:
    document: Doc
    document_id: str
    source: str
    alignment: object = None
    meaning: object = None


@dataclass(frozen=True)
class Assessment:
    status: str
    score: float = 1.0
    relevant_documents: int = 1


@dataclass(frozen=True)
class Coverage:
    status: str
    question_type: str = "descriptive"


@dataclass(frozen=True)
class Package:
    status: str
    context: str
    ready_for_generation: bool
    items: tuple = ()


@dataclass(frozen=True)
class Generation:
    answer: str
    model_calls: int


@dataclass(frozen=True)
class Grounding:
    status: str


@dataclass(frozen=True)
class Guard:
    answer: str
    status: str
    fallback_used: bool
    reason: str


class FakeAnswerRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_dependencies(
    *,
    evidence_status="supported",
    coverage_status="supported",
    generated_answer="The fee is ₹50,000.",
    grounding_status="grounded",
    fallback="Configured fallback.",
):
    calls = {
        "dense": [],
        "bm25": [],
        "fuse": 0,
        "verify": 0,
        "rank": 0,
        "generate": 0,
        "ground": 0,
        "guard": 0,
    }

    d1 = Doc("The fee is ₹50,000 for the current fee structure.", {"source": "fees.docx"})
    c1 = Candidate(d1, "id-1", "fees.docx")

    def conversation_resolver(question, chat_history):
        return {
            "resolved_question": question,
            "mode": "standalone",
            "active_topic": "",
            "active_entity": "",
        }

    def multi_intent(question):
        return [SimpleNamespace(question=question, topics=(), entities=())]

    query = SimpleNamespace(
        original_query="What is the fee?",
        search_query="fee current structure",
        target=None,
        request_type="quantitative",
        confidence=1.0,
        resolution_state="resolved",
    )

    def query_understander(question):
        return query

    def dense(query_text):
        calls["dense"].append(query_text)
        return [d1]

    def bm25(query_text):
        calls["bm25"].append(query_text)
        return [d1]

    def fuse(lists, **kwargs):
        calls["fuse"] += 1
        return [c1]

    def align(question, document, registry):
        return query, SimpleNamespace(), SimpleNamespace()

    def verify(frame, candidates):
        calls["verify"] += 1
        return SimpleNamespace(
            verified=tuple(candidates),
            uncertain=(),
            rejected=(),
        )

    def rank(query_text, candidates, **kwargs):
        calls["rank"] += 1
        return list(candidates)

    def build_groups(documents):
        return []

    def expand_groups(groups, chunks):
        return list(groups)

    def flatten(groups, **kwargs):
        return [d1]

    def scope_filter(query_object, documents, **kwargs):
        return list(documents)

    def assess_evidence(candidates, query=None):
        return Assessment(evidence_status)

    def assess_coverage(query_object, candidates):
        return Coverage(coverage_status, "quantitative")

    claim = SimpleNamespace(claim_id="claim-1")
    unit = SimpleNamespace(
        evidence_id="e-1",
        text="The fee is ₹50,000.",
        source="fees.docx",
        heading=None,
    )
    audit = SimpleNamespace(
        status="supported",
        claims=(claim,),
        matches=(),
        supported_claim_ids=("claim-1",),
        partial_claim_ids=(),
        unsupported_claim_ids=(),
        conflicting_claim_ids=(),
    )

    def extract_claims(query_object):
        return (claim,)

    def extract_units(candidates):
        return (unit,)

    def audit_claims(claims, units):
        return audit

    def build_package(units, *, claim_audit=None, **kwargs):
        return Package(
            status="ready",
            context="Evidence 1\nThe fee is ₹50,000.",
            ready_for_generation=True,
            items=units,
        )

    def generate(request, model):
        calls["generate"] += 1
        return Generation(generated_answer, 1)

    def ground(*, answer, evidence):
        calls["ground"] += 1
        return Grounding(grounding_status)

    def guard(answer, *, fallback):
        calls["guard"] += 1
        if not answer:
            return Guard(fallback, "empty_answer", True, "fallback")
        return Guard(answer, "clean", False, "ok")

    def institution(state):
        return SimpleNamespace(
            display_name="Example Institution",
            fallback_response=fallback,
            prompt_additions=(),
            answer_tone="student_friendly",
            answer_verbosity="moderate",
        )

    def model(state):
        return object()

    def fallback_provider(state, institution_obj):
        return fallback

    def canonical(state):
        return [d1]

    deps = NodeDependencies(
        conversation_resolver=conversation_resolver,
        multi_intent_decomposer=multi_intent,
        query_understander=query_understander,
        dense_retrieve=dense,
        keyword_retrieve=bm25,
        fuse_ranked_lists=fuse,
        align_query_to_document=align,
        rank_candidates=rank,
        verify_candidates=verify,
        build_evidence_groups=build_groups,
        expand_group_context=expand_groups,
        flatten_evidence_groups=flatten,
        filter_scope_conflicts=scope_filter,
        assess_evidence=assess_evidence,
        assess_coverage=assess_coverage,
        extract_claims=extract_claims,
        extract_evidence_units=extract_units,
        audit_claims=audit_claims,
        build_evidence_package=build_package,
        answer_generator=generate,
        answer_request_type=FakeAnswerRequest,
        assess_answer_grounding=ground,
        guard_answer=guard,
        institution_provider=institution,
        answer_model_provider=model,
        fallback_provider=fallback_provider,
        canonical_chunks_provider=canonical,
    )
    return CoreNodes(deps), calls


def test_primary_query_is_always_first_and_fanout_is_bounded():
    nodes, calls = make_dependencies()
    state = {
        "resolved_question": "What is the fee?",
        "retrieval_queries": (
            "What is the fee?",
            "fee current structure",
            "fee admissions",
            "fee extra forbidden fourth",
        ),
    }
    result = nodes.hybrid_retrieve_node(state)

    assert result["retrieval_queries"] == (
        "What is the fee?",
        "fee current structure",
        "fee admissions",
    )
    assert calls["dense"][:3] == list(result["retrieval_queries"])
    assert len(calls["dense"]) == 3
    assert len(calls["bm25"]) == 3


def test_empty_evidence_fails_closed_without_generation():
    nodes, calls = make_dependencies(evidence_status="insufficient")
    state = {
        "question": "What is the fee?",
        "resolved_question": "What is the fee?",
        "evidence_package": Package("empty", "", False),
        "evidence_assessment": Assessment("insufficient"),
        "evidence_coverage_status": "insufficient",
        "fallback_response": "Configured fallback.",
    }
    result = nodes.answer_node(state)

    assert result["answer"] == "Configured fallback."
    assert result["answer_guard_fallback_used"] is True
    assert result["model_calls"] == 0
    assert calls["generate"] == 0


def test_grounding_review_never_reaches_user():
    nodes, calls = make_dependencies(grounding_status="review")
    state = {
        "question": "What is the fee?",
        "resolved_question": "What is the fee?",
        "evidence_package": Package(
            "ready",
            "Evidence 1\nThe fee is ₹50,000.",
            True,
        ),
        "evidence_assessment": Assessment("supported"),
        "evidence_coverage_status": "supported",
        "evidence_question_type": "quantitative",
        "fallback_response": "Configured fallback.",
    }
    result = nodes.answer_node(state)

    assert result["answer"] == "Configured fallback."
    assert result["answer_grounding_status"] == "review"
    assert result["answer_guard_fallback_used"] is True
    assert calls["generate"] == 1
    assert calls["ground"] == 1


def test_grounded_answer_uses_new_answering_stack_once():
    nodes, calls = make_dependencies()
    state = {
        "question": "What is the fee?",
        "resolved_question": "What is the fee?",
        "evidence_package": Package(
            "ready",
            "Evidence 1\nThe fee is ₹50,000.",
            True,
        ),
        "evidence_assessment": Assessment("supported"),
        "evidence_coverage_status": "supported",
        "evidence_question_type": "quantitative",
        "fallback_response": "Configured fallback.",
    }
    result = nodes.answer_node(state)

    assert result["answer"] == "The fee is ₹50,000."
    assert result["answer_grounding_status"] == "grounded"
    assert result["answer_guard_fallback_used"] is False
    assert calls["generate"] == 1
    assert calls["ground"] == 1
    assert calls["guard"] == 1


def test_resolve_conversation_preserves_resolved_question():
    nodes, _ = make_dependencies()
    result = nodes.resolve_conversation_node(
        {
            "question": "What about fees?",
            "chat_history": [
                {"role": "user", "content": "Tell me about admissions."}
            ],
        }
    )
    assert result["resolved_question"] == "What about fees?"
    assert result["conversation_mode"] == "standalone"


def test_plan_multi_intent_keeps_each_question_separate():
    nodes, _ = make_dependencies()
    nodes.deps = make_dependencies()[0].deps

    def three_intents(question):
        return [
            SimpleNamespace(question="What are the Ph.D. requirements?", topics=(), entities=()),
            SimpleNamespace(question="Where is the library?", topics=(), entities=()),
            SimpleNamespace(question="What are the hostel fees?", topics=(), entities=()),
        ]

    object.__setattr__(nodes.deps, "multi_intent_decomposer", three_intents)
    result = nodes.plan_multi_intent_node(
        {"resolved_question": "Three requests"}
    )
    assert result["is_multi_intent"] is True
    assert result["intent_count"] == 3
    assert result["intent_questions"] == [
        "What are the Ph.D. requirements?",
        "Where is the library?",
        "What are the hostel fees?",
    ]


def test_package_node_calls_e6_package_not_raw_context():
    nodes, calls = make_dependencies()
    state = {
        "evidence_units": (
            SimpleNamespace(
                evidence_id="e-1",
                text="The fee is ₹50,000.",
                source="fees.docx",
            ),
        ),
        "claim_audit": SimpleNamespace(
            status="supported",
            supported_claim_ids=("claim-1",),
            partial_claim_ids=(),
        ),
    }
    result = nodes.package_evidence_node(state)
    assert result["evidence_package_status"] == "ready"
    assert result["ready_for_generation"] is True


def test_production_core_has_no_institution_specific_literals():
    source = Path(__file__).resolve().parents[2] / "backend" / "core" / "nodes.py"
    text = source.read_text(encoding="utf-8").casefold()
    forbidden = (
        "iit jodhpur",
        "iitj",
        "data_iitj",
        "fees_and_finance.docx",
        "mtech_admissions.docx",
        "collection_name",
    )
    assert not any(marker in text for marker in forbidden), [
        marker for marker in forbidden if marker in text
    ]


def test_node_module_is_orchestration_not_old_answer_chain():
    source = Path(__file__).resolve().parents[2] / "backend" / "core" / "nodes.py"
    text = source.read_text(encoding="utf-8").casefold()

    assert "answer_prompt" not in text
    assert "backend.prompts" not in text
    assert "answer_chain" not in text
    assert "from backend.answer_guard" not in text


def test_grounding_failure_is_recorded_not_hidden():
    nodes, _ = make_dependencies(grounding_status="review")
    state = {
        "question": "What is the fee?",
        "resolved_question": "What is the fee?",
        "evidence_package": Package(
            "ready",
            "Evidence 1\nThe fee is ₹50,000.",
            True,
        ),
        "evidence_assessment": Assessment("supported"),
        "evidence_coverage_status": "supported",
        "fallback_response": "Configured fallback.",
    }
    result = nodes.answer_node(state)
    assert result["answer_grounding_status"] == "review"
    assert result["answer_guard_reason"] == "answer_grounding_review"


def run_tests() -> None:
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]

    for test in tests:
        test()

    print(f"NODE 1 CORE ORCHESTRATION HARD TESTS: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    run_tests()
